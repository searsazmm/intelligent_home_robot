# -*- coding: utf-8 -*-
"""后端 B 核心逻辑单元测试。

跑法（在 backend_B 目录下）：
    python -m pytest tests/ -v
    python tests/test_core.py            # 不装 pytest 也能跑，见文件末尾

覆盖重点放在**容易出错又不容易发现**的地方：
    - 粘包/半包分帧
    - 视觉状态判定的防抖、迟滞、失联降级
    - 文本情绪对否定词/程度副词的处理
    - 状态融合的优先级
    - 历史 CSV 的写入与回读
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.dialogue import DialogueEngine
from core.history_store import CsvHistoryStore, TurnRecord
from core.protocol import LineBuffer, ProtocolError, decode_json_line, encode_line
from core.text_emotion import TextEmotionAnalyzer
from core.vision_state import VisionSample, VisionStateEvaluator


# ==========================================================================
# 报文分帧
# ==========================================================================

class TestProtocol(unittest.TestCase):

    def test_encode_line_ends_with_newline(self):
        """api_doc §3.3.3：每条 JSON 末尾必须带 \\n。"""
        self.assertTrue(encode_line({"a": 1}).endswith(b"\n"))

    def test_encode_line_keeps_chinese_readable(self):
        """中文要以 UTF-8 原文传输，不转成 \\uXXXX。"""
        self.assertIn("你好".encode("utf-8"), encode_line({"text": "你好"}))

    def test_decode_rejects_non_object(self):
        """顶层不是 JSON 对象的报文要拒绝，而不是当成合法数据。"""
        with self.assertRaises(ProtocolError):
            decode_json_line("[1,2,3]")
        with self.assertRaises(ProtocolError):
            decode_json_line("")

    def test_buffer_splits_sticky_packets(self):
        """三条粘在一个 TCP 包里，要能拆成三条。"""
        buf = LineBuffer()
        chunk = encode_line({"n": 1}) + encode_line({"n": 2}) + encode_line({"n": 3})
        lines = list(buf.feed(chunk))
        self.assertEqual(len(lines), 3)
        self.assertEqual([decode_json_line(x)["n"] for x in lines], [1, 2, 3])

    def test_buffer_handles_half_packet(self):
        """半包要缓存住，等下一段到了再拼出来 —— 这是防粘包的核心场景。"""
        buf = LineBuffer()
        whole = encode_line({"n": 42})
        head, tail = whole[:6], whole[6:]

        self.assertEqual(list(buf.feed(head)), [])          # 半包，不该吐出任何东西
        self.assertEqual(len(buf.pending), 6)               # 但也不能丢

        lines = list(buf.feed(tail))
        self.assertEqual(len(lines), 1)
        self.assertEqual(decode_json_line(lines[0])["n"], 42)

    def test_buffer_handles_multibyte_split(self):
        """中文字符被 TCP 从中间切开，也不能解码出错。"""
        buf = LineBuffer()
        whole = encode_line({"text": "你好世界"})
        # 在第 2 个字节处切开，正好落在"你"的 UTF-8 编码中间
        lines = list(buf.feed(whole[:2])) + list(buf.feed(whole[2:]))
        self.assertEqual(len(lines), 1)
        self.assertEqual(decode_json_line(lines[0])["text"], "你好世界")

    def test_buffer_ignores_blank_lines(self):
        """空行忽略，不当作错误。"""
        buf = LineBuffer()
        self.assertEqual(list(buf.feed(b"\n\n\n")), [])


# ==========================================================================
# 视觉样本解析
# ==========================================================================

class TestVisionSample(unittest.TestCase):

    def test_parses_full_payload(self):
        sample = VisionSample.from_payload({
            "timestamp": 12.5, "has_face": True, "ear": 0.31, "blink_cnt": 7,
            "pitch": 3.0, "yaw": -2.0, "roll": 1.0, "emo_feature": "normal",
        })
        self.assertTrue(sample.has_face)
        self.assertAlmostEqual(sample.ear, 0.31)
        self.assertEqual(sample.blink_cnt, 7)

    def test_tolerates_missing_and_null_fields(self):
        """A 端在无人脸时字段可能填 null —— 不能因此抛异常。"""
        sample = VisionSample.from_payload({"has_face": False, "ear": None})
        self.assertFalse(sample.has_face)
        self.assertEqual(sample.ear, 0.0)
        self.assertEqual(sample.emo_feature, "normal")

    def test_tolerates_string_types(self):
        """CSV 读进来全是字符串，布尔/数字要能正确还原。"""
        sample = VisionSample.from_payload({
            "has_face": "true", "ear": "0.25", "blink_cnt": "9",
        })
        self.assertTrue(sample.has_face)
        self.assertAlmostEqual(sample.ear, 0.25)
        self.assertEqual(sample.blink_cnt, 9)


# ==========================================================================
# 视觉状态判定
# ==========================================================================

def make_sample(t: float, has_face=True, ear=0.30, blink=0,
                pitch=0.0, yaw=0.0, feature="normal") -> VisionSample:
    """造一帧，received_at 直接用传入的时间轴，这样测试不用真的 sleep。"""
    return VisionSample(
        timestamp=t, has_face=has_face, ear=ear, blink_cnt=blink,
        pitch=pitch, yaw=yaw, roll=0.0, emo_feature=feature, received_at=t,
    )


def feed_range(ev, start, end, step=0.1, **kwargs):
    """从 start 到 end 按 step 连续喂帧，返回最后一个状态。"""
    state = None
    t = start
    while t < end:
        state = ev.push(make_sample(t, **kwargs))
        t = round(t + step, 3)
    return state


class TestVisionStateEvaluator(unittest.TestCase):

    def test_normal_when_face_present(self):
        ev = VisionStateEvaluator()
        state = feed_range(ev, 0.0, 10.0, ear=0.30)
        self.assertEqual(state.state, config.STATE_NORMAL)

    def test_tired_from_upstream_feature(self):
        """A 端直接给出 tired 特征时应判疲劳。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0, ear=0.30)
        state = feed_range(ev, 5.0, 10.0, ear=0.30, feature="tired")
        self.assertEqual(state.state, config.STATE_TIRED)

    def test_tired_from_sustained_low_ear(self):
        """EAR 持续偏低（A 端特征仍是 normal）也要能判出疲劳。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0, ear=0.30)
        state = feed_range(ev, 5.0, 20.0, ear=0.10, feature="normal")
        self.assertEqual(state.state, config.STATE_TIRED)

    def test_brief_low_ear_does_not_trigger(self):
        """眨一下眼 EAR 短暂变低，不能判成疲劳 —— 防抖的意义就在这里。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0, ear=0.30)
        state = feed_range(ev, 5.0, 5.5, ear=0.10)   # 只低 0.5 秒
        state = feed_range(ev, 5.5, 10.0, ear=0.30)
        self.assertEqual(state.state, config.STATE_NORMAL)

    def test_sad_from_upstream_low(self):
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        state = feed_range(ev, 5.0, 12.0, feature="low")
        self.assertEqual(state.state, config.STATE_SAD)

    # ------------------------------------------------------------------
    # emo_feature 两套枚举（api_doc §3.2 V1.3）
    #
    # 仓库里并存两套模块 A，出站格式一样、emo_feature 取值不一样：
    #   A-包       normal / low   / tired
    #   A-单文件   normal / tired / sad / blank
    # B 必须两套都认。这几条守住"接上另一套 A 之后不会静默漏判"。
    # ------------------------------------------------------------------

    def test_sad_from_upstream_sad(self):
        """A-单文件发的正式值 sad 必须能判出低落（此前只认 low，会静默漏掉）。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        state = feed_range(ev, 5.0, 12.0, feature="sad")
        self.assertEqual(state.state, config.STATE_SAD)
        self.assertIn("sad", state.reason)   # 确认是 A 端特征分支判的

    def test_low_and_sad_are_equivalent(self):
        """low 是 sad 的兼容别名 —— 两个值必须判出同一个状态，且都不回归。"""
        for feature in ("low", "sad"):
            with self.subTest(feature=feature):
                ev = VisionStateEvaluator()
                feed_range(ev, 0.0, 5.0)
                state = feed_range(ev, 5.0, 12.0, feature=feature)
                self.assertEqual(state.state, config.STATE_SAD)

    def test_blank_maps_to_absent(self):
        """blank（发呆/失神）判 absent —— C 端 absent 的标签本来就是「走神/无人」。

        理由串必须提到 blank：否则这台机器一直有人脸在位，absent 也可能
        是"丢脸超宽限期"那条分支蒙对的，测不出真正想守的东西。
        """
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        state = feed_range(ev, 5.0, 12.0, feature="blank")
        self.assertEqual(state.state, config.STATE_ABSENT)
        self.assertIn("blank", state.reason)

    def test_emo_feature_mapping_table(self):
        """表驱动：四个合法值各自的落点，外加一个非法值。"""
        cases = (
            ("normal", config.STATE_NORMAL),
            ("tired", config.STATE_TIRED),
            ("low", config.STATE_SAD),
            ("sad", config.STATE_SAD),
            ("blank", config.STATE_ABSENT),
        )
        for feature, expected in cases:
            with self.subTest(feature=feature):
                ev = VisionStateEvaluator()
                feed_range(ev, 0.0, 5.0)
                state = feed_range(ev, 5.0, 12.0, feature=feature)
                self.assertEqual(state.state, expected)

    def test_unknown_emo_feature_falls_back_to_normal(self):
        """没见过的取值不能崩、也不能误判成低落 —— 落到正常的兜底分支。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        state = feed_range(ev, 5.0, 12.0, feature="weird")
        self.assertEqual(state.state, config.STATE_NORMAL)

    def test_blank_is_checked_before_tired(self):
        """blank 与"困得睁不开眼"互斥，先判走神 —— 别被低 EAR 抢走。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        state = feed_range(ev, 5.0, 20.0, ear=0.10, feature="blank")
        self.assertEqual(state.state, config.STATE_ABSENT)
        self.assertIn("blank", state.reason)

    def test_absent_after_face_lost_grace(self):
        """人脸消失后，超过宽限期要判 absent。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        state = feed_range(ev, 5.0, 20.0, has_face=False)
        self.assertEqual(state.state, config.STATE_ABSENT)

    def test_absent_detection_is_not_delayed_by_window(self):
        """人离开后要"及时"判 absent。

        回归测试：早先的实现用"整个 10 秒窗口里有没有人脸"来判断，
        结果旧的人脸样本在窗口里赖了 10 秒，人走了 8 秒才发现。
        """
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 10.0)                    # 0~9.9s 都有人脸
        ev.push(make_sample(10.0, has_face=False))   # 第 10 秒人走了
        # 最后一张脸在 9.9s：宽限 2s → 11.9s 才够条件，再加防抖 1.5s → 13.4s 生效。
        # 喂到 14s，此时必须已经判出 absent（老实现要等到 18s 才切）。
        state = feed_range(ev, 10.0, 14.0, has_face=False)
        self.assertEqual(state.state, config.STATE_ABSENT)

    def test_absent_recovery_is_slower_than_entry(self):
        """从 absent 恢复要比进入 absent 慢（迟滞），避免人一闪而过就切回 normal。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        feed_range(ev, 5.0, 20.0, has_face=False)
        self.assertEqual(ev.get_state().state, config.STATE_ABSENT)

        # 人脸刚回来 1 秒，不该立刻恢复
        ev.push(make_sample(20.0))
        state = feed_range(ev, 20.1, 21.0)
        self.assertEqual(state.state, config.STATE_ABSENT)

        # 撑过 ABSENT_RECOVER_HOLD 之后才恢复
        state = feed_range(ev, 21.0, 25.0)
        self.assertEqual(state.state, config.STATE_NORMAL)

    def test_stale_data_degrades_to_absent(self):
        """数据源不发包了，get_state() 要降级成 absent，不能一直报 normal。"""
        clock = [1000.0]
        ev = VisionStateEvaluator(clock=lambda: clock[0])
        feed_range(ev, 0.0, 5.0)
        ev.push(VisionSample(timestamp=5.0, has_face=True, ear=0.30,
                             emo_feature="normal", received_at=1000.0))
        self.assertEqual(ev.get_state().state, config.STATE_NORMAL)

        # 把时钟往前拨，超过 VISION_STALE_SECONDS
        clock[0] += config.VISION_STALE_SECONDS + 1.0
        self.assertEqual(ev.get_state().state, config.STATE_ABSENT)

    def test_disconnect_forces_absent(self):
        """断开与 A 的连接后必须立刻 absent —— 看不见人时不能假装正常。"""
        ev = VisionStateEvaluator()
        feed_range(ev, 0.0, 5.0)
        state = ev.mark_disconnected()
        self.assertEqual(state.state, config.STATE_ABSENT)
        self.assertEqual(ev.get_state().state, config.STATE_ABSENT)

    def test_blink_rate_needs_long_window(self):
        """眨眼频率的误判回归测试。

        10 秒窗口里正常波动就能算出 40+ 次/分，早先把好好的人判成了疲劳。
        现在要求至少 20 秒、至少 6 次才认。
        """
        ev = VisionStateEvaluator()
        # 10 秒内眨 7 次（换算成频率是 42/分，但样本期太短，不该采信）
        blink = 0
        t = 0.0
        next_blink = 0.0
        state = None
        while t < 10.0:
            if t >= next_blink:
                blink += 1
                next_blink = t + 1.4
            state = ev.push(make_sample(t, blink=blink, ear=0.30))
            t = round(t + 0.1, 3)
        # 这里看 push() 的返回值而不是 get_state()：get_state() 会拿真实的墙钟
        # 去和测试里的合成时间轴比，必然判成"数据过期"。push() 才是被测对象。
        self.assertEqual(state.state, config.STATE_NORMAL)


# ==========================================================================
# 文本情绪
# ==========================================================================

class TestTextEmotion(unittest.TestCase):

    def setUp(self):
        self.analyzer = TextEmotionAnalyzer()

    def test_empty_text_is_neutral(self):
        self.assertEqual(self.analyzer.label_of(""), "neutral")
        self.assertEqual(self.analyzer.label_of("   "), "neutral")

    def test_plain_negative(self):
        self.assertEqual(self.analyzer.label_of("我今天有点累"), "negative")
        self.assertEqual(self.analyzer.label_of("心里难受，想哭"), "negative")

    def test_plain_positive(self):
        self.assertEqual(self.analyzer.label_of("我很开心，谢谢你"), "positive")

    def test_negation_flips_polarity(self):
        """否定词要翻转极性："不开心"是消极的。"""
        self.assertEqual(self.analyzer.label_of("我不开心"), "negative")
        self.assertEqual(self.analyzer.label_of("我挺开心的"), "positive")

    def test_intensifier_amplifies(self):
        """"很累"的分数要比"累"更消极。"""
        weak = self.analyzer.analyze("有点累").score
        strong = self.analyzer.analyze("非常累").score
        self.assertLess(strong, weak)

    def test_discomfort_is_flagged(self):
        """提到身体不适要单独打标，对话那边才会走健康关怀分支。"""
        result = self.analyzer.analyze("头疼得厉害")
        self.assertTrue(result.discomfort)
        self.assertEqual(result.label, "negative")

    def test_crisis_expression_is_strongly_negative(self):
        result = self.analyzer.analyze("活着没意思")
        self.assertEqual(result.label, "negative")
        self.assertLess(result.score, -3.0)

    def test_crisis_is_flagged(self):
        """求救信号要留下**独立标记**，不能只靠"分数很低"让下游去猜。

        这个标记是 core/dialogue.py 走危机关怀分支、并且**不把这句话交给
        大模型**的唯一依据（见 llm_eligible）。别的极端消极句也能凑到低分，
        但那些不需要走危机处理，所以必须能区分开。
        """
        for text in ("我不想活了", "活着没意思", "不如死了", "没人在乎我"):
            with self.subTest(text=text):
                self.assertTrue(self.analyzer.analyze(text).crisis)

    def test_ordinary_sadness_is_not_crisis(self):
        """普通抱怨不能被打成危机 —— 否则关怀语会变得很吓人。"""
        for text in ("今天有点闷", "心里难受", "我挺累的", "没什么意思"):
            with self.subTest(text=text):
                self.assertFalse(self.analyzer.analyze(text).crisis)

    def test_crisis_survives_whitespace_evasion(self):
        """插空格、全角不能绕过危机判定。

        匹配是在 _normalize() 之后做的（去空白、全角转半角），所以
        「我 不 想 活 了」同样命中。这条同时钉住"下游不要拿原文自己再匹配
        一遍"这个约定 —— 那样写就会漏掉这个用例。
        """
        for text in ("我 不 想 活 了", "不 想 活", "活着 没意思"):
            with self.subTest(text=text):
                self.assertTrue(self.analyzer.analyze(text).crisis)

    def test_crisis_does_not_imply_discomfort(self):
        """危机和身体不适是两个独立信号。

        "我疼得不想活了"两者都命中，而 _candidates 里危机优先级更高 ——
        这时候该回应的是"不想活"，不是"疼"。
        """
        result = self.analyzer.analyze("我不想活了")
        self.assertTrue(result.crisis)
        self.assertFalse(result.discomfort)

    def test_fine_grained_emotion(self):
        self.assertEqual(self.analyzer.analyze("我好困").emotion, "tired")
        self.assertEqual(self.analyzer.analyze("睡不着，很担心").emotion, "anxious")

    def test_neutral_smalltalk(self):
        self.assertEqual(self.analyzer.label_of("嗯"), "neutral")
        self.assertEqual(self.analyzer.label_of("随便聊聊"), "neutral")


# ==========================================================================
# 状态融合与对话
# ==========================================================================

class TestDialogue(unittest.TestCase):

    def setUp(self):
        self.engine = DialogueEngine()

    def test_fuse_vision_tired_wins_over_neutral_text(self):
        """用户嘴上说没事，但视觉已经判疲劳 —— 以视觉为准。"""
        from core.vision_state import VisionState
        state = self.engine.respond("我挺好的", VisionState(state=config.STATE_TIRED)).state
        self.assertEqual(state, config.STATE_TIRED)

    def test_fuse_negative_text_pulls_normal_down_to_sad(self):
        """视觉没看出问题，但文字明显消极 —— 要拉低成 sad。"""
        from core.vision_state import VisionState
        state = self.engine.respond("心里难受，想哭", VisionState(state=config.STATE_NORMAL)).state
        self.assertEqual(state, config.STATE_SAD)

    def test_fuse_tired_text_pulls_normal_to_tired(self):
        from core.vision_state import VisionState
        state = self.engine.respond("我好累啊", VisionState(state=config.STATE_NORMAL)).state
        self.assertEqual(state, config.STATE_TIRED)

    def test_active_chat_overrides_absent(self):
        """视觉判无人，但用户正在打字 —— 说明人在，不该继续报 absent。"""
        from core.vision_state import VisionState
        state = self.engine.respond("你好", VisionState(state=config.STATE_ABSENT)).state
        self.assertEqual(state, config.STATE_NORMAL)

    def test_empty_text_does_not_crash(self):
        reply = self.engine.respond("")
        self.assertTrue(reply.reply)

    def test_standalone_greeting_word_is_not_a_substring_match(self):
        """「早」只有整句就是它时才算打招呼。

        回归测试：词表里「早」是唯一的单字关键词，做**子串**匹配时
        「我们家那口子走得早」会被识别成 greeting —— 那是在说老伴去世，
        却被回一句热情的招呼。既答非所问，也不合适。
        """
        from core.dialogue import STANDALONE_KEYWORDS
        for text in ("早", "早！", "早，", "早上好"):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                self.assertEqual(self.engine.recognize_intent(text, emotion),
                                 "greeting")

        for text, expected in (("我们家那口子走得早", "chat"),
                               ("我起得早", "chat"),
                               ("你早点睡吧", "chat")):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                got = self.engine.recognize_intent(text, emotion)
                self.assertNotEqual(got, "greeting",
                                    "%r 被当成了打招呼" % text)
                self.assertEqual(got, expected)

        # 单字规则只该罩住真正有歧义的那几个，别顺手把整张表都改了
        self.assertEqual(STANDALONE_KEYWORDS, frozenset({"早"}))
        self.assertNotIn("你好", STANDALONE_KEYWORDS)

    def test_intent_recognition(self):
        for text, expected in (("你好呀", "greeting"),
                               ("那先这样吧，再见", "farewell"),
                               ("你叫什么名字", "identity"),
                               ("现在几点了", "time"),
                               ("头疼得厉害", "discomfort")):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                self.assertEqual(self.engine.recognize_intent(text, emotion), expected)

    def test_life_sharing_is_not_farewell(self):
        """「我今天出门散步了」是生活分享，不是告别 —— 不能被打发。

        回归测试：farewell 词表里原本是"出门""走了""我睡"三个短词，
        子串一撞就误判：老人分享「我今天出门散步了」，机器人回
        「行，您去忙吧」—— 分享日常恰恰是最该接住的话。「我睡不着」
        是失眠倾诉不是晚安；「我老伴走了」是悼念，答成"您慢点走"伤人。
        """
        for text in ("我今天出门散步了",
                     "我今天出门买了趟菜",
                     "我睡不着",
                     "我老伴走了"):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                got = self.engine.recognize_intent(text, emotion)
                self.assertNotEqual(got, "farewell", "%r 被当成了告别" % text)

        # 真正的告别说法不受影响：主语+动作的完整句子仍然命中
        for text in ("我出门了", "我得走了", "先走了", "我睡了", "要出门了"):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                self.assertEqual(self.engine.recognize_intent(text, emotion),
                                 "farewell", "%r 没被认成告别" % text)

    def test_borrowed_keyword_does_not_match(self):
        """被反例"借走"的关键词不作数（KEYWORD_BLOCKERS）。

        「我心疼孙子」是在说疼爱，不是身体不适 —— 误判的后果是回
        「要不要我帮您叫家里人」。「你好久没来啦」是嗔怪，不是打招呼。
        """
        emotion = self.engine.analyzer.analyze("我心疼孙子")
        self.assertNotEqual(self.engine.recognize_intent("我心疼孙子", emotion),
                            "discomfort")
        emotion = self.engine.analyzer.analyze("你好久没来啦")
        self.assertNotEqual(self.engine.recognize_intent("你好久没来啦", emotion),
                            "greeting")

        # 反例只否决挂着的那个关键词，不带它的长关键词照样命中：
        # 「头疼」不是"心疼"能借走的 —— 真不舒服仍然要被关心。
        emotion = self.engine.analyzer.analyze("头疼得厉害")
        self.assertEqual(self.engine.recognize_intent("头疼得厉害", emotion),
                         "discomfort")

    def test_negative_praise_is_not_praise(self):
        """「你真烦」带着"你真"，但情绪是负面的 —— 不是夸奖。

        误判的后果是回「谢谢您夸奖，我就想让您每天舒舒服服的」，
        比沉默还伤人。负面情绪应当落进 venting 的倾诉模板。
        """
        emotion = self.engine.analyzer.analyze("你真烦")
        self.assertEqual(emotion.label, "negative")
        self.assertEqual(self.engine.recognize_intent("你真烦", emotion),
                         "venting")

    def test_weather_question_gets_honest_answer(self):
        """问天气必须诚实说"看不了"，绝不能让大模型编一个晴天。

        模型没有实时数据，早年 chat 意图让它自由发挥时，老人问
        「今天天气怎么样」，它张嘴就答"今天阳光挺好" —— 编的，
        老人按它的话穿衣出门要挨淋。天气走确定性模板，永远不进大模型。
        """
        from core.dialogue import INTENT_FALLBACK
        from core.vision_state import VisionState
        for text in ("今天天气怎么样", "外面会下雨吗", "明天是不是要降温",
                     "今天冷不冷"):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                self.assertEqual(self.engine.recognize_intent(text, emotion),
                                 "weather")

        reply = self.engine.respond("今天天气怎么样",
                                    VisionState(state=config.STATE_NORMAL))
        self.assertEqual(reply.intent, "weather")
        self.assertIn(reply.reply, INTENT_FALLBACK["weather"])
        eligible, _ = self.engine.llm_eligible("weather", emotion, "今天天气怎么样")
        self.assertFalse(eligible, "天气问题绝不能交给大模型 —— 它只会编")

        # 反向钉住：天气**评价**仍是闲聊。「今天天气不错」不是在问天气，
        # 全仓一堆测试拿它当闲聊样本，别把老人的感慨拦成"我看不了天气"
        emotion = self.engine.analyzer.analyze("今天天气不错")
        self.assertEqual(self.engine.recognize_intent("今天天气不错", emotion),
                         "chat")

    def test_reminder_request_gets_honest_answer(self):
        """「提醒我吃药」不能应成"好嘞我记下了" —— 那是假承诺。

        机器人没有闹钟和日历。应下之后老人到点干等，误的是吃药这种
        正事。唯一诚实的回答是承认做不了，给个替代办法。
        """
        from core.dialogue import INTENT_FALLBACK
        from core.vision_state import VisionState
        for text in ("到时候提醒我吃药", "明天早上叫我起床"):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                self.assertEqual(self.engine.recognize_intent(text, emotion),
                                 "reminder")
        reply = self.engine.respond("到时候提醒我吃药",
                                    VisionState(state=config.STATE_NORMAL))
        self.assertIn(reply.reply, INTENT_FALLBACK["reminder"])
        eligible, _ = self.engine.llm_eligible("reminder", emotion, "到时候提醒我吃药")
        self.assertFalse(eligible, "提醒类请求不能交给大模型 —— 它只会假应承")

    def test_festival_question_gets_the_date_from_the_table(self):
        """「重阳节是多久」必须答出真实日期 —— 农历换算靠手抄表，不靠模型编。

        大模型记的农历日期是训练数据里的旧年份，问今年的重阳节能编出
        好几个版本。节日走 FESTIVALS 查表，永远不进大模型。
        """
        from datetime import date
        from core.dialogue import festival_reply
        from core.vision_state import VisionState
        for text in ("重阳节是多久", "今年中秋节是哪天", "过年是几号"):
            with self.subTest(text=text):
                emotion = self.engine.analyzer.analyze(text)
                self.assertEqual(self.engine.recognize_intent(text, emotion),
                                 "festival")

        # 纯函数用注入的"今天"测，不依赖真实系统日期：
        # 2026-10-06 距 2026 重阳节（10 月 18 日）还有 12 天
        got = festival_reply("重阳节是多久", today=date(2026, 10, 6))
        self.assertIn("10月18号", got)
        self.assertIn("农历九月初九", got)
        self.assertIn("还有12天", got)
        # 当天就是节日
        self.assertIn("今天就是重阳节",
                      festival_reply("重阳节是多久", today=date(2026, 10, 18)))
        # 今年过了答明年（2026-12-01 问春节 → 2027-02-06）
        self.assertIn("明年是2月6号",
                      festival_reply("春节是几号", today=date(2026, 12, 1)))
        # 公历节日不带农历说法
        self.assertEqual(festival_reply("国庆是哪天", today=date(2026, 9, 1)),
                         "国庆节是10月1号，还有30天。")
        # 年份超出表 → 诚实兜底（返回空串，由 INTENT_FALLBACK 接手）
        self.assertEqual(festival_reply("春节是几号", today=date(2028, 3, 1)), "")

        # 链路：意图 → 查表回答，且绝不进大模型
        reply = self.engine.respond("重阳节是多久",
                                    VisionState(state=config.STATE_NORMAL))
        self.assertEqual(reply.intent, "festival")
        eligible, _ = self.engine.llm_eligible("festival", emotion, "重阳节是多久")
        self.assertFalse(eligible, "节日日期不能交给大模型 —— 它只会编")

    def test_festival_table_dates_are_reviewed(self):
        """表里的日期是手抄的 —— 用已知锚点钉住，防止补表时抄错。

        这几个日期都核对过公开日历；改表（比如补 2028 年）时这条测试
        提醒你：改错了，老人照着错的日期盼儿女回家。
        """
        from core.dialogue import FESTIVALS
        anchors = {
            ("春节", 2026): (2, 17),
            ("春节", 2027): (2, 6),
            ("中秋节", 2026): (9, 25),
            ("重阳节", 2026): (10, 18),
            ("端午节", 2026): (6, 19),
            ("元宵节", 2026): (3, 3),
            ("除夕", 2026): (2, 16),
        }
        table = {name: dates for name, _, _, dates in FESTIVALS}
        for (name, year), expected in anchors.items():
            with self.subTest(festival=name, year=year):
                self.assertEqual(table[name][year], expected)

    def test_chat_with_a_topic_answers_the_topic(self):
        """老人聊到具体的事，就接那件事，不是「嗯嗯，然后呢」地应付。

        大模型不可用时，chat 原先只有三条通用回声 —— 聊什么都是
        复述一遍。话题池让它起码接得住散步、家人、吃饭这些日常话。
        """
        from core.dialogue import topic_replies_for
        from core.vision_state import VisionState
        for text in ("我今天出门散步了",
                     "我孙子来看我了",
                     "下午在家听戏",
                     "到点了，我有点饿了"):
            with self.subTest(text=text):
                pool = topic_replies_for(text)
                self.assertTrue(pool, "%r 没有命中任何话题池" % text)
                reply = self.engine.respond(
                    text, VisionState(state=config.STATE_NORMAL))
                self.assertIn(reply.reply, pool,
                              "%r 没接住话题：%s" % (text, reply.reply))

        # 没聊到具体话题时保持原行为：走 (normal, chat) 通用池
        self.assertEqual(topic_replies_for("今天天气不错"), [])

    def test_identity_question_gets_identity_answer(self):
        """问身份时必须答身份，不能被"状态通用兜底"的套话盖掉。

        回归测试：早先候选模板的挑选顺序是先看"状态兜底"，导致"你是谁"
        被回成了"嗯，我听着呢，您说。"这种完全答非所问的套话。
        """
        from core.dialogue import INTENT_FALLBACK
        from core.vision_state import VisionState
        reply = self.engine.respond("你是谁？", VisionState(state=config.STATE_NORMAL))
        self.assertEqual(reply.intent, "identity")
        self.assertIn(reply.reply, INTENT_FALLBACK["identity"])

    def test_reply_never_empty_and_no_unfilled_placeholder(self):
        """任何输入都不能返回空回复，也不能把 {topic} 这种占位符漏给用户。"""
        from core.vision_state import VisionState
        for text in ("你好", "我好累", "心里难受", "嗯", "你是谁", "头疼", "再见",
                     "随便说说", "。。。", "asdfghjkl", "12345", "我心率多少",
                     "我呼吸怎么样"):
            for state in config.VALID_STATES:
                reply = self.engine.respond(text, VisionState(state=state))
                self.assertTrue(reply.reply.strip(), f"{text}/{state} 回复为空")
                self.assertNotIn("{", reply.reply, f"{text}/{state} 占位符没替换")
                self.assertIn(reply.state, config.VALID_STATES)

    def test_topic_echo_skipped_for_filler(self):
        """只说"嗯"的时候不该回显话题。"""
        from core.vision_state import VisionState
        reply = self.engine.respond("嗯", VisionState(state=config.STATE_NORMAL))
        self.assertNotIn("嗯——", reply.reply)

    def test_crisis_gets_the_crisis_replies(self):
        """求救信号必须走专门的危机关怀文案。

        回归测试：加这条分支之前，CRISIS_PATTERNS 只把情绪分拉到 -5.0，
        下游没有任何特判 —— "我不想活了"和"今天有点闷"落进同一组模板，
        回的是一句普通的共情套话。
        """
        from core.dialogue import CRISIS_REPLIES
        from core.vision_state import VisionState
        for state in config.VALID_STATES:
            with self.subTest(state=state):
                reply = self.engine.respond("我不想活了", VisionState(state=state))
                self.assertIn(reply.reply, CRISIS_REPLIES)

    def test_crisis_outranks_discomfort(self):
        """"我疼得不想活了"该回应"不想活"，不是"疼"。"""
        from core.dialogue import CRISIS_REPLIES
        reply = self.engine.respond("我疼得不想活了")
        self.assertIn(reply.reply, CRISIS_REPLIES)

    def test_discomfort_still_wins_over_state_templates(self):
        """身体不适那条分支没有被危机分支挤掉。

        （原用例是「我胸口疼」；紧急求助上线后它被 emergency 意图接走
        并真实通知家人 —— 那是特性不是回归，不适分支改用普通不适语句测。）
        """
        from core.dialogue import DISCOMFORT_REPLIES
        reply = self.engine.respond("我肚子有点不舒服")
        self.assertIn(reply.reply, DISCOMFORT_REPLIES)

    def test_ack_replies_are_prewarmed(self):
        """应声词（当前为空表）必须在预合成缓存里 —— 空表时无断言，仅守住结构。

        历史约束：它整句的存在意义就是"立刻出声"：要是它自己还要等一次网络
        往返（本机实测 1.2~1.6 秒），那还不如不说 —— 用户听到的是"沉默，
        然后一句废话，然后沉默，然后回答"。恢复文案后此断言必须重新生效。
        """
        from core.dialogue import ACK_REPLIES, static_replies
        prewarmed = static_replies()
        for text in ACK_REPLIES:
            with self.subTest(text=text):
                self.assertIn(text, prewarmed)

    def test_crisis_and_discomfort_replies_are_prewarmed(self):
        """这两组必须进预合成集合。

        它们早先是内联在 _candidates 里返回的字面量，而 static_replies()
        只遍历那几张字典表 —— 于是**从来没被预合成过**，表现为
        "老人说不舒服，机器先沉默三四秒"。这两类话恰恰最不能等。
        """
        from core.dialogue import CRISIS_REPLIES, DISCOMFORT_REPLIES, static_replies
        prewarmed = static_replies()
        for text in CRISIS_REPLIES + DISCOMFORT_REPLIES:
            with self.subTest(text=text):
                self.assertIn(text, prewarmed)


# ==========================================================================
# 第②步：先答内容，再追加关怀
#
# 总逻辑是三步：① 先回答用户说的内容 → ② 结合视觉判断状态 →
# ③ 需要关心就**追加**一句，正常就不打扰。
#
# 这一族测试守的就是"①不能被②顶替、③只能是追加"。
# ==========================================================================

class TestContentOutranksState(unittest.TestCase):
    """内容回答不能被状态套话顶替。"""

    def setUp(self):
        self.engine = DialogueEngine()

    def test_tired_chat_answers_the_content_not_the_state(self):
        """回归测试：整个第②步就是为这一条做的。

        改之前 ``STATE_FALLBACK[tired]`` 排在意图兜底**之前**，于是老人说
        「今天天气不错」、摄像头看他有点乏，答的是「您看着有点累了，
        先歇一会吧」—— 老人说的话被完全无视了。
        """
        from core.dialogue import PROACTIVE_TEMPLATES, STATE_FALLBACK
        from core.vision_state import VisionState
        reply = self.engine.respond("今天天气不错",
                                    VisionState(state=config.STATE_TIRED))
        self.assertEqual(reply.intent, "chat")
        self.assertNotIn(reply.reply, STATE_FALLBACK[config.STATE_TIRED])
        # 追加的那句才是状态相关的内容，且来自预合成文案
        self.assertIn(reply.follow_up, PROACTIVE_TEMPLATES["care_tired"])

    def test_chat_always_uses_the_content_pool(self):
        """表驱动：chat 在**每个**状态下都走 (normal, chat) 的内容池。"""
        from core.dialogue import REPLY_TEMPLATES
        from core.vision_state import VisionState
        pool = REPLY_TEMPLATES[(config.STATE_NORMAL, "chat")]
        for state in config.VALID_STATES:
            with self.subTest(state=state):
                reply = self.engine.respond("今天天气不错",
                                            VisionState(state=state))
                if reply.intent != "chat":
                    continue        # 视觉没改意图，这里不该发生
                filled = [t.replace("{topic}", "今天天气不错") for t in pool]
                self.assertIn(reply.reply, filled,
                              "%s 状态下的闲聊没走内容池：%s" % (state, reply.reply))

    def test_only_normal_chat_template_exists(self):
        """不变量：REPLY_TEMPLATES 里只允许有 (normal, chat)。

        曾经还有 (sad, chat) 和 (absent, chat)，它们现在**永远选不到**
        （chat 属于 CONTENT_INTENTS，_candidates 会跳过精确匹配）。
        留着是死数据，还会让人以为"低落时闲聊有专门的答法"，所以删了。
        这条测试守住它不被重新加回来。
        """
        from core.dialogue import CONTENT_INTENTS
        from core.dialogue import REPLY_TEMPLATES
        chat_keys = sorted(state for (state, intent) in REPLY_TEMPLATES
                           if intent == "chat")
        self.assertEqual(chat_keys, [config.STATE_NORMAL])
        self.assertEqual(CONTENT_INTENTS, frozenset({"chat"}))

    def test_state_sensitive_intents_still_flavor_by_state(self):
        """反过来说：**不追加**的那些轮次里，状态照旧参与选词。

        "累的时候道别"就该说「那您早点休息」，这是对的状态敏感，
        和第②步要改的那件事不是一回事。

        注意它只在**不追加**时生效 —— 要追加的那一轮内容按 normal 生成
        （关怀交给追加句，免得同一件事说两遍）。这里先把额度用掉，
        于是第二轮被节流、不追加，状态敏感那两句才轮到。
        """
        from core.dialogue import REPLY_TEMPLATES
        from core.vision_state import VisionState
        engine = DialogueEngine(care_interval=180.0)
        engine.respond("今天天气不错",
                       VisionState(state=config.STATE_TIRED))    # 用掉额度
        reply = engine.respond("那先这样吧，再见",
                               VisionState(state=config.STATE_TIRED))
        self.assertEqual(reply.follow_up, "")
        self.assertEqual(reply.intent, "farewell")
        self.assertIn(reply.reply,
                      REPLY_TEMPLATES[(config.STATE_TIRED, "farewell")])

    def test_content_goes_plain_when_care_is_appended(self):
        """要追加的那一轮，内容按 normal 生成 —— 这是"不重复"的保证。

        反例（改之前的样子）：视觉判 tired、老人说「那先这样吧，再见」，
        回答是 (tired,farewell) 的「那您早点休息，睡够了精神就好了。晚安。」，
        再追一句 care_tired 的「您先躺下歇会儿」—— 同一件事说了两遍。

        所以追加的那一轮，内容一律从 normal 那套里挑：告别就说「有空再聊」，
        关心由追加句专门负责。两段各说各的，不打架。
        """
        from core.dialogue import REPLY_TEMPLATES
        from core.vision_state import VisionState
        reply = DialogueEngine(care_interval=0.0).respond(
            "那先这样吧，再见", VisionState(state=config.STATE_TIRED))
        self.assertTrue(reply.follow_up, "这一轮应当追加")
        self.assertIn(reply.reply,
                      REPLY_TEMPLATES[(config.STATE_NORMAL, "farewell")])


class TestCareFollowUp(unittest.TestCase):
    """第三步：追加关怀的判定。"""

    def build(self, care_interval=0.0):
        # care_interval=0：这些用例验的是"该不该追加"，不是节流。
        # 节流单独在 TestCareThrottle 里测。
        return DialogueEngine(care_interval=care_interval)

    @staticmethod
    def vision(state):
        from core.vision_state import VisionState
        return VisionState(state=state)

    def test_care_only_when_the_camera_saw_it(self):
        """摄像头看到才算 —— 光凭文字说累不追加。

        追加的关怀里全是**关于身体的断言**（「我看您眼皮都沉了」），
        只能在摄像头真看到的时候说。老人打字说"我有点累"而画面里什么都
        没看出来，回答本身就已经接住了，再蹦一句身体断言就是瞎编。
        """
        from core.dialogue import PROACTIVE_TEMPLATES
        engine = self.build()
        reply = engine.respond("我有点累", self.vision(config.STATE_NORMAL))
        self.assertEqual(reply.follow_up, "",
                         "视觉没看到，不该追加身体断言式的关怀")
        # 但内容/共情该接的还得接住
        self.assertTrue(reply.reply.strip())
        self.assertNotIn(reply.reply, PROACTIVE_TEMPLATES["care_tired"])

    def test_no_care_when_the_elder_already_said_it(self):
        """他自己已经说了累，就不再追一句 —— 那是同一件事说两遍。"""
        engine = self.build()
        reply = engine.respond("我有点累，想歇会儿",
                               self.vision(config.STATE_TIRED))
        self.assertEqual(reply.follow_up, "")

    def test_no_care_for_normal_or_absent(self):
        """正常/无人不打扰。absent **刻意**不在 CARE_KINDS 里。"""
        from core.proactive import CARE_KINDS
        self.assertNotIn(config.STATE_NORMAL, CARE_KINDS)
        self.assertNotIn(config.STATE_ABSENT, CARE_KINDS)
        for state in (config.STATE_NORMAL, config.STATE_ABSENT):
            with self.subTest(state=state):
                reply = self.build().respond("今天天气不错", self.vision(state))
                self.assertEqual(reply.follow_up, "")

    def test_care_kinds_cover_sad_and_tired(self):
        """摄像头判低落/疲劳时要追加，且文案来自主动关怀那张表。"""
        from core.dialogue import PROACTIVE_TEMPLATES
        for state, kind in ((config.STATE_SAD, "care_sad"),
                            (config.STATE_TIRED, "care_tired")):
            with self.subTest(state=state):
                reply = self.build().respond("今天天气不错", self.vision(state))
                self.assertIn(reply.follow_up, PROACTIVE_TEMPLATES[kind])

    def test_crisis_never_gets_a_follow_up(self):
        """危机语句绝不追加。

        那句话要走的是逐句审过的确定性回复，而且追加的关怀文案是
        「您今天心情不太好」这种**轻描淡写**的口吻 —— 接在"不想活了"
        后面是严重的降级。crisis 分支在 _candidates 里就返回了，
        追加必须同样让路。
        """
        engine = self.build()
        reply = engine.respond("活着没意思，不想活了",
                               self.vision(config.STATE_SAD))
        self.assertTrue(reply.reply)
        self.assertEqual(reply.follow_up, "")

    def test_discomfort_never_gets_a_follow_up(self):
        """身体不适同理：先在答"疼"，别急着说"您心情不好"。"""
        reply = self.build().respond("头疼得厉害",
                                     self.vision(config.STATE_SAD))
        self.assertEqual(reply.follow_up, "")

    def test_care_shows_up_in_reason_for_defence(self):
        """答辩时要能一眼看出这轮为什么追加/没追加。"""
        engine = self.build()
        said = engine.respond("我有点累", self.vision(config.STATE_TIRED))
        self.assertIn("关怀=无(他自己说了)", said.reason)

        not_seen = self.build().respond("今天天气不错",
                                        self.vision(config.STATE_NORMAL))
        self.assertIn("关怀=无(视觉=normal不需关心)", not_seen.reason)

        added = self.build().respond("今天天气不错",
                                     self.vision(config.STATE_TIRED))
        self.assertIn("关怀=care_tired", added.reason)

    def test_vision_none_does_not_append(self):
        """不传 vision（默认 normal）时不追加 —— 离线回放/单测的默认行为。"""
        self.assertEqual(DialogueEngine(care_interval=0.0).respond("你好").follow_up, "")


class TestCareThrottle(unittest.TestCase):
    """追加关怀的节流：省的是"唠叨"，不是"该说不说"。"""

    @staticmethod
    def vision(state):
        from core.vision_state import VisionState
        return VisionState(state=state)

    def test_repeat_within_the_interval_is_suppressed(self):
        engine = DialogueEngine(care_interval=180.0)
        first = engine.respond("今天天气不错", self.vision(config.STATE_TIRED))
        second = engine.respond("刚才那事挺有意思", self.vision(config.STATE_TIRED))
        self.assertTrue(first.follow_up, "第一次必然追加（计数器从 0 起算）")
        self.assertEqual(second.follow_up, "", "间隔内的第二次不该再追")
        self.assertIn("关怀=无(未到间隔)", second.reason)

    def test_zero_interval_always_appends(self):
        engine = DialogueEngine(care_interval=0.0)
        for text in ("今天天气不错", "刚才那事挺有意思", "我孙子来了"):
            with self.subTest(text=text):
                self.assertTrue(engine.respond(
                    text, self.vision(config.STATE_TIRED)).follow_up)

    def test_blocked_round_does_not_consume_the_budget(self):
        """⚠️ 只有**真的追加了**才该记账。

        不然"视觉判了 tired、但他自己已经说了累"这种被别处挡下的轮次
        也会白扣掉这 180 秒 —— 老人接着聊十句，一句关怀都等不到。
        """
        engine = DialogueEngine(care_interval=180.0)
        engine.respond("我有点累", self.vision(config.STATE_TIRED))   # 被挡，不记账
        engine.respond("今天天气不错", self.vision(config.STATE_NORMAL))  # 视觉没看到
        reply = engine.respond("刚才那事挺有意思",
                               self.vision(config.STATE_TIRED))
        self.assertTrue(reply.follow_up,
                        "前面两轮都没有真的追加，不该把额度用掉")

    def test_per_instance_not_class_level(self):
        """计时是**实例**状态：两台会话不该互相节流。"""
        a = DialogueEngine(care_interval=180.0)
        b = DialogueEngine(care_interval=180.0)
        self.assertTrue(a.respond("今天天气不错",
                                  self.vision(config.STATE_TIRED)).follow_up)
        self.assertTrue(b.respond("今天天气不错",
                                  self.vision(config.STATE_TIRED)).follow_up)

    def test_cooldowns_are_per_kind(self):
        """交互规则：**同类型**关怀 30 秒内不重复 —— 关键在"同类"。

        刚说过「困了就歇会儿」（care_tired），紧接着老人情绪低落，
        安抚（care_sad）照样要追加 —— 两类关怀各记各的账。
        若共用一个额度，低落那轮会被疲劳的记账误伤。
        """
        engine = DialogueEngine(care_interval=180.0)
        tired = engine.respond("今天天气不错", self.vision(config.STATE_TIRED))
        self.assertTrue(tired.follow_up, "第一轮疲劳，必追加")

        sad = engine.respond("刚才那事挺有意思", self.vision(config.STATE_SAD))
        self.assertTrue(sad.follow_up,
                        "另一类关怀不受疲劳那轮冷却的影响")

        again = engine.respond("再说一个吧", self.vision(config.STATE_TIRED))
        self.assertEqual(again.follow_up, "", "同类型在间隔内仍要拦住")


class TestSpokenParts(unittest.TestCase):
    """一次回复要说出来的全部内容。"""

    @staticmethod
    def vision(state):
        from core.vision_state import VisionState
        return VisionState(state=state)

    def test_parts_are_reply_then_follow_up(self):
        reply = DialogueEngine(care_interval=0.0).respond(
            "今天天气不错", self.vision(config.STATE_TIRED))
        self.assertEqual(reply.spoken_parts(), (reply.reply, reply.follow_up))

    def test_parts_skip_the_empty_follow_up(self):
        reply = DialogueEngine(care_interval=0.0).respond(
            "今天天气不错", self.vision(config.STATE_NORMAL))
        self.assertEqual(reply.follow_up, "")
        self.assertEqual(reply.spoken_parts(), (reply.reply,))

    def test_to_dict_text_is_the_whole_utterance(self):
        """``text`` 是合成一句（8002 的消费方拿它整段显示/朗读）。"""
        reply = DialogueEngine(care_interval=0.0).respond(
            "今天天气不错", self.vision(config.STATE_TIRED))
        payload = reply.to_dict()
        self.assertEqual(payload["text"], reply.reply + reply.follow_up)
        self.assertEqual(payload["follow_up"], reply.follow_up)

    def test_to_dict_keeps_the_existing_keys(self):
        """纯新增字段，别顺手把老键改名 —— 8002 的消费方在按老键读。"""
        reply = DialogueEngine(care_interval=0.0).respond("你好")
        payload = reply.to_dict()
        for key in ("type", "text", "state", "intent", "emotion", "timestamp"):
            with self.subTest(key=key):
                self.assertIn(key, payload)

    def test_care_kind_names_the_appended_line(self):
        """``care_kind`` 是写历史 CSV 那一列用的（比 intent 准）。

        主动开口那一行写的 intent 就是 kind（见 main._fire_proactive），
        应答里的追加理应一致 —— 否则 CSV 里分不出哪几行是关怀。
        """
        reply = DialogueEngine(care_interval=0.0).respond(
            "今天天气不错", self.vision(config.STATE_TIRED))
        self.assertEqual(reply.care_kind, "care_tired")
        self.assertEqual(reply.to_dict()["care_kind"], "care_tired")

    def test_care_kind_is_empty_when_nothing_is_appended(self):
        reply = DialogueEngine(care_interval=0.0).respond(
            "今天天气不错", self.vision(config.STATE_NORMAL))
        self.assertEqual(reply.care_kind, "")
        self.assertEqual(reply.to_dict()["care_kind"], "")


# ==========================================================================
# 历史记录
# ==========================================================================

class TestHistoryStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "history.csv")
        self.store = CsvHistoryStore(self.path, session_id="test-session")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_file_with_header(self):
        self.assertTrue(os.path.exists(self.path))
        with open(self.path, encoding="utf-8") as fp:
            header = fp.readline().strip()
        self.assertEqual(header.split(","), config.HISTORY_FIELDS)

    def test_append_turn_writes_two_rows(self):
        self.store.append_turn("你好", "您好呀", config.STATE_NORMAL, "neutral", 0.0, "greeting")
        records = self.store.load_recent(10)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].role, "user")
        self.assertEqual(records[0].text, "你好")
        self.assertEqual(records[1].role, "robot")
        self.assertEqual(records[1].text, "您好呀")

    def test_roundtrip_preserves_chinese_and_floats(self):
        """中文和浮点数要能原样读回来（编码/换行符最容易在这里出问题）。"""
        self.store.append_turn("头疼得厉害，吃不下饭", "要不要我帮您叫家里人？",
                               config.STATE_SAD, "negative", -3.3, "discomfort")
        user = self.store.load_recent(10)[0]
        self.assertEqual(user.text, "头疼得厉害，吃不下饭")
        self.assertAlmostEqual(user.emotion_score, -3.3)
        self.assertEqual(user.text_emotion, "negative")

    def test_load_recent_respects_limit(self):
        for i in range(10):
            self.store.append_turn(f"第{i}句", f"回复{i}", config.STATE_NORMAL, "neutral", 0.0, "chat")
        records = self.store.load_recent(4)
        self.assertEqual(len(records), 4)
        # 必须是"最近"的 4 条，即最后写入的
        self.assertEqual(records[-1].text, "回复9")

    def test_recent_dialogue_shape(self):
        self.store.append_turn("你好", "您好呀", config.STATE_NORMAL, "neutral", 0.0, "greeting")
        dialogue = self.store.recent_dialogue(5)
        self.assertEqual(set(dialogue[0].keys()), {"role", "text"})

    def test_last_robot_replies(self):
        self.store.append_turn("一", "回复一", config.STATE_NORMAL, "neutral", 0.0, "chat")
        self.store.append_turn("二", "回复二", config.STATE_NORMAL, "neutral", 0.0, "chat")
        self.assertEqual(self.store.last_robot_replies(2), ["回复二", "回复一"])

    def test_load_session_filters(self):
        self.store.append_turn("本会话", "回复", config.STATE_NORMAL, "neutral", 0.0, "chat")
        other = CsvHistoryStore(self.path, session_id="another")
        other.append_turn("别的会话", "回复", config.STATE_NORMAL, "neutral", 0.0, "chat")
        mine = self.store.load_session("test-session")
        self.assertEqual(len(mine), 2)
        self.assertTrue(all(r.session_id == "test-session" for r in mine))

    def test_survives_corrupt_file(self):
        """历史文件坏了不能影响主流程 —— 读到哪算哪，不抛异常。"""
        with open(self.path, "a", encoding="utf-8") as fp:
            fp.write("这不是合法CSV\n")
        self.store.append_turn("你好", "您好呀", config.STATE_NORMAL, "neutral", 0.0, "chat")
        self.assertIsInstance(self.store.load_recent(10), list)

    def test_from_row_tolerates_bad_numbers(self):
        record = TurnRecord.from_row({"timestamp": "not-a-number", "emotion_score": ""})
        self.assertEqual(record.timestamp, 0.0)
        self.assertEqual(record.emotion_score, 0.0)


# ==========================================================================
# 体征现答（vitals 意图）
# ==========================================================================

class TestVitalsIntent(unittest.TestCase):
    """「我心率多少」必须报 VitalsStore 里的真实数值，并给偏高偏低与建议。

    大模型看不见传感器读数，让它答只会编一个数 —— 所以 vitals 意图
    永远不进大模型，vitals_reply 按读数现算。
    """

    def setUp(self):
        self.engine = DialogueEngine(care_interval=0.0)

    @staticmethod
    def _reading(hr=None, rr=None):
        import time
        from core.typed_messages import RppgReading
        return RppgReading(timestamp=time.time(), hr=hr, rr=rr, ibi_ms=())

    def _ask(self, text):
        from core.vision_state import VisionState
        return self.engine.respond(text, VisionState(state=config.STATE_NORMAL))

    def test_heart_rate_question_reports_the_measured_value(self):
        self.engine._vitals_provider = lambda: self._reading(hr=75, rr=16.0)
        reply = self._ask("我心率多少")
        self.assertEqual(reply.intent, "vitals")
        self.assertIn("75", reply.reply)
        self.assertIn("60到100", reply.reply)          # 带静息参考范围
        self.assertNotIn("偏高", reply.reply)
        self.assertNotIn("偏低", reply.reply)
        self.assertIn("视觉测量", reply.reply)          # 免责声明随数值走
        # vitals 永不进大模型
        emotion = self.engine.analyzer.analyze("我心率多少")
        eligible, _ = self.engine.llm_eligible("vitals", emotion, "我心率多少")
        self.assertFalse(eligible, "体征数值不能交给大模型 —— 它看不见读数")

    def test_high_and_low_bands_carry_advice(self):
        self.engine._vitals_provider = lambda: self._reading(hr=110)
        reply = self._ask("我心率快不快")
        self.assertIn("110", reply.reply)
        self.assertIn("偏高", reply.reply)
        self.assertIn("告诉家里人", reply.reply)

        self.engine._vitals_provider = lambda: self._reading(hr=52)
        reply = self._ask("我的心跳怎么样")
        self.assertIn("52", reply.reply)
        self.assertIn("偏低", reply.reply)

    def test_respiration_question_reports_only_rr(self):
        self.engine._vitals_provider = lambda: self._reading(hr=75, rr=25.0)
        reply = self._ask("我呼吸怎么样")
        self.assertEqual(reply.intent, "vitals")
        self.assertIn("25", reply.reply)
        self.assertIn("快", reply.reply)
        self.assertNotIn("心率", reply.reply, "问呼吸就别把心率也塞进来")

        self.engine._vitals_provider = lambda: self._reading(hr=75, rr=14.0)
        reply = self._ask("我呼吸正常吗")
        self.assertIn("14", reply.reply)
        self.assertNotIn("偏", reply.reply)

    def test_not_measured_falls_back_without_a_number(self):
        from core.dialogue import INTENT_FALLBACK
        self.engine._vitals_provider = lambda: self._reading(hr=None, rr=None)
        reply = self._ask("我心率多少")
        self.assertEqual(reply.intent, "vitals")
        self.assertIn(reply.reply, INTENT_FALLBACK["vitals"])

    def test_missing_or_crashing_provider_falls_back(self):
        from core.dialogue import INTENT_FALLBACK
        # 默认构造（没有 provider）
        reply = self._ask("心率多少")
        self.assertIn(reply.reply, INTENT_FALLBACK["vitals"])

        # provider 抛异常也不能连累回复
        def boom():
            raise RuntimeError("store gone")
        self.engine._vitals_provider = boom
        reply = self._ask("心率多少")
        self.assertIn(reply.reply, INTENT_FALLBACK["vitals"])

    def test_pure_function_bands_and_none(self):
        from core.dialogue import HR_RANGE, RR_RANGE, _vitals_band, vitals_reply
        self.assertEqual(_vitals_band(100, HR_RANGE), "normal")
        self.assertEqual(_vitals_band(101, HR_RANGE), "high")
        self.assertEqual(_vitals_band(60, HR_RANGE), "normal")
        self.assertEqual(_vitals_band(59, HR_RANGE), "low")
        self.assertEqual(_vitals_band(12, RR_RANGE), "normal")
        self.assertEqual(_vitals_band(11.5, RR_RANGE), "low")
        self.assertEqual(_vitals_band(20, RR_RANGE), "normal")
        self.assertEqual(_vitals_band(21, RR_RANGE), "high")
        # 没读数 → 空串（调用方落回引导模板），绝不返回"心率 0"
        self.assertEqual(vitals_reply("我心率多少", None), "")
        self.assertEqual(
            vitals_reply("我心率多少", self._reading(hr=None, rr=16.0)), "")

    def test_palpitation_wording_is_not_vitals(self):
        """「我心跳得厉害」是不适陈述 —— 不能被抢去报一个数。"""
        self.engine._vitals_provider = lambda: self._reading(hr=75)
        reply = self._ask("我心跳得厉害")
        self.assertNotEqual(reply.intent, "vitals")


# ==========================================================================
# 时间 / 日期 / 星期现答（成都时间）
# ==========================================================================

class TestTimeReply(unittest.TestCase):
    """问时间必须按本机时钟现算，绝不让没有钟的大模型瞎编。"""

    def setUp(self):
        from datetime import datetime
        from core.dialogue import CHENGDU_TZ
        # 2025-10-07 是星期二，15:20
        self.now = datetime(2025, 10, 7, 15, 20, tzinfo=CHENGDU_TZ)
        self.engine = DialogueEngine(care_interval=0.0)

    def _ask(self, text):
        from core.vision_state import VisionState
        return self.engine.respond(text, VisionState(state=config.STATE_NORMAL))

    def test_clock_only(self):
        from core.dialogue import time_reply
        answer = time_reply("现在几点", now=self.now)
        self.assertIn("3点20分", answer)
        self.assertIn("下午", answer)
        self.assertNotIn("号", answer)
        self.assertNotIn("星期", answer)

    def test_date_includes_weekday(self):
        from core.dialogue import time_reply
        answer = time_reply("今天几号", now=self.now)
        self.assertIn("10月7号", answer)
        self.assertIn("星期二", answer)

    def test_weekday_only(self):
        from core.dialogue import time_reply
        answer = time_reply("今天星期几", now=self.now)
        self.assertEqual(answer, "今天是星期二。")

    def test_jidian_with_date_keyword_not_treated_as_date(self):
        # 「今天几点」带"今天几"，但不能被当成问日期
        from core.dialogue import time_reply
        answer = time_reply("今天几点了", now=self.now)
        self.assertNotIn("号", answer)
        self.assertIn("3点20分", answer)

    def test_morning_hour_and_zero_minute(self):
        from datetime import datetime
        from core.dialogue import CHENGDU_TZ, time_reply
        morning = datetime(2025, 10, 8, 7, 0, tzinfo=CHENGDU_TZ)
        self.assertIn("早上7点整", time_reply("几点", now=morning))

    def test_engine_route_time_intent(self):
        reply = self._ask("现在几点了")
        self.assertEqual(reply.intent, "time")
        # 答案是现算的真实时间，只要结构对（含"点"和"分/整/半"之一）
        self.assertIn("点", reply.reply)


# ==========================================================================
# 天气现答（实时查询，查不到就老实说看不到）
# ==========================================================================

class TestWeatherIntent(unittest.TestCase):

    def setUp(self):
        self.engine = DialogueEngine(care_interval=0.0)

    def _ask(self, text):
        from core.vision_state import VisionState
        return self.engine.respond(text, VisionState(state=config.STATE_NORMAL))

    def test_report_from_provider_is_spoken(self):
        # 青城山 23 度 → 晴（微凉）；只报天气，建议不掺进来
        self.engine._weather_provider = lambda: {"temp": 23, "code": 0, "wind": 1.2}
        reply = self._ask("今天天气怎么样")
        self.assertEqual(reply.intent, "weather")
        self.assertIn("青城山", reply.reply)
        self.assertIn("23度", reply.reply)
        self.assertIn("晴（微凉）", reply.reply)
        self.assertNotIn("记得戴好帽子", reply.reply)
        self.assertNotIn("出门", reply.reply)

    def test_activity_question_gets_activity_answer(self):
        """天气问句只报天气；散步/锻炼类问句才回出行与锻炼建议。"""
        self.engine._weather_provider = lambda: {"temp": 18, "code": 61, "wind": 0.0}
        plain = self._ask("今天天气怎么样")
        activity = self._ask("今天适合出门散步锻炼吗")
        self.assertEqual(plain.intent, "weather")
        self.assertEqual(activity.intent, "weather")
        self.assertNotIn("出门", plain.reply)          # 天气回答不含建议
        self.assertIn("地滑", activity.reply)          # 活动回答含出门话术
        self.assertIn("活动", activity.reply)          # 活动回答含锻炼话术
        self.assertNotEqual(plain.reply, activity.reply)

    def test_none_report_falls_back_to_honest_template(self):
        from core.dialogue import INTENT_FALLBACK
        reply = self._ask("天气怎么样")          # 默认无 provider
        self.assertEqual(reply.intent, "weather")
        self.assertIn(reply.reply, INTENT_FALLBACK["weather"])

    def test_crashing_provider_falls_back(self):
        from core.dialogue import INTENT_FALLBACK

        def boom():
            raise RuntimeError("network down")
        self.engine._weather_provider = boom
        reply = self._ask("外面冷不冷")
        self.assertIn(reply.reply, INTENT_FALLBACK["weather"])

    def test_weather_never_goes_to_llm(self):
        emotion = self.engine.analyzer.analyze("天气怎么样")
        eligible, _ = self.engine.llm_eligible("weather", emotion, "天气怎么样")
        self.assertFalse(eligible)

    def test_weather_commentary_is_still_chat(self):
        # 「今天天气不错」是评价不是问句，仍然闲聊，不该触发天气播报
        self.engine._weather_provider = lambda: {"temp": 23, "code": 0, "wind": 0.0}
        reply = self._ask("今天天气不错")
        self.assertNotEqual(reply.intent, "weather")

    def test_activity_question_gets_short_advice_not_full_broadcast(self):
        # 「适合出门散步吗」也命中 weather 意图（"适合出门"在关键词表里），
        # 但它是**活动类**问句 —— 要答"能不能去"，不是把天气播报全文
        # 再念一遍（同一时段两次播报一字不差，听感就是复读机）。
        # 实测 2026-10-09：问天气和问散步播的是同一段 40+ 字长文。
        self.engine._weather_provider = lambda: {"temp": 23, "code": 0, "wind": 1.2}
        full = self._ask("今天天气怎么样").reply
        short = self._ask("晚上适合出门散步锻炼吗").reply
        self.assertIn("晴（微凉）", short)
        self.assertIn("太极", short)              # 活动专属话术（_ACTIVITY_TEMPLATES）
        self.assertNotIn("23度", short)           # 不报气温全文
        self.assertNotIn("帽子", short)
        self.assertNotEqual(full, short, "问天气和问散步答得一模一样")

    def test_story_intent_tells_a_nostalgic_story(self):
        # 「讲个故事」走**确定性**回答：从预写池挑怀旧小故事，不走大模型
        # （编的长故事未必贴合老人年代记忆，且长文本合成更容易断流）。
        # 结尾必须带互动钩子 —— 故事讲完把话头递回给老人。
        reply = self._ask("给我讲个故事")
        self.assertEqual(reply.intent, "story")
        self.assertIn("您", reply.reply)          # 适老称呼
        self.assertTrue(any(q in reply.reply for q in ("？", "您听")),
                        "故事结尾没有互动钩子")

    def test_story_twice_does_not_repeat(self):
        # 连着点播两次，第二条必须换个故事 —— 连着听同一个显得没记性。
        # story_reply 用 last 排除上次讲过的那条，两次必不同。
        first = self._ask("讲个故事").reply
        second = self._ask("再讲一个故事").reply
        self.assertNotEqual(first, second)


# ==========================================================================
# 紧急求助：意图 + 标志位 + 真实通知由 main 接线（这里只测对话层）
# ==========================================================================

class TestEmergencyIntent(unittest.TestCase):

    def setUp(self):
        self.engine = DialogueEngine(care_interval=0.0)

    def _ask(self, text):
        from core.vision_state import VisionState
        return self.engine.respond(text, VisionState(state=config.STATE_NORMAL))

    def test_body_emergency_sets_flag_and_static_reply(self):
        from core.dialogue import EMERGENCY_REPLIES
        reply = self._ask("救命啊快来人")
        self.assertEqual(reply.intent, "emergency")
        self.assertTrue(reply.emergency)
        self.assertEqual(reply.emergency_kind, "emergency")
        self.assertIn(reply.reply, EMERGENCY_REPLIES)
        # 紧急轮不追加关怀语
        self.assertEqual(reply.follow_up, "")

    def test_ambulance_and_breath_phrases(self):
        for text in ("快叫救护车", "我喘不上气了", "打120", "我摔了一跤起不来了",
                     "胸口疼得厉害", "我胸口疼"):
            reply = self._ask(text)
            self.assertEqual(reply.intent, "emergency", text)
            self.assertTrue(reply.emergency, text)

    def test_crisis_emotion_also_flags_emergency(self):
        from core.dialogue import CRISIS_REPLIES
        reply = self._ask("我不想活了")
        self.assertTrue(reply.emergency)
        self.assertEqual(reply.emergency_kind, "crisis")
        self.assertIn(reply.reply, CRISIS_REPLIES)
        self.assertEqual(reply.follow_up, "")

    def test_short_words_do_not_false_alarm(self):
        # 完整词表刻意不收裸"报警/不行了"，防误伤
        for text in ("电视里在报警", "这把年纪不行了", "救命稻草这个词怎么来的"):
            reply = self._ask(text)
            self.assertNotEqual(reply.intent, "emergency", text)
            self.assertFalse(reply.emergency, text)

    def test_emergency_never_goes_to_llm(self):
        emotion = self.engine.analyzer.analyze("救命")
        eligible, _ = self.engine.llm_eligible("emergency", emotion, "救命")
        self.assertFalse(eligible)


# ==========================================================================
# 「不想听你说话」：安静意图，只承认 + 由调度器关主动开口
# ==========================================================================

class TestQuietIntent(unittest.TestCase):

    def setUp(self):
        self.engine = DialogueEngine(care_interval=0.0)

    def _ask(self, text):
        from core.vision_state import VisionState
        return self.engine.respond(text, VisionState(state=config.STATE_NORMAL))

    def test_quiet_phrases_recognized(self):
        from core.dialogue import INTENT_FALLBACK
        for text in (
            "喔不想听到你说话",       # 用户实测原句
            "我不想听你说话",
            "你别说话了",
            "闭嘴",
            "我想安静一会",
        ):
            reply = self._ask(text)
            self.assertEqual(reply.intent, "quiet", text)
            self.assertIn(reply.reply, INTENT_FALLBACK["quiet"], text)
            self.assertEqual(reply.follow_up, "", text)

    def test_quiet_never_goes_to_llm(self):
        emotion = self.engine.analyzer.analyze("别说话")
        eligible, _ = self.engine.llm_eligible("quiet", emotion, "别说话")
        self.assertFalse(eligible)

    def test_normal_chat_is_not_quiet(self):
        reply = self._ask("我想听听戏")
        self.assertNotEqual(reply.intent, "quiet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
