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
                     "随便说说", "。。。", "asdfghjkl", "12345"):
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
        """身体不适那条分支没有被危机分支挤掉。"""
        from core.dialogue import DISCOMFORT_REPLIES
        reply = self.engine.respond("我胸口疼")
        self.assertIn(reply.reply, DISCOMFORT_REPLIES)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
