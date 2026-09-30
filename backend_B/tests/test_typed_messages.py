# -*- coding: utf-8 -*-
"""api_doc §3.5 类型化报文在 B 侧的处理：分流、校验、丢弃。

跑法（在 backend_B 目录下）：
    python -m pytest tests/test_typed_messages.py -v
    python tests/test_typed_messages.py    # 不装 pytest 也能跑，见文件末尾

（文件名跟的是被测模块 ``core/typed_messages.py``。不叫 ``test_vitals.py``
是因为 ``backend_A/tests/`` 下已经有一个同名的 —— 两边的 ``tests/`` 都没有
``__init__.py``，pytest 会把同名文件当同一个顶层模块，直接收集失败。）

这个文件守的是一类**不会报错、只会静默走偏**的失效：

1. **类型化报文被当成帧。** ``VisionSample.from_payload`` 对未知键静默忽略、
   对缺失键回退默认，所以 ``{"type":"rppg",...}`` 兜出来的样本是
   ``has_face=False`` —— A 每秒钟发一条，等于**周期性伪造"老人不在"**。
   A 侧 v1 模式刻意关掉心跳躲的就是这个；新开的两条通道不能把它放回来。
2. **没人订阅时把报文当垃圾。** rppg 在 ③a 阶段确实没有消费者（B 的体征规则
   还没做），但"没消费者"必须是**计了数的丢弃**，不能是无声无息 ——
   否则联调时看不出这条通道是通的还是死的。
3. **实时对、回放错。** :class:`VisionClient` 与
   :class:`~core.vision_client.OfflineVisionFeeder` 共用同一个分流器；
   各写一份 if 的话，这种差异只会在演示当天被发现。

关于"状态有没有被搅乱"这条，**这里刻意不写断言**，理由与做法见
:meth:`TestTypedRouting.test_state_assertion_omitted_on_purpose`。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.typed_messages import (
    MSG_FOCUS,
    MSG_RPPG,
    TYPED_MESSAGE_TYPES,
    FocusReading,
    RppgReading,
    TypedMessageError,
    TypedMessageRouter,
    parse_typed,
)
from core.vision_client import OfflineVisionFeeder, VisionClient
from core.vision_state import VisionSample, VisionStateEvaluator


# --------------------------------------------------------------------------
# 报文构造
# --------------------------------------------------------------------------

def frame_line(ts: float, *, has_face: bool = True, emo: str = "normal",
               ear: float = 0.30) -> str:
    """一条 §3.2 帧报文（**不带 ``type``**）。"""
    return json.dumps({
        "timestamp": ts, "has_face": has_face, "ear": ear, "blink_cnt": int(ts * 10),
        "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "emo_feature": emo,
    })


def rppg_line(ts: float, hr=75, rr=19.0, ibi=(800, 800)) -> str:
    return json.dumps({
        "type": MSG_RPPG, "timestamp": ts, "hr": hr, "rr": rr, "ibi_ms": list(ibi),
    })


def focus_line(ts: float, gaze=0.10, quality=0.90) -> str:
    return json.dumps({
        "type": MSG_FOCUS, "timestamp": ts, "gaze": gaze, "gaze_quality": quality,
    })


def make_client() -> tuple[VisionClient, VisionStateEvaluator, list]:
    """一个连着真判定器的客户端，并记下每一个进判定器的样本。"""
    evaluator = VisionStateEvaluator()
    seen: list = []
    client = VisionClient(evaluator, on_sample=seen.append)
    return client, evaluator, seen


# ==========================================================================
# 分流
# ==========================================================================

class TestTypedRouting(unittest.TestCase):
    """带 ``type`` 的报文必须被摘走，且**不能**变成帧。"""

    def test_a_typed_packet_would_be_a_ghost_frame_if_it_fell_through(self):
        """**前提测试：先证明这个坑是真的，再证明我们堵上了。**

        这条不含任何被测代码，它钉的是"如果没人拦会怎样" ——
        ``from_payload`` 把一条 rppg 兜成什么样。写成断言而不是注释，
        是因为将来有人放宽 ``from_payload`` 的默认值时，
        这个文件里"为什么必须分流"的理由会跟着一起失效。
        """
        ghost = VisionSample.from_payload({
            "type": MSG_RPPG, "timestamp": 12.5, "hr": 75, "rr": 19.0,
            "ibi_ms": [800, 800],
        })
        self.assertFalse(ghost.has_face, "幽灵帧的前提变了，请重新读本文件的模块文档")
        self.assertEqual(ghost.ear, 0.0)
        self.assertEqual(ghost.emo_feature, "normal")

    def test_rppg_never_becomes_a_frame(self):
        """一条 rppg 都不许进判定器 —— 用 on_sample 计数，这是精确观测量。"""
        client, _ev, seen = make_client()
        client._handle_line(rppg_line(1.0))
        self.assertEqual(seen, [], "rppg 报文进了判定器")
        self.assertEqual(client.frames_received, 0)

    def test_focus_never_becomes_a_frame(self):
        client, _ev, seen = make_client()
        client._handle_line(focus_line(1.0))
        self.assertEqual(seen, [], "focus 报文进了判定器")
        self.assertEqual(client.frames_received, 0)

    def test_mixed_stream_counts_exactly_the_frames(self):
        """混合流：帧计数只数帧，两种类型各数各的。"""
        client, _ev, seen = make_client()
        for i in range(10):
            client._handle_line(frame_line(i * 0.1))
            client._handle_line(focus_line(i * 0.1))
            if i % 5 == 0:
                client._handle_line(rppg_line(i * 0.1))

        self.assertEqual(client.frames_received, 10)
        self.assertEqual(len(seen), 10, "进判定器的样本数与帧数对不上")
        self.assertEqual(client.typed.counts, {MSG_FOCUS: 10, MSG_RPPG: 2})
        self.assertEqual(client.bad_lines, 0, "类型化报文被误判成了脏数据")

    def test_unknown_type_is_dropped_not_treated_as_a_frame(self):
        """白名单外的 ``type``：丢弃 + 计数，**绝不**落进帧分支。"""
        client, _ev, seen = make_client()
        client._handle_line(json.dumps({"type": "vision_window", "timestamp": 1.0}))
        self.assertEqual(seen, [], "未知类型被当成帧了 —— 这正是幽灵帧的来源")
        self.assertEqual(client.frames_received, 0)
        self.assertEqual(client.typed.unknown, 1)
        self.assertEqual(client.typed.counts, {})

    def test_heartbeat_is_rejected_like_any_other_unknown_type(self):
        """``heartbeat`` 是 v2 的报文名，混进 v1 流必须被丢。

        A 侧 v1 模式**刻意不发**心跳，理由就是它会变成幽灵帧
        （见 backend_A 的 ``server.serve``）。这条从 B 这一侧把同一件事钉住。
        """
        client, _ev, seen = make_client()
        client._handle_line(json.dumps({"type": "heartbeat", "timestamp": 1.0}))
        self.assertEqual(seen, [])
        self.assertEqual(client.typed.unknown, 1)

    def test_type_null_is_not_a_frame(self):
        """``"type": null`` 也算"带 type"，按未知类型丢弃。

        边界情形：写 ``if payload.get("type")`` 而不是 ``if "type" in payload``
        的话，``null`` 会滑进帧分支。判据要按 api_doc §3.5.1 写成
        「**有没有** ``type``」，不是「``type`` 真不真」。
        """
        client, _ev, seen = make_client()
        client._handle_line(json.dumps({"type": None, "timestamp": 1.0}))
        self.assertEqual(seen, [], "type=null 滑进了帧分支")
        self.assertEqual(client.typed.unknown, 1)

    def test_malformed_typed_is_counted_apart_from_unknown(self):
        """字段不合规与类型不认识是两件事，计数分开 —— 排查方向不同。"""
        client, _ev, _seen = make_client()
        client._handle_line(rppg_line(1.0, hr=5000))       # 类型认识，值越界
        client._handle_line(json.dumps({"type": "nope"}))  # 类型不认识
        self.assertEqual(client.typed.malformed, 1)
        self.assertEqual(client.typed.unknown, 1)
        self.assertEqual(client.typed.counts, {})

    def test_bad_lines_still_counts_real_garbage(self):
        """分流不该把 ``bad_lines`` 的语义冲淡 —— 它只该数非 JSON。"""
        client, _ev, _seen = make_client()
        client._handle_line("这不是 JSON")
        self.assertEqual(client.bad_lines, 1)
        self.assertEqual((client.typed.unknown, client.typed.malformed), (0, 0))

    def test_state_assertion_omitted_on_purpose(self):
        """为什么这里**没有**"状态没被搅乱"的断言 —— 写下来免得后来人补一条假的。

        看起来该加一条："喂一堆 rppg 之后状态还是 normal，所以没被搅乱"。
        但那条**永远会通过**，不管有没有 bug：判定器的防抖要
        ``STATE_MIN_HOLD``（1.5 秒）的稳定时间才肯换状态，而在一个跑不到
        1 毫秒的测试里，几十条幽灵样本只够设上候选、不够生效。
        一条恒真的断言比没有断言更糟 —— 它会让人以为这里被守住了。

        真正能看见泄漏的是上面那些**计数**：``frames_received``、
        ``on_sample`` 的调用次数、``typed.counts``。幽灵帧一旦出现，
        它们的数字立刻不对，不需要等防抖。

        （端到端的"状态确实没抖"由 ``backend_A/tests/test_interop_with_b.py``
        的 :class:`TestAgainstRealEvaluator` 负责，它跑的是真实 socket
        与真实时间轴。这里只钉前提与计数。）
        """
        client, _ev, _seen = make_client()
        for i in range(50):
            client._handle_line(rppg_line(i * 0.1))
        self.assertEqual(client.typed.counts[MSG_RPPG], 50)
        self.assertEqual(client.frames_received, 0)


# ==========================================================================
# 解析与校验
# ==========================================================================

class TestRppgParsing(unittest.TestCase):

    def test_full_packet(self):
        reading = parse_typed({"type": MSG_RPPG, "timestamp": 12.5, "hr": 75,
                               "rr": 19.1, "ibi_ms": [780, 780, 546]})
        self.assertIsInstance(reading, RppgReading)
        self.assertEqual(reading.hr, 75)
        self.assertAlmostEqual(reading.rr, 19.1, places=4)
        self.assertEqual(reading.ibi_ms, (780, 780, 546))
        self.assertTrue(reading.has_pulse)

    def test_null_hr_is_normal_not_an_error(self):
        """**置灰是正常态。** 窗口未满 / SNR<1.5 / 帧率<5Hz 都会走这一支。

        任何"收到 rppg ⇒ hr 是数字"的实现都会在服务刚起来时误报 ——
        而服务刚起来恰恰是演示时最常看的那几秒。
        """
        reading = parse_typed({"type": MSG_RPPG, "timestamp": 1.0,
                               "hr": None, "rr": None, "ibi_ms": []})
        self.assertIsNone(reading.hr)
        self.assertIsNone(reading.rr)
        self.assertEqual(reading.ibi_ms, ())
        self.assertFalse(reading.has_pulse)

    def test_missing_optional_fields_are_tolerated(self):
        """只带 timestamp 也要收 —— A 置灰时就是这么发的。"""
        reading = parse_typed({"type": MSG_RPPG, "timestamp": 1.0})
        self.assertIsNone(reading.hr)

    def test_extra_keys_are_tolerated(self):
        """接收方按 Postel 定律容忍多余的键。

        A 将来给 rppg 加一个有意义的字段时，旧版 B 应当继续工作，
        而不是整条报文被拒（那是 A 侧 `assert_*_contract` 的严格度，
        用在接收方会把升级变成同步改造）。
        """
        reading = parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "hr": 75,
                               "spo2": 98})
        self.assertEqual(reading.hr, 75)

    def test_hr_out_of_range_is_rejected(self):
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "hr": 5000})
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "hr": 5})

    def test_bool_is_not_a_number(self):
        """``"hr": true`` 不许变成心率 1。

        ``isinstance(True, int)`` 是 ``True``，不显式排除 bool 的话，
        JSON 里的布尔值会被静默当成数字 —— 1 恰好越界所以这里能拦住，
        但同一个漏洞在 ``gaze`` 上是 ``true → 1.0``，也就是**专注度满分**。
        """
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "hr": True})

    def test_missing_timestamp_is_rejected(self):
        """时间戳**必填**：③b 的基线窗与节流全靠它，缺了只能偷偷换墙钟。"""
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "hr": 75})

    def test_ibi_is_validated(self):
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "ibi_ms": [10]})
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "ibi_ms": [800.5]})
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "ibi_ms": "八百"})
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_RPPG, "timestamp": 1.0,
                         "ibi_ms": [800] * 100})

    def test_non_finite_numbers_are_rejected_cleanly(self):
        """**``nan`` / ``inf`` 必须走 ``TypedMessageError``，不能漏成裸异常。**

        这条守的是一个真实的崩溃路径：``float("nan")`` 是能成功的，它会一路
        走到 ``int(value)`` 才炸，而那时抛的是裸的 ``ValueError``（``inf`` 则是
        ``OverflowError``）—— 不是 :class:`TypedMessageError`，路由器的 ``except``
        接不住。于是它会穿过 ``_read_loop`` 与 ``run()`` 的 except 集合，
        **把视觉读取线程打死**，状态停在上一次的值不再更新。
        """
        for bad in (float("nan"), float("inf"), "-inf"):
            with self.subTest(value=bad):
                with self.assertRaises(TypedMessageError) as ctx:
                    parse_typed({"type": MSG_RPPG, "timestamp": 1.0,
                                 "ibi_ms": [bad]})
                self.assertNotIsInstance(ctx.exception, (OverflowError,))
        for bad in ("nan", "inf", "NaN", "Infinity"):
            with self.subTest(field="hr", value=bad):
                with self.assertRaises(TypedMessageError):
                    parse_typed({"type": MSG_RPPG, "timestamp": 1.0, "hr": bad})
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_FOCUS, "timestamp": 1.0,
                         "gaze": "nan", "gaze_quality": 0.9})

    def test_numbers_as_strings_are_accepted(self):
        """CSV / 抓包 dump 里每个格子都是字符串，实时是数字 —— 两样都得认。

        ``--offline`` 指着 dump 跑时，这一条决定"回放能不能复现实时"。
        """
        reading = parse_typed({"type": MSG_RPPG, "timestamp": "12.50", "hr": "75",
                               "rr": "", "ibi_ms": "[780, 780]"})
        self.assertEqual(reading.timestamp, 12.5)
        self.assertEqual(reading.hr, 75)
        self.assertIsNone(reading.rr, "空串应当按 null 处理")
        self.assertEqual(reading.ibi_ms, (780, 780))


class TestFocusParsing(unittest.TestCase):

    def test_normal(self):
        reading = parse_typed({"type": MSG_FOCUS, "timestamp": 3.0,
                               "gaze": 0.12, "gaze_quality": 0.9})
        self.assertIsInstance(reading, FocusReading)
        self.assertAlmostEqual(reading.gaze, 0.12, places=4)
        self.assertTrue(reading.usable)

    def test_null_gaze_needs_zero_quality(self):
        reading = parse_typed({"type": MSG_FOCUS, "timestamp": 3.0,
                               "gaze": None, "gaze_quality": 0.0})
        self.assertIsNone(reading.gaze)
        self.assertFalse(reading.usable)

    def test_gaze_with_zero_quality_is_rejected(self):
        """有值却说"没质量" —— 不自洽，丢弃。

        这一支对应的现实是：A 在**没有虹膜关键点**的模型上估不出视线，
        而"估不出来"和"完全对正"在数值上都是 0.0。若有人把 None 改写回
        0.0，专注度就会把"看不见眼睛"读成满分。
        """
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_FOCUS, "timestamp": 3.0,
                         "gaze": 0.0, "gaze_quality": 0.0})

    def test_null_gaze_with_quality_is_rejected(self):
        """反方向一样要拦：说好了没值，就不该同时声称质量非零。"""
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_FOCUS, "timestamp": 3.0,
                         "gaze": None, "gaze_quality": 0.8})

    def test_gaze_range_is_enforced(self):
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_FOCUS, "timestamp": 3.0,
                         "gaze": 1.5, "gaze_quality": 0.9})

    def test_bool_gaze_would_be_perfect_attention(self):
        """``"gaze": true`` 的危险在于它是 **1.0 = 完全对正**，不是报错。"""
        with self.assertRaises(TypedMessageError):
            parse_typed({"type": MSG_FOCUS, "timestamp": 3.0,
                         "gaze": True, "gaze_quality": 0.9})


# ==========================================================================
# 路由器本身
# ==========================================================================

class TestRouter(unittest.TestCase):

    def test_frame_passes_through_untouched(self):
        router = TypedMessageRouter()
        self.assertFalse(router.handle(json.loads(frame_line(1.0))))
        self.assertEqual((router.counts, router.unknown, router.malformed), ({}, 0, 0))

    def test_callback_receives_kind_and_reading(self):
        got = []
        router = TypedMessageRouter(on_reading=lambda kind, r: got.append((kind, r)))
        router.handle(json.loads(focus_line(1.0)))
        router.handle(json.loads(rppg_line(2.0)))
        self.assertEqual([kind for kind, _ in got], [MSG_FOCUS, MSG_RPPG])
        self.assertIsInstance(got[0][1], FocusReading)
        self.assertIsInstance(got[1][1], RppgReading)

    def test_no_subscriber_means_counted_discard(self):
        """没订阅者时也必须计数 —— 否则联调时分不清"通道是通的"还是"死了"。"""
        router = TypedMessageRouter()
        for i in range(5):
            self.assertTrue(router.handle(json.loads(rppg_line(i))))
        self.assertEqual(router.counts[MSG_RPPG], 5)

    def test_callback_exception_does_not_break_the_router(self):
        def boom(kind, reading):
            raise RuntimeError("订阅方炸了")

        router = TypedMessageRouter(on_reading=boom)
        self.assertTrue(router.handle(json.loads(rppg_line(1.0))))
        self.assertEqual(router.counts[MSG_RPPG], 1)

    def test_parser_bug_is_swallowed_and_counted_separately(self):
        """**路由器承诺不抛。** 连它自己的解析器出 bug 也不许抛。

        这条用一个坏掉的解析器来验证兜底那一层：真放出去的话，
        异常会穿过 ``_read_loop`` 与 ``run()`` 的 except 集合，
        把视觉读取线程打死 —— 而那是"某一行数据脏了"演变成"整个视觉
        通道永久停摆"的路径。
        """
        import core.typed_messages as tm

        original = tm._PARSERS[MSG_RPPG]

        def broken(payload):
            raise OverflowError("模拟解析器 bug")

        tm._PARSERS[MSG_RPPG] = broken
        try:
            router = TypedMessageRouter()
            self.assertTrue(router.handle(json.loads(rppg_line(1.0))))
        finally:
            tm._PARSERS[MSG_RPPG] = original

        self.assertEqual(router.errors, 1, "解析器异常没被计进 errors")
        self.assertEqual(router.malformed, 0, "解析器 bug 被误记成了对端数据脏")
        self.assertIn("bug", router.summary())

    def test_summary_mentions_the_discards(self):
        router = TypedMessageRouter()
        router.handle(json.loads(rppg_line(1.0)))
        router.handle(json.loads(json.dumps({"type": "nope"})))
        summary = router.summary()
        self.assertIn(MSG_RPPG, summary)
        self.assertIn("未知类型", summary)

    def test_summary_says_so_when_nothing_arrived(self):
        """**这条是本文件的联调守卫。**

        体征/视线一条都没到时，摘要必须明说"未收到"，而不是打印一个空字符串 ——
        空字符串在终端里和"没打这行日志"看不出区别。
        """
        self.assertIn("未收到", TypedMessageRouter().summary())

    def test_all_known_types_are_parseable(self):
        """白名单里的每个类型都必须真有解析器。

        加类型时漏了 ``_PARSERS`` 的话，它会一直被计入 ``malformed`` 而
        永远解析不出来 —— 表现是"这条通道有数据但没人用"，不报错。
        判据是**报错内容**：说"未知报文类型"就说明解析器压根没登记。
        """
        for kind in TYPED_MESSAGE_TYPES:
            with self.subTest(kind=kind):
                with self.assertRaises(TypedMessageError) as ctx:
                    parse_typed({"type": kind})   # 缺 timestamp，必然被拒
                self.assertNotIn(
                    "未知报文类型", str(ctx.exception),
                    f"{kind} 在白名单里却没有解析器，会一直被丢弃",
                )


# ==========================================================================
# 离线回放：与实时对称
# ==========================================================================

class TestOfflineFeederSymmetry(unittest.TestCase):
    """``--offline`` 指着抓包 dump 时，typed 行不能变成幽灵帧。"""

    #: api_doc §3.4 的固定列序，再补上 §3.5 的四个类型化列。
    FIELDS = ("timestamp", "has_face", "ear", "blink_cnt", "pitch", "yaw", "roll",
              "emo_feature", "type", "hr", "rr", "ibi_ms", "gaze", "gaze_quality")

    def _write(self, rows: list[str]) -> str:
        fd, path = tempfile.mkstemp(suffix=".csv", text=True)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fp:
            fp.write(",".join(self.FIELDS) + "\n")
            for row in rows:
                fp.write(row + "\n")
        self.addCleanup(os.unlink, path)
        return path

    @classmethod
    def _row(cls, **kw) -> str:
        """按固定列序拼一行。

        **不要手写逗号串。** 漏一个逗号的话 ``DictReader`` 会把尾巴的列填成
        ``None`` —— 而 ``"type": None`` 仍然满足"带 type"，于是帧行会被当成
        未知类型丢掉。那样测试会以"帧数对不上"的形式红，而真正的原因在逗号上，
        排查方向会被带偏。
        """
        unknown = set(kw) - set(cls.FIELDS)
        assert not unknown, f"列名不在 api_doc 的列序里：{sorted(unknown)}"
        return ",".join(str(kw.get(name, "")) for name in cls.FIELDS)

    def test_typed_rows_do_not_become_samples(self):
        path = self._write([
            self._row(timestamp=0.0, has_face="true", ear=0.3, blink_cnt=0,
                      emo_feature="normal"),
            self._row(timestamp=0.1, has_face="true", ear=0.3, blink_cnt=1,
                      emo_feature="normal", type=MSG_RPPG, hr=75, rr=19.0,
                      ibi_ms="[800]"),
            self._row(timestamp=0.2, has_face="true", ear=0.3, blink_cnt=2,
                      emo_feature="normal"),
            self._row(timestamp=0.3, has_face="true", ear=0.3, blink_cnt=3,
                      emo_feature="normal", type=MSG_FOCUS, gaze=0.12,
                      gaze_quality=0.9),
        ])
        evaluator = VisionStateEvaluator()
        seen: list = []
        feeder = OfflineVisionFeeder(path, evaluator, on_sample=seen.append,
                                     speed=1000.0, loop=False)

        feeder.run()

        self.assertEqual(len(seen), 2, f"dump 里的 typed 行变成了样本：{len(seen)} 条")
        self.assertEqual(feeder.frames_received, 2)
        self.assertEqual(feeder.typed.counts, {MSG_RPPG: 1, MSG_FOCUS: 1})

    def test_typed_rows_skip_the_pacing(self):
        """typed 行不该参与 ``previous_ts`` —— 它和同刻的帧共用时间戳。

        **靠耗时来判，不是靠帧数。** 两帧在 t=0.0 与 t=0.6，中间夹一条
        t=9999 的 rppg 行：

        * 正确 —— typed 行不碰 ``previous_ts``，下一帧按 0.0→0.6 睡 **0.6s**；
        * 泄漏 —— typed 行把 ``previous_ts`` 抬到 9999，下一帧的差值成了负数，
          落进"差值异常"的兜底分支，只睡 **0.1s**。

        帧数两种情况下都是 2，所以只有计时能分辨。
        """
        path = self._write([
            self._row(timestamp=0.0, has_face="true", ear=0.3, blink_cnt=0,
                      emo_feature="normal"),
            self._row(timestamp=9999.0, has_face="true", ear=0.3, blink_cnt=1,
                      emo_feature="normal", type=MSG_RPPG, hr=75, rr=19.0,
                      ibi_ms="[800]"),
            self._row(timestamp=0.6, has_face="true", ear=0.3, blink_cnt=2,
                      emo_feature="normal"),
        ])
        feeder = OfflineVisionFeeder(path, VisionStateEvaluator(),
                                     speed=1.0, loop=False)

        started = time.monotonic()
        feeder.run()
        elapsed = time.monotonic() - started

        self.assertEqual(feeder.frames_received, 2)
        self.assertEqual(feeder.typed.counts, {MSG_RPPG: 1})
        self.assertGreaterEqual(
            elapsed, 0.4,
            f"只用了 {elapsed:.2f}s，说明 typed 行参与了对齐 —— "
            f"它会把后面的帧时间轴算歪",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
