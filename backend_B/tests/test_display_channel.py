# -*- coding: utf-8 -*-
"""专注度**展示**通道（B→C，``vai`` 报文）：一条只读旁路。

为什么要单独一个文件、并且写得比一条"能发出去"的测试多得多：

  1. 这条通道**改了一条规格**。需求文档 §4 B2 与 ``--focus-silence`` 的帮助文本
     原本都写着「B 内部使用，**不下发 C**」。现在下发，是因为用户要拿它做前端
     右栏的展示。规格一改，最该被钉住的就是"**只**多了展示、别的什么都没变"。
  2. 它的危险不在于发不出去，而在于**发的时候顺手多做了事**。VAI 是 B 侧
     唯一能关掉主动关怀的信号（``should_stay_silent``）；展示路径每 0.2 秒跑一次，
     只要它顺手碰一下判决、或顺手推一下状态，就会以 5Hz 的频率污染静默边沿，
     而且**看起来一切正常**。
  3. ``vai`` 报文刻意不带 ``state`` 字段。带了它就会变成第二个状态源，
     去和 8001/8002 的 state 打架（那两条还有 C 端的优先级去重）。

所以下面四类断言都要在：**报文内容**（逐字段原样转发、无 state）、
**发送节流**（变化 / 心跳 / 补发）、**开关解耦**（与 --focus-silence 互不影响）、
以及四条硬约束（每条一条测试）。
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock

_BACKEND_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_B)

#: backend_B/main.py 在本进程里的专用模块名。**不能叫 "main"** ——
#: 整仓跑 pytest 时 backend_A 的测试会先注册 sys.modules["main"]。
#: 理由详见 tests/test_state_fanout.py 里同名函数的注释。
_MAIN_MODULE_NAME = "backend_b_main"

import config
import core.focus as focus_lib
import core.ui_channel as ui_channel
from core.focus import FocusSnapshot, FocusStatus
from core.ui_channel import ChatServer, VAI_NOTE, build_vai_message


def load_backend_b_main():
    """按文件路径加载 backend_B/main.py（理由同上）。"""
    cached = sys.modules.get(_MAIN_MODULE_NAME)
    if cached is not None:
        return cached
    path = os.path.join(_BACKEND_B, "main.py")
    spec = importlib.util.spec_from_file_location(_MAIN_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MAIN_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def valid_snapshot(**overrides) -> FocusSnapshot:
    """一个"真的算出来了"的快照（``usable`` 为真），字段可覆盖。"""
    fields = dict(
        index=63.4567,
        index_status="研究趋势（非认知专注）",
        status=FocusStatus.VALID,
        reason="校准已锁定，有效观察 42s",
        available_modalities=("head_pose", "eye_open", "gaze"),
        recent_modalities=("head_pose", "eye_open", "gaze"),
        modality_config_id="hp+eo+gaze",
        valid_seconds=42.0,
    )
    fields.update(overrides)
    return FocusSnapshot(**fields)


class SnapshotOnlyTracker:
    """**只肯交出 ``snapshot()``** 的 tracker 替身。别的属性一律炸。

    这是"四条硬约束"里最硬的一条的实现方式：不用断言"某个方法没被调用"
    （那只挡得住你想到的那个名字），而是让**任何**越界访问当场失败 ——
    包括将来才会写出来的新方法名。
    """

    def __init__(self, snapshot):
        self._snapshot = snapshot
        #: 记录被调用的次数，好确认这条通道真的跑到了
        self.snapshot_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return self._snapshot

    def __getattr__(self, name):
        raise AssertionError(
            f"展示路径不该碰 FocusTracker.{name} —— 它只允许调用 snapshot()"
        )


def make_server(**kwargs) -> ChatServer:
    """造一个没 start() 过的 ChatServer（不监听端口、不碰网络）。"""
    kwargs.setdefault("on_chat", lambda text: None)
    return ChatServer(**kwargs)


def sent_messages(client) -> list:
    """把发到假 socket 上的字节解回字典列表。"""
    return [json.loads(call[0][0].decode("utf-8"))
            for call in client.sendall.call_args_list]


# ===========================================================================
# 报文内容
# ===========================================================================

class TestBuildVaiMessage(unittest.TestCase):
    """``build_vai_message``：只做投影，一个字段都不许加工。"""

    def test_carries_every_field_the_frontend_may_need(self):
        msg = build_vai_message(valid_snapshot())
        self.assertEqual(msg["type"], "vai")
        self.assertEqual(msg["index"], 63.5)          # 一位小数
        self.assertEqual(msg["status"], FocusStatus.VALID.value)
        self.assertEqual(msg["modalities"], ["head_pose", "eye_open", "gaze"])
        self.assertEqual(msg["recent_modalities"], ["head_pose", "eye_open", "gaze"])
        self.assertEqual(msg["modality_config_id"], "hp+eo+gaze")
        self.assertEqual(msg["valid_seconds"], 42.0)
        self.assertEqual(msg["formula_version"], focus_lib.FORMULA_VERSION)
        self.assertEqual(msg["weight_version"], focus_lib.WEIGHT_VERSION)
        self.assertIn("timestamp", msg)

    def test_has_no_state_field(self):
        """**这条是防第二个状态源。**

        ``vai`` 一旦带上 ``state``，早晚有人把它接到表情上，于是「专注度」
        变成 B 的第三条状态通道，去和 8001/8002 打架 —— 而那两个是 api_doc
        定了的、C 端还按优先级去重。专注度只该是**另一栏数字**。
        """
        msg = build_vai_message(valid_snapshot())
        self.assertNotIn("state", msg)
        self.assertNotIn("reason_state", msg)

    def test_missing_index_stays_none_and_is_not_zero(self):
        """``index is None`` = **没有分数**（校准没锁定 / 有效观察不够），不是 0 分。

        前端要分开显示（「专注度：未提供」而不是「专注度 0.0」）——
        后者会让老人以为系统在说他走神。
        """
        msg = build_vai_message(FocusSnapshot())     # 全默认值 = 从没算出来过
        self.assertIsNone(msg["index"])
        self.assertEqual(msg["status"], FocusStatus.NO_FACE.value)

    def test_status_is_read_as_chinese_value_not_enum_repr(self):
        """枚举取 ``.value``；漏了它，C 端会显示 ``FocusStatus.VALID``。"""
        msg = build_vai_message(valid_snapshot())
        self.assertNotIn("FocusStatus", msg["status"])

    def test_index_status_is_forwarded_verbatim(self):
        """**不许改写这句话。** 它是这套指数自带的安全声明。

        「研究趋势（非认知专注）」把"这不是认知专注"写死在里面，
        换成"专注度良好"之类的顺口说法，等于替它下了它明确没下的结论。
        """
        msg = build_vai_message(valid_snapshot())
        self.assertEqual(msg["index_status"], "研究趋势（非认知专注）")

    def test_note_keeps_both_halves(self):
        """note 的两半各自有出处，缺一不可（见 ui_channel.VAI_NOTE 注释）。"""
        msg = build_vai_message(valid_snapshot())
        self.assertIn("非认知专注", msg["note"])
        self.assertIn("非医疗结论", msg["note"])
        self.assertEqual(msg["note"], VAI_NOTE)

    def test_works_on_a_duck_typed_snapshot(self):
        """鸭子类型：不 import FocusTracker 也能发（传输层不该认识领域对象）。

        给的是个**不是** FocusSnapshot 的对象 —— 只要它"长得像快照"。
        """
        duck = mock.Mock(
            index=None, index_status="不可用", status=None, reason="",
            available_modalities=(), recent_modalities=(),
            modality_config_id="", valid_seconds=0.0,
            formula_version="v1", weight_version="w1",
        )
        msg = build_vai_message(duck)
        self.assertEqual(msg["type"], "vai")
        self.assertIsNone(msg["index"])
        self.assertEqual(msg["status"], "")


# ===========================================================================
# 发送节流：变化 / 心跳 / 补发
# ===========================================================================

class TestPublishVaiThrottling(unittest.TestCase):
    """``publish_vai``：0.2 秒一拍，但只在**变化**或**心跳到期**时才真发。"""

    def setUp(self):
        self.server = make_server()
        self.addCleanup(self.server.stop)

    def test_first_snapshot_is_sent(self):
        self.assertTrue(self.server.publish_vai(valid_snapshot()))

    def test_unchanged_snapshot_is_not_resent(self):
        """指数是秒级平滑的慢变量。每拍都发的话 C 每秒要处理 5 条一模一样的东西。"""
        self.server.publish_vai(valid_snapshot())
        self.assertFalse(self.server.publish_vai(valid_snapshot()))

    def test_a_new_index_is_sent(self):
        self.server.publish_vai(valid_snapshot())
        self.assertTrue(self.server.publish_vai(valid_snapshot(index=71.2)))

    def test_a_status_change_alone_is_sent_even_with_same_index(self):
        """状态从"有效"掉到别的档（比如"眼部遮挡"）—— 也必须发。

        只比分数的话，C 端会继续把一个已经不可信的旧值显示成有效测量。
        """
        self.server.publish_vai(valid_snapshot())
        self.assertTrue(self.server.publish_vai(
            valid_snapshot(status=FocusStatus.EYE_OCCLUDED)))

    def test_a_modality_change_alone_is_sent(self):
        """模态一变，**同一个分值不再等价**（权重重归一化了）。

        签名里必须带 modality_config_id：只比分数会让 C 端停留在
        上一组证据的解读上。
        """
        self.server.publish_vai(valid_snapshot())
        degraded = valid_snapshot(modality_config_id="hp+eo")
        self.assertTrue(self.server.publish_vai(degraded))

    def test_index_becoming_unavailable_is_sent(self):
        """从"有分数"到"没有分数"是最需要让界面知道的一次变化。"""
        self.server.publish_vai(valid_snapshot())
        self.assertTrue(self.server.publish_vai(FocusSnapshot()))

    def test_heartbeat_resends_after_the_interval(self):
        """没变化也要按心跳重发 —— 否则 C 端分不清「数值没变」与「B 挂了」。

        把上次发送时刻推到很久以前来模拟心跳到期，不去 sleep。
        """
        self.server.publish_vai(valid_snapshot())
        self.server._last_vai_at -= config.VAI_HEARTBEAT_SECONDS + 1.0
        self.assertTrue(self.server.publish_vai(valid_snapshot()))

    def test_none_snapshot_is_not_sent(self):
        """没有 tracker（``--no-focus-silence``）时上游会给 None，别发空报文。"""
        self.assertFalse(self.server.publish_vai(None))

    def test_force_bypasses_dedup(self):
        """``force`` 是给补发留的口子（新客户端连上时用）。"""
        self.server.publish_vai(valid_snapshot())
        self.assertTrue(self.server.publish_vai(valid_snapshot(), force=True))


class TestSendInitialVai(unittest.TestCase):
    """新客户端连上时补发：与 ``_send_initial_state`` 同一条理由。

    指数变化才发，而它是个慢变量 —— 不补发的话，新连上的 C 右栏要盯着
    「专注度：未提供」一直等到下一次变化或心跳。
    """

    def setUp(self):
        self.server = make_server()
        self.addCleanup(self.server.stop)
        self.client = mock.Mock()
        self.addr = ("127.0.0.1", 55555)

    def test_resends_current_snapshot_and_logs(self):
        self.server._vai_provider = lambda: valid_snapshot()
        with self.assertLogs("core.ui_channel", level="INFO") as captured:
            self.server._send_initial_vai(self.client, self.addr)

        msgs = sent_messages(self.client)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["type"], "vai")
        self.assertEqual(msgs[0]["index"], 63.5)
        self.assertIn("补发专注度", "".join(captured.output))

    def test_does_nothing_without_a_provider(self):
        """没开专注静默 ⇒ 注入的是 None ⇒ 整条通道不存在，一条都不发。"""
        self.server._vai_provider = None
        self.server._send_initial_vai(self.client, self.addr)
        self.client.sendall.assert_not_called()

    def test_does_nothing_when_provider_returns_none(self):
        self.server._vai_provider = lambda: None
        self.server._send_initial_vai(self.client, self.addr)
        self.client.sendall.assert_not_called()

    def test_provider_failure_never_drops_the_connection(self):
        """取快照失败**不掐连接** —— 专注度只是一栏展示。

        状态那条（``_send_initial_state``）失败要断，因为它语义必需；
        这条是锦上添花，为它断掉整条对话通道是本末倒置。
        """
        def boom():
            raise RuntimeError("tracker 炸了")

        self.server._vai_provider = boom
        with self.assertLogs("core.ui_channel", level="ERROR"):
            self.server._send_initial_vai(self.client, self.addr)   # 不该抛
        self.client.sendall.assert_not_called()


# ===========================================================================
# 四条硬约束
# ===========================================================================

class TestDisplayPathIsReadOnly(unittest.TestCase):
    """``BackendB._publish_vai`` 只许读快照，碰别的任何东西都是 bug。

    这些测试拿 duck-typed 替身直接调未绑定方法，而不是构造整个 BackendB
    （那要拉起视觉源、历史文件、语音层）。替身刚好够钉住"它调了谁"。
    """

    def make_stub(self, *, vai_display=True, tracker=None):
        stub = mock.Mock()
        stub.args.vai_display = vai_display
        stub.focus_tracker = tracker
        return stub

    def _call(self, stub):
        load_backend_b_main().BackendB._publish_vai(stub)

    def test_only_snapshot_is_touched(self):
        """约束 1+2+3+4 的实现式断言：tracker 只肯给 ``snapshot()``。

        用 SnapshotOnlyTracker 而不是 assert_not_called，是因为它挡得住
        **任何**越界访问（含将来新加的方法名），而不只是我此刻想到的那几个。
        """
        tracker = SnapshotOnlyTracker(valid_snapshot())
        stub = self.make_stub(tracker=tracker)
        self._call(stub)

        self.assertEqual(tracker.snapshot_calls, 1)
        stub.chat_server.publish_vai.assert_called_once()
        stub._publish_state.assert_not_called()
        stub._focus_liveness.assert_not_called()

    def test_does_not_touch_the_silence_decision(self):
        """约束 2：不调 ``should_stay_silent`` / ``focus_is_live``。

        展示绝不能反过来影响静默 —— 那是 B 侧唯一的"闭嘴"信号，
        而展示路径每 0.2 秒跑一次。
        """
        main = load_backend_b_main()
        stub = self.make_stub(tracker=SnapshotOnlyTracker(valid_snapshot()))
        with mock.patch.object(main.focus_lib, "should_stay_silent") as silent, \
                mock.patch.object(main.focus_lib, "focus_is_live") as live:
            self._call(stub)
        silent.assert_not_called()
        live.assert_not_called()

    def test_does_not_log_silence_transitions(self):
        """约束 3：不调 ``_focus_liveness`` —— 它有日志副作用。

        它进出静默各打一条日志（靠 ``_focus_logged`` 在边沿上去重）。
        展示路径按 0.2 秒的节拍调它，会把日志刷爆，也会把静默边沿弄脏 ——
        真的进出静默时那两条日志反而不打了。
        """
        stub = self.make_stub(tracker=SnapshotOnlyTracker(valid_snapshot()))
        self._call(stub)
        stub._focus_liveness.assert_not_called()

    def test_does_not_publish_state(self):
        """约束 4：不调 ``_publish_state``。

        ``vai`` 报文不带 ``state`` 字段（见 TestBuildVaiMessage），
        发送路径也绝不能顺手推一个状态 —— 那会让 C 的表情跟着专注度跳。
        """
        stub = self.make_stub(tracker=SnapshotOnlyTracker(valid_snapshot()))
        self._call(stub)
        stub._publish_state.assert_not_called()
        stub.status_server.publish.assert_not_called()

    def test_sends_nothing_when_vai_display_is_off(self):
        """``--no-vai-display``：连快照都不取（tracker 一次都不该被碰）。"""
        tracker = SnapshotOnlyTracker(valid_snapshot())
        stub = self.make_stub(vai_display=False, tracker=tracker)
        self._call(stub)
        self.assertEqual(tracker.snapshot_calls, 0)
        stub.chat_server.publish_vai.assert_not_called()

    def test_sends_nothing_without_a_tracker(self):
        """``--no-focus-silence`` ⇒ tracker 是 None ⇒ 展示通道整个不存在。

        这是**配置**而非数据问题，所以静默返回、不刷日志。
        """
        stub = self.make_stub(tracker=None)
        self._call(stub)                      # 不该抛
        stub.chat_server.publish_vai.assert_not_called()


class TestDisplayIsHookedIntoThePublishTick(unittest.TestCase):
    """钉住**调用点**：专注度展示必须挂在既有的 0.2 秒节拍上。

    与 test_state_fanout 里那类测试同一条理由 —— 「方法写对了，但没人调」
    是这类改动最典型的失败方式，只测方法本身发现不了。挂在既有节拍上
    而不是新开线程，则是因为新线程只会多一处要同步的状态，不会更快。
    """

    def test_the_tick_publishes_vai_every_beat(self):
        BackendB = load_backend_b_main().BackendB

        stub = mock.Mock()
        stub._stop = threading.Event()
        stub._effective_state.return_value = (config.STATE_NORMAL, "")
        # 第一拍就把停止位立起来，循环跑一轮即退出（wait(0.2) 之后返回）
        stub._tick_proactive.side_effect = lambda *a, **k: stub._stop.set()

        BackendB._state_publish_loop(stub)

        stub._publish_vai.assert_called_once()
        stub._publish_state.assert_called_once()


# ===========================================================================
# 开关与规格文本
# ===========================================================================

class TestCliWiring(unittest.TestCase):
    """``--vai-display`` 与 ``--focus-silence`` **必须各自独立**。

    合在一起（用 focus_silence 兼职控制展示）的话，用户要么"为了看数字被迫
    开着静默"，要么"为了不被静默被迫不看数字"—— 两件事本来没关系。
    """

    def parse(self, argv):
        return load_backend_b_main().build_parser().parse_args(argv)

    def test_defaults_come_from_config(self):
        args = self.parse([])
        self.assertEqual(args.vai_display, config.VAI_DISPLAY_ENABLED)
        self.assertEqual(args.focus_silence, config.FOCUS_SILENT_ENABLED)

    def test_can_be_turned_off_and_back_on(self):
        self.assertFalse(self.parse(["--no-vai-display"]).vai_display)
        self.assertTrue(self.parse(["--vai-display"]).vai_display)

    def test_the_two_switches_are_decoupled(self):
        """四种组合都要能表达出来。"""
        # 单独关掉一个，另一个必须**停在配置默认值上**（不许被连坐）
        self.assertTrue(self.parse(["--no-focus-silence"]).vai_display)
        self.assertEqual(self.parse(["--no-vai-display"]).focus_silence,
                         config.FOCUS_SILENT_ENABLED)
        args = self.parse(["--focus-silence", "--vai-display"])
        self.assertTrue(args.focus_silence and args.vai_display)
        args = self.parse(["--no-focus-silence", "--no-vai-display"])
        self.assertFalse(args.focus_silence or args.vai_display)

    def test_focus_silence_help_no_longer_claims_it_never_reaches_c(self):
        """规格改了，帮助文本就得跟着改。

        原话是「**B 内部使用，不下发 C**」。留着它，下一个读 --help 的人
        会以为 C 上那个数字是幻觉。
        """
        help_text = load_backend_b_main().build_parser().format_help()
        block = self.option_help(help_text, "--focus-silence,")
        self.assertNotIn("不下发 C", block)
        self.assertIn("只用于静默判定", block)

    @staticmethod
    def option_help(help_text: str, header_prefix: str) -> str:
        """截出某个选项那一小段帮助（从它的选项行到下一条选项行）。

        不能简单 ``split`` 旗标名：``--help`` 里 usage 行、选项行、
        以及**别的选项的帮助正文**（``--vai-display`` 那条就写着
        「与 --focus-silence 解耦」）都会出现同一个名字。
        """
        lines = help_text.splitlines()
        start = next(i for i, line in enumerate(lines)
                     if line.lstrip().startswith(header_prefix))
        block = [lines[start]]
        for line in lines[start + 1:]:
            if line.startswith("  --"):     # 下一条选项开始
                break
            block.append(line)
        return "\n".join(block)


class TestBannerShowsTheRealState(unittest.TestCase):
    """开机那一屏要能分辨"没开"（配置）与"开了但 A 没发数据"（数据）。

    分辨不了的话，C 的右栏在两种情况下长得**一模一样**（都是"未提供"），
    而这一屏是唯一能一眼看出区别的地方 —— 和「专注静默」那行的理由相同。
    """

    def banner(self, *, vai_display, tracker):
        BackendB = load_backend_b_main().BackendB
        stub = mock.Mock()
        stub.args.vai_display = vai_display
        stub.args.offline = "data/sample_vision.csv"
        stub.args.no_voice = True
        stub.args.proactive = False
        stub.args.half_duplex = True
        stub.args.focus_silence = tracker is not None
        stub.focus_tracker = tracker
        stub.speaker = None
        stub.voice_loop = None
        stub.chatter.available = False
        stub.chatter.reason = "未配置"
        stub.proactive.policy.quiet_enabled = False
        stub.proactive.focus_suppress_max = 0.0
        stub.history = None

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            BackendB._print_banner(stub)
        return buffer.getvalue()

    def test_prints_the_switch_state(self):
        line = [l for l in self.banner(vai_display=True, tracker=object()).splitlines()
                if "专注度展示" in l]
        self.assertEqual(len(line), 1)
        self.assertIn("已开启", line[0])
        self.assertIn(f"{config.VAI_HEARTBEAT_SECONDS:g}s", line[0])

    def test_says_so_when_switched_off(self):
        line = [l for l in self.banner(vai_display=False, tracker=None).splitlines()
                if "专注度展示" in l]
        self.assertIn("--no-vai-display", line[0])

    def test_distinguishes_config_off_from_no_tracker(self):
        """开着展示、但没有 tracker —— 必须说出**为什么**，不能只写"未启用"。"""
        line = [l for l in self.banner(vai_display=True, tracker=None).splitlines()
                if "专注度展示" in l]
        self.assertIn("未启用", line[0])
        self.assertIn("--no-focus-silence", line[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
