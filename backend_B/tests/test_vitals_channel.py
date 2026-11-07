# -*- coding: utf-8 -*-
"""心率 / 呼吸（rppg）**展示**通道（B→C，``vitals`` 报文）。

A 端的 rppg 报文一直都在（约 1Hz），此前 B 收下、计数、然后**丢弃** ——
没有消费者。现在它经 :mod:`core.vitals` 接住并投影成 ``vitals`` 展示报文，
与 ``vai`` 走同一条 8002 线路、同一套"变化 + 心跳"节流。

写得比一条"能发出去"的测试多的理由，与 ``test_display_channel.py`` 相同：
这条通道的危险不在于发不出去，而在于**发的时候顺手多做了事** ——
心率是最像"该驱动点什么"的数字，一旦有人把它接到表情或判决上
（"心率高了就切 sad"），体征就成了第二个状态源。

所以下面钉住：**报文内容**（逐字段原样转发、无 state、null 不是 0）、
**存储语义**（只留最新一条）、**发送节流**（变化 / 心跳 / 补发）、
**只读旁路**（不碰 evaluator、不推状态）。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

_BACKEND_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_B)

import config
import core.vitals as vitals_lib
from core.typed_messages import RppgReading
from core.ui_channel import ChatServer
from core.vitals import VITALS_NOTE, VitalsStore, build_vitals_message


def make_server(**kwargs) -> ChatServer:
    """造一个没 start() 过的 ChatServer（不监听端口、不碰网络）。"""
    kwargs.setdefault("on_chat", lambda text: None)
    return ChatServer(**kwargs)


def sent_messages(client) -> list:
    """把发到假 socket 上的字节解回字典列表。"""
    return [json.loads(call[0][0].decode("utf-8"))
            for call in client.sendall.call_args_list]


def valid_reading(**overrides) -> RppgReading:
    """一条"真的测出来了"的读数，字段可覆盖。"""
    fields = dict(timestamp=1.0, hr=72, rr=16.0, ibi_ms=(833, 841, 828))
    fields.update(overrides)
    return RppgReading(**fields)


# ===========================================================================
# 报文内容
# ===========================================================================

class TestBuildVitalsMessage(unittest.TestCase):
    """``build_vitals_message``：只做投影，一个字段都不许加工。"""

    def test_carries_every_field_the_frontend_may_need(self):
        msg = build_vitals_message(valid_reading())
        self.assertEqual(msg["type"], "vitals")
        self.assertEqual(msg["hr"], 72)
        self.assertEqual(msg["rr"], 16.0)
        self.assertEqual(msg["ibi_ms"], [833, 841, 828])
        self.assertIn("timestamp", msg)

    def test_has_no_state_field(self):
        """**这条是防第二个状态源**（与 vai 报文同一条红线）。

        ``vitals`` 一旦带上 ``state``，心率迟早被接到表情上 ——
        体征只该是右栏的另一行数字，判决走 8001/8002 的 state。
        """
        msg = build_vitals_message(valid_reading())
        self.assertNotIn("state", msg)
        self.assertNotIn("reason_state", msg)

    def test_none_values_stay_none_and_are_not_zero(self):
        """``hr is None`` = **没有测量值**（窗口未满 / SNR 不足），不是心率 0。

        显示成 0 会让老人以为系统在报一个危险的心跳数 ——
        前端要分开显示（「——」而不是「0 bpm」）。
        """
        msg = build_vitals_message(valid_reading(hr=None, rr=None, ibi_ms=()))
        self.assertIsNone(msg["hr"])
        self.assertIsNone(msg["rr"])

    def test_none_reading_returns_none(self):
        """一条都没收到过 rppg 时上游给 None —— 别发空报文冲掉"未检出"语义。"""
        self.assertIsNone(build_vitals_message(None))

    def test_ibi_ms_becomes_a_json_list(self):
        """frozen dataclass 里是元组，json 序列化不认 —— 投影成 list。"""
        msg = build_vitals_message(valid_reading())
        self.assertIsInstance(msg["ibi_ms"], list)

    def test_note_is_the_disclaimer_verbatim(self):
        """这是摄像头估出来的数，不是医疗器械 —— 免责声明必须跟着数值走。"""
        msg = build_vitals_message(valid_reading())
        self.assertEqual(msg["note"], VITALS_NOTE)
        self.assertIn("非医疗结论", msg["note"])

    def test_works_on_a_duck_typed_reading(self):
        """鸭子类型：传输层不该认识 ``RppgReading`` 这个领域对象。"""
        duck = mock.Mock(hr=80, rr=17.5, ibi_ms=(750,))
        msg = build_vitals_message(duck)
        self.assertEqual(msg["type"], "vitals")
        self.assertEqual(msg["hr"], 80)
        self.assertEqual(msg["rr"], 17.5)


# ===========================================================================
# 存储语义
# ===========================================================================

class TestVitalsStore(unittest.TestCase):
    """``VitalsStore``：只留最新一条 —— 界面要"当下"，不要历史。"""

    def test_snapshot_starts_empty(self):
        store = VitalsStore()
        self.assertIsNone(store.snapshot())

    def test_update_then_snapshot(self):
        store = VitalsStore()
        reading = valid_reading()
        store.update(reading)
        self.assertIs(store.snapshot(), reading)

    def test_latest_reading_wins(self):
        store = VitalsStore()
        store.update(valid_reading(hr=72))
        newest = valid_reading(hr=75)
        store.update(newest)
        self.assertIs(store.snapshot(), newest)


# ===========================================================================
# 发送节流：变化 / 心跳 / 补发
# ===========================================================================

class TestPublishVitalsThrottling(unittest.TestCase):
    """``publish_vitals``：0.2 秒一拍，但只在**变化**或**心跳到期**时才真发。"""

    def setUp(self):
        self.server = make_server()
        self.addCleanup(self.server.stop)

    def test_first_reading_is_sent(self):
        self.assertTrue(self.server.publish_vitals(valid_reading()))

    def test_unchanged_reading_is_not_resent(self):
        """rppg 约 1Hz，而节拍 0.2 秒 —— 每拍都发的话 C 每秒要吃 5 条重复的。"""
        self.server.publish_vitals(valid_reading())
        self.assertFalse(self.server.publish_vitals(valid_reading()))

    def test_a_new_hr_is_sent(self):
        self.server.publish_vitals(valid_reading())
        self.assertTrue(self.server.publish_vitals(valid_reading(hr=75)))

    def test_a_new_rr_alone_is_sent(self):
        """呼吸数也在签名里 —— 只比心率的话呼吸的变化会被吞掉。"""
        self.server.publish_vitals(valid_reading())
        self.assertTrue(self.server.publish_vitals(valid_reading(rr=17.5)))

    def test_ibi_jitter_alone_is_not_sent(self):
        """ibi 每一跳都在变（同一个人相邻两跳差十几毫秒是常态）。

        比它等于每条都发，去重就失效了 —— 签名只允许是 ``(hr, rr)``。
        """
        self.server.publish_vitals(valid_reading())
        self.assertFalse(self.server.publish_vitals(
            valid_reading(ibi_ms=(830, 845, 829))))

    def test_values_becoming_none_is_sent(self):
        """从"测出来了"到"没检出"是最需要让界面知道的一次变化。"""
        self.server.publish_vitals(valid_reading())
        self.assertTrue(self.server.publish_vitals(
            valid_reading(hr=None, rr=None, ibi_ms=())))

    def test_heartbeat_resends_after_the_interval(self):
        """没变化也要按心跳重发 —— 否则 C 端分不清「数值没变」与「B 挂了」。"""
        self.server.publish_vitals(valid_reading())
        self.server._last_vitals_at -= config.VAI_HEARTBEAT_SECONDS + 1.0
        self.assertTrue(self.server.publish_vitals(valid_reading()))

    def test_none_reading_is_not_sent(self):
        self.assertFalse(self.server.publish_vitals(None))

    def test_force_bypasses_dedup(self):
        """``force`` 是给补发留的口子。"""
        self.server.publish_vitals(valid_reading())
        self.assertTrue(self.server.publish_vitals(valid_reading(), force=True))


class TestSendInitialVitals(unittest.TestCase):
    """新客户端连上时补发：与 ``_send_initial_vai`` 同一条理由。"""

    def setUp(self):
        self.server = make_server()
        self.addCleanup(self.server.stop)
        self.client = mock.Mock()
        self.addr = ("127.0.0.1", 55555)

    def test_resends_current_reading_and_logs(self):
        self.server._vitals_provider = lambda: valid_reading()
        with self.assertLogs("core.ui_channel", level="INFO") as captured:
            self.server._send_initial_vitals(self.client, self.addr)

        msgs = sent_messages(self.client)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["type"], "vitals")
        self.assertEqual(msgs[0]["hr"], 72)
        self.assertIn("补发体征", "".join(captured.output))

    def test_does_nothing_without_a_provider(self):
        self.server._vitals_provider = None
        self.server._send_initial_vitals(self.client, self.addr)
        self.client.sendall.assert_not_called()

    def test_does_nothing_when_provider_returns_none(self):
        """一条 rppg 都还没收到过 —— 别补发，让 C 显示「未提供」。"""
        self.server._vitals_provider = lambda: None
        self.server._send_initial_vitals(self.client, self.addr)
        self.client.sendall.assert_not_called()

    def test_provider_exception_keeps_the_connection(self):
        """取快照炸了**不**掐连接 —— 体征只是一栏展示（与 vai 同一条理由）。"""
        def boom():
            raise RuntimeError("存储炸了")
        self.server._vitals_provider = boom
        self.server._clients.append(self.client)   # 假装已在册
        with self.assertLogs("core.ui_channel", level="ERROR"):
            self.server._send_initial_vitals(self.client, self.addr)
        self.client.sendall.assert_not_called()
        self.assertIn(self.client, self.server._clients)   # 连接还在


# ===========================================================================
# main.BackendB 的接线
# ===========================================================================

def load_backend_b_main():
    """按文件路径加载 backend_B/main.py（**不能叫 "main"**，理由同 test_display_channel）。"""
    cached = sys.modules.get("backend_b_main")
    if cached is not None:
        return cached
    import importlib.util
    path = os.path.join(_BACKEND_B, "main.py")
    spec = importlib.util.spec_from_file_location("backend_b_main", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backend_b_main"] = module
    spec.loader.exec_module(module)
    return module


class TestBackendBWiring(unittest.TestCase):
    """``main.BackendB``：rppg 进存储、节拍推展示。"""

    @classmethod
    def setUpClass(cls):
        cls.main = load_backend_b_main()

    def _make_stub(self):
        stub = mock.Mock()
        stub.args = mock.Mock(vai_display=True, proactive=True)
        stub.vitals_store = VitalsStore()
        stub.chat_server = mock.Mock()
        return stub

    def test_on_typed_reading_routes_rppg_into_the_store(self):
        """``MSG_RPPG`` 进 ``VitalsStore``；别的类型化报文（focus）不归它管。"""
        stub = self._make_stub()
        reading = valid_reading()

        self.main.BackendB._on_typed_reading(stub, "rppg", reading)
        self.assertIs(stub.vitals_store.snapshot(), reading)

        self.main.BackendB._on_typed_reading(stub, "focus", object())
        self.assertIs(stub.vitals_store.snapshot(), reading)

    def test_on_typed_reading_is_the_only_vitals_writer_here(self):
        """接线层对 vitals_store 的全部动作就是 ``update`` —— 不读、不算。"""
        stub = self._make_stub()
        stub.vitals_store = mock.Mock()
        self.main.BackendB._on_typed_reading(stub, "rppg", valid_reading())
        stub.vitals_store.update.assert_called_once()
        stub.vitals_store.snapshot.assert_not_called()

    def test_publish_vitals_forwards_the_snapshot(self):
        stub = self._make_stub()
        reading = valid_reading()
        stub.vitals_store = mock.Mock()
        stub.vitals_store.snapshot.return_value = reading

        self.main.BackendB._publish_vitals(stub)
        stub.chat_server.publish_vitals.assert_called_once_with(reading)

    def test_publish_vitals_forwards_none_when_nothing_has_arrived(self):
        """一条都没收到过时转发 ``None`` —— 真正的"不发"在 ChatServer 层
        （见 ``test_none_reading_is_not_sent``），接线层只管如实递快照。"""
        stub = self._make_stub()
        self.main.BackendB._publish_vitals(stub)
        stub.chat_server.publish_vitals.assert_called_once_with(None)

    def test_the_tick_publishes_vitals_every_beat(self):
        """0.2 秒节拍上必须挂着体征推送（去重在 ChatServer.publish_vitals 里）。"""
        stub = self._make_stub()
        stub._effective_state = mock.Mock(return_value=("normal", ""))
        stub._publish_state = mock.Mock()
        stub._publish_vai = mock.Mock()
        stub._publish_vitals = mock.Mock()
        stub._tick_proactive = mock.Mock()
        stub._stop = mock.Mock()
        # 真跑一小段循环：第 4 次 is_set 时退出（循环体跑满 3 拍）
        stub._stop.is_set.side_effect = [False, False, False, True, True]
        stub._stop.wait.side_effect = lambda _s: None

        import threading
        thread = threading.Thread(
            target=self.main.BackendB._state_publish_loop, args=(stub,), daemon=True)
        thread.start()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(stub._publish_vitals.call_count, 3)


if __name__ == "__main__":
    unittest.main()
