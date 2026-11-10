# -*- coding: utf-8 -*-
"""``main.BackendB`` 与家属数据落盘（:mod:`core.vitals_log`）的接线。

落盘 writer 本身的行为在 test_vitals_log.py 里已经钉死了；这里只回答
另一个问题：**B 的运行路径上有没有真的调它**。家属 Web 端的一周折线图、
状态时间线、预警中心，全靠这几处挂接喂数据 —— 少挂一处，家属端那一块
就永远是空的，而且没有任何报错（数据没了，不是数据错了）。

沿袭 test_state_fanout.py 的做法：用 duck-typed 替身直接调未绑定方法，
不构造整个 BackendB —— 那要拉起视觉源、历史文件、语音层，不值。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import unittest
from unittest import mock

_BACKEND_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_B)

_MAIN_MODULE_NAME = "_backend_b_main_for_family_hooks"


def load_backend_b_main():
    """按**文件路径**加载 backend_B/main.py（理由见 test_state_fanout.py：
    仓库里有两个 main.py，``import main`` 会撞上 backend_A 的）。"""
    path = os.path.join(_BACKEND_B, "main.py")
    spec = importlib.util.spec_from_file_location(_MAIN_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MAIN_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class TestLogFamilyData(unittest.TestCase):
    """``_log_family_data``：节拍上把体征快照交给 writer，节流在 writer 里。"""

    def test_forwards_current_snapshot(self):
        snapshot = object()
        stub = mock.Mock()
        stub.family_log = mock.Mock()
        stub.vitals_store.snapshot.return_value = snapshot
        load_backend_b_main().BackendB._log_family_data(stub)
        stub.family_log.maybe_write_vitals.assert_called_once_with(snapshot)

    def test_skips_silently_without_writer(self):
        """B_FAMILY_LOG=0（或初始化失败）时 family_log 为 None —— 必须静默跳过。

        落盘是旁路功能，它缺席绝不能变成节拍上的一个 AttributeError。
        """
        stub = mock.Mock(spec=["family_log", "vitals_store"])
        stub.family_log = None
        stub.vitals_store.snapshot.return_value = object()
        load_backend_b_main().BackendB._log_family_data(stub)   # 不应抛异常
        stub.vitals_store.snapshot.assert_not_called()


class TestPublishStateWritesStateEvent(unittest.TestCase):
    """``_publish_state``：真实状态变化要落一条 state_event。"""

    def make_stub(self, published: bool):
        stub = mock.Mock()
        stub.status_server.publish.return_value = published
        stub.chat_server.client_count.return_value = 0
        stub.family_log = mock.Mock()
        return stub

    def test_write_on_real_change(self):
        stub = self.make_stub(published=True)
        load_backend_b_main().BackendB._publish_state(stub, "sad", "对话判定")
        stub.family_log.write_state_event.assert_called_once_with("sad", "对话判定")

    def test_not_written_when_deduped(self):
        """8001 判为「没变化且没到心跳」时连镜像都不做，家属端同样不写。"""
        stub = self.make_stub(published=False)
        load_backend_b_main().BackendB._publish_state(stub, "sad", "对话判定")
        stub.family_log.write_state_event.assert_not_called()

    def test_skips_silently_without_writer(self):
        stub = self.make_stub(published=True)
        stub.family_log = None
        load_backend_b_main().BackendB._publish_state(stub, "sad", "对话判定")   # 不抛


class TestStatePublishLoopBeat(unittest.TestCase):
    """``_state_publish_loop``：家属落盘必须挂在 0.2 秒节拍上。

    少调一拍不会报错，只会让家属折线图悄悄缺点 —— 这是接线测试存在的意义。
    """

    def test_beat_includes_family_log(self):
        stub = mock.Mock()
        stub._stop = mock.Mock()
        stub._stop.is_set.side_effect = [False, True]   # 跑一拍就退出
        stub._stop.wait.return_value = True
        stub._effective_state.return_value = ("normal", "")
        load_backend_b_main().BackendB._state_publish_loop(stub)
        stub._log_family_data.assert_called_once_with()


class TestRaiseEmergencyWritesAlert(unittest.TestCase):
    """``_raise_emergency``：预警进家属库（60s 防抖在 writer 内部）。"""

    def make_stub(self):
        stub = mock.Mock()
        stub.family_alerter = mock.Mock()
        stub.chat_server = mock.Mock()
        stub.family_log = mock.Mock()
        return stub

    def test_alert_lands_with_kind_and_text(self):
        stub = self.make_stub()
        load_backend_b_main().BackendB._raise_emergency(stub, "emergency", "救命")
        stub.family_log.write_alert.assert_called_once_with("emergency", "救命")

    def test_alert_still_written_when_webhook_fails(self):
        """webhook 没配/网络炸了不能连累家属库这条记录 —— 两条通道互不影响。"""
        stub = self.make_stub()
        stub.family_alerter.alert.side_effect = OSError("network down")
        load_backend_b_main().BackendB._raise_emergency(stub, "emergency", "救命")
        stub.family_log.write_alert.assert_called_once()

    def test_skips_silently_without_writer(self):
        stub = self.make_stub()
        stub.family_log = None
        load_backend_b_main().BackendB._raise_emergency(stub, "emergency", "救命")
        stub.chat_server.broadcast_emergency.assert_called_once()   # 其余通道照常


class TestShutdownClosesWriter(unittest.TestCase):
    """``shutdown``：落盘要最后关，把队列里积压的状态/告警写完再退出。"""

    STUB_SPEC = ["_stop", "_threads", "family_log", "status_server",
                 "chat_server", "vision_source", "voice_loop", "speaker"]

    def test_writer_closed_and_cleared(self):
        stub = mock.Mock(spec=self.STUB_SPEC)
        stub._stop = threading.Event()
        stub._threads = []
        stub.voice_loop = None
        stub.speaker = None
        stub.vision_source = mock.Mock()
        writer = stub.family_log = mock.Mock()
        load_backend_b_main().BackendB.shutdown(stub)
        # shutdown 会把 family_log 置 None（防止重复关闭），所以要断言
        # 之前保存的引用，而不是 stub.family_log
        writer.close.assert_called_once_with()
        self.assertIsNone(stub.family_log)

    def test_close_skipped_when_writer_absent(self):
        stub = mock.Mock(spec=self.STUB_SPEC)
        stub._stop = threading.Event()
        stub._threads = []
        stub.voice_loop = None
        stub.speaker = None
        stub.vision_source = mock.Mock()
        stub.family_log = None
        load_backend_b_main().BackendB.shutdown(stub)   # 不应抛异常


if __name__ == "__main__":
    unittest.main()
