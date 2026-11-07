# -*- coding: utf-8 -*-
"""多实例自检的测试：find_live_listeners 要能抓出"占着端口的活实例"。

背景（2026-10-09 现场）：三次重启都没杀干净旧进程，Windows 的
SO_REUSEADDR 让三个 B 端静默绑死同一对端口，用户问一句有好几个
声音一起回答。自检必须在 bind 之前用 connect 探测把这种情况拦下。

沿用仓库既有测试的写法（unittest，自己把父目录塞进 sys.path）。
"""

import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import find_live_listeners                              # noqa: E402


class _FakeSock:
    """记录 close() 是否被调用的假 socket。"""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class TestFindLiveListeners(unittest.TestCase):
    """用可注入的 connect 替身测逻辑，不碰真实网络。"""

    def test_all_refused_means_nothing_busy(self):
        """全部连不上 → 端口都是空的，可以放心 bind。"""
        def refuse(address, timeout):
            raise ConnectionRefusedError("no listener")

        self.assertEqual(find_live_listeners(
            [("127.0.0.1", 8001), ("127.0.0.1", 8002)], _connect=refuse), [])

    def test_live_listener_is_reported(self):
        """连得上的端口要被点名，且探测用的 socket 必须被关闭。"""
        opened = []

        def accept_some(address, timeout):
            if address[1] == 8002:
                sock = _FakeSock()
                opened.append(sock)
                return sock
            raise ConnectionRefusedError("free")

        busy = find_live_listeners(
            [("127.0.0.1", 8001), ("127.0.0.1", 8002)], _connect=accept_some)
        self.assertEqual(busy, [("127.0.0.1", 8002)])
        self.assertTrue(opened[0].closed, "探测连接必须立刻关闭，不能挂着")

    def test_timeout_is_forwarded(self):
        """超时要传给底层 connect —— localhost 连接失败应当立即返回，
        但万一碰上防火墙吞包，也不能把启动卡死在默认的系统超时上。"""
        seen = {}

        def record(address, timeout):
            seen["timeout"] = timeout
            raise ConnectionRefusedError

        find_live_listeners([("127.0.0.1", 8001)],
                            timeout=0.25, _connect=record)
        self.assertEqual(seen["timeout"], 0.25)


class TestAgainstRealSockets(unittest.TestCase):
    """和真实 socket 对一遍：替身测的是我们的调用方式，
    这一组测的是 connect 探测对真实监听/真实空端口的行为。"""

    def test_real_listener_is_detected(self):
        with _listening_socket() as server:
            host, port = server.getsockname()
            self.assertEqual(
                find_live_listeners([(host, port)]), [(host, port)])

    def test_closed_port_is_not_busy(self):
        port = _grab_free_port()
        # socket 已关闭：connect 会被立刻拒绝（TIME_WAIT 不接受 connect）
        self.assertEqual(find_live_listeners([("127.0.0.1", port)]), [])


def _listening_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server


def _grab_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    unittest.main()
