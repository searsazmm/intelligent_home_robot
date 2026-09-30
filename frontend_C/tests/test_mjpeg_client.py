# -*- coding: utf-8 -*-
"""画面通道：multipart 分帧、单槽语义、重连、地址解析。

跑法（在 frontend_C 目录下）：
    python -m pytest tests/ -v
    python tests/test_mjpeg_client.py

这里起的是**真的** TCP 连接（临时端口上的假服务端），只对 socket 打桩。
理由与 ``tests/test_state_client.py`` 一致：这段代码要防的恰恰是半包、
分帧错位、重连时序这些真实网络行为，把 socket 换成假对象之后，
测试和实现就变成了自说自话。

**不需要真的 JPEG。** 分帧只看 ``FF D8 FF`` 这个头，所以假帧就是
``SOI + 一串字节``。真要解码的那一段在 ``tests/test_split_window.py``
（那里才有 Qt）。
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_core.mjpeg_client import (
    BOUNDARY_FALLBACK,
    DEFAULT_CAMERA_FPS,
    FrameSlot,
    MjpegClient,
    MultipartParser,
    StreamError,
    clamp_fps,
    parse_stream_url,
)

#: 所有测试的超时上限。够慢机器用，又不会让挂掉的测试拖住整条流水线。
TIMEOUT = 5.0

SOI = b"\xff\xd8\xff"
EOI = b"\xff\xd9"


def fake_frame(payload: bytes = b"payload", size: int | None = None) -> bytes:
    """一个"长得像 JPEG"的帧：以 SOI 开头、以 EOI 结尾。"""
    if size is not None:
        body = bytes([0x41 + (i % 26) for i in range(max(0, size - 5))])
        return SOI + body + EOI
    return SOI + payload + EOI


def part(frame: bytes, boundary: str = BOUNDARY_FALLBACK,
         with_length: bool = True) -> bytes:
    """一个完整的分区：``--frame`` + 头 + 帧 + CRLF。"""
    head = f"--{boundary}\r\nContent-Type: image/jpeg\r\n".encode("ascii")
    if with_length:
        head += f"Content-Length: {len(frame)}\r\n".encode("ascii")
    return head + b"\r\n" + frame + b"\r\n"


def wait_for(predicate, timeout: float = TIMEOUT, interval: float = 0.01) -> bool:
    """轮询等一个条件成立。返回是否等到了。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ==========================================================================
# 地址解析
# ==========================================================================

class TestParseStreamUrl(unittest.TestCase):

    def test_a_full_url(self):
        self.assertEqual(parse_stream_url("http://127.0.0.1:8010/stream.mjpeg"),
                         ("127.0.0.1", 8010, "/stream.mjpeg"))

    def test_host_and_port_without_scheme(self):
        """``--stream-url`` 是答辩前手敲的，少写个 http:// 不值得让人去查文档。"""
        self.assertEqual(parse_stream_url("127.0.0.1:8010"), ("127.0.0.1", 8010, "/stream.mjpeg"))

    def test_a_bare_host_gets_the_default_port_and_path(self):
        self.assertEqual(parse_stream_url("127.0.0.1"),
                         ("127.0.0.1", 8010, "/stream.mjpeg"))

    def test_a_hostname_is_kept_as_is(self):
        """不解析成 IP —— ``localhost`` 与 ``127.0.0.1`` 在某些机器上不是一回事。"""
        self.assertEqual(parse_stream_url("localhost:9000/cam")[0], "localhost")

    def test_https_is_rejected(self):
        with self.assertRaises(StreamError):
            parse_stream_url("https://127.0.0.1:8010/stream.mjpeg")

    def test_a_non_numeric_port_is_rejected(self):
        with self.assertRaises(StreamError):
            parse_stream_url("127.0.0.1:abc/stream.mjpeg")

    def test_an_empty_url_is_rejected(self):
        for empty in ("", "   ", None):
            with self.subTest(empty=empty):
                with self.assertRaises(StreamError):
                    parse_stream_url(empty)


class TestClampFps(unittest.TestCase):

    def test_it_clamps_instead_of_raising(self):
        """手敲的命令行不该能把定时器打成 0ms（一个 0 会让界面空转烧 CPU）。"""
        self.assertEqual(clamp_fps(0), 1)
        self.assertEqual(clamp_fps(-5), 1)
        self.assertEqual(clamp_fps(9999), 60)
        self.assertEqual(clamp_fps(15), 15)

    def test_garbage_falls_back_to_the_default(self):
        self.assertEqual(clamp_fps("abc"), DEFAULT_CAMERA_FPS)
        self.assertEqual(clamp_fps(None), DEFAULT_CAMERA_FPS)


# ==========================================================================
# multipart 状态机
# ==========================================================================

class TestMultipartParser(unittest.TestCase):
    """分帧。**只有这一个类知道分区的形状**，所以要把它的边界情况钉死。"""

    def test_one_part_in_one_chunk(self):
        parser = MultipartParser()
        frame = fake_frame()
        self.assertEqual(parser.feed(part(frame)), [frame])
        self.assertEqual(parser.overflows, 0)

    def test_several_parts_in_one_chunk(self):
        parser = MultipartParser()
        frames = [fake_frame(bytes([i]) * 8) for i in range(3)]
        self.assertEqual(parser.feed(b"".join(part(f) for f in frames)), frames)

    def test_a_part_split_across_chunks_byte_by_byte(self):
        """**这条是这个类存在的理由。**

        真机上偶发（一次 recv 正好切在分区中间），手测几乎撞不到 ——
        而它坏了的表现是"画面偶尔卡一下"，最难查的一类。
        """
        parser = MultipartParser()
        frame = fake_frame(b"x" * 300)
        blob = part(frame)
        got = []
        for i in range(len(blob)):
            got += parser.feed(blob[i:i + 1])
        self.assertEqual(got, [frame])
        self.assertEqual(parser.overflows, 0)

    def test_the_boundary_itself_split_across_chunks(self):
        """``--frame`` 这个词本身被切成两半。"""
        parser = MultipartParser()
        frame = fake_frame(b"y" * 40)
        blob = part(frame)
        cut = blob.index(b"--frame") + 4      # 切在 "--fr|ame" 中间
        got = parser.feed(blob[:cut]) + parser.feed(blob[cut:])
        self.assertEqual(got, [frame])

    def test_a_preamble_before_the_first_boundary_is_skipped(self):
        """有些服务端会先发一段 CRLF 才开始分区。"""
        parser = MultipartParser()
        frame = fake_frame()
        self.assertEqual(parser.feed(b"\r\n\r\n" + part(frame)), [frame])

    def test_the_parser_survives_garbage_before_a_boundary(self):
        parser = MultipartParser()
        frame = fake_frame()
        self.assertEqual(parser.feed(b"\x00\x01\x02 garbage " + part(frame)), [frame])

    def test_a_frame_without_content_length_falls_back_to_the_eoi_marker(self):
        """本项目的 A 侧总是给 Content-Length，这条是给别家的流留的活路。"""
        parser = MultipartParser()
        frame = fake_frame(b"z" * 50)
        self.assertEqual(parser.feed(part(frame, with_length=False)), [frame])

    def test_a_lying_content_length_is_caught_by_the_jpeg_header(self):
        """声明 5 字节、实际给的字节不是 JPEG 开头 → 这一帧丢掉、重新找边界。

        宁可丢一帧也不要解码一段错位的数据：错位数据的表现是"画面花一下"，
        比黑屏难查得多。
        """
        parser = MultipartParser()
        bad = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 5\r\n\r\n"
               b"HELLO\r\n")
        good = part(fake_frame())
        frames = parser.feed(bad + good)
        self.assertEqual(frames, [fake_frame()])
        self.assertEqual(parser.overflows, 1)

    def test_a_custom_boundary(self):
        parser = MultipartParser("mjpeg-boundary")
        frame = fake_frame()
        self.assertEqual(parser.feed(part(frame, boundary="mjpeg-boundary")), [frame])

    def test_an_oversized_buffer_is_dropped_instead_of_growing(self):
        """缓冲超限必须丢掉重来 —— 把"画面卡住"升级成"进程被 OOM 杀掉"更糟。"""
        parser = MultipartParser(max_buffer=2048)
        parser.feed(b"nonsense" * 1000)          # 没有任何边界
        self.assertLessEqual(parser.buffered, 4096)
        self.assertGreater(parser.overflows, 0)
        # 丢掉之后还能正常接上
        self.assertEqual(parser.feed(part(fake_frame())), [fake_frame()])

    def test_nothing_is_produced_from_an_incomplete_part(self):
        parser = MultipartParser()
        blob = part(fake_frame(b"q" * 100))
        self.assertEqual(parser.feed(blob[:-10]), [])
        self.assertEqual(parser.feed(blob[-10:]), [fake_frame(b"q" * 100)])


# ==========================================================================
# 单槽
# ==========================================================================

class TestFrameSlot(unittest.TestCase):

    def test_take_returns_none_when_there_is_nothing_new(self):
        """**这条是 GUI 那一侧不白烧 CPU 的全部依据**：没新帧就不解码。"""
        slot = FrameSlot()
        self.assertIsNone(slot.take())
        slot.put(b"jpeg")
        self.assertIsNotNone(slot.take())
        self.assertIsNone(slot.take(), "同一帧被取了两次")

    def test_only_the_newest_frame_survives(self):
        slot = FrameSlot()
        slot.put(b"old")
        slot.put(b"new")
        seq, jpeg = slot.take()
        self.assertEqual(jpeg, b"new")
        self.assertEqual(seq, 2)
        self.assertEqual(slot.stats["overwritten"], 1)

    def test_the_sequence_keeps_increasing_across_takes(self):
        """序号用完即弃但**不回绕**：界面靠它判断"这一帧画过没有"。"""
        slot = FrameSlot()
        slot.put(b"a")
        slot.take()
        slot.put(b"b")
        self.assertEqual(slot.take()[0], 2)

    def test_an_empty_frame_is_ignored(self):
        slot = FrameSlot()
        slot.put(b"")
        self.assertIsNone(slot.take())
        self.assertEqual(slot.stats["put"], 0)

    def test_concurrent_put_and_take_never_loses_the_last_frame(self):
        """socket 线程写、GUI 线程读，两个线程同时跑不许出现死锁或丢帧。"""
        slot = FrameSlot()
        stop = threading.Event()
        produced = {"n": 0}

        def producer():
            while not stop.is_set():
                produced["n"] += 1
                slot.put(b"jpeg-%d" % produced["n"])

        def consumer():
            while not stop.is_set():
                slot.take()

        threads = [threading.Thread(target=producer, daemon=True),
                   threading.Thread(target=consumer, daemon=True)]
        for t in threads:
            t.start()
        time.sleep(0.3)
        stop.set()
        for t in threads:
            t.join(TIMEOUT)

        slot.put(b"last")
        self.assertEqual(slot.take()[1], b"last")
        self.assertEqual(slot.stats["put"], produced["n"] + 1)


# ==========================================================================
# 假的 MJPEG 服务端
# ==========================================================================

class FakeStreamServer:
    """最小的假 A：listen、接受连接、按脚本发字节。

    刻意不做任何"聪明"的事（不发心跳、不校验请求头）。真实服务的花样越多，
    测试失败时就越难判断是实现错了还是假服务端错了。
    """

    def __init__(self, frames=None, raw=None, port: int = 0,
                 boundary: str = BOUNDARY_FALLBACK) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", port))
        self._server.listen(5)
        self._server.settimeout(0.2)
        self.port = self._server.getsockname()[1]
        self.boundary = boundary
        # 「一帧都不发」也是一个有用的场景（连上了、但流是静默的），
        # 所以空列表要**照原样**保留，不能用 `or` 兜底成默认值。
        self.frames = [fake_frame()] if frames is None else list(frames)
        #: 响应头之后要发的**原始字节**（用来伪造 503、非 MJPEG 之类的响应）。
        #: 给了它就完全不发分区。
        self.raw = raw
        self.requests: list[bytes] = []
        self._lock = threading.Lock()
        self._closed = False
        self._accept_count = 0
        self._accepted = threading.Event()
        #: 在这里就建好（而不是在接收线程里）—— ``close()`` 可能赶在
        #: 接收线程跑起来之前被调用，那时会 AttributeError。
        self._clients: list[socket.socket] = []
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/stream.mjpeg"

    @property
    def accept_count(self) -> int:
        with self._lock:
            return self._accept_count

    def wait_for_client(self, timeout: float = TIMEOUT) -> bool:
        return self._accepted.wait(timeout)

    def close(self) -> None:
        self._closed = True
        with self._lock:
            clients = list(self._clients)
        for sock in clients:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        try:
            self._server.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(TIMEOUT)
            with self._lock:
                self._clients.append(conn)
                self._accept_count += 1
            self._accepted.set()
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            request = conn.recv(4096)
            self.requests.append(request)
            if self.raw is not None:
                # 给了原始字节就**整条响应**都由它决定（含状态行与响应头）——
                # 先发一个正常的 200 再跟一段 503，客户端会把它当成"连上了
                # 但没画面"，那样就测不出"识别出错误状态并重连"了。
                conn.sendall(self.raw if isinstance(self.raw, bytes) else b"".join(self.raw))
                return
            conn.sendall(self._handshake_bytes())
            self._send_body(conn)
        except OSError:
            pass

    def _handshake_bytes(self) -> bytes:
        return (
            "HTTP/1.0 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; "
            f"boundary={self.boundary}\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")

    def _send_body(self, conn: socket.socket) -> None:
        for frame in self.frames:
            conn.sendall(part(frame, boundary=self.boundary))
            time.sleep(0.02)


# ==========================================================================
# 客户端
# ==========================================================================

class TestMjpegClient(unittest.TestCase):
    """真连接、真重连。socket 不打桩。"""

    def _client(self, server, **kwargs):
        params = dict(reconnect_initial=0.05, reconnect_max=0.2)
        params.update(kwargs)
        client = MjpegClient(server.url, **params)
        self.addCleanup(client.stop)
        return client

    def test_it_pulls_frames_into_the_slot(self):
        frames = [fake_frame(bytes([i]) * 32) for i in range(4)]
        server = FakeStreamServer(frames=frames)
        self.addCleanup(server.close)
        slot = FrameSlot()
        client = self._client(server, slot=slot)
        client.start()

        self.assertTrue(wait_for(lambda: slot.stats["put"] >= 1), "一帧都没收到")
        self.assertTrue(client.connected)

    def test_the_request_is_a_plain_get(self):
        server = FakeStreamServer()
        self.addCleanup(server.close)
        client = self._client(server)
        client.start()
        self.assertTrue(server.wait_for_client())
        self.assertTrue(wait_for(lambda: server.requests))
        request = server.requests[0]
        self.assertTrue(request.startswith(b"GET /stream.mjpeg HTTP/1.0\r\n"))
        self.assertIn(b"Host: 127.0.0.1:", request)
        self.assertIn(b"Accept: multipart/x-mixed-replace", request)

    def test_on_frame_is_used_when_no_slot_is_given(self):
        """不给单槽时用 ``on_frame``。**但 GUI 用它只能塞单槽**（模块文档）。"""
        server = FakeStreamServer(frames=[fake_frame(b"a"), fake_frame(b"b")])
        self.addCleanup(server.close)
        seen: list[bytes] = []
        done = threading.Event()

        def on_frame(jpeg: bytes) -> None:
            seen.append(jpeg)
            if len(seen) >= 2:
                done.set()

        client = self._client(server, on_frame=on_frame)
        client.start()
        self.assertTrue(done.wait(TIMEOUT), f"只收到 {len(seen)} 帧")
        self.assertIn(fake_frame(b"a"), seen)

    def test_the_link_callback_reports_both_edges(self):
        server = FakeStreamServer(frames=[fake_frame()])
        self.addCleanup(server.close)
        events: list[tuple[str, bool]] = []
        client = self._client(server, on_link=lambda s, c: events.append((s, c)))
        client.start()
        self.assertTrue(wait_for(lambda: any(c for _s, c in events)), "没有报告连上")
        self.assertEqual(events[0][0], "stream")

    def test_a_503_does_not_kill_the_client(self):
        """A 的展示流满了会回 503 —— 那不是"协议错了"，是"再等等"。

        这两件事必须分开：协议错了要报错退出，满了要退避重连。
        混在一起的话，四个客户端连着的时候第五个会安静地死掉。
        """
        raw = (b"HTTP/1.0 503 Service Unavailable\r\n"
               b"Content-Type: text/plain; charset=utf-8\r\n"
               b"Content-Length: 6\r\n\r\n" + "满了".encode("utf-8"))
        server = FakeStreamServer(raw=raw)
        self.addCleanup(server.close)
        client = self._client(server)
        client.start()
        self.assertTrue(server.wait_for_client())
        self.assertTrue(wait_for(lambda: server.accept_count >= 2), "没有重连")
        self.assertFalse(client.connected)

    def test_a_non_mjpeg_response_is_rejected(self):
        raw = (b"HTTP/1.0 200 OK\r\n"
               b"Content-Type: text/html\r\n"
               b"Content-Length: 2\r\n\r\nhi")
        server = FakeStreamServer(raw=raw)
        self.addCleanup(server.close)
        client = self._client(server)
        client.start()
        self.assertTrue(wait_for(lambda: server.accept_count >= 2), "没有重连")
        self.assertFalse(client.connected)

    def test_a_garbage_status_line_is_rejected(self):
        raw = b"NOT-HTTP AT ALL\r\n\r\n"
        server = FakeStreamServer(raw=raw)
        self.addCleanup(server.close)
        client = self._client(server)
        client.start()
        self.assertTrue(wait_for(lambda: server.accept_count >= 2), "没有重连")

    def test_it_reconnects_after_the_server_hangs_up(self):
        """A 挂了 → 退避重连 → 又起来了就接上。**这是右栏能自愈的全部依据。**"""
        server = FakeStreamServer(frames=[fake_frame()])
        self.addCleanup(server.close)
        slot = FrameSlot()
        client = self._client(server, slot=slot)
        client.start()
        self.assertTrue(wait_for(lambda: slot.stats["put"] >= 1))
        first = server.accept_count

        # 服务端把所有连接掐掉，但不关监听：模拟"A 重启"
        with server._lock:
            clients = list(server._clients)
        for sock in clients:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.assertTrue(wait_for(lambda: server.accept_count > first), "没有重连")
        self.assertTrue(wait_for(lambda: client.connected), "重连后没报告连上")

    def test_a_vanished_server_is_retried_not_fatal(self):
        """端口上什么都没有：客户端要一直退避重试，而不是线程退出。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()          # 现在这个端口没人监听

        client = MjpegClient(f"http://127.0.0.1:{port}/stream.mjpeg",
                             reconnect_initial=0.05, reconnect_max=0.1)
        self.addCleanup(client.stop)
        client.start()
        time.sleep(0.3)
        self.assertFalse(client.connected)
        # 线程还活着（没退出），所以起来之后还能接上 —— 用一个真服务端验证
        server = FakeStreamServer(port=port)
        self.addCleanup(server.close)
        self.assertTrue(server.wait_for_client(), "客户端已经放弃重试了")

    def test_stop_returns_promptly_while_blocked_on_recv(self):
        """``stop()`` 必须立刻醒 —— 否则退出要等满超时。

        这一条钉的是"先 shutdown 再 close"那半句：只 close 的话，
        阻塞在 ``recv`` 上的线程在 Windows 上不会返回。
        """
        # frames=[]：握手正常完成，之后一帧都不发（连接保持打开）。
        # socket_timeout 给到 10 秒，好让"stop() 是不是真的把它叫醒了"
        # 这件事有一个宽裕的判据 —— 靠超时自然醒的话这条会明显超时。
        server = FakeStreamServer(frames=[])
        self.addCleanup(server.close)
        client = self._client(server, socket_timeout=10.0)
        client.start()
        self.assertTrue(server.wait_for_client())
        self.assertTrue(wait_for(lambda: client.connected))

        started = time.time()
        client.stop()
        self.assertLess(time.time() - started, 2.0, "stop() 被 recv 拖住了")

    def test_stop_is_idempotent(self):
        server = FakeStreamServer()
        self.addCleanup(server.close)
        client = self._client(server)
        client.start()
        client.stop()
        client.stop()          # 不许抛
        self.assertFalse(client.connected)

    def test_a_callback_that_raises_does_not_break_the_stream(self):
        """回调出事只该丢那一帧。**画面通道不许因为界面代码的毛病而断掉。**"""
        server = FakeStreamServer(frames=[fake_frame(bytes([i])) for i in range(6)])
        self.addCleanup(server.close)
        calls = {"n": 0}

        def angry(_jpeg):
            calls["n"] += 1
            raise RuntimeError("界面代码炸了")

        client = self._client(server, on_frame=angry)
        client.start()
        self.assertTrue(wait_for(lambda: calls["n"] >= 3), f"只被调了 {calls['n']} 次")
        self.assertEqual(client.stats["callback_errors"], calls["n"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
