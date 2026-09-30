"""展示流（MJPEG）、像素同源合成源，以及**补上的那条隐私红线测试**。

标准库 unittest，不依赖 pytest、不依赖摄像头。HTTP 部分一律 ``port=0``
（让系统挑一个空闲端口），所以这些用例不会撞上任何正在跑的 A。

--------------------------------------------------------------------------
这个文件补的是两块从没被测过的地
--------------------------------------------------------------------------
1. **「像素不出模块」此前一条测试都没有。** ``privacy/guard.py`` 写了
   一整套 ``SENSITIVE_TOKENS`` / ``SAFE_KEYS`` / ``assert_clean``，
   ``backend_A/tests/`` 里却没有任何一个用例调过它们。展示流这一步恰好是
   这条红线唯一一次被开例外，所以红线**必须**先被钉住 —— 否则"例外没放宽
   红线"这句话只是一句声明。见 :class:`TestPrivacyRedLineStillHolds`。

2. **``scrub_frame``（"用完即毁"）全仓从无调用点。** 一条写着"用完即毁"
   却从未执行过的代码，与没有这条代码是一回事。它现在挂上了，
   对应用例见 :class:`TestScrub`。

运行::

    cd backend_A && python -m unittest discover -s tests -v
    cd backend_A && python tests/test_stream.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import socket
import struct
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

# 允许 `python tests/test_stream.py` 这种直接执行的方式。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _console  # noqa: E402,F401  （导入即生效：让中文输出不乱码）

import numpy as np  # noqa: E402

from module_a_vision import stream as stream_mod  # noqa: E402
from module_a_vision.capture.base import is_feature_source, is_frame_source  # noqa: E402
from module_a_vision.capture.base import CaptureError  # noqa: E402
from module_a_vision.capture.synthetic import SyntheticPixelSource  # noqa: E402
from module_a_vision.privacy.guard import (  # noqa: E402
    SENSITIVE_TOKENS,
    PrivacyViolationError,
    assert_clean,
    find_violations,
    scrub_frame,
)
from module_a_vision.server import build_server  # noqa: E402
from module_a_vision.stream import (  # noqa: E402
    BIND_HOST,
    FrameHub,
    MjpegStreamServer,
    draw_overlay,
    overlay_lines,
)
from shared.frame_features import FrameFeatures, FrameQuality  # noqa: E402


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


CV2 = _has_cv2()
#: 编/解码真的用得到 opencv 才跳过；纯逻辑用例不留在这里。
_needs_cv2 = unittest.skipUnless(CV2, "本机没装 opencv，展示流的像素用例跳过")


def frame(ts: float = 1.5, *, has_face: bool = True, bbox=(80, 60, 300, 260)):
    return FrameFeatures(
        ts=ts, wall_ts=ts, has_face=has_face, face_bbox=bbox if has_face else None,
        ear_left=0.29, ear_right=0.29, closure_ratio=0.2,
        pitch_deg=12.0, yaw_deg=-2.0, roll_deg=34.0,
        gaze_off_ratio=0.55 if has_face else None,
        quality=FrameQuality(),
    )


# ================================================================ 红线

class _QuietStdout(unittest.TestCase):
    """把被测代码的 stdout 吃掉。

    这个文件里好几类用例**故意**触发带有 ``[A] ✗`` / ``[A] ⚠`` 前言的
    路径（隐私闸、只读缓冲、展示流横幅）。让它们涌进测试输出，会让下一个
    人分不清"预期的报错"和"真的出事了" —— 一个把真实故障淹掉的测试套件
    是有害的。
    """

    def setUp(self):
        self._quiet = contextlib.redirect_stdout(io.StringIO())
        self._quiet.__enter__()
        self.addCleanup(lambda: self._quiet.__exit__(None, None, None))


class TestPrivacyRedLineStillHolds(_QuietStdout):
    """展示流是**唯一的窄例外**，不是"红线放宽了"。

    这里的每一条都在问同一个问题：那条总线（TCP 8000 的报文）**有没有
    因为多了展示流而变松**。答案必须一直是"没有"。
    """

    def test_the_sensitive_token_list_still_names_the_pixel_words(self):
        """例外是靠"走另一条路"实现的，**不是**靠从词表里删词。

        谁要是为了让某个报错消失把 ``jpeg`` 从词表里拿掉，这条会先炸。
        """
        for token in ("bbox", "landmarks", "image", "frame", "jpeg", "png",
                      "pixel", "snapshot", "video", "raw", "crop"):
            with self.subTest(token=token):
                self.assertIn(token, SENSITIVE_TOKENS)

    def test_broadcast_still_drops_a_message_carrying_a_bbox(self):
        """报文里塞了人脸框 → 被拦下、计数、发不出去。"""
        server = build_server(source_kind="synthetic", real_time=False)
        self.addCleanup(server.stop)
        payload = {"type": "vision_window", "device_id": "cam-01",
                   "face_bbox": [80, 60, 300, 260]}
        sent = server.broadcast(payload)
        self.assertEqual(sent, 0)
        self.assertEqual(server.stats["privacy_blocked"], 1)

    def test_broadcast_still_drops_jpeg_bytes_stuffed_into_a_message(self):
        """把 JPEG 字节塞进报文 —— 键名看着人畜无害，值一看就是像素。"""
        server = build_server(source_kind="synthetic", real_time=False)
        self.addCleanup(server.stop)
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64
        sent = server.broadcast({"type": "vision_window", "device_id": "cam-01",
                                 "data": jpeg})
        self.assertEqual(sent, 0)
        self.assertEqual(server.stats["privacy_blocked"], 1)

    def test_a_jpeg_blob_raises_even_under_an_innocent_key(self):
        """兜底那条（值是二进制就一定违规）必须还在。

        键名可以随便起，值兜不住 —— 否则"给字段起个 data 的名字就能绕过去"。
        """
        with self.assertRaises(PrivacyViolationError):
            assert_clean({"type": "vision_window", "payload": b"\xff\xd8\xff"})

    def test_a_numpy_frame_raises_too(self):
        with self.assertRaises(PrivacyViolationError):
            assert_clean({"type": "vision_window", "payload": np.zeros((4, 4, 3))})

    def test_the_clean_attestation_payload_still_passes(self):
        """反向的用例同样必要：闸门不许宽到拦下自己的干净报文。

        ``privacy.raw_frame_uploaded`` / ``frames_total`` 这些名字里带敏感词、
        语义却是"声明与计数"的字段，靠 :data:`SAFE_KEYS` 放行。它们要是被
        误拦，表现是**每一帧都发不出去**，而日志上只看到一片"被隐私闸拦下"。
        """
        assert_clean({
            "type": "vision_window",
            "privacy": {"raw_frame_uploaded": False, "face_image_uploaded": False,
                        "frame_retention": "memory_only"},
            "window": {"frames_total": 30, "frames_valid": 28},
        })
        self.assertEqual(find_violations({"type": "vision_window"}), [])

    def test_the_stream_path_does_not_touch_broadcast(self):
        """展示流开着的时候，总线的行为也必须一模一样。

        这是"两条路互不重叠"最直白的可执行版本：同一个脏报文，
        开不开展示流，结果都得是被拦下。
        """
        hub = FrameHub()
        server = build_server(source_kind="synthetic", real_time=False, stream=hub)
        self.addCleanup(server.stop)
        sent = server.broadcast({"type": "vision_window", "face_bbox": [1, 2, 3, 4]})
        self.assertEqual(sent, 0)
        self.assertEqual(server.stats["privacy_blocked"], 1)
        # 展示流那边一次都没有被喂过东西 —— 它只认 _publish_frame 那条路。
        self.assertEqual(hub.stats["offered"], 0)


# ================================================================ 叠加

class TestOverlay(unittest.TestCase):
    """叠加层：**只有 ASCII**，且画在副本上。"""

    def test_overlay_text_is_pure_ascii(self):
        """ASCII 这条不是洁癖：``cv2.putText`` 用的是 Hershey 字库，
        它画不出中文 —— 而且不报错，静默变成 ``????``。

        所以这里钉死：任何一个叠加字符的码点都必须 < 128。
        谁往叠加文案里加一句中文解释，上线时看到的就是一串问号。
        """
        for features in (frame(), frame(has_face=False)):
            for line in overlay_lines(features, (640, 480)):
                with self.subTest(line=line):
                    self.assertTrue(line.isascii(), f"叠加里有非 ASCII 字符：{line!r}")
                    self.assertTrue(all(ord(ch) < 128 for ch in line))

    def test_overlay_flags_a_missing_index_instead_of_printing_zero(self):
        """``gaze_off_ratio is None`` 是"这一帧估不出来"，不是"0"。

        与 B 侧 ``index is None ≠ 0`` 是同一条规矩，只是发生在叠加层。
        """
        feats = dataclasses.replace(frame(), gaze_off_ratio=None)
        text = " ".join(overlay_lines(feats, (640, 480)))
        self.assertIn("gaze=--", text)
        self.assertNotIn("gaze=0.00", text)

    def test_overlay_survives_garbage_features(self):
        """字段类型乱来也不许抛 —— 这条路径跑在采集主循环里。"""
        for bad in (None, object(), FrameFeatures()):
            with self.subTest(bad=type(bad).__name__):
                self.assertTrue(overlay_lines(bad, (640, 480)))

    @_needs_cv2
    def test_draw_overlay_paints_green_and_leaves_the_input_alone(self):
        pixels = np.zeros((480, 640, 3), dtype=np.uint8)
        before = pixels.copy()
        out = draw_overlay(pixels, frame())
        self.assertIsNot(out, pixels)
        np.testing.assert_array_equal(pixels, before, "原缓冲被改了")
        green = int(((out[:, :, 1] > 200) & (out[:, :, 0] < 80) & (out[:, :, 2] < 80)).sum())
        self.assertGreater(green, 500, "绿框与绿字都没画出来")

    @_needs_cv2
    def test_draw_overlay_without_a_bbox_does_not_crash(self):
        for feats in (frame(has_face=False, bbox=None), FrameFeatures()):
            with self.subTest(ts=feats.ts):
                out = draw_overlay(np.zeros((240, 320, 3), dtype=np.uint8), feats)
                self.assertEqual(out.shape, (240, 320, 3))


# ================================================================ 单槽缓存

class TestFrameHub(unittest.TestCase):

    def test_nothing_is_encoded_when_nobody_is_watching(self):
        """**这是这个类存在的第二个理由。** 没人看的时候连 cv2 都不该调。"""
        calls = []
        real = stream_mod.encode_jpeg
        stream_mod.encode_jpeg = lambda img, q=stream_mod.JPEG_QUALITY: (
            calls.append(1), real(img, q))[1]
        self.addCleanup(lambda: setattr(stream_mod, "encode_jpeg", real))

        hub = FrameHub()
        self.assertFalse(hub.wanted)
        self.assertFalse(hub.offer(np.zeros((16, 16, 3), dtype=np.uint8), frame()))
        self.assertEqual(calls, [])
        self.assertEqual(hub.stats["offered"], 0)

    def test_only_the_newest_frame_survives(self):
        """单槽：最新的胜出，**永不排队**。"""
        hub = FrameHub()
        hub.subscribe()
        hub.publish(b"old")
        hub.publish(b"new")
        self.assertEqual(hub.stats["replaced"], 1)
        got = hub.wait(0, timeout=0.1)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], b"new")

    def test_wait_honours_since_so_slow_clients_skip_frames_instead_of_queueing(self):
        hub = FrameHub()
        hub.subscribe()
        hub.publish(b"a")
        seq, jpeg = hub.wait(0, timeout=0.1)
        self.assertEqual(jpeg, b"a")
        self.assertIsNone(hub.wait(seq, timeout=0.05), "没有新帧时应当超时返回 None")
        hub.publish(b"b")
        self.assertEqual(hub.wait(seq, timeout=0.1)[1], b"b")

    def test_two_clients_both_see_the_same_latest_frame(self):
        """每个客户端自己记序号，所以不是互相抢同一份。"""
        hub = FrameHub()
        hub.subscribe()
        hub.subscribe()
        hub.publish(b"x")
        self.assertEqual(hub.wait(0, 0.1)[1], b"x")
        self.assertEqual(hub.wait(0, 0.1)[1], b"x")

    def test_the_client_cap_is_enforced(self):
        hub = FrameHub(max_clients=2)
        self.assertTrue(hub.subscribe())
        self.assertTrue(hub.subscribe())
        self.assertFalse(hub.subscribe(), "超过上限还在收")

    def test_close_wakes_every_waiter(self):
        """关服务时卡在 wait() 上的线程必须立刻醒 —— 否则退出要等满超时。"""
        hub = FrameHub()
        hub.subscribe()
        woke = threading.Event()

        def blocked():
            hub.wait(0, timeout=30.0)
            woke.set()

        threading.Thread(target=blocked, daemon=True).start()
        time.sleep(0.1)
        hub.close()
        self.assertTrue(woke.wait(2.0), "close() 没叫醒等着的线程")

    def test_health_line_carries_counts_and_nothing_else(self):
        hub = FrameHub()
        line = hub.health_line()
        self.assertTrue(line.startswith("ok clients=0/"))
        self.assertTrue(line.isascii())


# ================================================================ HTTP 服务

def _get(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read(200)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, exc.read(200)


@_needs_cv2
class TestStreamServer(_QuietStdout):
    """真起一个 HTTP 服务（``port=0``），把客户端的实际观感钉住。"""

    def setUp(self):
        super().setUp()          # 先消音：start() 会打一行横幅
        self.hub = FrameHub()
        self.server = MjpegStreamServer(self.hub, port=0)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://{BIND_HOST}:{self.server.port}"

    def test_it_binds_the_loopback_only_and_says_so(self):
        """**只绑回环**是这个例外的第一条边界，用真实 socket 地址来钉。"""
        self.assertEqual(self.server.host, BIND_HOST)
        self.assertEqual(self.server.host, "127.0.0.1")
        # 从真实绑定的 socket 上再确认一次（配置写对了但绑错的事真会发生）
        httpd = self.server._httpd
        self.assertEqual(httpd.server_address[0], BIND_HOST)

    def test_the_port_is_not_configurable_from_the_command_line(self):
        """地址不可配置是**故意的**：最窄的实现就是让它无法被放宽。"""
        from module_a_vision import main as a_main

        dests = {action.dest for action in a_main.build_parser()._actions}
        self.assertNotIn("stream_host", dests)
        self.assertIn("stream_port", dests)

    def test_healthz_reports_counts(self):
        status, body = _get(f"{self.base}/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"ok clients="), body)

    def test_an_unknown_path_gets_a_404_response_not_a_dropped_connection(self):
        """回归用例：这一条曾经真的坏过。

        ``send_error(404, "……中文说明……")`` 的中文会进**状态行**，而
        ``BaseHTTPRequestHandler`` 用 ``latin-1`` 编码那一行 —— 抛
        ``UnicodeEncodeError``、连接当场被掐。客户端看到的是
        ``RemoteDisconnected``（"服务器没响应"），而真正的原因在服务端。
        所以这里断言的是"拿到了一个 404 **响应**"，不只是"没成功"。
        """
        status, _body = _get(f"{self.base}/")
        self.assertEqual(status, 404)

    def test_the_stream_speaks_multipart(self):
        """分区格式要被**逐字节**钉住：C 侧那个手写的状态机就靠它切帧。

        两个细节值得说明：

        * 先投帧再连。MJPEG 是"服务器推"的，连上之后再投会有一个
          "连上了但暂时没数据"的窗口，测试就得多等一轮超时。
        * 用 ``read1`` 而不是 ``read``。``HTTPResponse.read(n)`` 会一直攒够
          ``n`` 字节才返回，而这个响应没有 ``Content-Length``，攒不够就是
          干等到超时 —— 一个 200 字节的分区配 ``read(4096)`` 必超时。
        """
        payload = b"\xff\xd8\xff\xe0" + b"z" * 128          # 4 + 128 = 132
        self.hub.publish(payload)
        req = urllib.request.Request(f"{self.base}/stream.mjpeg")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("multipart/x-mixed-replace", resp.headers["Content-Type"])
            self.assertIn("boundary=frame", resp.headers["Content-Type"])
            buf = b""
            deadline = time.time() + 5
            while b"\xff\xd8\xff" not in buf and time.time() < deadline:
                chunk = resp.read1(8192)
                if not chunk:
                    break
                buf += chunk
        self.assertIn(b"--frame\r\n", buf)
        self.assertIn(b"Content-Type: image/jpeg\r\n", buf)
        self.assertIn(b"Content-Length: 132\r\n\r\n", buf)
        self.assertIn(payload, buf, "分区里的 JPEG 与投进去的不是同一份字节")

    def test_too_many_clients_gets_a_503(self):
        hub = FrameHub(max_clients=0)
        server = MjpegStreamServer(hub, port=0)
        server.start()
        self.addCleanup(server.stop)
        status, body = _get(f"http://{BIND_HOST}:{server.port}/stream.mjpeg")
        self.assertEqual(status, 503)
        self.assertIn("上限".encode("utf-8"), body)

    def test_a_client_that_resets_before_asking_prints_no_traceback(self):
        """客户端"连上就断"不许在 A 的终端上打 traceback。

        这不是洁癖。C 侧 ``MjpegClient.stop()`` 走的是
        ``shutdown(SHUT_RDWR)`` + ``close()``，重连退避期间也随时可能掐掉
        一条刚建好的连接 —— 也就是说**这条噪声每次停 C 都会出现一次**。
        ``socketserver`` 的默认实现在读请求行失败时会往 stderr 打一整段
        traceback，看上去和"视觉模块崩了"一模一样。演示现场看见它，
        第一反应必然是去查一个根本没坏的东西。

        这里用 ``SO_LINGER=(1,0)`` 主动发 RST（普通的 ``close()`` 发的是
        FIN，``readline`` 收到空串会安静地收场，复现不出这个问题）。
        """
        sockserver = self.server._httpd
        before = sockserver.stats["client_resets"]

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            sock = socket.create_connection((BIND_HOST, self.server.port), timeout=5)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                            struct.pack("hh", 1, 0))     # 1 = 开启，0 = 不等待
            sock.close()
            deadline = time.time() + 5
            while sockserver.stats["client_resets"] == before and time.time() < deadline:
                time.sleep(0.02)

        self.assertNotIn("Traceback", err.getvalue(),
                         f"客户端断开被当成异常打了出去：\n{err.getvalue()}")
        self.assertEqual(sockserver.stats["client_resets"], before + 1,
                         "断开没有被记进 client_resets（说明它既没被吞、也没被计数）")

    def test_a_client_that_vanishes_does_not_leave_the_slot_taken(self):
        """客户端断了必须**归还名额** —— 否则反复重连会把 4 个位置占满。

        注意这里的驱动方式：**得喂帧**。断开这件事服务端只在 ``write`` 时
        才发现，而没帧可写时 handler 就一直在 ``wait`` 超时循环里转。
        真机上 15fps 不停有帧，所以这个"要喂帧"只存在于测试里；反过来说，
        它也说明名额的回收不是靠"定时探活"，而是靠"下一帧写不出去"。
        """
        sock = socket.create_connection((BIND_HOST, self.server.port), timeout=5)
        sock.sendall(b"GET /stream.mjpeg HTTP/1.0\r\n\r\n")
        sock.recv(1024)
        deadline = time.time() + 3
        while self.hub._clients == 0 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.hub._clients, 1)

        sock.close()
        deadline = time.time() + 10
        while self.hub._clients and time.time() < deadline:
            self.hub.publish(b"\xff\xd8\xff\xe0" + b"z" * 64)
            time.sleep(0.05)
        self.assertEqual(self.hub._clients, 0, "断开后名额没有归还")


# ================================================================ 同源像素源

class TestSyntheticPixelSource(unittest.TestCase):
    """``synthetic-pixels``：像素与特征**同一次剧本推进**的产物。"""

    def make(self, **kw):
        src = SyntheticPixelSource(scenario="drowsy", **kw)
        self.addCleanup(src.close)
        src.open()
        return src

    def test_it_is_a_frame_source_and_not_a_feature_source(self):
        """**这条决定了派发走哪条分支。**

        ``server._read_one`` 是"特征源优先"的：它只要还有
        ``read_features()``，那条分支就会抢先命中、像素永远拿不到。
        """
        src = self.make()
        self.assertTrue(is_frame_source(src))
        self.assertFalse(is_feature_source(src))

    @_needs_cv2
    def test_read_gives_pixels_and_process_gives_the_features_of_the_same_frame(self):
        src = self.make()
        pixels = src.read()
        self.assertIsInstance(pixels, np.ndarray)
        self.assertEqual(pixels.shape, (src.cfg.height, src.cfg.width, 3))
        feats = src.process(pixels)
        self.assertEqual(feats.ts, src.last_ts)

    @_needs_cv2
    def test_the_features_follow_the_script_not_a_constant(self):
        """drowsy 段必须真的歪头 —— 否则"画面上看得见头歪"就成了巧合。"""
        src = self.make()
        seen = []
        for _ in range(getattr(src, "_script").frame_index + 400):
            pixels = src.read()
            if pixels is None:
                break
            seen.append(src.process(pixels).roll_deg)
        self.assertTrue(seen)
        self.assertGreater(max(abs(r) for r in seen), 20.0, "合成剧本没走到 drowsy 段")

    def test_process_without_read_is_an_error(self):
        """顺序错了必须报错：静默返回上一帧的特征会让画面与文字差一帧。"""
        src = self.make()
        with self.assertRaises(CaptureError):
            src.process(np.zeros((8, 8, 3), dtype=np.uint8))

    @_needs_cv2
    def test_process_twice_in_a_row_is_an_error(self):
        src = self.make()
        pixels = src.read()
        src.process(pixels)
        with self.assertRaises(CaptureError):
            src.process(pixels)

    def test_build_server_wires_the_source_as_its_own_face_backend(self):
        server = build_server(source_kind="synthetic-pixels", real_time=False)
        self.addCleanup(server.stop)
        self.assertIs(server.backend, server.source)


# ================================================================ 读取与清零

class TestReadOneAndScrub(_QuietStdout):

    def test_a_feature_source_returns_no_pixels(self):
        server = build_server(source_kind="synthetic", real_time=False)
        self.addCleanup(server.stop)
        server.source.open()
        step = server._read_one()
        self.assertIsNotNone(step)
        feats, pixels = step
        self.assertIsNone(pixels, "特征源不该凭空多出像素")
        self.assertIsNotNone(feats)

    @_needs_cv2
    def test_a_frame_source_returns_both(self):
        server = build_server(source_kind="synthetic-pixels", real_time=False)
        self.addCleanup(server.stop)
        server.source.open()
        feats, pixels = server._read_one()
        self.assertIsInstance(pixels, np.ndarray)
        self.assertEqual(feats.ts, server.source.last_ts)

    def test_the_server_never_stashes_the_pixels(self):
        """**"模块内不留像素"必须是事实，而不是纪律。**

        挂一个 ``self._last_pixels`` 会让它在两次读之间一直活着 ——
        这条断言就是防止那种改动的。
        """
        server = build_server(source_kind="synthetic", real_time=False)
        self.addCleanup(server.stop)
        leftovers = [name for name in vars(server) if "pixel" in name or "frame_" in name]
        self.assertEqual(leftovers, [])
        self.assertFalse(hasattr(server, "_last_pixels"))

    def test_scrub_zeroes_the_buffer_and_counts_it(self):
        server = build_server(source_kind="synthetic", real_time=False)
        self.addCleanup(server.stop)
        pixels = np.full((4, 4, 3), 200, dtype=np.uint8)
        server._scrub(pixels)
        self.assertEqual(int(pixels.sum()), 0)
        self.assertEqual(server.stats["frames_scrubbed"], 1)
        self.assertEqual(server.stats["scrub_skipped"], 0)

    def test_a_read_only_frame_is_skipped_instead_of_killing_the_loop(self):
        """清不掉就跳过并计数 —— **绝不能**让采集主循环崩掉。

        ``scrub_frame`` 对只读缓冲是**故意**抛错的（"以为清零了其实没有"
        比报错危险）。这里接住它，但账要记下来。
        """
        server = build_server(source_kind="synthetic", real_time=False)
        self.addCleanup(server.stop)
        frozen = np.zeros((4, 4, 3), dtype=np.uint8)
        frozen.flags.writeable = False
        server._scrub(frozen)          # 不抛
        self.assertEqual(server.stats["scrub_skipped"], 1)
        self.assertEqual(server.stats["frames_scrubbed"], 0)

    def test_scrub_frame_still_refuses_a_read_only_buffer(self):
        """上一条接住的错误，本身也必须是**真的**抛出来的。"""
        frozen = np.zeros((2, 2, 3), dtype=np.uint8)
        frozen.flags.writeable = False
        with self.assertRaises(ValueError):
            scrub_frame(frozen)

    def test_without_a_stream_nothing_is_scrubbed(self):
        """不给 ``--stream`` 就没有人经手像素，也就没有"用完"这一说。"""
        server = build_server(source_kind="synthetic", real_time=False)
        self.addCleanup(server.stop)
        pixels = np.full((4, 4, 3), 7, dtype=np.uint8)
        server._publish_frame(FrameFeatures(), pixels)
        self.assertEqual(int(pixels.sum()), 4 * 4 * 3 * 7, "没开展示流却动了像素")
        self.assertEqual(server.stats["frames_scrubbed"], 0)

    @_needs_cv2
    def test_with_a_stream_the_pixels_are_zeroed_after_publishing(self):
        hub = FrameHub()
        hub.subscribe()
        server = build_server(source_kind="synthetic-pixels", real_time=False, stream=hub)
        self.addCleanup(server.stop)
        pixels = np.full((16, 16, 3), 200, dtype=np.uint8)
        server._publish_frame(frame(), pixels)
        self.assertEqual(int(pixels.sum()), 0, "编码之后没有清零")
        self.assertEqual(server.stats["frames_scrubbed"], 1)
        self.assertEqual(hub.stats["offered"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
