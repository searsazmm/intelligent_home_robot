"""展示用 MJPEG 流 —— 「像素不出模块 A」这条红线**唯一的窄例外**。

--------------------------------------------------------------------------
先把这件事说清楚：这不是把红线放宽，而是给它开了一条**可以被描述完**的口子
--------------------------------------------------------------------------

:mod:`module_a_vision.privacy.guard` 里那条闸门管的是 **TCP 8000 上给 B 的
报文**。展示流走的是另一条路：HTTP / MJPEG，只到 ``127.0.0.1``，只给同机的
frontend_C。两条路互不重叠，**``guard.py`` 一个字符都没有改** ——
``SENSITIVE_TOKENS`` 仍然拦着 ``bbox`` / ``pixel`` / ``jpeg`` / ``frame``，
``assert_clean`` 仍然逐条扫过每一份出站报文。任何人往 8000 的报文里塞一帧
像素，行为与从前完全一样：被拦下、被计数、被丢弃。

例外的边界写在这里，一条一条都收得很窄：

============================  ==========================================
只绑 ``127.0.0.1``            硬编码 :data:`BIND_HOST`，**没有 CLI 覆盖** ——
                              连"不小心听 0.0.0.0"这个可能性都不存在
默认关闭                      ``--stream`` 不给就不起来，8010 上**没有监听**
不落盘                        本模块只有 ``imencode``；没有 ``imwrite``，
                              没有 ``open(..., 'wb')``，没有文件名参数
没有索引                      不做 HTML 首页、不做目录列表；只有两个路径
用完即毁                      编完码调用方立刻 ``scrub_frame(pixels)``
============================  ==========================================

--------------------------------------------------------------------------
这个文件里唯一一件"不是像素"的事：叠加
--------------------------------------------------------------------------

画框要 ``face_bbox``，而它来自 :class:`~shared.frame_features.FrameFeatures`，
所以像素与特征必须**在同一处**相遇 —— 那处只能是这里（``_read_one`` 之后、
``scrub_frame`` 之前）。顺序因此是死的：**先 process 再发布**，
反过来的话框会比画面晚一帧，头一转就能看出来。

⚠️ **叠加只有 ASCII。** ``cv2.putText`` 用的是 Hershey 字库，它**画不出中文**
（不报错，静默变成 ``????``）。所以这里只出原始量（``pitch=12.0``），
中文解释留在 C 侧的 :mod:`c_core.display_text`。想在 A 这边看到中文标注的
话，得先换一套带中文字形的绘制库 —— 那是另一个量级的改动。
"""

from __future__ import annotations

import contextlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

try:  # 没装 opencv 时本模块仍可 import（测试与 /healthz 都不需要像素）
    import cv2
except ImportError:  # pragma: no cover - 本机已装
    cv2 = None  # type: ignore[assignment]

#: 展示流的默认端口。刻意避开 ``HTTP接口文档.md`` 给 B 预留的 8020。
DEFAULT_STREAM_PORT = 8010

#: 监听地址。**硬编码，不提供 CLI 覆盖** —— "例外写窄"最窄的实现，
#: 就是让它无法被命令行放宽。
BIND_HOST = "127.0.0.1"

#: 同时拉流的客户端上限。与 ``server.py`` 的 ``max_clients=4`` 同一个取舍：
#: 不无限收，否则文件描述符会被反复重连的客户端吃光。
MAX_STREAM_CLIENTS = 4

#: multipart 分界串（报文里写 ``--frame``）。
BOUNDARY = "frame"

#: JPEG 质量。80 在 640x480 下约 30KB/帧、15fps ≈ 450KB/s，
#: 回环上无压力；再高只是让 CPU 多转，人眼看不出区别。
JPEG_QUALITY = 80

#: 只认这两个路径。**不做 HTML 首页** —— 本需求用不上，且会给仓库引入
#: 一类新的资产。
PATH_STREAM = "/stream.mjpeg"
PATH_HEALTH = "/healthz"


class StreamError(RuntimeError):
    """展示流起不来或不可用。消息面向运维人员，写清楚怎么办。"""


def _require_cv2() -> Any:
    if cv2 is None:
        raise StreamError(
            "展示流需要 opencv（画叠加与编 JPEG）。请安装 opencv-contrib-python，"
            "或不加 --stream 直接跑（视觉主链路不需要 opencv）。"
        )
    return cv2


# ================================================================ 叠加

def _f(value: Any, digits: int = 1) -> str:
    """把数值格式化成叠加用的一小段文本。取不到就给 ``--``。

    刻意不让它抛：叠加是**展示**，一个字段类型不对不该让整条采集循环停下。
    """
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def overlay_lines(features: Any, size: tuple[int, int]) -> list[str]:
    """叠加要画的两行 **ASCII** 量。中文解释在 C 侧（见模块文档）。"""
    width, height = size
    ts = _f(getattr(features, "ts", None))
    if features is None or not getattr(features, "has_face", False):
        return [f"ts={ts}s face=0", f"{width}x{height} no-face"]

    gaze = getattr(features, "gaze_off_ratio", None)
    return [
        f"ts={ts}s face=1 ear={_f(getattr(features, 'ear_left', None), 2)}"
        f" close={_f(getattr(features, 'closure_ratio', None), 2)}"
        f" gaze={'--' if gaze is None else _f(gaze, 2)}",
        f"pitch={_f(getattr(features, 'pitch_deg', None))}"
        f" yaw={_f(getattr(features, 'yaw_deg', None))}"
        f" roll={_f(getattr(features, 'roll_deg', None))} {width}x{height}",
    ]


def draw_overlay(pixels: Any, features: Any) -> Any:
    """在像素的**副本**上画人脸框与两行量，返回副本。

    两条约定：

    * **画在副本上。** ``pixels`` 是采集源的缓冲，不是我们的；改一个不归
      自己管的缓冲是那种半年后才会咬人的写法。
    * ``face_bbox`` 是**原图分辨率**的框（:mod:`shared.frame_features` 的
      约定），所以直接画在同样的坐标系里，不需要任何缩放。
    """
    cxx = _require_cv2()
    img = pixels.copy()

    bbox = getattr(features, "face_bbox", None)
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cxx.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    height = img.shape[0] if getattr(img, "shape", None) else 480
    scale = max(0.45, min(1.0, height / 480.0 * 0.55))
    y = int(22 * scale) + 8
    for line in overlay_lines(features, (img.shape[1], img.shape[0])):
        cxx.putText(img, line, (10, y), cxx.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 0), 1, cxx.LINE_AA)
        y += int(26 * scale)
    return img


def encode_jpeg(image: Any, quality: int = JPEG_QUALITY) -> bytes:
    """BGR 图 → JPEG 字节。**这里就是"不落盘"那句话的技术形态**：
    整个模块只有 ``imencode``，没有任何写文件的分支。"""
    cxx = _require_cv2()
    ok, buffer = cxx.imencode(".jpg", image, [int(cxx.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise StreamError("JPEG 编码失败（cv2.imencode 返回 False）")
    return bytes(buffer)


# ================================================================ 单槽帧缓存

class FrameHub:
    """**单槽**帧缓存：最新一帧胜出，永不排队。

    为什么是单槽而不是队列：展示流的语义是"让我看到此刻的画面"。排队意味着
    观众看到的是历史，而且客户端越慢积压越多 —— 延迟会自己长大，界面上却
    看不出任何异常（画面一直在动，只是越来越晚）。单槽把"慢"直接翻译成
    "掉帧"，这是两种表现里唯一不骗人的那种。

    **无订阅者时一次 cv2 都不调**：:meth:`offer` 的第一句就是
    :attr:`wanted` 这个整数判断。这是本类存在的第二个理由 —— 让"没人在看"
    这件事在真机上真的不花代价。演示时顺手起 A 而不起 C，不该因此多烧
    一路 CPU。
    """

    def __init__(self, max_clients: int = MAX_STREAM_CLIENTS) -> None:
        self.max_clients = max_clients
        self._cond = threading.Condition()
        self._slot: Optional[bytes] = None
        self._seq = 0
        self._clients = 0
        self._closed = False
        self.stats = {
            #: 有订阅者时被交付上来的帧数（``offer`` 真的走到了编码那一步）
            "offered": 0,
            #: 上一帧还没被任何客户端取走就被顶掉的次数。
            #: **它不为零是正常的**（客户端比采集慢），但持续很大就说明
            #: 客户端卡住了 —— 那时画面会一顿一顿的。
            "replaced": 0,
            #: 被客户端取走的份数（4 个客户端各取一帧算 4）
            "served": 0,
            #: 编码/叠画出错被丢弃的帧数。非零就该看一眼日志。
            "encode_failed": 0,
            #: 接受过的客户端连接数（含已断开的）
            "subscribers": 0,
        }

    # ------------------------------------------------------------ 订阅

    @property
    def wanted(self) -> bool:
        """现在有没有人在看。**不加锁**：一个 int 的读写在 CPython 下是原子的，
        而这个属性在采集循环里每帧都要问一次，加锁反而是拿主链路的开销
        去换一个无关紧要的一致性。"""
        return self._clients > 0 and not self._closed

    @property
    def closed(self) -> bool:
        return self._closed

    def subscribe(self) -> bool:
        """登记一个客户端。超过上限或已关闭时返回 ``False``。"""
        with self._cond:
            if self._closed or self._clients >= self.max_clients:
                return False
            self._clients += 1
            self.stats["subscribers"] += 1
            return True

    def unsubscribe(self) -> None:
        with self._cond:
            if self._clients > 0:
                self._clients -= 1

    def close(self) -> None:
        """叫醒所有等着取帧的线程。幂等。"""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ------------------------------------------------------------ 生产

    def publish(self, jpeg: bytes) -> None:
        """存一帧并叫醒所有等着的客户端。"""
        with self._cond:
            if self._slot is not None:
                self.stats["replaced"] += 1
            self._slot = jpeg
            self._seq += 1
            self._cond.notify_all()

    def offer(self, pixels: Any, features: Any) -> bool:
        """把这一帧画上叠加、编成 JPEG、存进单槽。返回是否真的处理了它。

        **没人看时第一句就返回**，一次 cv2 都不调（见类文档）。

        整个过程包在 ``except Exception`` 里：展示流是**旁路**，
        它出任何问题都不该把视觉主链路带走 —— 那意味着老人的状态判定
        因为"演示用的画面"而停摆。
        """
        if pixels is None or not self.wanted:
            return False
        self.stats["offered"] += 1
        try:
            self.publish(encode_jpeg(draw_overlay(pixels, features)))
            return True
        except Exception as exc:  # noqa: BLE001 - 旁路不许崩主链路
            self.stats["encode_failed"] += 1
            if self.stats["encode_failed"] == 1:
                print(f"[A] ⚠ 展示流编码失败，已丢弃该帧（不影响采集）：{exc}")
            return False

    # ------------------------------------------------------------ 消费

    def wait(self, since: int, timeout: float) -> Optional[tuple[int, bytes]]:
        """等一帧比 ``since`` 新的画面，返回 ``(序号, JPEG 字节)``。

        每个客户端自己记住上一次拿到的序号，所以**所有客户端都能看到最新
        那一帧**，而不是互相抢同一份。超时返回 ``None``（调用方借这个空档
        回头看一眼"是不是该退出了"）。
        """
        with self._cond:
            if self._seq <= since and not self._closed:
                self._cond.wait(timeout)
            if self._seq <= since or self._slot is None:
                return None
            self.stats["served"] += 1
            return self._seq, self._slot

    def health_line(self) -> str:
        """``/healthz`` 的那一行。**只有计数，没有任何一帧的内容。**"""
        s = self.stats
        return (
            f"ok clients={self._clients}/{self.max_clients}"
            f" accepted={s['subscribers']} offered={s['offered']}"
            f" served={s['served']} replaced={s['replaced']}"
            f" encode_failed={s['encode_failed']}"
        )


# ================================================================ HTTP 服务

class _StreamRequestHandler(BaseHTTPRequestHandler):
    """只认 ``/stream.mjpeg`` 与 ``/healthz`` 两个路径，其余 404。

    ``protocol_version`` 取 **HTTP/1.0**：这个响应**没有** ``Content-Length``、
    也没有 ``Transfer-Encoding`` —— 它的结束信号就是"连接关掉"。HTTP/1.1 的
    默认语义是长连接，于是"连接什么时候结束"变成一件要额外声明的事；
    HTTP/1.0 把"关掉即结束"作为默认，正好对上 MJPEG 的形状。
    """

    protocol_version = "HTTP/1.0"
    #: 读写超时。**不是可有可无的**：客户端拔网线之后 ``wfile.write`` 会
    #: 一直阻塞，没有它，那条线程就永远挂在那个 write 上（daemon 线程不会
    #: 阻止进程退出，但会让"关掉一个客户端"变成只能整进程重启）。
    timeout = 5.0

    server_version = "AirVisionStream/1.0"

    # -------------------------------------------------------- 日志
    def log_message(self, fmt: str, *args: Any) -> None:
        """**故意什么都不打。** 一帧一条日志会瞬间刷屏，而且默认走 stderr，
        会与 ``[A]`` 的 stdout 交错成看不懂的东西。异常状态另有计数。"""

    def log_error(self, fmt: str, *args: Any) -> None:
        self.server.stats["handler_errors"] += 1

    # -------------------------------------------------------- 路由
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的约定
        if self.path == PATH_STREAM:
            self._serve_stream()
        elif self.path == PATH_HEALTH:
            self._plain(200, self.server.hub.health_line())
        else:
            self._plain(404, "Not Found —— 本服务只有 /stream.mjpeg 与 /healthz")

    def _plain(self, status: int, text: str) -> None:
        """回一段纯文本。

        ⚠️ **状态行的 reason phrase 必须是 ASCII。** ``send_response`` 用
        ``latin-1`` 编码那一行，塞中文进去会当场抛 ``UnicodeEncodeError``、
        把连接直接掐掉 —— 客户端看到的是"服务器没有响应"，而真正的原因在
        服务端。所以解释性文字一律进 **body**（UTF-8），状态行只用标准短语。
        这个坑踩过一次：`/` 返回 404 时客户端拿到的是 RemoteDisconnected。
        """
        # 用 ``send_response(status)`` 而不是 ``send_error``：后者的 message
        # 参数会进状态行，正中上面那个坑。
        body = (text + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        with contextlib.suppress(OSError):
            self.wfile.write(body)

    def _serve_stream(self) -> None:
        hub = self.server.hub
        if not hub.subscribe():
            self.server.stats["rejected"] += 1
            self._plain(503, f"展示流已满（上限 {hub.max_clients} 个，或 A 正在退出）")
            return

        self.server.stats["streams"] += 1
        sent = 0
        last = 0
        try:
            self.send_response(200)
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            while not self.server.stopping.is_set() and not hub.closed:
                got = hub.wait(last, 1.0)
                if got is None:
                    continue        # 超时：回去看一眼是不是该退出了
                last, jpeg = got
                self.wfile.write(f"--{BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                sent += 1
        except (OSError, ValueError):
            # 客户端走了（Windows 上多为 ConnectionResetError）、或者写超时。
            # 这是**正常结束**的一种，不是错误 —— 谁也不会跑过来通知一声。
            pass
        finally:
            hub.unsubscribe()
            self.server.stats["frames_sent"] += sent
            self.close_connection = True


class _StreamServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` 的定型：一个客户端一条线程，进程退出不等它们。"""

    daemon_threads = True
    allow_reuse_address = True
    #: ``ThreadingMixIn`` 默认会在 ``server_close()`` 里 join 所有 handler
    #: 线程。一个还连着流、正卡在 ``wfile.write`` 上的客户端会让"关服务"
    #: 卡满超时。关掉它：线程都是 daemon，进程退出时自然收场。
    block_on_close = False

    #: 客户端"连上又立刻断"时算正常收场，不打 traceback 的那几个异常。
    #: ``ConnectionResetError`` 是 C 侧 ``MjpegClient.stop()`` 的必然产物
    #: （它先 ``shutdown(SHUT_RDWR)`` 再 ``close()``），Windows 上会以
    #: ``WSAECONNRESET`` 的形式在**读请求行**时就炸出来。
    _QUIET_RESETS = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)

    def handle_error(self, request, client_address) -> None:
        """把"客户端断开"从错误降级成计数，其余照旧抛 traceback。

        默认实现在**读请求行**失败时会往 stderr 打一整段 traceback。这段
        噪声很坏：C 侧每次 stop()／重连都会触发一次，于是 A 的终端上看着
        像在崩，而实际上什么都没坏 —— 演示现场看到这个，第一反应一定是
        "视觉模块挂了"。

        **只吞这三种**。真出 bug 时留下的仍是完整 traceback，这条修改
        不会把任何真实故障藏起来。
        """
        if isinstance(sys.exc_info()[1], self._QUIET_RESETS):
            self.stats["client_resets"] += 1
            return
        super().handle_error(request, client_address)

    def __init__(self, addr, handler, hub: FrameHub) -> None:
        super().__init__(addr, handler)
        self.hub = hub
        self.stopping = threading.Event()
        self.stats = {
            "streams": 0,           # 成功建立的流
            "rejected": 0,          # 超过上限被 503 的
            "frames_sent": 0,       # 累计发出去的分区数
            "handler_errors": 0,
            "client_resets": 0,     # 连上就立刻断掉的客户端（见 handle_error）
        }


class MjpegStreamServer:
    """展示流服务。``start()`` 非阻塞；绑不上端口时**抛 OSError**，由调用方决定。

    ⚠️ 端口绑定失败**绝不能**拖垮 TCP 8000 那条主链路。这个类的调用方
    （``main.py``）接住 OSError 之后只打印一条提示就继续跑 —— 视觉功能的
    可用性不能取决于一个演示用的旁路。
    """

    def __init__(self, hub: FrameHub, port: int = DEFAULT_STREAM_PORT,
                 host: str = BIND_HOST) -> None:
        self.hub = hub
        self.host = host          # 不给 CLI 覆盖：见 BIND_HOST 的说明
        self.port = port
        self._httpd: Optional[_StreamServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{PATH_STREAM}"

    @property
    def stats(self) -> dict:
        return self._httpd.stats if self._httpd is not None else {}

    def start(self) -> None:
        """绑定并开始接受连接。**非阻塞**。

        :raises OSError: 端口被占用或地址不可绑。调用方应当只打印提示，
            不要让整个进程失败。
        """
        _require_cv2()
        httpd = _StreamServer((self.host, self.port), _StreamRequestHandler, self.hub)
        # 传 port=0 时由系统挑端口；记下真实值，否则横幅会打印一个看起来
        # 像"没起来"的 8010:0。
        self.port = httpd.server_address[1]
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever,
                                        name="a-mjpeg", daemon=True)
        self._thread.start()
        print(
            f"[A] 展示流已监听 {self.host}:{self.port}{PATH_STREAM}"
            f"（**只绑本机回环**，不进任何报文；像素不落盘）"
        )

    def stop(self) -> None:
        """停下服务。幂等。"""
        httpd = self._httpd
        if httpd is None:
            return
        httpd.stopping.set()
        self.hub.close()          # 叫醒所有卡在 wait() 上的 handler
        with contextlib.suppress(Exception):
            httpd.shutdown()
        with contextlib.suppress(Exception):
            httpd.server_close()
        self._httpd = None
        self._thread = None


__all__ = [
    "BIND_HOST",
    "BOUNDARY",
    "DEFAULT_STREAM_PORT",
    "JPEG_QUALITY",
    "MAX_STREAM_CLIENTS",
    "PATH_HEALTH",
    "PATH_STREAM",
    "FrameHub",
    "MjpegStreamServer",
    "StreamError",
    "draw_overlay",
    "encode_jpeg",
    "overlay_lines",
]
