# -*- coding: utf-8 -*-
"""从 A 侧拉 MJPEG 展示流，把一帧帧 JPEG 交出来。

**纯标准库、零 Qt。** 与 :mod:`c_core.state_client` 同一条约定：这里只做
传输，不碰任何控件、不解码图像。解码在 ``ui/camera.py``，用的是 PyQt5 自带的
JPEG 插件（**不是 cv2** —— 理由见 ``main.py`` 顶部：cv2 与 PyQt5 同进程会
互相顶掉对方的平台插件）。

--------------------------------------------------------------------------
为什么手写 HTTP 与 multipart，而不用 ``http.client``
--------------------------------------------------------------------------
那条响应**既没有 ``Content-Length`` 也没有 ``Transfer-Encoding``** ——
MJPEG 的结束信号就是"连接被关掉"。``http.client`` 对这种情况会退化成
read-until-EOF：``resp.read()`` 一直不返回，直到流断掉为止。也就是说，
拿到的是一整条流的字节（而且攒在一个没有上限的内存里），切帧的工作
还是得自己做。既然如此，直接用原始 socket 更短、也更不容易出错。

--------------------------------------------------------------------------
线程铁律（与 ``ui/window.py`` 的 StateBridge 是同一条）
--------------------------------------------------------------------------
本模块的一切回调都在 **socket 线程**里执行。所以 ``on_frame`` **只允许**
把字节塞进 :class:`FrameSlot`（或任何等价的无锁单槽），**绝不能**去调
``QWidget.update()`` / ``QLabel.setPixmap()`` 之类。

理由不只是"跨线程操作控件会崩"，还有一个更隐蔽的：**信号不会合并。**
每帧 emit 一个带 payload 的信号，等于把 Qt 的事件队列当成帧队列 ——
GUI 一旦慢下来，队列只会越堆越长，画面越来越晚，而屏幕上完全看不出来
（画面一直在动，只是延迟在悄悄变大）。所以是 GUI 用 ``QTimer`` **主动拉**：
拉得慢就自然丢掉中间那些帧，这与 A 侧单槽缓存的语义是同一套。
"""

from __future__ import annotations

import logging
import re
import socket
import threading
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: 连接状态回调里的来源标识。必须与 ``ui/window.py`` 的 ``LINK_LABELS``
#: 和 ``c_core/display_text.py`` 的 ``LINK_LABELS`` 里的键**逐字一致** ——
#: 三张表说的是同一条通道。
SOURCE_STREAM = "stream"

#: A 侧展示流的默认端口与路径（``module_a_vision/stream.py`` 的
#: ``DEFAULT_STREAM_PORT`` / ``PATH_STREAM``）。**刻意避开 8010 以外的一切**：
#: 8020 是 ``HTTP接口文档.md`` 给 B 预留的，8001/8002 是 B 的既有通道。
DEFAULT_STREAM_PORT = 8010
DEFAULT_PATH = "/stream.mjpeg"

#: 服务端没在 ``Content-Type`` 里给 boundary 时的兜底（A 侧总是给 "frame"）。
BOUNDARY_FALLBACK = "frame"

#: GUI 默认的取帧频率。**定义在这里而不是 ``ui/camera.py``**，因为
#: ``main.py`` 的 ``build_parser()`` 要拿它当 ``--camera-fps`` 的默认值，
#: 而那个函数跑在 ``_import_qt()`` **之前** —— 从 ``ui.camera`` 里取这个
#: 常数会顺带把 PyQt5 拉进来，在一个还没确认装没装 Qt 的时刻。
#:
#: 它**不跟 A 的 fps 对齐、也不追求对齐**：两边各自的节奏由单槽解耦，
#: 对齐了反而会引入相位问题（C 恰好总是取到刚被顶掉的那一帧）。
DEFAULT_CAMERA_FPS = 15
MIN_CAMERA_FPS = 1
MAX_CAMERA_FPS = 60


def clamp_fps(value: int) -> int:
    """把 ``--camera-fps`` 钳到合理区间。手敲的命令行不该能把定时器打成 0ms。"""
    try:
        fps = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CAMERA_FPS
    return max(MIN_CAMERA_FPS, min(MAX_CAMERA_FPS, fps))

#: 单次 recv 的大小。
RECV_SIZE = 65536
#: 响应头（含状态行）的长度上限。本机回环上它只有几十字节，
#: 超了说明对面根本不是这条流。
MAX_HEADER = 16 * 1024
#: 缓冲上限。正常情况下 BODY 状态最多只攒一帧（有 Content-Length）；
#: 到 4 MiB 还没切出一帧，说明分帧已经错位了，宁可丢掉重来也不要
#: 让内存无上限地涨 —— 那是把"画面卡住"升级成"进程被 OOM 杀掉"。
MAX_BUFFER = 4 * 1024 * 1024

#: JPEG 的起始与结束标记
SOI = b"\xff\xd8\xff"
EOI = b"\xff\xd9"

_BOUNDARY_RE = re.compile(r'boundary="?([^";,]+)"?', re.I)


class StreamError(RuntimeError):
    """展示流的配置或协议出了不可恢复的问题（地址写错、对面不是 MJPEG）。"""


def _noop(*_args, **_kwargs) -> None:
    """默认回调：什么都不做。"""


# ==========================================================================
# 地址
# ==========================================================================

def parse_stream_url(url: str,
                     default_port: int = DEFAULT_STREAM_PORT) -> Tuple[str, int, str]:
    """``http://127.0.0.1:8010/stream.mjpeg`` → ``("127.0.0.1", 8010, "/stream.mjpeg")``。

    写得宽容一点是刻意的：``--stream-url`` 是答辩前手敲的，少写个 ``http://``
    或漏掉路径都很正常，为这两种情况让人去查文档不值得。

    :raises StreamError: 协议不是 http、端口不是数字、或者压根没给地址。
    """
    text = str(url or "").strip()
    if not text:
        raise StreamError("展示流地址是空的")
    # 没有 ``//`` 时补一个，否则 urlsplit 会把 "127.0.0.1" 当成 scheme
    parts = urlsplit(text if "//" in text else f"//{text}")
    if parts.scheme and parts.scheme.lower() != "http":
        raise StreamError(f"展示流只支持 http（收到 {parts.scheme!r}）：{url}")
    try:
        port = parts.port or default_port
    except ValueError as exc:
        raise StreamError(f"展示流地址里的端口不是数字：{url}（{exc}）") from exc
    host = parts.hostname or "127.0.0.1"
    return host, int(port), parts.path or DEFAULT_PATH


# ==========================================================================
# 单槽
# ==========================================================================

class FrameSlot:
    """**单槽**帧缓存：GUI 取慢了就丢，永不排队。

    A 侧已经用单槽顶掉旧帧了，这里再挡一道是因为两边的节奏仍然不同：
    A 按它的 fps 推，C 按 ``--camera-fps`` 拉，中间还隔着一个 TCP。
    队列在这个位置是纯粹的负债 —— 它只会让画面越来越晚，而屏幕上
    完全看不出来（画面一直在动）。

    **本类只有两个方法会被跨线程调用**：socket 线程调 :meth:`put`，
    GUI 线程调 :meth:`take`。两者都用同一把锁，临界区里只有赋值，
    不存在"谁等谁"的问题。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._jpeg: Optional[bytes] = None
        self.stats: Dict[str, int] = {
            #: 放进来过多少帧（含被顶掉的）
            "put": 0,
            #: 上一帧还没被取走就被顶掉的次数。**不为零是正常的**
            #: （GUI 比网络慢），持续很大说明界面卡住了。
            "overwritten": 0,
            #: 被 GUI 取走的次数
            "taken": 0,
        }

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def put(self, jpeg: bytes) -> None:
        """socket 线程调用。**只做这一件事。**"""
        if not jpeg:
            return
        with self._lock:
            if self._jpeg is not None:
                self.stats["overwritten"] += 1
            self._seq += 1
            self._jpeg = jpeg
            self.stats["put"] += 1

    def take(self) -> Optional[Tuple[int, bytes]]:
        """GUI 线程调用。有**新**帧时返回 ``(序号, JPEG)``，否则 ``None``。

        取走即清空，所以"同一帧被画两遍"和"没新帧却重复解码"这两件事
        都不可能发生 —— 15Hz 的定时器比流的 fps 快时不会白解 JPEG。
        """
        with self._lock:
            if self._jpeg is None:
                return None
            jpeg = self._jpeg
            self._jpeg = None
            self.stats["taken"] += 1
            return self._seq, jpeg

    def clear(self) -> None:
        with self._lock:
            self._jpeg = None


# ==========================================================================
# multipart 状态机
# ==========================================================================

class MultipartParser:
    """字节级 multipart 状态机：喂进去字节，吐出 JPEG。

    只有这一个类知道分区的形状，所以 ``tests/test_mjpeg_client.py``
    可以脱离网络把分帧钉死（包括"一个分区分两次到达"这种真机上偶发、
    手测几乎撞不到的情况）。

    分区形状::

        --frame\\r\\n
        Content-Type: image/jpeg\\r\\n
        Content-Length: 12345\\r\\n
        \\r\\n
        <12345 字节的 JPEG>\\r\\n
        --frame\\r\\n
        ...

    长度以 ``Content-Length`` 为准，**不扫 EOI 标记**：JPEG 的压缩数据里
    完全可能出现 ``FF D9``，靠扫标记迟早会切出一个半截的帧，而那种帧
    解码失败的样子是"画面偶尔花一下" —— 最难查的一类问题。没有
    ``Content-Length`` 时才退回扫 EOI（本项目的 A 侧总是给，这条是给
    别家的流留的活路）。
    """

    _SEEK = 0
    _HEADERS = 1
    _BODY = 2

    def __init__(self, boundary: str = BOUNDARY_FALLBACK,
                 max_buffer: int = MAX_BUFFER) -> None:
        self.boundary = str(boundary or BOUNDARY_FALLBACK)
        self.max_buffer = max(1024, int(max_buffer))
        self._sep = f"--{self.boundary}".encode("ascii", "replace")
        self._buf = bytearray()
        self._state = self._SEEK
        self._length = 0
        #: 分帧失配的次数（缓冲超限 / 长度与 JPEG 头对不上）。**应当恒为 0**，
        #: 非零说明对面给的东西不是它声明的那样。
        self.overflows = 0

    @property
    def buffered(self) -> int:
        return len(self._buf)

    def feed(self, chunk: bytes) -> List[bytes]:
        """喂一段字节，返回这段字节里切出来的所有完整帧。"""
        if chunk:
            self._buf += chunk
        frames: List[bytes] = []
        while True:
            advanced, frame = self._step()
            if frame is not None:
                frames.append(frame)
            # **只有"这一步什么都没推动"才收手。** 一次 feed 常常要跨好几个
            # 状态（找边界 → 读头 → 取帧），中间每一步都不产出帧；
            # 把"没产出帧"当成"该收手了"，就会每次 feed 只推进一步 ——
            # 表现是分帧看上去是对的、但慢得离谱（一次 recv 一帧都切不出来）。
            if not advanced:
                break
        return frames

    # ------------------------------------------------------------------

    def _step(self) -> Tuple[bool, Optional[bytes]]:
        """走一步。返回 ``(这一步有没有推进状态机, 切出来的帧或 None)``。"""
        if self._state == self._SEEK:
            return self._seek()
        if self._state == self._HEADERS:
            return self._read_headers()
        return self._read_body()

    def _seek(self) -> Tuple[bool, Optional[bytes]]:
        idx = self._buf.find(self._sep)
        if idx < 0:
            # 尾巴要留着：分隔符可能正好被一次 recv 切成两半。
            keep = len(self._sep) + 4
            if len(self._buf) > self.max_buffer:
                self.overflows += 1
            if len(self._buf) > keep:
                del self._buf[:len(self._buf) - keep]
            return False, None

        del self._buf[:idx + len(self._sep)]
        # 分隔符后面跟 CRLF（正常分区）或 "--"（流的结尾），两种都吃掉。
        # 缓冲暂时不够就先不管：下面的 _read_headers 能容忍行首空行。
        self._eat_terminator()
        self._state = self._HEADERS
        return True, None

    def _eat_terminator(self) -> None:
        for tail in (b"\r\n", b"\n", b"--"):
            if self._buf.startswith(tail):
                del self._buf[:len(tail)]
                return

    def _read_headers(self) -> Tuple[bool, Optional[bytes]]:
        idx = self._buf.find(b"\r\n\r\n")
        width = 4
        if idx < 0:
            idx = self._buf.find(b"\n\n")
            width = 2
        if idx < 0:
            if len(self._buf) > MAX_HEADER:
                self.overflows += 1
                self._reset()
            return False, None

        head = bytes(self._buf[:idx])
        del self._buf[:idx + width]
        self._length = 0
        for line in head.replace(b"\r\n", b"\n").split(b"\n"):
            name, sep, value = line.partition(b":")
            if not sep:
                continue                       # 空行、或者不是头的一行
            if name.strip().lower() == b"content-length":
                try:
                    self._length = max(0, int(value.strip()))
                except ValueError:
                    self._length = 0
        self._state = self._BODY
        return True, None

    def _read_body(self) -> Tuple[bool, Optional[bytes]]:
        if self._length:
            if len(self._buf) < self._length:
                if len(self._buf) > self.max_buffer:
                    self.overflows += 1
                    self._reset()
                return False, None
            frame = bytes(self._buf[:self._length])
            del self._buf[:self._length]
        else:
            end = self._buf.find(EOI, 2)
            if end < 0:
                if len(self._buf) > self.max_buffer:
                    self.overflows += 1
                    self._reset()
                return False, None
            frame = bytes(self._buf[:end + 2])
            del self._buf[:end + 2]

        self._state = self._SEEK
        if not frame.startswith(SOI):
            # 长度和 JPEG 头对不上：这一帧不可信，丢掉并重新找边界。
            # 这一步**是推进**（字节已经被吃掉了），继续往下走。
            self.overflows += 1
            return True, None
        return True, frame

    def _reset(self) -> None:
        """缓冲超限：清空重来。当前这一帧注定丢了，但状态机得回到能对齐的地方。"""
        self._buf.clear()
        self._length = 0
        self._state = self._SEEK


def _header_value(head: bytes, name: bytes) -> str:
    for line in head.split(b"\r\n")[1:]:
        key, sep, value = line.partition(b":")
        if sep and key.strip().lower() == name:
            return value.strip().decode("latin-1", "replace")
    return ""


# ==========================================================================
# 客户端
# ==========================================================================

class MjpegClient:
    """到 A 侧展示流的一条连接，带自动重连。

    线程模型**逐条对齐** :class:`c_core.state_client.BackendLink`：
    一个 daemon 线程、指数退避、``stop()`` 先 ``shutdown(SHUT_RDWR)`` 再
    ``close()``（只 close 的话，阻塞在 ``recv`` 上的线程在 Windows 上不会
    醒过来）。

    回调（都在 socket 线程里执行）::

        on_frame(jpeg)              收到一帧完整的 JPEG
        on_link(source, connected)  连上 / 断开，供界面显示

    ``slot`` 是 ``on_frame`` 的语法糖：只给它、不给 ``on_frame`` 时，
    帧直接进 :class:`FrameSlot`。这是推荐用法，因为 GUI 那条铁律就是
    "``on_frame`` 里只许放进单槽"（见模块文档）。
    """

    def __init__(
        self,
        url: str,
        slot: Optional[FrameSlot] = None,
        on_frame: Optional[Callable[[bytes], None]] = None,
        on_link: Optional[Callable[[str, bool], None]] = None,
        reconnect_initial: float = 1.0,
        reconnect_max: float = 10.0,
        socket_timeout: float = 1.0,
    ) -> None:
        self.url = url
        self.host, self.port, self.path = parse_stream_url(url)
        self.slot = slot

        if on_frame is None and slot is not None:
            on_frame = slot.put
        self._on_frame = on_frame or _noop
        self._on_link = on_link or _noop

        self._reconnect_initial = reconnect_initial
        self._reconnect_max = reconnect_max
        self._socket_timeout = socket_timeout

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        self.stats: Dict[str, int] = {
            #: 交付给 on_frame 的帧数
            "frames": 0,
            #: 从网络上收到的总字节数（含分区头）
            "bytes": 0,
            #: 成功握手过多少次（含重连）
            "connections": 0,
            #: on_frame 抛异常被吞掉的次数。非零就该看一眼日志。
            "callback_errors": 0,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        """起线程。连接是异步建立的，本方法立刻返回。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._guarded, name="C-画面通道", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        """停止并等待线程收尾。幂等。"""
        self._stop.set()
        with self._lock:
            sock = self._sock
        if sock is not None:
            _shutdown_socket(sock)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _guarded(self) -> None:
        """包一层异常保护：子线程崩了要留日志，而不是静默死掉。"""
        try:
            self._loop()
        except Exception:
            logger.exception("画面通道线程异常退出")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        backoff = self._reconnect_initial
        while not self._stop.is_set():
            sock = self._connect()
            if sock is None:
                backoff = self._backoff_wait(backoff)
                if backoff is None:
                    return
                continue
            backoff = self._reconnect_initial     # 连上了，退避重置

            try:
                boundary, leftover = self._handshake(sock)
                self.stats["connections"] += 1
                self._set_connected(True)
                logger.info("已连上画面通道 %s（boundary=%s）", self.url, boundary)
                self._pump(sock, boundary, leftover)
            except socket.timeout:
                logger.info("画面通道等待数据超时（%.1fs），重连", self._socket_timeout)
            except StreamError as exc:
                # 协议层面的错（对面根本不是 MJPEG）：退避重连，因为对面
                # 可能只是还没起来 / 正在退出，下一次就好了。
                logger.info("画面通道协议错误：%s", exc)
            except OSError as exc:
                if not self._stop.is_set():
                    logger.info("画面通道中断：%s（稍后重连）", exc)
            finally:
                self._set_connected(False)
                self._drop(sock)

            if self._stop.is_set():
                return
            backoff = self._backoff_wait(backoff)
            if backoff is None:
                return

    def _connect(self) -> Optional[socket.socket]:
        """建立连接。失败只记日志并返回 None（由调用方退避重试）。"""
        if self._stop.is_set():
            return None
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5.0)
        except OSError as exc:
            logger.info("连接画面通道 %s 失败：%s（稍后重试）", self.url, exc)
            return None

        sock.settimeout(self._socket_timeout)   # 必须设超时，否则 stop() 唤醒不了它
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        with self._lock:
            self._sock = sock
        return sock

    def _handshake(self, sock: socket.socket) -> Tuple[str, bytes]:
        """发请求、读响应头，返回 ``(boundary, 已经多读进来的字节)``。

        返回值里为什么要带 leftover：``recv`` 不保证在响应头结束处停下，
        第一个分区很可能已经在同一个 chunk 里了。把它丢掉的话，表现是
        "画面要等到第二帧才出来" —— 一个不会报错、只会让人以为
        "摄像头有延迟"的现象。
        """
        sock.sendall(self._request().encode("ascii"))

        buffer = bytearray()
        while True:
            idx = buffer.find(b"\r\n\r\n")
            if idx >= 0:
                break
            chunk = sock.recv(RECV_SIZE)
            if not chunk:
                raise StreamError("连接在握手阶段就被关掉了")
            buffer += chunk
            if len(buffer) > MAX_HEADER:
                raise StreamError(f"HTTP 响应头超过 {MAX_HEADER} 字节，对面不是这条流")

        head = bytes(buffer[:idx])
        leftover = bytes(buffer[idx + 4:])

        status_line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        fields = status_line.split(" ", 2)
        try:
            code = int(fields[1])
        except (IndexError, ValueError):
            raise StreamError(f"看不懂的状态行：{status_line!r}") from None
        if code != 200:
            raise StreamError(f"展示流返回 {code}（{status_line.strip()}）")

        content_type = _header_value(head, b"content-type")
        if "multipart/x-mixed-replace" not in content_type.lower():
            raise StreamError(f"返回的不是 MJPEG（Content-Type={content_type!r}）")
        match = _BOUNDARY_RE.search(content_type)
        return (match.group(1) if match else BOUNDARY_FALLBACK), leftover

    def _pump(self, sock: socket.socket, boundary: str, leftover: bytes) -> None:
        """连着的时候一直读，逐帧交给 on_frame。返回即表示连接结束。"""
        parser = MultipartParser(boundary)
        for frame in parser.feed(leftover):
            self._deliver(frame)

        while not self._stop.is_set():
            try:
                chunk = sock.recv(RECV_SIZE)
            except socket.timeout:
                continue      # 暂时没数据，回去检查 stop 标志
            if not chunk:
                raise StreamError("展示流被对端关闭")
            self.stats["bytes"] += len(chunk)
            for frame in parser.feed(chunk):
                self._deliver(frame)

    def _deliver(self, frame: bytes) -> None:
        self.stats["frames"] += 1
        try:
            self._on_frame(frame)
        except Exception:  # noqa: BLE001 - 回调出错只该丢这一帧，不该断流
            self.stats["callback_errors"] += 1
            if self.stats["callback_errors"] == 1:
                logger.exception("画面帧回调出错（已丢弃该帧，不影响拉流）")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _request(self) -> str:
        return (
            f"GET {self.path} HTTP/1.0\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Accept: multipart/x-mixed-replace, image/jpeg\r\n"
            "User-Agent: frontend_C/1.0\r\n"
            "Connection: close\r\n"
            "\r\n"
        )

    def _set_connected(self, value: bool) -> None:
        if self._connected == value:
            return
        self._connected = value
        try:
            self._on_link(SOURCE_STREAM, value)
        except Exception:  # noqa: BLE001 - 同上：回调出事不许把拉流带走
            logger.exception("画面通道的连接状态回调出错")

    def _backoff_wait(self, current: float) -> Optional[float]:
        """退避等待，返回下一次该用的退避值；``None`` 表示收到停止信号。"""
        if self._stop.wait(current):
            return None
        return min(current * 2.0, self._reconnect_max)

    def _drop(self, sock: socket.socket) -> None:
        with self._lock:
            if self._sock is sock:
                self._sock = None
        _shutdown_socket(sock)


def _shutdown_socket(sock: socket.socket) -> None:
    """先 ``shutdown`` 再 ``close``。

    只 close 的话，另一个线程可能正阻塞在 ``recv`` 上，而 ``recv`` 不会
    因为本地 close 就返回（Windows 上尤其明显）。``shutdown(SHUT_RDWR)``
    会让对端收到 FIN，阻塞中的 recv 立刻返回 0。
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


__all__ = [
    "BOUNDARY_FALLBACK",
    "DEFAULT_CAMERA_FPS",
    "DEFAULT_PATH",
    "DEFAULT_STREAM_PORT",
    "MAX_CAMERA_FPS",
    "MIN_CAMERA_FPS",
    "SOURCE_STREAM",
    "FrameSlot",
    "MjpegClient",
    "MultipartParser",
    "StreamError",
    "clamp_fps",
    "parse_stream_url",
]
