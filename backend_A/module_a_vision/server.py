"""模块 A · TCP 8000 服务端（A 是服务端，B 是客户端）。

帧格式沿用《系统总接口文档》§2.1：一行一条 JSON，末尾 ``\\n``。
A 用**同步** socket 而不是 asyncio——这不是偷懒：采集循环本身是
阻塞的（``read_features()`` 会按真实帧率 ``sleep``），推理是 CPU 密集的，
把它塞进 asyncio 只会让事件循环被同步推理卡住。

线程分工只有两条：

* **接受线程**：``accept()`` 新连接，登记进客户端表。
* **生产线程（主线程）**：读帧 → 推理 → 聚合 → 到点广播。
  外加一个心跳线程，周期发 ``heartbeat``。

**两种出站格式，由 ``emit_mode`` 选**（见 :mod:`module_a_vision.wire`）：

``v2``（本模块原生）
    三条报文：

    ===========================  ==============================================
    ``vision_window``            每 10 秒一个窗口（正常路径）
    ``vision_unusable``          看得见设备，但画面不可用（遮挡/全黑/过暗）
    ``heartbeat``                进程存活证明，**不参与 L4 判定**
    ===========================  ==============================================

    ``heartbeat`` 与 ``vision_window`` 必须分开，这一点在 B 侧同样被强调：
    心跳回答"进程还活着吗"，窗口回答"我们还能看见老人吗"。一个摄像头被
    毛巾盖住的系统心跳一切正常，如果让它重置 L4 计时器，这种失效**永远
    不会告警**——而它恰恰是最需要告警的一种。

``v1``（团队仓库默认）
    api_doc §3.2 的平铺 8 字段，**逐帧一条**。

    ⚠️ **``v1`` 模式下不发 ``heartbeat``，也不发 ``vision_window``。**
    B 的 ``VisionSample.from_payload`` 会把任何一条缺字段的报文兜成一个
    ``has_face=false`` 的样本**推进判定器** —— 心跳会被当成"看不见老人"。
    B 不需要心跳：A 死掉由它那个 5 秒失联判据兜住。这是"一种模式一种报文"
    的必然结论，不是省事。

**隐私**：每一份出站报文在 ``send`` 之前都要过
:func:`~module_a_vision.privacy.guard.assert_clean`。这不是可选的：
它是"图像不出模块 A"这条红线在代码里的最后一道闸，而**闸门装在
出口而不是入口**，意味着将来无论谁新增了什么字段，都必须先过这一关。

**契约**：``v1`` 模式的报文还要过 :func:`~module_a_vision.wire.assert_v1_contract`，
和隐私闸并排装在同一个出口。理由一样，但针对的是另一种失效：B 对缺失字段
**静默回退默认值**，字段名拼错的表现不是崩溃，而是"状态永远停在 absent"。
契约闸把这个静默失效变成出口处一次响亮的拒绝。
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from typing import Any

from shared.framing import bind_listen_with_retry, encode
from shared.schema import WindowState

from .aggregate.window import AggregatorConfig, WindowAggregator
from .capture.base import CaptureConfig, is_feature_source, is_frame_source
from .privacy.guard import PrivacyViolationError, assert_clean
from .wire import assert_v1_contract, to_v1_sample

#: 默认监听地址与端口（《系统总接口文档》§2.2：8000）。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: 心跳周期（秒）。取 30 是刻意的：远小于 B 的断流阈值（600 秒），
#: 因此"心跳停了"总能先于"判定断流"被观察到，便于排查。
HEARTBEAT_SEC = 30.0

#: 报文类型。
MSG_WINDOW = "vision_window"
MSG_UNUSABLE = "vision_unusable"
MSG_HEARTBEAT = "heartbeat"

#: 出站格式。``v2`` 是本模块原生的 10 秒嵌套窗口；``v1`` 是 api_doc §3.2
#: 的逐帧平铺 8 字段。默认 ``v1`` —— 团队仓库的消费方是 ``backend_B``，
#: 它认的是 §3.2。
EMIT_V1 = "v1"
EMIT_V2 = "v2"
EMIT_MODES = (EMIT_V1, EMIT_V2)


class VisionServer:
    """把采集源、人脸后端、聚合器串起来，并把窗口推给 B。

    :param source: 采集源：帧源（有 ``read()``）或特征源（有 ``read_features()``）。
    :param backend: 人脸后端。帧源必传；特征源可以不传。
    :param aggregator: 窗口聚合器。``None`` 时按 ``elder_id`` / ``device_id`` 新建。
    :param heartbeat_sec: 心跳周期；``0`` 关闭心跳（测试用）。**``v1`` 模式下
        心跳被无条件关闭**，见模块文档。
    :param emit_mode: ``"v1"``（默认，api_doc §3.2 逐帧平铺）或 ``"v2"``
        （原生 10 秒嵌套窗口）。
    """

    def __init__(
        self,
        source: Any,
        backend: Any = None,
        aggregator: WindowAggregator | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        heartbeat_sec: float = HEARTBEAT_SEC,
        max_clients: int = 4,
        emit_mode: str = EMIT_V1,
    ) -> None:
        if emit_mode not in EMIT_MODES:
            raise ValueError(
                f"未知出站格式 {emit_mode!r}。可选：{' / '.join(EMIT_MODES)}"
            )

        self.source = source
        self.backend = backend
        self.host = host
        self.port = port
        self.heartbeat_sec = heartbeat_sec
        self.max_clients = max_clients
        self.emit_mode = emit_mode

        self.aggregator = aggregator or WindowAggregator()

        self._sock: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        #: 运行统计。``windows``（v2）与 ``v1_frames``（v1）是验收时最常看的数。
        self.stats = {
            "frames": 0,
            "windows": 0,
            "unusable_windows": 0,
            "clients_accepted": 0,
            "send_failures": 0,
            "privacy_blocked": 0,
            "v1_frames": 0,
            #: 被契约闸拦下的报文数。**非零就是 bug**，不是可调参数。
            "v1_contract_blocked": 0,
        }

    # ------------------------------------------------------------ 生命周期

    def serve(self) -> socket.socket:
        """绑定并开始接受连接。**非阻塞**，返回后由调用方跑 :meth:`run`。"""
        sock = bind_listen_with_retry(self.host, self.port)
        self._sock = sock

        # 记下**真实**绑定的端口。传 port=0 时操作系统会挑一个空闲端口，
        # 这时 self.port 还是 0，横幅会打印 "127.0.0.1:0" —— 一个看起来
        # 像"没起来"的地址。测试与自动化脚本也要靠这个值去连。
        self.port = sock.getsockname()[1]

        acceptor = threading.Thread(target=self._accept_loop, name="a-accept", daemon=True)
        acceptor.start()
        self._threads.append(acceptor)

        # v1 模式**不发心跳**：B 会把任何一条缺 8 字段的报文兜成一个
        # has_face=false 的样本推进判定器，等于周期性地伪造"看不见老人"。
        # 详见模块文档。
        if self.heartbeat_sec > 0 and self.emit_mode != EMIT_V1:
            beater = threading.Thread(target=self._heartbeat_loop, name="a-heartbeat", daemon=True)
            beater.start()
            self._threads.append(beater)

        print(
            f"[A] 视觉服务已监听 {self.host}:{self.port}"
            f"（源：{_describe(self.source)}  出站：{self.emit_mode}）"
        )
        return sock

    def run(self) -> None:
        """生产循环。阻塞直到源耗尽或被 :meth:`stop`。

        源耗尽**不等于**进程该退出：B 可能还在跑，而"看不见了"本身
        就是一条需要上报的状态。所以耗尽后不立刻 close，而是等
        ``stop()``——真到那时，B 的 L4 计时器会把这件事报出来。
        """
        self.source.open()
        try:
            self._produce()
        finally:
            self.source.close()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for client in self._clients:
                with contextlib.suppress(OSError):
                    client.close()
            self._clients.clear()
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None

    def close(self) -> None:
        self.stop()

    # ------------------------------------------------------------ 生产

    def _produce(self) -> None:
        while not self._stop.is_set():
            frame = self._read_one()
            if frame is None:
                # 源耗尽（视频播完 / 合成剧本走完 / 已达帧上限）。
                print(f"[A] 采集源已耗尽（共 {self.stats['frames']} 帧）")
                break

            self.stats["frames"] += 1

            # 摄入**必须**排在投影前面：本帧若恰好完成一次眨眼，
            # 累计计数器是在 add_frame 里自增的。反过来会永远晚一帧。
            self.aggregator.add_frame(frame)

            if self.emit_mode == EMIT_V1:
                self._emit_v1(frame)
            else:
                # 关窗判定用**帧自己的时间戳**，不用墙钟——合成源与 CSV 回放
                # 的时间轴从 0 开始，与墙钟无关。
                if self.aggregator.should_close(frame.ts):
                    self._emit(self.aggregator.close_window(frame.ts))

    def _read_one(self):
        """从任意种类的源读一帧特征。"""
        if is_feature_source(self.source):
            return self.source.read_features()

        if is_frame_source(self.source):
            pixels = self.source.read()
            if pixels is None:
                return None
            if self.backend is None:
                raise RuntimeError(
                    "给了帧源却没有给人脸后端：请传 backend=MediaPipeFaceBackend(...) "
                    "或 backend=SyntheticFaceBackend(...)。"
                )
            return self.backend.process(pixels)

        raise RuntimeError(
            "采集源既不是帧源也不是特征源：它既没有 read() 也没有 read_features()。"
        )

    def _emit_v1(self, frame: Any) -> None:
        """把一帧投影成 api_doc §3.2 报文并广播。"""
        payload = to_v1_sample(
            frame,
            blink_total=self.aggregator.blink_total,
            emotion=self.aggregator.classify_frame(frame),
        )
        sent = self.broadcast(payload)
        if sent:
            self.stats["v1_frames"] += 1

        # **缓冲必须照常排空。** v1 不消费窗口，可 `add_frame` 每一帧都往
        # `_buffer.frames` 里追加：不排空它就是一个只涨不落的内存泄漏，
        # 同时 `_stable` 与 `_attention` 这些跨窗口状态也再不会推进。
        #
        # 聚合结果在这里**丢弃**。这不是漏掉了什么，而是"v1 模式下窗口
        # 根本不是出站格式"的直接后果 —— 所以 `stats["windows"]` 在 v1
        # 模式下会一直是 0，而 `frames` 照常增长。看到这个组合不要以为是坏了。
        if self.aggregator.should_close(frame.ts):
            self.aggregator.close_window(frame.ts)

    def _emit(self, window: WindowState) -> None:
        """广播一个窗口。"""
        self.stats["windows"] += 1
        payload = window.to_dict()

        if not window.quality.usable:
            self.stats["unusable_windows"] += 1
            # 画面不可用**照发窗口**，让 B 知道"这一刻我看不清"；
            # 另外补一条 vision_unusable 让 B 走画面侧的 L4 路径。
            # 两者都要：前者维持时序，后者触发告警语义。
            self.broadcast({**payload, "type": MSG_UNUSABLE})
        else:
            self.broadcast({**payload, "type": MSG_WINDOW})

        self._log_window(window)

    def _log_window(self, window: WindowState) -> None:
        """窗口级别的日志。只打印一行摘要——10 秒一条，不该刷屏。"""
        quality = "可用" if window.quality.usable else "不可用"
        print(
            f"[A] 窗口 #{self.stats['windows']:>4}  "
            f"帧 {window.window.frames_valid}/{window.window.frames_total}  "
            f"质量 {quality}  提示 {describe_hints(window)}"
        )

    # ------------------------------------------------------------ 广播

    def broadcast(self, message: dict[str, Any]) -> int:
        """把一条报文发给所有已连接的 B。返回成功发送的份数。

        **每一份出站报文都在这里过两道闸。** 放在这一个函数里，是为了
        让"有没有绕过去"成为一个可以一眼回答的问题：只有这一个出口。

        1. **隐私闸** —— 图像不出模块 A。
        2. **契约闸**（仅 v1）—— 报文必须严格是 §3.2 的 8 个字段。
           针对的失效不一样：B 对缺字段静默回退默认值，字段拼错不报错、
           不崩溃，只表现为"状态永远停在 absent"。这道闸把那个静默失效
           挪到出口，变成一次响亮的拒绝。
        """
        try:
            assert_clean(message)
        except PrivacyViolationError as exc:
            self.stats["privacy_blocked"] += 1
            # 绝不"改一改再发"——那会掩盖调用方的错误。
            # 响亮地失败，让写错字段的人立刻知道。
            print(f"[A] ✗ 出站报文被隐私闸拦下，已丢弃：\n{exc}")
            return 0

        if self.emit_mode == EMIT_V1:
            problems = assert_v1_contract(message)
            if problems:
                self.stats["v1_contract_blocked"] += 1
                # 同样绝不"补两个字段再发"：那会让对端收到一份看起来
                # 正常、实际有一个字段是瞎填的报文，比直接不发更糟。
                print(
                    "[A] ✗ v1 报文不符合 api_doc §3.2，已丢弃：\n  "
                    + "\n  ".join(problems)
                )
                return 0

        data = encode(message)
        sent = 0
        dead: list[socket.socket] = []

        with self._lock:
            clients = list(self._clients)

        for client in clients:
            try:
                client.sendall(data)
                sent += 1
            except OSError:
                dead.append(client)

        if dead:
            with self._lock:
                for client in dead:
                    if client in self._clients:
                        self._clients.remove(client)
                    with contextlib.suppress(OSError):
                        client.close()
            self.stats["send_failures"] += len(dead)
            if sent == 0 and clients:
                print("[A] 所有 B 连接均已断开，等待重连")

        return sent

    # ------------------------------------------------------------ 连接

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            sock = self._sock
            if sock is None:
                break
            try:
                client, addr = sock.accept()
            except OSError:
                break  # socket 被 close() 关掉，正常退出

            with self._lock:
                if len(self._clients) >= self.max_clients:
                    # 不无限收：B 反复重启而不清理连接时，
                    # 无上限会让 A 的文件描述符被吃光。
                    with contextlib.suppress(OSError):
                        client.close()
                    print(f"[A] 连接数已达上限 {self.max_clients}，拒绝 {addr}")
                    continue
                self._clients.append(client)
                self.stats["clients_accepted"] += 1

            print(f"[A] B 已连接 {addr[0]}:{addr[1]}")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_sec):
            if not self._clients:
                continue
            self.broadcast(
                {
                    "type": MSG_HEARTBEAT,
                    "ts": time.time(),
                    "device_id": self.aggregator.cfg.device_id,
                    "windows": self.stats["windows"],
                }
            )


# ================================================================ 辅助

def describe_hints(window: WindowState) -> str:
    """把 A 的本地初判压成一行。

    措辞用"疑似"而不是断言——A 的初判**只是建议**，B 可以推翻它。
    日志里也不该写得像已经确诊了：这套系统最不该做的事，就是让读到
    日志的人以为它做了医学判断。
    """
    h = window.event_hints
    bits = []
    if h.suspected_low_mood:
        bits.append("疑似情绪低")
    if h.suspected_drowsy:
        bits.append("疑似困倦")
    if h.suspected_body_abnormal:
        bits.append("疑似身体异常")
    if h.consecutive_abnormal_windows:
        bits.append(f"连续异常 {h.consecutive_abnormal_windows} 窗口")
    return "、".join(bits) if bits else "无"


def _describe(source: Any) -> str:
    describe = getattr(source, "describe", None)
    if callable(describe):
        with contextlib.suppress(Exception):
            return str(describe())
    return type(source).__name__


def build_server(
    source_kind: str = "synthetic",
    scenario: str = "normal",
    video: str | None = None,
    csv: str | None = None,
    camera_index: int = 0,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    elder_id: str = "E1001",
    device_id: str = "cam-livingroom-01",
    no_face_backend: bool = False,
    real_time: bool = True,
    emit_mode: str = EMIT_V1,
    **kwargs: Any,
) -> VisionServer:
    """按种类装配一个服务端。供 ``main.py`` 与 ``tools/`` 使用。

    ``source_kind`` 取 ``synthetic`` / ``video`` / ``csv`` / ``camera``。
    真实人脸后端只在 ``camera`` 与 ``video`` 下才需要——合成源与 CSV
    直接产出特征，不经过像素。

    :param real_time: 是否按真实节奏产出帧。**实时演示必须为 ``True``**——
    否则 180 秒的剧本会在几十毫秒内全部走完，B 侧看到的是一串
    时间戳在瞬间跳完的窗口，冷却与持续性判定全部失去意义。
    离线验证数据契约时设 ``False``。
    :param emit_mode: ``"v1"``（默认）或 ``"v2"``，见 :class:`VisionServer`。
    """
    cfg = CaptureConfig(real_time=real_time, **kwargs)

    if source_kind == "synthetic":
        from .capture.synthetic import SyntheticFeatureSource

        return VisionServer(
            source=SyntheticFeatureSource(scenario=scenario, cfg=cfg),
            aggregator=WindowAggregator(AggregatorConfig(elder_id=elder_id, device_id=device_id)),
            host=host,
            port=port,
            emit_mode=emit_mode,
        )

    if source_kind == "csv":
        if not csv:
            raise ValueError("csv 源需要 --csv 指定文件路径")
        from .capture.csv_replay import CsvReplaySource

        return VisionServer(
            source=CsvReplaySource(csv, cfg=cfg),
            aggregator=WindowAggregator(AggregatorConfig(elder_id=elder_id, device_id=device_id)),
            host=host,
            port=port,
            emit_mode=emit_mode,
        )

    # camera 与 video 都要走像素 → 人脸后端。
    backend = None
    if not no_face_backend:
        from .face.mediapipe_backend import MediaPipeFaceBackend

        backend = MediaPipeFaceBackend()

    if source_kind == "camera":
        from .capture.camera import CameraSource

        source = CameraSource(index=camera_index, cfg=cfg)
    elif source_kind == "video":
        if not video:
            raise ValueError("video 源需要 --video 指定文件路径")
        from .capture.video_file import VideoFileSource

        source = VideoFileSource(video, cfg=cfg)
    else:
        raise ValueError(
            f"未知采集源 {source_kind!r}。可选：synthetic / video / csv / camera"
        )

    return VisionServer(
        source=source,
        backend=backend,
        aggregator=WindowAggregator(AggregatorConfig(elder_id=elder_id, device_id=device_id)),
        host=host,
        port=port,
        emit_mode=emit_mode,
    )


__all__ = [
    "VisionServer",
    "build_server",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "HEARTBEAT_SEC",
    "EMIT_V1",
    "EMIT_V2",
    "EMIT_MODES",
    "MSG_WINDOW",
    "MSG_UNUSABLE",
    "MSG_HEARTBEAT",
    "describe_hints",
]
