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
    api_doc §3.2 的平铺 8 字段，**逐帧一条**；外加 §3.5 的**类型化报文**
    （目前是 ``rppg`` 体征，见 :mod:`module_a_vision.vitals`）。

    ⚠️ 类型化报文是**额外**发的，帧报文**仍然不带 ``type``** ——
    理由见 :mod:`module_a_vision.wire` 的模块文档（一句话：带上了就会
    撞上 §3.2 的键集合精确相等校验）。

    ⚠️ **``v1`` 模式下不发 ``heartbeat``，也不发 ``vision_window``。**
    B 的 ``VisionSample.from_payload`` 会把任何一条缺字段的报文兜成一个
    ``has_face=false`` 的样本**推进判定器** —— 心跳会被当成"看不见老人"。
    B 不需要心跳：A 死掉由它那个 5 秒失联判据兜住。这是"一种模式一种报文"
    的必然结论，不是省事。

**隐私**：每一份出站报文在 ``send`` 之前都要过
:func:`~module_a_vision.privacy.guard.assert_clean`。这不是可选的：
它是"图像不出模块 A"这条红线在代码里的最后一道闸，而**闸门装在
出口而不是入口**，意味着将来无论谁新增了什么字段，都必须先过这一关。

**契约**：``v1`` 模式的报文还要过 :func:`~module_a_vision.wire.check_contract`，
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
from .privacy.guard import PrivacyViolationError, assert_clean, scrub_frame
from .wire import KIND_FRAME, check_contract, to_focus_payload, to_v1_sample

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
    :param vitals: 体征通道（:class:`~module_a_vision.vitals.RppgMonitor`）。
        给了它，v1 模式就会在帧报文之外**另发** §3.5 的 ``rppg`` 报文。
        ``None`` 表示这条通道关闭 —— 例如 CSV 回放（没有 RGB 也没有像素）。
    :param focus: 是否发 §3.5 的 ``focus`` 视线报文（**逐帧**）。
        默认开。它不需要任何额外数据源——视线就在每一帧的
        ``gaze_off_ratio`` 里——所以没有"挂不上"的情况。
        关掉它会让 B 的 VAI 校准**永远做不完且不报错**，只应在排查
        "B 到底有没有被类型化报文影响"时临时关掉。
    :param stream: 展示流的帧缓存（:class:`~module_a_vision.stream.FrameHub`）。
        ``None``（默认）表示不开展示流 —— 这时 ``_publish_frame`` 整个是
        空转，连一次 ``scrub_frame`` 都不会发生。**展示流走 HTTP、只绑
        回环，与 8000 那条报文链路互不重叠**（见 :mod:`module_a_vision.stream`
        开头的边界说明）。
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
        vitals: Any = None,
        focus: bool = True,
        stream: Any = None,
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
        self.vitals = vitals
        self.focus = focus
        self.stream = stream

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
            #: 已发出的体征报文数（§3.5）。**0 就说明体征通道是死的** ——
            #: 演示时它应该以 1Hz 稳定增长。
            "rppg_packets": 0,
            #: 已发出的视线报文数（§3.5）。它与帧**同频**，所以
            #: ``focus_packets`` 与 ``v1_frames`` 在 ``focus=True`` 时应当
            #: **完全相等**（两者都只在真发出去之后才自增）。对不上就是
            #: 有帧没走到 :meth:`_emit_focus` —— 那是一个静默故障，
            #: 靠这两个数相减才看得见。
            "focus_packets": 0,
            #: 被契约闸拦下的**帧**报文数。**非零就是 bug**，不是可调参数。
            #:
            #: ⚠️ 引入 §3.5 类型化报文后，这个计数器**只统计帧报文**。
            #: 别把类型化报文的违规也记进来：那样它就同时表达两种
            #: 完全不同的故障（"帧字段写错了"与"体征数值越界"），
            #: 而"非零就是 bug"这句判断正是靠语义单一才成立的。
            "v1_contract_blocked": 0,
            #: 被契约闸拦下的**类型化**报文数（rppg / focus）。
            "typed_contract_blocked": 0,
            #: 已就地清零的帧数（"用完即毁"真的发生的次数）。
            #: 只在 ``stream`` 开着时增长 —— 没有展示流就没人经手像素，
            #: 也就没有"用完"这一说。
            "frames_scrubbed": 0,
            #: 清不掉的帧数（缓冲只读）。**应当恒为 0**；非零说明某个采集源
            #: 在拿只读缓冲，那时"用完即毁"是失效的（见 :meth:`_scrub`）。
            "scrub_skipped": 0,
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
        if self.emit_mode == EMIT_V1:
            # 体征通道**单独打一行**，而且把"这是自检不是测量"写在上面：
            # 它平时只在日志里出现一次，却是最容易被截图当结论的一行。
            print(f"[A] 体征：{_describe(self.vitals) if self.vitals else '关闭'}")
            # 视线通道单独一行，并且**把速率写出来**：它是 B 侧 VAI 能否
            # 出指数的前提，"关掉了"这句话必须当场可见，而不是等半小时后
            # 从 B 的日志里反推。
            print(f"[A] 视线：{'逐帧（§3.5 focus）' if self.focus else '关闭'}")
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
            step = self._read_one()
            if step is None:
                # 源耗尽（视频播完 / 合成剧本走完 / 已达帧上限）。
                print(f"[A] 采集源已耗尽（共 {self.stats['frames']} 帧）")
                break
            frame, pixels = step

            self.stats["frames"] += 1

            # 摄入**必须**排在投影前面：本帧若恰好完成一次眨眼，
            # 累计计数器是在 add_frame 里自增的。反过来会永远晚一帧。
            self.aggregator.add_frame(frame)

            # 展示流排在**推理之后**：叠加要画 face_bbox，它来自 frame。
            # 反过来框会比画面晚一帧，头一转就看得出来。
            self._publish_frame(frame, pixels)

            if self.emit_mode == EMIT_V1:
                self._emit_v1(frame)
            else:
                # 关窗判定用**帧自己的时间戳**，不用墙钟——合成源与 CSV 回放
                # 的时间轴从 0 开始，与墙钟无关。
                if self.aggregator.should_close(frame.ts):
                    self._emit(self.aggregator.close_window(frame.ts))

    def _read_one(self) -> tuple[Any, Any] | None:
        """从任意种类的源读一帧，返回 ``(特征, 像素)``；``None`` 表示已耗尽。

        ``pixels is None`` 表示这个源根本没有像素（特征源、CSV 回放）——
        展示流那边于是什么都不做，这是正常情况而不是缺数据。

        ⚠️ **像素的生命周期就是这个返回值。** 刻意不挂
        ``self._last_pixels``：这个实例是长驻的，一旦挂上去，"A 模块内不留
        像素"就从一条**事实**降级成一条**纪律**。调用方（``_produce``）
        拿到之后马上编码，随即 :meth:`_scrub` 当场清零。
        """
        if is_feature_source(self.source):
            return self.source.read_features(), None

        if is_frame_source(self.source):
            pixels = self.source.read()
            if pixels is None:
                return None
            if self.backend is None:
                raise RuntimeError(
                    "给了帧源却没有给人脸后端：请传 backend=MediaPipeFaceBackend(...) "
                    "或 backend=SyntheticFaceBackend(...)。"
                )
            # **先 process 再返回** —— 叠加要画 face_bbox，而它来自这一步。
            return self.backend.process(pixels), pixels

        raise RuntimeError(
            "采集源既不是帧源也不是特征源：它既没有 read() 也没有 read_features()。"
        )

    # ------------------------------------------------------------ 展示流

    def _publish_frame(self, frame: Any, pixels: Any) -> None:
        """把这一帧交给展示流，然后**当场把像素清零**。

        三步的顺序换不得：画叠加 + 编 JPEG（没人看时它一次 cv2 都不调）→
        ``scrub_frame`` 就地清零 → 参数出栈，这一帧的像素从此没有引用者。
        "用完即毁"到这一步才第一次真的发生（此前 ``scrub_frame`` 全仓
        没有任何调用点）。
        """
        hub = self.stream
        if hub is None or pixels is None:
            return
        hub.offer(pixels, frame)
        self._scrub(pixels)

    def _scrub(self, pixels: Any) -> None:
        """就地清零一帧。**清不掉也绝不能把视觉主循环带走。**

        只读缓冲是清不掉的，而 ``scrub_frame`` 对这种情况**故意**抛
        ``ValueError``（"以为清零了其实没有"比报错危险得多）。这里接住它，
        理由是这个循环比"这一帧有没有被清干净"重要得多：像素在这之后已经
        没有别的引用者，清零只是多加一层保险。**但不装作没发生** ——
        第一次会打一行，计数也留着。
        """
        try:
            scrub_frame(pixels)
            self.stats["frames_scrubbed"] += 1
        except (TypeError, ValueError) as exc:
            self.stats["scrub_skipped"] += 1
            if self.stats["scrub_skipped"] == 1:
                print(f"[A] ⚠ 有帧无法就地清零，已跳过（不影响采集）：{exc}")

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

        self._emit_vitals(frame)
        self._emit_focus(frame)

        # **缓冲必须照常排空。** v1 不消费窗口，可 `add_frame` 每一帧都往
        # `_buffer.frames` 里追加：不排空它就是一个只涨不落的内存泄漏，
        # 同时 `_stable` 与 `_attention` 这些跨窗口状态也再不会推进。
        #
        # 聚合结果在这里**丢弃**。这不是漏掉了什么，而是"v1 模式下窗口
        # 根本不是出站格式"的直接后果 —— 所以 `stats["windows"]` 在 v1
        # 模式下会一直是 0，而 `frames` 照常增长。看到这个组合不要以为是坏了。
        if self.aggregator.should_close(frame.ts):
            self.aggregator.close_window(frame.ts)

    def _emit_vitals(self, frame: Any) -> None:
        """体征通道（api_doc §3.5 的 ``rppg``）。

        **必须挂在逐帧路径上，而且必须与帧共用一个时间轴。**
        :class:`~module_a_vision.vitals.RppgMonitor` 内部按**帧时间戳**
        决定发不发、并按帧时间戳做 FFT，所以喂给它的 ``ts`` 只能是
        ``frame.ts``，不能是墙钟 —— 合成源与 CSV 回放的时间轴都从 0 开始，
        混进墙钟会让 ``Rppg`` 算出一个跨越两套时钟的帧率，然后
        **一直**走置灰分支（``fs < 5.0``），表现是"体征一直出不来"。

        ⚠️ **这里的 ``self.vitals is None`` 只关掉自己。** 原先两条通道
        共用 :meth:`_emit_side_channels`、由 ``vitals is None`` 一并早退；
        视线通道挂进来之后那个写法就成了一个静默陷阱：CSV 回放与 camera
        源都没有体征（``build_server`` 只给合成源挂），于是**每帧都被那道
        早退挡在视线通道之外**，表现是"VAI 一直不工作"，而"体征没开"
        看起来完全正常。两条通道各自守各自的门，谁也不替谁决定。
        """
        if self.vitals is None:
            return
        payload = self.vitals.feed(frame.ts)
        if payload is None:
            return
        if self.broadcast(payload):
            self.stats["rppg_packets"] += 1

    def _emit_focus(self, frame: Any) -> None:
        """视线通道（api_doc §3.5 的 ``focus``）。**逐帧，与帧报文同频。**

        速率与判据都写在 :func:`~module_a_vision.wire.to_focus_payload`：
        掉到 1Hz 以下会让 B 的 ``PassiveCalibrator`` 每次 ``update()`` 都
        ``reset()``，"校准永远做不完"且不报错。所以这里**不做任何抽稀**
        （不要因为"每帧都发是不是太密"给它加个 ``every`` 计数器）。

        时间戳同样取 ``frame.ts`` —— 这条流的两个消费者（校准用的头姿窗口
        与视线窗口）都会跨帧做差，混进墙钟等于让一个窗口横跨两套时钟。
        """
        if not self.focus:
            return
        if self.broadcast(to_focus_payload(frame)):
            self.stats["focus_packets"] += 1

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
        2. **契约闸**（仅 v1）—— 报文必须严格是 §3.2 的 8 个字段
           （帧），或 §3.5 的类型化形状（``rppg`` / ``focus``）。
           针对的失效不一样：B 对缺字段静默回退默认值，字段拼错不报错、
           不崩溃，只表现为"状态永远停在 absent"。这道闸把那个静默失效
           挪到出口，变成一次响亮的拒绝。

        分流由 :func:`~module_a_vision.wire.check_contract` 做，**和
        ``--dry-run`` 用的是同一个函数**——两者一旦各写一遍，
        那个"最快的排查工具"就会开始输出与实跑不符的"一切正常"。

        :raises wire.UnknownMessageTypeError: v1 模式下收到一个带 ``type``
            却不在 §3.5 白名单里的报文。**刻意让它抛出去**：兜底成
            "当帧处理"会让它撞上"多余字段"从而在这里被静默丢弃，
            故障于是表现成"某个功能一直不工作"，而不是一次启动后
            几秒内就能看到的失败。
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
            kind, problems = check_contract(message)
            if problems:
                # 两类报文分开计数：帧违规与体征/视线违规是两种完全不同的
                # 故障，混进一个计数器会让"非零就是 bug"这句判断失去意义。
                self.stats[
                    "v1_contract_blocked" if kind == KIND_FRAME
                    else "typed_contract_blocked"
                ] += 1
                # 同样绝不"补两个字段再发"：那会让对端收到一份看起来
                # 正常、实际有一个字段是瞎填的报文，比直接不发更糟。
                print(
                    f"[A] ✗ {kind} 报文不符合 api_doc，已丢弃：\n  "
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
    vitals: bool = True,
    focus: bool = True,
    stream: Any = None,
    **kwargs: Any,
) -> VisionServer:
    """按种类装配一个服务端。供 ``main.py`` 与 ``tools/`` 使用。

    ``source_kind`` 取 ``synthetic`` / ``synthetic-pixels`` / ``video`` /
    ``csv`` / ``camera``。
    真实人脸后端只在 ``camera`` 与 ``video`` 下才需要——合成源与 CSV
    直接产出特征，不经过像素。

    :param real_time: 是否按真实节奏产出帧。**实时演示必须为 ``True``**——
    否则 180 秒的剧本会在几十毫秒内全部走完，B 侧看到的是一串
    时间戳在瞬间跳完的窗口，冷却与持续性判定全部失去意义。
    离线验证数据契约时设 ``False``。
    :param emit_mode: ``"v1"``（默认）或 ``"v2"``，见 :class:`VisionServer`。
    :param vitals: 是否挂 §3.5 的体征通道。**只有合成源挂得上** ——
        它是目前唯一能给出 RGB 的源（合成体征直出 ``(t, R, G, B)``）。
        CSV 回放既没有 RGB 列也没有像素；camera / video 的真像素取色
        本步尚未实现。传 ``False`` 可以要一条只发帧的干净流（排查用）。
    :param focus: 是否发 §3.5 的视线报文。**所有源都发得起**（视线来自
        每一帧自己的 ``gaze_off_ratio``，不需要额外数据源），所以默认开，
        而且**与 ``vitals`` 无关** —— 想得到"只含帧报文"的干净流，
        两个都要传 ``False``。
    :param stream: 展示流的帧缓存（:class:`~module_a_vision.stream.FrameHub`），
        透传给 :class:`VisionServer`。``None`` = 不开展示流。
    """
    cfg = CaptureConfig(real_time=real_time, **kwargs)

    if source_kind == "synthetic":
        from .capture.synthetic import SyntheticFeatureSource
        from .vitals import RppgMonitor, SyntheticVitalsSource

        return VisionServer(
            source=SyntheticFeatureSource(scenario=scenario, cfg=cfg),
            aggregator=WindowAggregator(AggregatorConfig(elder_id=elder_id, device_id=device_id)),
            host=host,
            port=port,
            emit_mode=emit_mode,
            vitals=RppgMonitor(SyntheticVitalsSource()) if vitals else None,
            focus=focus,
            stream=stream,
        )

    if source_kind == "synthetic-pixels":
        # **本机没有摄像头时的展示流数据源。** 与 synthetic 的区别只有一条：
        # 它多给一路玩具像素（见 capture/synthetic.py 的类文档）。
        #
        # ⚠️ ``backend=source`` —— 它**自己是自己的人脸后端**。这不是偷懒：
        # 像素与特征必须来自同一次剧本推进，分两次读会让时间轴走快一倍。
        # 也因此它**不能**配 MediaPipeFaceBackend（那会去检一张合成脸，
        # 而这里画的本来就不是人脸），更不能配 SyntheticFaceBackend
        # （它收到 ndarray 直接报错）。
        from .capture.synthetic import SyntheticPixelSource

        pixel_source = SyntheticPixelSource(scenario=scenario, cfg=cfg)
        return VisionServer(
            source=pixel_source,
            backend=pixel_source,
            aggregator=WindowAggregator(AggregatorConfig(elder_id=elder_id, device_id=device_id)),
            host=host,
            port=port,
            emit_mode=emit_mode,
            # 体征要 RGB，而这条路径**没有** RGB（玩具图是 BGR 三通道但
            # 与体征无关）。宁可不发，也不发一条从玩具图里"取色"出来的假体征。
            focus=focus,
            stream=stream,
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
            # CSV 回放没有体征（没有 RGB 列也没有像素），**但有视线**：
            # 若 CSV 带 ``gaze_off_ratio`` 列，这一路就是真实的视线数据。
            focus=focus,
            stream=stream,
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
            f"未知采集源 {source_kind!r}。可选：synthetic / synthetic-pixels / "
            "video / csv / camera"
        )

    return VisionServer(
        source=source,
        backend=backend,
        aggregator=WindowAggregator(AggregatorConfig(elder_id=elder_id, device_id=device_id)),
        host=host,
        port=port,
        emit_mode=emit_mode,
        focus=focus,
        stream=stream,
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
