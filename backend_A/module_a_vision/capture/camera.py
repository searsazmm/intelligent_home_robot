"""摄像头采集源。

⚠️ 本机（开发机）**没有摄像头设备**，这是预期情况而不是缺陷。因此本文件
的错误分支是**被认真对待的主路径**之一：没有摄像头时给出的提示必须直接
告诉运维人员改用什么，而不是抛一个 "cannot open camera"。

后端选择
--------

Windows 上 OpenCV 有多个后端（DirectShow / MSMF / 自动）。同一个摄像头
在不同后端下的打开成功率、首帧延迟都不一样，所以这里**依次尝试**，
第一个能真正读出帧的胜出，并把后端名记进 :meth:`CameraSource.describe`。
真机排查"昨天还能用"类故障时，这个信息往往就是答案。
"""

from __future__ import annotations

import time

import numpy as np

from .base import BaseCapture, CaptureConfig, CaptureError

try:  # 没装 opencv 时仍可 import 本模块（合成/回放路径不需要它）
    import cv2
except ImportError:  # pragma: no cover - 本机已装
    cv2 = None  # type: ignore[assignment]


class NoCameraError(CaptureError):
    """没有可用摄像头。

    单独成类是为了让上层能**区分**"设备不存在"与"设备存在但读取失败"：
    前者应当直接换成合成源继续跑，后者应当重试或告警。
    """


#: 尝试顺序。DirectShow 在 Windows 上最稳且首帧最快，MSMF 次之。
_BACKENDS: tuple[tuple[int, str], ...] = (
    (getattr(cv2, "CAP_DSHOW", 700), "CAP_DSHOW"),
    (getattr(cv2, "CAP_MSMF", 1400), "CAP_MSMF"),
    (getattr(cv2, "CAP_ANY", 0), "CAP_ANY"),
)

#: 打开后连续读多少帧才算"真的能用"。有些设备 open() 成功但一帧都读不出来。
_PROBE_FRAMES = 3


class CameraSource(BaseCapture):
    """``cv2.VideoCapture`` 包装。"""

    def __init__(
        self,
        index: int = 0,
        cfg: CaptureConfig | None = None,
        backend: int | None = None,
    ) -> None:
        """
        :param index: 摄像头设备号。默认 0（系统默认摄像头）。
        :param backend: 指定 OpenCV 后端；``None`` 表示按 :data:`_BACKENDS` 依次尝试。
        """
        super().__init__(cfg)
        self._index_device = index
        self._forced_backend = backend
        self._cap = None
        self._backend_name = ""
        #: 单调时钟基准，用于把墙钟换算成"自采集开始起的秒数"。
        self._mono_start = time.monotonic()
        #: 连续读取失败的次数。上层可据此判断"偶发丢帧"还是"设备掉线"。
        self.consecutive_failures = 0

    # ------------------------------------------------------------ 生命周期

    def open(self) -> None:
        """打开摄像头。失败时抛 :class:`NoCameraError` 并给出替代方案。"""
        if cv2 is None:
            raise CaptureError(
                "未安装 opencv，无法使用摄像头采集。请安装 opencv-contrib-python，"
                "或改用 SyntheticSource / CsvReplaySource / VideoFileSource。"
            )
        if self._cap is not None:
            self.close()

        tried: list[str] = []
        candidates = (
            [(self._forced_backend, f"backend={self._forced_backend}")]
            if self._forced_backend is not None
            else [(code, name) for code, name in _BACKENDS]
        )

        for code, name in candidates:
            tried.append(name)
            cap = cv2.VideoCapture(self._index_device, code)
            if not cap.isOpened():
                cap.release()
                continue

            self._apply_props(cap)
            if self._probe(cap):
                self._cap = cap
                self._backend_name = name
                self._index = 0
                self._exhausted = False
                self._last_ts = 0.0
                self._pace_ts = 0.0
                self._last_wall = None
                self._mono_start = time.monotonic()
                self._opened = True
                self.consecutive_failures = 0
                return

            cap.release()

        raise NoCameraError(
            f"未检测到可用摄像头（设备号 {self._index_device}，依次尝试了 {'、'.join(tried)}）。\n"
            "本机（开发机）没有摄像头设备属于预期情况，不是缺陷。请改用以下任一采集源：\n"
            '  SyntheticSource(scenario="drowsy")      —— 合成剧本，本机验证过的主路径\n'
            '  CsvReplaySource("data/xxx.csv")         —— 回放模块 A 导出的 CSV\n'
            '  VideoFileSource("data/xxx.mp4")         —— 回放视频文件\n'
            "若确实接了摄像头，请检查：设备号是否正确、是否被其它程序独占、"
            "Windows 隐私设置里是否允许桌面应用访问摄像头。"
        )

    def _apply_props(self, cap) -> None:
        """请求目标分辨率与帧率。设备不支持的项会被忽略，属正常现象。"""
        if self.cfg.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
        if self.cfg.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))
        if self.cfg.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))

    def _probe(self, cap) -> bool:
        """连读几帧确认设备真的出图。

        这一步不能省：部分虚拟摄像头/被占用的设备 ``isOpened()`` 为真，
        但 ``read()`` 永远返回失败，于是故障会推迟到采集循环里才暴露。
        """
        for _ in range(_PROBE_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._backend_name = ""
        super().close()

    def describe(self) -> str:
        if self._cap is None:
            return f"摄像头（设备 {self._index_device}，未打开）"
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (
            f"摄像头：设备={self._index_device}，后端={self._backend_name}，"
            f"实际分辨率={w}x{h}"
        )

    # ------------------------------------------------------------ 读取

    def read(self) -> np.ndarray | None:
        """读一帧 BGR 图。

        ``None`` 表示**本帧**没取到（设备偶发丢帧），此时源仍然活着、可以继续读；
        连续失败次数见 :attr:`consecutive_failures`，由上层决定何时告警或重连。
        真正拔掉设备时 ``cv2`` 会持续返回失败，这里不会抛异常——采集循环
        不该因为一次读取失败就整个崩掉。
        """
        if self._cap is None:
            raise CaptureError(
                "摄像头尚未打开。请先调用 open() —— 或在 with 语句里使用它。"
            )
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            self.consecutive_failures += 1
            return None

        self.consecutive_failures = 0
        # 摄像头是**实时**源：时间戳取自单调时钟，而不是"帧号 ÷ 帧率"。
        # 设备实际帧率常常达不到请求值（光照不足时自动降帧），用帧号算出来的
        # 时间戳会系统性地偏慢，而持续时长判定全部建立在它上面。
        self._index += 1
        self._last_ts = time.monotonic() - self._mono_start
        return frame
