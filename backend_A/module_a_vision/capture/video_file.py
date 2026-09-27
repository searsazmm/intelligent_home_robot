"""视频文件采集源——按目标帧率回放一段视频，可循环。

时间轴口径（**这是最容易搞错的一点**）
----------------------------------------

``cfg.fps`` 是**输出**帧率，不是原视频帧率：它决定每帧推进多少时间
（``ts = 帧号 ÷ cfg.fps``），也就决定了 10 秒窗口里落多少帧。
原视频帧率只在 :meth:`VideoFileSource.describe` 里报告，供人工核对。

想让时间轴与原视频严格一致，就把 ``cfg.fps`` 设成原视频帧率
（``open()`` 会把探测到的值放进 ``source_fps``）。

回放节奏
--------

默认 ``real_time=False``：**全速**读，时间轴按帧率虚拟推进。离线回归、
批量重放都靠它。要实时演示再打开 ``real_time``（可选 ``speed`` 倍速）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseCapture, CaptureConfig, CaptureError

try:  # 没装 opencv 时仍可 import 本模块
    import cv2
except ImportError:  # pragma: no cover - 本机已装
    cv2 = None  # type: ignore[assignment]


class VideoFileSource(BaseCapture):
    """循环回放一个视频文件。"""

    def __init__(
        self,
        path: str | Path,
        cfg: CaptureConfig | None = None,
    ) -> None:
        """
        :param path: 视频文件路径。
        :param cfg: ``loop=False`` 时播完即 :attr:`exhausted`。
        """
        super().__init__(cfg)
        self._path = Path(path)
        self._cap = None
        #: 原视频帧率（探测值，仅用于报告）。
        self.source_fps: float = 0.0
        #: 原视频总帧数（探测值）。
        self.source_frames: int = 0
        #: 已循环次数。
        self.loops: int = 0

    # ------------------------------------------------------------ 生命周期

    def open(self) -> None:
        if cv2 is None:
            raise CaptureError(
                "未安装 opencv，无法回放视频文件。请安装 opencv-contrib-python。"
            )
        if not self._path.exists():
            raise CaptureError(
                f"视频文件不存在：{self._path.resolve()}\n"
                "请确认路径；相对路径是相对于**进程的工作目录**，不是本文件所在目录。"
            )
        if self._path.is_dir():
            raise CaptureError(f"这是一个目录而不是视频文件：{self._path.resolve()}")

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise CaptureError(
                f"无法打开视频文件：{self._path.resolve()}\n"
                "常见原因：编码格式不受当前 OpenCV 构建支持（如 H.265/HEVC）、文件损坏。"
                "可用 ffmpeg 转成 H.264 的 mp4 再试。"
            )

        self._cap = cap
        self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._index = 0
        self._exhausted = False
        self._last_ts = 0.0
        self._pace_ts = 0.0
        self._last_wall = None
        self.loops = 0
        self._opened = True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().close()

    def describe(self) -> str:
        if self.source_fps > 0:
            src = f"，原视频 {self.source_fps:g} fps"
            if self.source_frames:
                src += f" / {self.source_frames} 帧"
        else:
            src = "，原视频帧率未知"
        return (
            f"视频文件：{self._path.name}（输出 {self.cfg.fps:g} fps{src}，"
            f"{'循环' if self.cfg.loop else '不循环'}）"
        )

    # ------------------------------------------------------------ 读取

    def read(self) -> np.ndarray | None:
        """读下一帧；``None`` 表示已耗尽（不循环且播完）或本帧解码失败。"""
        if self._cap is None:
            raise CaptureError("视频文件尚未打开。请先调用 open()。")
        if self._over_budget():
            self._exhausted = True
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            if not self.cfg.loop:
                self._exhausted = True
                return None
            # 循环：回到开头重读一次。只重试一次——若开头也读不出来，
            # 说明文件本身有问题，继续重试会变成死循环。
            self.loops += 1
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None or frame.size == 0:
                self._exhausted = True
                return None

        ts = self._tick()
        self._pace(ts)
        return frame
