"""直通人脸后端——把上游构造好的 :class:`FrameFeatures` 原样交给下游。

它是本机的**主路径**，也是全部回归测试的数据来源。之所以让一个"什么都
不做"的后端占据这么重要的位置，是因为接缝被刻意抬到了 ``FrameFeatures``
这一层：合成 478 个地标是在测试几何提取代码，而不是在测试策略
（详见 :mod:`shared.frame_features` 的模块文档）。

它**不处理像素**，收到 ``np.ndarray`` 会直接报错而不是返回一张空特征——
后者的表现是"人脸永远检不出来"，属于最难定位的一类静默故障。
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from shared.frame_features import FrameFeatures, missing_blendshapes

from .base import BaseFaceBackend, FaceBackendError


class SyntheticFaceBackend(BaseFaceBackend):
    """直通后端：``process(features) -> features``。"""

    name = "synthetic-passthrough"

    def __init__(
        self,
        auto_stamp_fps: float | None = None,
        warn_missing_blendshapes: bool = False,
    ) -> None:
        """
        :param auto_stamp_fps: 若给出，则给 ``ts <= 0`` 的帧补一个按该帧率
            递增的时间戳。帧特征不带时间戳时，聚合器会把整个窗口当成
            "所有帧都在 0 秒"，于是持续时长恒为 0、窗口永远不关——
            这个兜底就是为了不让那类故障静默发生。
        :param warn_missing_blendshapes: 是否对缺失的 blendshape 名字告警。
            手工构造特征的测试常只填几个名字，默认关闭避免噪声；
            启动自检与合成源联调时建议打开。
        """
        super().__init__()
        self._fps = auto_stamp_fps
        self._warn_missing = warn_missing_blendshapes
        self._frame_index = 0
        self._last_ts = 0.0

    def open(self) -> None:
        super().open()
        self._frame_index = 0
        self._last_ts = 0.0

    def process(self, frame_or_features: np.ndarray | FrameFeatures) -> FrameFeatures:
        """直通返回特征；收到像素则报错，并指出该换哪个后端。"""
        if not isinstance(frame_or_features, FrameFeatures):
            shape = getattr(frame_or_features, "shape", None)
            raise FaceBackendError(
                "SyntheticFaceBackend 直通特征、不处理像素"
                f"（收到 {type(frame_or_features).__name__}"
                + (f"，shape={shape}" if shape else "")
                + "）。\n"
                "两种改法：\n"
                "  1. 走合成路径：用 SyntheticSource(...).read_features() 直接取 FrameFeatures；\n"
                "  2. 走真实路径：改用 MediaPipeFaceBackend（需要摄像头与模型文件）。"
            )

        frame = frame_or_features

        if self._warn_missing and frame.has_face and frame.blendshapes:
            missing = missing_blendshapes(frame.blendshapes)
            if missing:
                msg = (
                    f"帧特征缺少 {len(missing)} 个期望的 blendshape："
                    f"{'、'.join(missing)}。缺失的名字会被下游读成 0，"
                    "与“真的为 0”无法区分，表情判定会整体偏向 normal。"
                )
                self._note(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

        if self._fps and frame.ts <= 0.0:
            frame = replace(
                frame,
                ts=self._frame_index / self._fps,
                wall_ts=frame.wall_ts or 0.0,
            )

        self._frame_index += 1
        if frame.ts > self._last_ts:
            self._last_ts = frame.ts
        return frame

    def describe(self) -> str:
        extra = f"，自动补时间戳 {self._fps:g} fps" if self._fps else ""
        return f"合成直通后端（不处理像素{extra}）"


__all__ = ["SyntheticFaceBackend"]
