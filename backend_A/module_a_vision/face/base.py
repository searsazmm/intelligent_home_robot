"""人脸后端契约——像素与特征之间的那道接缝。

两种后端，同一个入口：

* :class:`~module_a_vision.face.synthetic.SyntheticFaceBackend`——直通，
  上游已经给出 :class:`~shared.frame_features.FrameFeatures` 时用它。
  本机（无摄像头）的**主路径**，也是全部回归测试的数据来源。
* :class:`~module_a_vision.face.mediapipe_backend.MediaPipeFaceBackend`——
  把像素变成特征。唯一的真实后端，需要模型文件与摄像头。

接口刻意只有一个方法
--------------------

``process(frame_or_features) -> FrameFeatures``

因为"接缝"就在返回值上：调用方只关心拿到一份特征，不关心它是从像素推出来的
还是被直接声明的。多一个 ``process_pixels`` / ``process_features`` 只会让
调用方在两种后端之间写分支，接缝的意义就没了。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from shared.frame_features import FrameFeatures


class FaceBackendError(RuntimeError):
    """人脸后端不可用（模型缺失、依赖缺失、已关闭）。

    消息面向部署人员，一律用中文写清楚缺什么、怎么补。
    """


@runtime_checkable
class FaceBackend(Protocol):
    """人脸后端协议。"""

    def process(self, frame_or_features: np.ndarray | FrameFeatures) -> FrameFeatures: ...

    def close(self) -> None: ...


class BaseFaceBackend(ABC):
    """后端公共实现：生命周期幂等、上下文管理、统一的自检日志出口。

    ``open()`` / ``close()`` 都做成**幂等**的：采集循环里"重连"是很常见的
    动作，让重连路径写两遍 try/except 只会出错。
    """

    #: 后端名，进日志与 :meth:`describe`。
    name: str = "face-backend"

    def __init__(self) -> None:
        self._opened = False
        self._closed = False
        #: 启动自检发现的问题，供上层原样打印到日志。
        self.self_check_notes: list[str] = []

    # ------------------------------------------------------------ 生命周期

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def closed(self) -> bool:
        return self._closed

    def open(self) -> None:
        """准备资源。重复调用不报错。"""
        self._opened = True
        self._closed = False

    def close(self) -> None:
        """释放资源。重复调用不报错。"""
        self._opened = False
        self._closed = True

    def __enter__(self) -> "BaseFaceBackend":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ 处理

    @abstractmethod
    def process(self, frame_or_features: np.ndarray | FrameFeatures) -> FrameFeatures:
        """产出这一帧的特征。"""

    def describe(self) -> str:
        return f"{self.name}（{'已打开' if self._opened else '未打开'}）"

    def _note(self, message: str) -> None:
        """记一条自检问题（去重、限量）。"""
        if message not in self.self_check_notes and len(self.self_check_notes) < 16:
            self.self_check_notes.append(message)


def as_features(obj: np.ndarray | FrameFeatures) -> FrameFeatures | None:
    """入参已经是特征就原样返回，否则返回 ``None``。

    用于让"直通"判断只写一处——它其实就是 ``isinstance``，
    但取个名字能让调用点自解释。
    """
    return obj if isinstance(obj, FrameFeatures) else None
