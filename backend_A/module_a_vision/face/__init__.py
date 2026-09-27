"""人脸后端子包——像素与帧特征之间的那道接缝。

两种后端实现同一个入口 ``process(frame_or_features) -> FrameFeatures``：

* :class:`SyntheticFaceBackend`——直通。上游已经给出
  :class:`~shared.frame_features.FrameFeatures` 时用它。
  **本机（无摄像头）的主路径**，零模型依赖、完全确定性。
* :class:`MediaPipeFaceBackend`——真实后端，把像素变成特征。
  ⚠️ 在本机**未经验证**（没有摄像头、模型文件也未随仓库提供），
  详见 :mod:`module_a_vision.face.mediapipe_backend` 的模块文档。

``MediaPipeFaceBackend`` 是**延迟导入** mediapipe 的：无摄像头场景下
``import module_a_vision.face`` 不需要装 mediapipe（见 requirements.txt 的说明）。
"""

from .base import (
    BaseFaceBackend,
    FaceBackend,
    FaceBackendError,
    as_features,
)
from .mediapipe_backend import (
    DEFAULT_MODEL_PATH,
    MODEL_ENV_VAR,
    MediaPipeFaceBackend,
)
from .synthetic import SyntheticFaceBackend

__all__ = [
    "BaseFaceBackend",
    "FaceBackend",
    "FaceBackendError",
    "as_features",
    # 后端实现
    "SyntheticFaceBackend",
    "MediaPipeFaceBackend",
    # 模型位置
    "DEFAULT_MODEL_PATH",
    "MODEL_ENV_VAR",
]
