"""采集子包——四种可互换的数据来源。

====================  ============  ==================================================
源                    种类          用途
====================  ============  ==================================================
:class:`CameraSource`      帧（BGR）     真机实时采集（本机无摄像头，属预期情况）
:class:`VideoFileSource`   帧（BGR）     回放视频文件，可循环
:class:`SyntheticSource`   帧 + 特征    **本机主路径**：按剧本产出，确定性、零模型
:class:`CsvReplaySource`   特征          回放 A 自己导出的 CSV，供 B 离线开发
====================  ============  ==================================================

两条协议（:class:`FrameSource` / :class:`FeatureSource`）的取舍见
:mod:`module_a_vision.capture.base` 的模块文档。用 :func:`read_any` 可以在
不知道源的种类时读一帧，但**必须**先判定返回的到底是像素还是特征。
"""

from .base import (
    BaseCapture,
    BaseFeatureSource,
    CaptureConfig,
    CaptureError,
    FeatureSource,
    FrameSource,
    is_feature_source,
    is_frame_source,
    read_any,
)
from .camera import CameraSource, NoCameraError
from .csv_replay import (
    EMO_FEATURE_TO_STATE,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    CsvReplaySource,
)
from .synthetic import (
    BUILTIN_SCENARIOS,
    INF,
    STATE_SPECS,
    Scenario,
    Segment,
    StateSpec,
    SyntheticFeatureSource,
    SyntheticSource,
    parse_timeline,
)
from .video_file import VideoFileSource

__all__ = [
    # 契约
    "BaseCapture",
    "BaseFeatureSource",
    "CaptureConfig",
    "CaptureError",
    "FeatureSource",
    "FrameSource",
    "is_feature_source",
    "is_frame_source",
    "read_any",
    # 四种源
    "CameraSource",
    "NoCameraError",
    "VideoFileSource",
    "CsvReplaySource",
    "SyntheticSource",
    "SyntheticFeatureSource",
    # 合成剧本
    "BUILTIN_SCENARIOS",
    "INF",
    "STATE_SPECS",
    "Scenario",
    "Segment",
    "StateSpec",
    "parse_timeline",
    # CSV 格式常量
    "EMO_FEATURE_TO_STATE",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
]
