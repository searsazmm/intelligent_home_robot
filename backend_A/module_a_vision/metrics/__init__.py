"""逐帧指标计算——把一帧的人脸信息变成可判定的标量。

五个指标各占一个文件，对应设计方案里的五类观测：

===========================  ==========================================
:mod:`~module_a_vision.metrics.expression`  表情四分类（blendshape 启发式）
:mod:`~module_a_vision.metrics.head_pose`   头部姿态（roll/pitch/yaw）
:mod:`~module_a_vision.metrics.attention`   注意力（专注 / 失神）
:mod:`~module_a_vision.metrics.eye`         眼部状态与眨眼统计
:mod:`~module_a_vision.metrics.fatigue`     疲劳评分（PERCLOS 加权）
===========================  ==========================================

这些是**指标**，不是结论。它们算的是"闭合度 0.93""roll 34°"，
而"这算不算异常"是 :mod:`~module_a_vision.aggregate` 与 B 的闸门的事。
把这两层分开，是为了让阈值全部集中在一处、可审可调。

⚠️ **表情分类是启发式基线，不是训练模型。** 设计方案 §8.3 已把
"缺少老人表情数据"列为首要风险。这里不伪造精度：接口
:class:`~module_a_vision.metrics.expression.ExpressionClassifier` 留好了，
等真实数据的模型接进来即可替换。
"""

from __future__ import annotations

from .attention import AttentionMetrics, AttentionTracker
from .expression import (
    BlendshapeExpressionClassifier,
    ExpressionClassifier,
    NeutralCalibrator,
)
from .eye import EyeMetrics, EyeTracker, fuse_or_raw
from .fatigue import (
    FatigueMetrics,
    assess,
    blink_abnormality,
    compute_fatigue_score,
    level_from_score,
)
from .head_pose import HeadPoseEstimate, HeadPoseEstimator, euler_from_matrix

__all__ = [
    "AttentionMetrics",
    "AttentionTracker",
    "BlendshapeExpressionClassifier",
    "ExpressionClassifier",
    "NeutralCalibrator",
    "EyeMetrics",
    "EyeTracker",
    "fuse_or_raw",
    "FatigueMetrics",
    "assess",
    "blink_abnormality",
    "compute_fatigue_score",
    "level_from_score",
    "HeadPoseEstimate",
    "HeadPoseEstimator",
    "euler_from_matrix",
]
