"""头部姿态：从帧特征得到角度并三分类。

角度来源有两条路，优先级如下：

1. **MediaPipe 的 ``facial_transformation_matrixes``**（4×4 齐次矩阵）。
   这是 MediaPipe 专门为头部姿态输出的量，含公制平移，比 solvePnP 更稳、
   更省算力。真实后端走这条。
2. 帧特征里直接给出的 ``pitch_deg`` / ``yaw_deg`` / ``roll_deg``
   （合成后端与 CSV 回放走这条）。

刻意保留 :func:`euler_from_matrix` 而不只用标签，是因为 B 侧要**独立复核**
几何条件（``|roll| ≥ 25°``）。如果只传标签，B 就只能盲信上游。

分类本身调用 :mod:`shared.geometry`——A 与 B 共用同一份判定式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.enums import HeadPose
from shared.frame_features import FrameFeatures
from shared.geometry import classify_head_pose, is_tilted, is_upright


@dataclass(frozen=True, slots=True)
class HeadPoseEstimate:
    """一次头姿估计的结果。"""

    label: HeadPose
    confidence: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


def euler_from_matrix(m: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    """从 4×4（行主序，长度 16）或 3×3（长度 9）旋转矩阵解欧拉角。

    :returns: ``(pitch_deg, yaw_deg, roll_deg)``

    约定（与 :mod:`shared.geometry` 一致）：

    * ``pitch`` 正值 = 低头
    * ``yaw`` 正值 = 向右偏
    * ``roll`` 正值 = 向右倾（歪头）

    使用 ZYX 内旋序列（yaw → pitch → roll），这是头部姿态解的常规选择。
    万向节死锁附近（``|pitch| ≈ 90°``）时 yaw 与 roll 不可分离，
    此时给出钳制后的结果并让置信度自然下降——正常人脸不会到这个角度。
    """
    if len(m) == 16:
        # 取左上 3×3 旋转块。
        r = [
            [m[0], m[1], m[2]],
            [m[4], m[5], m[6]],
            [m[8], m[9], m[10]],
        ]
    elif len(m) == 9:
        r = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
    else:
        raise ValueError(f"矩阵长度必须是 9 或 16，收到 {len(m)}")

    sy = math.sqrt(r[0][0] ** 2 + r[1][0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(-r[2][0], sy)
        roll = math.atan2(r[1][0], r[0][0])
    else:
        # 万向节死锁。
        pitch = math.atan2(-r[1][2], r[1][1])
        yaw = math.atan2(-r[2][0], sy)
        roll = 0.0

    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll),
    )


def normalize_angles(
    pitch_deg: float, yaw_deg: float, roll_deg: float
) -> tuple[float, float, float]:
    """把角度规范到 [-90, 90]。**不做符号翻转。**

    :func:`euler_from_matrix` 解出的符号**已经**是"pitch 正值 = 低头"，
    与 :mod:`shared.geometry` 的约定一致——真机标定实测：低头 pitch 为正、
    抬头为负、平视接近 0。

    这里原本有一个 ``-pitch_deg``，理由是"MediaPipe 的变换矩阵解出的 pitch
    符号与'低头为正'相反"。**那个前提是错的**，它引入的翻转把抬头判成了低头：
    ``is_bowed``（``pitch >= 25``）于是在抬头时才成立、真低头时反而不成立。
    不要把这个负号加回去——除非你先用手边的摄像头重新测一遍低头/抬头的符号。
    """
    return (
        _clamp_angle(pitch_deg),
        _clamp_angle(yaw_deg),
        _clamp_angle(roll_deg),
    )


def _clamp_angle(v: float) -> float:
    return max(-90.0, min(90.0, v))


class HeadPoseEstimator:
    """头姿估计器。

    本身无状态——角度逐帧独立，稳定性由聚合层的跨窗口持续时长负责。
    """

    def __init__(
        self,
        roll_tilt_deg: float = 25.0,
        pitch_bow_deg: float = 25.0,
    ) -> None:
        self._roll_tilt = roll_tilt_deg
        self._pitch_bow = pitch_bow_deg

    def estimate(self, frame: FrameFeatures) -> tuple[HeadPose, float, tuple[float, float, float]]:
        """返回 ``(标签, 置信度, (pitch, yaw, roll))``。"""
        if not frame.has_face or not frame.quality.valid:
            return HeadPose.UPRIGHT, 0.0, (0.0, 0.0, 0.0)

        pitch, yaw, roll = frame.pitch_deg, frame.yaw_deg, frame.roll_deg
        label = classify_head_pose(
            pitch, yaw, roll,
            roll_tilt=self._roll_tilt,
            pitch_bow=self._pitch_bow,
        )
        confidence = self._confidence(label, pitch, yaw, roll)
        return label, confidence, (pitch, yaw, roll)

    def _confidence(
        self, label: HeadPose, pitch: float, yaw: float, roll: float
    ) -> float:
        """置信度 = 该分类离边界有多远。

        贴着阈值（例如 roll 正好 25.2°）时不给高置信度——那个角度上
        "端正"与"歪倒"本就难分，硬给高置信度会让下游的持续性判定
        失去意义。
        """
        if label == HeadPose.TILTED:
            margin = abs(roll) - self._roll_tilt
            return min(0.70 + 0.25 * min(margin / 20.0, 1.0), 0.95)
        if label == HeadPose.BOWED:
            margin = pitch - self._pitch_bow
            if margin < 0:
                # 由"三者都不满足"回落而来，本身就不确定。
                return 0.55
            return min(0.65 + 0.25 * min(margin / 25.0, 1.0), 0.95)
        # UPRIGHT：离三个上界越远越自信。
        from shared.geometry import UPRIGHT_PITCH_DEG, UPRIGHT_ROLL_DEG, UPRIGHT_YAW_DEG

        slack = min(
            UPRIGHT_PITCH_DEG - abs(pitch),
            UPRIGHT_ROLL_DEG - abs(roll),
            UPRIGHT_YAW_DEG - abs(yaw),
        )
        return min(0.70 + 0.25 * min(max(slack, 0.0) / 15.0, 1.0), 0.95)
