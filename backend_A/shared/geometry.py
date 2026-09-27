"""头姿与眼部的几何判定——A 与 B 共用的**唯一**实现。

为什么必须共用：A 用这些判定式产出标签，B 用**同一组判定式复核**标签
（B 不盲信上游）。如果两边各写一份，阈值一改就漂移，于是出现
"A 说歪倒、B 说没歪倒"这类最难排查的故障。把判定式收敛到一处之后，
A 与 B 的分歧就只可能来自数据本身。

本模块是纯函数，无状态、无 I/O、无依赖。
"""

from __future__ import annotations

from .enums import EyeState, HeadPose

# ================================================================ 默认阈值
# 取自《系统设计方案》§4.2。A 与 B 的配置都从这里取默认值，
# 保证「同一个数字只定义一次」。

#: |roll| ≥ 此值判为"头部歪倒"。这是最高危规则 R6 的组成条件。
ROLL_TILT_DEG = 25.0

#: pitch ≥ 此值判为"低头"。
PITCH_BOW_DEG = 25.0

#: 头部端正的三个上界。
UPRIGHT_PITCH_DEG = 20.0
UPRIGHT_ROLL_DEG = 15.0
UPRIGHT_YAW_DEG = 25.0

#: 闭眼程度阈值。
CLOSURE_CLOSED = 0.80      # ≥ 此值算"闭眼"
CLOSURE_HALF_LOW = 0.55    # 半闭眼下界
CLOSURE_HALF_HIGH = 0.80   # 半闭眼上界

#: EAR 的典型取值范围，用于把 EAR 线性映射到闭眼程度。
EAR_OPEN = 0.30
EAR_CLOSED = 0.10


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ================================================================ 头姿

def is_tilted(roll_deg: float, threshold: float = ROLL_TILT_DEG) -> bool:
    """头部是否歪倒（左右倾）。"""
    return abs(roll_deg) >= threshold


def is_bowed(pitch_deg: float, threshold: float = PITCH_BOW_DEG) -> bool:
    """头部是否低垂（前倾）。"""
    return pitch_deg >= threshold


def is_upright(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    pitch_max: float = UPRIGHT_PITCH_DEG,
    roll_max: float = UPRIGHT_ROLL_DEG,
    yaw_max: float = UPRIGHT_YAW_DEG,
) -> bool:
    """头部是否端正。"""
    return (
        abs(pitch_deg) < pitch_max
        and abs(roll_deg) < roll_max
        and abs(yaw_deg) < yaw_max
    )


def classify_head_pose(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    roll_tilt: float = ROLL_TILT_DEG,
    pitch_bow: float = PITCH_BOW_DEG,
) -> HeadPose:
    """三分类头部姿态。

    优先级刻意是 **歪倒 > 低头 > 端正**：

    * 歪倒排在最前，因为它承载安全含义（R6 的组成条件），
      不能被"同时也低着点头"掩盖掉。
    * 三者都不满足时（例如 pitch=22、yaw=40），回落到 ``BOWED``。
      这是刻意的保守选择——只有 ``UPRIGHT`` 是良性状态，
      在拿不准时不应把状态判成良性。
    """
    if is_tilted(roll_deg, roll_tilt):
        return HeadPose.TILTED
    if is_bowed(pitch_deg, pitch_bow):
        return HeadPose.BOWED
    if is_upright(pitch_deg, yaw_deg, roll_deg):
        return HeadPose.UPRIGHT
    return HeadPose.BOWED


# ================================================================ 眼部

def closure_from_ear(ear: float) -> float:
    """把 EAR 线性映射为闭眼程度 [0,1]。"""
    span = EAR_OPEN - EAR_CLOSED
    if span <= 0:
        return 0.0
    return _clamp((EAR_OPEN - ear) / span, 0.0, 1.0)


def fuse_closure(ear: float, blink_blendshape: float) -> float:
    """融合 EAR 与 ``eyeBlink`` blendshape 得到闭眼程度。

    两个信号各有短板，融合能互补：

    * **EAR 是几何量**，对光照不敏感，但戴老花镜时关键点漂移会失真。
    * **blendshape 是模型回归量**，对遮挡更鲁棒，但存在个体基线偏移
      （老人眼睑下垂会让它常年偏高）。

    因此 EAR 占 0.6、blendshape 占 0.4——这是**未做中性标定时的临时权重**。
    完成个体标定后应当把 blendshape 的权重调高。
    """
    return _clamp(
        0.6 * closure_from_ear(ear) + 0.4 * blink_blendshape,
        0.0,
        1.0,
    )


def classify_eye_state(
    closure_ratio: float,
    closed_th: float = CLOSURE_CLOSED,
    half_low: float = CLOSURE_HALF_LOW,
) -> EyeState:
    """三分类眼部状态。"""
    if closure_ratio >= closed_th:
        return EyeState.CLOSED
    if closure_ratio >= half_low:
        return EyeState.HALF_CLOSED
    return EyeState.NORMAL_BLINK


def is_closed(closure_ratio: float, threshold: float = CLOSURE_CLOSED) -> bool:
    return closure_ratio >= threshold


def is_half_closed(
    closure_ratio: float,
    low: float = CLOSURE_HALF_LOW,
    high: float = CLOSURE_HALF_HIGH,
) -> bool:
    return low <= closure_ratio < high
