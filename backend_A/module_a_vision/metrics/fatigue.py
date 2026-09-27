"""疲劳度综合评分——严格实现《系统设计方案》§4.3。

::

    fatigue_score =
        0.35 × min(perclos / 0.40, 1.0)          # 闭眼占比
      + 0.25 × P(tired)                           # 表情疲惫概率
      + 0.20 × min(long_closure_sec / 10.0, 1.0)  # 最长持续闭眼
      + 0.10 × (1 − P(focused))                   # 注意力涣散
      + 0.10 × blink_abnormality                  # 眨眼频率异常度

为什么要加权融合而不是只看 PERCLOS
------------------------------------

单看 PERCLOS 会被两种正常情形误伤：**午睡**和**闭眼养神**。
两者都会让闭眼占比飙高，但都不该触发"休息提醒"——老人本来就在休息。
加入表情、持续闭眼、注意力三个维度后，单纯的"安静闭眼"得分会显著低于
"坐着打瞌睡"（后者通常伴随头部下沉、表情疲惫、注意力涣散）。

权重与分档都是可调的：老人个体差异很大，建议在 M5 阶段用真实数据校准
（设计方案 §4.10）。
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.enums import FatigueLevel

# ================================================================ 权重
# 与设计方案 §4.3 一一对应。改这里等于改系统的疲劳判定，务必回归测试。

W_PERCLOS = 0.35
W_TIRED_EXPRESSION = 0.25
W_LONG_CLOSURE = 0.20
W_INATTENTION = 0.10
W_BLINK_ABNORMALITY = 0.10

#: PERCLOS 的归一化参考线。达到此值即认为该项满分。
PERCLOS_REFERENCE = 0.40

#: 持续闭眼的归一化参考线（秒）。
LONG_CLOSURE_REFERENCE = 10.0

#: 正常眨眼频率（次/分）与允许的偏离幅度。
NORMAL_BLINK_RATE = 15.0
BLINK_RATE_SPAN = 15.0

# ================================================================ 分档
NONE_UPPER = 0.35    # score < 0.35 → 无疲劳
SEVERE_LOWER = 0.70  # score ≥ 0.70 → 重度疲劳


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def blink_abnormality(blink_rate_per_min: float) -> float:
    """眨眼频率异常度 [0,1]。

    0 表示频率正常（15 次/分），1 表示偏离 15 次以上。
    频率过低（困倦）与过高（眼干、紧张）都算异常。
    """
    return _clamp(abs(blink_rate_per_min - NORMAL_BLINK_RATE) / BLINK_RATE_SPAN)


def compute_fatigue_score(
    perclos: float,
    p_tired: float,
    long_closure_sec: float,
    p_focused: float,
    blink_rate_per_min: float,
) -> float:
    """按设计方案 §4.3 的公式计算综合疲劳分。

    :param perclos: 最近 60 秒闭眼帧占比（**不是** 10 秒窗口内的占比）。
    :param p_tired: 表情分类器给出的"疲惫"概率。
    :param long_closure_sec: 当前持续闭眼秒数。
    :param p_focused: 注意力分类器给出的"专注"概率。
    :param blink_rate_per_min: 最近 60 秒眨眼频率。
    """
    score = (
        W_PERCLOS * min(perclos / PERCLOS_REFERENCE, 1.0)
        + W_TIRED_EXPRESSION * _clamp(p_tired)
        + W_LONG_CLOSURE * min(long_closure_sec / LONG_CLOSURE_REFERENCE, 1.0)
        + W_INATTENTION * (1.0 - _clamp(p_focused))
        + W_BLINK_ABNORMALITY * blink_abnormality(blink_rate_per_min)
    )
    return _clamp(score)


def level_from_score(
    score: float,
    none_upper: float = NONE_UPPER,
    severe_lower: float = SEVERE_LOWER,
) -> FatigueLevel:
    """把疲劳分映射为等级。"""
    if score >= severe_lower:
        return FatigueLevel.SEVERE
    if score >= none_upper:
        return FatigueLevel.MILD
    return FatigueLevel.NONE


def confidence_from_score(
    score: float, none_upper: float = NONE_UPPER, severe_lower: float = SEVERE_LOWER
) -> float:
    """置信度 = 距离最近分档边界的裕度。

    分数落在档位正中间时最自信；贴着边界时最不自信。
    这样"刚好 0.35 分"不会被当成一个确定的轻度疲劳。
    """
    if score >= severe_lower:
        span = 1.0 - severe_lower
        margin = min(score - severe_lower, span)
        return _clamp(0.6 + 0.35 * (margin / span if span > 0 else 0.0), 0.0, 0.95)
    if score >= none_upper:
        span = severe_lower - none_upper
        center = (severe_lower + none_upper) / 2.0
        margin = 1.0 - abs(score - center) / (span / 2.0) if span > 0 else 0.0
        return _clamp(0.55 + 0.35 * margin, 0.0, 0.95)
    span = none_upper
    margin = min(none_upper - score, span)
    return _clamp(0.6 + 0.35 * (margin / span if span > 0 else 0.0), 0.0, 0.95)


@dataclass(frozen=True, slots=True)
class FatigueMetrics:
    """疲劳评估结果。"""

    level: FatigueLevel = FatigueLevel.NONE
    score: float = 0.0
    confidence: float = 0.0


def assess(
    perclos: float,
    p_tired: float,
    long_closure_sec: float,
    p_focused: float,
    blink_rate_per_min: float,
) -> FatigueMetrics:
    """一步算出疲劳等级、分数与置信度。"""
    score = compute_fatigue_score(
        perclos=perclos,
        p_tired=p_tired,
        long_closure_sec=long_closure_sec,
        p_focused=p_focused,
        blink_rate_per_min=blink_rate_per_min,
    )
    return FatigueMetrics(
        level=level_from_score(score),
        score=score,
        confidence=confidence_from_score(score),
    )
