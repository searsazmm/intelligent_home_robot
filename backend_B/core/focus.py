# -*- coding: utf-8 -*-
"""VAI 专注度指数 —— 从 ``non-contact/专注度指数/focus_index/`` 移植的内核。

指数叫什么
----------

``VAI`` = Visual Attention Index，**视觉注意行为趋势指数**，取值 0-100。
它在需求文档里叫「专注度指数」（§12.5 第 9 项），归宿是 **B**：

* 需求文档 §12 开篇：「A 的职责边界：只采信号、只出特征，**不做指数合成**」；
* 需求文档 §4 B2：``focus`` 是 **B 内部使用、不下发 C** 的静默模式判据；
* `HTTP接口文档.md` 把 ``focusScore`` 列在 B 的 HTTP 指标里。

所以 A 侧只发视线原始量（api_doc §3.5.3），融合与指数全在这里。

⚠️ 这份实现与参考实现**不可比**
--------------------------------

逐条对齐的是**公式与常量**，不是**输入**。有四处是必须照实说明的适配，
把其中任何一条丢掉，算出来的就只是"一个像 VAI 的数"：

1. **头姿基线由 B 自己建（``PassiveCalibrator``）。**
   参考实现的 ``relative_yaw_deg`` 来自它自己的采集层（已做基线校正）；
   A-包的头姿是**原始角度**（``metrics/head_pose.py:normalize_angles`` 只
   把角度 clamp 到 ±90，**不减基线**—— 基线校正在 GYZ 的 ``vision_a.py`` 里，
   是另一套实现）。所以 A→B 这条线上，"相对角度"必须由 B 用
   :class:`PassiveCalibrator` 现场建立个人基线，否则 ``pose_alignment``
   量的是"这台设备装在哪"，不是"老人头转了多少"。
   注意 ``需求文档.md §12.5 #4`` 曾写"A 提供基线相对角度"，**那是错的**。
2. **``eye_open`` 是单眼版。** 参考实现是左右眼各自建 EAR 基线、各自跑状态机，
   再 ``fuse_eye_states`` 融合（避免平均 EAR 掩盖单眼遮挡）。B 手里只有
   **双眼均值** ``ear``（api_doc §3.2 就一个 ``ear`` 字段），所以这里是单路。
   代价：单眼遮挡在 B 侧看不出来。
3. **门控做了裁剪**，见 :class:`FocusGate` 的文档 —— 像素类判据
   （模糊、过曝、人脸框面积、眼周对比度）B 一条都做不了。
4. **``gaze`` 是一次语义替换。** 参考实现优先用 ``target_roi_probability``
   （"有没有看向任务目标区"），``alignment`` 只是回退；B 只有
   「有没有看向正前方」。被动观察模式下参考实现本来也回退到 alignment，
   所以这个替换可辩护，但**指数不能声称与参考实现可比**。

**另外，静默的判据是本项目自己加的，参考实现里没有对应物**：
它只出分、不下判断，既没有阈值也没有"证据要齐"这一说。而 B 要用这个分去
**关掉一个功能**，所以 :func:`should_stay_silent` 在"指数可信"之上又加了两道 ——
证据齐全（:data:`SILENCE_REQUIRED_MODALITIES`）与够专注
（``config.FOCUS_SILENT_MIN_INDEX``）。缺任何一道，功能都会**反向**
（摄像头一通电就再也不主动开口 / 越困越安静），且不留任何报错。
两处都是**工程判据，不是从参考实现抄来的**，也因此不参与"与参考实现可比"的说法。

**没有移植**的部分（都是有意的）：工程可信状态（``_reliability`` /
``engineering_confidence_status``）、事件检测（持续闭眼 / 凝视偏离）、
``base_components``、会话报告与审计、ROI / 任务协议。它们服务于"研究级
可复现报告"，而这里要的是"这一刻要不要安静一点"。同理，
参考实现里 ``_stability`` 与 ``_raw_history`` 也没移植。

一条**结构性**的红线
--------------------

**体征（HR/RR）绝不进这个融合。** 参考实现里有 ``features/rppg_adapter.py``
且它的 ``RppgEvidence.validated_for_fusion`` 默认 ``False``；B 侧连
:class:`~core.typed_messages.RppgReading` 都不往这里传 —— 这个模块的签名里
**根本没有**体征的入口。体征走需求文档 §B3 那条线（现在也没做）。

时钟：两个钟，别混
------------------

* **指数内部全用报文里的 ``timestamp``**（A 的采集秒，从 0 开始）——
  校准窗口、有效观察时长、EMA 的 ``dt`` 都靠它做差。混进墙钟会让同一个
  窗口横跨两套时间轴。
* **"多久没更新了"用墙钟**（``received_at`` 一侧）。A 的 ``timestamp``
  是它自己的进程时间，A 一挂就永远停在那个值上，用它判新鲜度会得出
  "数据一直很新"。所以 :meth:`FocusTracker.last_observation_wall` 记的是
  本机墙钟，给 §静默模式的抑制闸用。
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Any, Callable, Deque, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 版本与权重（照搬参考实现，别改数值 —— 改了要同时改这里的版本号）
# ---------------------------------------------------------------------------

#: 公式与权重的版本标识。参考实现原样带过来，便于对照与追责。
FORMULA_VERSION = "VAI-F1"
WEIGHT_VERSION = "VAI-W1"

#: 三模态融合权重（研究工程初值，未经临床验证）。
DEFAULT_WEIGHTS: Dict[str, float] = {"gaze": 0.55, "pose": 0.25, "eye_open": 0.20}

#: 模态组合 → 可读名字。**不同模态组合下的同一个分值不等价**，
#: 所以名字要一路带出去，别只留一个数。
_MODALITY_CONFIG_MAP = {
    frozenset(["gaze", "pose", "eye_open"]): "完整模态",
    frozenset(["gaze", "pose"]): "凝视+头姿",
    frozenset(["gaze", "eye_open"]): "凝视+睁眼",
    frozenset(["pose", "eye_open"]): "头姿+睁眼",
    frozenset(["gaze"]): "仅凝视",
    frozenset(["pose"]): "仅头姿",
    frozenset(["eye_open"]): "仅睁眼",
}


class EyeState(str, Enum):
    """单路（双眼均值）眼睑状态。"""

    OPEN = "睁眼"
    CLOSED = "闭眼"
    UNKNOWN = "未知"


class FocusStatus(str, Enum):
    """一次观测的测量状态。

    **只保留 B 判得出来的那几个**（对照参考实现的 ``MeasurementStatus``）：
    丢掉了 ``CAMERA_MOVING`` / ``MULTIPLE_FACES`` / ``TOO_DARK`` /
    ``OVEREXPOSED`` / ``BLURRED`` / ``FACE_TOO_SMALL`` / ``FACE_TOO_CLOSE`` /
    ``LOW_QUALITY`` / ``SINGLE_EYE_DEGRADED`` —— 前七个要像素或人脸框，
    后两个要单眼质量分或 A 的画面质量分，B 手里都没有。
    **其中 ``LOW_QUALITY`` 不必单独存在**：A 的 ``has_face`` 已经是
    "有脸 **且** 画面质量合格"（``wire.to_v1_sample``），它不合格时
    B 收到的是 ``has_face=false``，走 :attr:`NO_FACE`。
    """

    VALID = "有效"
    WARMING_UP = "校准预热中"
    NO_FACE = "未检测到人脸"
    EYE_OCCLUDED = "眼部遮挡"
    OUT_OF_POSE_RANGE = "头姿超出范围"
    INSUFFICIENT_OBSERVATION = "观察不足"


#: ``index_status`` 的字符串。**照抄参考实现的措辞**，因为"研究趋势（非认知专注）"
#: 这句话本身是这套东西的安全声明，不该在这里被改写成更顺口的说法。
IDX_UNAVAILABLE = "不可用"
IDX_INSUFFICIENT = "有效观察时长不足"
IDX_TREND = "研究趋势（非认知专注）"

#: 有效观察时长达标后才出指数。与参考实现同值。
MIN_VALID_SECONDS_FOR_INDEX = 10.0

#: 静默模式**必须**拿到的两路证据（需求文档 §4 B2：「视线稳定 + 有规律眨眼」）。
#:
#: ⚠️ **为什么不是"有指数就行"。** ``EvidenceFusion`` 把缺失模态剔出分母再归一化，
#: 于是**只剩一路证据时那一路的权重要顶到 1.0** —— 实测：眼睛处于无法判定的
#: 状态时（EAR 基线锁在偏低值上，睁闭都看不出来），视线与睁眼两路一起被门控
#: 拿掉，只剩头姿，``pose_alignment=1.0`` ⇒ **指数 100.0**。
#: 也就是说"一位闭着眼、头却冲着镜头的长者"会拿到满分专注。
#: 那是研究用的趋势指数可以容忍的已知产物（它只出分、不下判断），
#: 但拿它去**关掉主动关怀**就成了反的：越困越安静。
#:
#: 两路证据在实现上其实是**同进同退**的（凝视门要求 ``eye_state == OPEN``，
#: 而 ``eye_open`` 只在 OPEN 时有值），写成两条是为了让判据读起来与需求文档
#: 一致，也为了将来改 ``eye_open`` 的判定方式时不必回头改这里的语义。
#:
#: ⚠️ **判的是"最近一段窗口里出现过"，不是"这一帧在"**（见
#: :attr:`FocusSnapshot.recent_modalities`）。逐帧判会让静默模式**每 4 秒抖一次**：
#: 眨眼那一帧 ``eye_state`` 不是 OPEN，视线与睁眼一起被门控拿掉，
#: 判据立刻变成"证据不全"→ 退出静默 → 下一帧又进去。实测就是这样，
#: 而且每次退出都给主动关怀开了一次门（一次眨眼就让机器人开口了）。
#: 眨眼本来还是需求文档里"有规律眨眼"那半个判据，逐帧判等于把它当成了干扰。
SILENCE_REQUIRED_MODALITIES = ("gaze", "eye_open")


@dataclass(frozen=True)
class FocusConfig:
    """可调常量。字段名与参考实现的 ``ResearchConfig`` 对齐，便于逐条对照。

    默认值**全部照搬**参考实现（见 ``focus_index/config/loader.py``）。
    测试要"快出 VAI"时必须另建一个 config 实例传入，
    **绝不改这里的默认值** —— 默认值一改，演示出来的数就不再是 VAI。
    """

    # --- 被动校准 ---
    calibration_seconds: float = 5.0
    calibration_min_samples: int = 30
    calibration_yaw_range_max: float = 8.0
    calibration_pitch_range_max: float = 8.0
    calibration_quality_min: float = 0.45

    # --- 平滑与出值 ---
    #: 一阶低通的时间常数（秒）：``alpha = 1 - exp(-dt/tau)``。
    #: 用它而不是逐帧固定 alpha，是为了让 10fps 与 15fps 下的平滑行为一致。
    ema_time_constant_seconds: float = 1.5
    min_valid_seconds_for_index: float = MIN_VALID_SECONDS_FOR_INDEX

    # --- 有效时间与重置 ---
    #: 两次有效观测间隔超过它就**不再计入**有效观察时长。
    #: 这也是"视线报文必须与帧同频"的原因：掉到 1Hz 以下这个门永远过不去，
    #: 表现为"校准/出值永远做不完"且不报错。
    max_sample_gap_seconds: float = 1.0
    #: 连续多久没有可评估的观测就重置整个会话（含校准基线）。
    state_reset_gap_seconds: float = 2.0

    # --- 眼部（单路 EAR）---
    ear_quality_min: float = 0.35
    ear_close_absolute: float = 0.20
    ear_open_absolute: float = 0.22
    ear_close_relative_drop: float = 0.30
    ear_open_relative_drop: float = 0.20
    ear_baseline_min: float = 0.20
    ear_baseline_samples: int = 30

    # --- 头姿硬门（原始角度）---
    #: 注意是**原始**角度不是相对角度：这道门问的是"关键点还准不准"，
    #: 而那是头相对**镜头**的角度决定的。见 :class:`FocusGate`。
    hard_yaw_deg: float = 45.0
    hard_pitch_deg: float = 35.0
    hard_roll_deg: float = 35.0

    # --- B 侧自加：把两路报文配成一次观测 ---
    #: 视线报文与帧报文的时间戳允许差多少还认为"说的是同一帧"。
    #: 实际上 A 是同一次 ``frame.ts`` 投影出来的，两个数**逐位相同**；
    #: 留这个容差是为了不把将来可能的抖动变成"配不上对"。
    pair_tolerance_seconds: float = 0.5

    #: 静默判决判"证据齐不齐"时回看的窗口（报文时间轴的秒）。
    #:
    #: 存在的理由是**眨眼**：一次眨眼 0.1–0.3s，那一帧 ``eye_state`` 不是 OPEN，
    #: 视线与睁眼两路一起被门控拿掉。逐帧判的话静默模式会跟着眨眼抖动，
    #: 而每次抖动都给主动关怀开一次门（实测：一次眨眼就让机器人开了口）。
    #: 2.0s 覆盖一次眨眼加采样抖动，又远短于"人真的闭眼躺下"的时间尺度
    #: （那种情况窗口里很快就只剩头姿了，静默会正确地解除）。
    evidence_window_seconds: float = 2.0


# ---------------------------------------------------------------------------
# 证据（纯函数，逐一对照参考实现）
# ---------------------------------------------------------------------------

def pose_alignment(
    rel_yaw_deg: Optional[float], rel_pitch_deg: Optional[float]
) -> Optional[float]:
    """头姿对正程度 ``[0,1]``。参考实现 ``features/head_pose.py`` 原文。

    ``clamp(1 - hypot(rel_yaw, rel_pitch)/20, 0, 1)`` —— 20 度是**工程初值**，
    不是医疗阈值。输入必须是**相对个人基线**的角度。
    """
    if rel_yaw_deg is None or rel_pitch_deg is None:
        return None
    deviation = math.hypot(float(rel_yaw_deg), float(rel_pitch_deg))
    return max(0.0, min(1.0, 1.0 - deviation / 20.0))


def gaze_evidence(gaze_off: Optional[float]) -> Optional[float]:
    """视线对正程度 ``[0,1]``。对应参考实现 ``features/gaze.py``。

    参考实现的签名是 ``gaze_evidence(alignment, target_roi_probability,
    gaze_quality)``：质量 ≤0 返回 ``None``，有 ROI 概率就优先用它、否则回退
    ``alignment``。B 侧**没有任务目标区**（那是屏幕任务场景才有的东西），
    所以只剩 alignment 这一支 —— 也就是把 ``1 - gaze_off`` 取回来。

    质量那一道**在这里不需要**：api_doc §3.5.3 的不变式已经保证
    "``gaze`` 为 ``null`` ⟺ ``gaze_quality == 0``"，两侧都校验，
    所以"取不到"这件事在 :class:`FocusTracker` 里就变成 ``None`` 了。
    """
    if gaze_off is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(gaze_off)))


class EarBaseline:
    """单路睁眼 EAR 基线。只接收"质量合格且明显睁眼"的样本。

    起点是 :attr:`FocusConfig.ear_baseline_samples` 个样本，取中位数 ——
    中位数而不是均值，因为眨眼/眯眼那些偏低的样本正是要挡掉的东西。
    """

    def __init__(self, config: FocusConfig) -> None:
        self.config = config
        self._samples: deque = deque(maxlen=max(90, config.ear_baseline_samples * 3))

    @property
    def value(self) -> Optional[float]:
        if len(self._samples) < self.config.ear_baseline_samples:
            return None
        return float(median(self._samples))

    def update(self, ear: Optional[float], quality: float) -> Optional[float]:
        if (
            ear is not None
            and quality >= self.config.ear_quality_min
            and ear >= self.config.ear_open_absolute
            and ear >= self.config.ear_baseline_min
        ):
            self._samples.append(float(ear))
        return self.value

    def reset(self) -> None:
        self._samples.clear()


class EyeStateTracker:
    """单路眼睑状态机：闭合与重开用**不同**阈值（迟滞）。

    照搬参考实现 ``features/eye_state.py``。迟滞是必要的：只有一个阈值时，
    EAR 在它附近抖动会让 ``eye_open`` 在 1.0/0.0 之间反复横跳，而
    ``eye_open`` 的权重是 0.20，跳一次就是 20 分的噪声。
    """

    def __init__(self, config: FocusConfig) -> None:
        self.config = config
        self.state = EyeState.UNKNOWN

    def update(
        self, ear: Optional[float], quality: float, baseline: Optional[float]
    ) -> EyeState:
        if ear is None or ear <= 0 or quality < self.config.ear_quality_min:
            self.state = EyeState.UNKNOWN
            return self.state

        relative_drop = (
            None
            if baseline is None or baseline <= 0
            else (baseline - ear) / baseline
        )
        close_relative = (
            relative_drop is not None
            and relative_drop >= self.config.ear_close_relative_drop
        )
        open_relative = (
            relative_drop is None
            or relative_drop <= self.config.ear_open_relative_drop
        )
        close_signal = ear < self.config.ear_close_absolute or close_relative
        open_signal = ear > self.config.ear_open_absolute and open_relative

        if self.state == EyeState.CLOSED:
            if open_signal:
                self.state = EyeState.OPEN
        elif close_signal:
            self.state = EyeState.CLOSED
        elif open_signal:
            self.state = EyeState.OPEN
        else:
            self.state = EyeState.UNKNOWN
        return self.state

    def reset(self) -> None:
        self.state = EyeState.UNKNOWN


# ---------------------------------------------------------------------------
# 被动校准：A 的头姿是原始角度，个人基线只能由 B 自己建
# ---------------------------------------------------------------------------

class PassiveCalibrator:
    """被动建立"这个人坐在这儿"的头姿基线。照搬参考实现 ``inference/calibration.py``。

    A-包的头姿是**原始角度**（不减基线，见模块文档第 1 条），所以这一步
    **不是可选的冗余**：没有它，``pose_alignment`` 量的是设备安装角度。

    ``ready()`` 是一个**与**条件：时长 ≥5s、样本 ≥30、``calibration_quality``
    ≥0.45、且 yaw/pitch 的中心 10–90 分位跨度都 ≤8 度。任何一条不满足都
    停在 :attr:`FocusStatus.WARMING_UP` —— 也就是**不出指数**，而不是出一个
    可疑的指数。这是有意的：早期样本的"基线"可能只是老人正在转头。

    ⚠️ 这里的 ``update`` 会因样本间隔超限而 ``reset()``。配合默认 1.0 秒的
    ``max_sample_gap_seconds``，**视线/帧报文的速率就是这条链路的功能前提**：
    掉到 1Hz 以下，基线永远建不起来，而且不报任何错。A 侧因此逐帧发送。
    """

    def __init__(self, config: FocusConfig) -> None:
        self.config = config
        self.samples: list = []
        self.started_at: Optional[float] = None
        self.reference_yaw: Optional[float] = None
        self.reference_pitch: Optional[float] = None
        self.calibration_quality: float = 0.0
        self.yaw_range: Optional[float] = None
        self.pitch_range: Optional[float] = None
        self.last_sample_at: Optional[float] = None

    def update(self, timestamp: float, yaw_deg: float, pitch_deg: float) -> None:
        if self.reference_yaw is not None and self.reference_pitch is not None:
            return
        if (
            self.last_sample_at is not None
            and timestamp - self.last_sample_at > self.config.max_sample_gap_seconds
        ):
            self.reset()
        if self.started_at is None:
            self.started_at = timestamp
        self.samples.append((float(yaw_deg), float(pitch_deg)))
        self.last_sample_at = timestamp
        self._update_quality()
        if self.ready(timestamp):
            self.reference_yaw = median(item[0] for item in self.samples)
            self.reference_pitch = median(item[1] for item in self.samples)

    @staticmethod
    def _central_range(values: list) -> float:
        """10–90 分位跨度。用分位而不是 min/max，是为了不被一两个离群样本拉爆。"""
        ordered = sorted(values)
        if len(ordered) < 2:
            return 0.0

        def percentile(q: float) -> float:
            position = (len(ordered) - 1) * q
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            fraction = position - lower
            return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

        low = percentile(0.10)
        high = percentile(0.90)
        return max(0.0, high - low)

    def _update_quality(self) -> None:
        if not self.samples:
            self.calibration_quality = 0.0
            self.yaw_range = None
            self.pitch_range = None
            return
        self.yaw_range = self._central_range([item[0] for item in self.samples])
        self.pitch_range = self._central_range([item[1] for item in self.samples])
        yaw_load = self.yaw_range / max(self.config.calibration_yaw_range_max, 1e-6)
        pitch_load = self.pitch_range / max(
            self.config.calibration_pitch_range_max, 1e-6
        )
        stability = max(0.0, min(1.0, 1.0 - 0.5 * (yaw_load + pitch_load)))
        sample_coverage = min(
            1.0, len(self.samples) / max(self.config.calibration_min_samples, 1)
        )
        self.calibration_quality = max(0.0, min(1.0, stability * sample_coverage))

    def ready(self, now: float) -> bool:
        return (
            self.started_at is not None
            and now - self.started_at >= self.config.calibration_seconds
            and len(self.samples) >= self.config.calibration_min_samples
            and self.calibration_quality >= self.config.calibration_quality_min
            and (self.yaw_range or 0.0) <= self.config.calibration_yaw_range_max
            and (self.pitch_range or 0.0) <= self.config.calibration_pitch_range_max
        )

    def reset(self) -> None:
        self.samples.clear()
        self.started_at = None
        self.reference_yaw = None
        self.reference_pitch = None
        self.calibration_quality = 0.0
        self.yaw_range = None
        self.pitch_range = None
        self.last_sample_at = None

    def relative_pose(
        self, yaw: Optional[float], pitch: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        if (
            yaw is None
            or pitch is None
            or self.reference_yaw is None
            or self.reference_pitch is None
        ):
            return None, None
        return yaw - self.reference_yaw, pitch - self.reference_pitch

    def metrics(self) -> Dict[str, Any]:
        return {
            "sample_count": len(self.samples),
            "yaw_range_deg": None if self.yaw_range is None else round(self.yaw_range, 3),
            "pitch_range_deg": (
                None if self.pitch_range is None else round(self.pitch_range, 3)
            ),
            "quality": round(self.calibration_quality, 4),
            "ready": self.reference_yaw is not None and self.reference_pitch is not None,
        }


# ---------------------------------------------------------------------------
# 融合：缺失模态不进分母
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionResult:
    value: float
    available_modalities: Tuple[str, ...]
    normalized_weights: Dict[str, float]
    modality_completeness: float
    evidence_consistency: float
    modality_config_id: str
    formula_version: str = FORMULA_VERSION
    weight_version: str = WEIGHT_VERSION


class EvidenceFusion:
    """加权融合。照搬参考实现 ``inference/evidence_fusion.py``。

    **缺失模态不进分母**（权重重归一化），所以只有头姿时也能出值 —— 但代价是
    "同一个分值在不同模态组合下不等价"，这就是 ``modality_config_id`` 与
    ``modality_completeness`` 必须一起带出去的原因。

    ⚠️ 由此推出一个**必须靠测试钉住**的陷阱：如果 ``eye_open`` 恒为 ``None``，
    权重会静默地从 ``0.55/0.25/0.20`` 变成 ``0.6875/0.3125``，指数照出、
    照得像模像样，**只是不再是 VAI**。所以 :class:`FocusTracker` 会把
    ``available_modalities`` 一路带进快照，并在模态长期不齐时打一条限频告警。
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)

    def combine(self, evidence: Dict[str, Optional[float]]) -> Optional[FusionResult]:
        available = {
            key: max(0.0, min(1.0, float(value)))
            for key, value in evidence.items()
            if key in self.weights and value is not None
        }
        if not available:
            return None
        total_weight = sum(self.weights[key] for key in available)
        normalized = {key: self.weights[key] / total_weight for key in available}
        value = sum(normalized[key] * score for key, score in available.items())
        pairs = [
            (a, b)
            for i, a in enumerate(available.values())
            for b in list(available.values())[i + 1 :]
        ]
        consistency = (
            1.0
            if not pairs
            else 1.0 - sum(abs(a - b) for a, b in pairs) / len(pairs)
        )
        config_key = frozenset(available.keys())
        return FusionResult(
            value=value,
            available_modalities=tuple(
                key for key in self.weights if key in available
            ),
            normalized_weights=normalized,
            modality_completeness=total_weight / sum(self.weights.values()),
            evidence_consistency=max(0.0, min(1.0, consistency)),
            modality_config_id=_MODALITY_CONFIG_MAP.get(config_key, "自定义配置"),
        )


# ---------------------------------------------------------------------------
# 门控（裁剪版）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateDecision:
    status: FocusStatus
    reason: str = ""
    assessable: bool = False


class FocusGate:
    """可评估性门控。**这是参考实现 ``MeasurementGate`` 的裁剪版。**

    参考实现有 15 条判据，B 手里只有 ``has_face`` / ``ear`` / ``pitch`` /
    ``yaw`` / ``roll``，外加焦点报文里的 ``gaze``，所以：

    ============================================ ======== ==================================
    参考实现的判据                                  B      理由
    ============================================ ======== ==================================
    ``face_count == 0``                           **保留**  → ``has_face == False``
    ``face_count > 1``（多人脸）                    裁掉     A 不报人脸个数，其人脸后端只取一张
    画面过暗 / 过曝 / 模糊 / 眼周对比度                裁掉     要像素
    人脸框面积过小 / 过大                            裁掉     要 bbox（而它是**敏感字段**，不出 A）
    头姿大角度硬门                                   **保留**  用**原始**角度，见下
    双眼均不可可靠观察                                **降级**  只有双眼均值 ``ear``：``ear <= 0`` 即"量不出眼宽"
    ``face_quality`` / ``image_quality`` / ``pose_quality`` / 综合质量  裁掉  A 的 ``has_face`` 已含画面质量闸
    单眼降级 ``SINGLE_EYE_DEGRADED``                裁掉     没有单眼量
    ============================================ ======== ==================================

    硬门用的是**原始**角度而不是相对基线，这一点值得写下来：这道门问的是
    "关键点还准不准"，而那是头相对**镜头**的转角决定的（侧脸 50 度时地标
    本身就不可信，与老人平时习惯怎么坐无关）。相对角度只用于
    ``pose_alignment`` 那一项证据。
    """

    def __init__(self, config: FocusConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        has_face: bool,
        ear: Optional[float],
        yaw: Optional[float],
        pitch: Optional[float],
        roll: Optional[float],
    ) -> GateDecision:
        if not has_face:
            return GateDecision(FocusStatus.NO_FACE, "未检测到人脸")
        for name, value, limit in (
            ("yaw", yaw, self.config.hard_yaw_deg),
            ("pitch", pitch, self.config.hard_pitch_deg),
            ("roll", roll, self.config.hard_roll_deg),
        ):
            if value is not None and abs(value) > limit:
                return GateDecision(
                    FocusStatus.OUT_OF_POSE_RANGE,
                    f"大角度头姿超出几何特征可靠范围（{name}={value:.1f}°，"
                    f"上限 {limit:g}°）",
                )
        if ear is None or ear <= 0:
            # 有脸却量不出眼宽：双眼关键点都不可靠。参考实现这一档是
            # "双眼均不可可靠观察"，B 只有均值，所以判据退化成 ear <= 0。
            return GateDecision(FocusStatus.EYE_OCCLUDED, "眼部不可可靠观察")
        return GateDecision(FocusStatus.VALID, "", True)


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FocusSnapshot:
    """一刻的专注度状态。**不可变**，可以跨线程传阅。

    单看 ``index`` 会误导，所以下面这些必须一起看：

    * ``index is None`` ⇒ 没有指数，``index_status`` 说明为什么（不可用 /
      有效观察时长不足）；
    * ``status`` 才是测量状态，``VALID`` 且 ``index is not None`` 才是"真的
      算出来了"；
    * ``available_modalities`` 说明这个数由哪几项证据组成 ——
      **少一项就换了一组权重**，分值不再等价。
    """

    index: Optional[float] = None
    index_status: str = IDX_UNAVAILABLE
    status: FocusStatus = FocusStatus.NO_FACE
    reason: str = ""
    available_modalities: Tuple[str, ...] = ()
    #: 最近一段窗口内**出现过**的模态（并集）。静默判决看的是它，不是
    #: 上面那一格 —— 理由见 :data:`SILENCE_REQUIRED_MODALITIES`：
    #: 眨眼那一帧视线与睁眼会同时消失，逐帧判会让静默模式每 4 秒抖一次。
    recent_modalities: Tuple[str, ...] = ()
    normalized_weights: Dict[str, float] = field(default_factory=dict)
    modality_completeness: float = 0.0
    evidence_consistency: float = 0.0
    modality_config_id: str = ""
    raw_value: Optional[float] = None
    valid_seconds: float = 0.0
    calibration: Dict[str, Any] = field(default_factory=dict)
    formula_version: str = FORMULA_VERSION
    weight_version: str = WEIGHT_VERSION

    @property
    def usable(self) -> bool:
        """这个快照能不能拿去做行为判决（静默模式）。

        三件事都要成立：**算出来了**、**状态是 VALID**、**观测时长够了**
        （后者已经体现在 ``index is not None`` 上，再写一次是为了让
        "为什么不能用"在调用处一眼可读）。
        """
        return self.index is not None and self.status == FocusStatus.VALID


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

#: 模态长期不齐时打几条 WARNING。A 一旦不再发视线（或帧与视线的
#: ``timestamp`` 对不上），模态会永久退化成"头姿+睁眼"，指数照出但已不是 VAI。
_DEGRADED_WARN_LIMIT = 3


class FocusTracker:
    """把两条报文流拼成一次观测，并产出 VAI。

    **输入是两条流，这一点是本类与参考实现最大的结构差异。**
    参考实现的 ``Observation`` 是采集层一次性装配好的；B 侧头姿/ear 在帧报文里、
    ``gaze`` 在 focus 报文里，得自己配对。

    配对发生在 :meth:`on_reading`（视线报文到达时），取**最近一帧**帧报文；
    两者的 ``timestamp`` 必须落在 :attr:`FocusConfig.pair_tolerance_seconds`
    之内。选视线侧触发而不是帧侧，是因为 A 的发送顺序是
    "帧 → 体征 → 视线"，同一帧的视线报文到得**晚**——挂在帧上永远配不到
    那一帧的视线。两条流都是逐帧的，所以触发频率一样。

    线程约定与 :class:`~core.vision_state.VisionStateEvaluator` 相同：
    ``on_sample`` / ``on_reading`` 只由视觉读取线程调，``snapshot()`` 可以被
    别的线程调（返回不可变快照）。
    """

    def __init__(
        self,
        config: Optional[FocusConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or FocusConfig()
        self._clock = clock

        self._gate = FocusGate(self.config)
        self._calibrator = PassiveCalibrator(self.config)
        self._ear_baseline = EarBaseline(self.config)
        self._eye_tracker = EyeStateTracker(self.config)
        self._fusion = EvidenceFusion(DEFAULT_WEIGHTS)

        #: 最近一帧帧报文（配对用）。**存的是不可变快照**，见 :meth:`on_sample`。
        self._last_sample: Optional[Any] = None
        self._last_snapshot = FocusSnapshot()

        # --- 时间与状态 ---
        self._last_seen_at: Optional[float] = None      # 报文时钟
        self._last_valid_at: Optional[float] = None
        self._invalid_since: Optional[float] = None
        self._valid_seconds = 0.0
        self._ema: Optional[float] = None
        self._last_ema_timestamp: Optional[float] = None
        #: **墙钟**上最后一次成功产出观测的时刻，给"数据还新不新"用（见模块文档）。
        self.last_observation_wall: Optional[float] = None

        # --- 可观测性 ---
        self.readings_seen = 0
        self.observations = 0
        #: 视线报文没配上帧报文的次数。**非零且持续增长**就说明配对逻辑失效，
        #: 结果是头姿证据悄悄消失（模态退化），指数照出。
        self.pair_misses = 0
        #: 因时间戳回退或观测中断而重置会话的次数。
        self.resets = 0
        self._degraded_warned = 0
        #: 最近 ``evidence_window_seconds`` 内每次观测实际拿到的模态
        #: ``[(报文时间戳, 模态元组)]``，给静默判决的"证据齐不齐"用
        #: （逐帧判会被眨眼打穿，见 :data:`SILENCE_REQUIRED_MODALITIES`）。
        self._recent_evidence: Deque[Tuple[float, Tuple[str, ...]]] = deque()

    # ------------------------------------------------------------ 输入

    def on_sample(self, sample: Any) -> None:
        """收一帧帧报文。只暂存，不做计算 —— 计算在 :meth:`on_reading`。"""
        self._last_sample = sample

    def on_reading(self, reading: Any) -> None:
        """收一条视线报文，并尝试配成一次观测。"""
        self.readings_seen += 1
        self._process(reading)

    # ------------------------------------------------------------ 主流程

    def _process(self, reading: Any) -> None:
        timestamp = float(reading.timestamp)
        now_wall = self._clock()

        if self._detect_discontinuity(timestamp):
            # 会话已被重置，这一条不再参与：重置后的第一条观测从零开始。
            return

        sample = self._last_sample
        paired = (
            sample is not None
            and abs(float(sample.timestamp) - timestamp)
            <= self.config.pair_tolerance_seconds
        )
        if not paired:
            self.pair_misses += 1

        has_face = bool(paired and sample.has_face)
        decision = self._gate.evaluate(
            has_face=has_face,
            ear=(sample.ear if paired else None),
            yaw=(sample.yaw if paired else None),
            pitch=(sample.pitch if paired else None),
            roll=(sample.roll if paired else None),
        )
        if not decision.assessable:
            self._record_invalid(timestamp)
            self._last_snapshot = FocusSnapshot(
                index=None,
                index_status=IDX_UNAVAILABLE,
                status=decision.status,
                reason=decision.reason,
                recent_modalities=self._recent_union(timestamp),
                calibration=self._calibrator.metrics(),
            )
            if decision.status == FocusStatus.EYE_OCCLUDED:
                # 眼睑状态必须跟着回落，否则"闭着眼"会一直留在上一态。
                self._eye_tracker.update(None, 0.0, None)
            return

        # 眼部在**校准锁定之前**就要开始积累：两个基线都各要 30 个样本，
        # 串行的话首个指数会比参考实现晚一大截。判据与参考实现的采集层一致
        # ——"有脸、且 EAR 是个正数"。
        quality = 1.0
        baseline = self._ear_baseline.update(sample.ear, quality)
        eye_state = self._eye_tracker.update(sample.ear, quality, baseline)

        self._calibrator.update(timestamp, sample.yaw, sample.pitch)
        if not self._calibrator.ready(timestamp):
            self._last_valid_at = None
            self._last_snapshot = FocusSnapshot(
                index=None,
                index_status=IDX_UNAVAILABLE,
                status=FocusStatus.WARMING_UP,
                reason="正在建立稳定的头姿基线",
                recent_modalities=self._recent_union(timestamp),
                calibration=self._calibrator.metrics(),
            )
            return

        rel_yaw, rel_pitch = self._calibrator.relative_pose(sample.yaw, sample.pitch)

        # 凝视门（参考实现 pipeline.py:354）：**眼没睁开就不采信视线**。
        # 闭着眼"看向正前方"不是一个观测，而 A 的视线估计在闭眼帧上本来
        # 也退化，所以这里必须两道都过。
        gaze = None
        if eye_state == EyeState.OPEN and reading.usable:
            gaze = gaze_evidence(reading.gaze)

        pose = pose_alignment(rel_yaw, rel_pitch)
        eye_open = (
            1.0
            if eye_state == EyeState.OPEN
            else 0.0
            if eye_state == EyeState.CLOSED
            else None
        )

        fused = self._fusion.combine(
            {"gaze": gaze, "pose": pose, "eye_open": eye_open}
        )
        if fused is None:
            self._last_valid_at = None
            self._last_snapshot = FocusSnapshot(
                index=None,
                index_status=IDX_UNAVAILABLE,
                status=FocusStatus.INSUFFICIENT_OBSERVATION,
                reason="没有足够的可用视觉证据",
                recent_modalities=self._recent_union(timestamp),
                calibration=self._calibrator.metrics(),
            )
            return

        self._note_modalities(fused)
        self._record_evidence(timestamp, fused.available_modalities)

        smoothed = self._time_constant_ema(timestamp, fused.value)
        valid_seconds = self._record_valid_time(timestamp)
        self.observations += 1
        self.last_observation_wall = now_wall

        # ⚠️ ``status`` **不在** common 里，两个分支各自显式给。
        # 放进 common 再 ``**common`` 展开的话，它会把下面 INSUFFICIENT 分支
        # 写在前面的 status 覆盖回 VALID —— 表现是"指数是 None，状态却说有效"，
        # 而 ``usable`` 要求两者都成立，于是静默模式会在"其实没算出来"的时候
        # 拿着一个 None 去判。测试 test_index_waits_for_min_valid_seconds 抓的就是它。
        common = dict(
            available_modalities=fused.available_modalities,
            recent_modalities=self._recent_union(timestamp),
            normalized_weights={
                key: round(value, 4)
                for key, value in fused.normalized_weights.items()
            },
            modality_completeness=round(fused.modality_completeness, 4),
            evidence_consistency=round(fused.evidence_consistency, 4),
            modality_config_id=fused.modality_config_id,
            raw_value=round(fused.value * 100.0, 1),
            valid_seconds=round(valid_seconds, 3),
            calibration=self._calibrator.metrics(),
            formula_version=fused.formula_version,
            weight_version=fused.weight_version,
        )

        if valid_seconds < self.config.min_valid_seconds_for_index:
            self._last_snapshot = FocusSnapshot(
                index=None,
                index_status=IDX_INSUFFICIENT,
                status=FocusStatus.INSUFFICIENT_OBSERVATION,
                reason=f"有效观察 {valid_seconds:.1f}s < "
                f"{self.config.min_valid_seconds_for_index:g}s",
                **common,
            )
            return

        self._last_snapshot = FocusSnapshot(
            index=round(smoothed * 100.0, 1),
            index_status=IDX_TREND,
            status=FocusStatus.VALID,
            reason="",
            **common,
        )

    # ------------------------------------------------------------ 内部

    def _detect_discontinuity(self, timestamp: float) -> bool:
        """时间戳回退 / 观测中断 → 重置整个会话（含校准基线）。

        照搬参考实现 ``_handle_context`` 的两条。回退为什么必须处理：
        A 重连摄像头时 ``ts`` 会从几百秒跳回 0（见 backend_A README §4.7），
        基线是旧时间轴上的，留着就会拿它去减新时间轴的角度。
        """
        reset = False
        reason = ""
        if self._last_seen_at is not None:
            gap = timestamp - self._last_seen_at
            if gap < 0:
                reset, reason = True, "timestamp_rollback"
            elif gap >= self.config.state_reset_gap_seconds:
                reset, reason = True, "observation_gap"
        self._last_seen_at = timestamp
        if reset:
            self._reset_session()
            self._last_snapshot = FocusSnapshot(
                index=None,
                index_status=IDX_UNAVAILABLE,
                status=FocusStatus.INSUFFICIENT_OBSERVATION,
                reason=f"会话已重置（{reason}）",
                calibration=self._calibrator.metrics(),
            )
        return reset

    def _record_invalid(self, timestamp: float) -> None:
        """记一次"这一刻不可评估"。与参考实现同：连续不可评估够久就重置。"""
        if self._invalid_since is None:
            self._invalid_since = timestamp
        self._last_valid_at = None
        if timestamp - self._invalid_since >= self.config.state_reset_gap_seconds:
            self._reset_session()
            self._invalid_since = timestamp

    def _reset_session(self) -> None:
        self._calibrator.reset()
        self._ear_baseline.reset()
        self._eye_tracker.reset()
        # 模态窗口也要清：新一轮的头一个窗口里没道理还留着上一轮的证据
        # （--loop 回放时否则会出现"刚重置就已经证据齐全"）。
        self._recent_evidence.clear()
        self._valid_seconds = 0.0
        self._last_valid_at = None
        self._invalid_since = None
        self._ema = None
        self._last_ema_timestamp = None
        self.resets += 1

    def _record_valid_time(self, timestamp: float) -> float:
        """累计"有效观察时长"。间隔超限的那一段**不计入**（见配置里的注释）。"""
        if self._last_valid_at is not None:
            gap = timestamp - self._last_valid_at
            if 0.0 <= gap <= self.config.max_sample_gap_seconds:
                self._valid_seconds += gap
        self._last_valid_at = timestamp
        self._invalid_since = None
        return self._valid_seconds

    def _time_constant_ema(self, timestamp: float, raw_value: float) -> float:
        """``alpha = 1 - exp(-dt/tau)``。**首帧 EMA 就等于首帧 raw。**

        用时间常数而不是逐帧固定 alpha，是为了不受帧率波动影响（10fps 与
        15fps 下平滑行为一致）。这条也解释了为什么数值基准测试里
        "首帧指数 == raw"是可以断言的。
        """
        if self._ema is None or self._last_ema_timestamp is None:
            self._ema = raw_value
        else:
            dt = max(0.0, timestamp - self._last_ema_timestamp)
            tau = max(1e-6, self.config.ema_time_constant_seconds)
            alpha = 1.0 - math.exp(-dt / tau)
            self._ema = alpha * raw_value + (1.0 - alpha) * self._ema
        self._last_ema_timestamp = timestamp
        return self._ema

    def _record_evidence(self, timestamp: float, modalities: Tuple[str, ...]) -> None:
        """把这次观测实际拿到的模态记进滑动窗口，并丢掉窗口外那些。

        只在**写线程**（``on_reading``）里调用；:meth:`_recent_union` 是只读的。
        """
        self._recent_evidence.append((timestamp, tuple(modalities)))
        cutoff = timestamp - self.config.evidence_window_seconds
        while self._recent_evidence and self._recent_evidence[0][0] < cutoff:
            self._recent_evidence.popleft()

    def _recent_union(self, timestamp: float) -> Tuple[str, ...]:
        """最近窗口内出现过的模态的**并集**，顺序照权重表（日志好读）。

        为什么是并集而不是"这一帧"：一次眨眼 0.1–0.3s，那一帧的视线与睁眼
        会同时消失；逐帧判会让静默模式跟着眨眼抖动，而每次抖动都放行一次
        主动关怀。并集判的是"这两路证据最近还活着吗"—— 眨眼不影响结论，
        而**长期**判不出眼部状态（基线锁死在偏低值上）仍然会被挡下。
        """
        cutoff = timestamp - self.config.evidence_window_seconds
        seen = set()
        for seen_at, modalities in self._recent_evidence:
            if seen_at >= cutoff:
                seen.update(modalities)
        return tuple(key for key in self._fusion.weights if key in seen)

    def _note_modalities(self, fused: FusionResult) -> None:
        """模态长期不齐 → 限频告警。

        这不是洁癖：缺一个模态会让权重重归一化，指数照出、照得像模像样，
        **只是不再是 VAI**。让它出声，比让它在报告里静静躺一个月好。
        """
        missing = [
            key for key in self._fusion.weights
            if key not in fused.available_modalities
        ]
        if not missing or self._degraded_warned >= _DEGRADED_WARN_LIMIT:
            return
        self._degraded_warned += 1
        # 用 print 而不是 logging：本模块是纯逻辑，不持有 logger 配置，
        # 而且这三行只在模态真的退化时出现，不该被日志级别过滤掉。
        print(
            f"[B] ⚠ 专注度模态不全：缺 {missing}（当前 {fused.modality_config_id}，"
            f"权重重归一化为 {fused.normalized_weights}）—— "
            f"指数仍会输出，但已不等价于完整模态 VAI"
        )

    # ------------------------------------------------------------ 输出

    def snapshot(self) -> FocusSnapshot:
        """最近一次算出来的状态。**永远返回一个快照，不返回 None。**

        没有任何观测时返回的是 :class:`FocusSnapshot` 的默认值
        （``index=None``、``status=NO_FACE``），调用方不必先判空。
        """
        return self._last_snapshot

    def seconds_since_last_observation(self, now_wall: Optional[float] = None) -> Optional[float]:
        """离上一次**成功产出观测**过了多少墙钟秒。``None`` = 从来没成功过。

        用墙钟而不是报文时间戳：A 一挂，它的 ``timestamp`` 就永远停在最后一个
        值上，"数据还新不新"这个问题只有墙钟答得了。
        """
        if self.last_observation_wall is None:
            return None
        now = self._clock() if now_wall is None else now_wall
        return max(0.0, now - self.last_observation_wall)


def focus_is_live(
    tracker: FocusTracker, stale_seconds: float, now_wall: Optional[float] = None
) -> Tuple[bool, str]:
    """**能不能拿这个指数去做行为判决**（静默模式）。返回 ``(可信, 不可信的原因)``。

    三个条件缺一不可，而且**每一条都不是多余的**：

    1. **有指数** —— ``index is None`` 意味着压根没算出来（校准没锁定、
       有效观察不够、门控拦了）。不是一个低分，是没有分数。
    2. **状态有效** —— ``status`` 必须是 :attr:`FocusStatus.VALID`。
       1 与 2 合起来就是 :attr:`FocusSnapshot.usable`。
    3. **数据够新** —— 这一条**只能靠墙钟**。``FocusTracker`` 在 A 掉线时
       不会报错，它只是**不再更新**：``snapshot()`` 会一直返回最后一刻的值，
       所以只看 1、2 的话，老人 20 分钟前的那次专注会把主动关怀
       **无限期**关掉，而且日志里一片安静。这是本函数存在的全部理由。

    返回原因而不只是 ``False``，是为了让"为什么机器人不说话了"有据可查
    （见 ``core/proactive.py`` 的 ``assess``）。
    """
    snapshot = tracker.snapshot()
    if snapshot.index is None:
        return False, f"没有可用指数（{snapshot.index_status}）"
    if snapshot.status != FocusStatus.VALID:
        return False, f"测量状态不是有效（{snapshot.status.value}）"
    age = tracker.seconds_since_last_observation(now_wall)
    if age is None:
        return False, "从未产出过观测"
    if age >= stale_seconds:
        return False, f"观测已过期 {age:.1f}s（上限 {stale_seconds:g}s）"
    return True, ""


def should_stay_silent(
    tracker: FocusTracker,
    *,
    stale_seconds: float,
    min_index: float,
    now_wall: Optional[float] = None,
) -> Tuple[bool, str]:
    """**该不该为了"老人在专注"而不打扰**。返回 ``(静默, 否则的原因)``。

    = :func:`focus_is_live` 可信 **且** 证据齐全（:data:`SILENCE_REQUIRED_MODALITIES`）
    **且** 指数达到 ``min_index``。

    ⚠️ **为什么不止 :func:`focus_is_live` 那一条。** 两种"看着都对、其实是反的"：

    1. **"指数可用"不等于"老人专注"。** 只要脸在画面里、管线没坏，指数就一直是
       VALID 的 —— 扭头看窗外、低头发呆照样出分（实测走神场景 43.1）。
       只判可信的话**机器人会在摄像头一通电之后就再也不主动开口**，
       而日志里写着"老人正在专注"。
    2. **"指数可用"也不等于"这个指数可信"。** 缺模态时权重重归一化，
       只剩头姿也能算出 **100.0**（详见 :data:`SILENCE_REQUIRED_MODALITIES`）——
       闭着眼的长者拿到满分专注，越困越安静。

    所以这里是在可信之上再加两道：证据齐全 + 够专注。前两道是"这个分能不能信"，
    第三道是"这个分够不够高"。把任意一道去掉，功能都会**反向**，
    而且不会有任何报错 —— 这是本函数存在的主要理由。

    ``min_index`` 是**工程初值，不是经过验证的临床切点** —— 参考实现自己就
    声明它的指数是"研究趋势（非认知专注）"，没有给任何阈值。所以这里只做
    "明显地没在看"与"明显地正看着"的分开，不做细粒度判读。

    顺序也是有意的：先可信、再齐全、最后比分数。不可信时连分数都不该被引用
    （可能是几十分钟前的旧值），缺证据时报"缺了什么"比报"分数不够"更能指向
    真正的原因。
    """
    live, reason = focus_is_live(tracker, stale_seconds, now_wall)
    if not live:
        return False, reason

    snapshot = tracker.snapshot()
    missing = [name for name in SILENCE_REQUIRED_MODALITIES
               if name not in snapshot.recent_modalities]
    if missing:
        return False, (
            f"专注度证据不全，缺 {'、'.join(missing)}"
            f"（最近 {tracker.config.evidence_window_seconds:g}s 内只有 "
            f"{'+'.join(snapshot.recent_modalities) or '无'}；"
            "缺证据时会权重重归一化，分数不可比）"
        )

    index = snapshot.index
    assert index is not None  # focus_is_live 已保证；防止将来改坏后静默比较 None
    if index < min_index:
        return False, f"专注度 {index:.1f} 未达静默门槛 {min_index:g}"
    return True, ""


__all__ = [
    "DEFAULT_WEIGHTS",
    "EarBaseline",
    "EvidenceFusion",
    "EyeState",
    "EyeStateTracker",
    "FocusConfig",
    "FocusGate",
    "FocusSnapshot",
    "FocusStatus",
    "FocusTracker",
    "FORMULA_VERSION",
    "FusionResult",
    "GateDecision",
    "IDX_INSUFFICIENT",
    "IDX_TREND",
    "IDX_UNAVAILABLE",
    "MIN_VALID_SECONDS_FOR_INDEX",
    "SILENCE_REQUIRED_MODALITIES",
    "PassiveCalibrator",
    "WEIGHT_VERSION",
    "focus_is_live",
    "gaze_evidence",
    "pose_alignment",
    "should_stay_silent",
]
