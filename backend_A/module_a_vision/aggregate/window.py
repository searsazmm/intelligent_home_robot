"""窗口聚合——把逐帧特征压成一份可上送的 :class:`WindowState`。

这是 A 模块的出口，也是隐私边界上的**最后一道闸门**：过了这里，
数据就要离开进程了。因此这里的输出刻意只有统计量，没有任何
可还原人像的字段（bbox / landmarks 在这里被丢弃）。

窗口内 vs 跨窗口的职责划分（重要）
------------------------------------

* **窗口内**：N-of-M 投票（``frames_ratio`` ≥ ρ）、持续性（``T_min``）。
  本文实现。
* **跨窗口**：连续 N 个窗口。由 B 负责（:mod:`module_b_agent.gate.rules`）。

只有 ``stable_sec`` 是跨窗口的——它由 :class:`StableDurationTracker`
跨窗口累计，**允许超过 ``window.duration_sec``**。

这一点是修掉的一处设计缺陷：早期把 ``stable_sec`` 读作"本窗口内持续的
秒数"，于是"头部歪倒 + 持续闭眼 ≥ 15 秒"这条最高危规则在一个 10 秒窗口上
永远无法成立。现在它从条件首次成立起累计，跨窗口不清零。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from shared.enums import (
    Attention,
    Emotion,
    EyeState,
    FatigueLevel,
    Glasses,
    HeadPose,
    Illumination,
    Occlusion,
)
from shared.frame_features import FrameFeatures
from shared.geometry import (
    CLOSURE_HALF_LOW,
    is_closed,
)
from shared.schema import (
    AttentionObs,
    EmotionObs,
    EventHints,
    EyeStateObs,
    FatigueObs,
    HeadPoseObs,
    Observations,
    PrivacyAttestation,
    Quality,
    WindowMeta,
    WindowState,
)

from ..metrics.attention import AttentionTracker
from ..metrics.expression import BlendshapeExpressionClassifier, NeutralCalibrator
from ..metrics.eye import PERCLOS_WINDOW_SEC, EyeTracker, fuse_or_raw
from ..metrics.fatigue import assess as assess_fatigue
from ..metrics.head_pose import HeadPoseEstimator

#: 各条件在跨窗口累计器里的键名。
K_EMOTION_LOW = "emotion_low"      # 难过 / 烦闷
K_EMOTION_TIRED = "emotion_tired"
K_HEAD_TILTED = "head_tilted"
K_HEAD_BOWED = "head_bowed"
K_ATTENTION_ABSENT = "attention_absent"
K_EYE_CLOSED = "eye_closed"
K_FATIGUE_SEVERE = "fatigue_severe"


class StableDurationTracker:
    """跨窗口累计的条件持续时长。

    与 :class:`~module_b_agent.gate.consecutive.ConsecutiveCounter` 的区别：
    前者计**秒数**，后者计**窗口数**。这里要秒数是因为
    "歪倒 + 闭眼 ≥ 15 秒"是一个时间量，而窗口是 10 秒——两者不是整数倍关系。
    """

    def __init__(self) -> None:
        self._start: dict[str, float] = {}

    def update(self, key: str, active: bool, now_ts: float) -> float:
        """推进一个窗口，返回该条件已持续的秒数。

        条件不成立时清零并返回 0——刻意不留"记忆"，
        因为断续出现的异常不是持续性异常。
        """
        if active:
            if key not in self._start:
                self._start[key] = now_ts
            return max(0.0, now_ts - self._start[key])
        self._start.pop(key, None)
        return 0.0

    def snapshot(self) -> dict[str, float]:
        return dict(self._start)


@dataclass
class AggregatorConfig:
    """聚合器可调参数。"""

    window_sec: float = 10.0
    #: 窗口内占比门槛 ρ（设计方案 §4.2）。
    min_frames_ratio: float = 0.60
    #: 各条件的窗口内最短持续（T_min）。
    min_emotion_low_sec: float = 4.0
    min_emotion_tired_sec: float = 6.0
    min_tilted_sec: float = 3.0
    min_bowed_sec: float = 8.0
    min_absent_sec: float = 6.0
    min_closed_sec: float = 5.0
    #: 帧有效率低于此值时整个窗口不可用。
    min_valid_ratio: float = 0.30
    #: 老花镜带来的置信度惩罚（设计方案 §4.9）。
    glasses_penalty: float = 0.15
    device_id: str = "cam-livingroom-01"
    elder_id: str = "E1001"
    model_name: str = "elder-face-mtl"
    model_version: str = "1.0.3"


@dataclass
class WindowBuffer:
    """当前窗口内累积的帧。"""

    frames: list[FrameFeatures] = field(default_factory=list)
    started_ts: float = 0.0


class WindowAggregator:
    """把逐帧特征聚合成 :class:`WindowState`。

    生命周期与采集进程一致：内部的有状态组件（眼部、注意力、
    持续时间）都需要跨窗口存活。
    """

    def __init__(self, cfg: AggregatorConfig | None = None) -> None:
        self.cfg = cfg or AggregatorConfig()

        self._calibrator = NeutralCalibrator()
        self._expression = BlendshapeExpressionClassifier(self._calibrator)
        self._eyes = EyeTracker(perclos_window_sec=PERCLOS_WINDOW_SEC)
        self._attention = AttentionTracker()
        self._head_pose = HeadPoseEstimator()
        self._stable = StableDurationTracker()

        self._buffer = WindowBuffer()
        self._window_index = 0
        self._last_ts = 0.0

    # ------------------------------------------------------------ 摄入

    def add_frame(self, frame: FrameFeatures) -> None:
        """摄入一帧。"""
        if not self._buffer.frames:
            self._buffer.started_ts = frame.ts
        self._buffer.frames.append(frame)
        self._last_ts = frame.ts

        if frame.has_face and frame.quality.valid:
            self._calibrator.feed(frame)
            self._eyes.update(frame, fuse_or_raw(frame))
            self._attention.update(frame)

    @property
    def window_open(self) -> bool:
        return bool(self._buffer.frames)

    def window_elapsed(self, now_ts: float) -> float:
        if not self._buffer.frames:
            return 0.0
        return now_ts - self._buffer.started_ts

    def should_close(self, now_ts: float) -> bool:
        """是否到了关窗时刻。"""
        return self.window_open and self.window_elapsed(now_ts) >= self.cfg.window_sec

    @property
    def calibrator(self) -> NeutralCalibrator:
        return self._calibrator

    @property
    def blink_total(self) -> int:
        """**进程内累计**的眨眼次数，只增不减。api_doc §3.2 的 ``blink_cnt``。

        转发给 :attr:`EyeTracker.blink_total`。之所以在这里开一个口子而不是
        让调用方自己去摸 ``self._eyes``：``_eyes`` 是私有实现细节，
        而"累计眨眼次数"是聚合器对外的语义之一。
        """
        return self._eyes.blink_total

    def classify_frame(self, frame: FrameFeatures) -> Emotion:
        """对**单帧**做表情分类，返回主标签。

        ``BlendshapeExpressionClassifier.classify`` 是纯读（只读标定器状态，
        不推进它），所以逐帧多调一次不会影响 ``close_window()`` 里的窗口级投票。

        存在的理由：api_doc §3.2 的 ``emo_feature`` 是**逐帧**的，而窗口级
        分类在 :meth:`_aggregate_observations` 里，那时帧已经被丢弃了。
        分类需要 :class:`NeutralCalibrator` 的标定状态，标定器归本类所有，
        所以外面拿不到 —— 这个方法是那个缺口。

        调用时机需在 :meth:`add_frame` 之后：本帧若恰好完成一次眨眼，
        :attr:`blink_total` 是在 ``add_frame`` 里自增的。
        """
        emotion, _confidence, _dist = self._expression.classify(frame)
        return emotion

    # ------------------------------------------------------------ 关窗

    def close_window(self, now_ts: float | None = None) -> WindowState:
        """关闭当前窗口并产出报文。"""
        now_ts = now_ts if now_ts is not None else self._last_ts
        frames = self._buffer.frames
        started = self._buffer.started_ts

        self._buffer = WindowBuffer()
        self._window_index += 1

        quality = self._aggregate_quality(frames)
        observations, hints = self._aggregate_observations(frames, now_ts, quality)

        # 每个观测项都要带上跨窗口累计的 stable_sec。
        observations = self._apply_stable_durations(observations, now_ts)

        return WindowState(
            trace_id=str(uuid.uuid4()),
            device_id=self.cfg.device_id,
            elder_id=self.cfg.elder_id,
            timestamp=now_ts,
            has_face=any(f.has_face for f in frames),
            window=WindowMeta(
                start_ts=_iso(started),
                end_ts=_iso(now_ts),
                duration_sec=max(0.0, now_ts - started) if frames else 0.0,
                frames_total=len(frames),
                frames_valid=sum(
                    1 for f in frames if f.has_face and f.quality.valid
                ),
                fps_effective=(
                    len(frames) / (now_ts - started)
                    if frames and now_ts > started
                    else 0.0
                ),
            ),
            quality=quality,
            observations=observations,
            event_hints=hints,
            privacy=PrivacyAttestation(
                raw_frame_uploaded=False,
                face_image_uploaded=False,
                on_device_processing=True,
                frame_retention="memory_only",
            ),
            model_name=self.cfg.model_name,
            model_version=self.cfg.model_version,
        )

    # ------------------------------------------------------------ 质量

    def _aggregate_quality(self, frames: list[FrameFeatures]) -> Quality:
        n = len(frames)
        if not n:
            return Quality(usable=False, face_found_ratio=0.0)

        face_found = sum(1 for f in frames if f.has_face) / n
        valid = sum(1 for f in frames if f.has_face and f.quality.valid) / n

        # 取窗口内的主导质量标记（出现次数最多者）。
        def dominant(attr: str, default):
            counts: dict = {}
            for f in frames:
                v = getattr(f.quality, attr)
                counts[v] = counts.get(v, 0) + 1
            return max(counts, key=lambda k: counts[k]) if counts else default

        illumination = dominant("illumination", Illumination.NORMAL)
        occlusion = dominant("occlusion", Occlusion.NONE)
        glasses = dominant("glasses", Glasses.NONE)

        penalty = 0.0
        if glasses != Glasses.NONE:
            penalty += self.cfg.glasses_penalty

        # 不可用的判定（设计方案 §4.9）：
        # 画面全黑 / 严重遮挡 / 有效帧太少 → 整个窗口丢弃。
        usable = True
        if illumination == Illumination.LOW:
            usable = False
        if occlusion == Occlusion.SEVERE:
            usable = False
        if valid < self.cfg.min_valid_ratio:
            usable = False

        return Quality(
            face_found_ratio=face_found,
            illumination=illumination,
            occlusion=occlusion,
            glasses=glasses,
            confidence_penalty=penalty,
            usable=usable,
        )

    # ------------------------------------------------------------ 观测

    def _aggregate_observations(
        self, frames: list[FrameFeatures], now_ts: float, quality: Quality
    ) -> tuple[Observations, EventHints]:
        valid_frames = [f for f in frames if f.has_face and f.quality.valid]
        n_valid = len(valid_frames)

        if not n_valid:
            return Observations(), EventHints()

        # ---- 表情：逐帧分类后投票 ----
        emo_votes: dict[str, int] = {}
        emo_conf_sum: dict[str, float] = {}
        dist_sum = {str(e): 0.0 for e in Emotion}
        for f in valid_frames:
            label, conf, dist = self._expression.classify(f)
            key = str(label)
            emo_votes[key] = emo_votes.get(key, 0) + 1
            emo_conf_sum[key] = emo_conf_sum.get(key, 0.0) + conf
            for k, v in dist.items():
                dist_sum[k] = dist_sum.get(k, 0.0) + v

        emo_label = max(emo_votes, key=lambda k: emo_votes[k])
        emo_ratio = emo_votes[emo_label] / n_valid
        emo_conf = emo_conf_sum[emo_label] / emo_votes[emo_label]
        emo_dist = {k: v / n_valid for k, v in dist_sum.items()}

        # ---- 头姿：投票 ----
        hp_votes: dict[HeadPose, int] = {}
        hp_conf: dict[HeadPose, float] = {}
        hp_angles: dict[HeadPose, list[tuple[float, float, float]]] = {}
        for f in valid_frames:
            pose, conf, angles = self._head_pose.estimate(f)
            hp_votes[pose] = hp_votes.get(pose, 0) + 1
            hp_conf[pose] = hp_conf.get(pose, 0.0) + conf
            hp_angles.setdefault(pose, []).append(angles)

        hp_label = max(hp_votes, key=lambda k: hp_votes[k])
        hp_ratio = hp_votes[hp_label] / n_valid
        hp_c = hp_conf[hp_label] / hp_votes[hp_label]
        pitch, yaw, roll = _mean_angles(hp_angles[hp_label])

        # ---- 眼部 ----
        eye_metrics = self._eyes.snapshot(frames, now_ts)

        # ---- 注意力 ----
        att_metrics = self._attention.snapshot(frames, now_ts)

        # ---- 疲劳 ----
        fatigue = assess_fatigue(
            perclos=eye_metrics.perclos,
            p_tired=emo_dist.get(str(Emotion.TIRED), 0.0),
            long_closure_sec=eye_metrics.long_closure_sec,
            p_focused=(
                att_metrics.confidence
                if att_metrics.label == Attention.FOCUSED
                else 1.0 - att_metrics.confidence
            ),
            blink_rate_per_min=eye_metrics.blink_rate_per_min,
        )

        observations = Observations(
            emotion=EmotionObs(
                label=Emotion(emo_label),
                confidence=emo_conf,
                distribution=emo_dist,
                stable_sec=0.0,  # 由 _apply_stable_durations 填
                frames_ratio=emo_ratio,
            ),
            head_pose=HeadPoseObs(
                label=hp_label,
                confidence=hp_c,
                pitch_deg=pitch,
                yaw_deg=yaw,
                roll_deg=roll,
                stable_sec=0.0,
                frames_ratio=hp_ratio,
            ),
            attention=AttentionObs(
                label=att_metrics.label,
                confidence=att_metrics.confidence,
                gaze_off_sec=att_metrics.gaze_off_sec,
                stable_sec=att_metrics.stable_sec,
                frames_ratio=att_metrics.frames_ratio,
            ),
            fatigue=FatigueObs(
                level=fatigue.level,
                score=fatigue.score,
                confidence=fatigue.confidence,
                perclos=eye_metrics.perclos,
                perclos_window_sec=PERCLOS_WINDOW_SEC,
                blink_rate_per_min=eye_metrics.blink_rate_per_min,
                avg_closure_ratio=eye_metrics.avg_closure_ratio,
                long_closure_sec=eye_metrics.long_closure_sec,
                stable_sec=0.0,
            ),
            eye_state=EyeStateObs(
                label=_eye_label(eye_metrics.closure_ratio),
                confidence=_eye_confidence(eye_metrics, n_valid),
                closure_ratio=eye_metrics.closure_ratio,
                blink_rate_per_min=eye_metrics.blink_rate_per_min,
                long_closure_sec=eye_metrics.long_closure_sec,
                stable_sec=0.0,
            ),
        )

        hints = self._build_hints(observations, eye_metrics, n_valid)
        return observations, hints

    def _apply_stable_durations(
        self, obs: Observations, now_ts: float
    ) -> Observations:
        """用跨窗口累计器填 ``stable_sec``。

        注意这里用的是**条件是否成立**，而不是"标签是否等于某值"——
        因为"标签=难过"但本窗口只占了 30% 的帧时，不该继续累加持续时长。
        """
        emo_low = obs.emotion.label in (Emotion.SAD, Emotion.UPSET) and (
            obs.emotion.frames_ratio >= self.cfg.min_frames_ratio
        )
        emo_tired = obs.emotion.label == Emotion.TIRED and (
            obs.emotion.frames_ratio >= self.cfg.min_frames_ratio
        )
        head_tilted = obs.head_pose.label == HeadPose.TILTED and (
            obs.head_pose.frames_ratio >= 0.50
        )
        head_bowed = obs.head_pose.label == HeadPose.BOWED and (
            obs.head_pose.frames_ratio >= self.cfg.min_frames_ratio
        )
        absent = obs.attention.label == Attention.ABSENT and (
            obs.attention.frames_ratio >= self.cfg.min_frames_ratio
        )
        closed = obs.eye_state.label == EyeState.CLOSED and (
            obs.eye_state.long_closure_sec >= self.cfg.min_closed_sec
        )
        severe = obs.fatigue.level == FatigueLevel.SEVERE and (
            obs.fatigue.confidence >= 0.0
        )

        s = self._stable
        d_emo_low = s.update(K_EMOTION_LOW, emo_low, now_ts)
        d_emo_tired = s.update(K_EMOTION_TIRED, emo_tired, now_ts)
        d_tilted = s.update(K_HEAD_TILTED, head_tilted, now_ts)
        d_bowed = s.update(K_HEAD_BOWED, head_bowed, now_ts)
        d_absent = s.update(K_ATTENTION_ABSENT, absent, now_ts)
        d_closed = s.update(K_EYE_CLOSED, closed, now_ts)
        d_severe = s.update(K_FATIGUE_SEVERE, severe, now_ts)

        return Observations(
            emotion=_replace(obs.emotion, stable_sec=d_emo_low if emo_low else d_emo_tired),
            head_pose=_replace(
                obs.head_pose,
                stable_sec=d_tilted if obs.head_pose.label == HeadPose.TILTED else d_bowed,
            ),
            attention=_replace(obs.attention, stable_sec=d_absent or obs.attention.stable_sec),
            fatigue=_replace(obs.fatigue, stable_sec=d_severe),
            eye_state=_replace(obs.eye_state, stable_sec=d_closed),
        )

    def _build_hints(
        self, obs: Observations, eye_metrics, n_valid: int
    ) -> EventHints:
        """产出**建议性**初判。

        ⚠️ B 的规则闸门**永不读取**本结构。它只用于日志、看板与人工排查。
        让闸门读它等于让 A 替 B 做判定，破坏分层（设计方案 §1.3）。
        """
        low_mood = obs.emotion.label in (Emotion.SAD, Emotion.UPSET)
        drowsy = obs.fatigue.level == FatigueLevel.SEVERE or (
            obs.eye_state.long_closure_sec >= self.cfg.min_closed_sec
        )
        # 联合条件：歪倒 **且** 持续闭眼。刻意不用任何单条件。
        body_abnormal = (
            obs.head_pose.label == HeadPose.TILTED
            and obs.eye_state.label == EyeState.CLOSED
        )

        return EventHints(
            consecutive_abnormal_windows=0,  # 跨窗口计数由 B 负责
            suspected_low_mood=low_mood,
            suspected_drowsy=drowsy,
            suspected_body_abnormal=body_abnormal,
            hint_confidence=min(
                1.0,
                (obs.emotion.confidence + obs.head_pose.confidence
                 + obs.eye_state.confidence) / 3.0,
            ),
        )


# ================================================================ 工具

def _replace(obs, **changes):
    """dataclass replace，保持 frozen 语义。"""
    from dataclasses import replace

    return replace(obs, **changes)


def _mean_angles(angles: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not angles:
        return 0.0, 0.0, 0.0
    n = len(angles)
    return (
        sum(a[0] for a in angles) / n,
        sum(a[1] for a in angles) / n,
        sum(a[2] for a in angles) / n,
    )


def _eye_label(closure_ratio: float) -> EyeState:
    from shared.geometry import classify_eye_state

    return classify_eye_state(closure_ratio)


def _eye_confidence(eye_metrics, n_valid: int) -> float:
    """眼部置信度。

    由"闭眼程度离阈值的距离"与"样本量"共同决定：样本太少时不给高置信度。
    """
    sample_factor = min(n_valid / 30.0, 1.0)
    closure = eye_metrics.closure_ratio
    if is_closed(closure):
        base = 0.70 + 0.25 * min((closure - 0.80) / 0.20, 1.0)
    elif closure >= CLOSURE_HALF_LOW:
        base = 0.60 + 0.15 * min((closure - 0.55) / 0.25, 1.0)
    else:
        base = 0.65 + 0.20 * min((0.55 - closure) / 0.55, 1.0)
    return max(0.0, min(base * (0.5 + 0.5 * sample_factor), 0.95))


def _iso(ts: float) -> str:
    """把单调秒转成可读的墙钟字符串（仅供人工排查）。"""
    if ts <= 0:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
