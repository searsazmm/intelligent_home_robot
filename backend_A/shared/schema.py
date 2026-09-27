"""模块间传输的数据结构。

这些结构**就是**线路协议，字段名与《系统设计方案》§3.2 / §7.4 / §7.5 严格一致。

两条贯穿始终的约定：

1. **前向兼容**：``from_dict`` 对缺失字段一律取默认值、对未知字段一律忽略
   （《系统总接口文档》通用约定）。这样 A 升级到 V2 后，旧版 B 仍能解析；
   反之 B 也能吞下 V1 的 8 字段报文。
2. **不变量内建**：``WindowState`` 携带 :class:`PrivacyAttestation`，
   B 侧收到后必须校验 ``raw_frame_uploaded`` 与 ``face_image_uploaded``
   均为 ``False``，否则触发 E5001 阻断。隐私不能只靠约定，要有可校验的字段。

关于 ``stable_sec`` 的语义（**这条曾是一处设计缺陷，务必遵守**）
------------------------------------------------------------------

``stable_sec`` 是**跨窗口累计**的持续时长，**允许大于 ``window.duration_sec``**。

早期设计里它被读作"本窗口内该状态持续的秒数"，那会导致
"头部歪倒 + 持续闭眼 ≥ 15 秒"这条最高危规则**在数学上不可观测**——
窗口只有 10 秒，累计量永远到不了 15。修复方式是把 ``stable_sec``
定义为从该状态**首次成立**起累计的时长，由 A 跨窗口维护。

B 侧因此不能只读 A 的标签，必须自己复核几何条件并跨窗口计数
（见 :mod:`module_b_agent.gate.rules`）。``event_hints`` 是**纯建议**，
闸门永不读取——否则等于让 A 替 B 做判定，破坏 §1.3 的分层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import (
    Attention,
    Blur,
    Emotion,
    EyeState,
    FatigueLevel,
    Glasses,
    HeadPose,
    Illumination,
    Occlusion,
)

SCHEMA_VERSION = "2.0"


def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    """取浮点字段，容忍 None 与字符串数字。"""
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(d: dict[str, Any], key: str, default: int = 0) -> int:
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _s(d: dict[str, Any], key: str, default: str = "") -> str:
    v = d.get(key, default)
    return default if v is None else str(v)


def _b(d: dict[str, Any], key: str, default: bool = False) -> bool:
    v = d.get(key, default)
    return default if v is None else bool(v)


def _sub(d: dict[str, Any], key: str) -> dict[str, Any]:
    """取嵌套对象，缺失或类型不对时返回空字典。"""
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def _enum(enum_cls, value: Any, default):
    """按值构造枚举，非法值回落到默认——线路上出现脏数据不应让进程崩溃。"""
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        return default


# ================================================================ 窗口元信息

@dataclass(frozen=True, slots=True)
class WindowMeta:
    """本窗口的统计口径。B 用它判断样本是否充足。"""

    start_ts: str = ""
    end_ts: str = ""
    duration_sec: float = 10.0
    frames_total: int = 0
    frames_valid: int = 0
    fps_effective: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_sec": self.duration_sec,
            "frames_total": self.frames_total,
            "frames_valid": self.frames_valid,
            "fps_effective": self.fps_effective,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowMeta:
        return cls(
            start_ts=_s(d, "start_ts"),
            end_ts=_s(d, "end_ts"),
            duration_sec=_f(d, "duration_sec", 10.0),
            frames_total=_i(d, "frames_total"),
            frames_valid=_i(d, "frames_valid"),
            fps_effective=_f(d, "fps_effective"),
        )

    @property
    def valid_ratio(self) -> float:
        """有效帧占比。"""
        return self.frames_valid / self.frames_total if self.frames_total else 0.0


@dataclass(frozen=True, slots=True)
class Quality:
    """画面质量。``usable=False`` 时 B 必须整个丢弃本窗口。"""

    face_found_ratio: float = 0.0
    illumination: Illumination = Illumination.NORMAL
    blur: Blur = Blur.LOW
    occlusion: Occlusion = Occlusion.NONE
    glasses: Glasses = Glasses.NONE
    #: 因眼镜、逆光等导致的置信度惩罚，B 用它抬高判定门槛（设计方案 §4.1）。
    confidence_penalty: float = 0.0
    usable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_found_ratio": self.face_found_ratio,
            "illumination": str(self.illumination),
            "blur": str(self.blur),
            "occlusion": str(self.occlusion),
            "glasses": str(self.glasses),
            "confidence_penalty": self.confidence_penalty,
            "usable": self.usable,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Quality:
        return cls(
            face_found_ratio=_f(d, "face_found_ratio"),
            illumination=_enum(Illumination, d.get("illumination"), Illumination.NORMAL),
            blur=_enum(Blur, d.get("blur"), Blur.LOW),
            occlusion=_enum(Occlusion, d.get("occlusion"), Occlusion.NONE),
            glasses=_enum(Glasses, d.get("glasses"), Glasses.NONE),
            confidence_penalty=_f(d, "confidence_penalty"),
            usable=_b(d, "usable", True),
        )


# ================================================================ 五项观测

@dataclass(frozen=True, slots=True)
class EmotionObs:
    label: Emotion = Emotion.NORMAL
    confidence: float = 0.0
    distribution: dict[str, float] = field(default_factory=dict)
    stable_sec: float = 0.0
    frames_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "confidence": self.confidence,
            "distribution": dict(self.distribution),
            "stable_sec": self.stable_sec,
            "frames_ratio": self.frames_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmotionObs:
        dist_raw = _sub(d, "distribution")
        return cls(
            label=_enum(Emotion, d.get("label"), Emotion.NORMAL),
            confidence=_f(d, "confidence"),
            distribution={k: _f(dist_raw, k) for k in dist_raw},
            stable_sec=_f(d, "stable_sec"),
            frames_ratio=_f(d, "frames_ratio"),
        )


@dataclass(frozen=True, slots=True)
class HeadPoseObs:
    label: HeadPose = HeadPose.UPRIGHT
    confidence: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    stable_sec: float = 0.0
    frames_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "confidence": self.confidence,
            "angles": {
                "pitch_deg": self.pitch_deg,
                "yaw_deg": self.yaw_deg,
                "roll_deg": self.roll_deg,
            },
            "stable_sec": self.stable_sec,
            "frames_ratio": self.frames_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HeadPoseObs:
        angles = _sub(d, "angles")
        return cls(
            label=_enum(HeadPose, d.get("label"), HeadPose.UPRIGHT),
            confidence=_f(d, "confidence"),
            pitch_deg=_f(angles, "pitch_deg"),
            yaw_deg=_f(angles, "yaw_deg"),
            roll_deg=_f(angles, "roll_deg"),
            stable_sec=_f(d, "stable_sec"),
            frames_ratio=_f(d, "frames_ratio"),
        )


@dataclass(frozen=True, slots=True)
class AttentionObs:
    label: Attention = Attention.FOCUSED
    confidence: float = 0.0
    gaze_off_sec: float = 0.0
    stable_sec: float = 0.0
    frames_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "confidence": self.confidence,
            "gaze_off_sec": self.gaze_off_sec,
            "stable_sec": self.stable_sec,
            "frames_ratio": self.frames_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttentionObs:
        return cls(
            label=_enum(Attention, d.get("label"), Attention.FOCUSED),
            confidence=_f(d, "confidence"),
            gaze_off_sec=_f(d, "gaze_off_sec"),
            stable_sec=_f(d, "stable_sec"),
            frames_ratio=_f(d, "frames_ratio"),
        )


@dataclass(frozen=True, slots=True)
class FatigueObs:
    """疲劳观测。

    ``perclos`` 必须在 **60 秒滚动窗**上计算，而不是在 10 秒报文窗口上。
    PERCLOS 本是一个 60–180 秒量级的指标；在 10 秒窗上算，只要老人闭眼
    几秒就会逼近 1.0，于是 ``min(perclos/0.40, 1)`` 项迅速饱和，
    既让 ``mild`` 档几乎不可达，又与眼部规则重复计分。
    ``perclos_window_sec`` 字段把这件事显式化，避免上下游各按各的理解实现。
    """

    level: FatigueLevel = FatigueLevel.NONE
    score: float = 0.0
    confidence: float = 0.0
    perclos: float = 0.0
    #: PERCLOS 的积分窗长（秒）。默认 60，与报文窗口（10s）不同。
    perclos_window_sec: float = 60.0
    blink_rate_per_min: float = 0.0
    avg_closure_ratio: float = 0.0
    long_closure_sec: float = 0.0
    stable_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": str(self.level),
            "score": self.score,
            "confidence": self.confidence,
            "metrics": {
                "perclos": self.perclos,
                "perclos_window_sec": self.perclos_window_sec,
                "blink_rate_per_min": self.blink_rate_per_min,
                "avg_closure_ratio": self.avg_closure_ratio,
                "long_closure_sec": self.long_closure_sec,
            },
            "stable_sec": self.stable_sec,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FatigueObs:
        m = _sub(d, "metrics")
        return cls(
            level=_enum(FatigueLevel, d.get("level"), FatigueLevel.NONE),
            score=_f(d, "score"),
            confidence=_f(d, "confidence"),
            perclos=_f(m, "perclos"),
            perclos_window_sec=_f(m, "perclos_window_sec", 60.0),
            blink_rate_per_min=_f(m, "blink_rate_per_min"),
            avg_closure_ratio=_f(m, "avg_closure_ratio"),
            long_closure_sec=_f(m, "long_closure_sec"),
            stable_sec=_f(d, "stable_sec"),
        )


@dataclass(frozen=True, slots=True)
class EyeStateObs:
    label: EyeState = EyeState.NORMAL_BLINK
    confidence: float = 0.0
    closure_ratio: float = 0.0
    blink_rate_per_min: float = 0.0
    long_closure_sec: float = 0.0
    stable_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "confidence": self.confidence,
            "closure_ratio": self.closure_ratio,
            "blink_rate_per_min": self.blink_rate_per_min,
            "long_closure_sec": self.long_closure_sec,
            "stable_sec": self.stable_sec,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EyeStateObs:
        return cls(
            label=_enum(EyeState, d.get("label"), EyeState.NORMAL_BLINK),
            confidence=_f(d, "confidence"),
            closure_ratio=_f(d, "closure_ratio"),
            blink_rate_per_min=_f(d, "blink_rate_per_min"),
            long_closure_sec=_f(d, "long_closure_sec"),
            stable_sec=_f(d, "stable_sec"),
        )


@dataclass(frozen=True, slots=True)
class Observations:
    """五项检测结果的集合。任一子项缺失时用默认值，不做 None 传播。"""

    emotion: EmotionObs = field(default_factory=EmotionObs)
    head_pose: HeadPoseObs = field(default_factory=HeadPoseObs)
    attention: AttentionObs = field(default_factory=AttentionObs)
    fatigue: FatigueObs = field(default_factory=FatigueObs)
    eye_state: EyeStateObs = field(default_factory=EyeStateObs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion.to_dict(),
            "head_pose": self.head_pose.to_dict(),
            "attention": self.attention.to_dict(),
            "fatigue": self.fatigue.to_dict(),
            "eye_state": self.eye_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observations:
        return cls(
            emotion=EmotionObs.from_dict(_sub(d, "emotion")),
            head_pose=HeadPoseObs.from_dict(_sub(d, "head_pose")),
            attention=AttentionObs.from_dict(_sub(d, "attention")),
            fatigue=FatigueObs.from_dict(_sub(d, "fatigue")),
            eye_state=EyeStateObs.from_dict(_sub(d, "eye_state")),
        )


@dataclass(frozen=True, slots=True)
class EventHints:
    """A 模块的**本地初判建议**，不是结论。B 可采纳也可推翻。"""

    consecutive_abnormal_windows: int = 0
    suspected_low_mood: bool = False
    suspected_drowsy: bool = False
    suspected_body_abnormal: bool = False
    hint_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "consecutive_abnormal_windows": self.consecutive_abnormal_windows,
            "suspected_low_mood": self.suspected_low_mood,
            "suspected_drowsy": self.suspected_drowsy,
            "suspected_body_abnormal": self.suspected_body_abnormal,
            "hint_confidence": self.hint_confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventHints:
        return cls(
            consecutive_abnormal_windows=_i(d, "consecutive_abnormal_windows"),
            suspected_low_mood=_b(d, "suspected_low_mood"),
            suspected_drowsy=_b(d, "suspected_drowsy"),
            suspected_body_abnormal=_b(d, "suspected_body_abnormal"),
            hint_confidence=_f(d, "hint_confidence"),
        )


@dataclass(frozen=True, slots=True)
class PrivacyAttestation:
    """隐私自证。B 侧强制校验，违反即 E5001 阻断。"""

    raw_frame_uploaded: bool = False
    face_image_uploaded: bool = False
    on_device_processing: bool = True
    frame_retention: str = "memory_only"
    local_face_template_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_frame_uploaded": self.raw_frame_uploaded,
            "face_image_uploaded": self.face_image_uploaded,
            "on_device_processing": self.on_device_processing,
            "frame_retention": self.frame_retention,
            "local_face_template_hash": self.local_face_template_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PrivacyAttestation:
        return cls(
            raw_frame_uploaded=_b(d, "raw_frame_uploaded"),
            face_image_uploaded=_b(d, "face_image_uploaded"),
            on_device_processing=_b(d, "on_device_processing", True),
            frame_retention=_s(d, "frame_retention", "memory_only"),
            local_face_template_hash=_s(d, "local_face_template_hash"),
        )

    @property
    def is_clean(self) -> bool:
        """是否满足"图像不出模块"的红线。"""
        return not self.raw_frame_uploaded and not self.face_image_uploaded


# ================================================================ 顶层报文

@dataclass(frozen=True, slots=True)
class WindowState:
    """A→B 的窗口观测报文——设计上**唯一**跨模块传输的视觉数据。

    刻意不包含 bbox、关键点、图像等任何可还原人像的字段。
    参见 :mod:`module_a_vision.privacy.guard` 中的出站断言。
    """

    trace_id: str = ""
    device_id: str = ""
    elder_id: str = ""
    timestamp: float = 0.0
    has_face: bool = False
    window: WindowMeta = field(default_factory=WindowMeta)
    quality: Quality = field(default_factory=Quality)
    observations: Observations = field(default_factory=Observations)
    event_hints: EventHints = field(default_factory=EventHints)
    privacy: PrivacyAttestation = field(default_factory=PrivacyAttestation)
    schema_version: str = SCHEMA_VERSION
    model_name: str = ""
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": "vision_window",
            "trace_id": self.trace_id,
            "device_id": self.device_id,
            "elder_id": self.elder_id,
            "timestamp": self.timestamp,
            "has_face": self.has_face,
            "window": self.window.to_dict(),
            "quality": self.quality.to_dict(),
            "observations": self.observations.to_dict(),
            "event_hints": self.event_hints.to_dict(),
            "privacy": self.privacy.to_dict(),
            "model": {"name": self.model_name, "version": self.model_version},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowState:
        model = _sub(d, "model")
        return cls(
            trace_id=_s(d, "trace_id"),
            device_id=_s(d, "device_id"),
            elder_id=_s(d, "elder_id"),
            timestamp=_f(d, "timestamp"),
            has_face=_b(d, "has_face"),
            window=WindowMeta.from_dict(_sub(d, "window")),
            quality=Quality.from_dict(_sub(d, "quality")),
            observations=Observations.from_dict(_sub(d, "observations")),
            event_hints=EventHints.from_dict(_sub(d, "event_hints")),
            privacy=PrivacyAttestation.from_dict(_sub(d, "privacy")),
            schema_version=_s(d, "schema_version", SCHEMA_VERSION),
            model_name=_s(model, "name"),
            model_version=_s(model, "version"),
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """B 的决策输出。字段与设计方案 §6.2 的输出 schema 一一对应。"""

    trigger_level: str = "L0"
    act: bool = False
    scene: str = "none"
    urgency: str = "low"
    intent: str = ""
    tone_hint: str = "warm"
    max_sentences: int = 2
    max_chars: int = 45
    wait_reply_sec: int = 30
    max_retry: int = 0
    retry_interval_sec: int = 60
    escalate_on_no_reply: bool = False
    family_push: bool = False
    family_push_reason: str | None = None
    family_hint: str | None = None
    cooldown_min: int = 30
    suppress_reason: str | None = None
    escalate_reason: str | None = None
    confidence: float = 0.0
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_level": self.trigger_level,
            "act": self.act,
            "scene": self.scene,
            "urgency": self.urgency,
            "intent": self.intent,
            "tone_hint": self.tone_hint,
            "max_sentences": self.max_sentences,
            "max_chars": self.max_chars,
            "wait_reply_sec": self.wait_reply_sec,
            "retry_policy": {
                "max_retry": self.max_retry,
                "interval_sec": self.retry_interval_sec,
                "escalate_on_no_reply": self.escalate_on_no_reply,
            },
            "family_push": self.family_push,
            "family_push_reason": self.family_push_reason,
            "family_hint": self.family_hint,
            "cooldown_min": self.cooldown_min,
            "suppress_reason": self.suppress_reason,
            "escalate_reason": self.escalate_reason,
            "confidence": self.confidence,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Decision:
        retry = _sub(d, "retry_policy")
        return cls(
            trigger_level=_s(d, "trigger_level", "L0"),
            act=_b(d, "act"),
            scene=_s(d, "scene", "none"),
            urgency=_s(d, "urgency", "low"),
            intent=_s(d, "intent"),
            tone_hint=_s(d, "tone_hint", "warm"),
            max_sentences=_i(d, "max_sentences", 2),
            max_chars=_i(d, "max_chars", 45),
            wait_reply_sec=_i(d, "wait_reply_sec", 30),
            max_retry=_i(retry, "max_retry"),
            retry_interval_sec=_i(retry, "interval_sec", 60),
            escalate_on_no_reply=_b(retry, "escalate_on_no_reply"),
            family_push=_b(d, "family_push"),
            family_push_reason=d.get("family_push_reason"),
            family_hint=d.get("family_hint"),
            cooldown_min=_i(d, "cooldown_min", 30),
            suppress_reason=d.get("suppress_reason"),
            escalate_reason=d.get("escalate_reason"),
            confidence=_f(d, "confidence"),
            trace_id=_s(d, "trace_id"),
        )
