"""合成帧源——本机（无摄像头）的**主路径**。

它按**剧本**产出数据。剧本是一串 ``(状态, 持续秒数)``：

::

    [("normal", 30.0), ("drowsy", 40.0)]

同一份剧本有三条输出，由三个类分别提供：

* :class:`SyntheticFeatureSource`——直接产出
  :class:`~shared.frame_features.FrameFeatures`。**这是被测试的路径**：
  零第三方依赖（除 numpy）、完全确定性、微秒级。
* :class:`SyntheticSource`——在它之上再加一条像素路径（``read()``），
  保证真实管线的形状不因"本机没摄像头"而悄悄分叉。
* :class:`SyntheticPixelSource`——同时给出像素与特征，并**充当自己的人脸
  后端**。它是给展示流（``--source synthetic-pixels --stream``）用的：
  本机没有摄像头，但右栏那张图总得有点东西可看。它**不是**
  ``SyntheticSource`` 的子类，理由写在那个类自己的文档里（一句话：
  服务器"特征源优先"的派发会让像素永远拿不到）。

像素路径刻意**不**生成能被 MediaPipe 检出的人脸
------------------------------------------------

画一张可信的人脸需要写一大堆几何代码，而它恰好会把真正的风险
（聚合、投票、规则判定）漏掉——那是在测试几何代码，不是在测试策略。
所以 :meth:`SyntheticSource.read` 画出的只是**一张形状正确的玩具图**：
背景亮度跟着照度走、一个按 ``roll`` 旋转的椭圆、两只按闭眼程度开合的
眼睛。把它喂给真后端只会得到 ``has_face=False``，这是预期行为。

状态的物理含义
--------------

各状态的配方见 :data:`STATE_SPECS`，每条都写清了它想覆盖的下游路径。
最容易搞混的是 ``drowsy`` 与 ``nap``：

* ``drowsy``（坐着打瞌睡）——头**歪倒**（``|roll| ≥ 25°``）+ 持续闭眼。
  这是最高危规则 R6 的两个条件同时成立。
* ``nap``（躺下午睡）——闭眼 + 低头，但**不歪**。它跑起来必须**不**触发
  R6，否则系统会把正常的午睡报成"身体异常"。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from random import Random
from typing import Iterable, Mapping

import numpy as np

from shared.enums import Blur, Illumination, Occlusion
from shared.frame_features import (
    EXPECTED_BLENDSHAPES,
    FrameFeatures,
    FrameQuality,
)
from shared.geometry import fuse_closure

from .base import BaseCapture, BaseFeatureSource, CaptureConfig, CaptureError
from ..metrics.expression import NEUTRAL_CALIBRATION_SEC

#: 表示"一直持续下去"。剧本的最后一段常取它。
INF = float("inf")


def _bs(**kw: float) -> dict[str, float]:
    """构造一组 blendshape：期望名单**全部在场**，未指定的取 0。

    刻意补齐而不是只给几个名字。缺失的名字会让表情分类器读到 0，而
    "缺失"与"真的是 0"在下游完全无法区分——诊断这类问题非常费时
    （见 :func:`shared.frame_features.missing_blendshapes` 的说明）。
    """
    full = {name: 0.0 for name in EXPECTED_BLENDSHAPES}
    full.update(kw)
    return full


@dataclass(frozen=True, slots=True)
class StateSpec:
    """一个"剧本状态"的帧特征配方。

    眼睛的两个信号（EAR 与 ``eyeBlink`` 系数）成对给出，闭眼程度由
    :func:`shared.geometry.fuse_closure` **现算**，而不是硬写一个数字——
    这样合成数据与真后端的数据在同一个公式下产生，融合权重一改两边同时生效。
    """

    #: 本状态是否有人脸。``False`` 时其余眼部/表情参数无意义。
    has_face: bool = True

    #: 睁眼基线：EAR 与 eyeBlink 系数。
    ear_open: float = 0.29
    blink_open: float = 0.05
    #: 闭眼基线。
    ear_closed: float = 0.10
    blink_closed: float = 0.90

    #: ``True`` = 持续闭眼；``False`` = 睁眼 + 周期性眨眼。
    eyes_closed: bool = False
    #: 眨眼周期（秒）。4.0 秒 ≈ 15 次/分，正是
    #: :data:`module_a_vision.metrics.fatigue.NORMAL_BLINK_RATE` 的正常值。
    blink_period: float = 4.0
    #: 单次眨眼时长（秒）。必须 < ``metrics.eye.MAX_BLINK_SEC`` (0.4)，
    #: 否则会被算成"持续闭眼"而不是"眨眼"。
    blink_sec: float = 0.22

    #: 头部姿态（度）。符号约定与 :mod:`shared.geometry` 一致：
    #: pitch 正=低头，yaw 正=向右偏，roll 正=向右倾。
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    #: 视线偏离正前方的程度 [0,1]。
    gaze_off: float = 0.0

    blendshapes: dict[str, float] = field(default_factory=dict)
    quality: FrameQuality = FrameQuality()
    #: 这个状态想覆盖的下游路径，写给自己和后来人看。
    note: str = ""


# ================================================================ 状态配方

STATE_SPECS: dict[str, StateSpec] = {
    "normal": StateSpec(
        pitch=2.0, yaw=-3.0, roll=1.5, gaze_off=0.05,
        blendshapes=_bs(browInnerUp=0.05, mouthSmileLeft=0.12, mouthSmileRight=0.10),
        note="安静坐着：头端正、正常眨眼（约 15 次/分）、表情中性",
    ),
    "drowsy": StateSpec(
        eyes_closed=True,
        pitch=12.0, yaw=-2.0, roll=34.0, gaze_off=0.55,
        blendshapes=_bs(
            eyeSquintLeft=0.50, eyeSquintRight=0.50,
            jawOpen=0.35, browDownLeft=0.25, browDownRight=0.25,
            mouthPressLeft=0.20, mouthPressRight=0.20,
        ),
        note="坐着打瞌睡：头歪倒 34° + 持续闭眼 —— R6 的两个条件同时成立",
    ),
    "nap": StateSpec(
        eyes_closed=True, ear_closed=0.11, blink_closed=0.88,
        pitch=42.0, yaw=1.0, roll=18.0, gaze_off=0.60,
        blendshapes=_bs(eyeSquintLeft=0.30, eyeSquintRight=0.30, jawOpen=0.15),
        note="躺下午睡：闭眼 + 低头，但**不歪**（roll 18° < 25°）——必须**不**触发 R6",
    ),
    "sad": StateSpec(
        pitch=6.0, yaw=-4.0, roll=3.0, gaze_off=0.25,
        blendshapes=_bs(
            browInnerUp=0.90,
            mouthFrownLeft=0.75, mouthFrownRight=0.75,
        ),
        note="难过：眉心上挑 + 嘴角下拉，强度足以越过表情判定门槛",
    ),
    "upset": StateSpec(
        pitch=4.0, yaw=2.0, roll=-2.0, gaze_off=0.20,
        blendshapes=_bs(
            browDownLeft=0.90, browDownRight=0.90,
            mouthPressLeft=0.80, mouthPressRight=0.80,
            noseSneerLeft=0.60, noseSneerRight=0.60,
            mouthPucker=0.50,
        ),
        note="烦闷：眉毛下压 + 嘴唇紧抿，与难过共用四分类的另一支",
    ),
    "tired": StateSpec(
        # 眼睛半睁不闭：EAR 略低、eyeBlink 中等，落在"半闭眼"档。
        ear_open=0.19, blink_open=0.45,
        pitch=14.0, yaw=-3.0, roll=5.0, gaze_off=0.45,
        blendshapes=_bs(
            eyeSquintLeft=1.00, eyeSquintRight=1.00,
            eyeWideLeft=0.0, eyeWideRight=0.0,
            jawOpen=1.00,
            browDownLeft=1.00, browDownRight=1.00,
        ),
        note=(
            "疲惫：半闭眼 + 眯眼 + 张嘴（打哈欠）。注意表情分类器对疲惫通道"
            "有 TIRED_DAMPING=0.6 的阻尼（见 metrics/expression.py 的模块文档），"
            "**表情标签不会变成 tired**——本状态真正的作用是抬高疲劳评分里的"
            " P(tired) 项与半闭眼帧占比。"
        ),
    ),
    "absent": StateSpec(
        pitch=1.0, yaw=8.0, roll=2.0, gaze_off=0.90,
        blendshapes=_bs(mouthSmileLeft=0.05),
        note="发呆失神：视线长时间偏离正前方，其余一切正常",
    ),
    "occluded": StateSpec(
        has_face=False,
        quality=FrameQuality(
            illumination=Illumination.NORMAL,
            blur=Blur.HIGH,
            occlusion=Occlusion.SEVERE,
            valid=False,
        ),
        note="镜头被挡住（手掌/衣物）：画面本身不可用 —— 对应 L4 的 vision_unusable",
    ),
    "no_face": StateSpec(
        has_face=False,
        note="老人离开房间：画面本身是好的，只是没有人 —— 与 occluded 的成因完全不同",
    ),
    "dark": StateSpec(
        has_face=False,
        quality=FrameQuality(
            illumination=Illumination.LOW,
            blur=Blur.MEDIUM,
            occlusion=Occlusion.NONE,
            valid=False,
        ),
        note="房间里没开灯：照度过低，窗口应被整体判为不可用",
    ),
}


# ================================================================ 剧本

#: 剧本开头的"安静坐着"时长。
#:
#: ⚠️ **这不是随手取的数。** 表情分类器的个体标定
#: （:class:`~module_a_vision.metrics.expression.NeutralCalibrator`）需要
#: ``NEUTRAL_CALIBRATION_SEC`` 秒的安静样本作为中性基线。剧本的开场段
#: 如果短于它，标定就会在剧本**已经进入异常状态之后**才完成——把要检测的
#: 那个表情当成基线减掉。
#:
#: 这个坑踩过一次：开场 15 秒 < 标定所需 20 秒，于是 ``sad`` 与 ``upset``
#: 两个剧本全程判不出情绪、L1 关怀路径整条不可达，而表面现象是
#: "判定逻辑失灵"，很难联想到病根在剧本时长上。
#:
#: 所以这里**派生**而不是写死一个数字：标定要求变了，剧本自动跟着变。
NEUTRAL_LEAD_IN_SEC = NEUTRAL_CALIBRATION_SEC + 5.0

BUILTIN_SCENARIOS: dict[str, tuple[tuple[str, float], ...]] = {
    "normal": (("normal", INF),),
    "drowsy": (("normal", NEUTRAL_LEAD_IN_SEC), ("drowsy", 120.0), ("normal", 40.0)),
    "nap": (("normal", NEUTRAL_LEAD_IN_SEC), ("nap", 1800.0)),
    "sad": (("normal", NEUTRAL_LEAD_IN_SEC), ("sad", 180.0)),
    "upset": (("normal", NEUTRAL_LEAD_IN_SEC), ("upset", 180.0)),
    "tired": (("normal", NEUTRAL_LEAD_IN_SEC), ("tired", 180.0)),
    "absent": (("normal", NEUTRAL_LEAD_IN_SEC), ("absent", 180.0)),
    "occluded": (("normal", NEUTRAL_LEAD_IN_SEC), ("occluded", 120.0)),
    "no_face": (("normal", NEUTRAL_LEAD_IN_SEC), ("no_face", 300.0)),
    "dark": (("normal", NEUTRAL_LEAD_IN_SEC), ("dark", 300.0)),
}

# 派生关系一旦被破坏（有人把 NEUTRAL_LEAD_IN_SEC 改回一个写死的短值），
# 这里立刻炸——比等两个剧本安静地判不出情绪要早得多。
assert NEUTRAL_LEAD_IN_SEC > NEUTRAL_CALIBRATION_SEC, (
    "剧本开场必须长于表情标定所需时长，否则标定会吸收掉要检测的表情"
)


@dataclass(frozen=True, slots=True)
class Segment:
    """剧本里的一段。"""

    state: str
    duration_sec: float

    @property
    def indefinite(self) -> bool:
        """是否"一直持续下去"。"""
        return not math.isfinite(self.duration_sec)

    def label(self) -> str:
        return f"{self.state} ∞" if self.indefinite else f"{self.state} {self.duration_sec:g}s"


@dataclass(frozen=True, slots=True)
class Scenario:
    """一串时间段，按顺序走完；``loop=True`` 时走完再从头来。"""

    name: str
    segments: tuple[Segment, ...]
    loop: bool = True

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("剧本至少要有一段")

    @property
    def total_sec(self) -> float:
        """剧本总长；含无限段时为 ``inf``。"""
        return sum(s.duration_sec for s in self.segments)

    def state_at(self, t: float) -> tuple[str, float]:
        """返回 ``(状态, 本段内已过的秒数)``。

        段内秒数用于眨眼相位——用全局秒数算相位会让"从 drowsy 切回 normal"
        的那一刻恰好落在眨眼中间，凭空多出一次眨眼。
        """
        total = self.total_sec
        if self.loop and math.isfinite(total) and total > 0:
            t = math.fmod(t, total)

        acc = 0.0
        for seg in self.segments:
            if seg.indefinite or t < acc + seg.duration_sec:
                return seg.state, t - acc
            acc += seg.duration_sec

        # 时间超出剧本且不循环：停在最后一段（正常情况下调用方早已判为耗尽）。
        last = self.segments[-1]
        return last.state, t

    def describe(self) -> str:
        tail = "，循环" if self.loop else "，不循环"
        return " → ".join(s.label() for s in self.segments) + tail


def parse_timeline(items: Iterable[tuple[str, float]]) -> tuple[Segment, ...]:
    """把 ``[("normal", 30.0), ("drowsy", 40.0)]`` 解析成段列表。

    校验失败一律给出中文说明并指出**是第几段**——剧本通常照抄自
    设计文档，"第 2 段时长不是正数"比"参数非法"有用得多。
    """
    segments: list[Segment] = []
    for i, item in enumerate(items):
        try:
            state, duration = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"剧本第 {i + 1} 段格式不对：应为 (状态名, 持续秒数)，收到 {item!r}"
            ) from exc
        if not isinstance(state, str) or not state:
            raise ValueError(f"剧本第 {i + 1} 段的状态名必须是非空字符串，收到 {state!r}")
        duration = float(duration)
        if not (duration > 0):
            raise ValueError(f"剧本第 {i + 1} 段（{state}）的持续秒数必须为正数，收到 {duration}")
        segments.append(Segment(state=state, duration_sec=duration))
    if not segments:
        raise ValueError("剧本为空：至少要给一段，例如 [(\"normal\", 30.0)]")
    return tuple(segments)


# ================================================================ 特征源

class SyntheticFeatureSource(BaseFeatureSource):
    """按剧本直出 :class:`FrameFeatures` 的合成源。

    这是本机（无摄像头）的**主路径**，也是全部回归测试的数据来源：
    没有像素、没有模型、没有随机性（``noise=0`` 时）。
    """

    def __init__(
        self,
        scenario: str = "normal",
        timeline: Iterable[tuple[str, float]] | None = None,
        cfg: CaptureConfig | None = None,
        states: Mapping[str, StateSpec] | None = None,
        noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        """
        :param scenario: 内建剧本名（见 :data:`BUILTIN_SCENARIOS`）。
        :param timeline: 自定义剧本，给了它就忽略 ``scenario``。
        :param states: 额外/覆盖的状态配方。
        :param noise: 给角度与比值叠加的均匀噪声幅度。默认 0，**保持完全确定性**；
            调大可用于检验判定对抖动的鲁棒性。
        """
        super().__init__(cfg)
        self._states: dict[str, StateSpec] = dict(STATE_SPECS)
        if states:
            self._states.update(states)

        if timeline is not None:
            segments = parse_timeline(timeline)
            self._scenario = Scenario(name="custom", segments=segments, loop=self.cfg.loop)
        elif scenario in BUILTIN_SCENARIOS:
            segments = parse_timeline(BUILTIN_SCENARIOS[scenario])
            self._scenario = Scenario(name=scenario, segments=segments, loop=self.cfg.loop)
        else:
            known = "、".join(BUILTIN_SCENARIOS)
            raise ValueError(
                f"未知剧本 {scenario!r}。可用：{known}；"
                f"要自定义请传 timeline=[(\"normal\", 30.0), (\"drowsy\", 40.0)]"
            )

        self._check_states()
        self._noise = float(noise)
        self._rng = Random(seed)
        self._wall_start = time.time()

    # ------------------------------------------------------------ 生命周期

    def open(self) -> None:
        """重置计数与时间轴。可重复调用（等价于从头播放）。"""
        self._index = 0
        self._exhausted = False
        self._last_ts = 0.0
        self._pace_ts = 0.0
        self._last_wall = None
        self._wall_start = time.time()
        self._opened = True

    def describe(self) -> str:
        return (
            f"合成源（特征直出）：剧本={self._scenario.name}"
            f"（{self._scenario.describe()}），fps={self.cfg.fps:g}"
            + (f"，噪声=±{self._noise:g}" if self._noise > 0 else "，确定性")
        )

    # ------------------------------------------------------------ 读取

    def read_features(self) -> FrameFeatures | None:
        """产出下一帧特征；``None`` 表示剧本已走完（``loop=False`` 时）或已达上限。"""
        step = self._advance()
        if step is None:
            return None
        spec, ts, t_local = step
        return self._build(spec, ts, t_local)

    # ------------------------------------------------------------ 时间轴

    def _advance(self) -> tuple[StateSpec, float, float] | None:
        """推进一步时间轴，返回 ``(本帧配方, ts, 段内秒数)``；``None`` = 已耗尽。

        **像素路径与特征路径共用这一段**（``read_features`` 与
        ``SyntheticSource.read``）。共用是刻意的：两条路径的换段时刻必须
        逐字相同，否则同一个剧本在"看画面"与"看数字"两边会在不同的秒数
        切状态 —— 那正是"合成源与真机形状悄悄分叉"的开始。
        """
        if not self._opened:
            raise CaptureError(
                "合成源尚未 open()。请先调用 open() —— 或在 with 语句里使用它。"
            )
        if self._over_budget():
            self._exhausted = True
            return None

        ts = self._tick()
        if not self._scenario.loop and ts >= self._scenario.total_sec:
            self._exhausted = True
            return None
        self._pace(ts)

        state, t_local = self._scenario.state_at(ts)
        return self._states[state], ts, t_local

    # ------------------------------------------------------------ 构造

    def _build(self, spec: StateSpec, ts: float, t_local: float) -> FrameFeatures:
        """把配方实例化成一帧特征。"""
        if not spec.has_face:
            return FrameFeatures(
                ts=ts,
                wall_ts=self._wall_start + ts,
                has_face=False,
                quality=spec.quality,
            )

        ear, blink = eye_values(spec, t_local)
        closure = fuse_closure(ear, blink)

        blendshapes = dict(spec.blendshapes)
        # 眼部系数由眼睛模型统一给出，避免配方里两处写不一致。
        blendshapes["eyeBlinkLeft"] = blink
        blendshapes["eyeBlinkRight"] = blink

        pitch, yaw, roll = self._noisy(spec.pitch), self._noisy(spec.yaw), self._noisy(spec.roll)
        gaze = _clamp01(self._noisy(spec.gaze_off))
        ear = _clamp01(self._noisy(ear))

        return FrameFeatures(
            ts=ts,
            wall_ts=self._wall_start + ts,
            has_face=True,
            face_bbox=self._bbox(pitch, yaw, roll),
            ear_left=ear,
            ear_right=ear,
            closure_ratio=closure,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            blendshapes=blendshapes,
            gaze_off_ratio=gaze,
            quality=spec.quality,
        )

    def _noisy(self, value: float) -> float:
        """按 ``noise`` 叠加抖动。``noise=0`` 时原样返回（确定性）。"""
        if self._noise <= 0:
            return value
        return value + self._rng.uniform(-self._noise, self._noise)

    def _bbox(self, pitch: float, yaw: float, roll: float) -> tuple[int, int, int, int]:
        """一个随头姿轻微漂移的人脸框。

        真后端会给出真实框；这里只保证"框会动"，让任何依赖它的调试代码
        （如画框预览）不至于是死的。**该字段敏感，禁止出模块**。
        """
        w, h = self.cfg.width, self.cfg.height
        cx = w * (0.5 + 0.002 * yaw)
        cy = h * (0.42 + 0.002 * pitch + 0.001 * abs(roll))
        half_w = w * 0.16
        half_h = h * 0.22
        return (
            int(max(0, cx - half_w)),
            int(max(0, cy - half_h)),
            int(min(w, cx + half_w)),
            int(min(h, cy + half_h)),
        )

    def _check_states(self) -> None:
        """剧本里引用的状态必须都有配方，否则启动即报错。

        刻意在这里拦住：一个拼错的状态名如果只是"读不到配方"，
        表现是运行中途崩溃或静默退化，而这类故障在无人值守设备上极难定位。
        """
        for seg in self._scenario.segments:
            if seg.state not in self._states:
                known = "、".join(sorted(self._states))
                raise ValueError(
                    f"剧本引用了未定义的状态 {seg.state!r}。可用状态：{known}"
                )

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def states(self) -> Mapping[str, StateSpec]:
        return self._states


# ================================================================ 帧源

class SyntheticSource(SyntheticFeatureSource):
    """合成帧源 = 特征直出 **+** 一条玩具像素路径。

    像素路径的目的只有一个：让上游循环在"真机 / 本机"两种情况下保持
    同一个形状（``read() -> np.ndarray``）。它画的**不是**一张能被
    MediaPipe 检出的人脸，详见模块文档。

    ⚠️ 两条路径共用同一条时间轴：混用 ``read()`` 与 ``read_features()``
    会各自推进一帧，时间轴因此走得比预期快一倍。请只选一条用。
    """

    def read(self) -> np.ndarray | None:
        """产出一张合成 BGR 图；``None`` 表示剧本已走完或已达上限。"""
        step = self._advance()
        if step is None:
            return None
        spec, _ts, t_local = step
        return render_toy_frame(spec, t_local, self.cfg)

    def describe(self) -> str:
        return super().describe().replace("特征直出", "特征直出 + 玩具像素")


class SyntheticPixelSource(BaseCapture):
    """合成 **像素 + 特征同源** 源：没有摄像头时，让展示流也有画面。

    为什么是 :class:`SyntheticSource` 的**兄弟**而不是子类
    -----------------------------------------------------

    理由很具体：服务器的派发（``server._read_one``）是"特征源优先"的，
    而 :class:`SyntheticSource` 同时有 ``read()`` 与 ``read_features()`` ——
    只要它还是 ``FeatureSource``，那条分支就会抢先命中，像素**永远拿不到**。
    本类只有 ``read()``，派发上不存在任何歧义。

    它同时也是**自己的人脸后端**（``process()``）
    --------------------------------------------

    一帧的像素与一帧的特征必须来自**同一次**剧本推进：分两次读会让时间轴
    走快一倍（这个坑 :class:`SyntheticSource` 的文档里已经写过一次）。
    所以 :meth:`read` 一次把两样都算出来，特征先存进一个单槽，紧跟着由
    :meth:`process` 取走 —— ``server._read_one`` 里
    ``read() → backend.process(pixels)`` 这个既有顺序**一行都不用改**。

    ⚠️ **它画不出能被 MediaPipe 检出的人脸**，也不该试图画。这里模拟的是
    "特征与像素可以同时拿到"这条理想管线（真机上 MediaPipe 就是一边收像素
    一边出特征），不是"合成一个人"。喂给真实后端只会得到
    ``has_face=False`` —— 那是预期行为。

    内层那个 ``_script`` 只借剧本、状态配方与构造器；**时间轴不借**：
    ``SyntheticSource`` 与 ``SyntheticFeatureSource`` 引用的私有件
    （``_advance`` / ``_build``）就是这条管线的零件，为它们各写一层公开
    包装只是把同一个东西写两遍。
    """

    def __init__(
        self,
        scenario: str = "normal",
        timeline: Iterable[tuple[str, float]] | None = None,
        cfg: CaptureConfig | None = None,
        states: Mapping[str, StateSpec] | None = None,
        noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(cfg)
        self._script = SyntheticFeatureSource(
            scenario=scenario, timeline=timeline, cfg=self.cfg,
            states=states, noise=noise, seed=seed,
        )
        #: 本帧的特征，等人脸后端（就是本类自己）取走。**只有一槽**：
        #: 取走即清空，下一次 ``read()`` 再放。存两帧以上没有意义 ——
        #: 像素与特征永远是成对消费的。
        self._pending: FrameFeatures | None = None

    # ------------------------------------------------------------ 生命周期

    def open(self) -> None:
        self._script.open()
        self._pending = None
        self._opened = True

    def close(self) -> None:
        self._pending = None
        super().close()

    def describe(self) -> str:
        return self._script.describe().replace("特征直出", "特征直出 + 玩具像素")

    # ------------------------------------------------------------ 时间轴

    @property
    def exhausted(self) -> bool:
        return self._script.exhausted

    @property
    def frame_index(self) -> int:
        return self._script.frame_index

    @property
    def last_ts(self) -> float:
        return self._script.last_ts

    # ------------------------------------------------------------ 读取

    def read(self) -> np.ndarray | None:
        """一次产出 **像素 + 那一帧的特征**；返回像素，特征存进单槽。

        分两步给（而不是返回一个二元组）是为了让 ``_read_one`` 保持它原本的
        形状：``pixels = source.read()`` → ``backend.process(pixels)``。
        """
        step = self._script._advance()
        if step is None:
            self._pending = None
            return None
        spec, ts, t_local = step
        self._pending = self._script._build(spec, ts, t_local)
        return render_toy_frame(spec, t_local, self.cfg)

    def process(self, pixels: np.ndarray | None) -> FrameFeatures:
        """人脸后端协议的那个 ``process()``：把刚读出的特征交出去。

        ``pixels`` 只用来判一次"调用顺序对不对" —— 特征不是从它算出来的，
        而是与它同一帧、同一次剧本推进的产物（见类文档）。**顺序错了必须
        报错**：静默返回上一帧的特征会让画面上的人和文字描述的人差一帧，
        而一帧的错位在界面上几乎看不出来，却能让人排一整天的队。
        """
        features = self._pending
        self._pending = None
        if features is None or pixels is None:
            raise CaptureError(
                "SyntheticPixelSource.process() 收到的像素与本帧特征对不上："
                "它必须紧跟在本类 read() 之后被调用一次（本类同时充当自己的"
                "人脸后端，见 server.build_server 里的 synthetic-pixels 分支）。"
            )
        return features


# ================================================================ 工具

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def eye_values(spec: StateSpec, t_local: float) -> tuple[float, float]:
    """返回 ``(EAR, eyeBlink 系数)``。特征构造与玩具绘图共用这一份。

    持续闭眼的状态没有眨眼相位 —— 否则会在"闭着的基础上再眨一下"，
    产生一个语义上不存在的开合片段。
    """
    if spec.eyes_closed:
        return spec.ear_closed, spec.blink_closed
    if spec.blink_period and spec.blink_period > 0:
        # 相位偏移半个周期：否则第一帧恰好落在眨眼中间，启动标定的
        # 第一个样本就是一张闭眼脸，个体基线从一开始就是偏的。
        phase = math.fmod(t_local + spec.blink_period / 2.0, spec.blink_period)
        if phase < spec.blink_sec:
            return spec.ear_closed, spec.blink_closed
    return spec.ear_open, spec.blink_open


def render_toy_frame(spec: StateSpec, t_local: float, cfg: CaptureConfig) -> np.ndarray:
    """画一张形状正确的玩具图。两个合成源共用（``read()`` 都走到这里）。

    按需导入 cv2：特征路径（本机的主路径）因此**不需要** opencv，
    只有真的要看像素时才要求它。
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - 本机已装
        raise CaptureError(
            "合成像素路径需要 opencv。请安装 opencv-contrib-python，"
            "或改用 read_features() 走特征路径（不需要 opencv）。"
        ) from exc

    w, h = cfg.width, cfg.height
    img = np.full((h, w, 3), _bg_level(spec.quality.illumination), dtype=np.uint8)

    if spec.has_face:
        ear, _ = eye_values(spec, t_local)
        cx, cy = int(w * 0.5), int(h * 0.45)
        axes = (int(w * 0.16), int(h * 0.22))
        # 椭圆按 roll 旋转：肉眼即可看出"头歪了"，便于人工排查。
        cv2.ellipse(img, (cx, cy), axes, spec.roll, 0, 360, (170, 165, 160), -1)
        # 眼睛：按 EAR 决定开口高度（EAR 越小越闭）。
        openness = _clamp01((ear - 0.10) / 0.20)
        eye_h = max(1, int(axes[1] * 0.10 * openness))
        for dx in (-axes[0] // 2, axes[0] // 2):
            cv2.ellipse(
                img, (cx + dx, cy - axes[1] // 4), (axes[0] // 6, eye_h),
                0, 0, 360, (40, 40, 45), -1,
            )

    if spec.quality.blur in (Blur.MEDIUM, Blur.HIGH):
        k = 7 if spec.quality.blur is Blur.MEDIUM else 21
        img = cv2.GaussianBlur(img, (k, k), 0)

    if spec.quality.occlusion is Occlusion.SEVERE:
        # 遮挡：糊成一团近似的纯色，与 "no_face"（画面正常但没人）区分开。
        img = np.full_like(img, 90)
    return img


def _bg_level(illumination: Illumination) -> int:
    """背景灰度：跟着照度走，让像素路径的亮度统计也是"对的"。"""
    if illumination is Illumination.LOW:
        return 18
    if illumination is Illumination.OVEREXPOSED:
        return 248
    return 128


__all__ = [
    "BUILTIN_SCENARIOS",
    "INF",
    "STATE_SPECS",
    "Scenario",
    "Segment",
    "StateSpec",
    "SyntheticFeatureSource",
    "SyntheticPixelSource",
    "SyntheticSource",
    "eye_values",
    "parse_timeline",
    "render_toy_frame",
]
