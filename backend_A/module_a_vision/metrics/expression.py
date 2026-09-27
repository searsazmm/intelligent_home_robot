"""表情识别——基于 blendshape 的**启发式基线**。

⚠️ 这不是一个训练好的分类器。

它是把 ARKit blendshape 系数按面部动作单元的常识组合成四个类别的透明规则。
诚实地说，它的准确率低于一个在老人数据上微调过的模型；《系统设计方案》
§8.3 已经把"缺少老人表情数据"列为首要工程风险。之所以仍然先做它：
它可解释、可调试、零训练成本，且 :class:`ExpressionClassifier` 接口
留好了位置——将来把 ONNX 模型接进来，下游一行都不用改。

关于 ``tired``（疲惫）的一个刻意设计
--------------------------------------

**表情通道对疲惫的贡献被刻意压低**（见 :data:`TIRED_DAMPING`）。

原因是疲惫的真正可靠信号在眼部时序里（PERCLOS、眨眼频率、持续闭眼），
而 blendshape 里的 ``eyeSquint``/``jawOpen`` 既可能来自困倦，也可能来自
眯眼看东西或正在说话。让表情通道去主导疲劳判定，会把"眯眼看电视的老人"
判成"重度疲劳"。因此疲惫由 :mod:`module_a_vision.metrics.fatigue` 主导，
表情只提供一个弱先验。

关于中性标定（**这是老人场景下最容易翻车的地方**）
----------------------------------------------------

ARKit 系数是相对"中性脸"定义的。而老人的面部松弛、眼睑下垂、法令纹深，
会让 ``eyeBlink``、``mouthFrown`` 的**基线整体偏高**——于是 ``sad`` 与
``tired`` 常年误报，系统变成一个不停搭话的唠叨机器。

因此本模块强制要求中性标定。**未标定时置信度封顶在**
:data:`UNCALIBRATED_CONF_CAP` (0.5)，低于判定门槛 τ=0.60，
于是系统安全地退回"不触发"。宁可什么都不做，也不要在没有基线的情况下
凭漂移的系数去打扰老人。
"""

from __future__ import annotations

import math
from typing import Protocol

from shared.enums import Emotion
from shared.frame_features import FrameFeatures

#: 各异常类别的"起算地板"。原始得分低于它一律算 0，
#: 避免微弱信号经 softmax 放大成高置信度。
SCORE_FLOOR = 0.25

#: softmax 温度。越小越锐利。0.20 是让 0.73 分映射到约 0.91 置信度的取值。
SOFTMAX_TEMPERATURE = 0.20

#: 未做中性标定时，置信度的封顶值。必须 < 判定门槛 τ=0.60。
UNCALIBRATED_CONF_CAP = 0.5

#: 疲惫通道的阻尼系数。见模块文档的说明。
TIRED_DAMPING = 0.6

#: 中性标定所需的累计时长（秒）。太短会被一个哈欠带偏。
NEUTRAL_CALIBRATION_SEC = 20.0

#: 完成中性标定所需的最少帧数。样本太少时，几帧异常就足以把均值拉偏。
MIN_CALIBRATION_FRAMES = 30


class ExpressionClassifier(Protocol):
    """表情分类器接口。

    实现者需返回 ``(主标签, 置信度, 四类分布)``，分布键固定为
    ``normal`` / ``tired`` / ``sad`` / ``upset`` 且和为 1。
    """

    def classify(
        self, frame: FrameFeatures
    ) -> tuple[Emotion, float, dict[str, float]]: ...


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def _softmax(scores: dict[str, float], temperature: float = SOFTMAX_TEMPERATURE) -> dict[str, float]:
    """数值稳定的 softmax。"""
    if not scores:
        return {}
    mx = max(scores.values())
    exps = {k: math.exp((v - mx) / temperature) for k, v in scores.items()}
    total = sum(exps.values())
    if total <= 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: v / total for k, v in exps.items()}


class _FrozenBaseline:
    """一个只读的"基线载体"，把任意一组系数当成标定基线喂给分类器。

    存在的唯一理由是让 :meth:`NeutralCalibrator._is_expressive` 能复用
    :meth:`BlendshapeExpressionClassifier.raw_scores` 的**同一份**公式：
    把公式抄到标定器里省不了几行，但两边迟早会各自演化，而"标定认为
    静息、判定认为异常"这种不一致排查起来极其费时。

    :class:`BlendshapeExpressionClassifier` 只向它的 ``calibrator`` 要
    ``baseline`` 这一个属性，所以这个载体也只需要这一个。
    """

    def __init__(self, baseline: dict[str, float]) -> None:
        self._baseline = baseline

    @property
    def baseline(self) -> dict[str, float]:
        return self._baseline


class NeutralCalibrator:
    """中性脸基线估计。

    在系统启动时累计一段"老人安静坐着"的 blendshape 均值，作为个体基线。
    之后所有系数都减去该基线再参与判定。

    刻意做成**显式**的：如果没跑标定就一直在用未标定状态，
    而不是悄悄用一组通用默认值假装标定过。

    标定只接受**截止时刻之内**的帧
    --------------------------------

    这是修掉的一处严重缺陷。早期实现只判"累计够 :data:`NEUTRAL_CALIBRATION_SEC`
    秒了吗"，判据一成立就拿手里**已有**的样本算基线——其中包含刚好
    踩过截止时刻的那一帧，于是等于承认"过了截止还在收样本"。

    病根在于截止时刻可能落在异常段里。剧本"安静 15 秒，然后难过"就是
    这么翻车的：20 秒的截止落在难过段之内，基线变成"15 秒中性 + 5 秒难过"
    的混合，然后 ``calibrated`` 报 ``True``。

    这比"没标定"更糟。没标定时置信度被封顶在 :data:`UNCALIBRATED_CONF_CAP`，
    系统明确知道自己没有基线，安全地退回不触发；而被污染的基线会
    **把要检测的那个表情本身当成中性减掉**——"难过"被算成"一直很正常"，
    上游于是永远拿不到 SAD/UPSET 标签，L1 情绪关怀整条路径不可达。
    一个自信的错误判定，比一个谦虚的"不知道"危险得多。

    所以现在有两条闸门：

    1. **截止时刻**。一到就不再收样本；若此时样本还不够，标定直接
       **作废**（见 :attr:`abandoned`）且不再累计。要重来必须显式调用
       :meth:`reset`——因为"过了截止"意味着这段样本本身就不可信，
       悄悄顺延到下一段只会再污染一次。
    2. **静息性**。窗口之内的每一帧还必须相对**窗口自身的运行均值**
       不像任何异常表情（判据见 :meth:`_is_expressive`）。

    为什么只挡住"截止时刻"那一帧还不够
    ------------------------------------

    因为污染并不发生在截止的那一帧，而发生在截止**之前**。剧本
    "安静 15 秒，然后难过"到 20 秒截止时手里已经有 200 帧——其中 50 帧
    是难过帧，样本数远超 :data:`MIN_CALIBRATION_FRAMES`，于是第 1 条
    闸门放行，基线照样被减掉三分之一。

    第 2 条闸门之所以拿"窗口自身的运行均值"当参照，而不是拿一组通用的
    中性系数当参照：**老人的静息脸本来就偏离标准中性**（见模块文档）。
    用绝对阈值会把"眼睑下垂的老人"整段判成表情、永远标定不出来，
    那等于把这个人群的整条情绪路径也关掉了。相对自身则没有这个问题——
    静息不动的人，每一帧都恰好等于自己的均值，永远不会触发。
    """

    def __init__(self, required_sec: float = NEUTRAL_CALIBRATION_SEC) -> None:
        self._required = required_sec
        self._sum: dict[str, float] = {}
        self._count = 0
        self._start_ts: float | None = None
        self._baseline: dict[str, float] | None = None
        #: 标定已作废：截止时刻已过而样本不够。见类文档。
        self._abandoned = False

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def abandoned(self) -> bool:
        """标定是否已作废（截止时刻已过，样本不足以形成可信基线）。

        与"尚未标定"不是一回事：未标定还可能标定成功，作废则是
        **给这段样本判了死刑**，在 :meth:`reset` 之前不会再产出基线。
        调用方据此可以决定是重开标定，还是本次开机就一直用封顶的置信度。
        """
        return self._abandoned

    @property
    def baseline(self) -> dict[str, float] | None:
        return self._baseline

    def feed(self, frame: FrameFeatures) -> None:
        """喂入一帧标定样本。标定完成或作废后不再累计。"""
        if (
            self._abandoned
            or self.calibrated
            or not frame.has_face
            or not frame.quality.valid
        ):
            return
        if self._start_ts is None:
            self._start_ts = frame.ts

        if frame.ts - self._start_ts >= self._required:
            # 截止时刻已到：这一帧本身**不**计入基线——它可能已经是异常帧了。
            if self._count >= MIN_CALIBRATION_FRAMES:
                self._baseline = {
                    k: v / self._count for k, v in self._sum.items()
                }
            else:
                self._abandoned = True
            return

        # 窗口之内的帧也要过静息性这一关：老人中途难过起来，不该把它
        # 当成"中性"减掉。见 :meth:`_is_expressive`。
        if self._is_expressive(frame):
            self._abandoned = True
            return

        for name, value in frame.blendshapes.items():
            self._sum[name] = self._sum.get(name, 0.0) + value
        self._count += 1

    def _is_expressive(self, frame: FrameFeatures) -> bool:
        """这一帧相对**本窗口的运行均值**是否已经明显像异常表情。

        判据复用分类器同一套打分公式（而不是复写一份系数），只把基线换成
        窗口自身的运行均值——差别在于：与静息脸的偏离量超过
        :data:`SCORE_FLOOR` 才算"像表情"。地板值本来就是干这个的：
        低于它的信号一律当噪声。

        ``_count == 0`` 时没有参照均值，无条件放行——否则老人到场的
        第一帧就会被拿绝对系数去比，眼睑偏重的人当场标定失败。
        """
        if self._count == 0:
            return False
        mean = {k: v / self._count for k, v in self._sum.items()}
        scores = BlendshapeExpressionClassifier(
            _FrozenBaseline(mean)
        ).raw_scores(frame)
        return max(scores.values()) > SCORE_FLOOR

    def reset(self) -> None:
        """清空基线与作废标记，重新开始标定。"""
        self._sum.clear()
        self._count = 0
        self._start_ts = None
        self._baseline = None
        self._abandoned = False


class BlendshapeExpressionClassifier:
    """基于 blendshape 规则的启发式表情分类器。"""

    def __init__(self, calibrator: NeutralCalibrator | None = None) -> None:
        self._calibrator = calibrator

    # ------------------------------------------------------------ 打分

    def _adjusted(self, frame: FrameFeatures, name: str) -> float:
        """读取一个 blendshape 并扣除个体中性基线。"""
        raw = frame.blend(name)
        base = self._calibrator.baseline if self._calibrator else None
        if not base:
            return raw
        return _clamp(raw - base.get(name, 0.0))

    def _pair(self, frame: FrameFeatures, base: str) -> float:
        return (self._adjusted(frame, f"{base}Left") + self._adjusted(frame, f"{base}Right")) / 2.0

    def raw_scores(self, frame: FrameFeatures) -> dict[str, float]:
        """计算三个异常类别的原始得分（未过 softmax）。"""
        # 难过：眉心上挑（内眉上扬）+ 嘴角下拉。设计文档 §5.5 的"难过"。
        sad = (
            0.50 * self._adjusted(frame, "browInnerUp")
            + 0.35 * self._pair(frame, "mouthFrown")
            + 0.15 * (1.0 - self._pair(frame, "mouthSmile"))
        )
        # 烦闷：眉毛下压 + 嘴唇紧抿 + 鼻翼提。区别于难过的关键在 browDown 与 mouthPress。
        upset = (
            0.45 * self._pair(frame, "browDown")
            + 0.30 * self._pair(frame, "mouthPress")
            + 0.15 * self._pair(frame, "noseSneer")
            + 0.10 * self._adjusted(frame, "mouthPucker")
        )
        # 疲惫：刻意阻尼，见模块文档。
        tired = TIRED_DAMPING * (
            0.35 * self._pair(frame, "eyeSquint")
            + 0.30 * (1.0 - self._pair(frame, "eyeWide"))
            + 0.20 * self._adjusted(frame, "jawOpen")
            + 0.15 * self._pair(frame, "browDown")
        )
        return {"sad": _clamp(sad), "upset": _clamp(upset), "tired": _clamp(tired)}

    # ------------------------------------------------------------ 分类

    def classify(
        self, frame: FrameFeatures
    ) -> tuple[Emotion, float, dict[str, float]]:
        """返回 ``(主标签, 置信度, 四类分布)``。"""
        if not frame.has_face or not frame.quality.valid:
            return Emotion.NORMAL, 0.0, _uniform_normal()

        raw = self.raw_scores(frame)

        # 低于地板的得分归零，避免微弱信号被 softmax 放大。
        gated = {
            k: _clamp((v - SCORE_FLOOR) / (1.0 - SCORE_FLOOR))
            for k, v in raw.items()
        }
        # "正常"是兜底类：没有任何异常信号足够强时由它胜出。
        normal = _clamp(1.0 - max(gated.values()))

        scores = {
            str(Emotion.NORMAL): normal,
            str(Emotion.TIRED): gated["tired"],
            str(Emotion.SAD): gated["sad"],
            str(Emotion.UPSET): gated["upset"],
        }
        dist = _softmax(scores)

        label_str = max(dist, key=lambda k: dist[k])
        confidence = dist[label_str]

        # 未标定则封顶，使其无法越过 τ=0.60。
        if self._calibrator is not None and not self._calibrator.calibrated:
            confidence = min(confidence, UNCALIBRATED_CONF_CAP)

        return Emotion(label_str), confidence, dist


def _uniform_normal() -> dict[str, float]:
    """无有效人脸时的分布：全部概率给"正常"。"""
    return {
        str(Emotion.NORMAL): 1.0,
        str(Emotion.TIRED): 0.0,
        str(Emotion.SAD): 0.0,
        str(Emotion.UPSET): 0.0,
    }
