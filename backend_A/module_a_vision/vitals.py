"""体征通道——把 :mod:`rppg` 的 CHROM 结果接上 api_doc §3.5 的线路。

算法早就在仓库里了（``backend_A/rppg.py``），一直卡在**协议**上：它的
HR / RR / IBI / SQI 只写进实验 CSV 和屏幕 overlay，从没上过 A→B 的
TCP 线路。本模块补的是那一段，不含任何新的信号处理。

为什么合成源是**一等公民**，不是像素路径的附赠品
------------------------------------------------

默认演示源是 ``--source synthetic``，它走的是
:class:`~module_a_vision.capture.synthetic.SyntheticFeatureSource` ——
**特征直出，全程不碰像素**（``build_server`` 连人脸后端都不给它传）。
如果把 rPPG 只挂在"有像素"的分支上，那么：

* 演示路径上它**一行都不会执行**（死代码）；
* 而 ``test_interop_with_b.py`` 整组测试照绿 —— 因为没有一条断言
  在问"体征到底发出来没有"。

所以 :class:`SyntheticVitalsSource` 直接产出 ``(R, G, B)`` 序列喂进
:class:`~rppg.Rppg`，让无摄像头的本机路径也有体征数据。

它是**环路自检**，不是生理测量
------------------------------

合成信号里的心率是**声明**出来的一个常数（:attr:`VitalsSpec.hr_bpm`），
算法不知道它。跑出来的 HR 若与声明值吻合，说明**算法与链路是通的**；
它**不能**用来论证这套 rPPG 在真人身上准不准。:meth:`describe` 会
把这句话打出来，免得有人把演示截图当成实验数据。

⚠️ 而这个"吻合"是**有条件的**，条件写在 :class:`VitalsSpec` 的类文档里，
动手调参数前请务必读完：CHROM 的 ``α`` 取幅度、没有符号，所以
"物理上正确"的那组通道比例恰好会让算法**什么也看不见**。默认值
是刻意挑的、能让整条链路真跑起来的那一组，别按直觉去"修正"它。

CSV 回放**不做** rPPG：``data/sample_vision.csv`` 里既没有 RGB 列也没有
像素，往里塞三列假数据会掩盖"这条路本来就走不通"。与
``capture/csv_replay.py`` 处理"没有视线列"的体例一致。

真像素路径（camera / video）本步**未实现**。将来要做的话，取色位置用
``frame.face_bbox`` 的上三分之一算 BGR 均值，**不要**复用
``vision_a.py`` 的地标 ROI —— 那一套的头姿是基线校正后的，语义与
A-包不同，两套报文混进同一条流会让 B 侧的校准基线漂移。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from rppg import Rppg

from .wire import build_rppg_payload


class VitalsSource(Protocol):
    """能给出一帧取色结果的东西。"""

    def read_rgb(self, ts: float) -> tuple[float, float, float] | None:
        """返回 ``(R, G, B)`` 三通道均值；``None`` 表示这一帧取不到色
        （真人路径下就是"这一帧没有可用的人脸 ROI"）。"""
        ...

    def describe(self) -> str: ...


@dataclass(frozen=True, slots=True)
class VitalsSpec:
    """合成体征信号的配方。体例照 :class:`~capture.synthetic.StateSpec`。

    信号模型（每帧在 ``ts`` 处求值）::

        f = hr_bpm / 60
        s = sin(2π f t)
        R = R0 · (1 + a_r·s)
        G = G0 · (1 + a_g·s)
        B = B0                    —— 常数

    为什么 ``a_r`` 与 ``a_g`` 的取值**不是随便填的**（这一段值得读完）
    ------------------------------------------------------------------

    CHROM 的核心是 ``S = X - α·Y``，``α = std(X)/std(Y)``。请特别注意
    **α 取的是幅度、没有符号**。把两个色度投影在基频上的系数记作

        p_X = 3·a_r - 2·a_g        X = 3r - 2g
        p_Y = 1.5·a_r + a_g        Y = 1.5r + g - 1.5b

    于是基频上的残差是 ``p_X - α·p_Y``，而 ``α = |p_X| / |p_Y|``：

    * **``p_X`` 与 ``p_Y`` 同号** → ``α = p_X/p_Y`` → 残差**恰好为 0**。
      基频被完整抵消，算法什么也看不见。此时 ``argmax`` 只会落在
      带边（0.7Hz ≈ 42bpm）或噪声上，实测恒为 44 —— 一个**算得出来、
      却毫无意义**的数字。这一支才是"物理上正确"的比例：真实皮肤的
      脉搏在 X 与 Y 上是同号相加的。
    * **``p_X`` 与 ``p_Y`` 反号** → ``α = -p_X/p_Y`` → 残差
      ``p_X + p_X = 2·p_X``，**不抵消**，基频完好地活下来。

    也就是说：这份"物理上正确"的配方恰好是算法看不见的那一个，
    而能跑通的那一个靠的是 α 的符号盲区。默认值取后者
    （``a_r = 0.01``、``a_g = 0.02`` ⇒ ``p_X = -0.01``、``p_Y = +0.035``），
    因为这份合成源的全部用途就是**让整条链路真的跑起来**：SQI 门控、
    峰值检测、IBI 过滤、报文投影，一个都别落下。

    :meth:`__post_init__` 会在同号时直接报错 —— 那不是"参数调得不好"，
    而是这份配方产不出任何 HR，静默下去只会得到一堆 44。

    由此也划出了这份自检**不能**证明什么，别读过头：

    * 它证明"算法被喂进去了、跑完了、链路通了"；
    * 它**不**证明 CHROM 在真人皮肤上准 —— 恰恰相反，上面那条
      "同号即抵消"说明单频双通道信号根本表达不了真实脉搏的色度方向。
      要验证算法精度，得用真视频与参考设备，那是另一件事。
    """

    #: **声明**的心率（bpm）。刻意取一个非整数，证明算法不是把声明值
    #: 原样抄回去的（实际会落在 FFT 分辨率格点上，见 README）。
    hr_bpm: float = 76.4
    #: 呼吸率（bpm）。本合成源**不建模**呼吸对信号的影响（见 note）。
    rr_bpm: float = 16.0

    #: 三通道直流分量（0-255）。
    dc_r: float = 150.0
    dc_g: float = 130.0
    dc_b: float = 120.0

    #: 相对脉搏幅度。**必须让 p_X 与 p_Y 反号**，见类文档。
    pulsatility_r: float = 0.010
    pulsatility_g: float = 0.020

    #: 幅度噪声（相对值）。默认 0 —— **保持完全确定性**，好让
    #: ``test_vitals.py`` 的数值断言稳定。
    #:
    #: ⚠️ 靠加噪声是**修不好**上面那条抵消的：实测噪声一大，α 的估计
    #: 误差确实让基频露出来一点，但露出来的和噪声一样是乱的（声明
    #: 76.4/110/58 分别跑出 89/74/96）。噪声改变的是"看得见多少"，
    #: 不改变"看不看得见"。
    noise: float = 0.0

    note: str = (
        "合成体征：只建模心搏引起的三通道周期性变化，**不建模呼吸**。"
        "所以 rr 一般会置灰，不要拿它当断言对象。"
    )

    def __post_init__(self) -> None:
        if self._chrominance_beats_same_sign():
            raise ValueError(
                "这组 pulsatility 让 X 与 Y 的基频投影同号，CHROM 的 "
                "alpha 会把脉搏完整抵消，hr 将恒为带边值。请调整 "
                "pulsatility_r / pulsatility_g（见 VitalsSpec 类文档）。"
            )

    def _chrominance_beats_same_sign(self) -> bool:
        """两个色度投影在基频上的系数是否同号（同号即抵消）。"""
        p_x = 3.0 * self.pulsatility_r - 2.0 * self.pulsatility_g
        p_y = 1.5 * self.pulsatility_r + self.pulsatility_g
        return p_x * p_y >= 0.0

    def chrominance_direction(self) -> tuple[float, float]:
        """``(p_X, p_Y)`` —— 诊断用。反号才是能跑通的配方。"""
        return (
            3.0 * self.pulsatility_r - 2.0 * self.pulsatility_g,
            1.5 * self.pulsatility_r + self.pulsatility_g,
        )


#: 默认配方。演示路径用它。
DEFAULT_VITALS = VitalsSpec()


class SyntheticVitalsSource:
    """确定性地产出 ``(R, G, B)`` 序列，供 :class:`RppgMonitor` 消费。

    纯函数式：只依赖传入的 ``ts``，没有内部状态、没有随机性（``noise=0``
    时）。所以同一串时间戳必然得到同一串通道值 —— 测试可以逐位比较。
    """

    def __init__(self, spec: VitalsSpec | None = None, seed: int = 0) -> None:
        self.spec = spec or DEFAULT_VITALS
        if self.spec.noise > 0:
            from random import Random

            self._rng: Any = Random(seed)
        else:
            self._rng = None

    def describe(self) -> str:
        s = self.spec
        p_x, p_y = s.chrominance_direction()
        return (
            f"合成体征（环路自检，**非生理测量**）：声明 hr={s.hr_bpm:g}bpm、"
            f"rr={s.rr_bpm:g}bpm；算法不知道这两个数。"
            f"色度投影 p_X={p_x:+.3f} p_Y={p_y:+.3f}（反号才不被 alpha 抵消）"
            + (f"，噪声=±{s.noise:g}" if s.noise > 0 else "，确定性")
        )

    def read_rgb(self, ts: float) -> tuple[float, float, float] | None:
        """在 ``ts`` 处求值。永远不返回 ``None`` —— 合成源里"总有一张脸"。"""
        s = self.spec
        f = float(s.hr_bpm) / 60.0
        base = math.sin(2.0 * math.pi * f * float(ts))

        r = s.dc_r * (1.0 + self._jitter(s.pulsatility_r) * base)
        g = s.dc_g * (1.0 + self._jitter(s.pulsatility_g) * base)
        b = s.dc_b
        return (r, g, b)

    def _jitter(self, amplitude: float) -> float:
        if self._rng is None:
            return amplitude
        return amplitude * (1.0 + self._rng.uniform(-self.spec.noise, self.spec.noise))


class RppgMonitor:
    """把取色 → :class:`~rppg.Rppg` → §3.5 报文的这一小段串起来。

    **帧报文每秒发 10 条，体征报文每秒发 1 条。** 两者速率不同是刻意的：
    :meth:`Rppg.compute` 自己就按 0.5 秒节流，再密也只会拿到同一个数；
    而 1Hz 远高于 B 判断体征失联所需的速率。速率参数
    :attr:`emit_interval` 与"帧率必须 ≥5Hz"（``Rppg.compute`` 里的
    ``fs < 5.0`` 置灰分支）是两回事，别混。
    """

    #: 体征报文的发送间隔（秒，按**帧时间轴**计，不是墙钟）。
    EMIT_INTERVAL_SEC = 1.0

    def __init__(
        self,
        source: VitalsSource,
        emit_interval: float = EMIT_INTERVAL_SEC,
        rppg_cls: type[Rppg] = Rppg,
    ) -> None:
        self.source = source
        self.emit_interval = float(emit_interval)
        #: 存**类**而不是实例：时间轴回退时要整个换一个新的
        #: （原因见 :meth:`feed` 里那一段，不是随手写成这样的）。
        self._rppg_cls = rppg_cls
        self._rppg = rppg_cls()
        self._last_emit: float | None = None
        self._last_ts: float | None = None
        #: 已产出的体征报文数、因时间轴回退而重建的次数（联调时看这两个数）。
        #: 正常演示里 ``resets`` 应当**一直是 0** —— 它不是"跑了几圈"的计数器，
        #: 它只在数据源的时间轴真的往回走时才加（见 :meth:`feed`）。
        self.packets = 0
        self.resets = 0

    def describe(self) -> str:
        return f"{self.source.describe()}，每 {self.emit_interval:g}s 一条体征报文"

    def feed(self, ts: float) -> dict[str, Any] | None:
        """喂一帧的取色结果。该发体征时报文就返回它，否则返回 ``None``。

        ``ts`` 必须是**单调时间轴**。往回走会被当作数据源重置并重建
        :class:`~rppg.Rppg`（原因见 :attr:`resets`）。"""
        ts = float(ts)

        if self._last_ts is not None and ts < self._last_ts:
            # **时间轴回退必须重置。** ``Rppg`` 内部全靠 ``t`` 的差值，
            # 回退之后有两层失效叠在一起：
            #
            # 1. ``fs = 1/median(diff(t))`` 会由一段跨越回退点的差值算出来
            #    （负数或极小值），此后**一直**走 ``fs < 5.0`` 的置灰分支；
            # 2. 更隐蔽的一层：``Rppg.reset()`` 清的是信号缓冲，
            #    **不清**它内部那个 0.5 秒的 ``compute`` 节流器
            #    （``_last_compute``）。时间戳从 0 重新开始后，节流器还记着
            #    回退前的值，于是 ``compute()`` 每次都判"距上次不到 0.5 秒"
            #    而直接返回 —— 一直到时间轴重新爬过那个值为止。
            #
            # 两层叠起来，表现是"回退之后体征就再也没有了"，每一步都不报错。
            # 所以这里**整个换一个新实例**，不去 reset() —— 换实例能把节流
            # 状态一起清掉，而且是唯一不需要伸手进 ``Rppg`` 私有字段的做法。
            # （``reset()`` 不加这一行不是它的 bug：它的契约是"人脸丢了，
            #  清缓冲"，那时时间轴是往前的，节流器本来就该留着。）
            #
            # ⚠️ 触发条件**不是** ``--loop``。合成源与视频源的时间轴是
            # ``_index / fps``（``capture/base.py:_tick``），``_index`` 只在
            # ``open()`` 里归零，而服务全程只 ``open()`` 一次 —— 所以
            # ``--loop`` 只是让剧本绕圈，ts 一路单调往上爬，实测跑到 415s
            # 也没有回退。**真正会回退的是摄像头重连**：
            # ``capture/camera.py:107`` 在 ``read()`` 里重连成功时把
            # ``_mono_start`` 重置成当时的 ``monotonic()``，而
            # ``_last_ts = monotonic() - _mono_start`` —— 于是拔插一次摄像头
            # 或一次 USB 抖动，ts 就从几百秒跳回 0 附近，而服务并不重启。
            # 这条分支守的是它。
            self._rppg = self._rppg_cls()
            self._last_emit = None
            self.resets += 1
        self._last_ts = ts

        rgb = self.source.read_rgb(ts)
        if rgb is None:
            # 真人路径下这里对应"这一帧没有可用的人脸 ROI"。**不**把
            # 未取到色的帧当作 0 填进去：那会在缓冲里留下一段人造的
            # 波形，比留个空缺危险得多。
            return None
        self._rppg.update(ts, *rgb)

        if self._last_emit is not None and ts - self._last_emit < self.emit_interval:
            return None
        self._last_emit = ts

        hr, rr, ibi_ms, _sqi = self._rppg.compute(ts)
        self.packets += 1
        return build_rppg_payload(ts, hr=hr, rr=rr, ibi_ms=ibi_ms)


__all__ = [
    "DEFAULT_VITALS",
    "RppgMonitor",
    "SyntheticVitalsSource",
    "VitalsSource",
    "VitalsSpec",
]
