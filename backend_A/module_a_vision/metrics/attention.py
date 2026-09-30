"""注意力判定：专注 / 发呆失神。

判据刻意不是"视线是否偏离"，而是"**偏离是否持续**"。
瞥一眼窗外、低头看一眼手里的东西，都是正常的；只有目光涣散且**长时间
保持**才算失神。否则系统会把老人一切不看镜头的时刻都当成走神。

因此本模块维护一个跨窗口的连续偏离计时器（:class:`AttentionTracker`），
而不是逐帧投票——后者会把"今天累计偏离 8 秒"误判成失神。
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.enums import Attention
from shared.frame_features import FrameFeatures

#: 单帧判定为"目光偏离"的阈值。
GAZE_OFF_THRESHOLD = 0.5

#: 连续偏离多久才算失神（设计方案 §4.2 的 T_min=6s）。
ABSENT_MIN_SEC = 6.0

#: 窗口内偏离帧占比达到多少才进入候选（设计方案 §4.2 的 ρ=0.60）。
ABSENT_FRAMES_RATIO = 0.60


@dataclass(frozen=True, slots=True)
class AttentionMetrics:
    """一次注意力统计的结果。"""

    label: Attention = Attention.FOCUSED
    confidence: float = 0.0
    gaze_off_sec: float = 0.0
    stable_sec: float = 0.0
    frames_ratio: float = 0.0
    #: 本窗口里**真的带视线信息**的帧数（``gaze_off_ratio is not None``）。
    #:
    #: 存在的理由是让"没有视线信息"这件事**看得出来**。``Attention`` 只有
    #: FOCUSED / ABSENT 两态，没有"无数据"，所以整窗都没有视线信息时
    #: ``frames_ratio`` 会算出 0.0、``label`` 是 ``FOCUSED`` —— 与"确实一直
    #: 很专注"在结构上无法区分。``gaze_frames == 0`` 就是那个区分的把手：
    #: **看到 ``label=FOCUSED`` 且 ``gaze_frames == 0``，应当读成"这一窗没有
    #: 测量"，而不是"老人很专注"。**
    #:
    #: 实测会走到这里的两条路：CSV 回放（api_doc §3.4 的 V1 CSV **没有视线
    #: 列**，于是每帧都是 ``None``）、以及两眼都量不出眼宽的退化检测。
    #: A-包实跑用的 ``face_landmarker.task`` 是 478 点（带虹膜），正常帧上
    #: 这个数等于本窗口的有效帧数。
    gaze_frames: int = 0


class AttentionTracker:
    """跨帧维护注意力状态。"""

    def __init__(
        self,
        gaze_threshold: float = GAZE_OFF_THRESHOLD,
        absent_min_sec: float = ABSENT_MIN_SEC,
    ) -> None:
        self._gaze_th = gaze_threshold
        self._absent_min = absent_min_sec
        #: 当前连续偏离段的起点；None 表示目光在正前方。
        self._off_start: float | None = None
        #: 失神状态是否已确认（连续偏离超过阈值）。确认后 stable_sec 持续累加。
        self._confirmed = False

    def update(self, frame: FrameFeatures) -> None:
        """摄入一帧。"""
        if not frame.has_face or not frame.quality.valid:
            # 无人脸不是"失神"，是"看不见"。直接断开偏离段，
            # 避免把老人离开房间算成发呆。
            self._off_start = None
            self._confirmed = False
            return

        # `gaze_off_ratio is None` = 这一帧没有视线信息。**当作"不偏离"处理，
        # 并断开偏离段** —— 与上面"无人脸"那条同一个道理：看不见不是失神。
        # 不能拿 0.0 去顶替（那是"完全对正"），也不能让它参与 `>=` 比较
        # （`None >= float` 抛 TypeError，会打死聚合主循环）。
        off = frame.gaze_off_ratio is not None and frame.gaze_off_ratio >= self._gaze_th
        if off:
            if self._off_start is None:
                self._off_start = frame.ts
        else:
            self._off_start = None
            self._confirmed = False

    def gaze_off_sec(self, now_ts: float) -> float:
        """当前连续偏离的持续秒数。"""
        if self._off_start is None:
            return 0.0
        return max(0.0, now_ts - self._off_start)

    def absorb(self, frame: FrameFeatures) -> None:
        """兼容别名，等价于 :meth:`update`。"""
        self.update(frame)

    def snapshot(
        self, window_frames: list[FrameFeatures], now_ts: float
    ) -> AttentionMetrics:
        """为刚结束的窗口产出注意力指标。"""
        n = len(window_frames)
        if not n:
            return AttentionMetrics()

        off_frames = sum(
            1
            for f in window_frames
            if f.has_face
            and f.quality.valid
            and f.gaze_off_ratio is not None
            and f.gaze_off_ratio >= self._gaze_th
        )
        # ⚠️ 分母**必须**把"有脸但没视线信息"的帧排除掉。这是这次改动里
        # 唯一会**静默**出错的地方：只给 `off_frames` 加 None 守卫、不动分母，
        # 代码不会崩、测试全绿、日志干净，但分母被那些永远进不了分子的帧
        # 撑大 → `ratio` 被系统性压低 → 失神初判整体偏向 FOCUSED。
        #
        # 实况（会真的走到的一条）：**CSV 回放**里 api_doc §3.4 的 V1 CSV
        # 没有视线列，于是每一帧的 `gaze_off_ratio` 都是 `None`；分母曾经是
        # "所有有脸的帧"而分子恒为 0。混合行（有些行有视线列、有些没有）时
        # 更隐蔽：ratio 被那些行压低，失神的初判整体偏向 FOCUSED，
        # 而分子分母各自单看都没毛病。
        valid_frames = sum(
            1
            for f in window_frames
            if f.has_face
            and f.quality.valid
            and f.gaze_off_ratio is not None
        )
        ratio = off_frames / valid_frames if valid_frames else 0.0

        off_sec = self.gaze_off_sec(now_ts)

        # 失神 = 窗口内偏离占比够高 **且** 连续偏离够久。
        # 两个条件缺一不可：只有占比高可能是频繁瞥视，只有时长久可能是
        # 中间夹杂了回正。
        is_absent = ratio >= ABSENT_FRAMES_RATIO and off_sec >= self._absent_min

        if is_absent:
            # 置信度由"超出阈值的程度"决定，并封顶在 0.95——
            # 单目视线估计本身有噪声，不应给出接近 1 的确定性。
            excess = min(
                (ratio - ABSENT_FRAMES_RATIO) / (1.0 - ABSENT_FRAMES_RATIO),
                1.0,
            )
            time_excess = min(off_sec / (self._absent_min * 3.0), 1.0)
            confidence = min(0.60 + 0.20 * excess + 0.15 * time_excess, 0.95)
            return AttentionMetrics(
                label=Attention.ABSENT,
                confidence=confidence,
                gaze_off_sec=off_sec,
                stable_sec=off_sec,
                frames_ratio=ratio,
                gaze_frames=valid_frames,
            )

        return AttentionMetrics(
            label=Attention.FOCUSED,
            confidence=min(0.5 + 0.5 * (1.0 - ratio), 0.95),
            gaze_off_sec=off_sec,
            stable_sec=0.0,
            frames_ratio=ratio,
            gaze_frames=valid_frames,
        )
