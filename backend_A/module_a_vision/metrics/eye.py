"""眼部时序指标：眨眼计数、持续闭眼、PERCLOS。

本模块只做**时序统计**，几何判定在 :mod:`shared.geometry`。
输入是 :class:`~shared.frame_features.FrameFeatures` 序列。

三个指标各自的用途与定义（口径必须写死，否则 A 与 B 会各按各的理解算）：

* ``blink_rate_per_min``——**最近 60 秒**内的眨眼次数 × (60/实际跨度)。
  眨眼定义为闭合段时长 < :data:`MAX_BLINK_SEC`。正常 10–25 次/分。
* ``long_closure_sec``——**当前**仍在持续的闭合段时长；未闭合时为 0。
  正常眨眼闭合 < 0.4 秒，所以只要这个值超过几秒就值得注意。
* ``perclos``——最近 :data:`PERCLOS_WINDOW_SEC`（60 秒）内闭眼帧占比。
  **不是**在 10 秒报文窗口上算的。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from shared.frame_features import FrameFeatures
from shared.geometry import CLOSURE_CLOSED, is_closed

#: 闭眼时长超过此值就不算"眨眼"，而算"持续闭眼"。真实眨眼闭合约 0.2–0.4 秒。
MAX_BLINK_SEC = 0.4

#: PERCLOS 与眨眼频率的积分窗长。PERCLOS 是 60–180 秒量级的指标，
#: 在 10 秒窗上计算会迅速饱和，见 `shared.schema.FatigueObs` 的说明。
PERCLOS_WINDOW_SEC = 60.0


@dataclass(frozen=True, slots=True)
class EyeMetrics:
    """一次眼部统计的结果。"""

    blink_rate_per_min: float = 0.0
    long_closure_sec: float = 0.0
    perclos: float = 0.0
    avg_closure_ratio: float = 0.0
    closure_ratio: float = 0.0
    #: 本窗口内闭眼帧的占比，用于窗口级投票（与 perclos 的窗长不同）。
    closed_frames_ratio: float = 0.0
    #: 本窗口内半闭眼帧的占比。
    half_closed_frames_ratio: float = 0.0
    #: 当前是否正处于一段超长闭合中。
    in_long_closure: bool = False
    #: **进程内累计**的眨眼次数（只增不减）。api_doc §3.2 的 ``blink_cnt`` 用它。
    #:
    #: 不要拿 ``len(EyeTracker._blinks)`` 代替它 —— 那个 deque 被
    #: :meth:`EyeTracker._evict` 按 60 秒滚动裁掉，**会变小**。
    #: 上报一个会倒退的累计值等于伪造"眨眼次数倒退"，而 B 的
    #: ``_blink_rate_per_minute`` 专门检测倒退，一旦倒退就返回 None，
    #: 等于静默关掉一条疲劳判据。
    blink_total: int = 0


class EyeTracker:
    """跨帧维护眼部状态。

    必须跨窗口长期存活——PERCLOS 与眨眼频率都需要 60 秒历史，
    而报文窗口只有 10 秒。
    """

    def __init__(
        self,
        perclos_window_sec: float = PERCLOS_WINDOW_SEC,
        max_blink_sec: float = MAX_BLINK_SEC,
        closed_threshold: float = CLOSURE_CLOSED,
    ) -> None:
        self._perclos_window = perclos_window_sec
        self._max_blink = max_blink_sec
        self._closed_th = closed_threshold

        #: (ts, closure_ratio, is_closed) 的滚动历史。
        self._history: deque[tuple[float, float, bool]] = deque()
        #: 已完成眨眼的时刻。**滚动 60 秒**，会被 `_evict` 裁掉。
        self._blinks: deque[float] = deque()
        #: 只增不减的累计眨眼次数。与 `_blinks` 并行记账：
        #: `_blinks` 服务"最近一分钟频率"，它服务 api_doc §3.2 的 `blink_cnt`。
        #: 语义是**进程内累计，A 重启即归零** —— B 侧已为此做了防御
        #: （vision_state.py 的 `_blink_rate_per_minute` 会检测倒退并放弃该判据），
        #: 所以不要去持久化它。
        self._blink_total: int = 0

        # 当前闭合段的起点；None 表示眼睛睁着。
        self._closure_start: float | None = None

    # ------------------------------------------------------------ 主流程

    def update(self, frame: FrameFeatures, closure_ratio: float) -> None:
        """摄入一帧。

        :param closure_ratio: 由 :func:`shared.geometry.fuse_closure` 算出的闭眼程度。
        """
        ts = frame.ts
        closed = is_closed(closure_ratio, self._closed_th)

        self._history.append((ts, closure_ratio, closed))
        self._evict(ts)

        if closed:
            if self._closure_start is None:
                self._closure_start = ts
        else:
            if self._closure_start is not None:
                duration = ts - self._closure_start
                if duration < self._max_blink:
                    self._blinks.append(ts)
                    # 一次眨眼在此刻完成并计数。累计值只在这里自增，
                    # 且**不受 `_evict` 影响** —— 这是它和 `_blinks` 的全部区别。
                    self._blink_total += 1
                self._closure_start = None
            self._evict(ts)

    def _evict(self, now_ts: float) -> None:
        """丢弃窗口外的历史。"""
        cutoff = now_ts - self._perclos_window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        # 眨眼记录保留同样长的时间即可。
        while self._blinks and self._blinks[0] < cutoff:
            self._blinks.popleft()

    # ------------------------------------------------------------ 查询

    def long_closure_sec(self, now_ts: float) -> float:
        """当前闭合段已持续的秒数；未闭合时为 0。"""
        if self._closure_start is None:
            return 0.0
        return max(0.0, now_ts - self._closure_start)

    def perclos(self) -> float:
        """最近 60 秒的闭眼帧占比。

        历史不足时按现有样本计算——启动后头几秒的 perclos 会偏不稳定，
        这是可接受的：判定门槛还有 ``event_hints`` 之外的持续性要求兜底。
        """
        if not self._history:
            return 0.0
        closed = sum(1 for _, _, c in self._history if c)
        return closed / len(self._history)

    def blink_rate_per_min(self) -> float:
        """最近 60 秒的眨眼频率（次/分）。

        跨度不足时按比例折算。样本过少（< 5 秒）时返回 0 而不是外推出
        一个夸张的数字——外推会让刚启动时误报"眨眼过速"。
        """
        if len(self._history) < 2:
            return 0.0
        span = self._history[-1][0] - self._history[0][0]
        if span < 5.0:
            return 0.0
        return len(self._blinks) * 60.0 / span

    @property
    def blink_total(self) -> int:
        """进程内累计的眨眼次数，**只增不减**。

        供 api_doc §3.2 的 ``blink_cnt`` 使用。与 :meth:`blink_rate_per_minute`
        的区别是后者只看最近 60 秒（所以会上下波动），而它是单调的。
        """
        return self._blink_total

    # ------------------------------------------------------------ 窗口快照

    def snapshot(self, window_frames: list[FrameFeatures], now_ts: float) -> EyeMetrics:
        """为刚刚结束的窗口产出指标。

        :param window_frames: 本窗口内的帧（用于算窗口级占比）。
        """
        n = len(window_frames)
        if n:
            closures = [
                fuse_or_raw(f) for f in window_frames
            ]
            avg_closure = sum(closures) / n
            closed_ratio = sum(
                1 for c in closures if is_closed(c, self._closed_th)
            ) / n
            half_ratio = sum(
                1
                for c in closures
                if not is_closed(c, self._closed_th) and c >= 0.55
            ) / n
            last_closure = closures[-1]
        else:
            avg_closure = closed_ratio = half_ratio = last_closure = 0.0

        long_sec = self.long_closure_sec(now_ts)
        return EyeMetrics(
            blink_rate_per_min=self.blink_rate_per_min(),
            long_closure_sec=long_sec,
            perclos=self.perclos(),
            avg_closure_ratio=avg_closure,
            closure_ratio=last_closure,
            closed_frames_ratio=closed_ratio,
            half_closed_frames_ratio=half_ratio,
            in_long_closure=long_sec >= 1.0,
            blink_total=self._blink_total,
        )


def fuse_or_raw(frame: FrameFeatures) -> float:
    """读取一帧的闭眼程度。

    后端可以直接给出 ``closure_ratio``；若为 0 则回落到按 EAR 与
    blendshape 现算，避免后端忘了填时整条链路静默失效。
    """
    if frame.closure_ratio > 0.0:
        return frame.closure_ratio
    from shared.geometry import fuse_closure

    ear = (frame.ear_left + frame.ear_right) / 2.0
    return fuse_closure(ear, frame.blend_pair("eyeBlink"))
