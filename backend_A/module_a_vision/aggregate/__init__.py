"""窗口聚合——把 10 秒的逐帧特征压成一个 :class:`~shared.schema.WindowState`。

这一层是 A 与 B 的**职责分界线**，值得说清：

* **A（本层）做窗口内的事**：投票占比 ρ、窗口内最短持续 T_min、
  跨窗口累计时长 ``stable_sec``、以及窗口整体质量是否可用。
* **B 做跨窗口的事**：连续 N 个窗口、冷却、升级、推送仲裁。

这么分是因为两者的**可测性条件完全不同**：窗口内的统计依赖帧率与
画面内容，只能在有数据时验证；跨窗口的判定依赖时间与历史，必须能
在没有摄像头、没有模型的机器上反复跑。把后者留在 B，安全逻辑就
永远测得了。

``stable_sec`` 有一个容易做错的地方：它是**跨窗口累计**的时长，
可能超过单个窗口的 ``duration_sec``。早期把它理解成"本窗口内持续了
多久"，会让"持续闭眼 15 秒"这类规则在 10 秒窗口下**在数学上不可观测**。
"""

from __future__ import annotations

from .window import (
    AggregatorConfig,
    StableDurationTracker,
    WindowAggregator,
    WindowBuffer,
)

__all__ = [
    "AggregatorConfig",
    "StableDurationTracker",
    "WindowAggregator",
    "WindowBuffer",
]
