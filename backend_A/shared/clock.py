"""可注入的时间源。

**所有涉及冷却期、每日打扰预算、跨窗口持续性的判定都必须经由 Clock 取时间。**
否则这些逻辑无法在测试中被确定性地驱动——而它们恰好是系统的安全关键部分。

用法::

    gate = RuleGate(clock=FakeClock())
    ...                       # 断言冷却期内不触发
    clock.advance(1800)       # 手动推进 30 分钟
    ...                       # 断言冷却已过期
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Protocol

#: 系统统一使用东八区，避免部署设备时区设置不一致导致夜间静默时段错乱。
CST = timezone(timedelta(hours=8))


class Clock(Protocol):
    """时间源协议。"""

    def now(self) -> datetime:
        """当前墙钟时间（带时区）。

        用于判断夜间静默时段、每日打扰预算的跨天重置、问候日程。
        """
        ...

    def monotonic(self) -> float:
        """单调递增秒数。

        用于测量时长（冷却剩余、评估窗口间隔）。不受系统时间调整影响，
        因此**时长计算必须用它，不能用 now() 相减**。
        """
        ...


class SystemClock:
    """生产环境时钟。"""

    def now(self) -> datetime:
        return datetime.now(CST)

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """测试用时钟：时间只在被显式推进时前进。"""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 9, 23, 9, 0, 0, tzinfo=CST)
        self._monotonic = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """把两个时间轴同时向前推进。"""
        self._now = self._now + timedelta(seconds=seconds)
        self._monotonic += seconds

    def set_time(self, when: datetime) -> None:
        """只改墙钟（用于跨天、切夜间时段），不动单调钟。"""
        self._now = when
