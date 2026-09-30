# -*- coding: utf-8 -*-
"""主动关怀：决定「机器人什么时候该先开口」。

任务要求 3：「检测用户呆滞/情绪低落时，主动发起关心安慰对话；日常主动问候」。

--------------------------------------------------------------------------
和 core/dialogue.py 的分工
--------------------------------------------------------------------------
    本模块 —— **什么时候**开口（时间驱动的状态机，阈值全在 config §9）
    dialogue.py —— **说什么**（proactive_reply()，中文文案）

分成两块是因为它们变化的节奏完全不同：调阈值不该动文案，改文案不该动时序。

--------------------------------------------------------------------------
三条贯穿全文的设计约束
--------------------------------------------------------------------------
1. **绝不走 respond() / handle_chat()。**
   这是一条会静默毁掉主动关怀的坑：``respond()`` 会更新
   ``_last_user_text_at``，而那个字段是「用户最近说过话」的证据，
   ``fuse_state`` 拿它来推翻视觉的 absent 判定。机器人自己说的话一旦被算进去，
   就变成 —— 对着空房间说一句"我陪着您"，系统立刻认定人在，
   ``absent`` 被改写成 ``normal``。等于用自己的回声证明了房间里有人。
   还会写进 ``role=user`` 的历史行（内容是机器人的话），
   并按 main.py 刷新 30 秒的状态覆盖 —— 主动关怀只要比 30 秒频繁，
   视觉状态就被永久钉住。
   正确做法：只写 ``role=robot``，**不设状态覆盖**（机器人的话不构成关于用户的证据）。

2. **不确定就不说。** 每一条门禁都是"任一不满足就沉默"。
   陪伴机器人说错话的代价，远高于少说一句。

3. **判定纯函数化。** :meth:`ProactivePolicy.evaluate` 不读时钟、不碰全局，
   全部输入从 :class:`ProactiveContext` 进来。于是「静默时段」「防骚扰间隔」
   「一小时内上限」这些最容易写错、又最难在真机上复现的规则，
   可以用假时钟表驱动测试（见 tests/test_proactive.py）。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, Optional, Tuple

import config

logger = logging.getLogger(__name__)

#: 状态 → 关怀类型。absent 刻意**不在**表里：
#: 房间里没人，对着空气说话是荒谬的。
#: 「人回来了」由 greeting 走另一条边沿判定，不是「absent 时慰问」。
CARE_KINDS = {
    config.STATE_SAD: "care_sad",
    config.STATE_TIRED: "care_tired",
}

#: 每小时上限的计数窗口（秒）。1 小时是刻意写死的，不走 config ——
#: 它是「一小时」这个语义本身，不是可调参数。
CAP_WINDOW_SECONDS = 3600.0


@dataclass(frozen=True)
class ProactiveContext:
    """一次判定的**全部**输入。冻结 + 无默认值，逼调用方把话说全。"""

    now: float                  #: 当前墙钟（单调），与 config §9 的秒数同一量纲
    hour: int                   #: 本地墙钟小时 0~23，静默时段用
    state: str                  #: 当前生效的视觉状态（已防抖）
    state_since: float          #: 当前状态从何时开始
    last_user_at: float         #: 用户最近一次说话；0 = 从未
    last_proactive_at: float    #: 上次主动开口；0 = 从未
    recent_proactive: Tuple[float, ...]   #: 最近一小时内的主动开口时刻
    greeting_pending: bool      #: 是否刚经历一次「离开很久后回来」
    speaking: bool              #: 机器正在采集或播报中
    started_at: float           #: 进程启动时刻（宽限期用）


@dataclass(frozen=True)
class ProactiveDecision:
    """决定开口。``kind`` 交给 DialogueEngine.proactive_reply() 挑文案。"""

    kind: str
    state: str
    reason: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "state": self.state, "reason": self.reason}


class ProactivePolicy:
    """纯判定：给定全部输入，决定要不要开口。无内部可变状态。

    所有秒数都是**墙钟秒**。离线回放时调用方负责按倍速缩放
    （见 main.py 的 _scale_for_speed），策略本身对此一无所知。
    """

    def __init__(
        self,
        sustain: float = config.PROACTIVE_SUSTAIN,
        min_interval: float = config.PROACTIVE_MIN_INTERVAL,
        max_per_hour: int = config.PROACTIVE_MAX_PER_HOUR,
        user_cooldown: float = config.PROACTIVE_USER_COOLDOWN,
        greeting_absent: float = config.PROACTIVE_GREETING_ABSENT,
        startup_grace: float = config.PROACTIVE_STARTUP_GRACE,
        quiet_hours: Tuple[int, int] = (config.PROACTIVE_QUIET_START,
                                        config.PROACTIVE_QUIET_END),
        quiet_enabled: bool = config.PROACTIVE_QUIET_ENABLED,
    ) -> None:
        self.sustain = sustain
        self.min_interval = min_interval
        self.max_per_hour = max_per_hour
        self.user_cooldown = user_cooldown
        self.greeting_absent = greeting_absent
        self.startup_grace = startup_grace
        self.quiet_hours = quiet_hours
        self.quiet_enabled = quiet_enabled

    # ------------------------------------------------------------------

    def in_quiet_hours(self, hour: int) -> bool:
        """是否处于静默时段。支持跨零点（22 → 7）。"""
        if not self.quiet_enabled:
            return False
        start, end = self.quiet_hours
        if start == end:
            return False                 # 起止相同视为「不静默」，而不是「全天静默」
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end    # 跨零点

    def evaluate(self, ctx: ProactiveContext) -> Optional[ProactiveDecision]:
        """返回该不该开口；不该开口返回 None。

        门禁按「先便宜后昂贵、先硬性后柔性」排列，
        每一条的失败都直接沉默，不做补救。
        """
        # ---- 门禁 1：正在说话就别插嘴 ----
        # TTS 播放中开口 = 自己盖住自己；STT 采集中开口 = 录进自己的声音。
        if ctx.speaking:
            return None

        # ---- 门禁 2：启动宽限期 ----
        # 刚开机那一小段时间，视觉状态还在 absent→normal 之间抖，
        # 而且一上来就热情打招呼会吓人一跳。
        if ctx.now - ctx.started_at < self.startup_grace:
            return None

        # ---- 门禁 3：静默时段 ----
        # 只静默**主动**开口，绝不静默应答（应答在别处，不受本模块影响）。
        # 凌晨两点的「我不舒服」必须回答 —— 这是安全属性，不是体验偏好。
        if self.in_quiet_hours(ctx.hour):
            return None

        # ---- 门禁 4：用户刚说过话，别抢话头 ----
        # **这是防骚扰最重要的一条。** 用户说了半句停下来想词，
        # 机器人立刻接话，是最让人恼火的失败模式。
        # 顺带保证用 --stdin 打字自测时机器人不插嘴。
        if ctx.last_user_at > 0 and ctx.now - ctx.last_user_at < self.user_cooldown:
            return None

        # ---- 门禁 5：两次主动之间要有间隔 ----
        if ctx.last_proactive_at > 0 and ctx.now - ctx.last_proactive_at < self.min_interval:
            return None

        # ---- 门禁 6：一小时内的次数上限 ----
        if len(ctx.recent_proactive) >= self.max_per_hour:
            return None

        # ---- 门禁 7：状态要真的稳住了一段时间 ----
        # 顺序上放在最后：前面的门禁都能在常数时间内否掉，
        # 不必先算持续时长。
        held = ctx.now - ctx.state_since

        # 优先级：问候 > 关怀。
        # 理由：「刚回来」是一个**事件**（边沿），而「一直低落」是一个**状态**。
        # 事件型的机会窗口只有一次，错过了就永远没了；状态型的一直在。
        if ctx.greeting_pending and ctx.state != config.STATE_ABSENT:
            # greeting_pending 已经在调度器里按「离开 ≥ greeting_absent」筛过，
            # 这里不再重复判断离开时长。
            return ProactiveDecision(
                kind="greeting",
                state=ctx.state,
                reason="离开一段时间后回到画面内",
            )

        kind = CARE_KINDS.get(ctx.state)
        if kind is not None and held >= self.sustain:
            return ProactiveDecision(
                kind=kind,
                state=ctx.state,
                reason=f"「{ctx.state}」已持续 {held:.0f} 秒（阈值 {self.sustain:.0f} 秒）",
            )

        return None


class ProactiveScheduler:
    """持有会变的时间记账，把每个节拍的观测交给 :class:`ProactivePolicy` 判定。

    挂在 main.py 已有的 0.2 秒状态发布节拍上，**不新增轮询线程**。

    线程安全：只有发布线程调用 :meth:`tick` 与 :meth:`note_proactive`，
    :meth:`set_speaking` 可能来自语音线程，用一个 Event 传递（原子操作）。
    """

    def __init__(
        self,
        policy: Optional[ProactivePolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        hour_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        self.policy = policy or ProactivePolicy()
        self._clock = clock
        # 静默时段要的是**本地墙钟小时**，不能用单调时钟换算。
        # 单独注入是为了测试能固定到凌晨 3 点。
        self._hour_provider = hour_provider or (lambda: datetime.now().hour)

        self._started_at = self._clock()
        self._state: Optional[str] = None
        self._state_since = self._started_at
        self._absent_since: Optional[float] = None
        self._greeting_pending = False

        self._last_proactive_at = 0.0
        self._proactive_times: Deque[float] = deque()

        self._speaking = False

    # ------------------------------------------------------------------

    def set_speaking(self, speaking: bool) -> None:
        """语音层在采集/播报时置真，避免机器人抢自己的话。"""
        self._speaking = bool(speaking)

    def note_proactive(self, now: Optional[float] = None) -> None:
        """记一次「真的开口了」。调用方在**成功发出**之后调用。"""
        moment = self._clock() if now is None else now
        self._last_proactive_at = moment
        self._proactive_times.append(moment)

    def note_care(self, now: Optional[float] = None) -> None:
        """记一次「应答里追加了一句关怀」。

        和 :meth:`note_proactive` **刻意不一样**：这里只推 ``last_proactive_at``
        （也就是 ``min_interval`` 那道"别刚说完又开口"的门），
        **不往 ``_proactive_times`` 里记**，所以不占 ``max_per_hour``
        那每小时 4 次的名额。

        为什么必须调：不推的话会撞车 —— t=0 的应答里追加一句 care_sad，
        ``user_cooldown`` 60 秒一过，主动关怀在 t=60 又来说同一句 care_sad。
        两句话隔一分钟说两遍，比不说更像个坏掉的复读机。

        为什么不占配额：这是**两笔预算**。``max_per_hour`` 管的是"机器人
        主动抢话头"，那是越少越好的东西；应答里附带的一句是跟着用户的话
        走的，用户自己开的口，不该因此少掉一次真正的主动关怀。
        """
        moment = self._clock() if now is None else now
        self._last_proactive_at = moment

    @property
    def last_proactive_at(self) -> float:
        return self._last_proactive_at

    def tick(
        self,
        state: str,
        last_user_at: float = 0.0,
        now: Optional[float] = None,
    ) -> Optional[ProactiveDecision]:
        """一个节拍。返回决定开口的内容，或 None。"""
        moment = self._clock() if now is None else now

        self._track_state(state, moment)
        self._prune(moment)

        ctx = ProactiveContext(
            now=moment,
            hour=self._hour_provider(),
            state=state,
            state_since=self._state_since,
            last_user_at=last_user_at,
            last_proactive_at=self._last_proactive_at,
            recent_proactive=tuple(self._proactive_times),
            greeting_pending=self._greeting_pending,
            speaking=self._speaking,
            started_at=self._started_at,
        )

        decision = self.policy.evaluate(ctx)

        # 问候是**一次性**的：不管这次有没有真的说出口（可能正赶上静默时段
        # 或用户刚说完话），这个边沿都算用掉了。
        # 否则会出现「您回来啦」迟到一分钟才说 —— 比不说更奇怪。
        self._greeting_pending = False
        return decision

    # ------------------------------------------------------------------

    def _track_state(self, state: str, now: float) -> None:
        """跟踪状态变化，并识别「离开很久后回来」这个边沿。"""
        if state == self._state:
            return

        if state == config.STATE_ABSENT:
            self._absent_since = now
        else:
            # 刚从 absent 恢复：看离开得够不够久
            if self._state == config.STATE_ABSENT and self._absent_since is not None:
                away = now - self._absent_since
                if away >= self.policy.greeting_absent:
                    self._greeting_pending = True
                    logger.debug("离开了 %.0f 秒后回来，待问候", away)
            self._absent_since = None

        self._state = state
        self._state_since = now

    def _prune(self, now: float) -> None:
        cutoff = now - CAP_WINDOW_SECONDS
        while self._proactive_times and self._proactive_times[0] < cutoff:
            self._proactive_times.popleft()
