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

--------------------------------------------------------------------------
门禁 8：专注静默（VAI）
--------------------------------------------------------------------------
需求文档 §4 B2 的 ``focus``：老人正专注于某件事时**暂不主动开口**。
它是**第八道**门禁（见 :meth:`ProactivePolicy.assess` 里的位置说明），
压住问候与关怀两者，**但绝不压应答**。

两个刻意的分工：

    谁算出"在专注"  core/focus.py（三合一：有指数 + 状态有效 + 数据够新）
    谁决定"因此沉默"  本模块的门禁 8，只认 ``ctx.focused`` 这**一个布尔量**

分成两处是因为"多新算新"必须只有一个答案。本模块连带**总时长上限**
（:meth:`ProactiveScheduler._apply_focus_cap`）—— 那是时间记账，
而本模块的定位就是"持有会变的时间记账"。
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
#: 走神（care_distracted）也刻意**不在这张表里** —— 它不是视觉状态映射，
#: 而是看 VAI 指数持续偏低触发（见 :meth:`ProactivePolicy.assess` 的
#: care_distracted 分支），状态多半是 normal：人还在画面里，只是分神了。
CARE_KINDS = {
    config.STATE_SAD: "care_sad",
    config.STATE_TIRED: "care_tired",
}

#: 每小时上限的计数窗口（秒）。1 小时是刻意写死的，不走 config ——
#: 它是「一小时」这个语义本身，不是可调参数。
CAP_WINDOW_SECONDS = 3600.0

#: 判定被**哪一道门**挡住的标识。**只是日志与测试的锚点，不是给用户看的文案** ——
#: 用户看到的是 dialogue.proactive_reply() 挑的那句话。
#: 有了它，「为什么机器人不说话」才有个可查的答案：``evaluate`` 的 ``None``
#: 把六种沉默混成了一个值。
GATE_SPEAKING = "speaking"
GATE_USER_QUIET = "user_quiet"
GATE_AWAITING_REPLY = "awaiting_reply"
GATE_STARTUP_GRACE = "startup_grace"
GATE_QUIET_HOURS = "quiet_hours"
GATE_USER_COOLDOWN = "user_cooldown"
GATE_MIN_INTERVAL = "min_interval"
#: 走神提醒自己的重复冷却（60 秒内不再重复走神提醒）。只压走神分支，
#: 不挡 idle_chat / greeting 等其他关怀 —— 那些有各自的门禁。
GATE_DISTRACT_COOLDOWN = "distract_cooldown"
GATE_HOURLY_CAP = "hourly_cap"
GATE_FOCUS = "focus"
GATE_NO_TRIGGER = "no_trigger"


@dataclass(frozen=True)
class ProactiveContext:
    """一次判定的**全部**输入。冻结；``user_quiet`` 之外无默认值。"""

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
    focused: bool               #: 老人正在专注（VAI，见 core/focus.py）
    #: 老人正在走神（VAI 持续偏低，见 core/focus.py 的 is_distracted）。
    #: 与 ``focused`` 互斥：高专注压住关怀，低专注触发走神提醒，
    #: 中间灰色地带既不静默也不提醒。
    distracted: bool = False
    #: 走神状态已持续的秒数（从首次检测到 VAI 低于阈值算起）。
    #: 由 :class:`ProactiveScheduler` 跟踪，策略只读。
    distracted_held: float = 0.0
    #: 用户明确说过"不想听你说话/别说话"，且静默时长还没到。
    #: 有默认值是为了不动旧的调用点（含大量测试里的位置参数构造）。
    #: 只压主动开口，**绝不压应答** —— 和静默时段同一条安全原则。
    user_quiet: bool = False
    #: 上一次主动开口还没得到老人回应。为真期间不再主动开口 ——
    #: 「问候没有回应就不重复问候」的落点。老人一接话即解除
    #: （Scheduler.note_user_reply），同样只压主动、不压应答。
    awaiting_reply: bool = False
    #: 上一次**走神提醒**发出的时刻；0 = 从未。走神豁免了 awaiting_reply /
    #: min_interval 两道闸门，防复读只能靠自己的重复冷却（60 秒）——
    #: 账本在 :class:`ProactiveScheduler`（note_proactive(kind=...) 记），
    #: 策略只读这一个数。
    last_distract_prompt_at: float = 0.0


@dataclass(frozen=True)
class ProactiveDecision:
    """决定开口。``kind`` 交给 DialogueEngine.proactive_reply() 挑文案。"""

    kind: str
    state: str
    reason: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "state": self.state, "reason": self.reason}


@dataclass(frozen=True)
class ProactiveAssessment:
    """一次判定的**完整**结果：开口的决定 + **沉默的原因**。

    为什么要在 ``Optional[ProactiveDecision]`` 之外再要一个类型：
    ``None`` 把多种沉默混成了一个值（正在说话／用户要求安静／宽限期／
    深夜／用户刚说完话／刚主动过／超配额／没到 sustain／正在专注）。联调时最高频的问题
    恰恰是"它为什么不说话"，而只看 ``None`` 无从回答。

    所以判定把原因一并交出来，:meth:`ProactivePolicy.evaluate` 保持只返回决定
    —— 既有调用方（含测试）一个字都不用改。
    """

    decision: Optional[ProactiveDecision]
    suppressed_by: str = ""

    @property
    def suppressed(self) -> bool:
        return self.decision is None


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
        distract_sustain: float = config.PROACTIVE_DISTRACT_SUSTAIN,
        distract_remind_gap: float = config.PROACTIVE_DISTRACT_REMIND_GAP,
    ) -> None:
        self.sustain = sustain
        self.min_interval = min_interval
        self.max_per_hour = max_per_hour
        self.user_cooldown = user_cooldown
        self.greeting_absent = greeting_absent
        self.startup_grace = startup_grace
        self.quiet_hours = quiet_hours
        self.quiet_enabled = quiet_enabled
        #: 走神要持续多久才触发提醒（秒）。和 sad/tired 的 sustain 同量级，
        #: 但走神是 VAI 低值的持续，不是视觉状态——所以单独配一个参数，
        #: 不复用 ``self.sustain``：将来想独立调走神门槛时不必动 sad/tired。
        self.distract_sustain = distract_sustain
        #: 走神提醒的重复冷却（秒）。走神豁免了 awaiting_reply / min_interval，
        #: 第一时间能说；但 TTS 播 4 秒 + 重新累计 3 秒 = 7-8 秒又说一遍，
        #: 一轮走神连说 3 次（实测）像复读机，还把每小时配额一口气吃光。
        #: 60 秒内不再重复走神提醒 —— 只压走神分支，不挡别的关怀。
        self.distract_remind_gap = distract_remind_gap

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

        想知道**为什么**沉默就用 :meth:`assess` —— 本方法只是它的一层薄包装，
        保留下来是因为调用方（含 29 处测试）只关心决定。
        """
        return self.assess(ctx).decision

    def assess(self, ctx: ProactiveContext) -> ProactiveAssessment:
        """与 :meth:`evaluate` 同一套判定，但把**被哪道门挡住**一起交出来。

        门禁的**顺序与语义一个字没动**，只是在最后、触发选择之前
        插了第 8 道（专注）—— 见该处的注释。
        """
        # ---- 门禁 1：正在说话就别插嘴 ----
        # TTS 播放中开口 = 自己盖住自己；STT 采集中开口 = 录进自己的声音。
        if ctx.speaking:
            return ProactiveAssessment(None, GATE_SPEAKING)

        # ---- 门禁 1b：用户明确要求安静（"别说话""不想听你说话"）----
        # 用户当面表达的意愿，比一切节奏门禁都优先。静默到什么时候由
        # ProactiveScheduler 的 quiet_until 控制（到期自动恢复；用户
        # 期间主动说话会立刻解除）。这里和静默时段是同一条安全原则：
        # 只压**主动开口**，应答在别处，根本不走这个判定。
        if ctx.user_quiet:
            return ProactiveAssessment(None, GATE_USER_QUIET)

        # ---- 门禁 1c：上一次主动开口还没得到回应 ----
        # 交互规则：主动问候抛出去，老人接了话就进入一问一答；
        # 老人没接话，**不再重复问候** —— 对着没反应的老人一遍遍问
        # 「您在呀？」比不问更糟。老人一开口（note_user_reply）即解除。
        # 同样只压主动开口，应答不经过这里。
        # 走神提醒豁免此门禁 —— 走神是视觉事件，idle_chat 后用户没回复
        # 但如果走神了应该立刻提醒，不应该被"等回复"挡住。
        if ctx.awaiting_reply and ctx.distracted_held < self.distract_sustain:
            return ProactiveAssessment(None, GATE_AWAITING_REPLY)

        # ---- 门禁 2：启动宽限期 ----
        # 刚开机那一小段时间，视觉状态还在 absent→normal 之间抖，
        # 而且一上来就热情打招呼会吓人一跳。
        if ctx.now - ctx.started_at < self.startup_grace:
            return ProactiveAssessment(None, GATE_STARTUP_GRACE)

        # ---- 门禁 3：静默时段 ----
        # 只静默**主动**开口，绝不静默应答（应答在别处，不受本模块影响）。
        # 凌晨两点的「我不舒服」必须回答 —— 这是安全属性，不是体验偏好。
        if self.in_quiet_hours(ctx.hour):
            return ProactiveAssessment(None, GATE_QUIET_HOURS)

        # ---- 门禁 4：用户刚说过话，别抢话头 ----
        # **这是防骚扰最重要的一条。** 用户说了半句停下来想词，
        # 机器人立刻接话，是最让人恼火的失败模式。
        # 顺带保证用 --stdin 打字自测时机器人不插嘴。
        if ctx.last_user_at > 0 and ctx.now - ctx.last_user_at < self.user_cooldown:
            return ProactiveAssessment(None, GATE_USER_COOLDOWN)

        # ---- 门禁 5：两次主动之间要有间隔 ----
        # 走神提醒（care_distracted）豁免此门禁 —— 走神是视觉事件触发的，
        # 不是机器人主动唠叨。idle_chat 刚说完 90 秒内如果用户走神了，
        # 应该立刻提醒，不应该被"两次间隔 90 秒"挡住。
        if ctx.last_proactive_at > 0 and ctx.now - ctx.last_proactive_at < self.min_interval:
            if ctx.distracted_held < self.distract_sustain:
                return ProactiveAssessment(None, GATE_MIN_INTERVAL)

        # ---- 门禁 5b：走神提醒自己的重复冷却 ----
        # 只压走神分支（此时 held 必然 >= sustain，其他分支不受影响 ——
        # 和门禁 1c 豁免走神的写法同风格，反过来用）。放这里而不是走神
        # 分支前，是为了让它在 hourly_cap **之前**就把重复提醒否掉：
        # 走神不占配额冷却结束时若配额也满了，日志该记 cap 而不是冷却。
        # 注意：held < sustain 时**不进**这条 —— idle_chat 等其他关怀
        # 照常走后面的门禁。
        if (ctx.last_distract_prompt_at > 0
                and ctx.now - ctx.last_distract_prompt_at < self.distract_remind_gap
                and ctx.distracted_held >= self.distract_sustain):
            return ProactiveAssessment(None, GATE_DISTRACT_COOLDOWN)

        # ---- 门禁 6：一小时内的次数上限 ----
        if len(ctx.recent_proactive) >= self.max_per_hour:
            return ProactiveAssessment(None, GATE_HOURLY_CAP)

        # ---- 门禁 7：状态要真的稳住了一段时间 ----
        # 顺序上放在最后：前面的门禁都能在常数时间内否掉，
        # 不必先算持续时长。
        held = ctx.now - ctx.state_since

        # ---- 门禁 8：老人正在专注，别打扰 ----
        #
        # 放在这里（而不是和 1~6 并排）有两个理由：
        #   1. 前面几道都是"不用想就该沉默"，专注是唯一一条**以用户意愿为由**
        #      的沉默，语义上离触发选择更近；
        #   2. 它同时压住问候与关怀 —— 一位刚回到座位就埋头看报的长者，
        #      不适合被"您回来啦"打断，所以必须在下面选触发**之前**。
        #
        # ⚠️ 这道门只认 ``ctx.focused`` 这**一个布尔量**。"指数可信 且 够专注"
        # （有指数 + 状态有效 + 数据够新 + 指数 ≥ 门槛）在 core/focus.py 的
        # should_stay_silent() 里判完才传进来，这里不做二次判断 ——
        # 否则"多新算新"与"多高算专注"会各有两个答案。
        #
        # ⚠️ 它**不压应答**。「老人先说了话」走 handle_chat，根本不经过这里。
        if ctx.focused:
            return ProactiveAssessment(None, GATE_FOCUS)

        # 优先级：问候 > 关怀。
        # 理由：「刚回来」是一个**事件**（边沿），而「一直低落」是一个**状态**。
        # 事件型的机会窗口只有一次，错过了就永远没了；状态型的一直在。
        if ctx.greeting_pending and ctx.state != config.STATE_ABSENT:
            # greeting_pending 已经在调度器里按「离开 ≥ greeting_absent」筛过，
            # 这里不再重复判断离开时长。
            return ProactiveAssessment(ProactiveDecision(
                kind="greeting",
                state=ctx.state,
                reason="离开一段时间后回到画面内",
            ))

        kind = CARE_KINDS.get(ctx.state)
        if kind is not None and held >= self.sustain:
            return ProactiveAssessment(ProactiveDecision(
                kind=kind,
                state=ctx.state,
                reason=f"「{ctx.state}」已持续 {held:.0f} 秒（阈值 {self.sustain:.0f} 秒）",
            ))

        # ---- 走神提醒：VAI 持续偏低 + 状态 normal（人在画面里只是分神）----
        #
        # 排在 sad/tired 关怀之后、idle_chat 之前 —— 走神比泛泛闲聊更具体，
        # 但不如 sad/tired 这种视觉状态明确（走神只看 VAI 数值，状态仍是 normal）。
        # 走神时状态多半是 normal：人还在画面里，只是分神了。所以这里**不要求**
        # 状态是某个特定值，只看 distracted 持续时长。absent 时 is_distracted
        # 本来就返 False（焦点不可信），不会走到这里。
        #
        # 用问句（『需要我陪您聊聊天吗』）—— 走神提醒要邀请老人搭话，
        # 和 idle_chat 同属"邀请式问候"语义，豁免"主动关怀不用问句"那条约束。
        # 注意：只看 ``distracted_held`` 不看 ``ctx.distracted`` —— VAI 高频跳动
        # 时当前帧可能 VAI>=68（distracted=False），但宽限期内 _distracted_since
        # 未清空，distracted_held 仍在累积。只看持续时长即可。
        if ctx.distracted_held >= self.distract_sustain:
            return ProactiveAssessment(ProactiveDecision(
                kind="care_distracted",
                state=ctx.state,
                reason=f"走神已持续 {ctx.distracted_held:.0f} 秒"
                      f"（阈值 {self.distract_sustain:.0f} 秒）",
            ))

        # ---- 空闲闲聊（交互规则③）：画面内有人、安静了一阵 → 主动问候 ----
        # 「连续 15 秒没有语音输入」不用在这里再判：门禁 4（user_cooldown
        # = 15 秒）已经保证用户至少安静了这么久；用户从未开过口时由
        # 门禁 2（启动宽限期 15 秒）兜底。走到这里，安静条件必然成立。
        #
        # 只在 normal 下触发：sad/tired 归关怀管（上面，等 sustain），
        # absent 对着空房间说话是荒谬的。「老人没回应就不重复」由
        # 门禁 1c（awaiting_reply）保证 —— 闲聊一次没人接，就闭嘴等老人。
        if ctx.state == config.STATE_NORMAL:
            return ProactiveAssessment(ProactiveDecision(
                kind="idle_chat",
                state=ctx.state,
                reason="画面内有人且持续安静，主动问候闲聊",
            ))

        return ProactiveAssessment(None, GATE_NO_TRIGGER)


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
        focus_suppress_max: float = config.FOCUS_SUPPRESS_MAX_SECONDS,
    ) -> None:
        self.policy = policy or ProactivePolicy()
        #: 专注静默的总时长上限。做成参数而不是直接读 config，
        #: 是为了让 main.py 在 --demo 下换成演示值（30 秒）；
        #: 15 分钟在演示里等于"看不出来"。
        self.focus_suppress_max = focus_suppress_max
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

        #: 用户要求安静**持续到何时**（单调时钟）；0 = 当前不在安静期。
        #: 到期自动放行；期间用户主动说一句话立刻 :meth:`clear_quiet`。
        self._quiet_until = 0.0

        #: 上一次主动问候是否还**没人接话**。置真于 :meth:`note_proactive`，
        #: 解除于 :meth:`note_user_reply`。为真期间不再主动开口 ——
        #  「问候没回应就不重复」的落点。
        self._awaiting_reply = False

        #: 专注静默这一轮的起点。**到点放行一次之后会重开**，见 :meth:`_apply_focus_cap`。
        self._focus_since: Optional[float] = None
        #: 走神这一轮的起点。走神是会瞬间波动的（眨眼、扭头一瞬），
        #: 只看一拍就提醒会变成唠叨 —— 用这个时间戳算"已持续多久"，
        #: 由 :meth:`_track_distracted` 维护。
        self._distracted_since: Optional[float] = None
        #: 最近一次判定为走神的时刻（带宽限期用）。
        self._last_distracted_at: Optional[float] = None
        #: 最近一次**走神提醒真的发出**的时刻（0 = 从未）。
        #: 只在 :meth:`note_proactive` 收到 ``kind="care_distracted"`` 时记 ——
        #: 走神豁免了 awaiting_reply / min_interval，防复读的 60 秒重复冷却
        #: 记在这里，经 ``ctx.last_distract_prompt_at`` 交给策略。
        self._last_distract_prompt_at = 0.0
        #: 最近一次判定的**完整**结果（决定 + 被哪道门挡住）。
        #: 给联调与测试用：``tick`` 只返回决定，而"为什么没说话"在
        #: :class:`ProactiveAssessment` 里。默认 ``None`` = 还没 tick 过。
        self.last_assessment: Optional[ProactiveAssessment] = None

    # ------------------------------------------------------------------

    def set_speaking(self, speaking: bool) -> None:
        """语音层在采集/播报时置真，避免机器人抢自己的话。"""
        self._speaking = bool(speaking)

    def enter_quiet(self, seconds: float) -> None:
        """用户说"不想听你说话"后调用：接下来 ``seconds`` 秒内不主动开口。

        只影响 :meth:`tick` 的主动开口判定；**应答路径不经过这里**，
        所以用户在安静期内主动说话照样回答。重复调用会续期。
        """
        if seconds and seconds > 0:
            self._quiet_until = self._clock() + seconds

    def clear_quiet(self) -> None:
        """用户重新主动说话时立即解除安静期（幂等）。"""
        self._quiet_until = 0.0

    def in_quiet(self, now: Optional[float] = None) -> bool:
        """当前是否处于用户要求的安静期（供接线层与测试查询）。"""
        moment = self._clock() if now is None else now
        return moment < self._quiet_until

    def note_proactive(self, now: Optional[float] = None,
                       kind: Optional[str] = None) -> None:
        """记一次「真的开口了」。调用方在**成功发出**之后调用。

        ``kind``：这次主动关怀的类型（``decision.kind``）。只有
        ``"care_distracted"`` 会额外记走神提醒的重复冷却账 ——
        走神豁免了 awaiting_reply / min_interval，60 秒内不再重复说。

        同时把「等回应」标记置真：主动问候是抛给老人的一个问题，
        老人接了话（:meth:`note_user_reply`）才解除。没接话就一直沉默 ——
        交互规则：问候没有回应，**不再重复问候**，避免变成对着发呆的
        老人一遍遍问「您在呀？」的复读机。
        """
        moment = self._clock() if now is None else now
        self._last_proactive_at = moment
        self._proactive_times.append(moment)
        self._awaiting_reply = True
        if kind == "care_distracted":
            self._last_distract_prompt_at = moment
        # 机器人一开口，走神计时清零重新累计：开口本身可能把用户从
        # 走神中拉回来；不清零的话 distracted_held 仍 >= distract_sustain，
        # 下一个 tick（0.25s 后）又会触发一次走神提醒，10 秒内连说 3 次。
        # 走神豁免了 awaiting_reply 和 min_interval 两道闸门，所以必须有
        # 自己的"已说过"记账 —— 这里就是那个记账。
        self._distracted_since = None
        self._last_distracted_at = None

    def note_user_reply(self) -> None:
        """老人接话了（任何真实语音/文字输入）。解除「等回应」标记。

        下一次主动问候恢复按常规节奏判定（user_cooldown 等）。
        与 :meth:`note_care` 一样**只改标记**，不动时间账本。
        """
        self._awaiting_reply = False

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

    @property
    def distracted_held(self) -> float:
        """走神已持续多久（秒，只读视图）。0 = 当前不在走神。

        给展示层（``main._publish_vai`` 下发走神状态给 C）用：读的是
        :meth:`tick` 记的账（:meth:`_track_distracted`），**不是**重新判定 ——
        展示层不参与判决，与 ``_publish_vai`` 的只读旁路约束同一条理由。
        """
        return self._distracted_held(self._clock())

    def tick(
        self,
        state: str,
        last_user_at: float = 0.0,
        now: Optional[float] = None,
        focused: bool = False,
        distracted: bool = False,
    ) -> Optional[ProactiveDecision]:
        """一个节拍。返回决定开口的内容，或 None。

        ``focused`` 由调用方按 ``core/focus.py`` 的 :func:`should_stay_silent`
        算好传进来（可信：有指数 + 状态有效 + 数据够新；且指数 ≥ 门槛）。
        ``distracted`` 同源，由 :func:`is_distracted` 算好（VAI 持续偏低）。
        调度器**不自己去问** FocusTracker —— 那会让本模块依赖视觉层，
        也会让"多新算新 / 多高算专注"出现两个答案。

        ⚠️ **本方法一定要被调用**，哪怕调用方已经知道要静默：
        :meth:`_track_state` 维护 ``_state_since``（``sustain`` 的判据就靠它），
        而下面会把 ``_greeting_pending`` 清掉。调用方在外层"提前 return"
        会让这台状态机停在原地 —— 状态一稳下来反而永远算不出持续时长。
        """
        moment = self._clock() if now is None else now

        self._track_state(state, moment)
        self._track_distracted(distracted, moment)
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
            focused=self._apply_focus_cap(focused, moment),
            distracted=distracted,
            distracted_held=self._distracted_held(moment),
            user_quiet=moment < self._quiet_until,
            awaiting_reply=self._awaiting_reply,
            last_distract_prompt_at=self._last_distract_prompt_at,
        )

        assessment = self.policy.assess(ctx)
        self.last_assessment = assessment

        # 问候是**一次性**的：不管这次有没有真的说出口（可能正赶上静默时段、
        # 用户刚说完话，或者老人正在专注），这个边沿都算用掉了。
        # 否则会出现「您回来啦」迟到一分钟才说 —— 比不说更奇怪。
        self._greeting_pending = False
        return assessment.decision

    # ------------------------------------------------------------------

    def _apply_focus_cap(self, focused: bool, moment: float) -> bool:
        """专注静默的**总时长上限**：连续专注够久就放行一次，并打日志。

        记账放在调度器里而不是策略里，理由与其它时间记账相同 ——
        策略是纯函数（模块开头第 3 条），而"已经连续专注了多久"必须跨节拍记住。

        为什么必须有这个上限：``focus_is_live`` 的新鲜度只管"数据还活着"，
        管不了"这个人真的连续专注了 40 分钟"。没有上限的话，一位一直埋头
        看书的长者会得到**无限期**的静默，而日志里一片安静 ——
        "把功能关掉了但不留痕迹"是最难查的一类失效。

        放行之后窗口**重开**，所以语义是"每连续专注 N 秒最多放行一次"，
        而不是"专注够久就永久恢复"：后者会让一个真正专注的下午
        变成不断被打扰的下午。
        """
        if not focused:
            self._focus_since = None
            return False
        if self._focus_since is None:
            self._focus_since = moment
            return True

        held = moment - self._focus_since
        if held < self.focus_suppress_max:
            return True

        logger.info(
            "专注静默已连续 %.0f 秒（上限 %.0f 秒），这一拍放行一次主动关怀，"
            "随后重新计时",
            held, self.focus_suppress_max,
        )
        self._focus_since = moment
        return False

    # ------------------------------------------------------------------

    def _track_distracted(self, distracted: bool, now: float) -> None:
        """跟踪走神的进入/退出，维护 ``_distracted_since`` 时间戳。

        带宽限期的边沿检测：进入走神时记下起点，退出时**不立刻清空**，
        而是等 2 秒宽限期过后才清空。这样 VAI 瞬时弹回（一拍 70+、
        下一拍又降到 68 以下）不会打断走神计时 —— 实测 VAI 在 60-72
        之间高频跳动，纯边沿检测会让 ``_distracted_since`` 从没累计到
        持续门槛。和 :meth:`_track_state` 跟踪 absent 是同一种"边沿 vs 状态"分工。
        """
        if distracted:
            if self._distracted_since is None:
                self._distracted_since = now
            self._last_distracted_at = now
            return
        # 不在走神：看宽限期过没过
        if self._distracted_since is not None:
            grace = getattr(self.policy, "distract_grace", 2.0)
            last = getattr(self, "_last_distracted_at", None)
            if last is None or (now - last) > grace:
                self._distracted_since = None

    def _distracted_held(self, now: float) -> float:
        """走神已持续多久（秒）。当前不在走神返回 0。"""
        if self._distracted_since is None:
            return 0.0
        return max(0.0, now - self._distracted_since)

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
