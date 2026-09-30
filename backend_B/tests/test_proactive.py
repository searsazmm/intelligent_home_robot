# -*- coding: utf-8 -*-
"""主动关怀的测试：假时钟 + 表驱动。

这里全部用**显式传入的 now**，不碰真实时钟 —— 静默时段、防骚扰间隔、
一小时上限这些规则靠 sleep 测会又慢又飘，而且凌晨 3 点那条根本没法测。

沿用仓库既有测试的写法（unittest，自己把父目录塞进 sys.path）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                                    # noqa: E402
from core.dialogue import DialogueEngine                         # noqa: E402
from core.proactive import (                                     # noqa: E402
    CARE_KINDS,
    CAP_WINDOW_SECONDS,
    GATE_FOCUS,
    GATE_NO_TRIGGER,
    GATE_SPEAKING,
    ProactiveContext,
    ProactiveDecision,
    ProactivePolicy,
    ProactiveScheduler,
)

# 测试用的固定基准时刻，避免浮点误差让边界断言变得奇怪
T0 = 100_000.0

#: 一个「所有门禁都放行」的上下文。各测试只覆盖自己关心的那一个字段，
#: 这样某条测试失败时原因一眼可见。
def open_context(**overrides) -> ProactiveContext:
    base = dict(
        now=T0,
        hour=15,                       # 下午 3 点，不在静默时段
        state=config.STATE_SAD,
        state_since=T0 - 100.0,        # 早就稳住了
        last_user_at=0.0,              # 从没说过话
        last_proactive_at=0.0,         # 从没主动过
        recent_proactive=(),
        greeting_pending=False,
        speaking=False,
        started_at=T0 - 1000.0,        # 早过了宽限期
        focused=False,                 # 老人没有在专注（门禁 8）
    )
    base.update(overrides)
    return ProactiveContext(**base)


class TestPolicyGates(unittest.TestCase):
    """每一条门禁单独测：满足时开口，不满足时沉默。"""

    def setUp(self):
        self.policy = ProactivePolicy(
            sustain=20.0,
            min_interval=90.0,
            max_per_hour=4,
            user_cooldown=60.0,
            greeting_absent=60.0,
            startup_grace=15.0,
            quiet_hours=(22, 7),
            quiet_enabled=True,
        )

    def test_fires_when_everything_is_clear(self):
        """基线：门禁全过时确实会开口。没有这条，其余测试可能是假绿。"""
        decision = self.policy.evaluate(open_context())
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "care_sad")

    # ---- 门禁 1：正在说话 ----

    def test_silent_while_speaking(self):
        """TTS 播报中不开口，否则自己盖住自己。"""
        self.assertIsNone(self.policy.evaluate(open_context(speaking=True)))

    # ---- 门禁 2：启动宽限期 ----

    def test_silent_during_startup_grace(self):
        ctx = open_context(started_at=T0 - 5.0)     # 才开机 5 秒
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_fires_right_after_grace_expires(self):
        """宽限期的边界：刚好到期就应该能开口，不能多等一个节拍。"""
        ctx = open_context(started_at=T0 - 15.0)
        self.assertIsNotNone(self.policy.evaluate(ctx))

    # ---- 门禁 3：静默时段 ----

    def test_silent_at_night(self):
        self.assertIsNone(self.policy.evaluate(open_context(hour=23)))
        self.assertIsNone(self.policy.evaluate(open_context(hour=3)))
        self.assertIsNone(self.policy.evaluate(open_context(hour=6)))

    def test_not_silent_at_daytime(self):
        for hour in (7, 12, 15, 21):
            with self.subTest(hour=hour):
                self.assertIsNotNone(self.policy.evaluate(open_context(hour=hour)))

    def test_quiet_hours_span_midnight(self):
        """22→7 跨零点。写错成 start <= hour < end 的话整晚都不会静默。"""
        self.assertTrue(self.policy.in_quiet_hours(22))
        self.assertTrue(self.policy.in_quiet_hours(0))
        self.assertTrue(self.policy.in_quiet_hours(6))
        self.assertFalse(self.policy.in_quiet_hours(7))
        self.assertFalse(self.policy.in_quiet_hours(21))

    def test_quiet_hours_non_wrapping(self):
        """起止不跨零点时（比如午休 13→14）也要对。"""
        policy = ProactivePolicy(quiet_hours=(13, 14))
        self.assertTrue(policy.in_quiet_hours(13))
        self.assertFalse(policy.in_quiet_hours(14))
        self.assertFalse(policy.in_quiet_hours(12))

    def test_same_start_and_end_means_never_quiet(self):
        """起止相同视为「不静默」，而不是「全天静默」。

        写反了会让机器人整天不说话，而且在测试里看不出来 ——
        因为大多数测试用的是 daytime 的 hour。单独钉住。
        """
        policy = ProactivePolicy(quiet_hours=(8, 8))
        for hour in range(24):
            with self.subTest(hour=hour):
                self.assertFalse(policy.in_quiet_hours(hour))

    def test_quiet_can_be_disabled(self):
        policy = ProactivePolicy(quiet_enabled=False, quiet_hours=(22, 7))
        for hour in range(24):
            with self.subTest(hour=hour):
                self.assertFalse(policy.in_quiet_hours(hour))

    # ---- 门禁 4：用户刚说过话 ----

    def test_silent_when_user_just_spoke(self):
        ctx = open_context(last_user_at=T0 - 10.0)   # 10 秒前说过
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_fires_after_user_cooldown(self):
        ctx = open_context(last_user_at=T0 - 61.0)
        self.assertIsNotNone(self.policy.evaluate(ctx))

    def test_zero_last_user_at_is_not_treated_as_now(self):
        """last_user_at=0 表示「从没说过」，不能算成「刚刚说过」。

        这是个真实存在过的错误形态：写成 ``now - last_user_at < cooldown``
        而不判零，0 会让差值变成 10 万秒…… 那倒是不拦。
        但反过来若有人把 0 当成「刚说过」，机器人将永远不开口。
        这条把语义钉死。
        """
        self.assertIsNotNone(self.policy.evaluate(open_context(last_user_at=0.0)))

    # ---- 门禁 5：两次主动之间的间隔 ----

    def test_silent_within_min_interval(self):
        ctx = open_context(last_proactive_at=T0 - 30.0)
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_fires_after_min_interval(self):
        ctx = open_context(last_proactive_at=T0 - 91.0)
        self.assertIsNotNone(self.policy.evaluate(ctx))

    # ---- 门禁 6：一小时上限 ----

    def test_silent_at_hourly_cap(self):
        times = tuple(T0 - 600 + i * 100 for i in range(4))   # 一小时窗口内 4 次
        ctx = open_context(recent_proactive=times)
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_fires_below_hourly_cap(self):
        times = tuple(T0 - 600 + i * 100 for i in range(3))
        ctx = open_context(recent_proactive=times)
        self.assertIsNotNone(self.policy.evaluate(ctx))

    # ---- 门禁 7：持续时长 ----

    def test_silent_before_sustain_elapsed(self):
        ctx = open_context(state_since=T0 - 19.0)
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_fires_at_sustain_boundary(self):
        ctx = open_context(state_since=T0 - 20.0)
        self.assertIsNotNone(self.policy.evaluate(ctx))

    # ---- 状态本身 ----

    def test_normal_never_triggers_care(self):
        ctx = open_context(state=config.STATE_NORMAL)
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_absent_never_triggers_care(self):
        """对着空房间说话是荒谬的。absent 不在 CARE_KINDS 里，且无问候待发。"""
        ctx = open_context(state=config.STATE_ABSENT)
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_absent_does_not_fire_even_with_greeting_pending(self):
        """greeting_pending 只跟踪「回来」的边沿；人在 absent 时不问候。"""
        ctx = open_context(state=config.STATE_ABSENT, greeting_pending=True)
        self.assertIsNone(self.policy.evaluate(ctx))

    def test_care_kinds_table(self):
        """CARE_KINDS 的映射是接口的一部分，会被 reason 和日志用。"""
        self.assertEqual(CARE_KINDS[config.STATE_SAD], "care_sad")
        self.assertEqual(CARE_KINDS[config.STATE_TIRED], "care_tired")
        self.assertNotIn(config.STATE_ABSENT, CARE_KINDS)
        self.assertNotIn(config.STATE_NORMAL, CARE_KINDS)

    def test_tired_fires_care_tired(self):
        ctx = open_context(state=config.STATE_TIRED)
        decision = self.policy.evaluate(ctx)
        self.assertEqual(decision.kind, "care_tired")
        self.assertEqual(decision.state, config.STATE_TIRED)

    # ---- 问候 ----

    def test_greeting_fires_immediately_without_sustain(self):
        """问候是边沿事件，不该等 sustain —— 那是给持续状态用的。

        「您回来啦」要说就得趁人刚进屋，等到 20 秒后再说，
        人已经坐下了，这句话就变得莫名其妙。
        """
        ctx = open_context(greeting_pending=True, state_since=T0)  # 刚刚才切过来
        decision = self.policy.evaluate(ctx)
        self.assertEqual(decision.kind, "greeting")

    def test_greeting_beats_care(self):
        """同时满足时先问候。

        理由：事件型的机会窗口只有一次（人刚回来），错过了永远没了；
        状态型的（一直低落）下次还会满足。
        """
        ctx = open_context(
            greeting_pending=True,
            state=config.STATE_SAD,
            state_since=T0 - 100.0,     # 关怀条件也满足
        )
        self.assertEqual(self.policy.evaluate(ctx).kind, "greeting")

    def test_greeting_still_obeys_the_gates(self):
        """问候不能凌驾于防骚扰之上 —— 静默时段回来也不说话。"""
        ctx = open_context(greeting_pending=True, hour=3)
        self.assertIsNone(self.policy.evaluate(ctx))


class TestDecisionPayload(unittest.TestCase):
    def test_to_dict_shape(self):
        d = ProactiveDecision(kind="care_sad", state=config.STATE_SAD, reason="测试")
        self.assertEqual(
            d.to_dict(),
            {"kind": "care_sad", "state": config.STATE_SAD, "reason": "测试"},
        )

    def test_reason_is_human_readable(self):
        """reason 会进日志和 proactive 报文，必须是能看懂的话，不是枚举名。"""
        policy = ProactivePolicy(sustain=20.0)
        ctx = open_context(state_since=T0 - 30.0)
        reason = policy.evaluate(ctx).reason
        self.assertIn("30", reason)
        self.assertIn(config.STATE_SAD, reason)


class FakeClock:
    """可手动推进的时钟。"""

    def __init__(self, start=T0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestScheduler(unittest.TestCase):
    """调度器：状态跟踪、问候边沿、计数清理。"""

    def setUp(self):
        self.clock = FakeClock()
        self.hour = 15
        self.policy = ProactivePolicy(
            sustain=20.0, min_interval=90.0, max_per_hour=4,
            user_cooldown=60.0, greeting_absent=60.0, startup_grace=15.0,
            quiet_hours=(22, 7),
        )
        self.scheduler = ProactiveScheduler(
            policy=self.policy, clock=self.clock, hour_provider=lambda: self.hour,
        )

    def establish(self, state, settle=True):
        """把调度器推进到「已过宽限期、且该状态已持续超过 sustain」。

        为什么需要它：sustain 的起点是调度器**第一次看见**这个状态的时刻，
        而不是状态在视觉判定器里真正开始的时刻 —— 调度器无从得知启动之前
        发生过什么。所以刚 tick 完 first 一次时 held 恒为 0。
        生产环境里调度器随进程启动、每 0.2 秒看一次，不存在这个偏差；
        但测试里必须显式补上这一段。
        """
        self.clock.advance(self.policy.startup_grace + 1.0)
        self.scheduler.tick(state)                       # 计时开始
        if settle:
            self.clock.advance(self.policy.sustain + 1.0)

    # ---- 宽限期 ----

    def test_startup_grace_uses_real_start_time(self):
        """宽限期从调度器创建算起，不是从第一次 tick 算起。

        sustain 设为 0，把 sustain 这道门禁从等式里去掉 ——
        否则「None」可能来自 sustain 而非宽限期，测试就没有鉴别力了。
        """
        policy = ProactivePolicy(
            sustain=0.0, startup_grace=15.0, min_interval=0.0,
            user_cooldown=0.0, quiet_enabled=False,
        )
        scheduler = ProactiveScheduler(
            policy=policy, clock=self.clock, hour_provider=lambda: 15,
        )
        self.assertIsNone(scheduler.tick(config.STATE_SAD))       # 刚建好
        self.clock.advance(14.0)
        self.assertIsNone(scheduler.tick(config.STATE_SAD))       # 还差 1 秒
        self.clock.advance(1.0)
        self.assertIsNotNone(scheduler.tick(config.STATE_SAD))    # 期满即开口

    # ---- 状态计时 ----

    def test_state_change_resets_sustain_timer(self):
        """状态一变，sustain 计时归零 —— 否则在不同状态间累计时长会误触发。"""
        self.clock.advance(20.0)
        self.scheduler.tick(config.STATE_TIRED)          # 计时开始
        self.clock.advance(15.0)
        self.assertIsNone(self.scheduler.tick(config.STATE_TIRED))   # 才 15 秒
        self.scheduler.tick(config.STATE_SAD)            # 换状态，归零
        self.clock.advance(19.0)
        self.assertIsNone(self.scheduler.tick(config.STATE_SAD))     # 才 19 秒
        self.clock.advance(1.0)
        self.assertIsNotNone(self.scheduler.tick(config.STATE_SAD))  # 满 20 秒

    # ---- 问候 ----

    def test_greeting_after_long_absence(self):
        """回来那一刻就问候，不等下一个节拍。

        「您回来啦」的时效在秒级；拖到下一拍还看不出来，
        但如果哪天为了别的原因改成延迟发出，这条会立刻红。
        """
        self.establish(config.STATE_ABSENT, settle=False)
        self.clock.advance(120.0)                        # 离开 2 分钟
        decision = self.scheduler.tick(config.STATE_NORMAL)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "greeting")
        self.assertEqual(decision.state, config.STATE_NORMAL)

    def test_no_greeting_after_short_absence(self):
        """扭头走开拿个东西就回来，不值得开口。"""
        self.establish(config.STATE_ABSENT, settle=False)
        self.clock.advance(5.0)                          # 只离开 5 秒
        self.assertIsNone(self.scheduler.tick(config.STATE_NORMAL))

    def test_greeting_is_one_shot(self):
        """问候只说一次 —— 不能每 0.2 秒的节拍都问一遍「您回来啦」。"""
        self.establish(config.STATE_ABSENT, settle=False)
        self.clock.advance(120.0)

        first = self.scheduler.tick(config.STATE_NORMAL)
        self.assertEqual(first.kind, "greeting")

        self.clock.advance(0.2)                          # 下一个发布节拍
        self.assertIsNone(self.scheduler.tick(config.STATE_NORMAL))

    def test_greeting_is_spent_even_if_gate_blocks_it(self):
        """被静默时段挡掉的问候不再补说。

        「您回来啦」迟到一小时才说，比不说更奇怪。
        """
        self.hour = 23                                   # 静默时段
        self.establish(config.STATE_ABSENT, settle=False)
        self.clock.advance(120.0)
        self.assertIsNone(self.scheduler.tick(config.STATE_NORMAL))   # 边沿被静默挡下

        self.hour = 9                                    # 天亮了，也不补说
        self.clock.advance(1.0)
        self.assertIsNone(self.scheduler.tick(config.STATE_NORMAL))

    def test_greeting_control_group(self):
        """上一条的对照组：同样的时序、非静默时段，确实会问候。

        没有这条，test_greeting_is_spent_even_if_gate_blocks_it 就算
        整个问候机制全坏了也照样绿。
        """
        self.establish(config.STATE_ABSENT, settle=False)
        self.clock.advance(120.0)
        self.assertIsNotNone(self.scheduler.tick(config.STATE_NORMAL))

    # ---- 防骚扰 ----

    def test_note_proactive_blocks_immediate_repeat(self):
        """开口之后立刻又开口 —— 必须被 min_interval 挡住。"""
        self.establish(config.STATE_SAD)
        self.assertIsNotNone(self.scheduler.tick(config.STATE_SAD))
        self.scheduler.note_proactive()
        self.clock.advance(30.0)
        self.assertIsNone(self.scheduler.tick(config.STATE_SAD))
        self.clock.advance(61.0)                         # 累计 91 秒
        self.assertIsNotNone(self.scheduler.tick(config.STATE_SAD))

    def test_hourly_cap_prunes_old_entries(self):
        """一小时窗口是滑动的：旧记录要过期，否则机器人说满 4 次就永久闭嘴。"""
        self.establish(config.STATE_SAD)
        for index in range(4):
            self.assertIsNotNone(
                self.scheduler.tick(config.STATE_SAD),
                f"第 {index + 1} 次仍应在上限之内",
            )
            self.scheduler.note_proactive()
            self.clock.advance(91.0)                     # 越过 min_interval

        # 说满 4 次，此时被上限挡住
        self.assertIsNone(self.scheduler.tick(config.STATE_SAD))

        # 再等一小时，最早的记录滑出窗口，应该重新可以开口
        self.clock.advance(CAP_WINDOW_SECONDS)
        self.assertIsNotNone(self.scheduler.tick(config.STATE_SAD))

    def test_speaking_flag_blocks(self):
        self.establish(config.STATE_SAD)
        self.scheduler.set_speaking(True)
        self.assertIsNone(self.scheduler.tick(config.STATE_SAD))
        self.scheduler.set_speaking(False)
        self.assertIsNotNone(self.scheduler.tick(config.STATE_SAD))

    def test_last_user_at_passed_through(self):
        """用户刚说完话，调度器不该开口。"""
        self.establish(config.STATE_SAD)
        self.assertIsNone(
            self.scheduler.tick(config.STATE_SAD, last_user_at=self.clock.now - 5.0)
        )
        self.assertIsNotNone(
            self.scheduler.tick(config.STATE_SAD, last_user_at=self.clock.now - 120.0)
        )

    def test_clock_injected(self):
        """确认调度器真的用了注入的时钟，而不是偷偷读 time.monotonic()。"""
        self.assertEqual(self.scheduler._started_at, T0)

    # ---- 应答里追加的关怀（第②步） ----

    def test_note_care_blocks_the_immediate_proactive_repeat(self):
        """⚠️ 不推这道门就会撞车。

        t=0 的应答里追加一句 care_sad，``user_cooldown`` 60 秒一过，
        主动关怀在 t=60 又来说同一句 —— 两句话隔一分钟说两遍，
        比不说更像个坏掉的复读机。
        """
        self.establish(config.STATE_SAD)
        self.scheduler.note_care()
        self.clock.advance(61.0)                        # 越过 user_cooldown
        self.assertIsNone(self.scheduler.tick(config.STATE_SAD),
                          "追加过关怀之后紧接着又主动开口了")
        self.clock.advance(30.0)                        # 越过 min_interval(90s)
        self.assertIsNotNone(self.scheduler.tick(config.STATE_SAD))

    def test_note_care_does_not_consume_the_hourly_quota(self):
        """两笔预算要分开。

        ``max_per_hour`` 管的是"机器人主动抢话头"，越少越好；应答里附带的
        一句是跟着用户的话走的 —— 老人自己开的口，不该因此少掉一次
        真正的主动关怀。
        """
        self.establish(config.STATE_SAD)
        for _ in range(20):                             # 远超每小时 4 次
            self.scheduler.note_care()
            self.clock.advance(0.1)

        self.assertEqual(len(self.scheduler._proactive_times), 0,
                         "追加关怀不该写进主动开口的小时配额")

        # 配额一次没用过，所以第一次主动开口不会被上限挡住
        self.clock.advance(self.policy.min_interval + 1.0)
        self.assertIsNotNone(self.scheduler.tick(config.STATE_SAD))

    def test_note_care_uses_the_injected_clock(self):
        self.scheduler.note_care()
        self.assertEqual(self.scheduler.last_proactive_at, self.clock.now)


class TestDemoScaling(unittest.TestCase):
    """倍速缩放的验收点（计划里那条第 5 项）。

    离线回放 --speed N 时数据时间跑得比墙钟快 N 倍，而所有门槛都是墙钟秒。
    main.py 的换算：**墙钟门槛 = 数据门槛 / 倍速**。

    这里要澄清一件容易搞反的事：**缩放不会让关怀变得更容易触发**，
    它只是把 1 倍速的判定余量原样搬到别的倍速上。

    背景实测（data/sample_vision.csv）：全长 119.9 数据秒，其中 sad 段 20.0 秒。
    而默认门槛 PROACTIVE_SUSTAIN 也是 20 秒 —— 两者基本相等，
    所以**即使在 1 倍速下，默认门槛也是卡在边界上的**：
    触发与否取决于那几十毫秒的采样相位。这不是缩放引入的问题，
    是默认阈值本身就选得太紧。
    答辩要靠的是 --demo 那组更宽松的预设（sustain 10 秒），而不是缩放。
    """

    SAD_SEGMENT_DATA_SECONDS = 20.0     # 实测 sample_vision.csv 的 sad 段长度

    @staticmethod
    def _policy(sustain_wall_seconds: float) -> ProactivePolicy:
        """只留 sustain 一道门禁，其余全部放行。"""
        return ProactivePolicy(
            sustain=sustain_wall_seconds, min_interval=0.0, startup_grace=0.0,
            user_cooldown=0.0, quiet_enabled=False,
        )

    def test_unscaled_threshold_never_fires_at_5x(self):
        """不缩放：5 倍速下 sad 段只播 4 墙钟秒，而门槛是 20 墙钟秒 → 永远不触发。

        这正是要解决的问题本身 —— 答辩时看起来像「主动关怀没实现」。
        """
        wall_span = self.SAD_SEGMENT_DATA_SECONDS / 5.0
        ctx = open_context(state_since=T0 - wall_span)     # 整段都播完了
        self.assertIsNone(self._policy(config.PROACTIVE_SUSTAIN).evaluate(ctx))

    def test_scaling_keeps_the_same_margin_at_every_speed(self):
        """缩放的作用是**保持一致**，不是更容易触发。

        每一档倍速下，门槛与 sad 段的比例都等于 1 倍速时的比例，
        所以「段内 99% 处不触发、刚过门槛就触发」这个行为处处相同。
        """
        for speed in (1.0, 2.0, 5.0, 10.0):
            with self.subTest(speed=speed):
                sustain_wall = config.PROACTIVE_SUSTAIN / speed
                segment_wall = self.SAD_SEGMENT_DATA_SECONDS / speed
                policy = self._policy(sustain_wall)

                self.assertIsNone(policy.evaluate(
                    open_context(state_since=T0 - segment_wall * 0.99)))
                self.assertIsNotNone(policy.evaluate(
                    open_context(state_since=T0 - sustain_wall * 1.01)))

    def test_default_sustain_is_borderline_even_at_1x(self):
        """把上面那段注释里的结论钉成断言，免得有人以为缩放修好了这个问题。

        默认门槛 20 秒 vs sad 段 20.0 秒 —— 1 倍速下同样卡边界。
        哪天有人把 sad 段改短，这条会红，提醒去调 --demo 预设。
        """
        self.assertGreaterEqual(
            config.PROACTIVE_SUSTAIN,
            self.SAD_SEGMENT_DATA_SECONDS * 0.9,
            "默认门槛如果明显短于 sad 段，那这条注释就该更新了",
        )

    def test_demo_preset_fires_at_every_speed(self):
        """--demo 的宽松门槛在每一档倍速下都真的会触发 ——
        这才是答辩时不再「有时行有时不行」的原因。"""
        for speed in (1.0, 2.0, 5.0, 10.0):
            with self.subTest(speed=speed):
                sustain_wall = config.PROACTIVE_DEMO_SUSTAIN / speed
                segment_wall = self.SAD_SEGMENT_DATA_SECONDS / speed
                policy = self._policy(sustain_wall)
                # 播到 sad 段 3/4 处就该开口
                ctx = open_context(state_since=T0 - segment_wall * 0.75)
                self.assertIsNotNone(policy.evaluate(ctx))

    def test_demo_preset_is_strictly_looser_than_default(self):
        """--demo 的每一项都必须比默认更宽松，否则「预设」名不副实。"""
        self.assertLess(config.PROACTIVE_DEMO_SUSTAIN, config.PROACTIVE_SUSTAIN)
        self.assertLess(config.PROACTIVE_DEMO_MIN_INTERVAL, config.PROACTIVE_MIN_INTERVAL)
        self.assertGreater(config.PROACTIVE_DEMO_MAX_PER_HOUR, config.PROACTIVE_MAX_PER_HOUR)
        self.assertLess(config.PROACTIVE_DEMO_GREETING_ABSENT,
                        config.PROACTIVE_GREETING_ABSENT)
        self.assertLess(config.PROACTIVE_DEMO_STARTUP_GRACE, config.PROACTIVE_STARTUP_GRACE)


class TestDialogueIntegration(unittest.TestCase):
    """主动文案与策略的接口。文案本身在 dialogue.py，这里只验形状。"""

    def setUp(self):
        self.engine = DialogueEngine()

    def test_every_care_kind_has_templates(self):
        from core.dialogue import PROACTIVE_TEMPLATES
        for kind in set(CARE_KINDS.values()) | {"greeting"}:
            with self.subTest(kind=kind):
                texts = PROACTIVE_TEMPLATES.get(kind)
                self.assertTrue(texts, f"{kind} 没有文案，策略判出来也没话说")
                for text in texts:
                    self.assertTrue(text.strip())

    def test_proactive_templates_ask_no_questions(self):
        """写作约束 1：主动开口不用问句。

        老人本来情绪就低，还要组织语言回答你 —— 追问是一种索取。
        """
        from core.dialogue import PROACTIVE_TEMPLATES
        for kind, texts in PROACTIVE_TEMPLATES.items():
            for text in texts:
                with self.subTest(kind=kind, text=text):
                    self.assertNotIn("？", text)
                    self.assertNotIn("?", text)

    def test_proactive_reply_returns_a_template(self):
        from core.dialogue import PROACTIVE_TEMPLATES
        for kind in PROACTIVE_TEMPLATES:
            with self.subTest(kind=kind):
                self.assertIn(self.engine.proactive_reply(kind), PROACTIVE_TEMPLATES[kind])

    def test_unknown_kind_falls_back_without_raising(self):
        from core.dialogue import PROACTIVE_TEMPLATES
        self.assertIn(
            self.engine.proactive_reply("no_such_kind"),
            PROACTIVE_TEMPLATES["greeting"],
        )

    def test_proactive_reply_does_not_mark_user_as_present(self):
        """**核心安全属性。**

        proactive_reply 绝不能更新 _last_user_text_at ——
        否则机器人安慰一句空房间，系统就认定「人在」，
        fuse_state 把 absent 改写成 normal。等于用自己的回声证明屋里有人。
        """
        for kind in ("care_sad", "care_tired", "greeting"):
            self.engine.proactive_reply(kind)
        self.assertEqual(self.engine.last_user_text_at, 0.0)

    def test_respond_from_user_still_records(self):
        """对照组：真正的用户输入必须记账，否则上一条测试可能只是因为整个机制坏了。"""
        self.engine.respond("我今天有点累", from_user=True)
        self.assertGreater(self.engine.last_user_text_at, 0.0)

    def test_respond_with_from_user_false_does_not_record(self):
        self.engine.respond("我陪着您", from_user=False)
        self.assertEqual(self.engine.last_user_text_at, 0.0)


class TestFocusGate(unittest.TestCase):
    """门禁 8：老人正在专注 → 不主动开口（需求文档 §4 B2 的 focus）。

    这一组的重点是**它压什么、不压什么**，以及"被它挡住"这件事是可观测的。
    """

    def setUp(self):
        self.policy = ProactivePolicy(
            sustain=20.0, min_interval=90.0, max_per_hour=4,
            user_cooldown=60.0, greeting_absent=60.0, startup_grace=15.0,
            quiet_hours=(22, 7), quiet_enabled=True,
        )

    def test_silent_while_focused(self):
        ctx = open_context(focused=True)
        self.assertIsNone(self.policy.evaluate(ctx))
        # 光看 None 分不出它和"深夜""没到 sustain"，所以要能问出原因。
        self.assertEqual(self.policy.assess(ctx).suppressed_by, GATE_FOCUS)

    def test_fires_when_not_focused(self):
        """对照组：同一份输入，只把 focused 翻过来，就该开口。

        没有这一条，上面那条测试在"整个策略坏掉、永远返回 None"时也会绿。
        """
        decision = self.policy.evaluate(open_context(focused=False))
        self.assertIsNotNone(decision)
        self.assertEqual(decision.kind, "care_sad")

    def test_focus_suppresses_greeting_too(self):
        """**问候也压**：刚回到座位就埋头看报的人，不该被"您回来啦"打断。

        所以门禁 8 必须在"选触发"**之前**（见 assess 里的位置说明）。
        它压的是**主动**开口；老人先开口说话的那条路走 handle_chat，
        根本不经过这里。
        """
        ctx = open_context(focused=True, greeting_pending=True,
                           state=config.STATE_NORMAL)
        self.assertIsNone(self.policy.evaluate(ctx))
        self.assertEqual(self.policy.assess(ctx).suppressed_by, GATE_FOCUS)

    def test_focus_sits_after_the_other_gates(self):
        """专注不是"最先被检查"的那条 —— 更硬的门禁仍然先报自己。

        为什么值得钉：门禁顺序决定了日志里看到的原因。把专注插到最前面
        也能让行为"正确"（一样是沉默），但联调时会看到"因为专注而沉默"
        却掩盖了"其实它正在说话"。顺序本身是诊断信息的一部分。
        """
        ctx = open_context(speaking=True, focused=True)
        self.assertEqual(self.policy.assess(ctx).suppressed_by, GATE_SPEAKING)
        # 一切正常但没到 sustain 时，报的是 no_trigger 而不是 focus。
        ctx = open_context(focused=False, state_since=T0)
        self.assertEqual(self.policy.assess(ctx).suppressed_by, GATE_NO_TRIGGER)

    def test_focus_does_not_leak_into_the_decision(self):
        """专注是**判决的输入**，不是判决的一部分：它不该出现在对外文案里。

        ``ProactiveDecision`` 会给 C 端（``to_dict``）与对话引擎用，
        多带一个 focus 字段就等于把 B 的内部信号泄出去 ——
        需求文档 §4 B2 明确要求 focus **不下发 C**。
        """
        decision = self.policy.evaluate(open_context(focused=False))
        self.assertEqual(set(decision.to_dict()), {"kind", "state", "reason"})


class TestFocusSuppression(unittest.TestCase):
    """调度器上的专注静默：时长上限、记账不能停。"""

    def setUp(self):
        self.clock = FakeClock()
        self.policy = ProactivePolicy(
            sustain=20.0, min_interval=90.0, max_per_hour=4,
            user_cooldown=60.0, greeting_absent=60.0, startup_grace=15.0,
            quiet_hours=(22, 7), quiet_enabled=True,
        )

    def _scheduler(self, focus_suppress_max=30.0):
        return ProactiveScheduler(
            policy=self.policy, clock=self.clock, hour_provider=lambda: 15,
            focus_suppress_max=focus_suppress_max,
        )

    def _establish(self, scheduler, state, settle=True):
        self.clock.advance(self.policy.startup_grace + 1.0)
        scheduler.tick(state, focused=False)
        if settle:
            self.clock.advance(self.policy.sustain + 1.0)

    def test_silent_while_focused(self):
        scheduler = self._scheduler()
        self._establish(scheduler, config.STATE_SAD)
        for _ in range(10):
            self.clock.advance(0.2)
            self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True))
        self.assertEqual(scheduler.last_assessment.suppressed_by, GATE_FOCUS)

    def test_cap_releases_within_the_limit(self):
        """连续专注超过上限 → 放行一次，随后重新计时。

        这条是"关掉一个功能但不留痕迹"的解药：没有它，一位一直看书的老人
        会得到**无限期**的静默，而日志里一片安静。
        """
        scheduler = self._scheduler(focus_suppress_max=30.0)
        self._establish(scheduler, config.STATE_SAD)

        self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True))
        self.clock.advance(29.0)
        self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True),
                          "还没到上限，仍应沉默")

        self.clock.advance(2.0)                       # 累计 31 秒 > 30
        decision = scheduler.tick(config.STATE_SAD, focused=True)
        self.assertIsNotNone(decision, "到了上限就该放行一次")
        self.assertEqual(decision.kind, "care_sad")

        # 放行之后窗口重开：紧接着的下一次仍然沉默（而不是永久恢复）。
        self.clock.advance(0.2)
        self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True))

    def test_cap_window_restarts_when_focus_ends(self):
        """专注中断一次就重新计时 —— 不能把前后两段拼成一段。

        拼起来的话，"专注 20 秒 → 走开 1 秒 → 再专注 20 秒"会累计到 40 秒
        而提前放行；对老人来说那是两回事。
        """
        scheduler = self._scheduler(focus_suppress_max=30.0)
        self._establish(scheduler, config.STATE_SAD)

        self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True))
        self.clock.advance(25.0)
        self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True))

        # 中断。这一拍**本来就该开口**（专注结束了，sustain 早满了），
        # 与窗口记账无关，所以这里不断言返回值 —— 而且 tick 不会自己
        # 调 note_proactive，所以它也不影响后面几拍。
        self.clock.advance(0.2)
        scheduler.tick(config.STATE_SAD, focused=False)

        self.clock.advance(0.2)
        self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True))   # 重新计时
        self.clock.advance(25.0)
        self.assertIsNone(scheduler.tick(config.STATE_SAD, focused=True),
                          "重新计时后 25 秒还不该放行（累计的话早过了）")

    def test_state_tracking_does_not_stop_while_focused(self):
        """专注期间 ``_state_since`` 照常维护（于是也照常被"状态变了"重置）。

        这条守的是 main.py 那边的用法：**绝不能因为反正要静默就跳过 tick**。
        跳过的话这台状态机停在原地，恢复之后关怀要重新等一整个 sustain。
        这里能直接观察的后果就是 `_state_since` 不再跟着状态变。
        """
        scheduler = self._scheduler(focus_suppress_max=1e9)   # 关掉上限这条线
        self._establish(scheduler, config.STATE_TIRED)

        self.clock.advance(5.0)
        scheduler.tick(config.STATE_TIRED, focused=True)
        since = scheduler._state_since

        self.clock.advance(5.0)
        scheduler.tick(config.STATE_SAD, focused=True)        # 状态变了
        self.assertGreater(scheduler._state_since, since,
                           "专注期间状态计时也必须归零")

    def test_greeting_edge_is_consumed_even_when_suppressed(self):
        """被专注挡掉的问候**算用掉了**，不会等专注结束再补说一句。

        老实说这是个有得有失的选择：那位老人确实永远听不到那句「您回来啦」。
        但迟到的问候比不说更奇怪（"您回来啦"——在三分钟后），
        而且这与静默时段下的既有行为一致（见 tick 的注释）。
        写下这条是为了让它成为一个**有意的**行为，而不是某天被人发现。
        """
        scheduler = self._scheduler()
        self._establish(scheduler, config.STATE_ABSENT, settle=False)
        self.clock.advance(120.0)                                   # 离开 2 分钟
        self.assertIsNone(scheduler.tick(config.STATE_NORMAL, focused=True))

        self.clock.advance(0.2)
        decision = scheduler.tick(config.STATE_NORMAL, focused=False)
        self.assertTrue(decision is None or decision.kind != "greeting",
                        "问候是一次性的边沿，被挡掉就该算用掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
