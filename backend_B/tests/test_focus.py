# -*- coding: utf-8 -*-
"""B 侧 VAI 专注度指数（``core/focus.py``）。

跑法（在 backend_B 目录下）：
    python -m pytest tests/test_focus.py -v
    python tests/test_focus.py          # 不装 pytest 也能跑，见文件末尾

这个文件要守的两类东西
----------------------

**一、移植保真度。** 指数是"看起来对"和"真的是它"差别最大的那种代码：
少一路证据、权重静默重归一化，出来的仍然是一个 0-100 的数、仍然平滑、
仍然像模像样，**只是不再是 VAI**。所以基准用的是参考实现自己的输入，
而且断言的是**算式的精确结果**（±0.05）而不是"大于 0"。

**二、几条不会报错的失效。** 见 :class:`TestSilentDegradation` 与
:class:`TestSessionContinuity`。

⚠️ 关于 `gaze` 的**极性**（最容易搞反的一处）
---------------------------------------------

* 报文里的 ``gaze``（api_doc §3.5.3）是**偏离量**：``0`` = 正对镜头。
* 参考实现的 ``gaze_alignment`` 是**对正量**：``0.85`` = 对得很正。
* 两者是 ``gaze_evidence = 1 - gaze_off`` 的关系（``core/focus.py`` 里
  明确记为一处**语义替换**）。

所以下面"对齐 0.85"的基准，喂进去的报文值是 **0.15**。搞反了这
一个减号，指数会从 91.8 掉到 53.3（见 :meth:`TestFidelity.test_polarity`），
所以两个方向都钉了断言。

本文件里的"快出 VAI"配置（1 秒校准、2 秒有效观察）**只属于测试**。
默认值一律照搬参考实现（5 秒 / 10 秒），改默认值就等于改了这套东西的定义。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.focus import (
    DEFAULT_WEIGHTS,
    FORMULA_VERSION,
    IDX_INSUFFICIENT,
    IDX_TREND,
    IDX_UNAVAILABLE,
    WEIGHT_VERSION,
    FocusConfig,
    FocusStatus,
    FocusTracker,
    focus_is_live,
    gaze_evidence,
    pose_alignment,
    should_stay_silent,
)
from core.typed_messages import FocusReading
from core.vision_client import OfflineVisionFeeder, VisionClient
from core.vision_state import VisionSample, VisionStateEvaluator

#: 测试专用。参考实现的单测用的就是这三个数
#: （``focus_index/tests/test_pipeline.py`` 的 ``setUp``）。
FAST = dict(
    calibration_seconds=1.0,
    calibration_min_samples=3,
    min_valid_seconds_for_index=2.0,
)

#: 基准场景里那个"正对镜头"的视线对齐度（= 参考实现的 gaze_alignment）。
ALIGNMENT = 0.85
#: 同一个意思，用报文的口径表达：偏离量。**这一个减号是本文件最重要的东西。**
GAZE_OFF = 1.0 - ALIGNMENT


def frame_at(ts, *, yaw=10.0, pitch=5.0, ear=0.29, roll=0.0, has_face=True):
    """一帧帧报文。默认值是"有脸、头微偏、眼睛正常睁着"。"""
    return VisionSample(
        timestamp=ts, has_face=has_face, ear=ear, blink_cnt=0,
        pitch=pitch, yaw=yaw, roll=roll, emo_feature="normal",
    )


def reading_at(ts, *, gaze_off=GAZE_OFF):
    """一条视线报文。``gaze_off=None`` 表示这一帧取不到视线（质量必为 0）。"""
    return FocusReading(
        timestamp=ts,
        gaze=gaze_off,
        gaze_quality=0.0 if gaze_off is None else 1.0,
    )


def feed(tracker, ts, *, gaze_off=GAZE_OFF, **frame_kwargs):
    """把一帧与其视线报文喂进去（顺序照抄 A 的实际发送顺序：帧在前）。"""
    tracker.on_sample(frame_at(ts, **frame_kwargs))
    tracker.on_reading(reading_at(ts, gaze_off=gaze_off))
    return tracker.snapshot()


def warm(tracker, *, count=3, step=0.5, first_ts=0.0, gaze_off=GAZE_OFF, **kwargs):
    """喂够校准所需，返回最后一帧的快照。

    ``FAST`` 配置下**第 3 个样本就会锁定校准**（1.0s 时长与 3 样本同时达标），
    所以默认调用返回的快照是 ``INSUFFICIENT_OBSERVATION`` —— 校准好了，
    只差有效观察时长。想停在 ``WARMING_UP`` 就传 ``count=2``。
    """
    snapshot = None
    for index in range(count):
        snapshot = feed(tracker, first_ts + index * step, gaze_off=gaze_off, **kwargs)
    return snapshot


def all_keys(value, _seen=None):
    """递归收集 dataclass / dict / tuple 里出现过的所有字符串键，用来查某个键在不在。"""
    keys = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys |= all_keys(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            keys |= all_keys(item)
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            keys.add(name)
            keys |= all_keys(getattr(value, name))
    return keys


class TestFidelity(unittest.TestCase):
    """数值基准：与参考实现逐位对齐。"""

    def test_index_matches_the_reference_arithmetic(self):
        """对齐 0.85 + 头姿正对（相对 0°/0°）+ 睁眼 → 91.8。

        算式（参考实现 ``pipeline.py`` 的 ``EvidenceFusion``）：

            0.55*0.85 + 0.25*1.0 + 0.20*1.0 = 0.9175 → 91.8

        首帧 EMA 等于首帧 raw（``_time_constant_ema`` 的约定），而且这里
        每一帧的 raw 都一样，所以 ``raw_value == index``。

        **头姿为什么是"正对"：** 喂进去的是**恒定**的原始角度 (yaw=10°,
        pitch=5°)，校准把基线锁在同一个值上，相对量因此是 (0°, 0°)。
        这一条同时钉住了"基线真的被减掉了"—— 要是谁把原始角度直接送进
        ``pose_alignment``，这里会算出 0.55*0.85 + 0.25*0.4409 + 0.2 = 0.7777
        → 77.8，而不是 91.8。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        self.assertEqual(warm(tracker, count=2).status, FocusStatus.WARMING_UP)

        snapshot = None
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            snapshot = feed(tracker, ts)

        expected = (
            DEFAULT_WEIGHTS["gaze"] * ALIGNMENT
            + DEFAULT_WEIGHTS["pose"] * pose_alignment(0.0, 0.0)
            + DEFAULT_WEIGHTS["eye_open"] * 1.0
        )
        self.assertAlmostEqual(expected, 0.9175, places=6)

        self.assertEqual(snapshot.status, FocusStatus.VALID)
        self.assertEqual(snapshot.index_status, IDX_TREND)
        self.assertAlmostEqual(snapshot.raw_value, round(expected * 100, 1), delta=0.05)
        self.assertAlmostEqual(snapshot.index, 91.8, delta=0.05)
        self.assertEqual(snapshot.available_modalities, ("gaze", "pose", "eye_open"))
        self.assertAlmostEqual(snapshot.modality_completeness, 1.0, places=4)
        self.assertEqual(snapshot.modality_config_id, "完整模态")
        self.assertEqual(snapshot.formula_version, FORMULA_VERSION)
        self.assertEqual(snapshot.weight_version, WEIGHT_VERSION)

    def test_polarity(self):
        """``gaze`` 是偏离量，不是对正量。喂反了会得到 53.3。"""
        # 正对镜头（偏离 0）必须给满分证据。
        self.assertEqual(gaze_evidence(0.0), 1.0)
        # 偏离到 1.0（画面边缘）给 0。
        self.assertEqual(gaze_evidence(1.0), 0.0)
        self.assertEqual(gaze_evidence(GAZE_OFF), ALIGNMENT)
        self.assertIsNone(gaze_evidence(None))
        # 越界的输入被夹住，不会算出负数或 >1 的证据。
        self.assertEqual(gaze_evidence(-0.5), 1.0)
        self.assertEqual(gaze_evidence(1.5), 0.0)

        flipped = (
            DEFAULT_WEIGHTS["gaze"] * GAZE_OFF
            + DEFAULT_WEIGHTS["pose"] * 1.0
            + DEFAULT_WEIGHTS["eye_open"] * 1.0
        )
        self.assertAlmostEqual(flipped * 100, 53.3, delta=0.05)

    def test_pose_alignment_scale(self):
        """20 度是工程初值：0° → 1.0，20° → 0.0，超过 20° 夹到 0。"""
        self.assertEqual(pose_alignment(0.0, 0.0), 1.0)
        self.assertAlmostEqual(pose_alignment(20.0, 0.0), 0.0, places=6)
        self.assertAlmostEqual(pose_alignment(0.0, 20.0), 0.0, places=6)
        self.assertEqual(pose_alignment(45.0, 45.0), 0.0)
        self.assertIsNone(pose_alignment(None, 1.0))
        self.assertIsNone(pose_alignment(1.0, None))


class TestWarmupAndDuration(unittest.TestCase):
    """没校准好 / 观察不够时**不出指数**，而且要说清是哪个原因。"""

    def test_no_index_before_calibration_locks(self):
        tracker = FocusTracker(FocusConfig(**FAST))
        first = warm(tracker, count=1)
        self.assertEqual(first.status, FocusStatus.WARMING_UP)
        self.assertIsNone(first.index)
        self.assertEqual(first.index_status, IDX_UNAVAILABLE)
        self.assertFalse(first.usable)

        second = warm(tracker, count=1, first_ts=0.5)
        self.assertEqual(second.status, FocusStatus.WARMING_UP)

        # 第三个样本（t=1.0）时长与样本数同时达标 → 校准锁定。
        third = warm(tracker, count=1, first_ts=1.0)
        self.assertEqual(third.status, FocusStatus.INSUFFICIENT_OBSERVATION)
        self.assertEqual(third.index_status, IDX_INSUFFICIENT)
        self.assertIsNotNone(third.raw_value, "校准好了就应当算出 raw，只是还不够时长")
        self.assertIsNone(third.index)

    def test_index_waits_for_min_valid_seconds(self):
        tracker = FocusTracker(FocusConfig(**FAST))
        snapshot = warm(tracker, count=2)
        self.assertEqual(snapshot.status, FocusStatus.WARMING_UP)

        for ts in (1.0, 1.5, 2.0, 2.5):
            snapshot = feed(tracker, ts)
            self.assertEqual(snapshot.index_status, IDX_INSUFFICIENT)
            self.assertEqual(snapshot.status, FocusStatus.INSUFFICIENT_OBSERVATION)
            self.assertIsNone(snapshot.index)
            self.assertFalse(snapshot.usable, "指数是 None 时绝不能 usable")

        snapshot = feed(tracker, 3.0)   # 有效观察累计到 2.0s
        self.assertEqual(snapshot.status, FocusStatus.VALID)
        self.assertEqual(snapshot.index_status, IDX_TREND)
        self.assertIsNotNone(snapshot.index)
        self.assertTrue(snapshot.usable)
        self.assertGreaterEqual(snapshot.valid_seconds, 2.0)

    def test_calibration_rejects_a_moving_head(self):
        """校准窗口里老人一直在转头 → 基线不可信 → 不出指数。

        样本跨度超限是**拒绝**而不是"凑合拿个中位数"：拿一个正在转头的人
        的中位数当基线，后面每一帧的头姿证据都是歪的，而且看不出来。

        角度取每秒 16°（0→40°），一路都贴着硬门（45°）内侧 —— 也就是说
        它会走到校准这一步，被**校准的范围判据**拒掉，而不是被硬门拦下。
        这也是本测试与 :meth:`TestGate.test_hard_pose_gate_uses_raw_angles_not_relative`
        的分工：那一条管"单人帧的可靠性"，这一条管"基线可不可信"。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        for index in range(6):
            snapshot = feed(tracker, index * 0.5, yaw=8.0 * index, pitch=0.0)
        self.assertEqual(snapshot.status, FocusStatus.WARMING_UP)
        self.assertIsNone(snapshot.index)
        self.assertGreater(snapshot.calibration["yaw_range_deg"], 8.0)
        self.assertFalse(snapshot.calibration["ready"])

    def test_default_config_keeps_the_reference_numbers(self):
        """默认值必须还是 5s / 30 样本 / 10s，否则演示出来的就不再是 VAI。"""
        config = FocusConfig()
        self.assertEqual(config.calibration_seconds, 5.0)
        self.assertEqual(config.calibration_min_samples, 30)
        self.assertEqual(config.min_valid_seconds_for_index, 10.0)
        self.assertEqual(config.ema_time_constant_seconds, 1.5)
        self.assertEqual(DEFAULT_WEIGHTS, {"gaze": 0.55, "pose": 0.25, "eye_open": 0.20})


class TestSilentDegradation(unittest.TestCase):
    """缺模态、极性、配对丢失 —— 都不报错，只让指数悄悄变成别的东西。"""

    def test_missing_gaze_renormalizes_the_weights(self):
        """取不到视线时权重从 0.55/0.25/0.20 变成 0.5556/0.4444。

        ``EvidenceFusion`` **不把缺失模态算进分母**（这是它的设计，不是 bug），
        所以同一分值的含义随模态组合而变。这就是 ``available_modalities``
        与 ``modality_completeness`` 必须一路带出去的原因。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker)
        snapshot = None
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            snapshot = feed(tracker, ts, gaze_off=None)

        self.assertEqual(snapshot.available_modalities, ("pose", "eye_open"))
        self.assertEqual(snapshot.modality_config_id, "头姿+睁眼")
        self.assertAlmostEqual(snapshot.normalized_weights["pose"], 0.25 / 0.45, places=4)
        self.assertAlmostEqual(snapshot.normalized_weights["eye_open"], 0.20 / 0.45, places=4)
        self.assertAlmostEqual(sum(snapshot.normalized_weights.values()), 1.0, places=4)
        self.assertAlmostEqual(snapshot.modality_completeness, 0.45, places=4)
        # 更狠的一档（连带 eye_open 一起丢）见下一条。

    def test_eye_state_unknown_collapses_to_pose_only_silently(self):
        """眼睑状态落到迟滞带 → ``gaze`` 与 ``eye_open`` **一起**消失，只剩头姿。

        这是**继承**来的行为，不是移植引入的：参考实现的状态机同样会在
        迟滞带里给 ``UNKNOWN``，而 ``gaze`` 的门是
        ``eye_state == OPEN and gaze_quality >= min``（``pipeline.py:354``），
        所以 ``UNKNOWN`` 会把两路证据一起带走，只剩 ``pose`` 一路、
        权重变成 ``{"pose": 1.0}``。指数照样出、照样平滑 —— 头姿正对就是
        100 分，**跟专注没有任何关系**。

        唯一能看出区别的就是 ``available_modalities`` / ``modality_completeness``，
        所以这条测试是那个哨兵。三个模态都齐的那种退化另见上一条。

        要走到这里得先有 EAR 基线（30 个样本）：先喂 31 帧正常睁眼（ear=0.29），
        再喂 ear=0.21 —— 它落在 ``ear_close_absolute``(0.20) 与
        ``ear_open_absolute``(0.22) **之间**，闭合与睁开两个信号都不成立
        → ``UNKNOWN``。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        for index in range(31):
            snapshot = feed(tracker, index * 0.1)
        self.assertEqual(snapshot.available_modalities, ("gaze", "pose", "eye_open"))

        snapshot = feed(tracker, 3.1, ear=0.21)
        self.assertEqual(snapshot.status, FocusStatus.VALID, "仍然算得出指数 —— 这正是危险之处")
        self.assertIsNotNone(snapshot.index)
        self.assertEqual(snapshot.available_modalities, ("pose",))
        self.assertEqual(snapshot.normalized_weights, {"pose": 1.0})
        self.assertAlmostEqual(snapshot.modality_completeness, 0.25, places=4)
        self.assertEqual(snapshot.modality_config_id, "仅头姿")

    def test_closed_eyes_suppress_gaze(self):
        """闭着眼"看向正前方"不是一个观测：凝视证据必须一并作废。

        与参考实现的 ``test_closed_eyes_suppress_gaze`` 同一件事。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker)
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            snapshot = feed(tracker, ts)
        self.assertEqual(snapshot.available_modalities, ("gaze", "pose", "eye_open"))

        snapshot = feed(tracker, 3.1, ear=0.15)
        self.assertEqual(snapshot.available_modalities, ("pose", "eye_open"))
        # 0.25*1.0(头姿正对) + 0.20*0.0(闭眼) = 0.25 → 权重重归一后 0.25/0.45
        self.assertAlmostEqual(snapshot.raw_value, round(0.25 / 0.45 * 100, 1), delta=0.05)

    def test_a_reading_without_its_frame_is_not_trusted(self):
        """视线报文配不上帧报文 → 当"看不见人"，而不是拿上一帧的头姿顶上。

        拿上一帧的 pose 顶上会得到一个**看起来正常**的指数，而实际输入
        已经断了一条 —— 那比不出指数坏得多。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        tracker.on_reading(reading_at(0.0))
        snapshot = tracker.snapshot()
        self.assertEqual(tracker.pair_misses, 1)
        self.assertEqual(snapshot.status, FocusStatus.NO_FACE)
        self.assertIsNone(snapshot.index)

    def test_stale_frame_does_not_pair(self):
        """过期很久的帧报文不能拿来配对新到的视线报文。"""
        tracker = FocusTracker(FocusConfig(**FAST))
        tracker.on_sample(frame_at(0.0))
        tracker.on_reading(reading_at(5.0))
        self.assertEqual(tracker.pair_misses, 1)
        self.assertEqual(tracker.snapshot().status, FocusStatus.NO_FACE)


class TestGate(unittest.TestCase):
    """裁剪版门控。裁掉了什么写在 ``core/focus.py`` 的 ``FocusGate`` 文档里。"""

    def test_no_face_is_never_a_zero_score(self):
        """没有人脸 ⇒ ``index is None``，**不是 0 分**。

        0 分是"极不专注"这样一个论断，而这里根本没有测量。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        snapshot = warm(tracker, has_face=False)
        self.assertEqual(snapshot.status, FocusStatus.NO_FACE)
        self.assertIsNone(snapshot.index)
        self.assertEqual(snapshot.index_status, IDX_UNAVAILABLE)

    def test_unassessable_frame_never_carries_an_index(self):
        """门控早退的那几档都必须 ``index is None``，而且**别停在上一档的分数上**。"""
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker)
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            snapshot = feed(tracker, ts)
        self.assertTrue(snapshot.usable)

        for kwargs, expected in (
            (dict(has_face=False), FocusStatus.NO_FACE),
            (dict(ear=0.0), FocusStatus.EYE_OCCLUDED),
            (dict(yaw=50.0), FocusStatus.OUT_OF_POSE_RANGE),
            (dict(pitch=-40.0), FocusStatus.OUT_OF_POSE_RANGE),
            (dict(roll=40.0), FocusStatus.OUT_OF_POSE_RANGE),
        ):
            snapshot = feed(tracker, 3.1, **kwargs)
            self.assertEqual(snapshot.status, expected, kwargs)
            self.assertIsNone(snapshot.index, kwargs)
            self.assertFalse(snapshot.usable, kwargs)

    def test_hard_pose_gate_uses_raw_angles_not_relative(self):
        """硬门量的是**相对镜头**的角度，所以拿原始角度判，不看基线。

        侧脸 50° 时地标本身就不可信，这跟老人平时习惯怎么坐无关。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        # 校准基线就建在 yaw=45° 上（贴着硬门内侧，能过门但过不了范围检查）——
        # 之后回到 45° 时"相对角度"是 0，但原始角度仍然越界 → 必须被拦。
        for index in range(3):
            snapshot = feed(tracker, index * 0.5, yaw=44.0, pitch=0.0)
        snapshot = feed(tracker, 1.5, yaw=46.0, pitch=0.0)
        self.assertEqual(snapshot.status, FocusStatus.OUT_OF_POSE_RANGE)
        self.assertIsNone(snapshot.index)


class TestSessionContinuity(unittest.TestCase):
    """断流、回退、长间隔 —— 基线必须跟着作废。"""

    def test_timestamp_rollback_resets_the_baseline(self):
        """A 重启（``--loop`` 每圈把 ts 归零、摄像头重连同理）→ 重新校准。

        基线是在旧时间轴上建的，留着它去减新时间轴的角度会得到一个
        **恒定偏移**，指数照出，谁也不会发现。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker)
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            snapshot = feed(tracker, ts)
        self.assertTrue(snapshot.usable)

        snapshot = feed(tracker, 0.1)   # 时间戳回退
        self.assertEqual(snapshot.status, FocusStatus.INSUFFICIENT_OBSERVATION)
        self.assertIsNone(snapshot.index)
        self.assertGreaterEqual(tracker.resets, 1)

        # 回退之后要**重新**做够校准才出指数。
        self.assertEqual(
            warm(tracker, count=2, first_ts=0.2).status, FocusStatus.WARMING_UP
        )

    def test_long_gap_resets_the_baseline(self):
        """观测中断超过 ``state_reset_gap_seconds`` → 会话重置。"""
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker)
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            snapshot = feed(tracker, ts)
        self.assertTrue(snapshot.usable)

        snapshot = feed(tracker, 6.0)   # 间隔 3.0s > 2.0s
        self.assertEqual(snapshot.status, FocusStatus.INSUFFICIENT_OBSERVATION)
        self.assertIsNone(snapshot.index)
        self.assertEqual(snapshot.index_status, IDX_UNAVAILABLE)

    def test_long_face_absence_resets_the_session(self):
        """连续看不见人够久 → 会话重置（不是"接着上次的分继续"）。"""
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker)
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            snapshot = feed(tracker, ts)
        self.assertTrue(snapshot.usable)
        resets_before = tracker.resets

        for ts in (3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2):
            snapshot = feed(tracker, ts, has_face=False)
        self.assertEqual(snapshot.status, FocusStatus.NO_FACE)
        self.assertGreater(tracker.resets, resets_before)

    def test_seconds_since_last_observation_uses_the_wall_clock(self):
        """新鲜度必须走墙钟：A 一挂，它的 ``timestamp`` 就永远停在最后一个值上。"""
        now = [1000.0]
        tracker = FocusTracker(FocusConfig(**FAST), clock=lambda: now[0])
        self.assertIsNone(tracker.seconds_since_last_observation())

        warm(tracker)
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            feed(tracker, ts)
        self.assertEqual(tracker.seconds_since_last_observation(), 0.0)

        # 报文时间戳不动（A 死了），墙钟往前走 —— 新鲜度要跟着变。
        now[0] += 30.0
        self.assertAlmostEqual(tracker.seconds_since_last_observation(), 30.0, places=3)

    def test_a_gap_long_enough_to_reset_is_treated_as_a_gap(self):
        """间隔 1.0s（恰好等于 ``max_sample_gap_seconds``）仍计入有效时长。"""
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker)
        for ts in (1.0, 2.0):
            snapshot = feed(tracker, ts)
        self.assertAlmostEqual(snapshot.valid_seconds, 1.0, places=3)


class TestStructuralRedLines(unittest.TestCase):
    """两条红线，用结构断言钉死，而不是靠"我们记得"。"""

    def test_no_virtual_baseline_ever_appears(self):
        """**绝不出现伪造基线。**

        ``non-contact/web_app.py`` 的 ``_build_virtual_baseline_observation``
        在没有人脸时伪造一个"假设正对镜头"的观测，好让门控与校准在演示时
        走通；识别它的唯一键是 ``metadata["virtual_baseline"]``。真实路径
        不产这个键 —— 包括**没有人脸**的那些帧。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        snapshots = [warm(tracker, has_face=False)]
        for ts in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            snapshots.append(feed(tracker, ts))
        for kwargs in (dict(has_face=False), dict(ear=0.0), dict(yaw=90.0)):
            snapshots.append(feed(tracker, 3.5, **kwargs))

        for snapshot in snapshots:
            keys = all_keys(snapshot)
            self.assertNotIn("virtual_baseline", keys)
            self.assertNotIn("virtual_baseline_observation", keys)

    def test_eye_evidence_is_a_declared_substitution(self):
        """凝视是"看没看正前方"，不是参考实现的"看没看任务目标区"。

        这条测试的作用是留个**路标**：读到这里的人应当去 ``core/focus.py``
        的模块文档看那条"不可比"的声明。
        """
        from core.focus import FocusTracker as _Tracker

        doc = _Tracker.__module__
        self.assertEqual(doc, "core.focus")
        self.assertIsNone(gaze_evidence(None))
        # 没有 target_roi_probability 这个入口 —— B 没有任务目标区。
        self.assertNotIn("target_roi", FocusReading.__dataclass_fields__)

    def test_snapshot_defaults_before_any_input(self):
        tracker = FocusTracker(FocusConfig(**FAST))
        snapshot = tracker.snapshot()
        self.assertIsNone(snapshot.index)
        self.assertEqual(snapshot.status, FocusStatus.NO_FACE)
        self.assertFalse(snapshot.usable)
        self.assertEqual(tracker.observations, 0)


class TestLiveness(unittest.TestCase):
    """``focus_is_live`` —— 能不能拿这个指数去做行为判决（静默模式）。

    三合一，而且**第三条（新鲜度）是这一组存在的全部理由**：
    ``FocusTracker`` 在 A 掉线时不会报错，它只是不再更新，``snapshot()``
    会一直返回最后一刻的值。只看前两条的话，老人 20 分钟前的那次专注
    会把主动关怀**无限期**关掉，而日志里一片安静。
    """

    def _live_tracker(self):
        """一个已经校准好、且有有效指数的 tracker，外加可推进的假墙钟。"""
        now = [5_000.0]
        tracker = FocusTracker(FocusConfig(**FAST), clock=lambda: now[0])
        warm(tracker, count=2)
        for ts in (1.0, 1.5, 2.0, 2.5, 3.0):
            feed(tracker, ts)
        self.assertTrue(tracker.snapshot().usable)
        return tracker, now

    def test_nothing_is_live_before_the_first_index(self):
        tracker = FocusTracker(FocusConfig(**FAST), clock=lambda: 0.0)
        live, why = focus_is_live(tracker, 2.0)
        self.assertFalse(live)
        self.assertTrue(why, "不可信时必须说得出原因（否则没法排查）")

    def test_live_once_the_index_is_out(self):
        tracker, _ = self._live_tracker()
        live, why = focus_is_live(tracker, 2.0)
        self.assertTrue(live)
        self.assertEqual(why, "")
        # 三合一的前两条合起来就是 usable —— 两者必须一致，
        # 否则"能不能用"会有两个答案。
        self.assertTrue(tracker.snapshot().usable)

    def test_not_live_while_observation_duration_is_short(self):
        """校准好了但有效观察时长不够 → 不可信，且原因要指得出这一条。"""
        tracker = FocusTracker(FocusConfig(**FAST), clock=lambda: 0.0)
        warm(tracker, count=2)
        feed(tracker, 1.0)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot.status, FocusStatus.INSUFFICIENT_OBSERVATION)
        live, why = focus_is_live(tracker, 2.0)
        self.assertFalse(live)
        self.assertIn(IDX_INSUFFICIENT, why)

    def test_stream_dying_makes_it_stale(self):
        """**A 掉线 → 静默必须自己解除。** 这是本函数最重要的一条。"""
        tracker, now = self._live_tracker()
        self.assertTrue(focus_is_live(tracker, 2.0)[0])

        # 报文不再来（A 挂了），墙钟继续走。tracker 自身**没有任何异常**：
        # snapshot() 照样返回上一次的分数。
        now[0] += 30.0
        self.assertIsNotNone(tracker.snapshot().index)
        self.assertTrue(tracker.snapshot().usable)

        live, why = focus_is_live(tracker, 2.0)
        self.assertFalse(live, "数据早就断了，不该继续拿旧分数关着关怀")
        self.assertIn("过期", why)

    def test_staleness_boundary_is_closed(self):
        """``age >= stale_seconds`` 即失效：等于门槛也判失效，不留灰区。"""
        tracker, now = self._live_tracker()
        now[0] += 2.0
        self.assertFalse(focus_is_live(tracker, 2.0)[0])
        # 差一点就还算数。
        tracker2, now2 = self._live_tracker()
        now2[0] += 1.9
        self.assertTrue(focus_is_live(tracker2, 2.0)[0])


class TestSilenceThreshold(unittest.TestCase):
    """``should_stay_silent`` —— **该不该为了"老人在专注"而不打扰**。

    这一组守的是一个**会让功能反向**的失效：把"指数可信"当成"老人专注"。
    脸在画面里、管线没坏，指数就一直有效 —— 扭头看窗外照样出分。
    只判可信的话，机器人在摄像头一通电之后就再也不主动开口，
    而日志里还写着"老人正在专注"。
    """

    #: 与 config.FOCUS_SILENT_MIN_INDEX 同源；这里写死是为了让测试
    #: 在有人改默认值时**照样**测的是语义，而不是跟着默认值一起漂。
    MIN_INDEX = 70.0

    def _focused_tracker(self):
        """正对镜头：VAI ≈ 91.8（见 :class:`TestFidelity`）。"""
        tracker = FocusTracker(FocusConfig(**FAST), clock=lambda: 5_000.0)
        warm(tracker, count=3)
        for ts in (1.5, 2.0, 2.5, 3.0):
            feed(tracker, ts)
        return tracker

    def _distracted_tracker(self):
        """扭头 + 视线离开：指数照样是有效值，但只有 40 分出头。"""
        tracker = FocusTracker(FocusConfig(**FAST), clock=lambda: 5_000.0)
        warm(tracker, count=3)
        for ts in (1.5, 2.0, 2.5, 3.0):
            feed(tracker, ts, gaze_off=0.9, yaw=22.0, pitch=14.0)
        return tracker

    def test_focused_elder_is_silenced(self):
        tracker = self._focused_tracker()
        silent, why = should_stay_silent(
            tracker, stale_seconds=2.0, min_index=self.MIN_INDEX)
        self.assertTrue(silent, f"91.8 分应当静默，实际：{why}")
        self.assertEqual(why, "")

    def test_distracted_elder_is_not_silenced(self):
        """**这一组存在的理由。** 有效但低分的指数绝不能关掉主动关怀。

        没有这道门槛的话，一个扭头看窗外 40 分钟的老人会得到 0 次主动关怀，
        而代码里每一个"指数可信"的断言都是绿的。
        """
        tracker = self._distracted_tracker()
        snapshot = tracker.snapshot()
        self.assertIsNotNone(snapshot.index, "前提：走神时指数是**有**的")
        self.assertLess(snapshot.index, self.MIN_INDEX)
        # 先确认它确实"可信" —— 否则这条测试会退化成在测可信性。
        self.assertTrue(focus_is_live(tracker, 2.0)[0])

        silent, why = should_stay_silent(
            tracker, stale_seconds=2.0, min_index=self.MIN_INDEX)
        self.assertFalse(silent, "走神时的指数不该让机器人闭嘴")
        self.assertIn("门槛", why)

    def test_threshold_is_inclusive(self):
        """``index >= min_index`` 即静默：正好等于门槛时算专注，不留灰区。"""
        tracker = self._distracted_tracker()
        index = tracker.snapshot().index
        self.assertTrue(should_stay_silent(
            tracker, stale_seconds=2.0, min_index=index)[0])
        self.assertFalse(should_stay_silent(
            tracker, stale_seconds=2.0, min_index=index + 0.05)[0])

    def test_pose_only_reading_must_not_silence(self):
        """**实测抓到的反向失效。** 缺模态 → 权重重归一化 → 只剩头姿时满分 100。

        场景：一位长者眼睛的状态判不出来（EAR 基线锁在偏低值上，睁闭都看不出），
        于是视线与睁眼两路一起被门控拿掉，只剩头姿正对镜头
        ⇒ ``pose_alignment=1.0``，权重归一化成 ``{"pose": 1.0}`` ⇒ **指数 100.0**。

        拿它去关掉主动关怀就是"越困越安静"。所以静默要求证据齐全，
        而不是"指数有值且够高"。
        """
        tracker = FocusTracker(FocusConfig(**FAST))
        warm(tracker, count=3)
        # 要喂够**一个证据窗口**（2s）：窗口里还留着校准时那几帧的
        # gaze+eye_open，而它们确实"最近还活着"。窗口本身是这组测试的前提。
        for ts in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
            # ear=0.20 恒定：基线锁在同一个值上，睁闭都判不出来。
            # **头仍然冲着镜头**（角度与校准时相同 ⇒ 相对量 0）—— 这正是
            # 现场那个画面：人低头/闭眼，头却还对着摄像头。
            feed(tracker, ts, ear=0.20)

        snapshot = tracker.snapshot()
        # 收敛值是 100.0（权重只剩头姿这一路，而它量到的是"正对"）。这里不钉
        # 精确值：EAR 基线还在微调，头几帧可能带着一路 eye_open=0.0 进来，
        # EMA 因此从略低的分数往上爬。要紧的是"**高于门槛**，但一路证据都不全"。
        self.assertGreater(snapshot.index, self.MIN_INDEX)
        self.assertNotIn("gaze", snapshot.available_modalities)
        self.assertNotIn("eye_open", snapshot.available_modalities)
        self.assertTrue(focus_is_live(tracker, 2.0)[0], "前提：它同时是'可信'的")

        silent, why = should_stay_silent(
            tracker, stale_seconds=2.0, min_index=self.MIN_INDEX)
        self.assertFalse(silent, "证据不全的满分不该让机器人闭嘴")
        self.assertIn("证据不全", why)
        self.assertIn("gaze", why)

    def test_a_blink_does_not_release_the_silence(self):
        """**眨眼不许把静默模式打穿。** 实测过：每 4 秒一次眨眼，静默跟着抖，
        而每次抖动都放行一次主动关怀 —— 一次眨眼就让机器人开口了。

        眨眼那一帧 ``eye_state`` 不是 OPEN，视线与睁眼两路一起被门控拿掉；
        判据看的是**窗口并集**，所以结论不受影响。而需求文档 §4 B2 里
        「有规律眨眼」本来就是**专注**的一半判据，逐帧判等于把它当成了干扰。
        """
        tracker = self._focused_tracker()
        self.assertTrue(should_stay_silent(
            tracker, stale_seconds=2.0, min_index=self.MIN_INDEX)[0])

        # 眨一下：这一帧 ear 掉到闭合区、且**不给视线**（照 A 的实际行为）。
        blink = feed(tracker, 3.1, ear=0.14, gaze_off=None)
        self.assertNotIn("gaze", blink.available_modalities,
                         "前提：眨眼那一帧视线确实被门控拿掉了")
        self.assertIn("gaze", blink.recent_modalities,
                      "窗口并集里应当还有视线 —— 它刚才是活的")

        silent, why = should_stay_silent(
            tracker, stale_seconds=2.0, min_index=self.MIN_INDEX)
        self.assertTrue(silent, f"眨一次眼不该解除静默，实际：{why}")

    def test_trust_is_checked_before_the_threshold(self):
        """不可信时先报不可信的原因，**不引用分数**。

        旧分数可能是几十分钟前那个 90 分，拿它跟门槛比会得出"很专注" ——
        这正是新鲜度那一条想防的事，所以顺序不能反。
        """
        now = [5_000.0]
        tracker = FocusTracker(FocusConfig(**FAST), clock=lambda: now[0])
        warm(tracker, count=3)
        for ts in (1.5, 2.0, 2.5, 3.0):
            feed(tracker, ts)
        self.assertGreaterEqual(tracker.snapshot().index, self.MIN_INDEX)

        now[0] += 30.0  # A 掉线，分数还停在 90 出头
        silent, why = should_stay_silent(
            tracker, stale_seconds=2.0, min_index=self.MIN_INDEX)
        self.assertFalse(silent)
        self.assertIn("过期", why)
        self.assertNotIn("门槛", why, "该说的是数据断了，不是分数不够")


class TestVisionWiring(unittest.TestCase):
    """§10 的接线：两条流都必须进 tracker，而且**两条路都要接**。

    实时（VisionClient）与离线（OfflineVisionFeeder）各写一遍接线是这套代码里
    最容易只接一半的地方 —— 而"回放时静默模式没生效"只会在演示当天发现。
    """

    #: 列序照 api_doc §3.4 再补 §3.5 的类型化列（与 test_typed_messages 一致）。
    FIELDS = ("timestamp", "has_face", "ear", "blink_cnt", "pitch", "yaw", "roll",
              "emo_feature", "type", "gaze", "gaze_quality")

    @classmethod
    def _row(cls, **kw) -> str:
        unknown = set(kw) - set(cls.FIELDS)
        assert not unknown, f"列名不在列序里：{sorted(unknown)}"
        return ",".join(str(kw.get(name, "")) for name in cls.FIELDS)

    def _write_csv(self, rows):
        fd, path = tempfile.mkstemp(suffix=".csv", text=True)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fp:
            fp.write(",".join(self.FIELDS) + "\n")
            for row in rows:
                fp.write(row + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_live_client_feeds_both_streams(self):
        """一帧 + 同一刻的视线报文 → 配对成功（``pair_misses == 0``）。"""
        tracker = FocusTracker(FocusConfig(**FAST))
        client = VisionClient(VisionStateEvaluator(), focus=tracker)

        frame = {"timestamp": 0.0, "has_face": True, "ear": 0.29, "blink_cnt": 0,
                 "pitch": 5.0, "yaw": 10.0, "roll": 0.0, "emo_feature": "normal"}
        focus = {"type": "focus", "timestamp": 0.0, "gaze": GAZE_OFF,
                 "gaze_quality": 1.0}

        client._handle_line(json.dumps(frame))
        client._handle_line(json.dumps(focus))

        self.assertEqual(client.frames_received, 1)
        self.assertEqual(tracker.readings_seen, 1)
        self.assertEqual(tracker.pair_misses, 0,
                         "帧报文没喂进 tracker —— 头姿证据会悄悄消失")
        self.assertEqual(tracker.snapshot().status, FocusStatus.WARMING_UP)

    def test_live_client_without_a_tracker_still_works(self):
        """关掉专注静默（``focus=None``）时，视线报文照收照计数、只是没人消费。"""
        client = VisionClient(VisionStateEvaluator(), focus=None)
        client._handle_line(json.dumps(
            {"type": "focus", "timestamp": 0.0, "gaze": 0.1, "gaze_quality": 1.0}))
        self.assertEqual(client.typed.counts.get("focus"), 1)
        self.assertEqual(client.frames_received, 0)

    def test_offline_feeder_feeds_both_streams(self):
        """离线回放同样要接 —— 否则 ``--offline`` 演示时静默模式根本不存在。"""
        path = self._write_csv([
            self._row(timestamp=0.0, has_face="true", ear=0.29, blink_cnt=0,
                      pitch=5.0, yaw=10.0, roll=0.0, emo_feature="normal"),
            self._row(timestamp=0.0, type="focus", gaze=GAZE_OFF, gaze_quality=1.0),
            self._row(timestamp=0.2, has_face="true", ear=0.29, blink_cnt=1,
                      pitch=5.0, yaw=10.0, roll=0.0, emo_feature="normal"),
            self._row(timestamp=0.2, type="focus", gaze=GAZE_OFF, gaze_quality=1.0),
        ])
        tracker = FocusTracker(FocusConfig(**FAST))
        feeder = OfflineVisionFeeder(path, VisionStateEvaluator(),
                                     speed=1000.0, loop=False, focus=tracker)
        feeder.run()

        self.assertEqual(feeder.frames_received, 2)
        self.assertEqual(tracker.readings_seen, 2)
        self.assertEqual(tracker.pair_misses, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
