"""api_doc §3.2 投影层与契约闸的单元测试。

标准库 unittest，不依赖 pytest、不依赖 socket、不依赖摄像头 —— 对齐
``backend_B/tests/test_core.py`` 的体例，让"跑测试"这件事在任何一台
机器上都是一条命令。

运行::

    cd backend_A && python -m unittest discover -s tests -v
    cd backend_A && python tests/test_v1_contract.py
"""

from __future__ import annotations

import os
import sys
import unittest

# 允许 `python tests/test_v1_contract.py` 这种直接执行的方式：
# 把 backend_A 放进 sys.path，好让 `module_a_vision` 与 `shared` 能被导入。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _console  # noqa: E402,F401  （导入即生效：让中文输出不乱码）

from shared.enums import Emotion  # noqa: E402
from shared.frame_features import FrameFeatures, FrameQuality  # noqa: E402

from module_a_vision.metrics.eye import EyeTracker  # noqa: E402
from module_a_vision.wire import (  # noqa: E402
    EMO_FEATURE_VALUES,
    EMOTION_TO_FEATURE,
    V1_FIELDS,
    assert_v1_contract,
    to_v1_sample,
)


def make_frame(
    ts: float = 1.0,
    *,
    has_face: bool = True,
    usable: bool = True,
    ear_left: float = 0.30,
    ear_right: float = 0.20,
    pitch: float = 1.0,
    yaw: float = -2.0,
    roll: float = 3.0,
) -> FrameFeatures:
    """造一帧特征。只填投影层关心的字段，其余走默认值。

    注意用的是 :class:`FrameQuality`（字段名 ``valid``），不是
    ``shared.schema.Quality``（字段名 ``usable``）—— 后者是**窗口级**质量。
    这两个类名字像、语义不同，混用会得到一个非常难查的 TypeError。
    """
    return FrameFeatures(
        ts=ts,
        has_face=has_face,
        ear_left=ear_left,
        ear_right=ear_right,
        pitch_deg=pitch,
        yaw_deg=yaw,
        roll_deg=roll,
        quality=FrameQuality(valid=usable),
    )


class TestProjection(unittest.TestCase):
    """投影：单帧 → §3.2 报文。"""

    def test_keys_exactly_match_contract(self) -> None:
        """键集合**精确等于** 8 个字段。

        这是整套测试里最重要的一条。B 对缺失字段静默回退默认值，
        所以"少一个键"不会报错、不会断连，只会让状态永远停在 absent。
        多一个键同样不行：那说明有人往 §3.2 里塞了规范外的东西
        （比如 V2 的 ``type``），而 B 认不出。
        """
        payload = to_v1_sample(
            make_frame(), blink_total=7, emotion=Emotion.NORMAL
        )
        self.assertEqual(set(payload), set(V1_FIELDS))
        self.assertEqual(len(V1_FIELDS), 8)

    def test_ear_is_mean_of_both_eyes(self) -> None:
        """``ear`` 是左右眼均值 —— 用不对称的输入才测得出来。"""
        payload = to_v1_sample(
            make_frame(ear_left=0.30, ear_right=0.20),
            blink_total=0,
            emotion=Emotion.NORMAL,
        )
        self.assertAlmostEqual(payload["ear"], 0.25, places=4)

    def test_pose_passthrough(self) -> None:
        payload = to_v1_sample(
            make_frame(pitch=12.5, yaw=-7.25, roll=33.0),
            blink_total=0,
            emotion=Emotion.NORMAL,
        )
        self.assertAlmostEqual(payload["pitch"], 12.5)
        self.assertAlmostEqual(payload["yaw"], -7.25)
        self.assertAlmostEqual(payload["roll"], 33.0)

    def test_timestamp_comes_from_frame(self) -> None:
        payload = to_v1_sample(
            make_frame(ts=42.125), blink_total=0, emotion=Emotion.NORMAL
        )
        self.assertAlmostEqual(payload["timestamp"], 42.125)

    def test_types_are_exact(self) -> None:
        """类型要**精确**，不能是"能被当成"。"""
        payload = to_v1_sample(
            make_frame(), blink_total=3, emotion=Emotion.NORMAL
        )
        self.assertIs(type(payload["has_face"]), bool)
        self.assertIs(type(payload["blink_cnt"]), int)
        for name in ("timestamp", "ear", "pitch", "yaw", "roll"):
            self.assertIs(type(payload[name]), float, f"{name} 应是 float")


class TestEmotionMapping(unittest.TestCase):
    """A 的四个表情标签 → 文档的三个取值。"""

    def test_all_four_labels_are_mapped(self) -> None:
        """每个标签都要有映射，不能靠 ``.get(..., "normal")`` 兜底蒙混。

        兜底那条路存在是为了防御"将来加了第五个标签"，但**现有的四个
        必须显式列出** —— 否则新加一个标签会被静默压成 normal，
        表现为"这个情绪系统看不见"，而不是一个显眼的 KeyError。
        """
        for emotion in Emotion:
            self.assertIn(emotion, EMOTION_TO_FEATURE, f"{emotion} 没有映射")

    def test_sad_and_upset_both_collapse_to_low(self) -> None:
        """文档只有三个取值，sad 与 upset 都得压到 low。"""
        self.assertEqual(EMOTION_TO_FEATURE[Emotion.SAD], "low")
        self.assertEqual(EMOTION_TO_FEATURE[Emotion.UPSET], "low")

    def test_projection_never_emits_value_outside_contract(self) -> None:
        for emotion in Emotion:
            payload = to_v1_sample(
                make_frame(), blink_total=0, emotion=emotion
            )
            self.assertIn(payload["emo_feature"], EMO_FEATURE_VALUES)

    def test_tired_maps_to_tired(self) -> None:
        """``Emotion.TIRED`` 这一支代码可达、数据不可达。

        表情分类器有 ``TIRED_DAMPING`` 阻尼，标签在结构上不会变成 tired。
        这条测试钉住的是**映射本身**没写错；不要因为它是绿的
        就以为 ``emo_feature=tired`` 会在实跑中出现 —— 它不会。
        """
        payload = to_v1_sample(
            make_frame(), blink_total=0, emotion=Emotion.TIRED
        )
        self.assertEqual(payload["emo_feature"], "tired")


class TestNoFaceFrames(unittest.TestCase):
    """无人脸 / 画面不可用的帧。"""

    def test_no_face_uses_defaults(self) -> None:
        payload = to_v1_sample(
            make_frame(has_face=False), blink_total=5, emotion=Emotion.NORMAL
        )
        self.assertIs(payload["has_face"], False)
        for name in ("ear", "pitch", "yaw", "roll"):
            self.assertEqual(payload[name], 0.0, f"无人脸时 {name} 应为 0")
        self.assertEqual(payload["emo_feature"], "normal")

    def test_unusable_quality_is_reported_as_no_face(self) -> None:
        """画面不可用要压成 ``has_face=False``，不能报 true + EAR 0。

        报 true 的话，B 的 ``_low_ear_seconds`` 会把那串 0 读成
        "EAR 持续低于阈值"，几秒后报出 tired —— 一个纯粹由缺字段
        制造出来的**假疲劳**。这条测试就是钉住那个失效模式。
        """
        payload = to_v1_sample(
            make_frame(has_face=True, usable=False),
            blink_total=0,
            emotion=Emotion.NORMAL,
        )
        self.assertIs(payload["has_face"], False)
        self.assertEqual(payload["ear"], 0.0)

    def test_blink_total_survives_no_face_frames(self) -> None:
        """无人脸时**不清零** blink_cnt：它是事件计数器，不是测量量。"""
        payload = to_v1_sample(
            make_frame(has_face=False), blink_total=9, emotion=Emotion.NORMAL
        )
        self.assertEqual(payload["blink_cnt"], 9)

    def test_no_face_frames_pass_contract(self) -> None:
        for has_face, usable in ((False, True), (True, False), (False, False)):
            payload = to_v1_sample(
                make_frame(has_face=has_face, usable=usable),
                blink_total=0,
                emotion=Emotion.NORMAL,
            )
            self.assertEqual(assert_v1_contract(payload), [])


class TestContractGate(unittest.TestCase):
    """契约闸：该拦的要拦住，不该拦的别误伤。"""

    def clean(self) -> dict:
        return to_v1_sample(
            make_frame(), blink_total=1, emotion=Emotion.NORMAL
        )

    def test_clean_payload_passes(self) -> None:
        self.assertEqual(assert_v1_contract(self.clean()), [])

    def test_missing_field_is_caught(self) -> None:
        payload = self.clean()
        del payload["blink_cnt"]
        problems = assert_v1_contract(payload)
        self.assertTrue(problems)
        self.assertIn("blink_cnt", " ".join(problems))

    def test_extra_field_is_caught(self) -> None:
        """V2 的 ``type`` 字段混进来要被拦下。"""
        payload = self.clean()
        payload["type"] = "vision_window"
        problems = assert_v1_contract(payload)
        self.assertTrue(problems)
        self.assertIn("多余", " ".join(problems))

    def test_bool_is_not_accepted_as_int(self) -> None:
        """``isinstance(True, int)`` 为真 —— 所以必须用 ``type() is int``。

        如果闸门用的是 isinstance，``blink_cnt=True`` 会被放行，
        B 那边 ``_blink_rate_per_minute`` 拿布尔做减法，得到一个
        算得出、但没有任何意义的数字。
        """
        payload = self.clean()
        payload["blink_cnt"] = True
        self.assertTrue(assert_v1_contract(payload))

    def test_non_bool_has_face_is_caught(self) -> None:
        payload = self.clean()
        payload["has_face"] = 1
        problems = assert_v1_contract(payload)
        self.assertTrue(problems)
        self.assertIn("has_face", " ".join(problems))

    def test_non_finite_float_is_caught(self) -> None:
        """NaN / inf 会被 JSON 编成 ``NaN`` / ``Infinity``，那不是合法 JSON。"""
        for bad in (float("nan"), float("inf"), float("-inf")):
            payload = self.clean()
            payload["ear"] = bad
            self.assertTrue(
                assert_v1_contract(payload), f"{bad} 应被拦下"
            )

    def test_string_number_is_caught(self) -> None:
        payload = self.clean()
        payload["ear"] = "0.3"
        self.assertTrue(assert_v1_contract(payload))

    def test_bad_emo_feature_is_caught(self) -> None:
        payload = self.clean()
        payload["emo_feature"] = "sad"
        problems = assert_v1_contract(payload)
        self.assertTrue(problems)
        self.assertIn("emo_feature", " ".join(problems))

    def test_gate_reports_every_problem_not_just_the_first(self) -> None:
        """一次报全，别让人修一个跑一次。"""
        payload = self.clean()
        del payload["ear"]
        payload["blink_cnt"] = True
        payload["emo_feature"] = "???"
        self.assertGreaterEqual(len(assert_v1_contract(payload)), 3)


class TestBlinkCounter(unittest.TestCase):
    """累计眨眼计数器 —— 这条链路上最容易写错的一个值。"""

    def _blink_once(self, tracker: EyeTracker, ts: float) -> None:
        """在 ``ts`` 附近制造一次完整眨眼（闭合 → 睁开）。"""
        tracker.update(make_frame(ts=ts), closure_ratio=0.9)
        tracker.update(make_frame(ts=ts + 0.1), closure_ratio=0.0)

    def test_counter_increases_by_one_per_blink(self) -> None:
        tracker = EyeTracker()
        self.assertEqual(tracker.blink_total, 0)
        for i in range(3):
            self._blink_once(tracker, 10.0 + i)
        self.assertEqual(tracker.blink_total, 3)

    def test_counter_never_decreases_past_the_rolling_window(self) -> None:
        """**这条是核心回归。**

        ``len(tracker._blinks)`` 被 ``_evict`` 按 60 秒滚动裁掉，会变小。
        拿它当 ``blink_cnt`` 就是伪造"眨眼次数倒退"，而 B 的
        ``_blink_rate_per_minute`` 检测到倒退会返回 None ——
        等于**静默关掉一条疲劳判据**，不报错、不告警。

        这里眨眼 + 推进远超 60 秒的时间，断言累计值只增不减。
        """
        tracker = EyeTracker()
        seen: list[int] = []
        ts = 0.0
        for i in range(80):
            self._blink_once(tracker, ts)
            seen.append(tracker.blink_total)
            ts += 5.0  # 400 秒，远超 60 秒窗口

        self.assertEqual(seen, sorted(seen), "blink_total 出现了倒退")
        self.assertEqual(tracker.blink_total, 80)
        # 而滚动窗口里的记录数远小于累计值 —— 两者确实不是一回事。
        self.assertLess(len(tracker._blinks), tracker.blink_total)

    def test_counter_ignores_long_closures(self) -> None:
        """闭眼超过 ``MAX_BLINK_SEC`` 算"持续闭眼"，不是眨眼，不计数。

        这一条重要：长时间闭眼是疲劳信号，把它算成眨眼会让
        ``blink_cnt`` 在"老人睡着了"的时候反而涨得最快。
        """
        tracker = EyeTracker()
        tracker.update(make_frame(ts=0.0), closure_ratio=0.9)
        tracker.update(make_frame(ts=5.0), closure_ratio=0.0)  # 闭了 5 秒
        self.assertEqual(tracker.blink_total, 0)

    def test_snapshot_carries_counter(self) -> None:
        """``close_window()`` 走的 ``snapshot()`` 也要带出累计值。"""
        tracker = EyeTracker()
        self._blink_once(tracker, 1.0)
        metrics = tracker.snapshot([], now_ts=1.1)
        self.assertEqual(metrics.blink_total, 1)

    def test_projected_counter_is_monotonic_over_a_frame_stream(self) -> None:
        """端到端那条断言，只是在纯函数层面先跑一遍：

        喂一串帧，逐帧投影，``blink_cnt`` 必须单调不减。
        """
        tracker = EyeTracker()
        counters: list[int] = []
        ts = 0.0
        for i in range(60):
            frame = make_frame(ts=ts)
            tracker.update(frame, closure_ratio=0.9 if i % 4 == 0 else 0.0)
            payload = to_v1_sample(
                frame, blink_total=tracker.blink_total, emotion=Emotion.NORMAL
            )
            counters.append(payload["blink_cnt"])
            # 步长必须 **小于** MAX_BLINK_SEC（0.4s），否则每次"闭合"都跨过
            # 眨眼上限、被判成持续闭眼，于是一次都不计数 —— 这条测试会
            # 因为"没有眨眼可数"而通过单调性断言，却测不到任何东西。
            ts += 0.3

        self.assertEqual(counters, sorted(counters))
        self.assertGreater(counters[-1], 0, "整段没有一次眨眼，测试本身失效了")


if __name__ == "__main__":
    unittest.main(verbosity=2)
