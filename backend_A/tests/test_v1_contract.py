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
    FOCUS_FIELDS,
    KIND_FRAME,
    MSG_FOCUS,
    MSG_RPPG,
    RPPG_FIELDS,
    V1_FIELDS,
    UnknownMessageTypeError,
    assert_focus_contract,
    assert_rppg_contract,
    assert_v1_contract,
    build_focus_payload,
    build_rppg_payload,
    check_contract,
    to_focus_payload,
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
    gaze: float | None = None,
) -> FrameFeatures:
    """造一帧特征。只填投影层关心的字段，其余走默认值。

    注意用的是 :class:`FrameQuality`（字段名 ``valid``），不是
    ``shared.schema.Quality``（字段名 ``usable``）—— 后者是**窗口级**质量。
    这两个类名字像、语义不同，混用会得到一个非常难查的 TypeError。

    ``gaze`` 默认 ``None``（"估不出来"），这是 :class:`FrameFeatures` 的
    默认值，也是 CSV 回放的真实取值。要在 ``focus`` 投影里得到一条可用
    视线就必须显式传它 —— 这个"必须显式"正是我们要的：默认值不能是
    ``0.0``，那会静默伪造"视线完全对正"。
    """
    return FrameFeatures(
        ts=ts,
        has_face=has_face,
        ear_left=ear_left,
        ear_right=ear_right,
        pitch_deg=pitch,
        yaw_deg=yaw,
        roll_deg=roll,
        gaze_off_ratio=gaze,
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
        """规范外的字段混进来要被拦下。"""
        payload = self.clean()
        payload["emotion"] = "happy"
        problems = assert_v1_contract(payload)
        self.assertTrue(problems)
        self.assertIn("多余", " ".join(problems))

    def test_type_key_still_makes_it_not_a_frame(self) -> None:
        """**即使 ``type`` 的值是 ``"frame"``，它也不再是一帧。**

        这条看着多余，其实是**唯一**能拦住"照着 §3.5 草案把
        ``type:"frame"`` 加回来"的守卫。草案里曾写过"现有视觉帧标
        ``type:"frame"``"，实现刻意没那么做（见
        :mod:`module_a_vision.wire` 的模块文档）。将来有人翻到旧文档、
        照着改回来时，契约闸会在这里把他拦下 —— 否则那一个键会一路
        走到 B，而 B 那边表现是"状态永远停在 absent"。
        """
        payload = {"type": "frame", **self.clean()}
        problems = assert_v1_contract(payload)
        self.assertTrue(problems, "带了 type 的帧报文必须被判为不合规")
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


class TestRouting(unittest.TestCase):
    """分流：帧 / rppg / focus，以及"第三样东西"。"""

    def test_frame_has_no_type_and_routes_to_frame(self) -> None:
        payload = to_v1_sample(make_frame(), blink_total=0, emotion=Emotion.NORMAL)
        kind, problems = check_contract(payload)
        self.assertEqual(kind, KIND_FRAME)
        self.assertEqual(problems, [])

    def test_focus_packet_is_not_sent_to_the_frame_gate(self) -> None:
        """**这条是分流 bug 的直接守卫。**

        曾经的写法是 ``if type == "rppg": … else: assert_v1_contract(…)``。
        那样 ``focus`` 会落进 ``else``、被 frame 闸判成"多余字段"、
        在 ``broadcast`` 里丢弃 —— 于是 VAI 永远拿不到视线数据，
        而**所有"报文格式正确"的测试全都是绿的**。

        这里断言 focus 走的是它自己那条路，而不是被 frame 闸判违规。
        """
        payload = build_focus_payload(1.0, gaze=0.1, gaze_quality=1.0)
        kind, problems = check_contract(payload)
        self.assertEqual(kind, MSG_FOCUS)
        self.assertEqual(problems, [])
        # 反过来确认：它**确实**过不了 frame 闸（所以"兜底成帧"
        # 那种写法一定会静默丢弃它，而不是碰巧能用）。
        self.assertTrue(assert_v1_contract(payload))

    def test_unknown_type_raises_instead_of_falling_back(self) -> None:
        """未知类型必须抛，不能兜底当帧处理。"""
        for bogus in ("vision_window", "heartbeat", "frame", "rppg2", ""):
            with self.subTest(type=bogus):
                with self.assertRaises(UnknownMessageTypeError):
                    check_contract({"type": bogus, **{f: 0 for f in V1_FIELDS}})

    def test_unknown_message_type_error_is_a_value_error(self) -> None:
        """继承 ``ValueError`` 是为了让 ``main()`` 现有的 except 接得住它。

        接不住的话，配置错误会以一整段栈回溯的形式砸在用户脸上，
        而不是 ``[A] ✗ 运行中断：…`` 那样一行说明。
        """
        self.assertTrue(issubclass(UnknownMessageTypeError, ValueError))


class TestRppgContract(unittest.TestCase):
    """§3.5 体征报文的契约闸。"""

    def clean(self) -> dict:
        return build_rppg_payload(12.5, hr=75, rr=19.1, ibi_ms=[780, 780, 546])

    def test_clean_payload_passes(self) -> None:
        payload = self.clean()
        self.assertEqual(set(payload), set(RPPG_FIELDS))
        self.assertEqual(assert_rppg_contract(payload), [])

    def test_null_hr_is_legal(self) -> None:
        """**``hr: null`` 是正常态。** 窗口未满 / SNR 不够 / 帧率过低时
        ``Rppg`` 都会置灰，而这三种情况在实跑里都很常见。任何断言
        "收到体征报文 ⇒ hr 是数字"的测试都是错的。
        """
        payload = build_rppg_payload(0.0, hr=None, rr=None, ibi_ms=[])
        self.assertIsNone(payload["hr"])
        self.assertEqual(assert_rppg_contract(payload), [])

    def test_hr_and_rr_range_is_enforced(self) -> None:
        for bad in (5, 300, -1):
            with self.subTest(hr=bad):
                payload = self.clean()
                payload["hr"] = bad
                self.assertTrue(assert_rppg_contract(payload), f"hr={bad} 应被拦下")
        for bad in (0.5, 90.0):
            with self.subTest(rr=bad):
                payload = self.clean()
                payload["rr"] = bad
                self.assertTrue(assert_rppg_contract(payload), f"rr={bad} 应被拦下")

    def test_bool_is_not_accepted_as_a_number(self) -> None:
        payload = self.clean()
        payload["hr"] = True
        self.assertTrue(assert_rppg_contract(payload))

    def test_ibi_entries_must_be_ints_in_range(self) -> None:
        for bad in ([700.5], [0], [99999], ["700"]):
            with self.subTest(ibi=bad):
                payload = self.clean()
                payload["ibi_ms"] = bad
                self.assertTrue(assert_rppg_contract(payload), f"ibi={bad} 应被拦下")

    def test_ibi_must_be_an_array(self) -> None:
        payload = self.clean()
        payload["ibi_ms"] = 700
        self.assertTrue(assert_rppg_contract(payload))

    def test_missing_and_extra_fields_are_caught(self) -> None:
        payload = self.clean()
        del payload["rr"]
        self.assertIn("缺少", " ".join(assert_rppg_contract(payload)))

        payload = self.clean()
        payload["sqi"] = 0.8
        self.assertIn("多余", " ".join(assert_rppg_contract(payload)))

    def test_wrong_type_tag_is_caught(self) -> None:
        payload = self.clean()
        payload["type"] = MSG_FOCUS
        self.assertTrue(assert_rppg_contract(payload))


class TestFocusContract(unittest.TestCase):
    """§3.5 视线报文的契约闸。"""

    def clean(self) -> dict:
        return build_focus_payload(1.0, gaze=0.05, gaze_quality=1.0)

    def test_clean_payload_passes(self) -> None:
        payload = self.clean()
        self.assertEqual(set(payload), set(FOCUS_FIELDS))
        self.assertEqual(assert_focus_contract(payload), [])

    def test_unavailable_gaze_is_legal_and_quality_must_be_zero(self) -> None:
        """``gaze: null`` 合法，但此时 ``gaze_quality`` 必须是 0。"""
        payload = build_focus_payload(2.0, gaze=None, gaze_quality=0.0)
        self.assertIsNone(payload["gaze"])
        self.assertEqual(assert_focus_contract(payload), [])

    def test_gaze_and_quality_must_agree(self) -> None:
        """**不变式：``gaze is None`` ⟺ ``gaze_quality == 0``。**

        两头都要查。只查"null ⇒ 0"会放过第二行那种：``gaze`` 明明有值
        却报质量为 0 —— 对端会把它当成不可信而丢掉一个有效观测，
        而且丢掉的过程没有任何痕迹。
        """
        payload = self.clean()
        payload["gaze"] = None
        payload["gaze_quality"] = 1.0
        self.assertTrue(assert_focus_contract(payload), "null + 质量 1.0 应被拦下")

        payload = self.clean()
        payload["gaze_quality"] = 0.0
        self.assertTrue(assert_focus_contract(payload), "有值 + 质量 0 应被拦下")

    def test_gaze_range_is_enforced(self) -> None:
        for bad in (-0.1, 1.5):
            with self.subTest(gaze=bad):
                payload = self.clean()
                payload["gaze"] = bad
                self.assertTrue(assert_focus_contract(payload))

    def test_gaze_accepts_zero_and_one(self) -> None:
        """0.0 是"正对着镜头"，必须是合法值 —— 它和 ``null`` 不是一回事。"""
        for good in (0.0, 1.0):
            with self.subTest(gaze=good):
                payload = build_focus_payload(1.0, gaze=good, gaze_quality=1.0)
                self.assertEqual(assert_focus_contract(payload), [])

    def test_bool_gaze_is_caught(self) -> None:
        payload = self.clean()
        payload["gaze"] = True
        self.assertTrue(assert_focus_contract(payload))

    def test_missing_and_extra_fields_are_caught(self) -> None:
        payload = self.clean()
        del payload["gaze_quality"]
        self.assertIn("缺少", " ".join(assert_focus_contract(payload)))

        payload = self.clean()
        payload["target_roi_probability"] = 0.7
        self.assertIn("多余", " ".join(assert_focus_contract(payload)))


class TestFocusFromFrame(unittest.TestCase):
    """``to_focus_payload``：一帧特征 → §3.5 视线报文。

    这是 ``focus`` 通道的**唯一**构造点（``server._emit_focus`` 与
    ``main._dry_run_v1`` 都调它）。两条路径共用它，是为了让"实跑的流"
    与"--dry-run 打出来的流"不可能分叉。
    """

    def test_usable_gaze_becomes_measured(self) -> None:
        payload = to_focus_payload(make_frame(ts=3.0, gaze=0.25))
        self.assertEqual(set(payload), set(FOCUS_FIELDS))
        self.assertEqual(payload["type"], MSG_FOCUS)
        self.assertAlmostEqual(payload["gaze"], 0.25, places=4)
        self.assertEqual(payload["gaze_quality"], 1.0)
        self.assertEqual(assert_focus_contract(payload), [])

    def test_frame_without_gaze_is_null_not_zero(self) -> None:
        """没有视线信息 → ``gaze=null``，**不是** ``0.0``。

        ``0.0`` 在这条链路上是"视线完全对正前方"（专注满分）。拿它当
        "估不出来"的哨兵值，等于对每一帧看不见眼睛的画面宣布专注满分。
        """
        payload = to_focus_payload(make_frame(ts=3.0, gaze=None))
        self.assertIsNone(payload["gaze"])
        self.assertEqual(payload["gaze_quality"], 0.0)
        self.assertEqual(assert_focus_contract(payload), [])

    def test_every_frame_produces_a_packet_even_without_gaze(self) -> None:
        """**不可用也要照发。** 跳过会让 B 的样本间隔凭空拉长。

        ``PassiveCalibrator`` 按 ``max_sample_gap_seconds`` 判定"这段观察
        时间还算不算数"，间隔超限就 ``reset()``。所以"这一帧没量到"必须
        是一条**报文**，而不是一条**空白**。
        """
        for frame in (
            make_frame(gaze=None),
            make_frame(has_face=False, gaze=0.3),
            make_frame(usable=False, gaze=0.3),
        ):
            with self.subTest(frame=repr(frame)[:60]):
                payload = to_focus_payload(frame)
                self.assertEqual(set(payload), set(FOCUS_FIELDS))
                self.assertIsNone(payload["gaze"])
                self.assertEqual(payload["gaze_quality"], 0.0)

    def test_gaze_never_contradicts_the_frame_message(self) -> None:
        """**单向**不变式：有视线 ⟹ 帧报文里有人脸。

        注意**不是等价**。反过来不成立，而且不成立才是对的：
        "看得见人"与"量得到视线"是两件事，CSV 回放里前者为真、后者恒假
        （V1 CSV 没有视线列）。

        单看代码不会觉得这里有什么问题，但**方向搞反就是一条真缺陷**：
        若视线报文在"帧报文说没人脸"的时候给出一个数，B 就会拿一条
        根本不存在的观测去做校准，而且两边各自的日志都干干净净。
        所以这条只钉一个方向：**帧说没人，视线必须是 null**。
        """
        cases = (
            (make_frame(gaze=0.4), True, True),
            (make_frame(gaze=None), True, False),
            (make_frame(has_face=False, gaze=0.4), False, False),
            (make_frame(usable=False, gaze=0.4), False, False),
            (make_frame(has_face=False, usable=False, gaze=None), False, False),
        )
        for frame, expect_face, expect_gaze in cases:
            with self.subTest(frame=repr(frame)[:60]):
                frame_payload = to_v1_sample(
                    frame, blink_total=0, emotion=Emotion.NORMAL
                )
                focus_payload = to_focus_payload(frame)
                self.assertEqual(frame_payload["has_face"], expect_face)
                self.assertEqual(focus_payload["gaze"] is not None, expect_gaze)
                if focus_payload["gaze"] is not None:
                    self.assertTrue(
                        frame_payload["has_face"],
                        "帧报文说没人脸，视线报文却给了一个数 —— "
                        "B 会拿这条不存在的观测去做校准",
                    )


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
