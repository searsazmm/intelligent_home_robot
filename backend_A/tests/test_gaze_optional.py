"""视线通道的「估不出来」语义：``gaze_off_ratio`` 必须能表达 ``None``。

标准库 unittest，不依赖 mediapipe、不依赖摄像头、不依赖模型文件。

这个文件补的是一个**真实的覆盖空洞**。在此之前 ``backend_A/tests/`` 里
**没有任何一帧碰到过 gaze**（``grep gaze_off_ratio backend_A/tests/`` 是空的），
于是下面两件事都不会让任何测试变红：

1. ``_gaze_off`` 在"估不出来"时返回 ``0.0`` —— 而 ``0.0`` 在这条链路上的
   意思是**视线完全对正前方**，也就是专注满分。"看不见眼睛"被读成
   "非常专注"，两个方向的后果都很坏，且没有任何提示。
2. ``AttentionTracker.snapshot`` 的分母把"有脸但没有视线信息"的帧也算进去
   —— 代码不崩、日志干净、结论被系统性推向 FOCUSED。

第 2 条尤其危险：它**不会**让任何东西报错，只会让 A 对"失神"的初判整体
偏松。所以这里最要紧的一条不是"``None`` 传下去了没有"，而是
:meth:`TestAttentionWindowDenominator.test_gazeless_frames_are_not_in_the_denominator`
—— 那条的判据会**翻转**：同样的窗口，改之前算 FOCUSED，改之后算 ABSENT。

运行::

    cd backend_A && python -m unittest discover -s tests -v
    cd backend_A && python tests/test_gaze_optional.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _console  # noqa: E402,F401  （导入即生效：让中文输出不乱码）

from module_a_vision.capture.csv_replay import CsvReplaySource  # noqa: E402
from module_a_vision.face.mediapipe_backend import (  # noqa: E402
    EYE_IDX_LEFT,
    EYE_IDX_RIGHT,
    IRIS_IDX_LEFT,
    IRIS_IDX_RIGHT,
    MediaPipeFaceBackend,
)
from module_a_vision.metrics.attention import (  # noqa: E402
    ABSENT_FRAMES_RATIO,
    AttentionTracker,
)
from shared.enums import Attention  # noqa: E402
from shared.frame_features import FrameFeatures, FrameQuality  # noqa: E402

#: 478 点模型（``face_landmarker.task``）的点数。虹膜点是 468–477。
N_LANDMARKS_478 = 478

#: 468 点模型：**没有虹膜**，最后 10 个点不存在。
N_LANDMARKS_468 = 468

#: 偏离阈值的两倍 —— 造"肯定算偏离"的帧时用它，不要贴着阈值写，
#: 否则改动阈值会让测试变得难懂。
OFF_ABOVE_THRESHOLD = 0.9


class _Pt:
    """最小地标：``_gaze_off`` 只读 ``.x``。"""

    __slots__ = ("x",)

    def __init__(self, x: float) -> None:
        self.x = x


def _flat_landmarks(count: int, x: float = 0.5) -> list:
    return [_Pt(x) for _ in range(count)]


def _gaze_off(landmarks) -> float | None:
    """调后端里的那个纯函数。

    用 ``__new__`` 绕过 ``__init__``：构造函数要加载模型文件，而
    ``_gaze_off`` 是纯粹的"地标进、比值出"，跟模型无关。
    """
    return MediaPipeFaceBackend.__new__(MediaPipeFaceBackend)._gaze_off(landmarks)


def _frame(ts: float, gaze: float | None) -> FrameFeatures:
    """一帧有脸、质量合格的画面，视线取给定值（可为 None）。"""
    return FrameFeatures(
        ts=ts,
        has_face=True,
        ear_left=0.3,
        ear_right=0.3,
        gaze_off_ratio=gaze,
        quality=FrameQuality(valid=True),
    )


class TestGazeOffIsOptional(unittest.TestCase):
    """``_gaze_off`` 的三种出口：一个数、或 ``None``。"""

    def test_missing_iris_landmarks_mean_none(self) -> None:
        """468 点模型上**必须**给 ``None``，不能给 ``0.0``。

        ``0.0`` 是"视线完全对正前方"（专注满分）。拿它当"估不出来"的
        哨兵值，等于对每一条根本看不见眼睛的帧宣布"非常专注"。
        """
        self.assertIsNone(
            _gaze_off(_flat_landmarks(N_LANDMARKS_468)),
            "没有虹膜点时返回的不是 None —— " "'看不见眼睛'会被读成'专注满分'",
        )

    def test_degenerate_eye_width_means_none(self) -> None:
        """两眼都量不出眼宽（全 0）时也是 ``None``。

        这是第二条早先写死成 ``0.0`` 的出口。它与上一条同样重要：
        走到这里的是一帧**检测退化**的画面，不是一帧"正对着镜头"的画面。
        """
        self.assertIsNone(_gaze_off(_flat_landmarks(N_LANDMARKS_478, x=0.0)))

    def test_a_real_estimate_is_a_number(self) -> None:
        """能估出来时必须**真的估出来** —— 否则上面两条可以靠"永远返回 None"骗过。"""
        landmarks = _flat_landmarks(N_LANDMARKS_478)
        # 右眼：外角 0.30、内角 0.50，虹膜中心摆在 0.44（明显偏外）。
        landmarks[EYE_IDX_RIGHT[0]] = _Pt(0.30)
        landmarks[EYE_IDX_RIGHT[3]] = _Pt(0.50)
        for i in IRIS_IDX_RIGHT:
            landmarks[i] = _Pt(0.44)
        # 左眼：外角 0.70、内角 0.50，虹膜正中（0.60）→ 这一只为 0。
        landmarks[EYE_IDX_LEFT[0]] = _Pt(0.70)
        landmarks[EYE_IDX_LEFT[3]] = _Pt(0.50)
        for i in IRIS_IDX_LEFT:
            landmarks[i] = _Pt(0.60)

        off = _gaze_off(landmarks)
        self.assertIsNotNone(off, "地标齐全却给不出估计")
        self.assertGreater(off, 0.0, "虹膜明显偏外，偏离度却是 0")
        self.assertLessEqual(off, 1.0, "偏离度必须归一化在 [0,1]")


class TestGazeNoneIsNotPerfectFocus(unittest.TestCase):
    """``None`` 与 ``0.0`` 在结构上必须分得开。"""

    def test_default_is_none_not_zero(self) -> None:
        self.assertIsNone(FrameFeatures().gaze_off_ratio)
        self.assertNotEqual(FrameFeatures().gaze_off_ratio, 0.0)

    def test_repr_survives_a_none_gaze(self) -> None:
        """``repr`` 不能在排错的时候崩。

        ``f"{None:.2f}"`` 会抛 ``TypeError``，而 ``__repr__`` 正是出问题时
        第一个被调用的东西 —— 在那里崩掉比打不出这个字段糟得多。
        """
        text = repr(_frame(1.0, None))
        self.assertIn("gaze_off=None", text)

    def test_repr_still_formats_a_number(self) -> None:
        self.assertIn("gaze_off=0.90", repr(_frame(1.0, OFF_ABOVE_THRESHOLD)))


class TestAttentionWindowDenominator(unittest.TestCase):
    """**这个文件里最要紧的一组。** 分母里的"没有视线信息"必须被剔除。"""

    def _ratio(self, frames: list) -> tuple:
        tracker = AttentionTracker()
        for f in frames:
            tracker.update(f)
        metrics = tracker.snapshot(frames, now_ts=len(frames) and frames[-1].ts)
        return metrics.frames_ratio, metrics.label, metrics.gaze_frames

    def test_gazeless_frames_are_not_in_the_denominator(self) -> None:
        """一半帧偏离、一半帧**没有视线信息** → 偏离占比是 ``1.0``，不是 ``0.5``。

        ``0.5`` 低于 ``ABSENT_FRAMES_RATIO``（0.60），所以按老算法这个窗口
        **永远**进不了失神候选 —— 而"进不了"这件事不会报错、不会告警。
        那些没有视线信息的帧既进不了分子也（从改动起）进不了分母，
        它们对"偏离占比"这个问题**没有发言权**。
        """
        frames = [_frame(float(i), OFF_ABOVE_THRESHOLD) for i in range(5)]
        frames += [_frame(float(i + 5), None) for i in range(5)]

        ratio, label, gaze_frames = self._ratio(frames)

        # 先断 ratio：它是本条的**语义**主张，失败信息应当直接指向"分母错了"。
        self.assertEqual(ratio, 1.0, f"分母里混进了没有视线信息的帧：ratio={ratio}")
        self.assertGreaterEqual(ratio, ABSENT_FRAMES_RATIO)
        self.assertEqual(gaze_frames, 5, "带视线信息的帧数报错了")

        # 把"老算法下会怎样"写下来：它不是"略低"，而是**跨过了判据**。
        diluted = 5 / 10
        self.assertLess(diluted, ABSENT_FRAMES_RATIO, "本条测试的前提已失效")

    def test_a_window_with_no_gaze_at_all_is_visibly_empty(self) -> None:
        """整窗都没有视线信息：算出来的仍是 ``FOCUSED``，**但看得出来不是测量**。

        这里不断言"应该判成别的" —— :class:`~shared.enums.Attention` 只有
        FOCUSED / ABSENT 两态，没有"无数据"，硬加一个态的爆炸半径远大于
        这一步的范围。所以做的是让这件事**可见**：
        ``gaze_frames == 0``。看到 ``FOCUSED`` 配 ``gaze_frames == 0``，
        应当读成"这一窗没有测量"，而不是"老人很专注"。
        """
        frames = [_frame(float(i), None) for i in range(10)]
        ratio, label, gaze_frames = self._ratio(frames)

        self.assertEqual(label, Attention.FOCUSED)
        self.assertEqual(gaze_frames, 0, "整窗没视线信息时 gaze_frames 必须是 0")
        self.assertEqual(ratio, 0.0)


class TestCsvReplayGazeColumn(unittest.TestCase):
    """回放里"没有这一列"与"这一列写着 0"必须是两件事。"""

    HEADER_BASE = "timestamp,has_face,ear,blink_cnt,pitch,yaw,roll,emo_feature"

    def _replay(self, *, columns: str, row: str) -> FrameFeatures:
        header = self.HEADER_BASE if not columns else f"{self.HEADER_BASE},{columns}"
        body = "1.0,true,0.30,0,2.0,-3.0,1.5,normal" if not row else f"1.0,true,0.30,0,2.0,-3.0,1.5,normal,{row}"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vis.csv")
            with open(path, "w", encoding="utf-8", newline="") as fp:
                fp.write(f"{header}\n{body}\n")
            src = CsvReplaySource(path)
            src.open()
            try:
                frame = src.read_features()
            finally:
                src.close()

        self.assertIsNotNone(frame, "回放没有产出帧")
        return frame

    def test_missing_column_stays_none(self) -> None:
        """V1 CSV 没有视线列 → ``None``（"估不出来"），**不是 ``0.0``**。

        填 ``0.0`` 是在伪造一个"老人一直很专注"的结论，而且全程无提示。
        """
        frame = self._replay(columns="", row="")
        self.assertIsNone(frame.gaze_off_ratio)

    def test_zero_in_the_column_stays_zero(self) -> None:
        """列存在且写着 ``0`` → ``0.0``。这一条与上一条配对，

        缺了它，把 ``None`` 一路写成"永远 None"也能让上一条通过 ——
        而那就等于把"确实正对着镜头"也丢掉了。
        """
        frame = self._replay(columns="gaze_off_ratio", row="0")
        self.assertIsNotNone(frame.gaze_off_ratio)
        self.assertEqual(frame.gaze_off_ratio, 0.0)

    def test_present_value_survives(self) -> None:
        frame = self._replay(columns="gaze_off_ratio", row="0.9")
        self.assertAlmostEqual(frame.gaze_off_ratio, 0.9, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
