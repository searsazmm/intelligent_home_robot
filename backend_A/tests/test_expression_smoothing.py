# -*- coding: utf-8 -*-
"""表情输出滑动窗口平滑器（:class:`ExpressionVoteFilter`）的单元测试。

需求口径：
  1. 缓存最近 5~8 帧的单帧表情结果；
  2. 多数投票选出主导类，**连续多帧一致才更新对外输出**；
  3. 过滤画面抖动 / 人脸微晃带来的瞬时误判；
  4. 切换延迟要小（实时性）。

滤波器只吃标签、不依赖帧数据，所以这里直接喂 :class:`Emotion`，
不需要构造 :class:`FrameFeatures`。与聚合器的接线
（``WindowAggregator.classify_frame``）由 test_interop_with_b.py 的
端到端用例覆盖。
"""

import os
import sys
import unittest

# 让 `from shared...` / `from module_a_vision...` 在任何工作目录下都能导入
# （pytest 从仓库根收集时，backend_A 不一定在 sys.path 里）。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.enums import Emotion
from module_a_vision.metrics.expression import ExpressionVoteFilter


def feed_all(f: ExpressionVoteFilter, labels) -> list[Emotion]:
    """依次喂入标签序列，返回每帧的对外输出（便于断言切换时机）。"""
    return [f.feed(label) for label in labels]


class TestWindowDefaults(unittest.TestCase):
    """默认参数（窗口 7 / 连续 3）落在需求的 5~8 帧口径内。"""

    def test_defaults_are_within_required_range(self):
        from module_a_vision.aggregate.window import AggregatorConfig

        cfg = AggregatorConfig()
        self.assertTrue(5 <= cfg.emo_smooth_window <= 8,
                        "需求要求窗口 5~8 帧")
        self.assertEqual(cfg.emo_smooth_window, 7)
        self.assertLess(cfg.emo_smooth_switch, cfg.emo_smooth_window)

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            ExpressionVoteFilter(window_size=0)

    def test_switch_need_clamped_to_window(self):
        """连续帧数钳位不超过窗口本身，否则永远切不过去。"""
        f = ExpressionVoteFilter(window_size=5, switch_consecutive=99)
        feed_all(f, [Emotion.SAD] * 5)          # 恰好占满窗口
        self.assertEqual(f.output, Emotion.SAD,
                         "钳位后仍应能在窗口全同类时切换")


class TestGlitchSuppression(unittest.TestCase):
    """约束 3：单帧毛刺不翻转输出。"""

    def test_single_frame_glitch_does_not_switch(self):
        f = ExpressionVoteFilter()
        feed_all(f, [Emotion.NORMAL] * 7)       # 填满窗口的稳态
        out = f.feed(Emotion.SAD)               # 一帧瞬时误判
        self.assertEqual(out, Emotion.NORMAL)

    def test_alternating_noise_stays_pinned(self):
        """normal/sad 逐帧交替（最坏噪声）——输出钉在初始标签上。"""
        f = ExpressionVoteFilter()
        noisy = [Emotion.NORMAL, Emotion.SAD] * 10
        outputs = feed_all(f, noisy)
        self.assertTrue(all(o == Emotion.NORMAL for o in outputs))

    def test_glitch_after_switch_does_not_flip_back(self):
        """切到 sad 后的一两帧 normal 毛刺，同样打不回去。"""
        f = ExpressionVoteFilter()
        feed_all(f, [Emotion.NORMAL] * 7)
        feed_all(f, [Emotion.SAD] * 4)          # 合法切换（连续+多数都满足）
        out = feed_all(f, [Emotion.NORMAL, Emotion.NORMAL])
        self.assertEqual(out, [Emotion.SAD, Emotion.SAD])


class TestSwitching(unittest.TestCase):
    """约束 2 + 4：连续多帧才切换，且延迟有界。"""

    def test_switches_after_consecutive_frames(self):
        f = ExpressionVoteFilter()
        feed_all(f, [Emotion.NORMAL] * 7)
        outputs = feed_all(f, [Emotion.SAD] * 4)
        # 第 4 帧才切换：连续性（3 帧）与多数投票（窗口内 4>3）**同时**
        # 满足的那一刻。25 fps 下约 160 毫秒，仍是"一眨眼"量级。
        self.assertEqual(outputs, [
            Emotion.NORMAL,
            Emotion.NORMAL,
            Emotion.NORMAL,
            Emotion.SAD,
        ])

    def test_majority_vote_takes_over_window(self):
        """窗口 7 帧：4 帧 sad（多数）+ 3 帧连续尾巴 → 切换。

        多数投票与连续性闸门是**与**的关系——本例两者同时满足。
        """
        f = ExpressionVoteFilter()
        feed_all(f, [Emotion.NORMAL] * 7)
        outputs = feed_all(f, [Emotion.SAD] * 4 + [Emotion.NORMAL])
        self.assertEqual(outputs[-1], Emotion.SAD,
                         "4 票 sad 在窗口内占多数，切换后保持")

    def test_two_frames_not_enough(self):
        """连续 2 帧不足 \`\`switch_consecutive``=3 —— 不切换。"""
        f = ExpressionVoteFilter()
        feed_all(f, [Emotion.NORMAL] * 7)
        outputs = feed_all(f, [Emotion.SAD, Emotion.SAD])
        self.assertEqual(outputs, [Emotion.NORMAL, Emotion.NORMAL])


class TestAbsentFrames(unittest.TestCase):
    """无脸帧投 normal 票：人离开画面后输出自然回落。"""

    def test_output_falls_back_to_normal_when_face_lost(self):
        f = ExpressionVoteFilter()
        feed_all(f, [Emotion.NORMAL] * 7)
        feed_all(f, [Emotion.SAD] * 4)          # 已切到 sad
        outputs = feed_all(f, [Emotion.NORMAL] * 4)
        self.assertEqual(outputs[-1], Emotion.NORMAL,
                         "连续 4 帧 normal（如人离开）应平滑回落")


if __name__ == "__main__":
    unittest.main(verbosity=2)
