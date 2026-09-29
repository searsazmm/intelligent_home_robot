# -*- coding: utf-8 -*-
"""颜文字表与状态规整的单元测试（不依赖 Qt）。

跑法（在 frontend_C 目录下）：
    python -m pytest tests/ -v
    python tests/test_expressions.py     # 不装 pytest 也能跑
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_core.expressions import (
    DEFAULT_STATE, EXPRESSIONS, STATE_ABSENT, STATE_NORMAL, STATE_SAD, STATE_TIRED,
    VALID_STATES, all_faces, expressions_for, normalize_state, state_label,
)


class TestNormalizeState(unittest.TestCase):
    """``normalize_state`` 是 api_doc §4.3 的落点：任何输入都必须有返回，绝不抛异常。"""

    def test_accepts_the_four_frozen_states(self):
        for state in VALID_STATES:
            self.assertEqual(normalize_state(state), state)

    def test_strips_and_lowercases(self):
        """前端收到的字符串可能带空白或大小写不一致，都该认。"""
        self.assertEqual(normalize_state("  SAD  "), STATE_SAD)
        self.assertEqual(normalize_state("Tired"), STATE_TIRED)
        self.assertEqual(normalize_state("\tAbsent\n"), STATE_ABSENT)

    def test_unknown_state_falls_back_to_normal(self):
        """api_doc §4.3：收到未知内容时默认展示 normal。

        注意 ``"sad "`` 这类**不在**这里 —— 带空白的合法状态会被 strip 后认出来，
        那是刻意的宽容，不是未知输入。
        """
        for junk in ("angry", "happy", "", "   ", "sadd", "normal2", "情绪低落", "SAD!"):
            self.assertEqual(normalize_state(junk), DEFAULT_STATE, msg=repr(junk))

    def test_non_string_input_falls_back_to_normal(self):
        """报文里的 state 字段可能是 None / 数字 / 列表 / 字典，都不许崩。"""
        for junk in (None, 123, 4.5, True, [], {}, ["sad"], {"state": "sad"}, object()):
            self.assertEqual(normalize_state(junk), DEFAULT_STATE, msg=repr(junk))

    def test_never_raises(self):
        """极端输入也不能抛 —— 前端的稳定性优先于「报告错误」。"""
        class Nasty:
            def __str__(self):
                raise RuntimeError("我不配合")
            __repr__ = __str__

        for junk in (Nasty(), b"sad", float("nan")):
            try:
                result = normalize_state(junk)
            except Exception as exc:                     # noqa: BLE001
                self.fail(f"normalize_state({junk!r}) 抛了 {exc!r}")
            self.assertEqual(result, DEFAULT_STATE)


class TestExpressionTable(unittest.TestCase):

    def test_every_state_has_expressions(self):
        """四态必须都有表情，否则某个状态下界面是空的。"""
        for state in VALID_STATES:
            self.assertIn(state, EXPRESSIONS)
            self.assertGreaterEqual(len(EXPRESSIONS[state]), 1, msg=state)

    def test_table_has_no_extra_states(self):
        """api_doc §4.2 只允许四种状态，禁止自造。"""
        self.assertEqual(set(EXPRESSIONS), set(VALID_STATES))

    def test_faces_are_unique_within_a_state(self):
        """同状态内轮换，重复的表情会让轮换看起来卡住。"""
        for state in VALID_STATES:
            faces = [e.face for e in EXPRESSIONS[state]]
            self.assertEqual(len(faces), len(set(faces)), msg=state)

    def test_open_frames_are_unique_across_states(self):
        """16 个睁眼帧两两不同。

        这条钉住的是「四个状态看起来得不一样」：如果 sad 和 tired 各有一个
        相同的睁眼帧，表情切换就会看起来毫无反应 —— 而状态和日志都是对的，
        极难排查。

        **眨眼帧不参与这条断言**，因为它本来就该重复：``(＾▽＾)`` 和 ``(￣▽￣)``
        闭眼后都是 ``(－▽－)``。两个笑脸共用一张闭眼的脸是合理的，
        强行要求它们不同反而会造出「左眼闭右眼睁」这种不像眨眼的帧。
        """
        open_frames = [e.face for state in VALID_STATES for e in EXPRESSIONS[state]]
        self.assertEqual(len(open_frames), len(set(open_frames)))

    def test_states_do_not_share_blink_frames_with_each_other(self):
        """同一状态内共用眨眼帧可以，跨状态共用不行 —— 那就是两个状态在互相冒充。"""
        per_state = {s: {e.blink for e in EXPRESSIONS[s]} for s in VALID_STATES}
        for i, left in enumerate(VALID_STATES):
            for right in VALID_STATES[i + 1:]:
                self.assertFalse(
                    per_state[left] & per_state[right],
                    msg=f"{left} 与 {right} 的眨眼帧有重叠：{per_state[left] & per_state[right]}",
                )

    def test_blink_frame_differs_from_open_frame(self):
        """眨眼帧必须真的不一样，否则眨眼动画看不出来。"""
        for state in VALID_STATES:
            for expression in EXPRESSIONS[state]:
                self.assertNotEqual(
                    expression.face, expression.blink,
                    msg=f"{state} 的 {expression.face!r} 眨眼帧和睁眼帧相同",
                )

    def test_all_faces_counts_both_frames(self):
        expected = sum(len(v) for v in EXPRESSIONS.values()) * 2
        self.assertEqual(len(all_faces()), expected)

    def test_all_faces_can_be_limited_to_one_state(self):
        for state in VALID_STATES:
            self.assertEqual(len(all_faces(state)), len(EXPRESSIONS[state]) * 2)

    def test_expressions_for_unknown_state_returns_normal_group(self):
        """未知状态取表情也不能抛（api_doc §4.3）。"""
        self.assertEqual(expressions_for("nonsense"), expressions_for(STATE_NORMAL))
        self.assertEqual(expressions_for(None), expressions_for(STATE_NORMAL))

    def test_expression_str_is_the_open_frame(self):
        """``__str__`` 返回睁眼帧，日志里打印才不会莫名其妙。"""
        for state in VALID_STATES:
            for expression in EXPRESSIONS[state]:
                self.assertEqual(str(expression), expression.face)

    def test_expression_is_immutable(self):
        """冻结的 dataclass：运行期改表情是 bug，应该直接报错。"""
        expression = EXPRESSIONS[STATE_NORMAL][0]
        with self.assertRaises(Exception):
            expression.face = "X"


class TestStateLabel(unittest.TestCase):

    def test_every_state_has_a_label(self):
        for state in VALID_STATES:
            label = state_label(state)
            self.assertTrue(label and isinstance(label, str), msg=state)

    def test_labels_are_distinct(self):
        """四态的中文说明必须互不相同，否则日志分不清是哪个状态。"""
        labels = [state_label(s) for s in VALID_STATES]
        self.assertEqual(len(labels), len(set(labels)))

    def test_unknown_state_gets_normal_label(self):
        self.assertEqual(state_label("bogus"), state_label(STATE_NORMAL))


class TestContentSanity(unittest.TestCase):
    """对文案本身的基本体检 —— 这些是「看起来对但其实坏了」的地方。"""

    def test_no_leading_or_trailing_whitespace(self):
        """首尾空白会让居中计算偏一点，而且看不出来。"""
        for face in all_faces():
            self.assertEqual(face, face.strip(), msg=repr(face))

    def test_no_newlines_or_tabs(self):
        """换行会把一行的脸撕成两行。"""
        for face in all_faces():
            self.assertNotIn("\n", face)
            self.assertNotIn("\t", face)

    def test_every_face_is_wrapped_in_parens(self):
        """颜文字都以括号包住，这是它看起来「是张脸」的最低要求。"""
        for face in all_faces():
            self.assertTrue(face.startswith("("), msg=repr(face))
            self.assertTrue(face.endswith((")", "?", "…", "Z", "z")), msg=repr(face))

    def test_faces_are_not_too_long_to_fit(self):
        """超长的脸在大字号下会顶出窗口。16 个字符是当前实测的安全上限。"""
        for face in all_faces():
            self.assertLessEqual(len(face), 16, msg=repr(face))


if __name__ == "__main__":
    unittest.main(verbosity=2)
