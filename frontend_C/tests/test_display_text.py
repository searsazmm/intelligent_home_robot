# -*- coding: utf-8 -*-
"""右栏文案的单元测试（纯函数，**不需要 PyQt5**）。

跑法（在 frontend_C 目录下）：
    python -m pytest tests/test_display_text.py -v

--------------------------------------------------------------------------
这个文件真正在守的三件事
--------------------------------------------------------------------------
1. **``index`` 是 ``None`` 不等于 0。** B 用 ``None`` 表示"没有分数"
   （校准没锁定 / 有效观察不够 / 门控拦了）。界面上把它显示成 ``0.0``
   等于对老人说"你走神了"，而事实是"这一项测不出来"。
2. **``note`` 一个字都不许改写。** 那是这套指数的安全声明，
   由 B 侧 ``ui_channel.VAI_NOTE`` 定稿。下面有一条**跨模块**的测试
   直接去读 B 的源文件比对字面量 —— 改文案时它会红。
3. **三态要能分辨**：没给 ``--stream-url`` / 给了但没连上 / 已连接。
   都写成"未接入"的话，演示时无法判断该去查 A 还是查参数。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_core import display_text as dt
from c_core.expressions import STATE_SAD, STATE_TIRED

#: 仓库根（frontend_C/tests/xxx.py → 上两级）
REPO_ROOT = Path(__file__).resolve().parents[2]

#: 一份"真的算出来了"的报文
VAI_OK = {
    "type": "vai",
    "index": 63.5,
    "index_status": "研究趋势（非认知专注）",
    "status": "有效",
    "modalities": ["gaze", "pose", "eye_open"],
    "modality_config_id": "凝视+头姿+睁眼",
    "valid_seconds": 42.0,
    "note": "研究趋势（非认知专注）；日常参考，非医疗结论",
}


class TestIndexIsNotZero(unittest.TestCase):
    """ ``None`` = 没有分数，不是 0 分。"""

    def test_none_is_not_shown_as_zero(self):
        self.assertEqual(dt.format_index(None), dt.NO_INDEX)
        self.assertNotEqual(dt.format_index(None), "0.0")
        self.assertNotIn("0", dt.format_index(None))

    def test_one_decimal_place(self):
        self.assertEqual(dt.format_index(63.4567), "63.5")
        self.assertEqual(dt.format_index(100), "100.0")
        self.assertEqual(dt.format_index(0), "0.0")

    def test_garbage_does_not_crash(self):
        """字段类型乱来时按"没有分数"显示，而不是抛出去把 GUI 线程带走。"""
        self.assertEqual(dt.format_index("高"), dt.NO_INDEX)
        self.assertEqual(dt.format_index([1, 2]), dt.NO_INDEX)

    def test_headline_says_so(self):
        self.assertEqual(dt.format_vai_headline(None), f"专注度 VAI {dt.NO_INDEX}")
        self.assertEqual(dt.format_vai_headline({"index": None}),
                         f"专注度 VAI {dt.NO_INDEX}")
        self.assertEqual(dt.format_vai_headline(VAI_OK), "专注度 VAI 63.5")

    def test_detail_explains_why_instead_of_showing_a_status(self):
        """没有分数时，细节行要显示 B 给的 ``index_status``（为什么没算出来），
        而不是拿 ``status``（"观察不足"）当结论 —— 后者容易被读成一种评价。"""
        detail = dt.format_vai_detail(
            {"index": None, "status": "观察不足", "index_status": "有效观察时长不足"})
        self.assertIn("有效观察时长不足", detail)
        self.assertNotIn("观察不足", detail)

    def test_detail_falls_back_when_status_missing(self):
        detail = dt.format_vai_detail({"index": None})
        self.assertIn("原因未知", detail)


class TestIndexBar(unittest.TestCase):

    def test_none_is_an_empty_bar_not_a_full_one(self):
        bar = dt.format_index_bar(None)
        self.assertEqual(bar, dt.BAR_EMPTY * dt.BAR_CELLS)

    def test_endpoints(self):
        self.assertEqual(dt.format_index_bar(0), dt.BAR_EMPTY * dt.BAR_CELLS)
        self.assertEqual(dt.format_index_bar(100), dt.BAR_FILLED * dt.BAR_CELLS)

    def test_middle(self):
        bar = dt.format_index_bar(63.5)
        self.assertEqual(len(bar), dt.BAR_CELLS)
        self.assertEqual(bar.count(dt.BAR_FILLED), 6)

    def test_out_of_range_is_clamped(self):
        """越界值不许画出一条比格子还长的条（会把版面撑破）。"""
        self.assertEqual(len(dt.format_index_bar(-20)), dt.BAR_CELLS)
        self.assertEqual(len(dt.format_index_bar(999)), dt.BAR_CELLS)
        self.assertEqual(dt.format_index_bar(999), dt.BAR_FILLED * dt.BAR_CELLS)


class TestModalities(unittest.TestCase):

    def test_keys_are_translated(self):
        self.assertEqual(dt.format_modalities(["gaze", "pose", "eye_open"]),
                         "凝视+头姿+睁眼")

    def test_unknown_key_is_kept_verbatim(self):
        """认不出的模态宁可原样显示 —— 吞掉它等于隐瞒"这分数换了一组权重"。"""
        self.assertEqual(dt.format_modalities(["gaze", "brand_new"]),
                         "凝视+brand_new")

    def test_empty(self):
        self.assertEqual(dt.format_modalities([]), "")
        self.assertEqual(dt.format_modalities(None), "")

    def test_a_bare_string_is_one_key_not_a_sequence_of_characters(self):
        """防呆：``"gaze"`` 必须翻成"凝视"，而不是逐字符变成 ``g+a+z+e``。"""
        self.assertEqual(dt.format_modalities("gaze"), "凝视")
        self.assertEqual(dt.format_modalities(""), "")

    def test_detail_prefers_b_config_id(self):
        """``modality_config_id`` 是 B 自己的总结，有就用它（少一处会漂移的重复）。"""
        self.assertIn("凝视+头姿+睁眼", dt.format_vai_detail(VAI_OK))

    def test_detail_falls_back_to_labels(self):
        msg = {k: v for k, v in VAI_OK.items() if k != "modality_config_id"}
        self.assertIn("凝视+头姿+睁眼", dt.format_vai_detail(msg))


class TestNoteIsNeverReworded(unittest.TestCase):
    """安全声明：C 只许原样显示，连兜底值都得逐字照抄 B 的那句。"""

    def test_uses_the_note_from_b(self):
        self.assertEqual(dt.format_note(VAI_OK), VAI_OK["note"])

    def test_fallback_matches_b_source_verbatim(self):
        """**跨模块断言**：兜底字符串必须与 ``backend_B/core/ui_channel.py``
        里的 ``VAI_NOTE`` 逐字相同。

        为什么要读源文件而不是 import：C 与 B 是两个独立模块（不同的运行进程、
        不同的 sys.path），C 不能 import B —— 而"两边各写一遍同样的中文"正是
        会漂移的地方。改 B 的那句话而忘了 C，这条测试会红。
        """
        path = REPO_ROOT / "backend_B" / "core" / "ui_channel.py"
        if not path.exists():
            self.skipTest(f"找不到 B 侧源文件：{path}")

        source = path.read_text(encoding="utf-8")
        expected = f'VAI_NOTE = "{dt.FALLBACK_NOTE}"'
        self.assertIn(expected, source,
                      "C 的兜底免责声明与 B 的 VAI_NOTE 不一致 —— 两边必须逐字相同")

    def test_fallback_used_when_note_missing(self):
        """旧版 B / 手写报文没有 note 字段时用兜底，而不是留空。"""
        self.assertEqual(dt.format_note({"index": 63.5}), dt.FALLBACK_NOTE)
        self.assertEqual(dt.format_note(None), dt.FALLBACK_NOTE)

    def test_note_block_always_has_both_halves(self):
        for note in (dt.format_note(VAI_OK), dt.format_note(None)):
            with self.subTest(note=note):
                self.assertIn("非认知专注", note)
                self.assertIn("非医疗结论", note)


class TestVaiLines(unittest.TestCase):

    def test_three_lines_always(self):
        for vai in (None, {}, VAI_OK, {"index": None, "index_status": "不可用"}):
            with self.subTest(vai=vai):
                self.assertEqual(len(dt.format_vai_lines(vai)), 3)

    def test_never_prints_zero_for_a_missing_index(self):
        """整块文本里都不许出现 ``0.0`` —— 这是本文件第一条约束的兜底版本。"""
        for vai in (None, {}, {"index": None, "index_status": "不可用"}):
            with self.subTest(vai=vai):
                block = "\n".join(dt.format_vai_lines(vai))
                self.assertNotIn("0.0", block)
                self.assertIn(dt.NO_INDEX, block)


class TestStateAndLinks(unittest.TestCase):

    def test_state_line_carries_the_reason(self):
        self.assertEqual(dt.format_state_line(STATE_TIRED), "状态 疲惫")
        self.assertEqual(dt.format_state_line(STATE_SAD, "连续打哈欠"),
                         "状态 情绪低落（连续打哈欠）")

    def test_unknown_state_still_prints_something(self):
        self.assertEqual(dt.format_state_line("angry"), "状态 状态正常")

    def test_link_block_order_is_fixed(self):
        """顺序固定，别让它在界面上跳来跳去。"""
        line = dt.format_link_block({"stream": False, "chat": True})
        self.assertTrue(line.startswith("对话通道 通"))
        self.assertIn("画面通道 断", line)

    def test_unknown_channel_is_still_shown(self):
        self.assertIn("future", dt.format_link_block({"future": True}))


class TestStreamText(unittest.TestCase):
    """画面占位区的三态：没配 / 配了没连上 / 已连接。"""

    def test_not_configured(self):
        text = dt.format_stream_text(None)
        self.assertIn(dt.STREAM_OFFLINE, text)
        self.assertIn("--stream-url", text)

    def test_configured_but_disconnected_names_the_url(self):
        text = dt.format_stream_text(False, "http://127.0.0.1:8010/stream.mjpeg")
        self.assertIn(dt.STREAM_OFFLINE, text)
        self.assertIn("http://127.0.0.1:8010/stream.mjpeg", text)
        self.assertIn("重连", text)

    def test_connected_says_waiting_for_the_first_frame(self):
        self.assertEqual(dt.format_stream_text(True), dt.STREAM_WAITING)

    def test_the_three_states_are_distinguishable(self):
        texts = {dt.format_stream_text(None),
                 dt.format_stream_text(False, "u"),
                 dt.format_stream_text(True)}
        self.assertEqual(len(texts), 3, "三种状态必须给出三种不同的文字")


if __name__ == "__main__":
    unittest.main(verbosity=2)
