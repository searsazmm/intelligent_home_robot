# -*- coding: utf-8 -*-
"""窗口渲染的单元测试（需要 PyQt5；没有图形环境时自动跳过）。

跑法（在 frontend_C 目录下）：
    python -m pytest tests/ -v
    python tests/test_window.py

    # 想在真机平台下跑（断言更强，见下），显式指定平台：
    QT_QPA_PLATFORM=windows python tests/test_window.py

--------------------------------------------------------------------------
关于「离屏平台什么都画不出来」—— 这条必须先讲清楚
--------------------------------------------------------------------------
默认用 ``QT_QPA_PLATFORM=offscreen`` 跑，好处是无需图形环境。
但实测该平台下 **``QFontDatabase().families()`` 返回 0 个字体族**，
于是 Qt 一个字都画不出来 —— 截出来的图是全黑的（实测非黑像素计数为 0）。

这意味着：**在离屏平台下断言「脸上有白色像素」是错的**，它必然失败，
而且失败信息会让人以为窗口坏了（其实是平台没字体）。所以本文件分两档：

  * 「结构」档 —— 背景纯黑、尺寸、不崩溃、按键、关闭清理。**任何平台都跑。**
  * 「字形」档 —— 真切出白色的脸。**只在字体库非空的平台跑**，否则跳过并说明原因。

字形是否真的可用，最终以 ``main.py --dump-glyphs``（或 ``--screenshot``）
在**演示机上**肉眼确认为准，不靠这个测试。这里的两档只是防回归。
"""

from __future__ import annotations

import os
import sys
import unittest

#: 默认走离屏，让无图形环境的机器也能跑。已经设了就用用户的设置。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_core.expressions import STATE_NORMAL, STATE_SAD, VALID_STATES, all_faces

try:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QFontDatabase, QImage, QKeyEvent, QPainter
    from PyQt5.QtWidgets import QApplication

    from ui.window import FaceWindow, GlyphDumpWindow, build_font, pick_font_family
    QT_IMPORT_ERROR = None
except Exception as exc:                                    # noqa: BLE001
    QT_IMPORT_ERROR = exc

_app = None
_family = ""
_font_count = 0
_platform = ""

# ⚠️ 这段必须在**类定义之前**执行，不能放进 setUpModule()。
# ``@unittest.skipUnless(has_real_fonts(), ...)`` 是在类定义时就求值的，
# 而 setUpModule() 要到测试开始才跑 —— 于是所有字形断言会永远跳过，
# 看起来「测试通过」，实际上一次都没执行过。这个坑很隐蔽，所以写在这里。
if QT_IMPORT_ERROR is None:
    _app = QApplication.instance() or QApplication([])
    _font_count = len(QFontDatabase().families())
    _platform = _app.platformName()
    _family = pick_font_family()


def has_real_fonts() -> bool:
    """当前平台有没有可用的字体库 —— 决定能不能断言字形。"""
    return _font_count > 0


def skip_reason() -> str:
    return (
        f"平台 {_platform!r} 的字体库为空（{_font_count} 个字体族），"
        f"Qt 在此平台画不出任何字，跳过字形断言。"
        f"用 QT_QPA_PLATFORM=windows 重跑可启用。"
    )


def count_bright_pixels(pixmap, step: int = 3) -> int:
    """统计采样点里明显偏白的像素数。"""
    image = pixmap.toImage()
    bright = 0
    for y in range(0, image.height(), step):
        for x in range(0, image.width(), step):
            if image.pixelColor(x, y).red() > 40:
                bright += 1
    return bright


def make_window(**kwargs) -> FaceWindow:
    return FaceWindow(_family, **kwargs)


#: 一个几乎不可能被任何字体覆盖的码位（Unicode 第 16 平面末尾），
#: 用来渲染出"缺字"到底长什么样 —— 也就是豆腐块 □。
#: 它同时兼容两种情况：Qt 画替换框，或者干脆什么都不画（全黑）。
TOFU_SENTINEL = "\U0010FFFD"

#: 组合类字符（U+0300/U+0301 这类重音符号）。它们**要和前一个字符叠加**才有意义，
#: 单独渲染时各字体的处理千奇百怪（画个孤立的重音、画个虚线圆圈、或者什么都不画）。
#: 所以单独渲染一个字来判断"有没有字形"对它们不成立，故不参与比对。
#:
#: 目前表里已经**没有**这类字符了：唯一用到它们的 (｡•́︿•̀｡) 因为
#: Microsoft YaHei UI 不肯把附加符号和 • 合成一个字（渲染成"点 + 右上飘个小撇"）
#: 被换成了 (T＿T)，见 c_core/expressions.py 的注释。
#: 保留这个过滤器是为了以后再加回带附加符号的表情时不会误报 —— 别删。
COMBINING_MARKS = frozenset(range(0x0300, 0x0370))


def render_signature(char: str, point_size: int = 40) -> bytes:
    """把单个字符画到一张小图上，返回原始字节 —— 用于比较两次渲染是否一样。

    比较**画出来的像素**而不是问字体库"有没有这个字形"，是因为
    Qt 有字体回退：``QFontMetrics.inFontUcs4`` 说"没有"的字符往往其实画得出来
    （``build_font`` 用的 ``setFamilies`` 就是为这个）。而豆腐块恰恰是
    "回退链全都没找到"的最终结果 —— 只有渲染出来才知道。
    """
    image = QImage(80, 80, QImage.Format_ARGB32)
    image.fill(Qt.black)
    painter = QPainter(image)
    try:
        painter.setFont(build_font(_family, point_size))
        painter.setPen(Qt.white)
        painter.drawText(image.rect(), Qt.AlignCenter, char)
    finally:
        painter.end()
    return image.bits().asstring(image.byteCount())


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PyQt5 不可用：{QT_IMPORT_ERROR}")
class TestBlackBackground(unittest.TestCase):
    """用户明确要求的第一条：**窗口背景纯黑色**。这条必须钉死。"""

    def test_background_is_pure_black(self):
        window = make_window()
        self.addCleanup(window.close)
        pixmap = window.snapshot(STATE_NORMAL)
        image = pixmap.toImage()
        for x, y in ((2, 2), (image.width() - 3, 2), (2, image.height() - 3),
                     (image.width() - 3, image.height() - 3),
                     (image.width() // 2, 5)):
            color = image.pixelColor(x, y)
            self.assertEqual(
                (color.red(), color.green(), color.blue()), (0, 0, 0),
                msg=f"({x},{y}) 不是纯黑而是 {color.name()}",
            )

    def test_background_is_black_in_every_state(self):
        """换表情时不能顺带把底色改掉 —— 每个状态都要重新验证。"""
        window = make_window()
        self.addCleanup(window.close)
        for state in VALID_STATES:
            image = window.snapshot(state).toImage()
            for x, y in ((2, 2), (image.width() - 3, image.height() - 3)):
                color = image.pixelColor(x, y)
                self.assertEqual(
                    (color.red(), color.green(), color.blue()), (0, 0, 0),
                    msg=f"状态 {state} 的 ({x},{y}) 是 {color.name()}",
                )

    def test_background_is_black_while_blinking(self):
        window = make_window()
        self.addCleanup(window.close)
        image = window.snapshot(STATE_SAD, blinking=True).toImage()
        color = image.pixelColor(2, 2)
        self.assertEqual((color.red(), color.green(), color.blue()), (0, 0, 0))


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PyQt5 不可用：{QT_IMPORT_ERROR}")
class TestRendering(unittest.TestCase):
    """结构档：不依赖字体库，任何平台都要过。"""

    def test_snapshot_has_requested_size(self):
        window = make_window()
        self.addCleanup(window.close)
        pixmap = window.snapshot(STATE_NORMAL, width=640, height=480)
        self.assertEqual((pixmap.width(), pixmap.height()), (640, 480))

    def test_all_four_states_render_without_crashing(self):
        window = make_window()
        self.addCleanup(window.close)
        for state in VALID_STATES:
            for blinking in (False, True):
                pixmap = window.snapshot(state, blinking=blinking)
                self.assertFalse(pixmap.isNull(), msg=f"{state} blinking={blinking}")

    def test_unknown_state_does_not_crash(self):
        """api_doc §4.3：收到未知内容也不能崩。"""
        window = make_window()
        self.addCleanup(window.close)
        for junk in ("angry", "", None, 123, ["sad"]):
            window.set_state(junk)
            self.assertEqual(window._state, STATE_NORMAL, msg=repr(junk))

    def test_set_state_accepts_all_four(self):
        window = make_window()
        self.addCleanup(window.close)
        for state in VALID_STATES:
            window.set_state(state)
            self.assertEqual(window._state, state)

    def test_set_state_is_idempotent(self):
        """重复设同一状态不该重启动画（否则 15 秒心跳会让脸闪一下）。"""
        window = make_window()
        self.addCleanup(window.close)
        window.set_state(STATE_SAD)
        window.snapshot(STATE_SAD)              # 把淡入停掉
        window.set_state(STATE_SAD)
        self.assertFalse(window._fade.state(), "同一状态不该重启淡入动画")

    def test_fixed_font_size_is_respected(self):
        window = make_window(font_size=42)
        self.addCleanup(window.close)
        self.assertEqual(window._font_point_size(), 42)

    def test_auto_font_size_scales_with_height(self):
        """自适应字号必须真的随窗口高度变 —— 全屏时要看得清。"""
        window = make_window()
        self.addCleanup(window.close)
        window.resize(800, 400)
        small = window._font_point_size()
        window.resize(800, 1200)
        self.assertGreater(window._font_point_size(), small)

    def test_auto_font_size_has_a_floor(self):
        """窗口被拉得极小时字号不能变成 0（会画不出来）。"""
        window = make_window()
        self.addCleanup(window.close)
        window.resize(800, 10)
        self.assertGreaterEqual(window._font_point_size(), 8)

    @unittest.skipUnless(has_real_fonts(), skip_reason())
    def test_face_is_actually_drawn(self):
        """真切出白色的字 —— 这是「白色颜文字」这条需求的自动化版本。"""
        window = make_window()
        self.addCleanup(window.close)
        for state in VALID_STATES:
            pixmap = window.snapshot(state)
            self.assertGreater(
                count_bright_pixels(pixmap), 0,
                msg=f"状态 {state} 什么都没画出来",
            )

    @unittest.skipUnless(has_real_fonts(), skip_reason())
    def test_snapshot_bypasses_the_fade_animation(self):
        """``snapshot()`` 必须绕开淡入动画。

        状态切换会把不透明度置 0 再动画到 1，而动画需要事件循环推进；
        在 ``grab()`` 这种同步渲染里没有事件循环，直接抓会得到**全黑**一张图。
        这个坑很隐蔽 —— 导出的一批 PPT 插图全是黑的，却没人知道为什么。
        """
        window = make_window()
        self.addCleanup(window.close)

        window.set_state(STATE_SAD)          # 启动淡入，此刻不透明度是 0
        self.assertEqual(window.grab().toImage().pixelColor(
            window.width() // 2, window.height() // 2).red(), 0,
            "前提不成立：直接 grab() 本应是黑的",
        )

        self.assertGreater(
            count_bright_pixels(window.snapshot(STATE_SAD)), 0,
            "snapshot() 没有绕开淡入，导出的图是黑的",
        )

    @unittest.skipUnless(has_real_fonts(), skip_reason())
    def test_open_and_blink_frames_differ_on_screen(self):
        """眨眼帧在屏幕上也必须和睁眼帧不同（不只是字符串不同）。"""
        window = make_window()
        self.addCleanup(window.close)
        open_pixels = count_bright_pixels(window.snapshot(STATE_NORMAL, blinking=False))
        blink_pixels = count_bright_pixels(window.snapshot(STATE_NORMAL, blinking=True))
        self.assertNotEqual(open_pixels, blink_pixels)

    @unittest.skipUnless(has_real_fonts(), skip_reason())
    def test_different_states_look_different(self):
        """四种状态在屏幕上必须看起来不一样，否则用户看不出情绪变化。"""
        window = make_window()
        self.addCleanup(window.close)
        counts = {s: count_bright_pixels(window.snapshot(s)) for s in VALID_STATES}
        self.assertEqual(
            len(set(counts.values())), len(VALID_STATES),
            msg=f"有状态画出来一样：{counts}",
        )


@unittest.skipUnless(has_real_fonts(), skip_reason())
class TestNoTofuGlyphs(unittest.TestCase):
    """颜文字里**一个豆腐块都不能有**。

    为什么单开一组：窗口"画出白色像素"不等于"画对了字"。缺字时 Qt 画的是
    替换框 □，**那也是白色像素**，所以 ``test_face_is_actually_drawn``
    照样通过。而颜文字恰恰是字体覆盖的重灾区 —— 它大量使用半角假名（｡）和
    全角符号（￣ ＿ ︿ ﹏），本机 24 个非 ASCII 字符里有好几个是"常见字体
    其实没覆盖"的。一旦缺字，屏幕上就是一排方框，而缩略图上完全看不出来。

    做法：拿一个保证缺字的码位当"豆腐块样板"，逐字比对渲染结果。
    这比问字体库可靠（Qt 会做字体回退，问它常常答"没有"但其实画得出来）。
    """

    def setUp(self):
        self.tofu = render_signature(TOFU_SENTINEL)

    def test_sentinel_is_a_usable_reference(self):
        """先证明样板本身可用，免得下面那条是在跟一个"正常字"比。

        只要求**和正常字符画出来不一样**，不要求它一定是个方框：
        Qt 遇到缺字时可能画替换框（本机实测是 322 个亮像素的方框），
        也可能干脆什么都不画。两种都是有效的"坏"参照，硬要求其中一种
        会让这个测试在别的机器上无谓地失败 —— 而它要守的是"能区分好坏"。
        """
        self.assertNotEqual(
            self.tofu, render_signature("A"),
            "缺字样板和普通字符渲染结果一样，说明它区分不出坏字形",
        )
        # 渲染必须是确定的，否则下面的逐字比对毫无意义
        self.assertEqual(
            self.tofu, render_signature(TOFU_SENTINEL),
            "同一个码位两次渲染结果不同，比对逻辑不可靠",
        )

    def test_every_character_has_a_real_glyph(self):
        """正式断言：表里每个字符画出来都不能长成豆腐块。"""
        missing = []
        cache = {}

        for face in all_faces():
            for char in face:
                if char.isspace() or ord(char) in COMBINING_MARKS:
                    continue
                if char not in cache:
                    cache[char] = render_signature(char)
                if cache[char] == self.tofu:
                    missing.append(f"{char!r}（U+{ord(char):04X}，出现在 {face}）")

        self.assertEqual(
            missing, [],
            "颜文字里有字形缺失的字符，屏幕上会显示成方块 □：\n  "
            + "\n  ".join(missing)
            + "\n换一个覆盖更全的字体（见 ui/window.py 的 FONT_CANDIDATES），"
            "\n或者把这些字符从 c_core/expressions.py 的字表里换掉。",
        )

    def test_the_check_can_actually_fail(self):
        """反向验证：把一个真缺字的字符塞进去，检查逻辑必须能发现。

        **没有这条，上面那条可能因为渲染恒为空/恒相等而永远通过。**
        """
        self.assertEqual(
            render_signature(TOFU_SENTINEL), self.tofu,
            "同一个码位两次渲染结果不一致，说明渲染不确定，比对逻辑不可靠",
        )
        self.assertNotEqual(
            render_signature("A"), self.tofu,
            "普通字符被误判成豆腐块，比对逻辑太宽",
        )


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PyQt5 不可用：{QT_IMPORT_ERROR}")
class TestInteraction(unittest.TestCase):
    """按键与关闭 —— 「答辩时卡在全屏黑窗口里」是必须留退路的事故。"""

    def test_q_key_closes_the_window(self):
        window = make_window()
        window.show()
        window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Q, Qt.NoModifier))
        self.assertTrue(window._closing)

    def test_escape_closes_the_window_when_not_fullscreen(self):
        window = make_window()
        window.show()
        window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        self.assertTrue(window._closing)

    def test_other_keys_are_ignored(self):
        window = make_window()
        self.addCleanup(window.close)
        window.show()
        window.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
        self.assertFalse(window._closing)

    def test_close_stops_all_timers(self):
        """关窗后定时器还在跑 = 窗口析构后回调继续碰控件 → 崩溃。"""
        window = make_window()
        window.show()
        window.close()
        self.assertTrue(window._closing)
        for name in ("_blink_off", "_blink_gap", "_cycle"):
            self.assertFalse(getattr(window, name).isActive(), msg=name)

    def test_link_updates_do_not_crash(self):
        window = make_window()
        self.addCleanup(window.close)
        window.set_link("chat", True)
        window.set_link("status", False)
        window.note_message("我在呢")
        self.assertIn("对话通道", window.windowTitle())
        self.assertIn("我在呢", window._last_message)


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PyQt5 不可用：{QT_IMPORT_ERROR}")
class TestGlyphDumpWindow(unittest.TestCase):
    """``--dump-glyphs`` 自检窗口。"""

    def test_window_is_tall_enough_for_its_labels(self):
        """窗口高度必须按表情数量算出来。

        写死高度会让码位标签落进下一行的字形里 —— 而码位标签恰恰是
        字形缺失时唯一的定位线索。
        """
        window = GlyphDumpWindow(_family)
        self.addCleanup(window.close)
        from c_core.expressions import all_faces

        rows = (len(all_faces()) + GlyphDumpWindow.COLUMNS - 1) // GlyphDumpWindow.COLUMNS
        self.assertGreaterEqual(
            window.height(),
            GlyphDumpWindow.HEADER_HEIGHT + rows * GlyphDumpWindow.MIN_CELL_HEIGHT,
        )

    def test_renders_without_crashing(self):
        window = GlyphDumpWindow(_family)
        self.addCleanup(window.close)
        self.assertFalse(window.snapshot().isNull())

    def test_background_is_black(self):
        window = GlyphDumpWindow(_family)
        self.addCleanup(window.close)
        image = window.snapshot().toImage()
        color = image.pixelColor(image.width() - 3, image.height() - 3)
        self.assertEqual((color.red(), color.green(), color.blue()), (0, 0, 0))


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PyQt5 不可用：{QT_IMPORT_ERROR}")
class TestFontBuilding(unittest.TestCase):

    def test_build_font_sets_the_requested_size(self):
        font = build_font(_family, 40)
        self.assertEqual(font.pointSize(), 40)

    def test_build_font_has_a_floor(self):
        """字号 0 或负数会让 Qt 画不出字，必须被兜住。"""
        self.assertGreaterEqual(build_font(_family, 0).pointSize(), 8)
        self.assertGreaterEqual(build_font(_family, -5).pointSize(), 8)

    def test_build_font_uses_a_fallback_chain(self):
        """用 setFamilies 给出回退链：缺个别字形时 Qt 会继续往下找。"""
        font = build_font(_family, 20)
        families = font.families() if hasattr(font, "families") else [font.family()]
        self.assertGreaterEqual(len(families), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
