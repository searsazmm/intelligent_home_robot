# -*- coding: utf-8 -*-
"""左右分栏窗口与右栏控件的单元测试（需要 PyQt5；没有图形环境时自动跳过）。

跑法（在 frontend_C 目录下）：
    python -m pytest tests/ -v
    python tests/test_split_window.py

    # 想在真机平台下跑（断言更强），显式指定平台：
    QT_QPA_PLATFORM=windows python tests/test_split_window.py

与 ``tests/test_window.py`` 同一套两档约定：**结构**类断言在任何平台都跑
（控件存在、转发到位、字号被钳制、绘制不崩），**字形**类断言只在字体库
非空的平台跑（离屏平台下 ``QFontDatabase().families()`` 是 0 个，
Qt 一个字都画不出来，那时候断言"有白色像素"必然失败且误导）。

--------------------------------------------------------------------------
这个文件真正在守的两件事
--------------------------------------------------------------------------
1. **分栏之后左栏的字号必须被宽度钳住**（``ui/window.py`` 的
   ``_largest_fitting_point_size``）。这是分栏带来的唯一排版风险：最长的那几个
   颜文字会在半宽的栏里横向溢出，而且只在个别表情上出现，一闪而过。
2. **右栏的专注度不许反过来影响左栏那张脸**（C 侧的"第二状态源"红线，
   对应 B 侧 ``vai`` 报文刻意不带 ``state`` 字段）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

#: 默认走离屏，让无图形环境的机器也能跑。已经设了就用用户的设置。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_FRONTEND_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FRONTEND_C)

from c_core.expressions import (
    STATE_NORMAL,
    STATE_SAD,
    STATE_TIRED,
    VALID_STATES,
    all_faces,
    expressions_for,
    state_label,
)
from c_core.state_client import SOURCE_CHAT, SOURCE_STATUS

#: frontend_C/main.py 在本进程里的专用模块名。**不能叫 "main"** ——
#: 整仓跑 pytest 时 backend_A / backend_B 的测试会先注册 sys.modules["main"]，
#: 三个模块都有一个 main.py。做法与 backend_B/tests/test_display_channel.py 一致。
_MAIN_MODULE_NAME = "frontend_c_main"


def load_frontend_c_main():
    """按文件路径加载 frontend_C/main.py（理由同上）。"""
    cached = sys.modules.get(_MAIN_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        _MAIN_MODULE_NAME, os.path.join(_FRONTEND_C, "main.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MAIN_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module

try:
    from PyQt5.QtGui import QFontDatabase
    from PyQt5.QtWidgets import QApplication

    from ui.camera import CameraPane
    from ui.layout import SplitWindow
    from ui.window import FaceWindow, StateBridge, pick_font_family
    QT_IMPORT_ERROR = None
except Exception as exc:                                    # noqa: BLE001
    QT_IMPORT_ERROR = exc

_app = None
_family = ""
_font_count = 0
_platform = ""

# ⚠️ 必须在**类定义之前**求值（``@unittest.skipUnless`` 在定义时就取结果），
# 理由见 tests/test_window.py 里同一段注释。
if QT_IMPORT_ERROR is None:
    _app = QApplication.instance() or QApplication([])
    _font_count = len(QFontDatabase().families())
    _platform = _app.platformName()
    _family = pick_font_family()


def has_real_fonts() -> bool:
    return _font_count > 0


def skip_reason() -> str:
    return (f"平台 {_platform!r} 的字体库为空（{_font_count} 个字体族），"
            f"Qt 在此平台画不出任何字。用 QT_QPA_PLATFORM=windows 重跑可启用。")


VAI = {
    "type": "vai",
    "index": 63.5,
    "index_status": "研究趋势（非认知专注）",
    "status": "有效",
    "modalities": ["gaze", "pose", "eye_open"],
    "modality_config_id": "凝视+头姿+睁眼",
    "valid_seconds": 42.0,
    "note": "研究趋势（非认知专注）；日常参考，非医疗结论",
}


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PyQt5 不可用：{QT_IMPORT_ERROR}")
class SplitWindowTestCase(unittest.TestCase):
    """公共脚手架：一对脸/右栏 + 分栏窗口，测试结束统一关闭。"""

    def setUp(self) -> None:
        self.face = FaceWindow(_family)
        self.pane = CameraPane(_family)
        self.window = SplitWindow(self.face, self.pane)
        self.addCleanup(self._close)

    def _close(self) -> None:
        # 先关子控件再关窗口：它们各自持有定时器（眨眼/轮换表情）
        self.face.close()
        self.pane.close()
        self.window.close()


class TestLayout(SplitWindowTestCase):

    def test_both_panes_are_present_and_owned(self):
        self.assertIs(self.window.face, self.face)
        self.assertIs(self.window.camera, self.pane)
        # addWidget 会把它们重新挂到分栏窗口下
        self.assertIs(self.face.parent(), self.window)
        self.assertIs(self.pane.parent(), self.window)

    def test_the_two_panes_share_the_width_equally(self):
        """「左右对半分」是需求原话 —— 用布局的 stretch 钉住它。"""
        layout = self.window.layout()
        stretches = [layout.stretch(i) for i in range(layout.count())]
        self.assertEqual(stretches[0], stretches[-1])
        self.assertGreater(stretches[0], 0)
        # 中间那条分隔线不参与拉伸
        self.assertEqual(stretches[1], 0)

    def test_real_geometry_after_show(self):
        """真显示出来之后两栏宽度应该接近 —— 布局确实生效了，不只是配置。"""
        self.window.resize(1200, 600)
        self.window.show()
        QApplication.processEvents()
        try:
            face_w, pane_w = self.face.width(), self.pane.width()
            if face_w <= 1 or pane_w <= 1:
                self.skipTest(f"平台 {_platform!r} 下控件没有真实几何尺寸")
            self.assertLessEqual(abs(face_w - pane_w), 8)
        finally:
            self.window.hide()


class TestForwarding(SplitWindowTestCase):
    """转发必须成对：一条更新落到两栏上，而不是只落到一栏。"""

    def test_set_state_reaches_both(self):
        self.window.set_state(STATE_SAD, "检测到低落")
        self.assertEqual(self.face._state, STATE_SAD)
        self.assertEqual(self.pane._state, STATE_SAD)
        self.assertEqual(self.pane._reason, "检测到低落")

    def test_set_link_reaches_both(self):
        self.window.set_link("chat", True)
        self.assertTrue(self.face._links["chat"])
        self.assertTrue(self.pane._links["chat"])

    def test_set_vai_reaches_the_right_pane(self):
        self.window.set_vai(VAI)
        self.assertEqual(self.pane._vai, VAI)

    def test_note_message_only_touches_the_hud(self):
        """回复文本只进左栏的调试 HUD。

        **右栏不是聊天记录区** —— 需求原话是「只展示表情」，右栏也只放
        画面与识别结果。所以这条转发是有意只走一半的。
        """
        self.window.note_message("我在呢")
        self.assertEqual(self.face._last_message, "我在呢")
        self.assertIsNone(self.pane._vai)


class TestNoSecondStateSource(SplitWindowTestCase):
    """专注度**不许**驱动表情（C 侧那条红线的可执行版本）。"""

    def test_vai_does_not_change_the_face(self):
        self.window.set_state(STATE_TIRED)
        before_state = self.face._state
        before_index = self.face._index

        self.window.set_vai({**VAI, "status": "有效"})
        self.window.set_vai({**VAI, "index": 100.0})
        self.window.set_vai({**VAI, "index": None, "status": "观察不足"})

        self.assertEqual(self.face._state, before_state)
        self.assertEqual(self.face._index, before_index)

    def test_vai_with_a_stray_state_field_still_does_not_change_the_face(self):
        """就算报文里混进了 ``state``，左栏也不许跟着变。"""
        self.window.set_state(STATE_NORMAL)
        self.window.set_vai({**VAI, "state": STATE_SAD})
        self.assertEqual(self.face._state, STATE_NORMAL)

    def test_face_expression_set_is_unchanged(self):
        """表情仍然只由四态决定（api_doc §4.2 不许自造）。"""
        self.window.set_vai(VAI)
        self.assertEqual(len(expressions_for(self.face._state)), 4)


class TestFontWidthClamp(SplitWindowTestCase):
    """分栏之后左栏只剩半宽：字号必须同时受宽度约束。

    断言一律按**目的**（最宽的颜文字真的放得下）而不是某个魔数 ——
    第一版实现用的是 ``width / 7.0`` 这个手调系数，实测偏大 50%，
    测试要是也照抄那个系数就一起错了。
    """

    def measured_width(self, point_size: int) -> int:
        """在给定字号下，最宽的颜文字要占多少像素。"""
        from ui.window import build_font
        from PyQt5.QtGui import QFontMetrics
        metrics = QFontMetrics(build_font(self.face._family, point_size))
        return max(metrics.horizontalAdvance(face) for face in all_faces())

    def test_a_wide_window_is_still_sized_by_height(self):
        """字号就是 ``min(按高度, 按宽度)`` —— 分栏只在"窄到会溢出"时才起作用。

        注意不能写死"1080p 下等于 ``height*0.16``"：最宽的颜文字要多少像素
        是**本机字体**的属性（实测开发机的 Microsoft YaHei UI 约 11px/磅，
        而离屏平台的回退字体约 16.6px/磅 —— 那个字体下 1920px 根本放不下
        172 磅）。所以这里按规则本身分两支断言，两支各自钉住一个目的。
        """
        width, height = 1920, 1080
        self.face.resize(width, height)
        size = self.face._font_point_size()
        by_height = int(height * 0.16)

        if self.measured_width(by_height) <= width:
            # 宽度不是瓶颈 → 仍然完全由高度决定（分栏前的老行为）
            self.assertEqual(size, by_height)
        else:
            # 宽度是瓶颈 → 钳制生效，且钳到的值确实放得下
            self.assertLess(size, by_height)
            self.assertLessEqual(self.measured_width(size), width)

    def test_the_widest_face_fits_in_a_half_width_pane(self):
        """半栏宽度下算出来的字号，要让最宽的那个颜文字放得下。

        1440 的窗口减去中间 1px 分隔线，一栏约 719px。
        """
        width = 719
        self.face.resize(width, 600)
        size = self.face._font_point_size()
        self.assertLessEqual(self.measured_width(size), width,
                             f"{size}pt 下最宽的颜文字放不进 {width}px 的栏")

    def test_a_narrow_pane_actually_shrinks_the_font(self):
        self.face.resize(320, 600)
        self.assertLess(self.face._font_point_size(), int(600 * 0.16),
                        "窄栏里没有触发宽度钳制")

    def test_every_face_of_every_state_fits(self):
        """逐个状态核对（含眨眼帧）—— 有一个放不下就会在切换时闪一下溢出。"""
        self.face.resize(719, 600)
        size = self.face._font_point_size()
        self.assertLessEqual(self.measured_width(size), 719)

    def test_the_floor_still_wins_in_an_absurdly_narrow_pane(self):
        """窗口被压到极窄时字号不许变成 0（会画不出来），也不许小于 24pt ——
        适老化要求字大，宁可溢出也不能小到看不清。"""
        self.face.resize(60, 600)
        self.assertGreaterEqual(self.face._font_point_size(), 24)

    def test_fixed_font_size_is_never_clamped(self):
        """显式 ``--font-size`` 时不许被悄悄改小 —— 那是用户的明确要求。"""
        face = FaceWindow(_family, font_size=42)
        self.addCleanup(face.close)
        face.resize(320, 600)
        self.assertEqual(face._font_point_size(), 42)


class TestCameraPaneRendering(unittest.TestCase):
    """右栏自己：能画、不崩、三种画面状态都画得出来。"""

    def setUp(self) -> None:
        self.pane = CameraPane(_family)
        self.addCleanup(self.pane.close)

    def test_snapshot_returns_a_pixmap_of_the_requested_size(self):
        pixmap = self.pane.snapshot(800, 600)
        self.assertFalse(pixmap.isNull())
        self.assertEqual((pixmap.width(), pixmap.height()), (800, 600))

    def test_paints_without_any_data_at_all(self):
        """从来没有收到过 vai / state / link —— B 没起来时就是这样，不许崩。"""
        for size in ((320, 240), (800, 600), (1600, 900), (200, 120)):
            with self.subTest(size=size):
                self.pane.snapshot(*size)

    def test_paints_all_three_stream_states(self):
        for url, connected in ((None, None), ("http://127.0.0.1:8010/stream.mjpeg", False),
                               ("http://127.0.0.1:8010/stream.mjpeg", True)):
            with self.subTest(url=url, connected=connected):
                pane = CameraPane(_family, stream_url=url)
                self.addCleanup(pane.close)
                if connected is not None:
                    pane.set_link("stream", connected)
                pane.set_vai(VAI)
                pane.set_state(STATE_TIRED, "连续打哈欠")
                self.assertFalse(pane.snapshot(600, 400).isNull())

    def test_paints_garbage_values(self):
        """字段类型乱来时也不许抛 —— 渲染是在 GUI 线程里跑的。"""
        pane = CameraPane(_family)
        self.addCleanup(pane.close)
        pane.set_vai({"index": "高", "status": None, "modalities": 42,
                      "valid_seconds": "很久", "note": 123})
        pane.set_vai("这不是 dict")
        pane.set_state(None)
        self.assertFalse(pane.snapshot(600, 400).isNull())

    def test_link_labels_cover_the_stream_channel(self):
        """``stream`` 必须在标签表里，否则标题栏/HUD 会打出英文原名。"""
        from ui.window import link_label
        self.assertEqual(link_label("stream"), "画面通道")

    @unittest.skipUnless(has_real_fonts(), skip_reason())
    def test_text_is_actually_drawn(self):
        """真切出白色的字（字体库非空的平台才跑）。"""
        self.pane.set_vai(VAI)
        self.pane.set_state(STATE_SAD, "检测到低落")
        self.pane.set_link("chat", True)
        pixmap = self.pane.snapshot(900, 600)
        image = pixmap.toImage()
        bright = sum(
            1
            for y in range(0, image.height(), 3)
            for x in range(0, image.width(), 3)
            if image.pixelColor(x, y).red() > 40
        )
        self.assertGreater(bright, 0, "右栏一个白点都没画出来")


def make_jpeg(width: int = 64, height: int = 48, color: str = "#20c0ff") -> bytes:
    """现编一张纯色 JPEG。

    **不 import cv2、也不读磁盘上的图片文件**：前者是 ``main.py`` 顶部那条
    硬约束（cv2 与 PyQt5 同进程会互相顶掉平台插件），后者会让这个仓库多出
    一类二进制资产。PyQt5 自己的 JPEG 插件编码一条纯色图绰绰有余，
    而且编解码走的是同一个插件 —— 与线上路径一致。
    """
    from PyQt5.QtCore import QBuffer
    from PyQt5.QtGui import QColor, QImage

    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor(color))
    buffer = QBuffer()
    buffer.open(QBuffer.WriteOnly)
    if not image.save(buffer, "JPG", 90):
        raise AssertionError("PyQt5 编不出 JPEG —— 这台机器的 JPEG 插件可能残缺")
    return bytes(buffer.data())


class TestCameraFrames(unittest.TestCase):
    """右栏的画面那条线：拉帧、解码、等比摆放、断开作废。

    这一层是**唯一**碰画面的地方，而且全部在 GUI 线程里 —— 所以这里
    直接调 ``pull_frame()`` 就等于"定时器到点了"，不需要真跑事件循环。
    """

    def _pane(self, fps: int = 15, slot=None, url: str = "http://127.0.0.1:8010/stream.mjpeg"):
        pane = CameraPane(_family, stream_url=url, stream_slot=slot, camera_fps=fps)
        self.addCleanup(pane.close)
        self.addCleanup(pane.stop_stream)
        return pane

    # ---------------------------------------------------------- 定时器

    def test_no_slot_means_no_timer_at_all(self):
        """没给 ``--stream-url`` 时连定时器都不该起（步骤①那条路径一行没变）。"""
        pane = self._pane(url=None)
        self.assertIsNone(pane._timer)
        self.assertIsNone(pane._pixmap)
        pane.pull_frame()          # 空调用也不许崩

    def test_a_slot_starts_the_pull_timer_at_the_requested_rate(self):
        from c_core.mjpeg_client import FrameSlot
        pane = self._pane(fps=20, slot=FrameSlot())
        self.assertIsNotNone(pane._timer)
        self.assertTrue(pane._timer.isActive())
        self.assertEqual(pane._timer.interval(), 50)      # 1000 / 20
        pane.stop_stream()
        self.assertIsNone(pane._timer, "stop_stream 之后定时器应当被丢掉")
        pane.stop_stream()          # 幂等

    def test_an_absurd_fps_is_clamped_not_honoured(self):
        """``--camera-fps 0`` 会让定时器空转烧 CPU，必须被钳住。"""
        from c_core.mjpeg_client import FrameSlot
        pane = self._pane(fps=0, slot=FrameSlot())
        self.assertEqual(pane.fps, 1)

    # ---------------------------------------------------------- 取帧与解码

    def test_a_frame_in_the_slot_is_decoded_and_drawn(self):
        from c_core.mjpeg_client import FrameSlot
        slot = FrameSlot()
        pane = self._pane(slot=slot)
        slot.put(make_jpeg())
        pane.pull_frame()
        self.assertIsNotNone(pane._pixmap)
        self.assertFalse(pane._pixmap.isNull())
        self.assertEqual(pane.stats["drawn"], 1)
        self.assertGreater(pane.frame_seq, 0)

    def test_nothing_new_means_no_decoding(self):
        """**这条是"定时器比流快也不花钱"的全部依据。**

        ``take()`` 取走即清空，所以第二、第三次拉不到东西 —— 不重新解码，
        也不重复 ``update()``。
        """
        from c_core.mjpeg_client import FrameSlot
        slot = FrameSlot()
        pane = self._pane(slot=slot)
        slot.put(make_jpeg())
        for _ in range(5):
            pane.pull_frame()
        self.assertEqual(pane.stats["drawn"], 1)

    def test_a_corrupt_frame_keeps_the_previous_picture(self):
        """一帧坏数据只该记账，**不该把画面清空**。

        闪一下黑屏看起来像"摄像头接触不良"，会把人引到完全错误的方向；
        而且这一帧之后立刻就有下一帧，清空的唯一效果就是闪。
        """
        from c_core.mjpeg_client import FrameSlot
        slot = FrameSlot()
        pane = self._pane(slot=slot)
        slot.put(make_jpeg())
        pane.pull_frame()
        good = pane._pixmap
        slot.put(b"\xff\xd8\xff" + b"not a jpeg at all")
        pane.pull_frame()
        self.assertEqual(pane.stats["decode_errors"], 1)
        self.assertEqual(pane.stats["drawn"], 1)
        self.assertIs(pane._pixmap, good, "坏帧把上一张好图顶掉了")

    def test_the_frame_sequence_tracks_the_slot(self):
        from c_core.mjpeg_client import FrameSlot
        slot = FrameSlot()
        pane = self._pane(slot=slot)
        for _ in range(3):
            slot.put(make_jpeg())
            pane.pull_frame()
        self.assertEqual(pane.frame_seq, slot.seq)

    # ---------------------------------------------------------- 断开作废

    def test_a_stream_disconnect_throws_the_picture_away(self):
        """**定格的画面和实时画面长得一模一样。**

        区别只在它不再更新，而屏幕上没有任何别的地方能看出这个区别 ——
        所以只能把它清掉，让占位文字去说"未连接，正在重连"。
        """
        from c_core.mjpeg_client import FrameSlot
        pane = self._pane(slot=FrameSlot())
        pane.set_link("stream", True)
        pane._pixmap = None
        slot = pane._slot
        slot.put(make_jpeg())
        pane.pull_frame()
        self.assertIsNotNone(pane._pixmap)

        pane.set_link("stream", False)
        self.assertIsNone(pane._pixmap)
        self.assertEqual(pane.stats["cleared"], 1)

    def test_connecting_does_not_clear_anything(self):
        from c_core.mjpeg_client import FrameSlot
        pane = self._pane(slot=FrameSlot())
        pane._slot.put(make_jpeg())
        pane.pull_frame()
        pane.set_link("stream", True)
        self.assertIsNotNone(pane._pixmap)

    def test_other_channels_never_touch_the_picture(self):
        """只有 ``stream`` 断开才作废画面。B 掉线不该影响它（两条链路无关）。"""
        from c_core.mjpeg_client import FrameSlot
        pane = self._pane(slot=FrameSlot())
        pane.set_link("stream", True)
        pane._slot.put(make_jpeg())
        pane.pull_frame()
        pane.set_link("chat", False)
        pane.set_link("status", False)
        self.assertIsNotNone(pane._pixmap)
        self.assertEqual(pane.stats["cleared"], 0)

    # ---------------------------------------------------------- 摆放

    def test_the_frame_is_letterboxed_not_stretched(self):
        """等比缩放到框内并居中 —— 拉伸会**把 A 烧进画面的那两行字拉变形**，
        而那两行恰恰是整块东西里唯一必须能读的部分。"""
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPixmap

        source = QPixmap(200, 100)                 # 2:1
        box = QRectF(0, 0, 400, 400)               # 1:1 的框
        rect = CameraPane._fit_rect(box, source)
        self.assertAlmostEqual(rect.width(), 400.0)      # 以宽为准
        self.assertAlmostEqual(rect.height(), 200.0)     # 高按比例，不拉满
        self.assertAlmostEqual(rect.top(), 100.0)        # 上下居中
        self.assertAlmostEqual(rect.left(), 0.0)

    def test_a_tall_frame_is_fitted_by_height(self):
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPixmap

        rect = CameraPane._fit_rect(QRectF(0, 0, 400, 400), QPixmap(100, 200))
        self.assertAlmostEqual(rect.height(), 400.0)
        self.assertAlmostEqual(rect.width(), 200.0)
        self.assertAlmostEqual(rect.left(), 100.0)

    def test_a_matching_aspect_ratio_fills_the_box(self):
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPixmap

        rect = CameraPane._fit_rect(QRectF(10, 20, 640, 480), QPixmap(640, 480))
        self.assertAlmostEqual(rect.width(), 640.0)
        self.assertAlmostEqual(rect.height(), 480.0)

    # ---------------------------------------------------------- 画出来

    def test_the_real_frame_reaches_the_canvas(self):
        """端到端：单槽 → 解码 → 画。用一张**颜色很跳**的纯色图，
        在黑底加白字的界面上一眼就能数出它有没有真的上屏。"""
        from c_core.mjpeg_client import FrameSlot
        pane = self._pane(slot=FrameSlot())
        pane.set_vai(VAI)
        pane.set_state(STATE_TIRED, "连续打哈欠")
        pane._slot.put(make_jpeg(color="#20c0ff"))
        pane.pull_frame()
        image = pane.snapshot(900, 600).toImage()
        blue = sum(
            1
            for y in range(0, image.height(), 2)
            for x in range(0, image.width(), 2)
            if image.pixelColor(x, y).blue() > 150 and image.pixelColor(x, y).red() < 120
        )
        self.assertGreater(blue, 200, "画面没上屏（一个青色像素都没有）")

    def test_the_placeholder_is_gone_once_a_frame_arrives(self):
        """有画面之后那个占位说明就不该再画 —— 两个一起出现会让人以为
        画面是"叠"在未接入状态上的。"""
        from c_core.mjpeg_client import FrameSlot
        from c_core import display_text as text_lib

        pane = self._pane(slot=FrameSlot())
        pane.set_link("stream", True)
        pane._slot.put(make_jpeg(color="#ffffff"))
        pane.pull_frame()
        # 占位文字是灰的（140,140,140），画面是纯白 —— 数一下接近"灰字"的像素
        image = pane.snapshot(900, 600).toImage()
        grey = sum(
            1
            for y in range(0, image.height(), 2)
            for x in range(0, image.width(), 2)
            if abs(image.pixelColor(x, y).red() - 140) < 12
        )
        # 文本块的通道行也是这个灰，所以不可能数到 0；关键是别整块都是灰的
        self.assertLess(grey, 4000, f"占位文字好像还在（{text_lib.STREAM_WAITING}）")

    def test_it_paints_before_the_first_frame_arrives(self):
        """连上了但一帧还没来 —— 占位文字要说"等待第一帧"，且不许崩。"""
        from c_core.mjpeg_client import FrameSlot
        pane = self._pane(slot=FrameSlot())
        pane.set_link("stream", True)
        self.assertFalse(pane.snapshot(900, 600).isNull())
        self.assertEqual(pane.frame_seq, 0)
        self.assertIsNone(pane._pixmap)


class TestStreamWiring(unittest.TestCase):
    """``main.py`` 里画面通道的接线：CLI、通道来源、构造。"""

    def setUp(self):
        self.main = load_frontend_c_main()

    def test_the_cli_has_the_stream_flags(self):
        parser = self.main.build_parser()
        dests = {action.dest for action in parser._actions}
        self.assertIn("stream_url", dests)
        self.assertIn("camera_fps", dests)
        args = parser.parse_args([])
        self.assertIsNone(args.stream_url, "默认必须是**不拉流**")
        self.assertEqual(args.camera_fps, 15)

    def test_build_stream_link_feeds_the_slot_and_only_emits_on_link(self):
        """帧那条路**不经过信号** —— 它直接进单槽，GUI 用定时器主动拉。"""
        from c_core.mjpeg_client import FrameSlot, SOURCE_STREAM

        slot = FrameSlot()
        bridge = StateBridge()
        links = []
        bridge.linkChanged.connect(lambda s, c: links.append((s, c)))

        args = _FakeArgs()
        args.stream_url = "http://127.0.0.1:8010/stream.mjpeg"
        client = self.main._build_stream_link(args, slot, bridge)
        self.addCleanup(client.stop)

        self.assertIs(client.slot, slot)
        # 绑定方法每次访问都是新对象，只能比相等不能比同一
        self.assertEqual(client._on_frame, slot.put)   # 默认的 on_frame 就是单槽
        self.assertEqual((client.host, client.port, client.path),
                         ("127.0.0.1", 8010, "/stream.mjpeg"))

        # 线程没起，直接调回调：只 emit，不碰控件
        client._set_connected(True)
        self.assertEqual(links, [(SOURCE_STREAM, True)])
        client._set_connected(False)
        self.assertEqual(links[-1], (SOURCE_STREAM, False))

    def test_the_stream_channel_reaches_the_camera_pane(self):
        """``stream`` 那条连接状态必须真的走到右栏（它靠 ``_links["stream"]``
        决定显示"未连接（重连中）"还是"等待第一帧"）。"""
        pane = CameraPane(_family, stream_url="http://127.0.0.1:8010/stream.mjpeg")
        face = FaceWindow(_family)
        window = SplitWindow(face, pane)
        # addCleanup 是后进先出：窗口必须**最后**关（它会连子控件一起销毁，
        # 之后再 close() 子控件就是 'wrapped C/C++ object has been deleted'）。
        self.addCleanup(window.close)

        bridge = StateBridge()
        bridge.linkChanged.connect(window.set_link)
        bridge.linkChanged.emit("stream", True)
        self.assertIs(pane._stream_connected(), True)
        bridge.linkChanged.emit("stream", False)
        self.assertIs(pane._stream_connected(), False)


class TestBridgeSignals(unittest.TestCase):
    """信号桥：C 侧所有数据都经它排队回 GUI 线程。"""

    def test_vai_signal_carries_the_whole_message(self):
        bridge = StateBridge()
        received = []
        bridge.vaiArrived.connect(received.append)      # 同线程 = 直接调用
        bridge.vaiArrived.emit(VAI)
        self.assertEqual(received, [VAI])

    def test_state_reason_signal_carries_both(self):
        bridge = StateBridge()
        received = []
        bridge.stateReasonChanged.connect(lambda s, r: received.append((s, r)))
        bridge.stateReasonChanged.emit(STATE_TIRED, "连续打哈欠")
        self.assertEqual(received, [(STATE_TIRED, "连续打哈欠")])

    def test_the_existing_signals_are_unchanged(self):
        """``stateChanged`` 仍然是**单参数** —— FaceWindow.set_state 的签名
        依赖它，加参数会把这个公开接口改掉。"""
        bridge = StateBridge()
        received = []
        bridge.stateChanged.connect(received.append)
        bridge.stateChanged.emit(STATE_SAD)
        self.assertEqual(received, [STATE_SAD])


class _FakeArgs:
    """``_build_link`` 只读这四个属性，不需要真的跑一遍 argparse。"""

    host = "127.0.0.1"
    chat_port = 8002
    status_port = 8001

    def __init__(self, also_status: bool = False) -> None:
        self.also_status = also_status


class TestVaiIsDroppedWhenTheLinkDies(unittest.TestCase):
    """B 一断，右栏那条专注度**必须作废**（``main.py::_build_link`` 的 emit_link）。

    一个数值不会自己变成"过期"：它会一直显示着最后一个数，进度条也在，
    而界面上没有任何别的地方能看出它已经不更新了。状态那条有兜底
    （断线回落 ``normal``，``state_client`` 负责），专注度没有，
    所以只能在接线这里补 —— 这个文件是它唯一的落点。
    """

    def setUp(self):
        self.bridge = StateBridge()
        self.links = []
        self.vais = []
        self.bridge.linkChanged.connect(lambda s, c: self.links.append((s, c)))
        self.bridge.vaiArrived.connect(self.vais.append)
        self.link = load_frontend_c_main()._build_link(_FakeArgs(), self.bridge)

    def _emit(self, source, connected):
        """直接调 B 侧会调的那个回调（不在测试里真起 socket）。"""
        self.link._on_link(source, connected)

    def test_chat_disconnect_drops_the_value(self):
        self._emit(SOURCE_CHAT, True)
        self.vais.clear()
        self._emit(SOURCE_CHAT, False)
        self.assertEqual(self.vais, [None])
        self.assertEqual(self.links[-1], (SOURCE_CHAT, False))

    def test_status_disconnect_does_not_touch_the_value(self):
        """``--also-status`` 时 8001 掉了不影响 8002 上的专注度，不该清掉它。"""
        self._emit(SOURCE_STATUS, False)
        self.assertEqual(self.vais, [])

    def test_reconnecting_does_not_clear_it(self):
        """连上时不清 —— B 会立刻补发当前值，清一下只会闪一次"未提供"。"""
        self._emit(SOURCE_CHAT, True)
        self.assertEqual(self.vais, [])


class _RecordingWindow:
    """``_install_demo_cycle`` 只调 ``set_state``，不需要真的控件。

    故意**不是** MagicMock：这里要断言的是"它被调到了、且参数是四态之一"，
    一个记名单比一个自动生成属性的假对象更能说明问题（假对象会吞掉
    拼错的属性名，而 ``set_state`` 拼错恰恰是这条路径当年坏掉的方式）。
    """

    def __init__(self) -> None:
        self.states = []

    def set_state(self, state, reason="") -> None:
        self.states.append(state)


@unittest.skipIf(QT_IMPORT_ERROR is not None, f"PyQt5 不可用：{QT_IMPORT_ERROR}")
class TestDemoCycle(unittest.TestCase):
    """``--demo`` 必须真能起来 —— 它是「B 起不来时的兜底」。

    2026-09-30 修掉的那个 bug：``_install_demo_cycle`` 是**模块级**函数，
    而 ``state_label`` 只在 ``main()`` 里被 import，于是 ``--demo``
    **每次启动都在第一行 tick 里 NameError 崩掉**。这条路径从步骤①
    重构之后就一直是坏的，也没人发现，因为常规跑法（连 B）根本不走它。

    这条用例不问"切得对不对"（那是 ``set_state`` 的职责），只问
    **"跑不跑得完"** —— 那种错误只有一个症状：抛异常。
    """

    def setUp(self):
        self.window = _RecordingWindow()
        self.timer = None

    def tearDown(self):
        if self.timer is not None:
            self.timer.stop()

    def test_the_first_tick_sets_a_state_without_raising(self):
        main = load_frontend_c_main()
        # 不抛异常就是这条用例的全部意义（NameError 会在这里冒出来）
        self.timer = main._install_demo_cycle(self.window, VALID_STATES, _app)
        self.assertEqual(len(self.window.states), 1,
                         "tick() 应当在安装时立刻切一次，不用等第一个间隔")
        self.assertIn(self.window.states[0], VALID_STATES)

    def test_it_cycles_and_wraps_around(self):
        """连续 tick 要把四态走一遍再回到第一个 —— 循环的下标算错的话，
        演示到一半会停在一个状态上不动。"""
        main = load_frontend_c_main()
        self.timer = main._install_demo_cycle(self.window, VALID_STATES, _app)
        for _ in range(len(VALID_STATES) - 1):
            self.timer.timeout.emit()
        self.assertEqual(self.window.states, list(VALID_STATES))

    def test_it_logs_a_chinese_label_for_every_state(self):
        """``state_label`` 查不到某个状态会 KeyError —— 四态必须都有中文名。"""
        for state in VALID_STATES:
            with self.subTest(state=state):
                self.assertTrue(state_label(state))

    def test_the_interval_is_the_documented_one(self):
        main = load_frontend_c_main()
        self.timer = main._install_demo_cycle(self.window, VALID_STATES, _app)
        self.assertEqual(self.timer.interval(), int(main.DEMO_INTERVAL * 1000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
