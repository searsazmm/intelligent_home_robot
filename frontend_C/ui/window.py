# -*- coding: utf-8 -*-
"""纯黑背景 + 白色颜文字的窗口。

需求（用户原话）：
    「窗口背景纯黑色；显示白色颜文字表情，根据 backend_B 发来的指令切换表情；
      只展示表情，不要输入框、聊天文字框。」

所以这个窗口**只有一张脸**。没有输入框、没有聊天记录、没有按钮、没有状态文字。
连接状态只出现在**标题栏**（全屏时看不到，所以另有 ``--log-file``）和日志里；
``--debug-hud`` 可以在左上角叠一行小号灰字，默认关闭，只供联调。

两个实现选择值得说明：

1. **不用 QLabel，直接在 paintEvent 里画。**
   QLabel 的样式表背景继承很容易盖掉窗口的黑色底（要额外写
   ``background:transparent`` 去补），而且它按 bbox 居中 —— 颜文字的包围盒
   上下留白不对称（``(￣▽￣)`` 和 ``(╥﹏╥)`` 的高度差很多），按 bbox 居中的结果是
   每换一个表情整张脸就上下跳一下。直接绘制可以按「基线 + 视觉中心」定位。

2. **字体在运行时探测，不硬编码。**
   开发机上 ``Microsoft YaHei`` / ``SimSun`` / ``SimHei`` **都不存在**，
   而 ``Microsoft YaHei`` 是网上最常见的写法。写死它不会报错，只会静默回退到
   无衬线体，颜文字里的 ``＾ ▽ ￣ ﹏`` 就可能变成豆腐块。
   实测可用的是 ``Microsoft YaHei UI``（首选）。
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from PyQt5.QtCore import (
    QEasingCurve, QObject, QPropertyAnimation, QRectF, QTimer, Qt, pyqtProperty,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget

from c_core.expressions import (
    DEFAULT_STATE, Expression, expressions_for, normalize_state, state_label, all_faces,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 字体候选（按优先级）。中英文名都列上 —— 同一个字体在不同系统上
# 可能只以本地化名出现（实测：请求 "Microsoft YaHei" 会解析成 "微软雅黑"）。
# --------------------------------------------------------------------------
FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "微软雅黑",
    "Microsoft YaHei",
    "MS Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "Malgun Gothic",
    "Segoe UI Symbol",
)

#: 眨眼持续时间（毫秒）。太快看不见，太慢像卡住。
BLINK_MS = 130
#: 两次眨眼的间隔区间（毫秒）。随机化，否则像节拍器。
BLINK_GAP_MS = (2200, 6000)
#: 同状态内轮换表情的间隔（毫秒）
CYCLE_MS = 7000
#: 状态切换时的淡入时长（毫秒）
FADE_MS = 180

#: 通道名 → 中文。BackendLink 传的是 "chat"/"status"，直接显示不好读。
LINK_LABELS = {"chat": "对话通道", "status": "状态通道"}


def link_label(source: str) -> str:
    return LINK_LABELS.get(source, source)


def pick_font_family() -> str:
    """返回本机第一个可用的颜文字字体名；都不在就退回 Qt 默认。

    注意：Qt 的字体回退会让「字体不存在」这件事不报错、只是字形悄悄变样，
    所以这里必须主动探测，并在日志里说清楚选到了哪个。
    """
    available = set(QFontDatabase().families())
    for name in FONT_CANDIDATES:
        if name in available:
            logger.info("颜文字字体：%s", name)
            return name
    fallback = QFont().defaultFamily()
    logger.warning(
        "没有找到任何候选颜文字字体（尝试过 %s），退回 Qt 默认字体 %r；"
        "颜文字可能出现豆腐块，建议跑一次 --dump-glyphs 检查",
        " / ".join(FONT_CANDIDATES), fallback,
    )
    return fallback


def build_font(family: str, point_size: int) -> QFont:
    """构造字体。

    用 ``setFamilies`` 而不是 ``setFamily``：前者能给出**回退链**，
    某个字体缺个别字形时 Qt 会继续往下找，而不是直接画豆腐块。
    Qt 5.13+ 支持，5.15 上稳定可用。
    """
    font = QFont()
    families = [family] + [f for f in FONT_CANDIDATES if f != family]
    try:
        font.setFamilies(families)
    except AttributeError:
        # 老版本 Qt 没有 setFamilies，退化为单一字体
        font.setFamily(family)
    font.setPointSize(max(8, point_size))
    return font


class StateBridge(QObject):
    """把后台线程的事件转成 Qt 信号。

    ⚠️ **这是本模块唯一允许被 socket 线程触碰的对象。**

    信号槽的跨线程投递有个容易踩的坑：连接到一个**不带 QObject 上下文**的
    可调用体（lambda / functools.partial）时，Qt 无法判断接收者在哪个线程，
    会退化成 DirectConnection —— 于是槽函数就在 socket 线程里执行了。
    如果那个槽碰了控件，轻则闪烁重则崩溃。

    所以上层的连接一律是 ``bridge.stateChanged.connect(self._on_state)`` 这种
    **绑定方法**（self 是 QWidget，属于 GUI 线程），Qt 才会用 QueuedConnection
    把调用排进 GUI 线程的事件循环。emit 本身跨线程是安全的。
    """

    stateChanged = pyqtSignal(str)          # 状态字符串（已规整）
    linkChanged = pyqtSignal(str, bool)     # 通道名, 是否已连接
    replyArrived = pyqtSignal(str)          # 机器人回复文本（仅记日志用）
    proactiveArrived = pyqtSignal(str)      # 主动关怀文本（仅记日志用）


class FaceWindow(QWidget):
    """只有一张白色颜文字的纯黑窗口。"""

    def __init__(
        self,
        family: str,
        font_size: Optional[int] = None,
        debug_hud: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._family = family
        self._fixed_font_size = font_size       # None = 按窗口高度自适应
        self._debug_hud = debug_hud

        self._state = DEFAULT_STATE
        self._index = 0
        self._blinking = False
        self._opacity = 1.0
        self._links: dict = {}                  # 通道名 → 是否已连接
        self._last_message = ""
        self._closing = False

        self.setWindowTitle("居家陪伴机器人")
        # 黑底靠 paintEvent 自己刷，这里关掉自动擦除，避免首帧闪白
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.resize(900, 600)

        # ---- 淡入动画（状态切换时用） ----
        self._fade = QPropertyAnimation(self, b"faceOpacity", self)
        self._fade.setDuration(FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)

        # ---- 眨眼：两个单次定时器轮流接力 ----
        self._blink_off = QTimer(self)
        self._blink_off.setSingleShot(True)
        self._blink_off.timeout.connect(self._end_blink)
        self._blink_gap = QTimer(self)
        self._blink_gap.setSingleShot(True)
        self._blink_gap.timeout.connect(self._start_blink)
        self._schedule_blink()

        # ---- 同状态内轮换表情 ----
        self._cycle = QTimer(self)
        self._cycle.timeout.connect(self._next_expression)
        self._cycle.start(CYCLE_MS)

    # ------------------------------------------------------------------
    # 对外
    # ------------------------------------------------------------------

    def set_state(self, state: str) -> None:
        """切换状态。收到的字符串一律先规整（api_doc §4.3）。"""
        state = normalize_state(state)
        if state == self._state:
            return
        logger.info("表情切换：%s → %s", self._state, state)
        self._state = state
        self._index = 0
        self._blinking = False
        self._refresh_title()
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self.update()

    def set_link(self, source: str, connected: bool) -> None:
        self._links[source] = connected
        self._refresh_title()
        self.update()

    def note_message(self, text: str) -> None:
        """记下最近一条回复/关怀文本，只在 HUD 里显示，不进入窗口主体。"""
        self._last_message = text

    def snapshot(self, state: str, width: int = 900, height: int = 600,
                 blinking: bool = False) -> QPixmap:
        """把某个状态渲染成一帧位图（``--screenshot`` 与单元测试用）。

        两个必须做的动作，否则抓到的是**纯黑**的一帧：

        1. **停掉并跳过淡入动画。** 状态切换会把 ``faceOpacity`` 置 0 再动画到 1，
           而 ``QPropertyAnimation`` 需要事件循环来推进 —— 在 ``grab()`` 这种
           同步渲染里根本没有事件循环，于是永远停在 0，脸是全透明的。
        2. **不经 ``set_state()`` 直接赋值。** ``set_state`` 会因为「状态没变」
           提前返回，截图时想连续抓两帧同一状态就失效了。
        """
        self._fade.stop()
        self._state = normalize_state(state)
        self._index = 0
        self._blinking = bool(blinking)
        self._opacity = 1.0
        self.resize(width, height)
        return self.grab()

    # ------------------------------------------------------------------
    # Qt 属性（供淡入动画驱动）
    # ------------------------------------------------------------------

    def _get_opacity(self) -> float:
        return self._opacity

    def _set_opacity(self, value: float) -> None:
        self._opacity = max(0.0, min(1.0, float(value)))
        self.update()

    faceOpacity = pyqtProperty(float, fget=_get_opacity, fset=_set_opacity)

    # ------------------------------------------------------------------
    # 动画
    # ------------------------------------------------------------------

    def _schedule_blink(self) -> None:
        if self._closing:
            return
        self._blink_gap.start(random.randint(*BLINK_GAP_MS))

    def _start_blink(self) -> None:
        if self._closing:
            return
        self._blinking = True
        self.update()
        self._blink_off.start(BLINK_MS)

    def _end_blink(self) -> None:
        self._blinking = False
        self.update()
        self._schedule_blink()

    def _next_expression(self) -> None:
        faces = expressions_for(self._state)
        if len(faces) > 1:
            self._index = (self._index + 1) % len(faces)
            self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def _current_expression(self) -> Expression:
        faces = expressions_for(self._state)
        return faces[self._index % len(faces)]

    def _font_point_size(self) -> int:
        """算出字号。

        颜文字的「视觉大小」取决于窗口高度而不是宽度，所以按高度取比例。
        0.16 是实测试出来的：全屏 1080p 下约 170pt，三四米外也看得清，
        且最长的那几个（``(∪.∪ )...zzz``）不会顶到边缘。
        """
        if self._fixed_font_size:
            return self._fixed_font_size
        return max(24, int(self.height() * 0.16))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # 1) 纯黑底。用 fillRect 而不是 stylesheet —— 不受系统主题影响。
        painter.fillRect(self.rect(), Qt.black)

        # 2) 白色的脸
        expression = self._current_expression()
        text = expression.blink if self._blinking else expression.face

        font = build_font(self._family, self._font_point_size())
        painter.setFont(font)
        color = QColor(255, 255, 255)
        color.setAlphaF(self._opacity)
        painter.setPen(color)

        # 垂直居中要按「整行的高度」而不是字形包围盒，
        # 否则 (￣▽￣) 和 (╥﹏╥) 会因为字形高度不同而上下跳。
        metrics = QFontMetrics(font)
        line_height = metrics.height()
        baseline = (self.height() + metrics.ascent() - metrics.descent()) / 2
        rect = QRectF(0, baseline - metrics.ascent(), self.width(), line_height)
        painter.drawText(rect, int(Qt.AlignHCenter | Qt.AlignVCenter), text)

        # 3) 调试用 HUD（默认关闭）
        if self._debug_hud:
            self._paint_hud(painter)

        painter.end()

    def _paint_hud(self, painter: QPainter) -> None:
        hud_font = QFont(self._family)
        hud_font.setPointSize(11)
        painter.setFont(hud_font)
        painter.setPen(QColor(120, 120, 120))   # 小号灰字，不抢戏
        lines = [f"{self._state} / {state_label(self._state)}"]
        if self._links:
            lines.append(" ".join(
                f"{link_label(k)}:{'通' if v else '断'}" for k, v in self._links.items()
            ))
        else:
            lines.append("未连接模块 B")
        if self._last_message:
            lines.append(self._last_message[:40])
        for row, line in enumerate(lines):
            painter.drawText(12, 22 + row * 18, line)

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        """ESC / Q 退出。

        **全屏时必须留一条退路。** 答辩现场卡在一个全屏黑窗口里是不可恢复的，
        所以 ESC 在全屏时退出全屏、在窗口模式下退出程序；Q 一律直接退出。
        """
        key = event.key()
        if key == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        elif key == Qt.Key_Q or (key == Qt.Key_Q and event.modifiers() & Qt.ControlModifier):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """停掉定时器并标记关闭，避免窗口析构后定时器还在回调。"""
        self._closing = True
        for timer in (self._blink_off, self._blink_gap, self._cycle):
            timer.stop()
        self._fade.stop()
        super().closeEvent(event)

    def _refresh_title(self) -> None:
        links = "，".join(
            f"{link_label(k)}{'通' if v else '断'}" for k, v in self._links.items()
        ) or "未连接"
        self.setWindowTitle(f"居家陪伴机器人 · {state_label(self._state)} · {links}")


class GlyphDumpWindow(QWidget):
    """``--dump-glyphs``：把每个颜文字连同它用到的字符码位平铺出来。

    存在的理由：Qt 的字体回退会让「某个字形缺失」这件事**不报错** ——
    只是那一个字悄悄变成方块。所以答辩前必须在**演示机上**用肉眼过一遍。

    窗口高度**按表情数量算出来**，而不是随手写一个数：字形 + 码位标签
    一共需要约 64px 的垂直空间，写死窗口高度会让标签落进下一行的字形里，
    而码位标签恰恰是字形缺失时唯一的定位线索 —— 偏偏在那时读不清就白做了。
    """

    #: 两列排版；每格垂直空间下限（字形 + 码位标签 + 间隔）
    COLUMNS = 2
    MIN_CELL_HEIGHT = 64
    HEADER_HEIGHT = 56

    def __init__(self, family: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._family = family
        self.setWindowTitle("颜文字字形自检 —— 看不到方块即为正常")

        faces = all_faces()
        rows = (len(faces) + self.COLUMNS - 1) // self.COLUMNS
        self._rows_per_column = rows
        self.resize(1180, self.HEADER_HEIGHT + rows * self.MIN_CELL_HEIGHT)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.fillRect(self.rect(), Qt.black)

        faces = all_faces()
        rows_per_column = self._rows_per_column
        cell_w = self.width() // self.COLUMNS
        # 取「算出来的高度」和「窗口实际高度」里较大的那个，
        # 这样用户把窗口拉高只会更宽松，拉矮也不会让标签叠字。
        cell_h = max(
            self.MIN_CELL_HEIGHT,
            (self.height() - self.HEADER_HEIGHT) // max(1, rows_per_column),
        )

        face_font = build_font(self._family, 26)
        label_font = QFont(self._family)
        label_font.setPointSize(10)   # 9pt 在投影上太小，码位会糊成一团

        # 表头：说清楚在看什么
        painter.setFont(label_font)
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(24, 26, f"字体 {self._family} · 共 {len(faces)} 个颜文字（含眨眼帧）")
        painter.drawText(24, 42, "每一行都应是完整的颜文字；出现 □ ▯ 或空白即为该字形缺失")

        for i, face in enumerate(faces):
            col = i // rows_per_column
            row = i % rows_per_column
            x = col * cell_w + 24
            y = self.HEADER_HEIGHT + row * cell_h

            painter.setFont(face_font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x, y + 24, face)

            # 标出每个字符的码位：一旦有豆腐块，能直接对着码位去查是哪个字
            codes = " ".join(f"U+{ord(ch):04X}" for ch in face if ch not in "()")
            painter.setFont(label_font)
            painter.setPen(QColor(140, 140, 140))
            painter.drawText(x, y + 44, codes)

        painter.end()

    def snapshot(self, width: int = 1180, height: Optional[int] = None) -> QPixmap:
        """渲染成位图（``--screenshot`` 用）。"""
        self.resize(width, height or self.height())
        return self.grab()
