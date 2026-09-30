# -*- coding: utf-8 -*-
"""右栏：画面 + 识别结果（面部 / 专注度 / 状态文字）。

--------------------------------------------------------------------------
画面从哪儿来
--------------------------------------------------------------------------
来自 A 侧那条 MJPEG 展示流，由 :mod:`c_core.mjpeg_client` 拉（纯标准库），
本类只做两件事：**按自己的节奏取最新一帧**、**解码并画进那个框**。

不用 QLabel 装画面，也不 import cv2：

* 不用 QLabel 是沿用本模块既有约定（见下第 1 条）；
* 不用 cv2 是硬性的 —— ``main.py`` 顶部写了原因（cv2 与 PyQt5 同进程会
  互相顶掉对方的平台插件）。解码走 ``QPixmap.loadFromData(..., "JPG")``，
  用的是 PyQt5 自带的 JPEG 插件，**多这一路画面不会让那条约束松动**。

--------------------------------------------------------------------------
为什么不 emit 带图片的信号，而是 QTimer 主动拉
--------------------------------------------------------------------------
见 ``c_core/mjpeg_client.py`` 的模块文档：**Qt 的信号不会合并**，
每帧 emit 一个带 payload 的信号等于把事件队列当成帧队列，GUI 一慢就越堆
越多。所以 socket 线程只往 :class:`~c_core.mjpeg_client.FrameSlot` 里放，
本类用 ``QTimer`` 以 ``--camera-fps`` 的节奏主动 ``take()``。拉得慢就自然
掉帧 —— 这与 A 侧单槽缓存的语义是同一套。

**拉不到新帧时一次解码都不做**（``take()`` 返回 ``None`` 即无新帧），
所以定时器比流的 fps 快不会白烧 CPU。

--------------------------------------------------------------------------
沿用本模块的两条既有约定
--------------------------------------------------------------------------
1. **不用 QLabel，直接 paintEvent 画**（理由见 ``ui/window.py`` 顶部第 1 条：
   QLabel 的样式表背景继承很容易盖掉黑底，要额外打补丁）。
2. **文字内容全部来自** :mod:`c_core.display_text` 的纯函数。这里只负责
   选字体、排版、上色 —— 一个字的文案都不写死在这个文件里，
   否则文案就没法脱离图形环境单测了。

--------------------------------------------------------------------------
线程铁律（与 ``ui/window.py`` 的 ``StateBridge`` 同一条）
--------------------------------------------------------------------------
本类的方法**只允许在 GUI 线程里被调用**。socket 线程来的数据一律先经
``StateBridge`` 的 ``vaiArrived`` / ``stateReasonChanged`` / ``linkChanged``
信号排队回 GUI 线程，或经 ``FrameSlot`` 这一道单槽，再由本类在 GUI 线程里
取走。绝不能在 socket 线程里直接调 ``set_vai()`` 或 ``pull_frame()``。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget

from c_core import display_text as text_lib
from c_core.expressions import DEFAULT_STATE
from c_core.mjpeg_client import DEFAULT_CAMERA_FPS, FrameSlot, clamp_fps

logger = logging.getLogger(__name__)

#: 四边留白
MARGIN = 26
#: 段落之间的额外间隔
SECTION_GAP = 16

#: 正文字号 = 栏宽 / 这个值。900px 宽的栏 → 26pt，隔着两三米也读得清。
#: 之所以按**宽度**取（左栏那张脸是按高度取的），是因为这里的排版是
#: 一列文本：字太大就会横向溢出、或者被迫折行，折行会让整个区块高度失控。
TEXT_WIDTH_DIVISOR = 34.0
MIN_TEXT_POINT = 12
MAX_TEXT_POINT = 26

#: 灰色（标题、通道状态、免责声明这类次要信息）
GREY = QColor(140, 140, 140)
DIM = QColor(100, 100, 100)


def _line_height(metrics: QFontMetrics) -> int:
    return metrics.height()


class CameraPane(QWidget):
    """右栏控件：上半个占位框（将来放画面），下半个文本块。"""

    TITLE = "画面 / 识别结果"

    def __init__(
        self,
        family: str,
        stream_url: Optional[str] = None,
        stream_slot: Optional[FrameSlot] = None,
        camera_fps: int = DEFAULT_CAMERA_FPS,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._family = family
        self._stream_url = stream_url
        #: 本次运行是否**打算**拉流。None 的 stream_url 表示"没给地址"，
        #: 与"给了地址但没连上"是两件事，占位文字不同（见 display_text）。
        self._stream_planned = stream_url is not None

        self._vai: Optional[dict] = None
        self._state = DEFAULT_STATE
        self._reason = ""
        self._links: Dict[str, bool] = {}

        # ---- 画面 ----
        self._slot = stream_slot
        #: 最近一帧解出来的图。**它只在 GUI 线程里被读写**（socket 线程
        #: 碰的是 FrameSlot，碰不到这里）。
        self._pixmap: Optional[QPixmap] = None
        self._frame_seq = 0
        self._fps = clamp_fps(camera_fps)
        self._timer: Optional[QTimer] = None
        self.stats: Dict[str, int] = {
            #: 真的解码并画上去的帧数
            "drawn": 0,
            #: 解码失败被丢掉的帧数（JPEG 坏了 / 对面推了别的东西）
            "decode_errors": 0,
            #: 因为断开而把画面清空的次数
            "cleared": 0,
        }

        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setMinimumWidth(320)

        if self._slot is not None:
            self.start_stream()

    # ------------------------------------------------------------------
    # 画面：取帧与解码（全部在 GUI 线程）
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        """实际生效的取帧频率（``--camera-fps`` 被钳过之后的那个值）。"""
        return self._fps

    @property
    def frame_seq(self) -> int:
        """最近画上去的那一帧的序号。0 = 一帧都还没画过。"""
        return self._frame_seq

    def start_stream(self) -> None:
        """开始按 ``--camera-fps`` 的节奏取帧。幂等。"""
        if self._slot is None or self._timer is not None:
            return
        self._timer = QTimer(self)
        self._timer.setInterval(int(round(1000.0 / self._fps)))
        self._timer.timeout.connect(self.pull_frame)   # 绑定方法 → 队列连接，留在 GUI 线程
        self._timer.start()

    def stop_stream(self) -> None:
        """停下取帧定时器（退出时调）。幂等。"""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def pull_frame(self) -> None:
        """从单槽取**最新**一帧并解码。**只能在 GUI 线程里调用。**

        没有新帧时立刻返回，一次解码都不做 —— 定时器比流的 fps 快是常态，
        这条早退就是"多出来的定时器不花钱"的全部实现。

        解码失败（``QPixmap`` 是空的）时**保留上一帧**而不是清空：一帧坏
        数据就闪一下黑屏，比继续显示上一帧糟糕得多，而且那种闪烁看起来
        像"摄像头接触不良"，会把人引到完全错误的方向。坏帧只记账。
        """
        if self._slot is None:
            return
        got = self._slot.take()
        if got is None:
            return
        seq, jpeg = got

        pixmap = QPixmap()
        if not pixmap.loadFromData(jpeg, "JPG"):
            self.stats["decode_errors"] += 1
            if self.stats["decode_errors"] == 1:
                logger.warning("画面帧解码失败（%d 字节），已丢弃并保留上一帧", len(jpeg))
            return

        self._frame_seq = seq
        self._pixmap = pixmap
        self.stats["drawn"] += 1
        self.update()

    # ------------------------------------------------------------------
    # 对外（全部是 GUI 线程的槽）
    # ------------------------------------------------------------------

    def set_vai(self, message: Optional[dict]) -> None:
        """收到 B 的专注度展示报文（``{"type":"vai", ...}``）。

        ``None`` 是合法输入 —— 表示"从来没收到过"，界面显示「未提供」，
        与"收到了但 index 是 None"（B 明确说没算出来）分开显示。
        """
        self._vai = message if isinstance(message, dict) else None
        self.update()

    def set_state(self, state: str, reason: str = "") -> None:
        """状态文字。**注意与左栏那张脸是两条独立的更新路径**：
        脸只听 ``stateChanged``，文字听 ``stateReasonChanged``（带原因）。
        """
        self._state = state
        self._reason = reason or ""
        self.update()

    def set_link(self, source: str, connected: bool) -> None:
        """连接状态变化（GUI 线程）。**画面通道一断就清掉画面。**

        与 ``main.py`` 里「对话通道一断就把专注度作废」是同一条规矩：一帧
        定格的画面看起来和实时画面一模一样，区别只在它不再更新 ——
        而屏幕上没有任何别的地方能看出这个区别。留着它比留空更危险：
        老人摔倒了，屏幕上还是一张他好好坐着的照片。

        清空之后占位文字会说明"未连接，正在重连"，所以这一下闪空是
        **有解释的**，不是花屏。
        """
        was = self._links.get(source)
        self._links[source] = bool(connected)
        if source == "stream" and was and not connected and self._pixmap is not None:
            self._pixmap = None
            self.stats["cleared"] += 1
            logger.info("画面通道已断开，右栏画面作废（改显示未连接）")
        self.update()

    def configure_stream(self, stream_url: Optional[str]) -> None:
        """运行期改地址用（当前只有启动时调一次）。"""
        self._stream_url = stream_url
        self._stream_planned = stream_url is not None
        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def _text_point_size(self) -> int:
        return max(MIN_TEXT_POINT,
                   min(MAX_TEXT_POINT, int(self.width() / TEXT_WIDTH_DIVISOR)))

    def _stream_connected(self) -> Optional[bool]:
        """三态：``None`` 没打算拉流 / ``True`` 连着 / ``False`` 断了或还没连上。"""
        if not self._stream_planned:
            return None
        return bool(self._links.get("stream"))

    def _build_text_block(self, painter: QPainter, point: int) -> List[Tuple[str, QFont, QColor]]:
        """文本块的每一行（文字、字体、颜色）。排版与绘制共用这一份定义。"""
        big = QFont(self._family)
        big.setPointSize(point)
        small = QFont(self._family)
        small.setPointSize(max(MIN_TEXT_POINT - 2, int(point * 0.62)))

        lines: List[Tuple[str, QFont, QColor]] = []
        for row in text_lib.format_vai_lines(self._vai):
            # 第一行（含数值）用主字号，条与细节行小一号 —— 数值是主角
            lines.append((row, big if not lines else small,
                          QColor(255, 255, 255) if not lines else QColor(200, 200, 200)))
        lines.append(("", small, GREY))     # 空行 = 段落间隔
        lines.append((text_lib.format_state_line(self._state, self._reason),
                      big, QColor(255, 255, 255)))
        lines.append((text_lib.format_note(self._vai), small, GREY))
        if self._links:
            lines.append((text_lib.format_link_block(self._links), small, DIM))
        return lines

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.fillRect(self.rect(), Qt.black)

        point = self._text_point_size()

        # 1) 先量文本块要多高 —— 画面框拿剩下的空间，而不是写死比例。
        #    写死 55% 的话，字号一变（窗口小、或演示机 DPI 不同）就会重叠。
        block = self._build_text_block(painter, point)
        block_height = 0
        for text, font, _color in block:
            block_height += _line_height(QFontMetrics(font)) if text else SECTION_GAP

        block_top = max(self.height() // 2, self.height() - MARGIN - block_height)
        body_width = max(1, self.width() - 2 * MARGIN)

        # 2) 画面：有帧就画帧，没有就画占位
        box = QRectF(MARGIN, MARGIN, body_width, max(0, block_top - MARGIN - SECTION_GAP))
        if self._pixmap is not None and box.height() >= 8:
            self._paint_frame(painter, box)
        else:
            self._paint_placeholder(painter, box, point)

        # 3) 文本块
        self._paint_lines(painter, block, MARGIN, block_top)

        painter.end()

    def _paint_frame(self, painter: QPainter, box: QRectF) -> None:
        """把最新一帧按比例铺进框里，居中留黑边（letterbox）。

        **不拉伸填满**：A 的叠加层已经把两行识别结果烧进了画面（见
        ``module_a_vision/stream.py`` 的 ``overlay_lines``）。拉伸会让
        那些字跟着变形 —— 而叠加层恰恰是这一整块东西里唯一"必须能读"的部分。

        缩放用 ``SmoothPixmapTransform``：颜文字那边不怕糊，这里怕 ——
        画面缩小后脸上会出现锯齿，隔着投影仪看像是识别错了。
        """
        pixmap = self._pixmap
        if pixmap is None or pixmap.isNull():
            return
        if box.width() <= 0 or box.height() <= 0:
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(self._fit_rect(box, pixmap), pixmap, QRectF(pixmap.rect()))
        painter.setPen(DIM)
        painter.drawRect(box)

    @staticmethod
    def _fit_rect(box: QRectF, pixmap: QPixmap) -> QRectF:
        """等比缩放到框内并居中。宽高比一致时就是框本身。"""
        pw = max(1, pixmap.width())
        ph = max(1, pixmap.height())
        scale = min(box.width() / pw, box.height() / ph)
        width = pw * scale
        height = ph * scale
        return QRectF(box.left() + (box.width() - width) / 2.0,
                       box.top() + (box.height() - height) / 2.0,
                       width, height)

    def _paint_placeholder(self, painter: QPainter, box: QRectF, point: int) -> None:
        """带边框的空框 + 居中说明。没有画面时**必须**看得出是"没接入"，
        而不是"接到了但画面全黑"（后者会让人去查摄像头，其实是没起流）。"""
        if box.height() < 8:
            return
        painter.setPen(DIM)
        painter.drawRect(box)

        hint_font = QFont(self._family)
        hint_font.setPointSize(max(MIN_TEXT_POINT, int(point * 0.7)))
        painter.setFont(hint_font)
        painter.setPen(GREY)

        lines = text_lib.format_stream_text(self._stream_connected(), self._stream_url).split("\n")
        metrics = QFontMetrics(hint_font)
        total = len(lines) * metrics.height()
        y = box.center().y() - total / 2 + metrics.ascent()
        for line in lines:
            painter.drawText(QRectF(box.left(), y - metrics.ascent(), box.width(), metrics.height()),
                             int(Qt.AlignHCenter | Qt.AlignVCenter), line)
            y += metrics.height()

    def _paint_lines(self, painter: QPainter, block, left: int, top: int) -> None:
        y = top
        for text, font, color in block:
            if not text:
                y += SECTION_GAP
                continue
            metrics = QFontMetrics(font)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(left, y + metrics.ascent(), text)
            y += metrics.height()

    # ------------------------------------------------------------------

    def snapshot(self, width: int = 900, height: int = 600,
                 vai: Optional[dict] = None, state: Optional[str] = None,
                 jpeg: Optional[bytes] = None) -> QPixmap:
        """渲染成一帧位图（``--screenshot`` 与单元测试用）。

        与 ``FaceWindow.snapshot`` 同样的理由：截图时**直接赋值**，
        不走 ``set_*``（那里会因为"值没变"提前返回，连续抓两帧就失效了）。

        ``jpeg`` 给了就把它当作"当前这一帧"来画 —— 测试靠它把"解码 + 等比
        摆放"这段钉住，而不必真的起一个流。
        """
        if vai is not None:
            self._vai = vai
        if state is not None:
            self._state = state
        if jpeg is not None:
            pixmap = QPixmap()
            if pixmap.loadFromData(jpeg, "JPG"):
                self._pixmap = pixmap
        self.resize(width, height)
        return self.grab()
