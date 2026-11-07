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
``StateBridge`` 的 ``vaiArrived`` / ``vitalsArrived`` / ``stateReasonChanged`` /
``linkChanged`` 信号排队回 GUI 线程，或经 ``FrameSlot`` 这一道单槽，
再由本类在 GUI 线程里取走。绝不能在 socket 线程里直接调 ``set_vai()``
或 ``pull_frame()``。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap,
)
from PyQt5.QtWidgets import QWidget

from c_core import display_text as text_lib
from c_core import theme
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

# ---- 颜色（色值全部来自 c_core/theme.py —— 与左栏状态光同一套主题） ----
GREY = QColor(*theme.hex_to_rgb(theme.TEXT_MUTED))
DIM = QColor(*theme.hex_to_rgb(theme.TEXT_FAINT))
BODY = QColor(*theme.hex_to_rgb(theme.TEXT_PRIMARY))
SOFT = QColor(*theme.hex_to_rgb(theme.TEXT_SECONDARY))
ACCENT_VAI = QColor(*theme.hex_to_rgb(theme.ACCENT_VAI))
TRACK = QColor(*theme.hex_to_rgb(theme.TRACK))
TRACK_BORDER = QColor(*theme.hex_to_rgb(theme.TRACK_BORDER))
PANEL_BORDER = QColor(*theme.hex_to_rgb(theme.PANEL_BORDER))
LIVE_OK = QColor(*theme.hex_to_rgb(theme.LIVE_OK))
LIVE_BAD = QColor(*theme.hex_to_rgb(theme.LIVE_BAD))
BG_TOP = QColor(*theme.hex_to_rgb(theme.BG_TOP))


def _accent_qcolor(state: object) -> QColor:
    """状态 → 主题色 QColor（胶囊、光晕等处共用，与左栏同源）。"""
    return QColor(*theme.hex_to_rgb(theme.state_accent(state)))


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
        self._vitals: Optional[dict] = None
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

    def set_vitals(self, message: Optional[dict]) -> None:
        """收到 B 的体征展示报文（``{"type":"vitals", ...}``）。

        与 ``set_vai`` 同一条红线：**不驱动表情** —— 报文不带 ``state``
        字段，心率呼吸只是右栏的另一行数字。

        ``None`` 是合法输入 —— 表示"从来没收到过"，界面显示「未提供」，
        与"收到了但 hr/rr 都是 None"（A 明确说没检出）分开显示。
        """
        self._vitals = message if isinstance(message, dict) else None
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

    def _index_value(self) -> Optional[float]:
        """``vai.index`` → 0–100 的 float；没有分数或值非法 → ``None``。

        与 ``display_text.format_index`` 同一条兜底脾气：类型乱来的字段按
        "没有分数"处理，绝不让 GUI 线程为脏数据抛异常。进度条的实心段
        长度与数值行的颜色都由它驱动。
        """
        if not isinstance(self._vai, dict):
            return None
        try:
            return max(0.0, min(100.0, float(self._vai.get("index"))))
        except (TypeError, ValueError):
            return None

    def _fonts(self, point: int) -> Tuple[QFont, QFont, QFont]:
        """右栏的三档字体：区块标签（小）/ 数值（主角，加粗大字）/ 细节（小）。

        排版层级是这次美化最重要的改动：原来整块只有两档字号、全部挤在一起，
        现在数值独占一档并加粗 —— 隔着两三米，第一眼看到的就是它。
        """
        label = QFont(self._family)
        label.setPointSize(max(10, int(point * 0.55)))
        number = QFont(self._family)
        number.setPointSize(max(20, min(40, int(point * 1.55))))
        number.setBold(True)
        small = QFont(self._family)
        small.setPointSize(max(MIN_TEXT_POINT - 2, int(point * 0.62)))
        return label, number, small

    def _block_rows(self, point: int) -> List[dict]:
        """文本块的每一行（种类、内容、字体、颜色、占高）。

        测量与绘制共用这一份定义（原来叫 ``_build_text_block``）——
        先量总高把画面框的空间让出来，再逐行画，两步永远说同一套话。

        文字内容仍然**全部**来自 :mod:`c_core.display_text` 的纯函数，
        这里只负责排版与上色；文案一个字都不写死在本文件。
        """
        label_f, number_f, small_f = self._fonts(point)
        label_m = QFontMetrics(label_f)
        number_m = QFontMetrics(number_f)
        small_m = QFontMetrics(small_f)
        value = self._index_value()
        vai = self._vai if isinstance(self._vai, dict) else None

        rows: List[dict] = []
        # ① 区块标签：青绿短竖条 + 小字（给整块一个视觉锚点）
        rows.append({"kind": "label", "text": text_lib.VAI_LABEL,
                     "font": label_f, "color": GREY,
                     "height": label_m.height() + 6})
        # ② 数值（主角）：有分数 → 白色大字；没有 → 灰色破折号（**不是 0**）
        rows.append({"kind": "number",
                     "text": text_lib.format_index(vai.get("index") if vai else None),
                     "font": number_f,
                     "color": BODY if value is not None else GREY,
                     "height": number_m.height() + 2})
        # ③ 图形化进度条（取代 ■□ 字符条，见 _paint_bar）
        bar_h = max(10, min(14, int(self.width() * 0.014)))
        rows.append({"kind": "bar", "value": value, "height": bar_h + 10})
        # ④ 细节行（状态 / 模态 / 已测时长；没分数时是"为什么没分数"）
        rows.append({"kind": "text", "text": text_lib.format_vai_detail(self._vai),
                     "font": small_f, "color": GREY, "height": small_m.height()})
        # ⑤ 状态胶囊：底色与左栏氛围光是同一个状态主题色（c_core.theme）。
        #    走神叠加：B 在 vai 报文里明确说走神、而视觉状态还是 normal 时，
        #    在原因括号里补「走神中」，并把胶囊色调切到走神紫 —— 用户要的
        #    就是这个可见变化：人还坐在画面里（所以 state 仍是 normal，
        #    协议四态不动），但状态行要能看出"走神了"。
        reason = self._reason
        accent = _accent_qcolor(self._state)
        if vai and vai.get("distracted") and self._state == "normal":
            reason = ("走神中" if not reason else f"走神中，{reason}")
            accent = _accent_qcolor("absent")
        rows.append({"kind": "gap", "height": SECTION_GAP})
        rows.append({"kind": "pill",
                     "text": text_lib.format_state_line(self._state, reason),
                     "font": small_f,
                     "accent": accent,
                     "height": small_m.height() + 12})
        # ⑤' 心率 / 呼吸：与专注度同构的 label + 数值 + 细节三行，插在
        #     状态胶囊下面。同样**不驱动表情**（vitals 报文不带 state 字段，
        #     见 set_vitals）；hr/rr 都没有时数值是灰色破折号，**不是 0**。
        vitals = self._vitals if isinstance(self._vitals, dict) else None
        has_vitals = vitals is not None and (
            vitals.get("hr") is not None or vitals.get("rr") is not None
        )
        rows.append({"kind": "gap", "height": SECTION_GAP})
        rows.append({"kind": "label", "text": text_lib.VITALS_LABEL,
                     "font": label_f, "color": GREY,
                     "height": label_m.height() + 6})
        rows.append({"kind": "number",
                     "text": text_lib.format_vitals_values(self._vitals),
                     "font": number_f,
                     "color": BODY if has_vitals else GREY,
                     "height": number_m.height() + 2})
        rows.append({"kind": "text", "text": text_lib.format_vitals_detail(self._vitals),
                     "font": small_f, "color": GREY, "height": small_m.height()})
        # ⑥ 免责声明（一个字不许改写 —— 兜底值也来自 display_text）
        rows.append({"kind": "gap", "height": SECTION_GAP})
        rows.append({"kind": "text", "text": text_lib.format_note(self._vai),
                     "font": small_f, "color": DIM, "height": small_m.height()})
        # ⑦ 通道行：彩色圆点 + 文字
        if self._links:
            rows.append({"kind": "gap", "height": max(6, SECTION_GAP // 2)})
            rows.append({"kind": "links", "font": small_f,
                         "height": small_m.height()})
        return rows

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # 背景：纯黑里加一层极轻的纵向渐变 —— 是"黑里的层次"，不是灰底。
        # 顶部 #0a0a0c 与左栏的状态光晕同为低亮度，两栏摆在一起不突兀。
        bg = QLinearGradient(0.0, 0.0, 0.0, float(self.height()))
        bg.setColorAt(0.0, BG_TOP)
        bg.setColorAt(1.0, QColor(0, 0, 0))
        painter.fillRect(self.rect(), bg)

        point = self._text_point_size()

        # 1) 先量文本块要多高 —— 画面框拿剩下的空间，而不是写死比例。
        #    写死 55% 的话，字号一变（窗口小、或演示机 DPI 不同）就会重叠。
        rows = self._block_rows(point)
        block_height = sum(row["height"] for row in rows)
        block_top = max(self.height() // 2, self.height() - MARGIN - block_height)
        available = self.height() - MARGIN - block_top

        # 空间不足时的优雅降级，两级：
        #   ① 压段间距与装饰 padding（_compact_rows）；
        #   ② 仍不够 → 缩字号重排一次（行高随字号缩，条与间距是定值，
        #      所以重排后再补一轮 ① 收余量）。内容行一个不丢 ——
        #      挤可以，免责声明 / 通道状态不能被挤出屏幕。
        overflow = block_height - available
        if overflow > 0:
            self._compact_rows(rows, overflow)
            block_height = sum(row["height"] for row in rows)
            if block_height > available:
                scaled = max(MIN_TEXT_POINT, int(point * available / block_height))
                if scaled < point:
                    rows = self._block_rows(scaled)
                    block_height = sum(row["height"] for row in rows)
                    self._compact_rows(rows, max(0.0, block_height - available))
                    block_height = sum(row["height"] for row in rows)
            block_top = max(self.height() // 2, self.height() - MARGIN - block_height)

        body_width = max(1, self.width() - 2 * MARGIN)

        # 2) 画面：有帧就画帧，没有就画占位
        box = QRectF(MARGIN, MARGIN, body_width, max(0, block_top - MARGIN - SECTION_GAP))
        if self._pixmap is not None and box.height() >= 8:
            self._paint_frame(painter, box, point)
        else:
            self._paint_placeholder(painter, box, point)

        # 3) 文本块
        self._paint_rows(painter, rows, body_width, block_top)

        painter.end()

    @staticmethod
    def _compact_rows(rows: List[dict], overflow: float) -> None:
        """空间不足时按"先装饰后内容"的顺序压缩行高（就地修改 rows）。

        压缩顺序：段间距（每段保底 2px）→ 标签 / 胶囊 / 进度条 / 数值行的
        装饰性 padding（各让 8px）。数值、细节、免责声明、通道这些**内容
        行的字体与文字一律不动** —— 挤可以，丢不行。``rows`` 同时供测量
        与绘制使用，就地改完两边自动一致。
        """
        remaining = float(overflow)
        for row in rows:
            if remaining <= 0:
                break
            if row["kind"] == "gap" and row["height"] > 2:
                cut = min(row["height"] - 2, remaining)
                row["height"] -= cut
                remaining -= cut
        for row in rows:
            if remaining <= 0:
                break
            if row["kind"] in ("label", "pill", "bar", "number"):
                cut = min(8.0 if row["kind"] != "number" else 2.0, remaining)
                row["height"] -= cut
                remaining -= cut

    def _paint_rows(self, painter: QPainter, rows: List[dict],
                    body_width: int, top: int) -> None:
        y = float(top)
        for row in rows:
            kind = row["kind"]
            if kind == "gap":
                y += row["height"]
            elif kind == "label":
                m = QFontMetrics(row["font"])
                painter.setPen(Qt.NoPen)
                painter.setBrush(ACCENT_VAI)
                painter.drawRoundedRect(QRectF(MARGIN, y + 1, 3, m.height() - 2),
                                        1.5, 1.5)
                painter.setFont(row["font"])
                painter.setPen(row["color"])
                painter.drawText(QRectF(MARGIN + 11, y, body_width - 11, m.height()),
                                 int(Qt.AlignVCenter), row["text"])
                y += row["height"]
            elif kind in ("text", "number"):
                m = QFontMetrics(row["font"])
                painter.setFont(row["font"])
                painter.setPen(row["color"])
                painter.drawText(QRectF(MARGIN, y, body_width, m.height()),
                                 int(Qt.AlignVCenter), row["text"])
                y += row["height"]
            elif kind == "bar":
                self._paint_bar(painter, MARGIN, y + 5, body_width,
                                row["height"] - 10, row["value"])
                y += row["height"]
            elif kind == "pill":
                self._paint_pill(painter, y, row, body_width)
                y += row["height"]
            elif kind == "links":
                self._paint_links(painter, y, row["font"])
                y += row["height"]

    def _paint_bar(self, painter: QPainter, x: float, y: float,
                   width: float, height: float, value: Optional[float]) -> None:
        """0–100 的图形化进度条。

        取代 ``■□`` 字符条的两个理由：字符条的对齐与粗细取决于字体，
        U+25A0 缺字形时会静默变豆腐块；圆角矩形 + 渐变在任何平台画出来
        都一模一样，也更适合投影仪。

        ``value is None``（没有分数）时**整条轨道全空** —— 空条加破折号，
        和"低分"在视觉上不会混。这条语义与 ``display_text.format_index_bar``
        一致；``value == 0`` 同样不画实心段（0 分就是空条，不是一个圆头）。
        """
        if width <= 0 or height <= 0:
            return
        radius = height / 2.0
        painter.setPen(QPen(TRACK_BORDER, 1.0))
        painter.setBrush(TRACK)
        painter.drawRoundedRect(QRectF(x, y, width, height), radius, radius)
        if value is None or value <= 0:
            return
        filled = max(height, width * (value / 100.0))   # 至少留一个圆头
        fill = QRectF(x, y, min(filled, width), height)
        grad = QLinearGradient(fill.topLeft(), fill.topRight())
        grad.setColorAt(0.0, ACCENT_VAI)
        grad.setColorAt(1.0, LIVE_OK)
        painter.setPen(Qt.NoPen)
        painter.setBrush(grad)
        painter.drawRoundedRect(fill, radius, radius)

    def _paint_pill(self, painter: QPainter, y: float, row: dict,
                    body_width: int) -> None:
        """状态胶囊：状态主题色的圆角底 + 白字。

        底色与左栏那张脸背后的氛围光是**同一个颜色**（都来自
        ``c_core.theme.STATE_ACCENTS``）—— 老人扫一眼右栏这个色块，
        就知道左栏的表情为什么变了。两栏由同一条状态驱动，颜色也必须同源。
        """
        font = row["font"]
        accent: QColor = row["accent"]
        m = QFontMetrics(font)
        h = float(row["height"])
        w = min(float(body_width), m.horizontalAdvance(row["text"]) + 24.0)
        radius = h / 2.0
        bg = QColor(accent)
        bg.setAlphaF(0.16)
        border = QColor(accent)
        border.setAlphaF(0.55)
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(bg)
        painter.drawRoundedRect(QRectF(MARGIN, y, w, h), radius, radius)
        painter.setFont(font)
        painter.setPen(BODY)
        painter.drawText(QRectF(MARGIN + 12, y, w - 24, h),
                         int(Qt.AlignVCenter), row["text"])

    def _paint_links(self, painter: QPainter, y: float, font: QFont) -> None:
        """通道行：彩色圆点 + 文字（文字仍来自 ``display_text.format_link_line``）。

        圆点用颜色编码健康度 —— 绿 = 通，红 = 断。原来纯靠「通/断」两个字，
        投影仪上一旦糊了就什么信息都没有；颜色加文字是冗余编码，丢一路
        还剩一路。顺序沿用 ``format_link_block`` 的固定排序，不让它跳位置。
        """
        m = QFontMetrics(font)
        painter.setFont(font)
        order = ("chat", "status", "stream")
        sources = ([s for s in order if s in self._links]
                   + sorted(s for s in self._links if s not in order))
        x = float(MARGIN)
        cy = y + m.height() / 2.0
        for source in sources:
            connected = bool(self._links[source])
            painter.setPen(Qt.NoPen)
            painter.setBrush(LIVE_OK if connected else LIVE_BAD)
            painter.drawEllipse(QRectF(x, cy - 3.0, 6.0, 6.0))
            x += 12.0
            text = text_lib.format_link_line(source, connected)
            painter.setPen(DIM)
            painter.drawText(QPointF(x, y + m.ascent()), text)
            x += m.horizontalAdvance(text) + 18.0

    def _paint_frame(self, painter: QPainter, box: QRectF, point: int) -> None:
        """把最新一帧按比例铺进框里，居中留黑边（letterbox）。

        **不拉伸填满**：A 的叠加层已经把两行识别结果烧进了画面（见
        ``module_a_vision/stream.py`` 的 ``overlay_lines``）。拉伸会让
        那些字跟着变形 —— 而叠加层恰恰是这一整块东西里唯一"必须能读"的部分。

        缩放用 ``SmoothPixmapTransform``：颜文字那边不怕糊，这里怕 ——
        画面缩小后脸上会出现锯齿，隔着投影仪看像是识别错了。

        美化升级：圆角裁切 + 一圈极轻的白色描边，让框在黑底上"存在"
        但不抢画面的戏；右上角另有一枚「实时画面」徽标（见下）。
        """
        pixmap = self._pixmap
        if pixmap is None or pixmap.isNull():
            return
        if box.width() <= 0 or box.height() <= 0:
            return
        radius = min(14.0, box.width() / 6.0, box.height() / 6.0)
        path = QPainterPath()
        path.addRoundedRect(box, radius, radius)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.save()
        painter.setClipPath(path)
        painter.drawPixmap(self._fit_rect(box, pixmap), pixmap, QRectF(pixmap.rect()))
        painter.restore()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 26), 1.0))
        painter.drawRoundedRect(box, radius, radius)
        self._paint_live_badge(painter, box, point)

    def _paint_live_badge(self, painter: QPainter, box: QRectF, point: int) -> None:
        """画面框右上角的「实时画面」徽标（半透明黑底 + 绿点）。

        放**右上**而不是左上：A 烧进画面的 ASCII 叠加层就在左上角，
        徽标叠上去会把这块东西里唯一"必须能读"的部分盖掉。
        框太小时整枚省略 —— 装饰不能挤占内容。"""
        if box.width() < 160 or box.height() < 60:
            return
        font = QFont(self._family)
        font.setPointSize(max(9, int(point * 0.45)))
        m = QFontMetrics(font)
        text = text_lib.STREAM_LIVE_BADGE
        h = m.height() + 8.0
        w = m.horizontalAdvance(text) + 30.0
        left = box.right() - 10.0 - w
        top = box.top() + 10.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(QRectF(left, top, w, h), h / 2.0, h / 2.0)
        painter.setBrush(LIVE_OK)
        painter.drawEllipse(QRectF(left + 12.0, top + h / 2.0 - 3.0, 6.0, 6.0))
        painter.setFont(font)
        painter.setPen(SOFT)
        painter.drawText(QRectF(left + 24.0, top, w - 28.0, h),
                         int(Qt.AlignVCenter), text)

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
        """虚线圆角框 + 简笔画摄像机 + 居中说明。

        没有画面时**必须**看得出是"没接入"，而不是"接到了但画面全黑"
        （后者会让人去查摄像头，其实是没起流）。"""
        if box.height() < 8:
            return
        painter.setPen(QPen(PANEL_BORDER, 1.5, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        radius = min(14.0, box.width() / 6.0, box.height() / 6.0)
        painter.drawRoundedRect(box, radius, radius)

        hint_font = QFont(self._family)
        hint_font.setPointSize(max(MIN_TEXT_POINT, int(point * 0.7)))
        painter.setFont(hint_font)
        painter.setPen(GREY)
        metrics = QFontMetrics(hint_font)
        lines = text_lib.format_stream_text(self._stream_connected(), self._stream_url).split("\n")
        text_h = len(lines) * metrics.height()

        # 简笔画图标：框太小就整个省略，说明文字永远要保住
        icon = int(min(64.0, box.width() * 0.16, box.height() * 0.22))
        if box.height() < 90:
            icon = 0
        gap = 12.0 if icon else 0.0
        total = icon + gap + text_h
        y = box.center().y() - total / 2.0 + metrics.ascent()
        if icon:
            self._paint_camera_icon(painter, box.center().x(), y - metrics.ascent() + icon / 2.0, icon)
            y += icon + gap
        for line in lines:
            painter.drawText(QRectF(box.left(), y - metrics.ascent(), box.width(), metrics.height()),
                             int(Qt.AlignHCenter | Qt.AlignVCenter), line)
            y += metrics.height()

    def _paint_camera_icon(self, painter: QPainter, cx: float, cy: float,
                           size: int) -> None:
        """简笔画摄像机：机身圆角矩形 + 镜头圆 + 顶部取景凸起。

        纯形状、**零字形** —— 占位图标恰恰是字体缺失时最先露馅的地方，
        所以它一个字符都不用，任何平台画出来都一样。"""
        body_w = float(size)
        body_h = size * 0.68
        body = QRectF(cx - body_w / 2.0, cy - body_h / 2.0, body_w, body_h)
        painter.setPen(QPen(GREY, 2.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(body, size * 0.14, size * 0.14)
        r = size * 0.16
        painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        notch_w = size * 0.34
        notch_h = size * 0.10
        painter.drawRoundedRect(
            QRectF(cx - notch_w / 2.0, body.top() - notch_h - 2.0, notch_w, notch_h + 4.0),
            2.0, 2.0)

    # ------------------------------------------------------------------

    def snapshot(self, width: int = 900, height: int = 600,
                 vai: Optional[dict] = None, state: Optional[str] = None,
                 jpeg: Optional[bytes] = None,
                 vitals: Optional[dict] = None) -> QPixmap:
        """渲染成一帧位图（``--screenshot`` 与单元测试用）。

        与 ``FaceWindow.snapshot`` 同样的理由：截图时**直接赋值**，
        不走 ``set_*``（那里会因为"值没变"提前返回，连续抓两帧就失效了）。

        ``jpeg`` 给了就把它当作"当前这一帧"来画 —— 测试靠它把"解码 + 等比
        摆放"这段钉住，而不必真的起一个流。
        """
        if vai is not None:
            self._vai = vai
        if vitals is not None:
            self._vitals = vitals
        if state is not None:
            self._state = state
        if jpeg is not None:
            pixmap = QPixmap()
            if pixmap.loadFromData(jpeg, "JPG"):
                self._pixmap = pixmap
        self.resize(width, height)
        return self.grab()
