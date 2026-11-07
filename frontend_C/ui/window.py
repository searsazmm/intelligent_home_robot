# -*- coding: utf-8 -*-
"""纯黑背景上的白色颜文字表情窗口。

需求（用户原话）：
    「窗口背景纯黑色；显示白色颜文字表情，根据 backend_B 发来的指令切换表情；
      只展示表情，不要输入框、聊天文字框。」

颜文字表见 :mod:`c_core.expressions`：四种状态各 4 个表情（共 16 个），
每个都带显式写死的眨眼帧。「根据 B 的状态切换表情」的职责划分不变：
B 发四态 → 表情只由四态决定；同状态内每 :data:`CYCLE_MS` 轮换一个变体。

所以这个窗口**只有一张脸**。没有输入框、没有聊天记录、没有按钮、没有状态文字。
连接状态只出现在**标题栏**（全屏时看不到，所以另有 ``--log-file``）和日志里；
``--debug-hud`` 可以在左上角叠一行小号灰字，默认关闭，只供联调。

两个实现选择值得说明：

1. **不用 QLabel，直接在 paintEvent 里画。**
   QLabel 的样式表背景继承很容易盖掉窗口的黑色底（要额外写
   ``background:transparent`` 去补）；而且颜文字的状态淡入、眨眼、呼吸、
   轮换交叉淡化是四路并行的连续动画，自绘才能让它们共用同一组透明度相位
   （``alpha = opacity × breath × cross``），与背景的状态色光晕同步浮现。

2. **字体在运行时探测，不硬编码。**
   颜文字大量使用全角/CJK 字形，开发机上 ``Microsoft YaHei`` / ``SimSun`` /
   ``SimHei`` **都不存在**，写死只会静默回退、颜文字变豆腐块。
   实测可用的是 ``Microsoft YaHei UI``（首选），HUD、``--dump-glyphs``
   字形自检窗口和右栏中文也都用它。
"""

from __future__ import annotations

import collections
import logging
import random
from typing import Optional

from PyQt5.QtCore import (
    QEasingCurve, QObject, QPropertyAnimation, QPointF, QRectF, QTimer, Qt,
    pyqtProperty, pyqtSignal,
)
from PyQt5.QtGui import (
    QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPixmap, QRadialGradient,
)
from PyQt5.QtWidgets import QMessageBox, QWidget

from c_core import theme
from c_core.expressions import (
    DEFAULT_STATE, STATE_ABSENT, STATE_NORMAL,
    Expression, expressions_for, normalize_state, state_label, all_faces,
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
BLINK_MS = 150
#: 两次眨眼的间隔区间（毫秒）。随机化，否则像节拍器。
#: 拉长到 3~7.5 秒：眨眼与更慢的表情变换协调，避免眨眼密集到像在抽搐。
BLINK_GAP_MS = (3000, 7500)
#: 同状态内轮换颜文字变体的间隔（毫秒）。表情主体由状态决定，
#: 每态另有 4 个变体（``c_core.expressions.EXPRESSIONS``），轮换避免死板。
#: 拉到 10 秒：老人盯着看时不会觉得脸在频繁换样。
CYCLE_MS = 10000
#: 状态切换时的淡入时长（毫秒）
FADE_MS = 220

# --------------------------------------------------------------------------
# 美化动效（2026-10 界面升级引入的三个参数）
#
# 共同的约束：**四角与顶边中点必须保持纯黑** —— 那是单元测试钉住的第一条
# （tests/test_window.py::TestBlackBackground）。所以氛围光晕的半径按
# ``min(宽, 高)`` 取，光到不了角落；淡入动画从 0 开始时，光晕和脸一起从
# 全黑里浮现（``snapshot_bypasses_the_fade`` 那条测试依赖这一点）。
# --------------------------------------------------------------------------

#: 呼吸动画周期（毫秒）。脸的透明度在 1.0 与 :data:`BREATH_MIN` 之间缓慢往复，
#: 让那张脸「活着」。幅度刻意压在 6% —— 老人隔两米看不该察觉它在动，
#: 只该觉得它不像一张贴图。周期拉到 4.2 秒，更平缓。
BREATH_MS = 4200
BREATH_MIN = 0.94

#: 同状态内轮换表情时的交叉淡化时长（毫秒）。表情换组不再硬切，
#: 而是先沉到 :data:`CROSS_DIP` 再浮回来，像一次缓慢的眨眼。
#: 拉到 380ms / 沉到 0.30：过渡更柔和，不再像快闪。
CROSS_MS = 380
CROSS_DIP = 0.30

#: 中心氛围光晕：半径比例（相对 ``min(宽, 高)``）与中心不透明度。
#: 0.40 保证在默认 900×600 下光晕到不了顶边中点（距离 295px > 240px）。
GLOW_RADIUS_RATIO = 0.40
GLOW_ALPHA = 0.16

#: 状态切换确认窗口大小：连续这么多次收到同一**新**状态才真正切换。
#: 2 = 过滤单次毛刺（B 端报文约 5Hz，2 次确认 ≈ 400ms 延迟）。
#: 设 1 等于关闭确认（旧行为）；设太大则状态切换明显滞后，老人表情对不上。
STATE_CONFIRM_NEED = 2

#: 通道名 → 中文。BackendLink 传的是 "chat"/"status"；"stream" 是画面通道
#: （A 侧 MJPEG，步骤② 才真的连上，但标签先在这里定好，
#: 免得 ``link_label`` 把它原样打成一行英文）。
#: ⚠️ ``c_core/display_text.py`` 里有一张**并行的**表（右栏文本块用），
#: 加新通道时两边都要加 —— 两张表的键必须一致。
LINK_LABELS = {"chat": "对话通道", "status": "状态通道", "stream": "画面通道"}

#: 颜文字左右两侧留出的空白（像素）。字号会被钳到"最宽的颜文字 + 这些留白
#: 刚好塞得进栏宽"为止。
#:
#: 为什么需要这一项：单脸窗口的宽度总是远大于高度，按高度取的字号横向够用；
#: 左右分栏之后左栏只剩一半宽，最长的那几个颜文字（``(∪.∪ )...zzz``）
#: 就会**横向顶出栏外**被右栏盖住 —— 而且只在个别表情上出现、切换时一闪而过，
#: 很容易被当成"偶尔的花屏"。
#:
#: ⚠️ 不写"字号 ÷ 系数"那种经验值 —— 字体宽度随字体、全角字形数量、DPI
#: 一起变，常量猜不准。由 :meth:`FaceWindow._largest_fitting_point_size`
#: 用 ``QFontMetrics`` 真量一次再按比例缩放，换字体也不用回来改。
EDGE_MARGIN = 16


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

    #: 状态 + 判定原因。**与上面的 stateChanged 并存，不是替代它。**
    #:
    #: 脸只需要状态（四态表决定表情），右栏的文本块却想显示"为什么"
    #: （「状态 疲惫（连续打哈欠 3 次）」）。让 stateChanged 多带一个参数
    #: 会改掉 ``FaceWindow.set_state`` 的签名 —— 那是本模块对外的公开接口，
    #: 而且它已经有一批用例按单参数在用。所以另开一条带原因的信号给右栏，
    #: 两条互不干扰。
    stateReasonChanged = pyqtSignal(str, str)   # 状态, 原因

    #: 专注度展示报文（B→C，api_doc §5.6）。**payload 是整个报文 dict。**
    #: 用 ``object`` 而不是 ``dict`` 作参数类型：PyQt5 的 ``pyqtSignal(dict)``
    #: 在跨线程排队时要把 dict 转成 QVariant，边界情况（None、嵌套）容易出岔子；
    #: ``object`` 是原样投递引用，而 dict 本身在 emit 之后不再被线程写。
    vaiArrived = pyqtSignal(object)

    #: 走神确认（B→C）。**是 vai 报文里 ``distracted`` 字段的专用转发**：
    #: B 端只在走神确认（VAI 持续低于走神门槛达到 ``distract_sustain``）
    #: 后才置真 —— 和主动开口说"您看起来走神了"同一本判定账。
    #:
    #: ⚠️ 这是对"vai 不许驱动表情"红线的一次**明示授权的例外**，边界收得很窄：
    #: vai 报文里混进的 ``state`` 字段仍然不许碰脸（防第二状态源），但走神
    #: 是用户点名要的表情变化 —— 人还坐在画面里（视觉状态仍是 normal），
    #: 老人抬头看到的却还是笑脸就不对了。仍走**独立信号**而不是让
    #: ``FaceWindow.set_vai`` 直接消费报文：脸不认识 vai 报文结构，
    #: 只认识一个布尔。断连时发 ``False``，脸回到状态机表情。
    distractedArrived = pyqtSignal(bool)

    #: 心率 / 呼吸展示报文（B→C）。同样不带 state 字段，同样只投右栏。
    vitalsArrived = pyqtSignal(object)

    #: 紧急事件报文（B→C，emergency/crisis）。payload 是整个报文 dict，
    #: 理由同 vaiArrived：object 跨线程原样投递，不经 QVariant 转换。
    emergencyArrived = pyqtSignal(object)


class FaceWindow(QWidget):
    """纯黑窗口上的一个白色颜文字，表情由 B 的四态驱动。"""

    def __init__(
        self,
        family: str,
        font_size: Optional[int] = None,
        debug_hud: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._family = family
        self._fixed_font_size = font_size       # None = 按窗口高度/宽度自适应
        self._debug_hud = debug_hud

        self._state = DEFAULT_STATE
        self._index = 0
        self._blinking = False
        self._distracted = False                # 走神叠加（见 set_distracted）
        self._opacity = 1.0
        self._breath = 1.0                      # 呼吸相位（faceBreath 属性驱动）
        self._cross = 1.0                       # 表情轮换的交叉淡化
        self._links: dict = {}                  # 通道名 → 是否已连接
        self._last_message = ""
        self._closing = False
        # 状态确认窗口：连续 STATE_CONFIRM_NEED 次收到同一**新**状态才切换。
        # 过滤 B 端单次状态毛刺（normal↔sad 抖一帧就回来的那种），
        # 让颜文字不会因为后端一次抖动就闪一下。B 端报文约 5Hz，
        # 2 次确认 ≈ 400ms 延迟，肉眼几乎无感。
        self._state_votes: collections.deque = collections.deque(maxlen=STATE_CONFIRM_NEED)

        self.setWindowTitle("居家陪伴机器人")
        # 黑底靠 paintEvent 自己刷，这里关掉自动擦除，避免首帧闪白
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.resize(900, 600)

        # ---- 淡入动画（状态切换时用） ----
        self._fade = QPropertyAnimation(self, b"faceOpacity", self)
        self._fade.setDuration(FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)

        # ---- 交叉淡化（同状态内轮换表情时用，见 _next_expression） ----
        self._cross_anim = QPropertyAnimation(self, b"faceCross", self)
        self._cross_anim.setDuration(CROSS_MS)
        self._cross_anim.setEasingCurve(QEasingCurve.OutCubic)

        # ---- 呼吸（常驻循环：1.0 → 0.94 → 1.0，正弦缓动，不硬回弹） ----
        self._breath_anim = QPropertyAnimation(self, b"faceBreath", self)
        self._breath_anim.setDuration(BREATH_MS)
        self._breath_anim.setLoopCount(-1)
        self._breath_anim.setKeyValueAt(0.0, 1.0)
        self._breath_anim.setKeyValueAt(0.5, BREATH_MIN)
        self._breath_anim.setKeyValueAt(1.0, 1.0)
        self._breath_anim.setEasingCurve(QEasingCurve.InOutSine)
        self._breath_anim.start()

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
        """切换状态。收到的字符串一律先规整（api_doc §4.3）。

        带**状态确认窗口**：连续 :data:`STATE_CONFIRM_NEED` 次收到同一
        新状态才真正切换，过滤后端单次状态毛刺（normal↔sad 抖一帧就回来
        的那种）。当前状态下的重复报文直接忽略（不进窗口），所以确认
        只对"新"状态生效——稳态下的常规心跳不会触发任何切换。
        """
        state = normalize_state(state)
        if state == self._state:
            # 当前状态的重复报文：清空待确认票，避免「normal→sad×1→normal×2」
            # 这种序列误判成需要切换。
            self._state_votes.clear()
            return
        # 状态确认窗口：攒够连续 N 票才切。
        self._state_votes.append(state)
        if len(self._state_votes) < STATE_CONFIRM_NEED:
            logger.debug("表情待确认：%s（%d/%d）", state,
                         len(self._state_votes), STATE_CONFIRM_NEED)
            return
        # 窗口已满，但全相同才放行；中间掺了别的状态就重来。
        if not all(v == state for v in self._state_votes):
            self._state_votes.clear()
            return
        logger.info("表情切换：%s → %s（已确认 %d 票）",
                    self._state, state, len(self._state_votes))
        self._state_votes.clear()
        self._apply_state(state)

    def set_distracted(self, distracted: bool) -> None:
        """走神叠加（B 端确认后才为真，见 ``StateBridge.distractedArrived``）。

        **不改 ``_state``、不改标题、不改氛围光** —— 视觉状态机说的是
        "人在不在画面里、情绪如何"，走神是 VAI 层的另一件事，两者叠加
        而不是替换。只影响 :meth:`_faces` 取哪组表情：

        - ``state=normal`` + 走神 → 用「走神/无人」组的表情
          （用户点名要求：走神时左边的小表情要变）
        - ``state=tired/sad`` 时**不叠加** —— 疲惫、低落各有自己的表情，
          比"走神"更贴近老人当下，状态机优先
        - 走神消失（B 端 ``distracted=False`` 或断连）→ 回状态机表情

        与 ``set_state`` 一样做去重：重复值直接忽略，避免心跳重发让脸闪。
        """
        distracted = bool(distracted)
        if distracted == self._distracted:
            return
        self._distracted = distracted
        self._index = 0
        # 换表情组的淡入节奏与换状态一致（沉下去再浮回来），
        # 但不碰 _state / 标题 / 氛围光 —— 那些只归状态机管。
        self._cross_anim.stop()
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self.update()

    def _faces(self):
        """当前该用的表情组。走神叠加发生在这里（唯一的叠加点）。"""
        if self._distracted and self._state == STATE_NORMAL:
            return expressions_for(STATE_ABSENT)
        return expressions_for(self._state)

    def _apply_state(self, state: str) -> None:
        """真正把状态写下去并触发淡入。``set_state`` 确认通过后调用，
        也供测试与 :meth:`snapshot` 绕过确认窗口直接切换。

        同一状态不重启动画（与旧 ``set_state`` 的去重语义一致）——
        15 秒心跳重发同一状态时不该让脸闪一下。
        """
        if state == self._state:
            return
        self._state = state
        self._index = 0
        self._blinking = False
        # 换状态时把交叉淡化归位：状态本身有淡入，两个动画相乘会过度变暗
        self._cross_anim.stop()
        self._cross = 1.0
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

    def show_emergency(self, msg: dict) -> None:
        """B 报来紧急事件（socket 线程 → emergencyArrived 信号 → 本槽）。

        弹窗必须在 GUI 线程做：信号在 main.py 里连到这个绑定方法，
        Qt 以排队方式投递，所以到这里时已经在 GUI 线程。
        ``msg`` 为 ``{"kind": "emergency"|"crisis", "text": 原话, ...}``。
        """
        msg = msg or {}
        kind = str(msg.get("kind") or "")
        text = str(msg.get("text") or "").strip()
        if kind == "crisis":
            title = "情绪危机提醒"
            head = "老人说了危险的话，请立刻电话联系并陪伴"
        else:
            title = "紧急求助"
            head = "老人发出紧急求助，请立刻电话联系或回家查看"
        body = head + ("\n原话：" + text if text else "")
        # 模态警示框：需要家属/操作者明确点掉，避免一闪而过被忽略。
        QMessageBox.warning(self, title, body)

    def snapshot(self, state: str, width: int = 900, height: int = 600,
                 blinking: bool = False) -> QPixmap:
        """把某个状态渲染成一帧位图（``--screenshot`` 与单元测试用）。

        两个必须做的动作，否则抓到的是**纯黑**的一帧：

        1. **停掉并跳过淡入动画。** 状态切换会把 ``faceOpacity`` 置 0 再动画到 1，
           而 ``QPropertyAnimation`` 需要事件循环来推进 —— 在 ``grab()`` 这种
           同步渲染里根本没有事件循环，于是永远停在 0，脸是全透明的。
        2. **不经 ``set_state()`` 直接赋值。** ``set_state`` 会因为「状态没变」
           提前返回，截图时想连续抓两帧同一状态就失效了。

        美化动效（呼吸 / 交叉淡化）同理：**全部停掉并归位到 1.0** ——
        ``--screenshot`` 导出的 PPT 插图必须是全亮、可复现的一帧。
        """
        self._fade.stop()
        self._breath_anim.stop()
        self._cross_anim.stop()
        self._state_votes.clear()            # 截图直接赋值，绕过确认窗口
        self._apply_state(normalize_state(state))
        # snapshot 专属归位：blinking 参数要生效，opacity/breath/cross 全亮
        self._blinking = bool(blinking)
        self._opacity = 1.0
        self._breath = 1.0
        self._cross = 1.0
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

    def _get_breath(self) -> float:
        return self._breath

    def _set_breath(self, value: float) -> None:
        self._breath = max(0.0, min(1.0, float(value)))
        self.update()

    faceBreath = pyqtProperty(float, fget=_get_breath, fset=_set_breath)

    def _get_cross(self) -> float:
        return self._cross

    def _set_cross(self, value: float) -> None:
        self._cross = max(0.0, min(1.0, float(value)))
        self.update()

    faceCross = pyqtProperty(float, fget=_get_cross, fset=_set_cross)

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
        faces = self._faces()
        if len(faces) > 1:
            self._index = (self._index + 1) % len(faces)
        # 轮换不硬切：先沉一下再浮回来（像一次缓慢的眨眼）。
        # 只动 _cross，不碰 _opacity —— 状态淡入与轮换是两件事。
        self._cross_anim.stop()
        self._cross_anim.setStartValue(CROSS_DIP)
        self._cross_anim.setEndValue(1.0)
        self._cross_anim.start()
        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def _current_expression(self) -> Expression:
        """当前要画的那一个颜文字（按轮换序号取，越界自动回绕）。"""
        faces = self._faces()
        return faces[self._index % len(faces)]

    def _font_point_size(self) -> int:
        """算出字号。

        颜文字的「视觉大小」取决于窗口高度而不是宽度，所以按高度取比例。
        0.16 是实测试出来的：全屏 1080p 下约 170pt，三四米外也看得清。

        **宽度那一项是左右分栏之后加的**：分栏把这张脸挤到半宽，宽表情就会
        横向溢出。两者取小 —— 只要栏够宽，尺寸还是由高度决定，即分栏对排版
        的影响仅限"窄到会溢出时"。

        ``--font-size`` 显式指定时**不做任何钳制**：那是明确要求，溢出是选择。
        """
        if self._fixed_font_size:
            return self._fixed_font_size
        by_height = int(self.height() * 0.16)
        by_width = self._largest_fitting_point_size()
        if by_width is None:             # 量不出宽度就别钳制
            return max(24, by_height)
        return max(24, min(by_height, by_width))

    def _largest_fitting_point_size(self) -> Optional[int]:
        """用**字体实际度量**算出"最宽的颜文字 + 两侧留白刚好放得下"的字号。

        量不出来时返回 ``None``（**不是**某个猜测值）：字体库为空的平台上
        度量全是 0，那时钳制只会平白把字号按一个假上限压住。

        在 100pt 上量一次再按比例缩放 —— 字形宽度与字号成正比，一次测量就够，
        不用二分。取 ``all_faces()``（四态全部，含眨眼帧）里最宽的那个：
        脸会轮换、会换组，只要有一个放不下，它就会在某个时刻溢出。
        """
        available = self.width() - 2 * EDGE_MARGIN
        if available <= 0:
            return None
        probe = 100                      # 探测字号：宽度与字号成正比，量一次即可
        metrics = QFontMetrics(build_font(self._family, probe))
        needed = max(metrics.horizontalAdvance(face) for face in all_faces())
        if needed <= 0:
            return None
        return int(probe * available / needed)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # 1) 纯黑底。用 fillRect 而不是 stylesheet —— 不受系统主题影响。
        painter.fillRect(self.rect(), Qt.black)

        # 2) 中心氛围光晕（状态主题色）。三处克制：
        #    - 半径 = min(宽,高) × 比例，光到不了四角与顶边中点（背景测试钉住）；
        #    - 中心不透明度只有 0.16，是"黑里的颜色"而不是"彩底"；
        #    - 乘上 _opacity：淡入时光晕和脸一起浮现，半途抓帧仍是全黑。
        alpha = self._opacity * self._breath * self._cross
        accent = QColor(*theme.hex_to_rgb(theme.state_accent(self._state)))
        if self._opacity > 0.01:
            radius = min(self.width(), self.height()) * GLOW_RADIUS_RATIO
            if radius >= 1.0:
                core = QColor(accent)
                core.setAlphaF(GLOW_ALPHA * self._opacity)
                edge = QColor(accent)
                edge.setAlphaF(0.0)
                grad = QRadialGradient(
                    QPointF(self.width() / 2.0, self.height() / 2.0), radius)
                grad.setColorAt(0.0, core)
                grad.setColorAt(1.0, edge)
                painter.fillRect(self.rect(), grad)

        # 3) 白色颜文字。表情只由四态驱动（api_doc §4.2），眨眼帧瞬切，
        #    同状态内 4 个变体轮换。整体透明度沿用 淡入 × 呼吸 × 轮换 的
        #    连乘相位，与光晕同步浮现。
        #    垂直居中按「整行高度」而不是字形包围盒，否则 (￣▽￣) 和
        #    (╥﹏╥) 会因字形高度不同而上下跳。dy 是状态切换的入场：淡入时
        #    从下方轻微上浮（全亮归零；snapshot() 直接置 1.0，截图在最终位置）。
        if alpha > 0.01:
            expression = self._current_expression()
            text = expression.blink if self._blinking else expression.face
            font = build_font(self._family, self._font_point_size())
            painter.save()
            painter.setFont(font)
            painter.setOpacity(alpha)
            painter.setPen(QColor(255, 255, 255))
            metrics = QFontMetrics(font)
            baseline = (self.height() + metrics.ascent() - metrics.descent()) / 2
            baseline += (1.0 - self._opacity) * self.height() * 0.02
            rect = QRectF(0, baseline - metrics.ascent(),
                          self.width(), metrics.height())
            painter.drawText(rect, int(Qt.AlignHCenter | Qt.AlignVCenter), text)
            painter.restore()

        # 4) 调试用 HUD（默认关闭）
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
        for anim in (self._fade, self._breath_anim, self._cross_anim):
            anim.stop()
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
