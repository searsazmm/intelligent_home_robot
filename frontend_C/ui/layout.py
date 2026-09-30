# -*- coding: utf-8 -*-
"""左右 1:1 分栏：左栏小表情（``FaceWindow``）+ 右栏画面与识别结果（``CameraPane``）。

--------------------------------------------------------------------------
这个类只做三件事，且刻意只做三件事
--------------------------------------------------------------------------
1. 用 ``QHBoxLayout`` 把两栏按 1:1 摆好，中间一条细分隔线；
2. 把 ``set_state`` / ``set_link`` / ``note_message`` **原样转发**给两栏；
3. 额外转发 ``set_vai`` 给右栏（左栏那张脸**不认**专注度 —— 见下）。

不做布局以外的任何逻辑：没有定时器、不碰 socket、不改文案。
分栏是个纯排版问题，把它和状态机混在一起之后，"答辩时想退回单脸窗口"
就不再是一个开关能解决的事了（那个开关是 ``--no-camera``）。

--------------------------------------------------------------------------
为什么左栏那张脸**不**跟着专注度变
--------------------------------------------------------------------------
表情只有四种状态（api_doc §4.2），由 8001/8002 的 ``state`` 报文驱动。
专注度是**另一栏数字**。一旦让脸也跟着 VAI 变，就等于给 B 加了第二条
状态通道 —— 两边不一致时（VAI 说专注、状态说疲惫）脸该听谁的？
那个问题没有好答案，所以从一开始就不让它存在：``vai`` 报文里
连 ``state`` 字段都没有（见 ``backend_B/core/ui_channel.py``）。
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QFrame, QHBoxLayout, QWidget

from ui.camera import CameraPane
from ui.window import FaceWindow

#: 分隔线颜色。两栏都是纯黑底，没有这条线就看不出哪里是分界。
DIVIDER_COLOR = "#3a3a3a"


class SplitWindow(QWidget):
    """左脸 + 右画面的主窗口。"""

    def __init__(
        self,
        face: FaceWindow,
        camera: CameraPane,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._face = face
        self._camera = camera

        divider = QFrame(self)
        divider.setFrameShape(QFrame.VLine)
        divider.setFixedWidth(1)
        divider.setStyleSheet(f"background-color: {DIVIDER_COLOR}; border: none;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # 1:1 —— 用户原话是「界面左右对半分」。给 face 和 camera 相同的 stretch，
        # 分隔线固定 1px 不参与拉伸。
        layout.addWidget(face, 1)
        layout.addWidget(divider, 0)
        layout.addWidget(camera, 1)

        # 标题栏由窗口持有（子控件 setWindowTitle 在嵌入后是不生效的），
        # 所以脸的 _refresh_title 那一套在分栏模式下看不到 —— 通道状态因此
        # 也显示在右栏的文本块里（见 display_text.format_link_block）。
        self.setWindowTitle("居家陪伴机器人")
        self.resize(1440, 600)

    # ------------------------------------------------------------------

    @property
    def face(self) -> FaceWindow:
        """给 ``--demo`` 与测试用：演示模式直接往脸上打状态。"""
        return self._face

    @property
    def camera(self) -> CameraPane:
        return self._camera

    # ------------------------------------------------------------------
    # 转发（与 FaceWindow 的同名方法语义一致）
    # ------------------------------------------------------------------

    def set_state(self, state: str, reason: str = "") -> None:
        """切状态。脸只吃状态字符串；右栏多要一个原因是给"状态 疲惫（连续打哈欠）"用的。"""
        self._face.set_state(state)
        self._camera.set_state(state, reason)

    def set_link(self, source: str, connected: bool) -> None:
        self._face.set_link(source, connected)
        self._camera.set_link(source, connected)

    def set_vai(self, message: Optional[dict]) -> None:
        self._camera.set_vai(message)

    def note_message(self, text: str) -> None:
        """只进 HUD，不上屏 —— 右栏**不是**聊天记录区（需求原话：只展示表情）。"""
        self._face.note_message(text)
