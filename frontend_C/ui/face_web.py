# -*- coding: utf-8 -*-
"""BoopFace 表情窗口 —— QtWebEngine 内嵌本地离线 HTML 的第一版（Phase 1）。

需求（用户原话）：
    「背景是黑色，小表情为白色」；「normal → idle、tired → sleep、
    failure（sad）→ 垂眼垂头（关切）、absent → bored」；
    「用 runJavaScript 通信即可，不用 QWebChannel」；
    「关闭鼠标指针跟随、关闭自动睡觉，保留 idle 自动眨眼和眼神游移」。

表情库是 boopface v0.8.0（MIT，纯 SVG Web Component，零运行时依赖），
lite 构建（``boopface-lite.js``，无 skins）放在
``intelligent_home_robot/static/family/js/boopface/``。

--------------------------------------------------------------------------
为什么用自定义 URL scheme，而不是 file://
--------------------------------------------------------------------------
``boopface-lite.js`` 不是单文件 bundle —— 它只有 1.7KB，是带**相对路径
ES module 引用**（``./src/boopface-extension-host.js``）的入口壳，
整个 ``src/`` 目录都要随行。Chromium 对 ``file://`` 页面里的 ES module
加载默认按 CORS 拦（origin 是 null），能不能加载取决于版本与设置 ——
靠它就是靠运气。

所以这里注册自定义协议 ``boopface://``，配一个
:class:`_FaceResourceHandler` 从本地磁盘回源：

    boopface://local/face.html                          → 宿主页（face_web.html）
    boopface://local/static/family/js/boopface/...      → boopface 资源目录

同源之内 module 加载畅通，且**绝不触网**（scheme 处理器只读本地文件）。
附带的好处是 exe 打包路径问题一并解决：资源根按
:func:`_candidate_roots` 解析，冻结环境（PyInstaller）下指向
``sys._MEIPASS``。

PyInstaller 打包时需要把两样东西带进去（数据文件的相对布局要与
:func:`_candidate_roots` 的解析一致）::

    --add-data "intelligent_home_robot/static;static"
    --add-data "intelligent_home_robot/frontend_C/ui/face_web.html;ui"

--------------------------------------------------------------------------
QtWebEngine 的两条硬约束（main.py 已处理，这里记录原因）
--------------------------------------------------------------------------
1. ``Qt.AA_ShareOpenGLContexts`` 必须在 QApplication 构造**之前**设置，
   否则 QtWebEngine 首个 view 构造时打警告甚至崩。
2. ``PyQt5.QtWebEngineWidgets`` 必须在 QApplication 构造**之前** import。
   所以 main.py 在建 app 前就 ``from ui import face_web``（本模块），
   顺带用 ImportError 作为「WebEngine 不可用」的判定信号。

--------------------------------------------------------------------------
对外契约：镜像 :class:`ui.window.FaceWindow`
--------------------------------------------------------------------------
main.py / SplitWindow 只认这几个方法（鸭子类型），新组件必须齐全：

- ``set_state(state)``     四态切换（含与 FaceWindow 相同的 2 票确认窗口）
- ``set_distracted(bool)`` Phase 1 **不接受**（见方法 docstring）
- ``set_link(s, ok)``      仅记日志（黑底网页上没有标题栏/HUD 可画）
- ``note_message(text)``   仅记日志（同上）
- ``show_emergency(msg)``  与 FaceWindow 相同的模态弹窗 —— 危机提示是
                           安全路径，web 表情模式下**绝不能**悄悄消失

``--dump-glyphs`` / ``--screenshot`` 两条一次性任务继续用裸 FaceWindow
（main.py 原样未动）：它们导出的是颜文字的确定性位图，与 web 表情无关。
"""

from __future__ import annotations

import collections
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

from PyQt5.QtCore import (
    QByteArray, QBuffer, QEvent, QEventLoop, QIODevice, QTimer, QUrl, Qt,
    pyqtSignal,
)
from PyQt5.QtGui import QColor
from PyQt5.QtWebEngineCore import (
    QWebEngineUrlRequestJob, QWebEngineUrlScheme, QWebEngineUrlSchemeHandler,
)
from PyQt5.QtWebEngineWidgets import QWebEngineProfile, QWebEngineView
from PyQt5.QtWidgets import QMessageBox

from c_core.expressions import (
    STATE_ABSENT, STATE_NORMAL, STATE_SAD, STATE_TIRED, normalize_state,
)

logger = logging.getLogger(__name__)

#: 自定义协议名。宿主页与 boopface 资源都从它下面取（见模块 docstring）。
SCHEME = b"boopface"

#: 状态确认窗口大小 —— 与 ``ui.window.STATE_CONFIRM_NEED`` 同值同理由：
#: B 端报文约 5Hz，2 票 ≈ 400ms，过滤单次状态毛刺。**两处必须同步改。**
STATE_CONFIRM_NEED = 2

#: ``wait_ready`` 的默认等待上限。首次加载要跑起 Chromium + 整个 module
#: 链，冷启 1~3 秒；给到 10 秒兜住慢盘 / 杀毒扫描，超时即回落。
FACE_READY_TIMEOUT_MS = 10_000

#: 四态 → boopface 动作。**Phase 1 的全部映射就在这张表里**
#: （api_doc §4.2 的四种状态，禁止自造；boopface 动作名见其 README）：
#: - normal → ``idle``   自动眨眼 + 眼神游移（用户点名保留的两个动效）
#: - sad    → ``failure`` 头与眼下垂 —— 即「垂眼垂头（关切）」
#: - tired  → ``sleep``  闭眼睡（页面已关 auto-sleep，睡到我们叫醒为止）
#: - absent → ``bored``  缓慢左右扫视
STATE_ACTIONS = {
    STATE_NORMAL: "idle",
    STATE_SAD: "failure",
    STATE_TIRED: "sleep",
    STATE_ABSENT: "bored",
}

#: scheme 处理器认识的扩展名 → MIME。**刻意手写而不用 mimetypes**：
#: Windows 注册表会污染 mimetypes 的结果（.js 可能被猜成 text/plain），
#: 而 ES module 加载对 MIME 是硬校验 —— 必须是 ``text/javascript``。
_MIME_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".map": "application/json",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".md": "text/plain",
}


# --------------------------------------------------------------------------
# 纯逻辑（不依赖 Qt 实例，可脱离 GUI 单测）
# --------------------------------------------------------------------------

def state_confirmed(
    current: str, votes: Sequence[str], need: int = STATE_CONFIRM_NEED,
) -> Optional[str]:
    """状态确认窗口的纯函数版：攒够 :arg:`need` 张**相同**票才放行。

    与 ``ui.window.FaceWindow.set_state`` 里的内联逻辑同语义（两处保持
    一致是刻意的 —— 颜文字与 web 表情必须对同一条状态流给出同样的反应）：

    - 票数不足 → ``None``（继续等）
    - 票里掺了别的状态 → ``None``（清票重来，由调用方负责）
    - 全相同 → 返回该状态
    """
    if len(votes) < need:
        return None
    first = votes[0]
    if any(v != first for v in votes):
        return None
    return first


# --------------------------------------------------------------------------
# 资源路径（开发环境与 PyInstaller 冻结环境通吃）
# --------------------------------------------------------------------------

def _candidate_roots() -> list:
    """资源根的候选目录，按优先级排列。

    - PyInstaller 冻结：``sys._MEIPASS``（--add-data 解包后的临时目录，
      布局 ``static/...`` 与 ``ui/face_web.html``，见模块 docstring）；
    - 开发环境：本文件向上两级 = 仓库包根（含 ``static/``）；
    - 兜底：本文件所在目录（html 与 face_web.py 同目录）。
    """
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    roots.append(Path(__file__).resolve().parents[2])
    roots.append(Path(__file__).resolve().parent)
    return roots


def html_path() -> Path:
    """宿主页 face_web.html 的绝对路径。找不到时**返回路径而不抛异常**
    —— 加载失败由 :meth:`BoopFaceWindow.wait_ready` 探针兜住并回落。"""
    for root in _candidate_roots():
        candidate = root / "face_web.html"
        if candidate.is_file():
            return candidate
        if (root / "ui" / "face_web.html").is_file():
            return root / "ui" / "face_web.html"
    # 都不在（打包漏了 --add-data）：返回源码旁路径，加载必然失败 → 回落
    return Path(__file__).resolve().parent / "face_web.html"


def static_root() -> Path:
    """boopface 资源所属的 ``static/`` 目录绝对路径。判定标准是真实的
    ``boopface-lite.js`` 存在（不是目录存在 —— 空目录会造成更难懂的报错）。"""
    for root in _candidate_roots():
        candidate = root / "static"
        if (candidate / "family" / "js" / "boopface" / "boopface-lite.js").is_file():
            return candidate
    return Path(__file__).resolve().parents[2] / "static"


# --------------------------------------------------------------------------
# 自定义 scheme：boopface:// → 本地文件
# --------------------------------------------------------------------------

_scheme_registered = False
_scheme_handler = None          # 模块级持有，防止被垃圾回收（Qt 不持 Python 引用）


def register_face_scheme() -> None:
    """注册 ``boopface://``。**必须在 QApplication 构造之前调用**
    （Qt 要求：scheme 要在 WebEngine 使用前注册好）。幂等。"""
    global _scheme_registered
    if _scheme_registered:
        return
    scheme = QWebEngineUrlScheme(SCHEME)
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    scheme.setFlags(
        QWebEngineUrlScheme.SecureScheme
        | QWebEngineUrlScheme.LocalScheme
        | QWebEngineUrlScheme.LocalAccessAllowed
        | QWebEngineUrlScheme.CorsEnabled
    )
    QWebEngineUrlScheme.registerScheme(scheme)
    _scheme_registered = True
    logger.debug("已注册自定义协议 %s://", SCHEME.decode())


def resolve_resource_path(path: str, root: Path, html: Path) -> Optional[Path]:
    """把 ``boopface://`` 的 URL path 解析为本地文件；越界 / 不存在 → ``None``。

    纯函数（不碰 Qt 实例），handler 与单元测试共用同一份实现。只放行两类
    路径：``/face.html``（宿主页）与 ``/static/...``（资源根内）；
    ``..`` 穿越与绝对路径注入都挡在前缀校验上。
    """
    if path == "/face.html":
        return html if html.is_file() else None
    prefix = "/static/"
    if not path.startswith(prefix):
        return None
    candidate = (root / path[len(prefix):]).resolve()
    # 越界校验：解析后的真实路径必须仍在资源根内（大小写不敏感的 Windows）
    root_str = str(root)
    candidate_str = str(candidate)
    if os.path.normcase(candidate_str).startswith(os.path.normcase(root_str + os.sep)):
        return candidate if candidate.is_file() else None
    return None


class _FaceResourceHandler(QWebEngineUrlSchemeHandler):
    """把 ``boopface://local/...`` 回源成本地文件，路径越界一律 404。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root = static_root().resolve()
        self._html = html_path().resolve()

    def _resolve(self, path: str) -> Optional[Path]:
        return resolve_resource_path(path, self._root, self._html)

    def requestStarted(self, job) -> None:  # noqa: N802（Qt 命名）
        path = job.requestUrl().path()
        local = self._resolve(path)
        if local is None:
            logger.debug("boopface:// 404：%s", path)
            job.fail(QWebEngineUrlRequestJob.UrlNotFound)
            return
        try:
            data = local.read_bytes()
        except OSError as exc:
            logger.error("读取表情资源失败 %s：%s", local, exc)
            job.fail(QWebEngineUrlRequestJob.UrlNotFound)
            return
        mime = _MIME_TYPES.get(local.suffix.lower(), "application/octet-stream")
        # QBuffer 的生命周期：reply 是异步消费的，buffer 必须活到读完 ——
        # 挂到 job 上（job 销毁时一起回收）。
        buf = QBuffer()
        buf.setParent(job)
        buf.setData(data)
        buf.open(QIODevice.ReadOnly)
        job.reply(QByteArray(mime.encode("ascii")), buf)
        logger.debug("boopface:// 200 %s（%d 字节）", path, len(data))


def install_scheme_handler() -> None:
    """把资源处理器挂到默认 profile 上。幂等；在 QApplication 之后调用。"""
    global _scheme_handler
    if _scheme_handler is not None:
        return
    _scheme_handler = _FaceResourceHandler()
    QWebEngineProfile.defaultProfile().installUrlSchemeHandler(SCHEME, _scheme_handler)


# --------------------------------------------------------------------------
# 窗口
# --------------------------------------------------------------------------

class BoopFaceWindow(QWebEngineView):
    """黑底白脸的 BoopFace 窗口，四态经 ``runJavaScript`` 驱动。"""

    #: 页面加载失败（探测超时 / 资源缺失）。``--face=web`` 强制模式下
    #: main.py 据此报错退出；auto 模式下 main.py 在建窗口**之前**就用
    #: :meth:`wait_ready` 同步探活，所以运行期这个信号只做日志。
    loadFailed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        install_scheme_handler()

        self._state = STATE_NORMAL
        self._votes: deque = collections.deque(maxlen=STATE_CONFIRM_NEED)
        self._ready = False
        self._hooked = []               # 已装键盘过滤的 focusProxy（防重复装）

        self.setWindowTitle("居家陪伴机器人")
        self.resize(900, 600)
        # 页面底色必须是黑的：boopface 自己只画脸，宿主 CSS 负责黑背景，
        # 而这行兜住「页面加载前一瞬间的默认白底闪光」。
        self.page().setBackgroundColor(QColor("#000000"))

        self.loadFinished.connect(self._on_load_finished)
        self.load(QUrl(f"{SCHEME.decode()}://local/face.html"))

    # ------------------------------------------------------------------
    # 对外接口（镜像 FaceWindow，main.py / SplitWindow 靠鸭子类型调用）
    # ------------------------------------------------------------------

    def set_state(self, state: str) -> None:
        """切换状态。规整 + 2 票确认，与 ``FaceWindow.set_state`` 同语义。"""
        state = normalize_state(state)
        if state == self._state:
            # 当前状态的重复报文：清票，避免「normal→sad×1→normal×2」误切换
            self._votes.clear()
            return
        self._votes.append(state)
        confirmed = state_confirmed(self._state, self._votes)
        if confirmed is None:
            logger.debug("表情待确认：%s（%d/%d）", state,
                         len(self._votes), STATE_CONFIRM_NEED)
            return
        self._votes.clear()
        logger.info("表情切换（boopface）：%s → %s（动作 %s）",
                    self._state, confirmed, STATE_ACTIONS[confirmed])
        self._state = confirmed
        self._push_state()

    def set_distracted(self, distracted: bool) -> None:
        """走神叠加 —— **Phase 1 不接受**，只记日志。

        用户规格原话：「先只实现第一阶段，情绪意图联动（第二阶段）暂时
        不做」。走神表情属于第二阶段；这里保持 no-op 保证接口齐全
        （main.py 的 ``distractedArrived`` 信号连得上、不崩），第二阶段
        只需要改这一个方法。
        """
        logger.debug("boopface 忽略走神叠加（Phase 2 未实现）：%s", distracted)

    def set_link(self, source: str, connected: bool) -> None:
        """通道状态：web 表情是纯黑一张脸，没有标题栏/HUD 可画，仅记日志。"""
        logger.debug("boopface 通道 %s：%s", source, "已连接" if connected else "已断开")

    def note_message(self, text: str) -> None:
        """回复/关怀文本：同上，黑底网页不显示文字，仅记日志。"""
        logger.debug("boopface 收到文本（不上屏）：%s", text[:80])

    def show_emergency(self, msg: dict) -> None:
        """紧急事件弹窗 —— 与 ``FaceWindow.show_emergency`` 相同的实现。

        这是安全路径：危机/紧急报文在任何表情模式下都必须**模态弹窗**
        提醒家属/操作者，绝不能因为换了渲染方式就静默降级成一条日志。
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
        QMessageBox.warning(self, title, body)

    def wait_ready(self, timeout_ms: int = FACE_READY_TIMEOUT_MS) -> bool:
        """同步等待页面就绪（``window.__face.ready === true``）。

        轮询 ``runJavaScript`` 直到 boopface 完成 custom element 注册，
        或超时。启动期（窗口还没 show）用嵌套事件循环是安全的 —— 这正是
        main.py 在 auto 模式下决定「用 web 表情还是回落颜文字」的依据。
        """
        if self._ready:
            return True
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if self._probe_once(wait_ms=800):
                logger.info("boopface 页面已就绪（耗时 %.1f 秒）",
                            timeout_ms / 1000.0 - (deadline - time.monotonic()))
                return True
        logger.error("boopface 页面在 %.1f 秒内未就绪", timeout_ms / 1000.0)
        return False

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _probe_once(self, wait_ms: int) -> bool:
        """探测一次页面就绪标志。超时 / 回调未到都算未就绪。"""
        loop = QEventLoop()
        holder = {}

        def done(value) -> None:
            holder["v"] = bool(value)
            if loop.isRunning():
                loop.quit()

        QTimer.singleShot(wait_ms, loop.quit)
        self.page().runJavaScript("!!(window.__face && window.__face.ready)", done)
        loop.exec_()
        self._ready = bool(holder.get("v"))
        return self._ready

    def _push_state(self) -> None:
        """把当前状态经 runJavaScript 推给页面（页面未就绪时跳过 ——
        就绪探针通过后 main.py 会补一次 :meth:`set_state`）。"""
        if not self._ready:
            logger.debug("页面未就绪，暂缓推送状态 %s", self._state)
            return
        action = STATE_ACTIONS.get(self._state, "idle")
        # json.dumps 负责引号转义 —— action 来自白名单表，这里只是把规矩立死
        self.page().runJavaScript(
            f"window.__face.setState({json.dumps(action)}); undefined"
        )

    def _on_load_finished(self, ok: bool) -> None:
        self._install_key_hook()
        if not ok:
            logger.error("boopface 宿主页加载失败（检查 %s 与 static 资源）", html_path())
            self.loadFailed.emit(str(html_path()))

    # ---- ESC 退全屏 / Q 退出：QWebEngineView 的键盘事件落在内部渲染子控件
    # 上，重写 keyPressEvent 收不到 —— 必须对 focusProxy 挂事件过滤器。
    def _install_key_hook(self) -> None:
        proxy = self.focusProxy()
        if proxy is not None and proxy not in self._hooked:
            proxy.installEventFilter(self)
            self._hooked.append(proxy)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802（Qt 命名）
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Escape:
                if self.isFullScreen():
                    self.showNormal()
                else:
                    self.close()
                return True
            if key == Qt.Key_Q:
                self.close()
                return True
        return super().eventFilter(obj, event)


def web_engine_available() -> bool:
    """QtWebEngine 绑定是否可导入（main.py 在建 app 前已实际试过，这是给
    测试与诊断用的显式布尔）。"""
    try:
        import PyQt5.QtWebEngineWidgets  # noqa: F401
        return True
    except ImportError:
        return False
