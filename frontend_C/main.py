# -*- coding: utf-8 -*-
"""模块 C（前端界面）入口。

界面默认**左右对半分**：左栏是小表情（``FaceWindow``），右栏是画面与识别结果
（``CameraPane``）。右栏的画面来自 A 侧那条 MJPEG 展示流，**要显式给**
``--stream-url`` 才拉；不给就是带边框的占位框（识别结果那一半照常显示真实数据）。
``--no-camera`` 退回原来的单脸窗口。

启动顺序按 api_doc §6.2：**先 A → 再 B → 最后 C**。
不过 C 不依赖 A，甚至不强依赖 B —— 两个都没起来时它会安静地退避重连，
界面上照常显示 normal 表情与"画面未连接（正在重连）"，谁一起来就自动接上。
所以启动顺序反了不会坏。

    python main.py                     # 默认：左右分栏，连 127.0.0.1:8002（不拉画面）
    python main.py --stream-url http://127.0.0.1:8010/stream.mjpeg
                                       # 再加上右栏的实时画面（A 要给 --stream）
    python main.py --no-camera         # 只要那张脸（不分栏）
    python main.py --fullscreen        # 全屏无边框（答辩用）
    python main.py --demo              # 不连 B，四态循环（B 起不来时的兜底）
    python main.py --dump-glyphs       # 字形自检：看不到方块即为正常
    python main.py --screenshot out/   # 导出四态 + 右栏 PNG（做 PPT 用），然后退出

⚠️ **不要 import cv2**。cv2 自带一套 Qt5 插件，和 PyQt5 的插件同处一个进程时
会互相顶掉对方的平台插件，症状是「Qt platform plugin could be initialized」
这种看不懂的崩。requirements.txt 里只放 PyQt5 也是这个原因。
右栏的画面走 MJPEG，解码用 PyQt5 自带的 JPEG 插件，
**同样不需要 cv2** —— 这条约束不会因为多了个画面就松动。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

#: 只取一个常数给 ``--camera-fps`` 当默认值。**故意放在模块级**：
#: ``build_parser()`` 跑在 ``_import_qt()`` 之前，而 ``c_core.mjpeg_client``
#: 是纯标准库的（零 Qt），在这里 import 它不会把 PyQt5 提前拉进来。
from c_core.mjpeg_client import DEFAULT_CAMERA_FPS

#: 让 Windows 控制台/管道都能打印中文。本机默认 cp936，
#: 输出被重定向成管道时打中文会直接 UnicodeEncodeError 把程序带崩。
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

logger = logging.getLogger("frontend_C")

EXIT_OK = 0
EXIT_QT_MISSING = 2

#: 四态循环的间隔（`--demo` 用），秒
DEMO_INTERVAL = 6.0

#: ``--screenshot`` 导出右栏用的**样例**专注度报文（不连 B）。
#: 数值故意取中间档（不是 100、也不是 None），这样进度条两头都留了空格 ——
#: 一眼就能看出格子的对齐对不对。note 逐字照抄 B 侧的 VAI_NOTE。
#:
#: 每个值都要是 B **真会发出来的**那种。``modality_config_id`` 这里写
#: "完整模态"而不是自己拼一个"凝视+头姿+睁眼"：三路齐全时 B 发的就是
#: ``core/focus.py`` 的 ``_MODALITY_CONFIG_MAP`` 里那个名字，编一个更好看的
#: 会让这张截图（答辩要用）和真机对不上。
_SAMPLE_VAI = {
    "type": "vai",
    "index": 63.5,
    "index_status": "研究趋势（非认知专注）",
    "status": "有效",
    "reason": "校准已锁定",
    "modalities": ["gaze", "pose", "eye_open"],
    "recent_modalities": ["gaze", "pose", "eye_open"],
    "modality_config_id": "完整模态",
    "valid_seconds": 42.0,
    "note": "研究趋势（非认知专注）；日常参考，非医疗结论",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frontend_C",
        description="居家陪伴机器人 · 前端界面（纯黑底 + 白色颜文字）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "启动顺序：先 backend_A，再 backend_B，最后 frontend_C（api_doc §6.2）。\n"
            "B 没起来也能跑：轮询重连，期间显示 normal 表情。\n"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="模块 B 的地址（默认 127.0.0.1）")
    parser.add_argument("--chat-port", type=int, default=8002,
                        help="模块 B 的对话通道端口（默认 8002，api_doc §5）")
    parser.add_argument("--status-port", type=int, default=8001,
                        help="模块 B 的状态通道端口（默认 8001，api_doc §4）")
    parser.add_argument("--also-status", action="store_true",
                        help="额外连 8001 状态通道（默认只连 8002，单一状态源更稳）")

    parser.add_argument("--font-size", type=int, default=None,
                        help="颜文字字号（磅）。默认按窗口高度自适应")
    parser.add_argument("--fullscreen", action="store_true",
                        help="无边框全屏。ESC 退出全屏，Q 退出程序")
    parser.add_argument("--debug-hud", action="store_true",
                        help="左上角叠一行调试灰字（默认关，只供联调）")
    parser.add_argument("--log-file", default=None,
                        help="同时写日志到文件（全屏时看不到控制台，建议加上）")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    parser.add_argument("--stream-url", default=None, metavar="URL",
                        help="A 侧展示流的地址（如 http://127.0.0.1:8010/stream.mjpeg）。"
                             "默认不拉流，右栏显示占位")
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS,
                        help=f"右栏取帧频率（默认 {DEFAULT_CAMERA_FPS}）。画面不受 B 影响，"
                             "--demo 下也照拉")

    parser.add_argument("--demo", action="store_true",
                        help="离线演示：不连 B，四态循环切换（答辩现场 B 起不来时的兜底）")
    parser.add_argument("--no-camera", action="store_true",
                        help="退回单脸窗口（不分栏）。分栏是默认；这个开关是兜底："
                             "投影仪分辨率奇葩、或临时只想给观众看那张脸时用")

    mode = parser.add_argument_group("一次性任务（做完就退出）")
    mode.add_argument("--dump-glyphs", action="store_true",
                      help="字形自检窗口：把所有颜文字平铺出来，看不到方块即为正常")
    mode.add_argument("--screenshot", metavar="DIR",
                      help="把四种状态各导出成一张 PNG 到 DIR，然后退出。"
                           "另附一张 pane.png（右栏排版与中文/进度条字符的自检）")

    parser.add_argument("--version", action="version", version="frontend_C 1.0")
    return parser


def setup_logging(level: str, log_file: str | None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if level == "DEBUG" else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info("日志同时写入 %s", os.path.abspath(log_file))


def _import_qt():
    """导入 PyQt5，失败时给一条能照做的中文提示。

    这台开发机出现过「``import PyQt5`` 成功但 ``PyQt5.QtWidgets`` 不存在」的
    残缺安装（PyQt5 目录下只有 Qt5/ 和 sip，没有绑定），所以提示里特地写了
    验证命令 —— 用 ``import PyQt5`` 验证是不作数的。
    """
    try:
        from PyQt5.QtCore import Qt  # noqa: F401
        from PyQt5.QtWidgets import QApplication  # noqa: F401
        return True
    except ImportError as exc:
        print(
            f"\n无法导入 PyQt5：{exc}\n\n"
            "本模块的界面依赖 PyQt5，请先安装：\n"
            "    pip install --force-reinstall PyQt5\n\n"
            "装完请用下面这条**验证**（不要用 `import PyQt5` —— 残缺安装时它会假成功）：\n"
            "    python -c \"from PyQt5.QtWidgets import QApplication; print('OK')\"\n",
            file=sys.stderr,
        )
        return False


def _run_screenshots(window, dump_window, out_dir: str, camera=None) -> int:
    """导出四态 PNG。做完就退出，不需要事件循环。

    ``camera`` 给了就再导一张右栏（``pane.png``）：那个栏位里出现了
    **新的字符类别** —— 中文正文、进度条的 ``■ □``、破折号。
    ``--dump-glyphs`` 只检查颜文字，管不到它们；而字体回退让缺字形
    静默变方块，所以答辩前必须有一张能对着看的图。

    内容是**写死的样例**（不连 B、不取实时数据）：这一张的用途是核对
    排版与字形，掺进实时数据就不可复现了。
    """
    from c_core.expressions import VALID_STATES

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for state in VALID_STATES:
        for blinking in (False, True):
            pixmap = window.snapshot(state, blinking=blinking)
            suffix = "_blink" if blinking else ""
            path = os.path.abspath(os.path.join(out_dir, f"face_{state}{suffix}.png"))
            if not pixmap.save(path, "PNG"):
                logger.error("写入失败：%s", path)
                return 1
            written.append(path)
            logger.info("已导出 %s", path)

    glyph_path = os.path.abspath(os.path.join(out_dir, "glyphs.png"))
    if dump_window.snapshot().save(glyph_path, "PNG"):
        written.append(glyph_path)
        logger.info("已导出 %s", glyph_path)

    if camera is not None:
        pane_path = os.path.abspath(os.path.join(out_dir, "pane.png"))
        pane = camera.snapshot(vai=_SAMPLE_VAI, state="tired")
        if pane.save(pane_path, "PNG"):
            written.append(pane_path)
            logger.info("已导出 %s", pane_path)

    print(f"\n共导出 {len(written)} 张到 {os.path.abspath(out_dir)}")
    print("请**肉眼**确认：背景纯黑、颜文字是白色且没有任何方块（豆腐块）；")
    print("pane.png 里的中文、进度条方块（■ □）与破折号也都要能看清。")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.log_file)

    if not _import_qt():
        return EXIT_QT_MISSING

    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtWidgets import QApplication

    # ⚠️ 高 DPI 属性必须在 QApplication 构造**之前**设置。之后再设是静默无效的
    # —— Windows 150% 缩放下会得到一张糊掉的颜文字，而且没有任何报错。
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv[:1])

    # Ctrl+C 能杀掉这个程序。Qt 的事件循环会挡住 Python 的信号处理，
    # 所以留一个空转定时器让解释器有机会跑信号处理器。
    # 答辩时卡在一个全屏黑窗口里、Ctrl+C 又没反应，是不可恢复的。
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _signal_timer = QTimer()
    _signal_timer.start(200)
    _signal_timer.timeout.connect(lambda: None)

    from c_core.expressions import DEFAULT_STATE, VALID_STATES, state_label
    from c_core.mjpeg_client import FrameSlot
    from ui.window import FaceWindow, GlyphDumpWindow, StateBridge, pick_font_family

    family = pick_font_family()

    # ---- 一次性任务：字形自检 / 截图 ----
    # ⚠️ 这两条路径**继续用裸 FaceWindow**（不进分栏）。
    # 它们是给 PPT 出确定性图片的：混进实时画面、或让排版随分栏变化，
    # 输出就不再可复现，而"每次导出的图都一样"正是它们存在的意义。
    if args.dump_glyphs or args.screenshot:
        from ui.camera import CameraPane
        window = FaceWindow(family, font_size=args.font_size, debug_hud=args.debug_hud)
        dump_window = GlyphDumpWindow(family)
        if args.screenshot:
            return _run_screenshots(window, dump_window, args.screenshot,
                                    camera=CameraPane(family))
        window.show()
        dump_window.show()
        # 两个窗口并排，免得字形表盖住主窗口
        window.move(0, 0)
        dump_window.move(940, 0)
        logger.info("字形自检已打开：逐个核对，没有方块即为正常")
        return app.exec_()

    # ---- 信号桥：socket 线程 → GUI 线程 ----
    # 只连接「绑定方法」（window / camera 的方法），Qt 才知道接收者在 GUI 线程，
    # 从而用 QueuedConnection 投递。连 lambda 会退化成 direct connection，
    # 于是在 socket 线程里直接画控件 —— Qt 最经典的崩溃来源。
    #
    # 它建在主窗口**之前**：画面通道要用它（那条线程的 on_link 只 emit 不碰控件），
    # 而 channel 是随窗口一起进来的，没法先建窗口再回头补。
    bridge = StateBridge()

    # ---- 主窗口：默认左右分栏，--no-camera 退回单脸 ----
    camera = None
    stream_link = None
    face = FaceWindow(family, font_size=args.font_size, debug_hud=args.debug_hud)
    if args.no_camera:
        window = face
        logger.info("单脸模式（--no-camera）：只显示表情，不显示画面栏")
        if args.stream_url:
            logger.info("--no-camera 已生效，--stream-url 忽略")
    else:
        from ui.camera import CameraPane
        from ui.layout import SplitWindow
        # 有地址才建单槽 —— 没给 --stream-url 时连定时器都不会起，
        # 右栏就是一个纯占位框（这条路径与步骤①完全一致）。
        slot = FrameSlot() if args.stream_url else None
        camera = CameraPane(family, stream_url=args.stream_url, stream_slot=slot,
                            camera_fps=args.camera_fps)
        window = SplitWindow(face, camera)
        logger.info("左右分栏：左=表情，右=画面 / 识别结果")

    bridge.stateChanged.connect(window.set_state)
    bridge.linkChanged.connect(window.set_link)
    bridge.replyArrived.connect(window.note_message)
    bridge.proactiveArrived.connect(window.note_message)
    if camera is not None:
        # 右栏的状态文字要带判定原因。两条信号都会打到 camera 上：
        # stateChanged 先发（只带状态），stateReasonChanged 后发（带原因），
        # Qt 的排队投递在同一接收线程里是先进先出，所以最终留下的是带原因的那条。
        bridge.stateReasonChanged.connect(camera.set_state)
        bridge.vaiArrived.connect(camera.set_vai)

    # ---- 画面通道：A 侧 MJPEG（真的给 --stream-url 了才拉）----
    # ⚠️ 顺序：**先接信号，再起线程**。反过来的话，第一帧连接状态可能在
    # 连接建立之前就发出去了，界面会一直停在"未连接"直到下一次重连。
    if camera is not None and slot is not None:
        stream_link = _build_stream_link(args, slot, bridge)
        stream_link.start()
        logger.info("正在连接画面通道 %s（%d fps）；A 未启动时会自动重连",
                    args.stream_url, camera.fps)

    link = None
    if args.demo:
        logger.info("离线演示模式：不连接模块 B，四态循环切换（间隔 %.0f 秒）", DEMO_INTERVAL)
        _install_demo_cycle(window, VALID_STATES, app)
    else:
        link = _build_link(args, bridge)
        link.start()
        logger.info(
            "正在连接模块 B（%s:%d%s）；模块 B 未启动时会自动重连，界面不受影响",
            args.host, args.chat_port,
            f" + 状态通道 {args.status_port}" if args.also_status else "",
        )

    if args.fullscreen:
        window.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        window.showFullScreen()
        logger.info("已进入全屏：ESC 退出全屏，Q 退出程序")
    else:
        window.show()

    window.set_state(DEFAULT_STATE)   # 初始就是 normal（api_doc §4.3）
    logger.info("字体=%s 字号=%s", family, args.font_size or "自适应")

    try:
        code = app.exec_()
    finally:
        # 先停连接再让进程退出：socket 线程都是 daemon，但 shutdown(SHUT_RDWR)
        # 才能立刻唤醒阻塞在 recv 上的线程，避免退出时卡住或留下半开的连接。
        if camera is not None:
            camera.stop_stream()   # 定时器先停：免得它去取一个正在被关掉的东西
        if stream_link is not None:
            logger.info("正在关闭画面通道……")
            stream_link.stop()
        if link is not None:
            logger.info("正在关闭到模块 B 的连接……")
            link.stop()
        logger.info("前端界面已退出")
    return code


def _build_stream_link(args, slot, bridge):
    """构造到 A 侧展示流的连接。

    回调同样跑在 socket 线程里，所以 ``on_link`` **只做一件事：emit 信号**。
    帧那条路根本不经过这里 —— 它由 ``MjpegClient`` 直接写进 ``slot``
    （``on_frame`` 的默认值就是 ``slot.put``），GUI 侧用定时器主动拉。
    这不是偷懒：经信号投递的图片会排队，而队列不会合并，GUI 一慢
    画面就会越拖越晚（见 ``c_core/mjpeg_client.py`` 的模块文档）。
    """
    from c_core.mjpeg_client import MjpegClient

    def emit_stream_link(source: str, connected: bool) -> None:
        bridge.linkChanged.emit(source, connected)

    return MjpegClient(args.stream_url, slot=slot, on_link=emit_stream_link)


def _build_link(args, bridge):
    """构造到模块 B 的连接，把回调接到信号桥上。

    这些回调是在 socket 线程里被调用的，所以**它们只做一件事：emit 信号**。
    emit 是线程安全的，真正的界面更新由 Qt 排队回 GUI 线程执行。
    """
    from c_core.state_client import SOURCE_CHAT, BackendLink

    def emit_state(state: str, reason: str, _source: str) -> None:
        """状态发两条：只带状态的（驱动脸）+ 带原因的（驱动右栏文字）。

        理由见 ``StateBridge.stateReasonChanged``：给 ``stateChanged`` 加参数
        会改掉 ``FaceWindow.set_state`` 的公开签名。
        """
        bridge.stateChanged.emit(state)
        bridge.stateReasonChanged.emit(state, reason)

    def emit_vai(msg: dict) -> None:
        """专注度展示报文（api_doc §5.6）：整条原样投到 GUI 线程。

        这里**不做任何解释**（不挑字段、不算文案）—— 显示什么由
        ``c_core/display_text.py`` 决定，而它是可以脱离 Qt 单测的。
        """
        bridge.vaiArrived.emit(msg)

    def emit_link(source: str, connected: bool) -> None:
        """连接状态变化 → GUI 线程。

        **对话通道一断，右栏那条专注度必须作废**（发 ``None`` → 界面转
        "未提供"）。它是一栏"当下的数值"，拿掉连接之后继续显示上一个数
        是最坏的一种过期：数字还在、进度条还在，只有它不再更新 ——
        而界面上没有任何别的地方能看出这个区别。

        状态那条有兜底（``state_client`` 断线回落 normal），专注度没有，
        所以补在这里。**只认对话通道**：``--also-status`` 下 8001 掉了
        不影响 8002 上的专注度，那时不该把它清掉。
        """
        bridge.linkChanged.emit(source, connected)
        if not connected and source == SOURCE_CHAT:
            # 这一行是为了"怎么看出来"：右栏只是从 `专注度 VAI 63.5` 变成
            # `专注度 VAI ——`，两者都不报错，没有日志就只能靠盯屏幕。
            logger.info("对话通道已断开，右栏专注度作废（改显示未提供）")
            bridge.vaiArrived.emit(None)

    return BackendLink(
        host=args.host,
        chat_port=args.chat_port,
        status_port=args.status_port if args.also_status else None,
        on_state=emit_state,
        on_vai=emit_vai,
        on_link=emit_link,
        on_reply=lambda msg: bridge.replyArrived.emit(str(msg.get("text") or "")),
        on_proactive=lambda msg: bridge.proactiveArrived.emit(str(msg.get("text") or "")),
    )


def _install_demo_cycle(window, states, app):
    """``--demo``：不连 B，自己按四态循环，每轮都经 ``set_state()``。

    走 ``set_state()`` 而不是直接改属性，是为了和真实路径共用同一套
    日志、标题栏、淡入逻辑 —— 演示模式坏掉而真实路径好的情况就不会发生。

    返回那个 ``QTimer``：主流程用不上（它归 ``app`` 管），但**测试要拿它
    停下来**，否则一次用例跑完定时器还在往一个已关闭的窗口上打。
    """
    from PyQt5.QtCore import QTimer
    # ⚠️ 这个 import **必须在本函数里**（与 ``main()`` 里那行是两份）。
    # 本函数是**模块级**函数，``main()`` 里那句局部 import 不会进它的作用域，
    # 少这一行就是 NameError：``--demo`` 每次启动都崩，而它恰恰是
    # "B 起不来时"的兜底路径 —— 兜底路径自己崩掉是最不该发生的一种坏。
    from c_core.expressions import state_label

    position = {"i": 0}

    def tick():
        state = states[position["i"] % len(states)]
        position["i"] += 1
        window.set_state(state)
        logger.info("演示模式：切到 %s（%s）", state, state_label(state))

    timer = QTimer(app)
    timer.timeout.connect(tick)
    timer.start(int(DEMO_INTERVAL * 1000))
    tick()   # 立刻切一次，不用等第一个间隔
    return timer


if __name__ == "__main__":
    sys.exit(main())
