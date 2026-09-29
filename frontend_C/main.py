# -*- coding: utf-8 -*-
"""模块 C（前端界面）入口。

启动顺序按 api_doc §6.2：**先 A → 再 B → 最后 C**。
不过 C 不依赖 A，甚至不强依赖 B —— B 没起来时它会安静地退避重连，
界面上照常显示 normal 表情，B 一起来就自动接上。所以启动顺序反了不会坏。

    python main.py                     # 默认：连 127.0.0.1:8002
    python main.py --fullscreen        # 全屏无边框（答辩用）
    python main.py --demo              # 不连 B，四态循环（B 起不来时的兜底）
    python main.py --dump-glyphs       # 字形自检：看不到方块即为正常
    python main.py --screenshot out/   # 导出四态 PNG（做 PPT 用），然后退出

⚠️ **不要 import cv2**。cv2 自带一套 Qt5 插件，和 PyQt5 的插件同处一个进程时
会互相顶掉对方的平台插件，症状是「Qt platform plugin could be initialized」
这种看不懂的崩。requirements.txt 里只放 PyQt5 也是这个原因。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

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

    parser.add_argument("--demo", action="store_true",
                        help="离线演示：不连 B，四态循环切换（答辩现场 B 起不来时的兜底）")

    mode = parser.add_argument_group("一次性任务（做完就退出）")
    mode.add_argument("--dump-glyphs", action="store_true",
                      help="字形自检窗口：把所有颜文字平铺出来，看不到方块即为正常")
    mode.add_argument("--screenshot", metavar="DIR",
                      help="把四种状态各导出成一张 PNG 到 DIR，然后退出")

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


def _run_screenshots(window, dump_window, out_dir: str) -> int:
    """导出四态 PNG。做完就退出，不需要事件循环。"""
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

    print(f"\n共导出 {len(written)} 张到 {os.path.abspath(out_dir)}")
    print("请**肉眼**确认：背景纯黑、颜文字是白色且没有任何方块（豆腐块）。")
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
    from ui.window import FaceWindow, GlyphDumpWindow, StateBridge, pick_font_family

    family = pick_font_family()
    window = FaceWindow(family, font_size=args.font_size, debug_hud=args.debug_hud)

    # ---- 一次性任务：字形自检 / 截图 ----
    if args.dump_glyphs or args.screenshot:
        dump_window = GlyphDumpWindow(family)
        if args.screenshot:
            return _run_screenshots(window, dump_window, args.screenshot)
        window.show()
        dump_window.show()
        # 两个窗口并排，免得字形表盖住主窗口
        window.move(0, 0)
        dump_window.move(940, 0)
        logger.info("字形自检已打开：逐个核对，没有方块即为正常")
        return app.exec_()

    # ---- 信号桥：socket 线程 → GUI 线程 ----
    # 只连接「绑定方法」（window 的方法），Qt 才知道接收者在 GUI 线程，
    # 从而用 QueuedConnection 投递。连 lambda 会退化成 direct connection，
    # 于是在 socket 线程里直接画控件 —— Qt 最经典的崩溃来源。
    bridge = StateBridge()
    bridge.stateChanged.connect(window.set_state)
    bridge.linkChanged.connect(window.set_link)
    bridge.replyArrived.connect(window.note_message)
    bridge.proactiveArrived.connect(window.note_message)

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
        if link is not None:
            logger.info("正在关闭到模块 B 的连接……")
            link.stop()
        logger.info("前端界面已退出")
    return code


def _build_link(args, bridge):
    """构造到模块 B 的连接，把回调接到信号桥上。

    这些回调是在 socket 线程里被调用的，所以**它们只做一件事：emit 信号**。
    emit 是线程安全的，真正的界面更新由 Qt 排队回 GUI 线程执行。
    """
    from c_core.state_client import BackendLink

    return BackendLink(
        host=args.host,
        chat_port=args.chat_port,
        status_port=args.status_port if args.also_status else None,
        on_state=lambda state, reason, source: bridge.stateChanged.emit(state),
        on_link=lambda source, connected: bridge.linkChanged.emit(source, connected),
        on_reply=lambda msg: bridge.replyArrived.emit(str(msg.get("text") or "")),
        on_proactive=lambda msg: bridge.proactiveArrived.emit(str(msg.get("text") or "")),
    )


def _install_demo_cycle(window, states, app) -> None:
    """``--demo``：不连 B，自己按四态循环，每轮都经 ``set_state()``。

    走 ``set_state()`` 而不是直接改属性，是为了和真实路径共用同一套
    日志、标题栏、淡入逻辑 —— 演示模式坏掉而真实路径好的情况就不会发生。
    """
    from PyQt5.QtCore import QTimer

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


if __name__ == "__main__":
    sys.exit(main())
