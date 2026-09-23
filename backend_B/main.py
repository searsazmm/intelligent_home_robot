# -*- coding: utf-8 -*-
"""后端 B 启动入口。

职责：把各个部件装起来，管好线程和退出。

    ┌──────────────┐  TCP 8000（B 是客户端）  ┌─────────────────────────┐
    │  模块 A 视觉  │ ───────────────────────▶ │  VisionClient           │
    └──────────────┘                          │    ↓                    │
                                              │  VisionStateEvaluator   │  → normal/sad/tired/absent
    ┌──────────────┐  TCP 8001（B 是服务端）  │    ↓                    │
    │  模块 C 前端  │ ◀─────────────────────── │  StatusBroadcaster      │
    │              │  TCP 8002（B 是服务端）  │    ↓                    │
    │              │ ◀───────────────────────▶ │  ChatServer → Dialogue  │
    └──────────────┘                          └─────────────────────────┘
                                                        ↓
                                                  CsvHistoryStore
                                                  data/history.csv

线程模型（全部是 daemon 线程，主线程负责收尾）：
    1. 视觉读取线程   1 个，跑 VisionClient.run() 或 OfflineVisionFeeder.run()
    2. 状态发布线程   1 个，轮询判定结果并按需推送
    3. 状态服务线程   1(accept) + N(每客户端)，对 C 的 8001
    4. 对话服务线程   1(accept) + N(每客户端)，对 C 的 8002
    5. 控制台线程     1 个，可选，--stdin 时启动

用法：
    python main.py                              # 连模块 A，正常模式
    python main.py --offline data/sample_vision.csv --speed 5
    python main.py --stdin                      # 顺带支持控制台打字自测
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Optional

# 允许 `python backend_B/main.py` 与 `cd backend_B && python main.py` 两种方式都能跑：
# 保证本目录在 sys.path 里，这样 `import config` / `from core import ...` 都能工作。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.dialogue import DialogueEngine, DialogueReply
from core.history_store import CsvHistoryStore
from core.ui_channel import ChatServer, StatusBroadcaster
from core.vision_client import OfflineVisionFeeder, VisionClient
from core.vision_state import VisionState, VisionStateEvaluator

logger = logging.getLogger("backend_b")


class BackendB:
    """后端 B 的应用装配与生命周期管理。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # ---- 核心组件 ----
        self.evaluator = VisionStateEvaluator()

        self.history: Optional[CsvHistoryStore] = None
        if not args.no_history:
            self.history = CsvHistoryStore(args.history)
            logger.info("历史记录文件：%s（已有 %d 条）", self.history.path, self.history.count())
            self._preload_history_hint()

        self.dialogue = DialogueEngine(history=self.history)

        self.status_server = StatusBroadcaster(
            host=args.status_host, port=args.status_port
        )
        self.chat_server = ChatServer(
            on_chat=self.handle_chat,
            host=args.chat_host, port=args.chat_port,
        )

        # 视觉数据来源：实时 socket 或离线 CSV
        if args.offline:
            self.vision_source = OfflineVisionFeeder(
                csv_path=args.offline,
                evaluator=self.evaluator,
                speed=args.speed,
                loop=args.loop,
            )
            logger.info("离线模式：回放 %s（%.1fx 倍速，loop=%s）",
                        args.offline, args.speed, args.loop)
        else:
            self.vision_source = VisionClient(
                evaluator=self.evaluator,
                host=args.vision_host, port=args.vision_port,
            )

        # ---- 状态覆盖 ----
        # 对话得出的融合状态比纯视觉状态更贴近现实（比如视觉没看出来但文字很消极），
        # 但视觉线程下一秒就会推回它自己的判定，两边打架会让前端状态乱跳。
        # 所以对话产生的状态用"限时覆盖"的方式生效，过期后交还给视觉。
        self._state_lock = threading.Lock()
        self._override_state: Optional[str] = None
        self._override_reason: str = ""
        self._override_until: float = 0.0

        self._stop = threading.Event()
        self._threads = []
        self._last_published_state: Optional[str] = None

    # ==================================================================
    # 启动 / 停止
    # ==================================================================

    def run(self) -> None:
        """启动全部组件并阻塞直到收到退出信号。"""
        try:
            self.status_server.start()
        except OSError as exc:
            logger.error("状态端口 %s:%d 启动失败：%s", self.args.status_host,
                         self.args.status_port, exc)
            logger.error("端口可能已被占用，用 netstat -ano | findstr :%d 查一下",
                         self.args.status_port)
            raise

        try:
            self.chat_server.start()
        except OSError as exc:
            logger.error("对话端口 %s:%d 启动失败：%s", self.args.chat_host,
                         self.args.chat_port, exc)
            self.status_server.stop()
            raise

        self._spawn(self.vision_source.run, "视觉读取")
        self._spawn(self._state_publish_loop, "状态发布")

        if self.args.stdin:
            self._spawn(self._stdin_loop, "控制台输入")

        self._print_banner()
        self._wait_for_exit()
        self.shutdown()

    def shutdown(self) -> None:
        """优雅关闭：先停数据源，再停服务，最后落盘。"""
        if self._stop.is_set() and not self._threads:
            return
        logger.info("正在关闭后端 B ...")
        self._stop.set()

        # 先停视觉源（可能正在 recv 阻塞）
        stop = getattr(self.vision_source, "stop", None)
        if callable(stop):
            stop()

        self.status_server.stop()
        self.chat_server.stop()

        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

        logger.info("后端 B 已停止")

    def _wait_for_exit(self) -> None:
        """主线程守候，处理 Ctrl+C。"""
        try:
            while not self._stop.is_set():
                time.sleep(0.3)
        except KeyboardInterrupt:
            print()   # 把 ^C 后面的光标换行，避免日志和提示符粘在一起
            logger.info("收到 Ctrl+C，准备退出")

    def _spawn(self, target, name: str) -> threading.Thread:
        thread = threading.Thread(target=self._guarded(target, name), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)
        return thread

    def _guarded(self, target, name: str):
        """包一层异常保护：任何子线程崩了都要留下日志，而不是静默死掉。"""
        def wrapper():
            try:
                target()
            except Exception:
                logger.exception("线程 %s 异常退出", name)
        return wrapper

    # ==================================================================
    # 状态发布
    # ==================================================================

    def _state_publish_loop(self) -> None:
        """定期取当前状态并推给 C。变化时推，没变化时靠心跳推。"""
        while not self._stop.is_set():
            try:
                state, reason = self._effective_state()
                if self.status_server.publish(state):
                    # 顺手往对话端口也推一份，C 端只用一条连接也能拿全信息
                    self.chat_server.broadcast_state(state, reason)
            except Exception:
                logger.exception("状态发布出错")
            self._stop.wait(0.2)

    def _effective_state(self):
        """当前对外的状态：优先用未过期的对话覆盖，否则用视觉判定。"""
        now = time.monotonic()
        with self._state_lock:
            if self._override_state and now < self._override_until:
                return self._override_state, self._override_reason

        vision: VisionState = self.evaluator.get_state()
        return vision.state, vision.reason

    def _set_state_override(self, state: str, reason: str, seconds: float = 30.0) -> None:
        """用对话结论临时覆盖视觉状态。"""
        with self._state_lock:
            self._override_state = state
            self._override_reason = reason
            self._override_until = time.monotonic() + seconds

    # ==================================================================
    # 对话处理（ChatServer 的回调）
    # ==================================================================

    def handle_chat(self, text: str) -> DialogueReply:
        """收到用户一句话：生成回复 → 写历史 → 更新状态覆盖 → 返回。"""
        vision = self.evaluator.get_state()
        reply = self.dialogue.respond(text, vision)

        logger.info("用户：%s", text)
        logger.info("机器人：%s  [%s]", reply.reply, reply.reason)

        # 落盘历史（任务要求 4 的另一半：写进去，之后才读得出来）
        if self.history is not None:
            try:
                self.history.append_turn(
                    user_text=text,
                    reply_text=reply.reply,
                    vision_state=reply.state,
                    text_emotion=reply.emotion_label,
                    emotion_score=reply.emotion_score,
                    intent=reply.intent,
                )
            except OSError:
                logger.exception("写入历史记录失败（不影响本次回复）")

        # 对话结论比纯视觉更准，限时覆盖一下，避免前端状态和对话内容对不上
        if reply.state != vision.state:
            self._set_state_override(reply.state, f"对话判定：{reply.reason}")
            self.status_server.publish(reply.state)

        return reply

    # ==================================================================
    # 控制台输入（开发自测用）
    # ==================================================================

    def _stdin_loop(self) -> None:
        """从控制台读一行当作"用户说的话"，方便不开前端 C 就能测对话。"""
        print("\n[控制台输入已开启] 直接打字回车，就当作是用户说的话；输入 :q 退出。\n")
        while not self._stop.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line:
                continue
            if line in (":q", ":quit", "exit"):
                self._stop.set()
                break
            reply = self.handle_chat(line)
            print(f"  → {reply.reply}   [状态={reply.state} 意图={reply.intent}]\n")

    # ==================================================================
    # 杂项
    # ==================================================================

    def _preload_history_hint(self) -> None:
        """启动时读一把历史，验证"预留的 CSV 读取接口"确实能用。"""
        try:
            recent = self.history.load_recent(limit=config.HISTORY_PRELOAD_ROWS)
            if recent:
                logger.info("已预加载最近 %d 条历史记录（供对话上下文使用）", len(recent))
                last = recent[-1]
                logger.info("  最近一条：%s：%s", last.role, last.text[:40])
            else:
                logger.info("历史记录为空，这是第一次运行")
        except Exception:
            logger.exception("预加载历史记录失败（不影响启动）")

    def _print_banner(self) -> None:
        source = f"离线回放 {self.args.offline}" if self.args.offline else \
            f"模块 A {self.args.vision_host}:{self.args.vision_port}"
        print("=" * 62)
        print("  居家陪伴机器人 · 后端 B")
        print("-" * 62)
        print(f"  视觉来源    {source}")
        print(f"  状态推送    {self.args.status_host}:{self.args.status_port}  (B=服务端, 推 normal/sad/tired/absent)")
        print(f"  对话通道    {self.args.chat_host}:{self.args.chat_port}  (B=服务端, 收发 JSON)")
        print(f"  历史记录    {self.history.path if self.history else '已禁用'}")
        print("-" * 62)
        print("  Ctrl+C 退出")
        print("=" * 62)
        print()


# ======================================================================
# 命令行参数
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backend_B",
        description="居家陪伴机器人 · 后端 B（视觉接收 + 文本情绪 + 对话管理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py                                      正常模式，连模块 A\n"
            "  python main.py --offline data/sample_vision.csv     离线回放，无需模块 A\n"
            "  python main.py --offline data/sample_vision.csv --speed 5 --stdin\n"
            "  python main.py --stdin                              控制台打字自测对话\n"
        ),
    )

    parser.add_argument("--offline", metavar="CSV", default=None,
                        help="离线模式：从该 CSV 回放视觉数据，不连模块 A")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="离线回放倍速（默认 1.0）")
    parser.add_argument("--loop", action="store_true",
                        help="离线回放循环播放（默认只播一遍；单帧测试时很有用）")
    parser.add_argument("--stdin", action="store_true",
                        help="开启控制台输入，直接打字测试对话")

    parser.add_argument("--vision-host", default=config.VISION_HOST,
                        help=f"模块 A 地址（默认 {config.VISION_HOST}）")
    parser.add_argument("--vision-port", type=int, default=config.VISION_PORT,
                        help=f"模块 A 端口（默认 {config.VISION_PORT}）")
    parser.add_argument("--status-host", default=config.STATUS_HOST,
                        help=f"状态推送监听地址（默认 {config.STATUS_HOST}）")
    parser.add_argument("--status-port", type=int, default=config.STATUS_PORT,
                        help=f"状态推送监听端口（默认 {config.STATUS_PORT}）")
    parser.add_argument("--chat-host", default=config.CHAT_HOST,
                        help=f"对话通道监听地址（默认 {config.CHAT_HOST}）")
    parser.add_argument("--chat-port", type=int, default=config.CHAT_PORT,
                        help=f"对话通道监听端口（默认 {config.CHAT_PORT}）")

    parser.add_argument("--history", default=config.HISTORY_CSV,
                        help=f"历史记录 CSV 路径（默认 {config.HISTORY_CSV}）")
    parser.add_argument("--no-history", action="store_true",
                        help="不读写历史记录 CSV")
    parser.add_argument("--log-level", default=config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help=f"日志级别（默认 {config.LOG_LEVEL}）")

    return parser


def setup_logging(level: str) -> None:
    """统一日志格式。中文日志在 Windows 控制台要注意编码。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    # Windows 控制台默认可能是 GBK，日志里有中文会乱码或直接抛 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    app = BackendB(args)
    try:
        app.run()
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
