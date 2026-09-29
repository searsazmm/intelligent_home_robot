# -*- coding: utf-8 -*-
"""模拟模块 C（桌面前端）—— 用命令行代替桌面前端，验证 B 的两个服务。

同时连 B 的两个端口：
    8001  只读，收 normal/sad/tired/absent 纯文本状态字符串（api_doc §4）
    8002  双向，发用户对话文本、收机器人回复（api_doc §5，V1.1 新增）

用法：
    python tools/mock_c_client.py                  # 交互式打字聊天
    python tools/mock_c_client.py --auto           # 自动发一串预设对话
    python tools/mock_c_client.py --chat-file data/sample_chat.csv --auto
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.protocol import LineBuffer, ProtocolError, decode_json_line, encode_line

# 状态字符串 → 显示用的中文，纯粹为了控制台好读
STATE_LABELS = {
    "normal": "状态正常",
    "sad": "情绪低落",
    "tired": "疲惫",
    "absent": "走神/无人",
}


class StatusListener(threading.Thread):
    """8001 端口：只收不发，把状态变化打印出来。"""

    def __init__(self, host: str, port: int) -> None:
        super().__init__(name="status-listener", daemon=True)
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self.last_state = None

    def run(self) -> None:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5.0)
        except OSError as exc:
            print(f"[8001] 连接失败：{exc}")
            return

        print(f"[8001] 已连接状态端口 {self.host}:{self.port}")
        buffer = LineBuffer()
        try:
            while not self._stop.is_set():
                chunk = sock.recv(1024)
                if not chunk:
                    print("[8001] 状态端口被服务端关闭")
                    break
                for line in buffer.feed(chunk):
                    self._on_status(line.strip())
        except OSError:
            pass
        finally:
            sock.close()

    def _on_status(self, state: str) -> None:
        """收到状态字符串。api_doc §4.3.2 要求：收到未知内容时默认显示 normal。"""
        if state not in config.VALID_STATES:
            print(f"[8001] 收到未知状态 {state!r}，按约定回退显示 normal")
            state = config.STATE_NORMAL
        if state == self.last_state:
            return   # 心跳重复，不刷屏
        self.last_state = state
        print(f"[8001] 状态 → {state:7s} ({STATE_LABELS.get(state, '?')})")

    def stop(self) -> None:
        self._stop.set()


class ChatClient:
    """8002 端口：发对话、收回复。"""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.sock = None

    def connect(self) -> bool:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
        except OSError as exc:
            print(f"[8002] 连接失败：{exc}")
            return False
        print(f"[8002] 已连接对话端口 {self.host}:{self.port}")
        return True

    def send_chat(self, text: str) -> None:
        """发一句用户说的话，然后等 B 的回复。"""
        if self.sock is None:
            return
        self.sock.sendall(encode_line({"type": "chat", "text": text, "timestamp": time.time()}))

        buffer = LineBuffer()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self.sock.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                print("[8002] 等待回复超时")
                return
            except OSError:
                return
            if not chunk:
                print("[8002] 对话端口被服务端关闭")
                return

            for line in buffer.feed(chunk):
                if self._handle(line):
                    return

    def _handle(self, line: str) -> bool:
        """处理一行回复。返回 True 表示这轮对话结束了。"""
        try:
            msg = decode_json_line(line)
        except ProtocolError:
            print(f"[8002] 无法解析：{line[:80]!r}")
            return False

        msg_type = msg.get("type")
        if msg_type == "reply":
            emotion = msg.get("emotion") or {}
            print(f"[8002] 机器人：{msg.get('text')}")
            print(f"       └ 状态={msg.get('state')} 意图={msg.get('intent')} "
                  f"情绪={emotion.get('label')}/{emotion.get('detail')}")
            return True
        if msg_type == "state":
            # 对话过程中 B 也会顺手推状态，跳过继续等真正的 reply
            print(f"[8002] （状态更新 {msg.get('state')}：{msg.get('reason', '')}）")
            return False
        if msg_type == "error":
            print(f"[8002] 服务端报错：{msg.get('message')}")
            return True
        return False

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


def load_chat_script(path: str):
    """读自动对话脚本 CSV：一列 text，可选一列 delay（秒）。"""
    script = []
    with open(path, "r", newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                delay = float(row.get("delay") or 2.0)
            except ValueError:
                delay = 2.0
            script.append((text, delay))
    return script


def run_auto(chat: ChatClient, script, between: float) -> None:
    """自动发预设对话。"""
    for text, delay in script:
        time.sleep(min(delay, between) if between else delay)
        print(f"\n[8002] 用户：{text}")
        chat.send_chat(text)
    print("\n自动对话结束")


def run_interactive(chat: ChatClient) -> None:
    """手动打字。"""
    print("\n直接打字回车，当作是用户说的话；输入 :q 退出。\n")
    while True:
        try:
            text = input("[用户] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in (":q", ":quit", "exit"):
            break
        chat.send_chat(text)


def main() -> int:
    # Windows 控制台默认可能是 GBK，日志里有中文会乱码。
    # ⚠️ 这行不只是好看：输出**重定向到文件**时 Python 用的是 ANSI 代码页（本机 GBK），
    #    而 tools/e2e_offline_check.sh 是 UTF-8 的、里面 grep 的是中文字面量 ——
    #    两边编码不一致，脚本会一直报「没收到回复」，而功能其实完全正常。
    #    同一个坑也让 `python tools/mock_c_client.py > log.txt` 出来的文件在
    #    VSCode / Git Bash 里打开是乱码。
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="模拟模块 C：连 B 的状态口和对话口")
    parser.add_argument("--host", default="127.0.0.1", help="B 的地址")
    parser.add_argument("--status-port", type=int, default=config.STATUS_PORT)
    parser.add_argument("--chat-port", type=int, default=config.CHAT_PORT)
    parser.add_argument("--auto", action="store_true", help="自动发预设对话，不交互")
    parser.add_argument("--chat-file", default=os.path.join(config.DATA_DIR, "sample_chat.csv"),
                        help="自动对话脚本 CSV")
    parser.add_argument("--between", type=float, default=4.0,
                        help="自动模式下每句之间最多等几秒")
    args = parser.parse_args()

    listener = StatusListener(args.host, args.status_port)
    listener.start()

    chat = ChatClient(args.host, args.chat_port)
    if not chat.connect():
        print("提示：后端 B 启动了吗？先跑 python main.py")
        listener.stop()
        return 1

    try:
        if args.auto:
            if not os.path.exists(args.chat_file):
                print(f"对话脚本不存在：{args.chat_file}")
                return 1
            script = load_chat_script(args.chat_file)
            print(f"已载入 {len(script)} 句预设对话：{args.chat_file}")
            run_auto(chat, script, args.between)
        else:
            run_interactive(chat)
    finally:
        chat.close()
        listener.stop()
        # 等状态线程把最后几行日志打完，避免输出被截断
        time.sleep(0.3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
