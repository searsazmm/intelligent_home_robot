# -*- coding: utf-8 -*-
"""连接后端 B，把状态与对话事件派发出来。

连法（api_doc §5.4.1 允许「两个都连」或「只连 8002」）：

    默认只连 8002  —— 它是双向 JSON 通道，状态与 reply/proactive 都在一条连接上，
                      只需要一套重连状态机、一个 JSON 解析器。
    --also-status 再连 8001 —— api_doc §4 规定的专用状态通道（纯文本），
                      连上即被服务端补发当前状态。用于演示 §4 时打开。

**为什么不默认两个都连**：两条通道都会推同一个状态，于是有两个独立视图可能不一致
（B 在对话里会立刻往 8001 推，而 8002 要等下一个 0.2 秒节拍），表情就会来回抖。
一个状态源、一个去重点，是更稳的选择。

⚠️ **本模块在后台线程里运行，绝不碰 Qt 控件。**
回调是在 socket 线程里被调用的，上层（ui/window.py 的 StateBridge）负责把事件转成
Qt 信号投递回 GUI 线程。这条边界必须守住 —— 在非 GUI 线程里操作控件是 Qt 最经典的
崩溃来源。
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Optional

from c_core.expressions import DEFAULT_STATE, normalize_state
from c_core.protocol import LineBuffer, ProtocolError, decode_json_line, encode_line

logger = logging.getLogger(__name__)

#: 状态来源标识，回调里用它区分是哪条通道给的
SOURCE_CHAT = "chat"      # 8002
SOURCE_STATUS = "status"  # 8001


def _noop(*_args, **_kwargs) -> None:
    """默认回调：什么都不做。"""


class BackendLink:
    """到模块 B 的连接集合（8002 必需，8001 可选），带自动重连。

    线程模型：每个端口一个 daemon 线程，各自独立重连，互不影响。
    ``stop()`` 会立刻唤醒阻塞在 ``recv`` 上的线程（靠 ``shutdown``），
    所以退出通常在毫秒级完成。

    回调（全部在 socket 线程里执行，上层负责切回 GUI 线程）：
        on_state(state, reason, source)  状态变化时调用；断开时也会调用（回落 normal）
        on_reply(msg)                    收到 B 的对话回复报文
        on_proactive(msg)                收到 B 的主动关怀报文（api_doc §5 V1.2）
        on_link(source, connected)       连接建立/断开时调用，供界面显示
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        chat_port: int = 8002,
        status_port: Optional[int] = None,
        on_state: Optional[Callable[[str, str, str], None]] = None,
        on_reply: Optional[Callable[[dict], None]] = None,
        on_proactive: Optional[Callable[[dict], None]] = None,
        on_link: Optional[Callable[[str, bool], None]] = None,
        reconnect_initial: float = 1.0,
        reconnect_max: float = 10.0,
        socket_timeout: float = 1.0,
    ) -> None:
        self.host = host
        self.chat_port = chat_port
        self.status_port = status_port  # None 表示不连 8001

        self._on_state = on_state or _noop
        self._on_reply = on_reply or _noop
        self._on_proactive = on_proactive or _noop
        self._on_link = on_link or _noop

        self._reconnect_initial = reconnect_initial
        self._reconnect_max = reconnect_max
        self._socket_timeout = socket_timeout

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sockets: list[socket.socket] = []
        self._sockets_lock = threading.Lock()

        # 8001 是否连着 —— 决定状态以哪条通道为准
        self._status_connected = False
        # 最近一次派发出去的状态，用于去重（B 每 15 秒有一次心跳重发同一状态）
        self._last_state: Optional[str] = None
        self._state_lock = threading.Lock()

    # ==================================================================
    # 生命周期
    # ==================================================================

    def start(self) -> None:
        """起线程。连接是异步建立的，本方法立刻返回。"""
        self._spawn(self._chat_loop, "B-对话通道(8002)")
        if self.status_port is not None:
            self._spawn(self._status_loop, "B-状态通道(8001)")

    def stop(self, timeout: float = 1.5) -> None:
        """停止并等待线程收尾。

        先 ``shutdown`` 再 ``close`` 是关键：只 close 的话，另一个线程可能正阻塞在
        ``recv`` 上，而 ``recv`` 不会因为本地 close 而返回（Windows 上尤其明显）。
        ``shutdown(SHUT_RDWR)`` 会让对端收到 FIN，阻塞中的 recv 立刻返回 0。
        """
        self._stop.set()
        with self._sockets_lock:
            sockets = list(self._sockets)
        for sock in sockets:
            self._shutdown_socket(sock)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=self._guarded(target, name), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _guarded(self, target, name: str):
        """包一层异常保护：子线程崩了要留日志，而不是静默死掉。"""
        def wrapper():
            try:
                target()
            except Exception:
                logger.exception("%s 线程异常退出", name)
        return wrapper

    # ==================================================================
    # 8002：对话通道（默认唯一的连接）
    # ==================================================================

    def _chat_loop(self) -> None:
        backoff = self._reconnect_initial
        while not self._stop.is_set():
            sock = self._connect(self.chat_port, "对话通道(8002)")
            if sock is None:
                backoff = self._backoff_wait(backoff)
                if backoff is None:
                    return
                continue
            backoff = self._reconnect_initial  # 连上了，退避重置

            buffer = LineBuffer()  # 新连接用新缓冲，避免上条连接的半包污染
            try:
                # 连上先探活，顺便让日志能明确区分「连上了」和「连上了但没数据」
                sock.sendall(encode_line({"type": "ping"}))
                while not self._stop.is_set():
                    if not self._recv_into(sock, buffer, self._dispatch_chat_line):
                        break
            finally:
                self._drop_socket(sock)

            self._on_link(SOURCE_CHAT, False)
            self._apply_state(DEFAULT_STATE, "与模块 B 的对话通道已断开", SOURCE_CHAT)
            backoff = self._backoff_wait(backoff)
            if backoff is None:
                return

    def _dispatch_chat_line(self, line: str) -> None:
        """处理 8002 来的一行 JSON。非法报文只记日志，绝不让线程退出。"""
        try:
            msg = decode_json_line(line)
        except ProtocolError as exc:
            logger.warning("对话通道收到非法报文：%s", exc)
            return

        msg_type = str(msg.get("type") or "").strip().lower()

        if msg_type == "state":
            self._apply_state(
                msg.get("state"),
                str(msg.get("reason") or ""),
                SOURCE_CHAT,
            )
        elif msg_type == "reply":
            self._on_reply(msg)
        elif msg_type == "proactive":
            # B 主动发起的关怀（api_doc §5 V1.2）。它同时带状态，一并更新表情，
            # 免得「B 说关心的话、表情却还是正常的」这种割裂。
            self._on_proactive(msg)
            if msg.get("state"):
                self._apply_state(msg.get("state"), str(msg.get("reason") or ""), SOURCE_CHAT)
        elif msg_type == "pong":
            logger.debug("对话通道心跳正常")
        elif msg_type == "error":
            logger.warning("模块 B 报错：%s", msg.get("message"))
        else:
            logger.debug("对话通道收到未知报文类型 %r，已忽略", msg_type)

    # ==================================================================
    # 8001：状态通道（可选，--also-status）
    # ==================================================================

    def _status_loop(self) -> None:
        source_name = "状态通道(8001)"
        backoff = self._reconnect_initial
        while not self._stop.is_set():
            sock = self._connect(self.status_port, source_name)
            if sock is None:
                backoff = self._backoff_wait(backoff)
                if backoff is None:
                    return
                continue
            backoff = self._reconnect_initial

            buffer = LineBuffer()
            self._status_connected = True
            try:
                while not self._stop.is_set():
                    # 8001 是纯文本状态行，不是 JSON
                    if not self._recv_into(sock, buffer, self._dispatch_status_line):
                        break
            finally:
                self._status_connected = False
                self._drop_socket(sock)

            self._on_link(SOURCE_STATUS, False)
            # 8001 断了不一定代表 B 全挂了（8002 可能还活着），
            # 所以这里不主动回落成 normal —— 交给 8002 那条通道去决定。
            backoff = self._backoff_wait(backoff)
            if backoff is None:
                return

    def _dispatch_status_line(self, line: str) -> None:
        """8001 只发四种纯文本状态字符串（api_doc §4.2）。"""
        self._apply_state(line, "来自状态通道(8001)", SOURCE_STATUS)

    # ==================================================================
    # 状态派发（单一入口 —— 去重与优先级都在这里）
    # ==================================================================

    def _apply_state(self, raw_state, reason: str, source: str) -> None:
        """规整 → 判优先级 → 去重 → 派发。

        api_doc §4.3：收到未知内容时默认展示 normal。所以在 ``normalize_state``
        之后**不存在**非法状态，这里不需要再判一次。
        """
        state = normalize_state(raw_state)

        # 8001 连上时它是权威源（api_doc §4 的专用状态通道）；
        # 8002 只在 8001 没连的时候说话，避免两条通道各说各话、表情来回抖。
        if source == SOURCE_CHAT and self._status_connected:
            logger.debug("忽略来自对话通道的状态 %s（8001 已连接，以它为准）", state)
            return

        with self._state_lock:
            if state == self._last_state:
                return  # 心跳重发，不重复派发（否则表情每 15 秒闪一下）
            self._last_state = state

        logger.info("状态 → %s（%s）%s", state, source, f"：{reason}" if reason else "")
        self._on_state(state, reason, source)

    # ==================================================================
    # 发送（供测试与将来的扩展；界面本身没有输入框）
    # ==================================================================

    def send_chat(self, text: str) -> bool:
        """往 8002 发一句用户的话。当前没有连接时返回 False。"""
        with self._sockets_lock:
            targets = list(self._sockets)
        if not targets:
            return False
        payload = encode_line({"type": "chat", "text": text, "timestamp": time.time()})
        for sock in targets:
            try:
                sock.sendall(payload)
                return True
            except OSError:
                continue
        return False

    # ==================================================================
    # 内部工具
    # ==================================================================

    def _connect(self, port: int, name: str) -> Optional[socket.socket]:
        """建立连接。失败只记日志并返回 None（由调用方退避重试）。"""
        if self._stop.is_set():
            return None
        try:
            sock = socket.create_connection((self.host, port), timeout=5.0)
        except OSError as exc:
            logger.info("连接模块 B 的%s失败：%s（稍后重试）", name, exc)
            return None

        sock.settimeout(self._socket_timeout)  # 必须设超时，否则 stop() 唤醒不了它
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        with self._sockets_lock:
            self._sockets.append(sock)
        logger.info("已连接模块 B 的%s %s:%d", name, self.host, port)
        self._on_link(SOURCE_STATUS if port == self.status_port else SOURCE_CHAT, True)
        return sock

    def _recv_into(self, sock: socket.socket, buffer: LineBuffer, handler) -> bool:
        """收一轮数据并逐行交给 handler。返回 False 表示连接该结束了。"""
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            return True   # 暂时没数据，回去检查 stop 标志
        except OSError:
            return False

        if not chunk:
            logger.info("模块 B 关闭了连接")
            return False

        try:
            for line in buffer.feed(chunk):
                handler(line)
        except ProtocolError as exc:
            logger.warning("分帧出错：%s", exc)
            buffer.clear()
        return True

    def _backoff_wait(self, current: float) -> Optional[float]:
        """退避等待，返回下一次该用的退避值；返回 None 表示收到停止信号。

        退避值由调用方以局部变量持有，**不放在实例属性上** —— 否则 8002 和 8001
        两条线程会共用一个退避进度，互相把对方的等待时间推长。
        """
        if self._stop.wait(current):
            return None
        return min(current * 2.0, self._reconnect_max)

    def _drop_socket(self, sock: socket.socket) -> None:
        with self._sockets_lock:
            if sock in self._sockets:
                self._sockets.remove(sock)
        self._shutdown_socket(sock)

    @staticmethod
    def _shutdown_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
