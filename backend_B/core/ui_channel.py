# -*- coding: utf-8 -*-
"""对模块 C（桌面前端）的两个 TCP 服务。

端口 8001 —— 状态推送  [api_doc §4]
    B 是服务端，C 是客户端。B 只发 4 种固定字符串：normal / sad / tired / absent，
    末尾带 \\n。纯文本，不是 JSON。

端口 8002 —— 双向对话  [api_doc §5]
    api_doc 原本只定义了 B→C 的单向状态，没有"用户说的话"怎么进 B 的通道。
    这里新增 8002：C 发 JSON 对话文本给 B，B 回 JSON 回复。
    格式：
        C→B  {"type":"chat","text":"我今天有点累","timestamp":1234.5}
        B→C  {"type":"reply","text":"...","state":"tired","intent":"...","emotion":{...}}
        B→C  {"type":"state","state":"tired","reason":"..."}   状态变化时主动推
        B→C  {"type":"proactive","text":"...","state":"sad","reason":"...","kind":"care_sad"}
             ↑ 机器人**主动**开口（api_doc §5.5 V1.2）。和 reply 的区别见
               broadcast_proactive 的注释。
               ⚠️ kind 只有 care_sad / care_tired / greeting 三种取值（api_doc §5.5
                  的表里冻结了这三个字面量，core/proactive.py 的 CARE_KINDS 与之对应）。
                  别在这里随手写 "care" —— 前端若按 kind 分派文案会静默走到兜底分支。
        C→B  {"type":"ping"}  /  B→C {"type":"pong"}

两个服务都是"多客户端"的：C 端可能反复重启，所以每个连接单开一个线程处理，
写失败就把该连接剔除，绝不因为一个客户端掉线影响其它客户端。

**两条通道都会补发当前状态**（8001 的 _client_loop、8002 的 _send_initial_state）：
新连上的 C 不必等到下一次状态变化或心跳才有数据。
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, List, Optional

import config
from core.protocol import LineBuffer, ProtocolError, decode_json_line, encode_line, encode_text_line

logger = logging.getLogger(__name__)


class _BaseTcpServer:
    """两个服务共用的骨架：监听、接受连接、管理客户端列表、优雅关闭。"""

    def __init__(self, host: str, port: int, name: str) -> None:
        self.host = host
        self.port = port
        self.name = name
        self._server: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动监听线程。端口被占用时抛 OSError，由调用方决定是否致命。"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR：程序重启后不用等 TIME_WAIT 结束就能立刻再监听同一端口，
        # 联调期间频繁重启时非常关键。
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(config.LISTEN_BACKLOG)
        server.settimeout(config.SOCKET_TIMEOUT)
        self._server = server

        thread = threading.Thread(target=self._accept_loop, name=f"{self.name}-accept", daemon=True)
        thread.start()
        self._threads.append(thread)
        logger.info("%s 已监听 %s:%d", self.name, self.host, self.port)

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            self._close_client(client)

    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                assert self._server is not None
                client, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                # stop() 关掉 server socket 时会走到这里，属于正常退出
                break

            client.settimeout(config.SOCKET_TIMEOUT)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._clients_lock:
                self._clients.append(client)
            logger.info("%s：模块 C 已连接 %s:%d", self.name, addr[0], addr[1])

            handler = threading.Thread(
                target=self._client_loop, args=(client, addr), name=f"{self.name}-client", daemon=True
            )
            handler.start()
            self._threads.append(handler)

    def _client_loop(self, client: socket.socket, addr) -> None:
        """单个客户端的读循环；子类覆写 _on_line 决定收到数据后干什么。"""
        buffer = LineBuffer()
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.recv(4096)
                except socket.timeout:
                    self._on_idle(client)
                    continue
                except OSError:
                    break

                if not chunk:
                    break

                try:
                    for line in buffer.feed(chunk):
                        self._on_line(client, line)
                except ProtocolError as exc:
                    logger.warning("%s：协议错误 %s", self.name, exc)
                    buffer.clear()
        finally:
            self._remove_client(client)
            logger.info("%s：客户端 %s:%d 已断开", self.name, addr[0], addr[1])

    def _on_line(self, client: socket.socket, line: str) -> None:
        """子类实现：处理一行输入。"""
        raise NotImplementedError

    def _on_idle(self, client: socket.socket) -> None:
        """子类可实现：recv 超时（暂时没数据）时做点什么，比如发心跳。"""
        return

    def _remove_client(self, client: socket.socket) -> None:
        with self._clients_lock:
            if client in self._clients:
                self._clients.remove(client)
        self._close_client(client)

    @staticmethod
    def _close_client(client: socket.socket) -> None:
        try:
            client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            client.close()
        except OSError:
            pass

    def _broadcast(self, payload: bytes) -> int:
        """给所有已连接的 C 端发同一份数据，返回成功发送的客户端数。

        发送失败的客户端会被剔除 —— 对端进程已经死了但 TCP 还没感知到，
        不剔除的话客户端列表会越积越多。
        """
        with self._clients_lock:
            clients = list(self._clients)

        sent = 0
        dead: List[socket.socket] = []
        for client in clients:
            try:
                client.sendall(payload)
                sent += 1
            except OSError:
                dead.append(client)

        for client in dead:
            self._remove_client(client)
        return sent


class StatusBroadcaster(_BaseTcpServer):
    """[api_doc §4] 8001 端口：向 C 推 normal/sad/tired/absent 四种状态字符串。

    只在状态变化时发送，并每隔 STATUS_HEARTBEAT_SECONDS 补发一次心跳。
    心跳的作用：C 端如果长时间收不到任何数据，可能误以为 B 挂了；而且前端
    要"断连时默认展示 normal"，有稳定心跳它才好判断到底断没断。
    """

    def __init__(
        self,
        host: str = config.STATUS_HOST,
        port: int = config.STATUS_PORT,
    ) -> None:
        super().__init__(host, port, "状态推送服务(8001)")
        self._current: str = config.STATE_NORMAL
        self._last_sent_at: float = 0.0
        #: 保护 _current / _last_sent_at 的读-改-写。
        #: 曾经只有一个生产者（发布循环），现在是三个：
        #: 发布循环、handle_chat（对话判定状态）、主动关怀。
        #: 没有这把锁，两条线程会同时判定为「状态变了」而重复发送，
        #: 或者更糟 —— 同时更新 _last_sent_at 而丢掉一次心跳记账。
        self._publish_lock = threading.Lock()

    # ------------------------------------------------------------------

    def publish(self, state: str, force: bool = False,
                mirror_clients: Optional[int] = None) -> bool:
        """推送状态。状态没变且没到心跳时间则不重复发。返回是否真的发了。

        ``mirror_clients`` 是**另一条通道**（8002 对话端口）此刻的客户端数，
        只用于日志。8002 上那一份由调用方在返回 True 之后发送 —— 见
        :meth:`main.BackendB._publish_state`，它把「推 8001」和「镜像到 8002」
        绑成一个动作，顺便把两边的客户端数一起带进来。

        ⚠️ 为什么是 ``None`` 而不是 ``0``：这两种情况的日志**必须不一样**。
        ``None`` = 调用方不做 8002 镜像，此时报"对话通道 N 个"就是假话；
        只有真的镜像了（哪怕镜像到 0 个客户端）才该报两个数字。
        用 0 当哨兵会让"没有镜像"和"镜像了但那边没人"这两件截然不同的事
        长得一模一样 —— 而这条日志存在的全部意义就是区分它们。
        """
        # 防御性检查：api_doc §4.2 只允许这 4 个值，非法值一律退回 normal
        if state not in config.VALID_STATES:
            logger.warning("非法状态 %r 已拦截，改用 %s", state, config.STATE_NORMAL)
            state = config.STATE_NORMAL

        now = time.monotonic()

        # 判定与记账放在锁内，**发送放在锁外**：
        # sendall 可能因为对端进程卡住而阻塞到 socket 超时（默认 1 秒），
        # 持锁发送会把 0.2 秒的状态发布节拍整个拖垮。
        # 锁外发送的代价只是极小概率的重复广播，而 C 端本来就按状态去重。
        with self._publish_lock:
            changed = state != self._current
            heartbeat_due = (now - self._last_sent_at) >= config.STATUS_HEARTBEAT_SECONDS

            if not force and not changed and not heartbeat_due:
                return False

            self._current = state
            self._last_sent_at = now

        sent = self._broadcast(encode_text_line(state))   # 纯文本 + \n

        if changed:
            if mirror_clients is None:
                logger.info("状态变更 → %s（状态通道 %d 个客户端）", state, sent)
            else:
                # 两个数字分开写，而不是加起来。C 端默认**只连 8002**（除非加
                # --also-status），所以本机演示配置下这里通常长这样：
                #     状态变更 → sad（状态通道 0 个客户端，对话通道 1 个）
                # 那个 0 是**正常的** —— C 根本不在 8001 上。
                # 早先这里只报 8001 的那个数，于是日志一路写着「已推送给 0 个客户端」，
                # 而 C 正在正常收、正常画。答辩时看到这行，第一反应必然是
                # 「广播断了」，然后花半天去查一条根本没坏的链路。
                logger.info("状态变更 → %s（状态通道 %d 个客户端，对话通道 %d 个）",
                            state, sent, mirror_clients)
        return True

    @property
    def current_state(self) -> str:
        with self._publish_lock:
            return self._current

    # 覆写：C 端连上后立刻把当前状态发过去，否则新客户端要等到下次变化才有数据
    def _on_line(self, client: socket.socket, line: str) -> None:
        """C 端理论上不该往 8001 发东西，收到就忽略（但不断开）。"""
        logger.debug("状态端口收到来自 C 的意外数据，已忽略：%r", line[:60])

    def _client_loop(self, client: socket.socket, addr) -> None:
        """覆写：连上先补发当前状态。"""
        try:
            client.sendall(encode_text_line(self._current))
            logger.info("已向新客户端 %s:%d 补发当前状态 %s", addr[0], addr[1], self._current)
        except OSError:
            self._remove_client(client)
            return
        super()._client_loop(client, addr)


class ChatServer(_BaseTcpServer):
    """[新增] 8002 端口：与 C 双向通信，收用户对话文本、回机器人回复。

    收到一条 chat 就交给 on_chat 回调（由 main.py 注入对话管理逻辑），
    回调返回 DialogueReply，这里负责序列化发回。

    state_provider 与「连上即补发」见 _send_initial_state 的注释 ——
    少了它，只连 8002 的前端最长要等一个心跳周期（15 秒）才拿到第一个状态。
    """

    def __init__(
        self,
        on_chat: Callable[[str], object],
        state_provider: Optional[Callable[[], tuple]] = None,
        host: str = config.CHAT_HOST,
        port: int = config.CHAT_PORT,
    ) -> None:
        super().__init__(host, port, "对话服务(8002)")
        self.on_chat = on_chat
        #: 返回 (state, reason) 的可调用体，由 main.py 注入。
        #: 用回调而不是直接持有 BackendB —— 服务端不该认识应用层。
        self._state_provider = state_provider

    # ------------------------------------------------------------------

    def _send_initial_state(self, client: socket.socket, addr) -> bool:
        """新客户端连上时先补发一次当前状态。

        **这是一个真实存在过的缺口**：状态变化时确实会往 8002 广播，
        但「刚连上、状态又还没变」的新客户端要一直等到下一次心跳
        （``STATUS_HEARTBEAT_SECONDS``，默认 15 秒）才有数据。
        只连 8002 的前端（即 frontend_C 的默认配置）于是开局有 15 秒是空白的，
        而 api_doc §4.3 要求的前端行为是「连上就能拿到当前状态」。

        8001 的 StatusBroadcaster 早就有同样的补发逻辑，8002 一直漏了。
        """
        if self._state_provider is None:
            return True
        try:
            state, reason = self._state_provider()
        except Exception:
            # 取状态失败不该把新连接掐掉 —— 客户端留着，等下一次广播
            logger.exception("取当前状态失败，跳过补发（连接保留）")
            return True

        if state not in config.VALID_STATES:
            logger.warning("补发时发现非法状态 %r，改用 %s", state, config.STATE_NORMAL)
            state = config.STATE_NORMAL

        ok = self._send(client, {
            "type": "state",
            "state": state,
            "reason": reason or "连接建立时补发",
            "timestamp": time.time(),
        })
        if ok:
            logger.info("已向新客户端 %s:%d 补发当前状态 %s", addr[0], addr[1], state)
        return ok

    def _client_loop(self, client: socket.socket, addr) -> None:
        """覆写：连上先补发当前状态，再进入正常的读循环。"""
        if not self._send_initial_state(client, addr):
            self._remove_client(client)
            return
        super()._client_loop(client, addr)

    def _on_line(self, client: socket.socket, line: str) -> None:
        """处理 C 发来的一行 JSON。"""
        try:
            payload = decode_json_line(line)
        except ProtocolError as exc:
            logger.warning("对话端口收到非法报文：%s", exc)
            self._send(client, {"type": "error", "message": "报文不是合法 JSON"})
            return

        msg_type = str(payload.get("type") or "chat").strip().lower()

        if msg_type == "ping":
            self._send(client, {"type": "pong", "timestamp": time.time()})
            return

        if msg_type != "chat":
            logger.debug("未知消息类型 %r，已忽略", msg_type)
            self._send(client, {"type": "error", "message": f"未知消息类型 {msg_type}"})
            return

        text = str(payload.get("text") or "").strip()
        if not text:
            self._send(client, {"type": "error", "message": "text 字段为空"})
            return

        started = time.monotonic()
        try:
            reply = self.on_chat(text)
        except Exception:
            # 对话逻辑出问题也要给 C 端一个交代，不能让前端干等
            logger.exception("对话处理抛出异常")
            self._send(client, {"type": "error", "message": "内部错误，已记录日志"})
            return

        payload_out = reply.to_dict() if hasattr(reply, "to_dict") else {"type": "reply", "text": str(reply)}
        self._send(client, payload_out)
        logger.info(
            "对话：%r → %r（%s，耗时 %dms）",
            text[:30], payload_out.get("text", "")[:30],
            payload_out.get("state", "?"),
            int((time.monotonic() - started) * 1000),
        )

    def broadcast_state(self, state: str, reason: str = "") -> None:
        """状态变化时也往对话端口推一份，C 端在一个连接里就能拿全信息。"""
        self._broadcast(encode_line({
            "type": "state",
            "state": state,
            "reason": reason,
            "timestamp": time.time(),
        }))

    def broadcast_proactive(self, text: str, state: str, reason: str = "",
                            kind: str = "") -> int:
        """广播一条**机器人主动发起**的话（api_doc §5 V1.2）。

        ``type`` 与 ``reply`` 分开是有意的：``reply`` 一定是「用户说了什么之后的
        应答」，``proactive`` 是「用户什么都没说、机器人自己开口」。
        前端据此可以把主动关怀和普通应答区分对待（比如将来加提示音），
        而不是被迫从上下文里猜。

        同时带上 ``state``：否则会出现「B 正在说关心的话、C 的表情却还是正常脸」
        这种割裂。返回成功送达的客户端数。
        """
        return self._broadcast(encode_line({
            "type": "proactive",
            "text": text,
            "state": state,
            "reason": reason,
            "kind": kind,
            "timestamp": time.time(),
        }))

    @staticmethod
    def _send(client: socket.socket, obj: dict) -> bool:
        try:
            client.sendall(encode_line(obj))
            return True
        except OSError:
            return False
