"""TCP 上的换行分隔 JSON 收发。

沿用《系统总接口文档》§2.1：**每条 JSON 数据末尾必须携带 `\\n` 分隔符**，
用于防止粘包。本模块是全系统唯一的收发实现，五个模块共用。

之所以必须共用一份，是因为下面这些 Windows 特有的坑如果各写一遍，
必然有模块踩中：

1. **绝不能 ``recv().split(b"\\n")`` 再逐个 ``decode()``。**
   一个中文字符占 3 个 UTF-8 字节，可能正好跨在两个 TCP 分段之间，
   逐段解码会抛 ``UnicodeDecodeError``。必须用面向行的读取器。

2. **越界异常的类型不统一。** ``readuntil`` 抛 ``LimitOverrunError``；
   ``readline`` 会把它转成 ``ValueError``。两者都要捕获。

3. **开发期重启带来的是 ``ConnectionResetError``（WinError 10054），
   而不是干净的 EOF。** 只捕获 ``IncompleteReadError`` 的读循环
   会在第一次重启时崩掉整个进程。

4. **Windows 上 ``reuse_address`` 默认为 False**，快速重启会撞
   ``WinError 10048``。但**不能**图省事把 ``SO_REUSEADDR`` 设成 True——
   在 Windows 上那会允许两个进程绑定同一端口，客户端被静默分流到
   两个进程，产生极难排查的诡异行为。正确做法是**带退避的重试绑定**。

另外，本模块提供同步与异步两套接口：A 与 C 是线程模型（A 是采集-推理
流水线，C 被 Tk 主循环约束），B 是 asyncio 模型。强行统一并发模型
得不偿失，统一**帧格式**才是关键。
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any

#: 单条报文的字节上限。实际报文（10s 窗口聚合结果）约 1–2 KB，此处留足余量。
MAX_MESSAGE_BYTES = 1 << 20  # 1 MiB

#: 连续空行的容忍上限。超过就按协议违规报错，而不是一直陪着读下去。
#: 取 64 是因为真实场景最多遇到一两个多余换行——真刷到这个量，
#: 对端已经不是我们自己的模块了。
MAX_CONSECUTIVE_BLANK_LINES = 64

#: 端口占用时的重试次数与退避。
_BIND_ATTEMPTS = 6
_BIND_BACKOFF_SEC = 0.5

#: 读循环应当捕获的异常集合。
#: 覆盖：对端干净关闭、对端重置连接、Windows 连接中止、报文越界。
READ_ERRORS = (
    asyncio.IncompleteReadError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    ValueError,          # readline 把 LimitOverrunError 转成 ValueError
    OSError,             # 兜底：包含 WinError 10054 等
)


class ProtocolError(Exception):
    """报文不符合协议约定。"""


# ================================================================ 编解码

def encode(msg: dict[str, Any]) -> bytes:
    """把消息编码为 ``JSON + \\n``。

    ``ensure_ascii=False`` 保留中文原样，便于抓包排查；
    ``separators`` 去掉多余空格，缩小报文体积。
    """
    return (
        json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def decode(line: bytes) -> dict[str, Any]:
    """把一行字节解码为消息字典。"""
    # strip() 同时处理 Windows 风格的行尾 \r\n。
    text = line.decode("utf-8", errors="strict").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"报文不是合法 JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ProtocolError("报文顶层必须是 JSON 对象")
    return obj


# ================================================================ 异步接口（B / D / E）

async def read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """读取一条消息。

    :returns: 消息字典；对端关闭时返回 ``None``。
    :raises ProtocolError: 报文超长或非法。
    """
    blanks = 0
    while True:
        try:
            line = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError:
            # 对端关闭，缓冲区里没有完整的一行。
            return None
        except asyncio.LimitOverrunError as exc:
            raise ProtocolError(f"报文超过 {MAX_MESSAGE_BYTES} 字节上限") from exc

        if line.strip():
            return decode(line)
        # 空行属于心跳噪声，**跳过**——即继续读下一条，而不是返回。
        #
        # ⚠️ 这里原先是 ``return None``，与同步的 :class:`LineReader`
        # 不一致：那个是真的 ``continue`` 跳过。两套实现打架的后果不对称——
        # ``None`` 在 B 的读循环里语义是"对端关闭"，于是一个多出来的空行
        # 会让 B 把视觉链路或老人端当成断线重连，而 C/A 那边什么事都没有。
        # 查起来就是"B 偶尔莫名其妙重连"，且只有发空行的那一端会撞上。
        # 按协议空行本就不该出现，但既然同步侧定义为跳过，异步侧就得一样。
        #
        # 跳过就带来一个新的风险：一直发空行的对端会让这个循环**永远不返回**。
        # 改动之前它是 fail-closed 的（一遇到空行就断），所以这个上限是
        # 这次改动的配套责任。容忍偶发/多余的换行（真实会遇到的就这种），
        # 但连续刷空行按协议违规处理——报错，别静默地陪着它转。
        blanks += 1
        if blanks > MAX_CONSECUTIVE_BLANK_LINES:
            raise ProtocolError(
                f"连续收到超过 {MAX_CONSECUTIVE_BLANK_LINES} 个空行，判定为对端异常"
            )


async def write_message(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    """写出一条消息并 flush。"""
    writer.write(encode(msg))
    await writer.drain()


async def start_server_with_retry(
    handler,
    host: str,
    port: int,
) -> asyncio.AbstractServer:
    """启动 TCP 服务端，端口被占用时退避重试。

    ``limit`` 与 :data:`MAX_MESSAGE_BYTES` 保持一致，保证 ``readuntil``
    不会先于我们自己的长度校验抛出 ``LimitOverrunError``。
    """
    last_exc: OSError | None = None
    for attempt in range(_BIND_ATTEMPTS):
        try:
            return await asyncio.start_server(
                handler, host, port, limit=MAX_MESSAGE_BYTES
            )
        except OSError as exc:
            last_exc = exc
            if attempt < _BIND_ATTEMPTS - 1:
                delay = _BIND_BACKOFF_SEC * (2**attempt)
                print(
                    f"[framing] 端口 {host}:{port} 绑定失败({exc})，"
                    f"{delay:.1f}s 后重试 ({attempt + 1}/{_BIND_ATTEMPTS})"
                )
                await asyncio.sleep(delay)
    raise OSError(f"无法绑定 {host}:{port}，已重试 {_BIND_ATTEMPTS} 次") from last_exc


# ================================================================ 同步接口（A / C）

def bind_listen_with_retry(
    host: str,
    port: int,
    backlog: int = 4,
) -> socket.socket:
    """绑定并监听，端口被占用时退避重试。供线程模型模块使用。

    ⚠️ 刻意**不**设置 ``SO_REUSEADDR``：在 Windows 上它允许两个进程
    绑定同一端口，客户端会被静默分流。宁可重试，也不要静默串线。
    """
    last_exc: OSError | None = None
    for attempt in range(_BIND_ATTEMPTS):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            sock.listen(backlog)
            return sock
        except OSError as exc:
            last_exc = exc
            sock.close()
            if attempt < _BIND_ATTEMPTS - 1:
                delay = _BIND_BACKOFF_SEC * (2**attempt)
                print(
                    f"[framing] 端口 {host}:{port} 绑定失败({exc})，"
                    f"{delay:.1f}s 后重试 ({attempt + 1}/{_BIND_ATTEMPTS})"
                )
                time.sleep(delay)
    raise OSError(f"无法绑定 {host}:{port}，已重试 {_BIND_ATTEMPTS} 次") from last_exc


class LineReader:
    """同步的按行读取器。

    内部维护字节缓冲，只在读到完整的 ``\\n`` 时才解码——这样即使
    一个中文的 3 个字节跨在两次 ``recv`` 之间也不会解码失败。
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = bytearray()

    def read_message(self) -> dict[str, Any] | None:
        """读一条消息；对端关闭返回 ``None``。"""
        while True:
            idx = self._buf.find(b"\n")
            if idx >= 0:
                line = bytes(self._buf[:idx])
                del self._buf[: idx + 1]
                if not line.strip():
                    continue
                return decode(line)

            if len(self._buf) > MAX_MESSAGE_BYTES:
                raise ProtocolError(f"报文超过 {MAX_MESSAGE_BYTES} 字节上限")

            try:
                chunk = self._sock.recv(65536)
            except TimeoutError:
                # ⚠️ 超时**不是**"对端关闭"，必须让调用方看见。
                #
                # ``socket.timeout`` 是 ``OSError`` 的子类，若被下面那条
                # 兜底捕获，就会返回 ``None``——而 ``None`` 的语义是
                # "连接断了"。两者的处置天差地别：超时可能只是"暂时没有
                # 消息"，断连则要重连。把它们混为一谈，就会得到
                # "安静一会儿就重连"这种极难排查的行为（真的发生过，
                # 见 :func:`connect_with_retry` 的注释）。
                raise
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                return None

            if not chunk:
                return None  # 对端正常关闭
            self._buf.extend(chunk)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def write_message_blocking(sock: socket.socket, msg: dict[str, Any]) -> bool:
    """同步写出一条消息。成功返回 True，连接已断返回 False。"""
    try:
        sock.sendall(encode(msg))
        return True
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
        return False


def connect_with_retry(
    host: str,
    port: int,
    attempts: int = _BIND_ATTEMPTS,
    timeout: float = 5.0,
) -> socket.socket:
    """带退避重试的**同步**连接。供 C 的事件循环外场景使用。

    ⚠️ **连上之后必须把 socket 的超时清掉（``settimeout(None)``）。**

    ``timeout`` 是**连接**超时，而 ``socket.create_connection`` 会把它
    设成这个 socket 的**读超时**并一直留着。于是：连接用 3 秒超时 →
    连上之后每一帧都必须 3 秒内到达，否则 ``recv`` 抛 ``socket.timeout``。

    而 ``socket.timeout`` 是 ``OSError`` 的子类，会被
    :meth:`LineReader.read_message` 的 recv 兜底捕获，**返回 ``None``**——
    而 ``None`` 的语义是"对端关闭"。这两件事叠起来就是一个很隐蔽的故障：

        B 本来就不会每一帧都说话。C 在安静期超过 3 秒 → 读循环拿到
        ``None`` → 认为"B 断开了" → 拆掉连接重连。

    表现是 C 每 3 秒重连一次，B 的日志里刷满"老人端已连接/已断开"，
    而**真正的代价**是重连的空窗期里 B 发出的播报会被丢掉——
    最可能被丢掉的恰恰是"您还好吧？"那一条。

    所以：连接超时归连接，读超时必须交给上层决定。这里清掉它。
    """
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(_BIND_BACKOFF_SEC * (2**attempt))
            continue
        sock.settimeout(None)
        return sock
    raise OSError(f"无法连接 {host}:{port}") from last_exc


async def connect_async_with_retry(
    host: str,
    port: int,
    attempts: int = _BIND_ATTEMPTS,
    timeout: float = 5.0,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """带退避重试的**异步**连接，返回 ``(reader, writer)``。

    与同步版并存而不是互相包装：asyncio 下用 ``time.sleep`` 会阻塞
    整个事件循环，把 B 的所有定时器一起卡住——在这个系统里，
    被卡住的定时器意味着无应答升级链不会推进。
    """
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=MAX_MESSAGE_BYTES),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            last_exc = exc  # type: ignore[assignment]
            if attempt < attempts - 1:
                await asyncio.sleep(_BIND_BACKOFF_SEC * (2**attempt))
    raise OSError(f"无法连接 {host}:{port}") from last_exc
