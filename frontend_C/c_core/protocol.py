# -*- coding: utf-8 -*-
"""TCP 报文分帧与编解码。

⚠️ **这是 ``backend_B/core/protocol.py`` 的一个刻意副本，不是疏忽。**

模块 C 不能 import 模块 B 的源码，理由有三条：

1. 三个模块之间**只许走 Socket**（api_doc §1）。C 直接 import B 的包，
   等于在架构上把两个模块粘死，B 换语言或 C 换机器都立刻崩。
2. C 必须能单独运行 —— 答辩时可能只把 frontend_C/ 拷到演示机上跑。
3. C 的单元测试不该要求 ``backend_B`` 出现在 ``sys.path`` 上。

同步风险很低：分帧规则由 api_doc §2.1 冻结（``\\n`` 分隔），不会再变。
如果哪天真的改了分帧规则，**两个文件必须一起改**。

api_doc §2.1：所有接口数据以换行符 ``\\n`` 作为数据包分隔，防止粘包。
TCP 是字节流，不保证「发一次 = 收一次」，所以必须自己按 ``\\n`` 切分并缓存残包。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional


class ProtocolError(Exception):
    """报文不符合约定格式时抛出。"""


def encode_line(obj: Any) -> bytes:
    """把 Python 对象编码成 ``JSON + \\n`` 的字节串。"""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def decode_json_line(line: str) -> Dict[str, Any]:
    """解析一行 JSON。解析失败或顶层不是对象时抛 ProtocolError。

    C 端不允许因为 B 发了一条脏数据就崩掉，调用方应捕获并跳过该条。
    """
    line = line.strip()
    if not line:
        raise ProtocolError("空行不是合法报文")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"JSON 解析失败: {exc.msg} | 原文={line[:200]!r}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"报文顶层必须是 JSON 对象，实际是 {type(obj).__name__}")
    return obj


class LineBuffer:
    """按 ``\\n`` 切分的接收缓冲区，解决粘包/半包问题。

    用法::

        buf = LineBuffer()
        for line in buf.feed(sock.recv(4096)):
            ...  # line 是不含 \\n 的完整一行

    不完整的尾巴留在内部，等下次 feed。
    """

    #: 单行长度上限，防止对端不发 \n 导致内存被撑爆
    MAX_LINE_BYTES = 1024 * 1024  # 1 MB

    def __init__(self, encoding: str = "utf-8") -> None:
        self._buf = b""
        self._encoding = encoding

    def feed(self, chunk: bytes) -> Iterator[str]:
        """喂入一段刚收到的字节，产出其中所有完整行（已解码为 str，无换行符）。"""
        if not chunk:
            return
        self._buf += chunk

        if len(self._buf) > self.MAX_LINE_BYTES and b"\n" not in self._buf:
            raise ProtocolError(
                f"单行超过 {self.MAX_LINE_BYTES} 字节仍未出现分隔符 \\n，丢弃缓冲区"
            )

        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            # 兼容对端误发 \r\n
            raw = raw.rstrip(b"\r")
            if not raw:
                continue  # 空行直接忽略
            yield self._decode(raw)

    def _decode(self, raw: bytes) -> str:
        """解码一行；非法 UTF-8 用替换字符兜底，不让解码错误中断整个连接。"""
        try:
            return raw.decode(self._encoding)
        except UnicodeDecodeError:
            return raw.decode(self._encoding, errors="replace")

    @property
    def pending(self) -> bytes:
        """缓冲区里还没拼成完整行的残留字节（调试用）。"""
        return self._buf

    def clear(self) -> None:
        """重连时清空残留，避免上一条连接的半包污染新连接。"""
        self._buf = b""


def first_of(obj: Dict[str, Any], keys, default: Optional[Any] = None) -> Any:
    """按顺序取第一个存在且非 None 的键，用于兼容字段别名。"""
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default
