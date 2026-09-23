# -*- coding: utf-8 -*-
"""TCP 报文分帧与编解码。

api_doc §2.1：所有接口数据以换行符 `\\n` 作为数据包分隔，防止粘包。

TCP 是字节流，不保证"发一次 = 收一次"。A 端如果连续快速发 3 条 JSON，
B 端 recv 可能一次读到 3 条粘在一起，也可能只读到半条。所以必须自己按
`\\n` 切分并缓存残包 —— 本模块的 LineBuffer 就是干这个的。
"""

import json
from typing import Any, Dict, Iterator, Optional


class ProtocolError(Exception):
    """报文不符合约定格式时抛出。"""


def encode_line(obj: Any) -> bytes:
    """把 Python 对象编码成 `JSON + \\n` 的字节串，用于发送。

    ensure_ascii=False 让中文以 UTF-8 原文传输（可读性好，且 api_doc 未要求转义）；
    separators 去掉多余空格，减小包体。
    """
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def encode_text_line(text: str) -> bytes:
    """把纯文本编码成 `文本 + \\n`（api_doc §4 要求 B→C 发纯文本状态串）。"""
    return (text + "\n").encode("utf-8")


def decode_json_line(line: str) -> Dict[str, Any]:
    """解析一行 JSON。解析失败或顶层不是对象时抛 ProtocolError。

    B 端不允许因为 A 发了一条脏数据就崩掉，调用方应捕获并跳过该条。
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
    """按 `\\n` 切分的接收缓冲区，解决粘包/半包问题。

    用法：
        buf = LineBuffer()
        for line in buf.feed(sock.recv(4096)):
            ...  # line 是不含 \\n 的完整一行

    feed() 返回本次能拼出的所有完整行；不完整的尾巴留在内部，等下次 feed。
    """

    # 单行长度上限，防止对端不发 \\n 导致内存被撑爆
    MAX_LINE_BYTES = 1024 * 1024  # 1 MB

    def __init__(self, encoding: str = "utf-8") -> None:
        self._buf = b""
        self._encoding = encoding

    def feed(self, chunk: bytes) -> Iterator[str]:
        """喂入一段刚收到的字节，产出其中所有完整行（已解码为 str，无换行符）。"""
        if not chunk:
            return
        self._buf += chunk

        # 防止恶意/异常的超长行
        if len(self._buf) > self.MAX_LINE_BYTES and b"\n" not in self._buf:
            raise ProtocolError(
                f"单行超过 {self.MAX_LINE_BYTES} 字节仍未出现分隔符 \\n，丢弃缓冲区"
            )

        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            # 兼容对端误发 \r\n（Windows 上很容易出现）
            raw = raw.rstrip(b"\r")
            if not raw:
                continue  # 空行（比如对端发了 \n\n）直接忽略
            yield self._decode(raw)

    def _decode(self, raw: bytes) -> str:
        """解码一行字节；非法 UTF-8 用替换字符兜底，不让解码错误中断整个连接。"""
        try:
            return raw.decode(self._encoding)
        except UnicodeDecodeError:
            return raw.decode(self._encoding, errors="replace")

    @property
    def pending(self) -> bytes:
        """当前缓冲区里还没拼成完整行的残留字节（调试用）。"""
        return self._buf

    def clear(self) -> None:
        """重连时清空残留，避免上一条连接的半包污染新连接。"""
        self._buf = b""


def first_of(obj: Dict[str, Any], keys, default: Optional[Any] = None) -> Any:
    """按顺序取第一个存在且非 None 的键，用于兼容字段别名。

    例如 A 端可能把 emo_feature 写成 emotion_feature，这里做兼容读取，
    但 api_doc 规定的字段名永远是首选。
    """
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default
