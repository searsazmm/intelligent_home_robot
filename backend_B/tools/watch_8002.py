# -*- coding: utf-8 -*-
"""被动监听 8002，把 B 发出来的每一条报文原样打出来。

    python tools/watch_8002.py                # 听 30 秒
    python tools/watch_8002.py --seconds 90   # 听久一点（等主动关怀触发用）
    python tools/watch_8002.py --chat "你好"  # 顺便发一句话

--------------------------------------------------------------------------
为什么需要它（和 mock_c_client.py 的区别）
--------------------------------------------------------------------------
``mock_c_client.py`` 是**主动**的：它会发一串对话，用来验证 "C→B→C" 的回环。

但要看**主动关怀**就不能用它：

  1. 它发完预设对话就退出，而主动关怀往往在它退出之后才触发；
  2. 更根本的是 —— 它每发一句话，都会刷新 B 的 ``PROACTIVE_USER_COOLDOWN``
     （默认 60 秒）。**机器人在这 60 秒内不会主动开口，这是刻意的**
     （用户刚说完话就抢话头是最糟的失败模式）。

所以想验证 ``proactive`` 报文，得用一个**一句话都不说**的监听者。
本工具默认只发一个 ping（确认链路通），然后安静地听。

--------------------------------------------------------------------------
它同时也是排错工具
--------------------------------------------------------------------------
「C 的表情不动」这类问题的第一步永远是：
**B 到底发了没有？** 用本工具看 8002，再对照 8001，就能立刻区分
是 B 没发、还是 C 没收到、还是 C 收到了没画。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.protocol import LineBuffer, ProtocolError, decode_json_line, encode_line  # noqa: E402

#: 报文类型 → 显示前缀。没有 emoji，Windows 控制台编码不可靠。
LABELS = {
    "state": "状态",
    "reply": "回复",
    "proactive": "★主动开口",
    "pong": "心跳",
    "error": "错误",
}


def describe(payload: dict) -> str:
    kind = str(payload.get("type") or "?")
    label = LABELS.get(kind, kind)
    text = payload.get("text")
    state = payload.get("state", "")

    if kind == "proactive":
        # 主动关怀是本次要验证的主角，多打两行
        return (f"{label}  kind={payload.get('kind', '?')}  state={state}\n"
                f"          「{text}」\n"
                f"          reason: {payload.get('reason', '')}")
    if text:
        return f"{label}  [{state}] 「{text}」"
    if kind == "state":
        return f"{label}  {state}  ({payload.get('reason', '')})"
    return f"{label}  {json.dumps(payload, ensure_ascii=False)}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="watch_8002",
        description="被动监听模块 B 的 8002 对话端口，原样打印所有报文",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="听多久后退出（默认 30 秒；0 = 一直听直到 Ctrl+C）")
    parser.add_argument("--chat", default="",
                        help="可选：连上后发一句话（注意会触发 60 秒主动关怀静默）")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    try:
        sock = socket.create_connection((args.host, args.port), timeout=3.0)
    except OSError as exc:
        print(f"连不上 {args.host}:{args.port} —— {exc}")
        print("  B 起了吗？端口对吗？（--port）")
        return 1

    sock.settimeout(0.5)
    print(f"已连上 {args.host}:{args.port}，开始监听"
          f"{'（%.0f 秒）' % args.seconds if args.seconds else '（Ctrl+C 退出）'}...")
    if args.chat:
        print(f"  ⚠️ 将发送一句话：「{args.chat}」")
        print("     这会刷新 B 的 60 秒 user_cooldown，主动关怀在此期间不会触发。")
    print()

    buffer = LineBuffer()
    deadline = time.monotonic() + args.seconds if args.seconds else None
    sent_chat = False
    proactive_seen = 0

    try:
        while deadline is None or time.monotonic() < deadline:
            if not sent_chat:
                # 默认只发一个 ping：确认链路通，又不触发 user_cooldown
                first = ({"type": "chat", "text": args.chat} if args.chat
                         else {"type": "ping"})
                sock.sendall(encode_line(first))
                sent_chat = True

            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                print(f"[连接中断] {exc}")
                break

            if not chunk:
                print("[B 关闭了连接]")
                break

            try:
                for line in buffer.feed(chunk):
                    try:
                        payload = decode_json_line(line)
                    except ProtocolError as exc:
                        print(f"[非法报文] {exc}: {line[:80]}")
                        continue
                    if payload.get("type") == "proactive":
                        proactive_seen += 1
                    stamp = time.strftime("%H:%M:%S")
                    print(f"[{stamp}] {describe(payload)}")
            except ProtocolError as exc:
                print(f"[分帧错误] {exc}")
                buffer.clear()
    except KeyboardInterrupt:
        print("\n[Ctrl+C]")
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    print()
    if proactive_seen:
        print(f"✓ 共收到 {proactive_seen} 条 proactive 报文")
    else:
        print("✗ 没有收到 proactive 报文。可能原因：\n"
              "    · 刚发过 chat（user_cooldown 默认 60 秒内不主动开口）\n"
              "    · 视觉状态一直是 normal（sad/tired 持续够久才会关心）\n"
              "    · 处于启动宽限期或 22:00–07:00 静默时段\n"
              "    · 离线回放倍速过高，没加 --demo（阈值没跟着缩放会永远不触发）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
