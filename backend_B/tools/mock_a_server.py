# -*- coding: utf-8 -*-
"""模拟模块 A（视觉后端）—— 临时代替真 A，让 B 能先联调起来。

关键点：api_doc §3.1 规定 **A 是服务端、B 是客户端**，所以这里是 TCP 服务端，
监听 8000 等 B 连过来，而不是反过来。别写反了。

除了常规发数据，还内置了两个故障演练开关，专门用来验证 B 的容错：
    --drop-after N   发满 N 帧后主动断开，检验 B 的断线重连
    --refuse SEC     启动后先不监听 SEC 秒，检验 B 连不上时的退避重试

用法：
    python tools/mock_a_server.py                          # 回放 data/sample_vision.csv
    python tools/mock_a_server.py --drop-after 100         # 发 100 帧后断线
    python tools/mock_a_server.py --refuse 10              # 前 10 秒拒绝连接
    python tools/mock_a_server.py --speed 5                # 5 倍速
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.protocol import encode_line

DEFAULT_DELAY = 0.1   # 默认 10Hz


def load_frames(csv_path: str):
    """读 CSV 里的帧，返回 (timestamp, payload_dict) 列表。

    除了 CSV 里有的字段，再补一个 timestamp —— api_doc 要求它是
    "程序运行时间戳"，所以这里发之前会重新按当前时刻覆盖。
    """
    frames = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            frame = dict(row)
            frame["_ts"] = float(row.get("timestamp") or 0.0)
            frames.append(frame)
    return frames


def to_payload(frame: dict, base_ts: float) -> dict:
    """把 CSV 一行转成符合 api_doc §3.2 的 JSON 对象。

    字段类型要转对：A 端真实实现里 has_face 是 bool、ear/pitch/yaw/roll 是 float、
    blink_cnt 是 int。CSV 里全是字符串，这里做一次还原。
    """
    return {
        "timestamp": round(base_ts + frame["_ts"], 3),
        "has_face": str(frame.get("has_face", "false")).strip().lower() == "true",
        "ear": float(frame.get("ear") or 0.0),
        "blink_cnt": int(float(frame.get("blink_cnt") or 0)),
        "pitch": float(frame.get("pitch") or 0.0),
        "yaw": float(frame.get("yaw") or 0.0),
        "roll": float(frame.get("roll") or 0.0),
        "emo_feature": str(frame.get("emo_feature") or "normal").strip(),
    }


def serve(args) -> int:
    frames = load_frames(args.csv)
    print(f"已载入 {len(frames)} 帧：{args.csv}")

    if args.refuse > 0:
        print(f"[故障演练] 先拒绝连接 {args.refuse}s，用于验证 B 的退避重连 ...")
        time.sleep(args.refuse)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"模拟模块 A 已监听 {args.host}:{args.port}，等待 B 连入 ...")

    rounds = 0
    try:
        while True:
            client, addr = server.accept()
            rounds += 1
            print(f"\n[第 {rounds} 次连接] B 已接入 {addr[0]}:{addr[1]}")
            try:
                sent = stream_frames_looping(client, frames, args) if args.loop \
                    else stream_frames(client, frames, args)
                print(f"本轮发送 {sent} 帧")
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print(f"B 断开了连接：{exc.__class__.__name__}")

            try:
                client.close()
            except OSError:
                pass

            if not args.loop:
                break
            print("等待 B 重连 ...")
    except KeyboardInterrupt:
        print("\n模拟 A 已停止")
    finally:
        server.close()
    return 0


def stream_frames(client: socket.socket, frames, args) -> int:
    """按 CSV 的时间轴把帧发给 B。"""
    base_ts = time.time()
    sent = 0
    previous = None

    for frame in frames:
        if args.drop_after and sent >= args.drop_after:
            print(f"[故障演练] 已达 --drop-after={args.drop_after}，主动断开")
            return sent

        if previous is not None:
            delta = frame["_ts"] - previous
            if not (0 < delta <= 5.0):
                delta = DEFAULT_DELAY
            time.sleep(delta / args.speed)
        previous = frame["_ts"]

        client.sendall(encode_line(to_payload(frame, base_ts)))
        sent += 1

        if sent % 100 == 0:
            print(f"  已发送 {sent}/{len(frames)} 帧")

    return sent


def stream_frames_looping(client: socket.socket, frames, args) -> int:
    """循环模式：**保持连接不挂断**，一轮播完接着播下一轮。

    这点很重要：真实的视觉模块是持续推流的，不会每隔 40 秒挂断一次。
    如果这里每轮都断开重连，B 会不停地打印重连日志、状态在 absent 和正常之间
    反复跳，把真正要观察的对话逻辑淹掉。要演练重连请用 --drop-after。
    """
    from core.protocol import encode_line as _encode

    sent = 0
    # 虚拟时钟：每轮播完把 offset 往前推一轮的时长，保证跨轮次 timestamp 严格递增。
    # 否则第二轮又从 0 开始，时间轴倒退 —— B 端会（正确地）判定数据源重置过，
    # 从而放弃眨眼频率统计，看起来就像"循环播放之后疲劳检测失效了"。
    offset = 0.0
    while True:
        base_ts = time.time()
        previous = None
        for frame in frames:
            if previous is not None:
                delta = frame["_ts"] - previous
                if not (0 < delta <= 5.0):
                    delta = DEFAULT_DELAY
                time.sleep(delta / args.speed)
            previous = frame["_ts"]

            if args.drop_after and sent >= args.drop_after:
                print(f"[故障演练] 已达 --drop-after={args.drop_after}，主动断开")
                return sent

            payload = to_payload(frame, base_ts)
            payload["timestamp"] = round(base_ts + offset + frame["_ts"], 3)
            client.sendall(_encode(payload))
            sent += 1
            if sent % 500 == 0:
                print(f"  已发送 {sent} 帧（循环中）")

        offset += (frames[-1]["_ts"] if frames else 0.0) + DEFAULT_DELAY
        print(f"  完成一轮（累计 {sent} 帧），继续下一轮")


def main() -> int:
    # Windows 控制台默认可能是 GBK，日志里有中文会乱码（同 main.py / watch_8002.py）
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="模拟模块 A（TCP 服务端，发视觉 JSON）")
    parser.add_argument("--csv", default=config.SAMPLE_VISION_CSV,
                        help=f"要回放的 CSV（默认 {config.SAMPLE_VISION_CSV}）")
    parser.add_argument("--host", default=config.VISION_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=config.VISION_PORT,
                        help=f"监听端口（默认 {config.VISION_PORT}）")
    parser.add_argument("--speed", type=float, default=1.0, help="回放倍速")
    parser.add_argument("--loop", action="store_true", help="循环，允许 B 反复重连")
    parser.add_argument("--drop-after", type=int, default=0,
                        help="发满 N 帧后主动断开，用于演练 B 的断线重连")
    parser.add_argument("--refuse", type=float, default=0.0,
                        help="启动后先拒绝连接 N 秒，用于演练 B 的退避重试")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV 不存在：{args.csv}")
        print("先运行：python tools/make_sample_vision.py")
        return 1

    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
