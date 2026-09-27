# -*- coding: utf-8 -*-
"""CSV 回放脚本：用 face_data.csv 模拟模块 A，把数据一帧帧发给后端 B。

    python tests/replay_script.py

--------------------------------------------------------------------------
为什么是「监听 8000」而不是「连 8001」
--------------------------------------------------------------------------
api_doc §3.1 规定 A→B 这条链路上 **模块 A 是服务端，模块 B 是客户端**。
所以一个"模拟模块 A"的脚本必须 listen 127.0.0.1:8000，等 B 主动连进来；
写成 connect 8001 是反的 —— 8001 是 B 推状态给 C 用的端口，B 在那儿是服务端，
从 8001 发数据等于假装自己是 B，而且 B 根本不会去连 8001。

--------------------------------------------------------------------------
CSV 字段 → api_doc §3.2 报文字段
--------------------------------------------------------------------------
CSV 只有三列 `timestamp,emotion,fatigue`，api_doc 要求的报文有 8 个字段，
中间靠一张映射表补齐：

    timestamp   → timestamp     原样透传
    emotion     → emo_feature   查 EMOTION_TO_FEATURE（只允许 normal/low/tired）
    fatigue     → ear           线性映射：0.0→0.35（清醒） … 1.0→0.10（困得睁不开眼）
    （无）      → has_face      emotion 取 absent/none 时为 false，否则 true
    （无）      → blink_cnt     恒为 0
    （无）      → pitch/yaw/roll 恒为 0.0

谁在编数据、谁没编，这里说清楚：
  - ear 是**有意**由 fatigue 推导的，两者本来就是同一件事的两种量纲；
    映射后疲劳度 0.6 以上会跌破 B 的 EAR 阈值 0.20，从而触发疲劳判定。
  - blink_cnt / pitch / yaw / roll 源数据里根本没有，所以填 0，
    **不编造**。代价是 B 的"眨眼频率"和"持续低头"两条疲劳判据不会被触发，
    tired 只会从 emo_feature 和 EAR 这两条路进来 —— 这已经够验证链路了。
    要连那两条判据一起测，用 data/sample_vision.csv（tools/make_sample_vision.py
    生成的完整格式数据）。

--------------------------------------------------------------------------
节奏
--------------------------------------------------------------------------
固定每行间隔 0.05 秒（= 20fps，模拟摄像头帧率），**不**按 CSV 里 timestamp
的差值走。所以如果 CSV 的时间戳间隔不是 0.05，回放出来的"数据时间"会跟
墙钟对不上。B 的窗口/防抖判定用的是收包时刻（墙钟），不受影响。
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from typing import List, Optional, Tuple

# 让 `python tests/replay_script.py` 和 `cd backend_B && python tests/replay_script.py`
# 两种方式都能 import 到 config / core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.protocol import encode_line

DEFAULT_CSV = os.path.join(config.DATA_DIR, "face_data.csv")

# 每帧间隔（秒）。0.05s = 20fps。
FRAME_INTERVAL = 0.05

# --------------------------------------------------------------------------
# 映射表
# --------------------------------------------------------------------------

# emotion 列 → api_doc 的 emo_feature（只允许这三个值）
EMOTION_TO_FEATURE = {
    "normal": "normal", "ok": "normal", "fine": "normal",
    "happy": "normal", "neutral": "normal", "calm": "normal",
    "tired": "tired", "fatigue": "tired", "sleepy": "tired", "drowsy": "tired",
    "sad": "low", "low": "low", "unhappy": "low", "down": "low", "upset": "low",
}

# 这些 emotion 取值表示"画面里没人脸"，转成 has_face=false
NO_FACE_EMOTIONS = frozenset({"absent", "none", "no_face", "noface", "miss"})

# fatigue → ear 的线性映射端点
# B 的 config.EAR_TIRED_THRESHOLD = 0.20，所以 fatigue > 0.6 就会跌破阈值
EAR_ALERT = 0.35       # fatigue = 0.0，精神饱满
EAR_EXHAUSTED = 0.10   # fatigue = 1.0，困得睁不开眼


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fatigue_to_ear(fatigue: float) -> float:
    """疲劳度 0~1 → 眼睑开合度。疲劳越高，眼睛睁得越小。"""
    if fatigue <= 0.0:
        return EAR_ALERT
    if fatigue >= 1.0:
        return EAR_EXHAUSTED
    return EAR_ALERT - fatigue * (EAR_ALERT - EAR_EXHAUSTED)


def to_payload(row: dict, warn) -> Optional[dict]:
    """CSV 一行 → api_doc §3.2 报文。字段缺失/非法时返回 None 并报警。"""
    try:
        timestamp = float(row.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        warn(f"timestamp 不是数字：{row.get('timestamp')!r}，跳过该行")
        return None

    emotion = (row.get("emotion") or "").strip().lower()

    # 有人脸才需要 fatigue；无人脸的行把 fatigue 当 0 处理
    raw_fatigue = row.get("fatigue")
    try:
        fatigue = clamp(float(raw_fatigue), 0.0, 1.0) if raw_fatigue not in (None, "") else 0.0
    except (TypeError, ValueError):
        warn(f"fatigue 不是数字：{raw_fatigue!r}，按 0 处理")
        fatigue = 0.0

    has_face = emotion not in NO_FACE_EMOTIONS

    # emotion 不认识时按 normal 走，但提醒一次，免得拼错列值却看不出来
    if emotion and emotion not in EMOTION_TO_FEATURE and has_face:
        warn(f"未知的 emotion 取值 {emotion!r}，按 normal 处理")
    emo_feature = EMOTION_TO_FEATURE.get(emotion, "normal")

    # 无人脸时按 api_doc §3.2 注释：其余字段填默认值，连接不断
    ear = fatigue_to_ear(fatigue) if has_face else 0.0

    return {
        "timestamp": timestamp,
        "has_face": has_face,
        "ear": round(ear, 4),
        "blink_cnt": 0,      # 源数据没有眨眼信息，不编
        "pitch": 0.0,        # 同上，头部姿态也拿不到
        "yaw": 0.0,
        "roll": 0.0,
        "emo_feature": emo_feature,
    }


# --------------------------------------------------------------------------
# CSV 读取
# --------------------------------------------------------------------------

def load_frames(csv_path: str) -> List[Tuple[dict, bytes]]:
    """读取 CSV，返回 [(原始行, 待发送的字节串)]。

    注意存的是 encode_line() 的**原始字节**，末尾的 \\n 是分帧标记，一个字节都不能少：
    B 端按 \\n 切分报文，少了它整条流会一直堵在缓冲区里，表现为"连上了但一帧都不处理"。

    文件不存在抛 FileNotFoundError，由 main() 统一处理成友好提示。
    """
    frames: List[Tuple[dict, bytes]] = []
    warned: set = set()

    def warn(message: str) -> None:
        # 同类问题只提醒一次，避免上千行里刷屏
        if message not in warned:
            warned.add(message)
            print(f"  ⚠ {message}")

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as fp:
        # utf-8-sig：兼容用 Excel 另存过、带 BOM 的 CSV
        reader = csv.DictReader(fp)
        if reader.fieldnames:
            missing = {"timestamp", "emotion", "fatigue"} - {
                name.strip().lstrip("﻿") for name in reader.fieldnames if name
            }
            if missing:
                print(f"  ⚠ CSV 缺少约定的列：{'、'.join(sorted(missing))}"
                      f"（现有列：{reader.fieldnames}）")

        for row in reader:
            payload = to_payload(row, warn)
            if payload is None:
                continue
            frames.append((row, encode_line(payload)))

    return frames


# --------------------------------------------------------------------------
# 回放
# --------------------------------------------------------------------------

def replay(frames: List[Tuple[dict, str]], host: str, port: int,
           interval: float, wait: float) -> int:
    """监听等待 B 连入，然后把 frames 逐帧发出去。返回进程退出码。"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ---- 绑定监听端口。失败通常是端口被占用或权限不够 ----
    try:
        server.bind((host, port))
        server.listen(1)
    except OSError as exc:
        print(f"\n✗ 无法监听 {host}:{port} —— {exc}")
        if getattr(exc, "errno", None) in (48, 98, 10048):
            print("  端口已被占用。先确认是不是已经在跑另一个模拟 A（或真的模块 A）：")
            print(f"    netstat -ano | findstr :{port}")
        else:
            print("  请检查地址是否合法、端口是否在 1024 以上。")
        server.close()
        return 1

    print(f"✓ 已监听 {host}:{port}（模块 A 的角色是服务端，等 B 连进来）")

    # ---- 等待 B 连接 ----
    if wait > 0:
        server.settimeout(wait)
    print(f"  等待后端 B 连接{'（最多 %.0f 秒）' % wait if wait > 0 else '（Ctrl+C 取消）'} ...")

    try:
        conn, addr = server.accept()
    except socket.timeout:
        print(f"\n✗ 等待 {wait:.0f} 秒仍没有客户端连入，退出。")
        print(f"  请确认后端 B 已启动：cd backend_B && python main.py")
        print(f"  B 会主动连 {host}:{port}，如果它还在重试，把本脚本先跑起来即可。")
        server.close()
        return 1
    except KeyboardInterrupt:
        print("\n已取消。")
        server.close()
        return 130
    except OSError as exc:
        print(f"\n✗ 接受连接失败：{exc}")
        server.close()
        return 1

    print(f"✓ 后端 B 已连入（来自 {addr[0]}:{addr[1]}）")
    print(f"  开始回放 {len(frames)} 帧，间隔 {interval * 1000:.0f}ms\n")

    # 关掉 Nagle：小包高频场景下攒包只会平白增加延迟
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    sent = 0
    started = time.monotonic()
    next_at = started
    exit_code = 0

    try:
        for _row, line in frames:
            # 按绝对时刻排程，而不是每次 sleep(interval)：
            # 后者会把发送耗时累加进去，帧率越跑越慢。
            next_at += interval
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            try:
                conn.sendall(line)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
                print(f"\n✗ 第 {sent + 1} 帧发送时连接被对端断开（{exc.__class__.__name__}）")
                print("  后端 B 可能已退出或被 Ctrl+C 中断。回放中止。")
                exit_code = 1
                break
            except OSError as exc:
                print(f"\n✗ 第 {sent + 1} 帧发送失败：{exc}")
                exit_code = 1
                break

            sent += 1
            if sent % 100 == 0:
                elapsed = time.monotonic() - started
                print(f"  已发送 {sent}/{len(frames)} 帧"
                      f"（{elapsed:.1f}s，{sent / elapsed:.1f} fps）")

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，回放中止。")
        exit_code = 130

    elapsed = time.monotonic() - started
    if sent == len(frames):
        print(f"\n✓ 回放完成：{sent} 帧，用时 {elapsed:.1f}s"
              f"（平均 {sent / elapsed:.1f} fps）")

    # 正常收尾：关掉写端让 B 读到 EOF，它会立刻进入重连等待。
    # 这里主动发一个 shutdown 而不是直接 close，避免 B 报 ConnectionReset。
    try:
        conn.shutdown(socket.SHUT_WR)
    except OSError:
        pass
    conn.close()
    server.close()
    return exit_code


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_script.py",
        description="读取 face_data.csv，模拟模块 A 通过 Socket 把视觉数据发给后端 B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tests/replay_script.py\n"
            "  python tests/replay_script.py --csv data/face_data.csv --interval 0.05\n"
            "  python tests/replay_script.py --host 127.0.0.1 --port 8000\n"
        ),
    )
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"要回放的 CSV（默认 {DEFAULT_CSV}）")
    parser.add_argument("--host", default=config.VISION_HOST,
                        help=f"监听地址（默认 {config.VISION_HOST}）")
    parser.add_argument("--port", type=int, default=config.VISION_PORT,
                        help=f"监听端口（默认 {config.VISION_PORT}，即 api_doc 的 A→B 端口）")
    parser.add_argument("--interval", type=float, default=FRAME_INTERVAL,
                        help=f"每帧间隔秒数（默认 {FRAME_INTERVAL}，即 20fps）")
    parser.add_argument("--wait", type=float, default=30.0,
                        help="等待 B 连入的秒数，0 表示一直等（默认 30）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Windows 控制台默认可能是 GBK，日志里有中文会乱码
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    print("=" * 62)
    print("  CSV 回放脚本 · 模拟模块 A")
    print("=" * 62)
    print(f"  数据文件    {args.csv}")
    print(f"  监听地址    {args.host}:{args.port}")
    print(f"  帧间隔      {args.interval * 1000:.0f}ms")
    print("-" * 62)

    # ---- CSV 不存在：打印提示后优雅退出，不留 traceback ----
    if not os.path.isfile(args.csv):
        print(f"\n✗ 找不到 CSV 文件：{args.csv}")
        print("  这个脚本需要一个包含 timestamp,emotion,fatigue 三列的文件。")
        print("  请用 --csv 指定路径，例如：")
        print(f"    python tests/replay_script.py --csv data/face_data.csv")
        return 1

    print("  正在读取 CSV ...")
    try:
        frames = load_frames(args.csv)
    except FileNotFoundError:
        # 上面查过一次，这里兜住"查到之后又被删掉"的竞态
        print(f"\n✗ 找不到 CSV 文件：{args.csv}")
        return 1
    except PermissionError:
        print(f"\n✗ 没有权限读取：{args.csv}")
        print("  文件可能被 Excel 打开了，关掉再试。")
        return 1
    except UnicodeDecodeError as exc:
        print(f"\n✗ CSV 编码无法识别：{exc}")
        print("  请把文件另存为 UTF-8 编码。")
        return 1
    except OSError as exc:
        print(f"\n✗ 读取 CSV 失败：{exc}")
        return 1

    if not frames:
        print(f"\n✗ CSV 里没有可用的数据行：{args.csv}")
        print("  表头需要是 timestamp,emotion,fatigue，且至少有一行有效数据。")
        return 1

    print(f"✓ 读到 {len(frames)} 帧（约 {len(frames) * args.interval:.1f} 秒）\n")

    try:
        return replay(frames, args.host, args.port, args.interval, args.wait)
    except KeyboardInterrupt:
        print("\n已取消。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
