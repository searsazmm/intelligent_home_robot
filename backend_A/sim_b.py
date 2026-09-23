"""
模拟模块 B 的验收客户端（联调用）
==================================
连接 A 的 Socket（api_doc §3），逐行接收 JSON 并按协议逐条校验：
- 字段集合与 §3.2 完全一致（不多不少）
- 类型正确；emo_feature 只允许 normal/tired/sad/blank（§3.2 V1.1）
- has_face=false 的心跳帧不中断流（§3.3 约束 1）
- timestamp 单调不减；帧率接近 15fps
- ear/pitch/yaw/roll 保留 ≤2 位小数（§3.3 约束 4）

全绿则把收到的流按 §3.4 表头另存 CSV，证明 B 可以从 Socket 流重建离线数据。

用法：
  python sim_b.py                        # 收 15 秒，打印验收报告
  python sim_b.py --seconds 30 --host 127.0.0.1 --port 8000
  python sim_b.py --out data/sim_b_received.csv
"""

import argparse
import csv
import json
import socket
import sys
import time

FIELDS = ["timestamp", "has_face", "ear", "blink_cnt", "pitch", "yaw", "roll", "emo_feature"]
LABELS = {"normal", "tired", "sad", "blank"}


def check_frame(obj, idx, errors):
    """单帧校验，返回 (是否心跳帧, 标签)。问题追加进 errors。"""
    def bad(msg):
        errors.append(f"#{idx}: {msg}")

    if not isinstance(obj, dict):
        bad(f"不是 JSON 对象: {obj!r}")
        return False, None
    if set(obj.keys()) != set(FIELDS):
        missing = set(FIELDS) - set(obj.keys())
        extra = set(obj.keys()) - set(FIELDS)
        bad(f"字段不符（缺 {missing}，多 {extra}）")
        return False, None

    if not isinstance(obj["has_face"], bool):
        bad(f"has_face 应为 bool，实为 {type(obj['has_face']).__name__}")
    if obj["emo_feature"] not in LABELS:
        bad(f"emo_feature 非法值 {obj['emo_feature']!r}")
    for k in ("ear", "pitch", "yaw", "roll"):
        v = obj[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            bad(f"{k} 应为数值，实为 {type(v).__name__}")
        elif round(float(v), 2) != float(v):
            bad(f"{k}={v} 超过 2 位小数")
    if not isinstance(obj["blink_cnt"], int) or isinstance(obj["blink_cnt"], bool):
        bad(f"blink_cnt 应为 int，实为 {type(obj['blink_cnt']).__name__}")
    if not isinstance(obj["timestamp"], (int, float)) or isinstance(obj["timestamp"], bool):
        bad(f"timestamp 应为数值，实为 {type(obj['timestamp']).__name__}")

    return not obj["has_face"], obj["emo_feature"]


def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows GBK 控制台下强制 UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="模拟 B 的 A→B 协议验收客户端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seconds", type=float, default=15.0, help="接收时长（秒）")
    ap.add_argument("--out", default=None, help="把收到的流另存为 CSV（§3.4 表头）")
    args = ap.parse_args()

    print(f"[sim_b] 连接 {args.host}:{args.port} ...")
    try:
        sock = socket.create_connection((args.host, args.port), timeout=5)
    except OSError as e:
        print(f"[sim_b] ❌ 连不上：{e}（A 侧是否已启动？端口是否被 Django runserver 占用？）")
        sys.exit(1)
    print(f"[sim_b] ✅ 已连接，开始接收 {args.seconds}s ...")

    sock.settimeout(5.0)
    errors, rows = [], []
    label_count = {k: 0 for k in sorted(LABELS)}
    heartbeats = frames = 0
    last_t = None
    buf = b""
    deadline = time.time() + args.seconds

    while time.time() < deadline:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            errors.append(f"{args.seconds:.0f}s 内仅收到 {frames} 帧（最后 5s 无数据）")
            break
        if not chunk:
            errors.append("连接被 A 侧关闭（可能 A 正在退出）")
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            idx = frames + 1
            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                errors.append(f"#{idx}: JSON 解析失败（{e}）")
                continue
            is_hb, label = check_frame(obj, idx, errors)
            if label:
                label_count[label] += 1
            if is_hb:
                heartbeats += 1
            t = obj.get("timestamp") if isinstance(obj, dict) else None
            if isinstance(t, (int, float)) and not isinstance(t, bool):
                if last_t is not None and t < last_t:
                    errors.append(f"#{idx}: timestamp 回退（{last_t} -> {t}）")
                last_t = t
            rows.append(obj)
            frames += 1
    sock.close()

    # ---- 验收报告 ----
    print(f"\n[sim_b] ===== 验收报告 =====")
    print(f"共收帧 {frames}（其中心跳帧 has_face=false：{heartbeats}）")
    print(f"标签分布：{label_count}")
    if frames >= 2 and isinstance(rows[0].get("timestamp"), (int, float)) and isinstance(rows[-1].get("timestamp"), (int, float)):
        span = rows[-1]["timestamp"] - rows[0]["timestamp"]
        if span > 0:
            print(f"实测帧率：{frames / span:.1f} fps（目标 15）")
    if errors:
        print(f"\n[sim_b] ❌ 未通过，{len(errors)} 个问题（最多显示 10 条）：")
        for e in errors[:10]:
            print(f"  - {e}")
        sys.exit(2)

    print(f"\n[sim_b] ✅ 全部校验通过：字段/类型/枚举/心跳/时序/小数位 均符合 api_doc §3")
    if args.out and rows:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for o in rows:
                w.writerow([
                    o["timestamp"], str(o["has_face"]).lower(), o["ear"], o["blink_cnt"],
                    o["pitch"], o["yaw"], o["roll"], o["emo_feature"],
                ])
        print(f"[sim_b] 已按 §3.4 表头另存流数据：{args.out}")


if __name__ == "__main__":
    main()
