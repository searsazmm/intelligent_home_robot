# -*- coding: utf-8 -*-
"""生成离线调试用的视觉 CSV（模拟模块 A 导出的数据）。

api_doc §5.1：开发阶段模块 A 导出标准 CSV，模块 B 开离线模式读文件并行开发。
模块 A 还没写好时，用这个脚本造一份格式完全一致的样例数据，B 就能单方面跑通。

生成的时间线（10Hz，共 120 秒），覆盖全部 4 种状态：

    0 – 20s    normal   正常坐着聊天
    20 – 50s   tired    EAR 逐渐下降、开始低头、眨眼变频繁（疲劳是渐进的，不是突变）
    50 – 58s   absent   人脸从画面消失
    58 – 80s   normal   人回来了，恢复
    80 – 100s  sad      A 端给出 emo_feature=low
    100 – 120s normal

用法：
    python tools/make_sample_vision.py                  # 写到 data/sample_vision.csv
    python tools/make_sample_vision.py --out other.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# CSV 表头必须与 api_doc §3.2 的字段一字不差
FIELDS = ["timestamp", "has_face", "ear", "blink_cnt", "pitch", "yaw", "roll", "emo_feature"]

SAMPLE_RATE = 10.0          # 每秒 10 帧，和摄像头常见帧率一致
DURATION = 120.0            # 总时长（秒）

# 各个阶段的定义：(结束时间, 阶段名)
# 顺序必须按时间递增，_phase_at() 靠它判断当前处于哪一段
PHASES = [
    (20.0, "normal"),
    (50.0, "tired"),
    (58.0, "absent"),
    (80.0, "normal2"),
    (100.0, "sad"),
    (120.0, "normal3"),
]

# 每个阶段的视觉特征基准值
PROFILES = {
    #                    ear    pitch  yaw   roll  眨眼/分  A端特征
    "normal":  dict(ear=0.310, pitch=0.0, yaw=0.0, roll=0.0, blink=18.0, feature="normal"),
    "normal2": dict(ear=0.305, pitch=1.0, yaw=1.0, roll=0.0, blink=18.0, feature="normal"),
    "normal3": dict(ear=0.300, pitch=0.0, yaw=0.0, roll=0.0, blink=17.0, feature="normal"),
    # 疲劳：EAR 压到阈值 0.20 以下，低头超过 20°，眨眼变频繁
    "tired":   dict(ear=0.140, pitch=26.0, yaw=2.0, roll=1.0, blink=38.0, feature="tired"),
    # 无人：字段填默认值，但照常发包（api_doc §3.3.1）
    "absent":  dict(ear=0.0, pitch=0.0, yaw=0.0, roll=0.0, blink=0.0, feature="normal"),
    # 低落：A 端给出 low 特征
    "sad":     dict(ear=0.290, pitch=3.0, yaw=-4.0, roll=0.0, blink=20.0, feature="low"),
}


def phase_at(t: float) -> str:
    """返回 t 时刻所处的阶段名。"""
    for end, name in PHASES:
        if t < end:
            return name
    return PHASES[-1][1]


def transition_ratio(t: float, ramp: float = 6.0) -> float:
    """疲劳是渐进的：进入 tired 阶段后 ramp 秒内从 0 平滑过渡到 1。

    真实数据不会是阶跃的，做平滑过渡才能验证 B 端的"持续 N 秒才判定"逻辑确实生效。
    """
    tired_start = PHASES[0][0]           # 20.0s，tired 阶段开始
    if t <= tired_start:
        return 0.0
    return min((t - tired_start) / ramp, 1.0)


def lerp(a: float, b: float, ratio: float) -> float:
    return a + (b - a) * ratio


def generate(out_path: str, seed: int = 20260923) -> int:
    """生成 CSV，返回写出的行数。

    固定随机种子，保证每次生成的样例文件一模一样 —— 联调时大家看到的
    数据一致，才能对着同一份现象讨论问题。
    """
    rng = random.Random(seed)
    rows = []
    blink_count = 0
    # 用"下一次眨眼的时间"来驱动计数，比按概率掷骰子更接近真实眨眼规律
    next_blink_at = 0.0

    total_frames = int(DURATION * SAMPLE_RATE)

    for index in range(total_frames):
        t = index / SAMPLE_RATE
        phase = phase_at(t)
        profile = PROFILES[phase]

        if phase == "absent":
            # 无人脸：照常发包，字段填默认值
            rows.append(dict(
                timestamp=round(t, 3), has_face="false", ear=0.0,
                blink_cnt=blink_count, pitch=0.0, yaw=0.0, roll=0.0,
                emo_feature="normal",
            ))
            continue

        ratio = transition_ratio(t) if phase == "tired" else 0.0

        # EAR：从正常值平滑降到疲劳值，再叠加一点抖动
        if phase == "tired":
            ear_base = lerp(PROFILES["normal"]["ear"], profile["ear"], ratio)
        else:
            ear_base = profile["ear"]
        ear = ear_base + rng.gauss(0, 0.012)
        ear = max(0.0, min(ear, 0.45))

        # 眨眼：E（每分钟次数）→ 平均间隔秒数
        blink_per_min = profile["blink"]
        if phase == "tired":
            blink_per_min = lerp(PROFILES["normal"]["blink"], profile["blink"], ratio)
        mean_interval = 60.0 / max(blink_per_min, 1.0)

        if t >= next_blink_at:
            blink_count += 1
            next_blink_at = t + rng.expovariate(1.0 / mean_interval)

        # 头部角度：基准值 + 缓慢的正弦摆动 + 抖动，模拟自然的小幅晃动
        pitch = profile["pitch"] + math.sin(t * 0.7) * 1.5 + rng.gauss(0, 0.8)
        yaw = profile["yaw"] + math.sin(t * 0.4) * 3.0 + rng.gauss(0, 1.2)
        roll = profile["roll"] + rng.gauss(0, 0.6)

        rows.append(dict(
            timestamp=round(t, 3),
            has_face="true",
            ear=round(ear, 4),
            blink_cnt=blink_count,
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            emo_feature=profile["feature"],
        ))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # newline="" 是 csv 模块在 Windows 上的要求，否则会多出空行
    # encoding="utf-8-sig" 让 Excel 打开不乱码
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> int:
    # Windows 控制台默认可能是 GBK，日志里有中文会乱码（同 main.py / watch_8002.py）
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="生成离线调试用视觉 CSV")
    parser.add_argument("--out", default=config.SAMPLE_VISION_CSV,
                        help=f"输出路径（默认 {config.SAMPLE_VISION_CSV}）")
    parser.add_argument("--seed", type=int, default=20260923, help="随机种子")
    args = parser.parse_args()

    count = generate(args.out, args.seed)
    print(f"已生成 {count} 帧（{DURATION:.0f}s @ {SAMPLE_RATE:.0f}Hz）→ {args.out}")
    print("时间线：0-20s normal / 20-50s tired / 50-58s absent / 58-80s normal / 80-100s sad / 100-120s normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
