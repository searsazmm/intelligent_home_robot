# -*- coding: utf-8 -*-
"""演示数据播种：造一周"起伏大"的家属端数据（体征 + 状态 + 预警）。

用法（仓库根目录）::

    python backend_B/tools/seed_demo_week.py                # 默认写到 backend_B/data/family_data.sqlite3
    python backend_B/tools/seed_demo_week.py --days 7       # 天数可调
    python backend_B/tools/seed_demo_week.py --db 其他路径.sqlite3

为什么需要它
------------
答辩/验收时希望折线图、条形图、词卡、评分、饮食建议都"有话可说"：
真实回放的数据太平稳（心率恒 75、呼吸恒 30），图表是一条直线，规则引擎
也只会触发一两条。本脚本直接按**设计好的周画像**写库：

- 心率逐日在 60~100 间大幅摆动，周均 ≈85 —— 刚好触发"心率偏高"规则；
- 呼吸周均 ≈20.3 —— 刚好触发"呼吸偏快"规则；
- 状态占比 疲惫 35% / 低落 32% / 平稳 33% —— 分别触发疲惫、低落两条规则；
- 逐日均值各不相同 —— 一周统计的折线/条形有起伏可看。

一致性约定
----------
家属端所有页面（实时监测 / 一周统计 / 健康报告 / 饮食建议）读的都是
**同一个 SQLite 库**，本脚本只改数据不改任何视图逻辑 —— 一个地方变了，
所有相关部分自动同步，不需要分别修改。

注意
----
- 状态事件按 60s 心跳续写：stats.day_segments 把相邻事件间隔超过
  ``db.STATE_GAP_SECONDS``（120s）的段视为"机器人没在跑"断档截断，
  所以演示数据必须每分钟续写一条同状态事件，占比才会接近设计值。
- 会**清空**三张表现有数据再写入（演示库，不保留历史）。
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import sqlite3
import sys
import time

# 让脚本既能从仓库根跑、也能从 tools 目录跑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "family_data.sqlite3")

#: 每天的体征画像（近 → 远当天在列表里的顺序：index 0 = 最早一天）。
#: 设计目标：周均 hr ≈87（≥85 触发偏高规则）、rr ≈21（≥20 触发偏快规则），
#: 且逐日差异大，折线图有明显的起伏。
#: 余量说明：最终周均值会被"今天只过了半天"稀释，所以基数要比阈值高出 2 左右。
HR_DAY_BASES = [102.0, 70.0, 96.0, 62.0, 100.0, 86.0, 94.0]
RR_DAY_BASES = [27.0, 14.5, 25.0, 13.0, 26.0, 16.5, 25.5]

#: 日内波动参数：心率 ±12（4 小时周期）、呼吸 ±2（6 小时周期），
#: 叠加少量噪声后 clamp 到生理上说得过去的范围。
HR_WAVE_AMP = 12.0
RR_WAVE_AMP = 2.0
HR_CLAMP = (45.0, 130.0)
RR_CLAMP = (10.0, 30.0)

#: 一天的状态块设计（时长秒）。周占比目标：疲惫 ≈37%、低落 ≈34%、平稳 ≈29%。
#: 余量说明：占比按"已计入的时长"算，今天只过半天会被稀释 1~2 个点，
#: 所以设计值要比触发阈值（疲惫 30%、低落 30%）高出 4~5 个点。
#: 每天三块的先后顺序按天轮换，看起来不像机械复制。
BLOCK_SECONDS = {"tired": 8.8 * 3600.0, "sad": 8.2 * 3600.0, "normal": 7.0 * 3600.0}
BLOCK_ORDER = ["tired", "sad", "normal"]

STATE_REASON = {
    "tired": "演示：表情/姿态判定为疲惫",
    "sad": "演示：表情/语调判定为情绪低落",
    "normal": "演示：状态平稳",
}


def _wave(t: float, period_sec: float, amp: float) -> float:
    return amp * math.sin(2.0 * math.pi * (t % period_sec) / period_sec)


def seed(db_path: str, days: int = 7, interval: float = 60.0) -> None:
    now = time.time()
    # 对齐到"今天本地 0 点"，让逐日统计边界干净
    day_start = time.mktime(datetime.date.today().timetuple())
    # 起点往前推 days 天
    start_ts = day_start - days * 86400.0

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for table in ("vitals_samples", "state_events", "alerts"):
            cur.execute(f"DELETE FROM {table}")

        # ---- 体征：每 interval 一条，日内正弦波 + 噪声 ----
        vitals = []
        ts = start_ts
        i = 0
        while ts < now:
            day_idx = min(int((ts - start_ts) // 86400.0), days - 1)
            hr_base = HR_DAY_BASES[day_idx]
            rr_base = RR_DAY_BASES[day_idx]
            hr = hr_base + _wave(ts, 4 * 3600.0, HR_WAVE_AMP) \
                + (0.9 * math.sin(i * 2.399))          # 伪随机小噪声（黄金角散列）
            rr = rr_base + _wave(ts, 6 * 3600.0, RR_WAVE_AMP) \
                + (0.25 * math.sin(i * 1.712))
            hr = int(round(min(max(hr, HR_CLAMP[0]), HR_CLAMP[1])))
            rr = round(min(max(rr, RR_CLAMP[0]), RR_CLAMP[1]), 1)
            sqi = round(0.92 + 0.05 * abs(math.sin(i * 0.911)), 3)
            vitals.append((ts, hr, rr, sqi))
            ts += interval
            i += 1
        cur.executemany(
            "INSERT INTO vitals_samples(ts, hr, rr, sqi) VALUES(?, ?, ?, ?)", vitals)

        # ---- 状态：每 60s 续写一条（避免 >120s 断档），块顺序按天轮换 ----
        # 注意 range(days + 1)：含今天 —— days=7 的"一周"窗口里今天是第 8 个
        # 自然日（起点是 7 天前的 0 点），漏掉它今天的分布就成了空白。
        states = []
        for d in range(days + 1):
            day0 = start_ts + d * 86400.0
            day_end = min(day0 + 86400.0, now)
            # 轮换块顺序，时长不变 → 周占比仍是设计值
            order = BLOCK_ORDER[d % len(BLOCK_ORDER):] + BLOCK_ORDER[:d % len(BLOCK_ORDER)]
            bounds, acc = [], day0
            for name in order:
                nxt = acc + BLOCK_SECONDS[name]
                bounds.append((name, acc, min(nxt, day_end)))
                acc = nxt
            # 按分钟走：落在哪个块就写哪个状态
            t = day0
            while t < day_end:
                for name, b0, b1 in bounds:
                    if b0 <= t < b1:
                        states.append((t, name, STATE_REASON[name]))
                        break
                t += 60.0
        # 补一条"此刻仍在持续"的结尾事件（最后一条的段会延伸到当前时间）
        cur.executemany(
            "INSERT INTO state_events(ts, state, reason) VALUES(?, ?, ?)", states)

        # ---- 预警：几条配合画像的记录（最近一小时内） ----
        alerts = [
            (now - 3200.0, "vitals", "检测到心率持续偏高（98 次/分），已记录"),
            (now - 1800.0, "vitals", "检测到呼吸偏快（25 次/分），建议留意休息"),
            (now - 600.0, "emergency", "老人按下紧急求助按钮（演示）"),
        ]
        cur.executemany("INSERT INTO alerts(ts, kind, text) VALUES(?, ?, ?)", alerts)

        conn.commit()
        print(f"已写入演示数据：{len(vitals)} 条体征、{len(states)} 条状态、{len(alerts)} 条预警")
        print(f"时间范围：{time.strftime('%m-%d %H:%M', time.localtime(start_ts))}"
              f" ~ {time.strftime('%m-%d %H:%M', time.localtime(now))}")
    finally:
        conn.close()


def live_loop(db_path: str, sample_interval: float = 15.0,
               state_interval: float = 60.0) -> None:
    """播种后续写：让"实时监测"页在演示期间一直在线。

    每 ``sample_interval`` 秒追加一条体征（延续当天画像的正弦波，衔接
    播种数据的相位），每 ``state_interval`` 秒续写一条当前状态（同样
    是为了不触发 120s 断档）。按 ``Ctrl+C`` 停止。
    """
    conn = sqlite3.connect(db_path)
    try:
        start_ts = conn.execute(
            "SELECT MIN(ts) FROM vitals_samples").fetchone()[0]
        state_ts, state = conn.execute(
            "SELECT ts, state FROM state_events ORDER BY ts DESC LIMIT 1").fetchone()
        print(f"续写中：体征每 {sample_interval:.0f}s、状态每 {state_interval:.0f}s"
              f"（当前状态 {state}）—— Ctrl+C 停止", flush=True)
        while True:
            now = time.time()
            cur = conn.cursor()
            last_vital = cur.execute(
                "SELECT MAX(ts) FROM vitals_samples").fetchone()[0]
            if now - last_vital >= sample_interval:
                # 与 seed() 同一套波形公式（按 ts 取相位，天然衔接）
                day_idx = min(int((now - start_ts) // 86400.0),
                              len(HR_DAY_BASES) - 1)
                i = int(round((now - start_ts) / sample_interval))
                hr = HR_DAY_BASES[day_idx] + _wave(now, 4 * 3600.0, HR_WAVE_AMP) \
                    + 0.9 * math.sin(i * 2.399)
                rr = RR_DAY_BASES[day_idx] + _wave(now, 6 * 3600.0, RR_WAVE_AMP) \
                    + 0.25 * math.sin(i * 1.712)
                hr = int(round(min(max(hr, HR_CLAMP[0]), HR_CLAMP[1])))
                rr = round(min(max(rr, RR_CLAMP[0]), RR_CLAMP[1]), 1)
                sqi = round(0.92 + 0.05 * abs(math.sin(i * 0.911)), 3)
                cur.execute(
                    "INSERT INTO vitals_samples(ts, hr, rr, sqi) VALUES(?, ?, ?, ?)",
                    (now, hr, rr, sqi))
            if now - state_ts >= state_interval:
                cur.execute(
                    "INSERT INTO state_events(ts, state, reason) VALUES(?, ?, ?)",
                    (now, state, "演示续写：状态保持"))
                state_ts = now
            conn.commit()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("已停止续写")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="播种一周起伏大的演示数据")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite 路径（默认家属端演示库）")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--interval", type=float, default=60.0, help="体征采样间隔秒")
    parser.add_argument("--live", action="store_true",
                        help="播种后进入续写循环，让实时监测页保持在线")
    args = parser.parse_args()
    seed(args.db, days=args.days, interval=args.interval)
    if args.live:
        live_loop(args.db)


if __name__ == "__main__":
    main()
