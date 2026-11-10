# -*- coding: utf-8 -*-
"""家属库统计查询：全部是**纯函数**（连接进、dict 出），方便测试与复用。

数据由 backend_B 的 core/vitals_log.py 写入，三张表的语义契约（含 NULL、
"仅切换写一条"、"断档"）都定义在那边 —— 本模块是消费者，按契约解读：

- ``vitals_samples``：hr/rr 为 NULL = 无测量值，统计均值时排除，
  并单独统计"有效率"（有测量的样本占比）；
- ``state_events``：仅状态**切换**时写一条 → 状态时长 = 相邻两条 ts 之差；
  **最后一条没有结束时间，结束时间取当前时间**（见 day_segments）；
  相邻记录间隔超过 :data:`db.STATE_GAP_SECONDS` 视为断档（程序没在跑），
  该段不计入任何状态时长；
- ``alerts``：kind/text 两个映射字段 + 60 秒同 kind 防抖由写入方负责。

空数据一律返回 None / 空列表 —— 页面据此诚实显示"暂无数据"。
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta

from .db import STATE_GAP_SECONDS

#: 已知状态集合（backend_B 侧 api_doc §4 冻结的四态；非法值同样计入，只是不参与配色）
KNOWN_STATES = ("normal", "sad", "tired", "absent")


def _day_start(date_str: str | None, days_back: int = 0) -> float:
    """某天的本地零点的 Unix 时间戳。date_str 为空表示今天。"""
    day = datetime.now().date() - timedelta(days=days_back)
    if date_str:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
    return datetime(day.year, day.month, day.day).timestamp()


def latest_vitals(conn: sqlite3.Connection) -> dict | None:
    """最新一条体征样本（含 NULL 字段原样透出，供前端显示"未检出"）。"""
    row = conn.execute(
        "SELECT ts, hr, rr, sqi FROM vitals_samples ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {"ts": row[0], "hr": row[1], "rr": row[2], "sqi": row[3]}


def latest_state(conn: sqlite3.Connection) -> dict | None:
    """最新一条状态事件（仅切换记录，同状态心跳不落库）。"""
    row = conn.execute(
        "SELECT ts, state, reason FROM state_events ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {"ts": row[0], "state": row[1], "reason": row[2]}


def day_series(conn: sqlite3.Connection, days: int = 7) -> list[dict]:
    """近 N 天逐日体征统计（含今天），按日期升序。

    每天给：样本数、有效率（hr 非空占比）、hr/rr 的均值与 min/max。
    hr/rr 为 NULL 的样本不进均值。没有任何样本的天也给出（数值为 None），
    折线图上断开 —— 缺数据就画缺口，不硬连。
    """
    today_start = _day_start(None)
    out: list[dict] = []
    for back in range(days - 1, -1, -1):
        start = today_start - back * 86400.0
        end = start + 86400.0
        count, valid, hr_avg, hr_min, hr_max, rr_avg = conn.execute(
            """
            SELECT COUNT(*),
                   SUM(hr IS NOT NULL),
                   AVG(hr),
                   MIN(hr),
                   MAX(hr),
                   AVG(rr)
            FROM vitals_samples
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        ).fetchone()
        day_label = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        out.append({
            "date": day_label,
            "count": count or 0,
            "valid": valid or 0,
            "validity": round((valid or 0) / count, 3) if count else None,
            "hr_avg": round(hr_avg, 1) if hr_avg is not None else None,
            "hr_min": hr_min,
            "hr_max": hr_max,
            "rr_avg": round(rr_avg, 1) if rr_avg is not None else None,
        })
    return out


def day_segments(conn: sqlite3.Connection,
                 start_ts: float, end_ts: float) -> list[tuple[str, float]]:
    """[start_ts, end_ts) 区间里的状态时长片段 [(state, seconds), ...]。

    状态切换记录的语义：
    - 每条 state_event 的持续段 = 本条 ts 到下一条 ts（或区间末尾）；
    - **最后一条没有结束时间 → 结束时间取 min(区间末尾, 当前时间)**；
    - 相邻两条间隔 > STATE_GAP_SECONDS 视为断档（机器人没在跑），
      该段不计入任何状态。
    """
    rows = conn.execute(
        "SELECT ts, state FROM state_events WHERE ts < ? ORDER BY ts",
        (end_ts,),
    ).fetchall()
    now = time.time()
    segments: list[tuple[str, float]] = []
    for i, (ts, state) in enumerate(rows):
        seg_start = max(ts, start_ts)
        if i + 1 < len(rows):
            seg_end = rows[i + 1][0]
        else:
            # 最后一条：结束时间取当前时间（正在持续的状态）
            seg_end = min(end_ts, now)
        if seg_end <= seg_start:
            continue
        if seg_end - seg_start > STATE_GAP_SECONDS and i + 1 < len(rows):
            # 断档：本条之后机器人长期没跑，只计入到断档起点为止
            seg_end = seg_start
        segments.append((state, seg_end - seg_start))
    return segments


def state_distribution(conn: sqlite3.Connection,
                       date_str: str | None = None,
                       days: int = 1) -> list[dict]:
    """状态时长分布（默认今天；``days=7`` 给一周）。按时长降序。

    长于 STATE_GAP_SECONDS 的持续段按断档截断，所以总占比可能 < 100% ——
    差额就是"机器人没在运行的时间"，页面如实说明。
    """
    start = _day_start(date_str, days_back=days - 1) if days > 1 else _day_start(date_str)
    end = start + days * 86400.0
    if date_str and days == 1:
        start = _day_start(date_str)
        end = start + 86400.0

    totals: dict[str, float] = {}
    for state, seconds in day_segments(conn, start, end):
        totals[state] = totals.get(state, 0.0) + seconds
    accounted = sum(totals.values())
    span = max(0.0, min(end, time.time()) - start)
    out = []
    for state, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        ratio = seconds / accounted if accounted else 0.0
        out.append({
            "state": state,
            "seconds": round(seconds, 1),
            "minutes": round(seconds / 60.0, 1),   # 模板展示用（秒太长不好读）
            "ratio": round(ratio, 3),              # 0~1（前端图表用）
            "pct": round(ratio * 100, 1),          # 百分数（模板直显用）
        })
    if out:
        out[-1]["span_covered"] = round(accounted / span, 3) if span else 0.0
    return out


def today_overview(conn: sqlite3.Connection) -> dict:
    """今日概览：平均体征 + 状态分布 + 预警数。字段缺失为 None。"""
    start = _day_start(None)
    end = start + 86400.0
    hr_avg, rr_avg, count = conn.execute(
        "SELECT AVG(hr), AVG(rr), COUNT(*) FROM vitals_samples WHERE ts >= ? AND ts < ?",
        (start, end),
    ).fetchone()
    return {
        "hr_avg": round(hr_avg, 1) if hr_avg is not None else None,
        "rr_avg": round(rr_avg, 1) if rr_avg is not None else None,
        "sample_count": count or 0,
        "state_distribution": state_distribution(conn),
        "alert_count": conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE ts >= ? AND ts < ?", (start, end)
        ).fetchone()[0],
    }


def recent_alerts(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """最近的预警，新→旧。只消费 kind/text 两个映射字段。"""
    rows = conn.execute(
        "SELECT ts, kind, text FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
        {"ts": row[0], "kind": row[1], "text": row[2],
         "time": datetime.fromtimestamp(row[0]).strftime("%m-%d %H:%M")}
        for row in rows
    ]


def overall_stats(conn: sqlite3.Connection, days: int = 7) -> dict:
    """近 N 天整体统计（报告页用）：均值 / 区间 / 状态占比 / 预警总数。"""
    start = _day_start(None, days_back=days - 1)
    end = start + days * 86400.0
    hr_avg, rr_avg, hr_min, hr_max, count, valid = conn.execute(
        """
        SELECT AVG(hr), AVG(rr), MIN(hr), MAX(hr), COUNT(*), SUM(hr IS NOT NULL)
        FROM vitals_samples WHERE ts >= ? AND ts < ?
        """,
        (start, end),
    ).fetchone()
    distribution = []
    accounted = 0.0
    for item in state_distribution(conn, days=days):
        distribution.append(item)
        accounted += item["seconds"]
    negative = sum(
        item["ratio"] for item in distribution if item["state"] in ("sad", "tired")
    )
    return {
        "days": days,
        "hr_avg": round(hr_avg, 1) if hr_avg is not None else None,
        "rr_avg": round(rr_avg, 1) if rr_avg is not None else None,
        "hr_min": hr_min,
        "hr_max": hr_max,
        "sample_count": count or 0,
        "valid": valid or 0,
        "state_distribution": distribution,
        "negative_ratio": round(negative, 3),
        "alert_count": conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE ts >= ? AND ts < ?", (start, end)
        ).fetchone()[0],
        "accounted_seconds": round(accounted, 1),
    }
