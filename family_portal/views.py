# -*- coding: utf-8 -*-
"""家属门户视图：3 个页面 + 4 个 REST 接口。

接口契约（《HTTP接口文档.md》§3.3）：统一
``{"status": int, "message": str, "data": dict, "timestamp": float}``，
**status = 0 成功，非 0 = 错误**（1 库不可用 / 无数据结构缺失等）。
HTTP 状态码与 status 分开：契约层面的错误（库缺表）用 status 表达，
HTTP 本身仍 200，前端好统一处理。

每个 REST 视图里都是"临时只读连接 → 查询 → close"三步 —— 连接不跨请求。
"""

from __future__ import annotations

import logging
import time

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from . import db, report as report_lib, stats as stats_lib

logger = logging.getLogger(__name__)


def _api(data=None, message: str = "ok", status: int = 0):
    """统一响应封装（契约 §3.3：0 = 成功，非 0 = 错误）。"""
    return JsonResponse({
        "status": status,
        "message": message,
        "data": data if data is not None else {},
        "timestamp": round(time.time(), 3),
    })


def _with_conn(handler):
    """通用骨架：开只读连接 → 调 handler(conn) → 关连接。

    库不存在 / 表未建好 → 返回 status=1 的空数据响应（前端据此显示
    "暂无数据"），绝不抛 500。
    """
    conn = db.connect_ro()
    try:
        if not db.ensure_tables_or_none(conn):
            return _api(None, "暂无数据（机器人未运行或尚未采集）", status=1)
        return handler(conn)
    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------

def home(request):
    """家属主页：实时监测卡 + 今日概览 + 最近预警。"""
    context = {}
    conn = db.connect_ro()
    try:
        if db.ensure_tables_or_none(conn):
            context = {
                "latest_vitals": stats_lib.latest_vitals(conn),
                "latest_state": stats_lib.latest_state(conn),
                "today": stats_lib.today_overview(conn),
                "alerts": stats_lib.recent_alerts(conn, limit=5),
            }
    finally:
        if conn is not None:
            conn.close()
    return render(request, "family/home.html", context)


def stats_page(request):
    """统计页：一周折线图 + 状态分布。"""
    return render(request, "family/stats.html")


def report_page(request):
    """报告页：一周报告 + 饮食建议（浏览器打印即 PDF）。"""
    conn = db.connect_ro()
    try:
        weather = report_lib.fetch_weather()
        if db.ensure_tables_or_none(conn):
            context = {"report": report_lib.weekly_report(conn, weather)}
        else:
            context = {"report": None}
    finally:
        if conn is not None:
            conn.close()
    return render(request, "family/report.html", context)


# --------------------------------------------------------------------------
# REST（前端 1.5s 轮询 api_live；其余按需）
# --------------------------------------------------------------------------

def api_live(request):
    """实时数据：最新体征 + 当前状态 + 今日概览 + 最新样本时间戳。

    ``data.latest_sample_ts`` 供前端做**离线判定**（超过 30s 判离线；
    浏览器时钟差异的局限见方案文档 —— 二期改为服务端时间差）。
    """
    def handler(conn):
        data = {
            "vitals": stats_lib.latest_vitals(conn),
            "state": stats_lib.latest_state(conn),
            "today": stats_lib.today_overview(conn),
            "offline_threshold": db.OFFLINE_AFTER_SECONDS,
        }
        vitals = data["vitals"]
        data["latest_sample_ts"] = vitals["ts"] if vitals else None
        return _api(data)
    return _with_conn(handler)


def api_stats(request):
    """一周统计：逐日体征序列 + 状态分布（供统计页图表）。"""
    def handler(conn):
        return _api({
            "series": stats_lib.day_series(conn, days=7),
            "distribution": stats_lib.state_distribution(conn, days=7),
        })
    return _with_conn(handler)


def api_alerts(request):
    """预警列表（新→旧）。"""
    def handler(conn):
        return _api({"alerts": stats_lib.recent_alerts(conn, limit=20)})
    return _with_conn(handler)


def api_report(request):
    """一周报告（含饮食建议）。天气走 30 分钟缓存，失败自动降级。"""
    def handler(conn):
        weather = report_lib.fetch_weather()
        return _api(report_lib.weekly_report(conn, weather))
    return _with_conn(handler)
