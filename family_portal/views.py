# -*- coding: utf-8 -*-
"""家属门户视图：3 个页面 + 4 个 REST 接口 + 登录/登出（二期鉴权）。

接口契约（《HTTP接口文档.md》§3.3）：统一
``{"status": int, "message": str, "data": dict, "timestamp": float}``，
**status = 0 成功，非 0 = 错误**（1 库不可用 / 无数据结构缺失等，
2 未授权）。HTTP 状态码与 status 分开：契约层面的错误（库缺表）用
status 表达，HTTP 本身仍 200，前端好统一处理；**未授权例外** ——
鉴权失败返回 HTTP 401 + status=2，让脚本端能第一时间区分"没带令牌"
和"没有数据"。

每个 REST 视图里都是"临时只读连接 → 查询 → close"三步 —— 连接不跨请求。

鉴权设计（方案二期项）：
- ``settings.FAMILY_API_TOKEN`` 为空 = 不启用（内网演示默认，零配置）；
- 非空时所有页面与接口都需授权，两条通道：
  · 浏览器：POST /login/ 校验令牌后写 session cookie（带 CSRF）；
  · 脚本/程序：请求头 ``X-API-Token`` 直传，无需登录态。
"""

from __future__ import annotations

import functools
import hmac
import logging
import socket
import time
from urllib.parse import urlparse, urlunparse

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render

from . import db, report as report_lib, stats as stats_lib

logger = logging.getLogger(__name__)

#: session 里标记"浏览器已通过令牌校验"的键。
SESSION_KEY = "family_authed"


def _api(data=None, message: str = "ok", status: int = 0, http_status: int = 200):
    """统一响应封装（契约 §3.3：0 = 成功，非 0 = 错误）。"""
    return JsonResponse({
        "status": status,
        "message": message,
        "data": data if data is not None else {},
        "timestamp": round(time.time(), 3),
    }, status=http_status)


def _auth_enabled() -> bool:
    """FAMILY_API_TOKEN 非空才启用鉴权 —— 空值维持"内网演示无鉴权"。"""
    return bool(getattr(settings, "FAMILY_API_TOKEN", ""))


def _render_page(request, template: str, context: dict | None = None):
    """页面渲染统一入口：注入鉴权开关，模板据此显示/隐藏"退出"。"""
    return render(request, template, {
        "auth_enabled": _auth_enabled(),
        **(context or {}),
    })


def _token_matches(supplied: str) -> bool:
    """常数时间令牌比较；非 ASCII 输入直接判不匹配。

    :func:`hmac.compare_digest` 的 str 版本要求两侧都是 ASCII-only，
    恶意/畸形的请求头（如含中文）会抛 TypeError —— 这里兜成"不匹配"，
    绝不让鉴权路径变成 500。
    """
    if not supplied:
        return False
    try:
        return hmac.compare_digest(supplied, settings.FAMILY_API_TOKEN)
    except TypeError:
        return False


def require_auth(view):
    """family_portal 的鉴权装饰器（方案二期：X-API-Token）。

    放行条件（满足其一）：未启用鉴权 / 请求头令牌正确 / session 已登录。
    API 路径未授权 → HTTP 401 + status=2；页面未授权 → 重定向登录页
    （带 ?next= 回跳）。只装饰 family_portal 自己的视图，admin 不受影响。
    """
    @functools.wraps(view)
    def wrapped(request, *args, **kwargs):
        if not _auth_enabled():
            return view(request, *args, **kwargs)
        supplied = request.headers.get("X-API-Token", "")
        if _token_matches(supplied) or request.session.get(SESSION_KEY) is True:
            return view(request, *args, **kwargs)
        if request.path.startswith("/api/"):
            return _api(None, "未授权：请携带 X-API-Token 请求头或先登录", status=2,
                        http_status=401)
        next_url = request.get_full_path() if request.path != "/" else ""
        return redirect(f"/login/?next={next_url}" if next_url else "/login/")
    return wrapped


def _safe_next(raw: str | None) -> str:
    """只回跳站内路径（防开放重定向）；"" 表示回首页。"""
    if raw and raw.startswith("/") and not raw.startswith("//"):
        return raw
    return ""


def login_view(request):
    """登录页：输入家属令牌，正确则写 session。

    错误提示**不区分**"令牌为空/错误"，不泄露任何校验细节。
    """
    error = None
    if request.method == "POST":
        if _auth_enabled() and _token_matches(request.POST.get("token", "")):
            request.session[SESSION_KEY] = True
            logger.info("家属门户登录成功（%s）", request.META.get("REMOTE_ADDR", "?"))
            return redirect(_safe_next(request.GET.get("next")) or "family_portal:home")
        error = "令牌不正确，请重新输入"
    return _render_page(request, "family/login.html",
                        {"error": error, "auth_enabled": True})


def logout_view(request):
    """登出：清 session 回登录页（未启用鉴权时回首页）。"""
    request.session.flush()
    return redirect("family_portal:login" if _auth_enabled() else "family_portal:home")


def _lan_ip() -> str | None:
    """探测本机局域网 IP（UDP connect 不发包，只查路由表）。失败返回 None。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))   # 不实际发包
            return sock.getsockname()[0]
    except OSError:
        return None


def _share_url(request) -> str:
    """报告页的扫码分享链接。

    优先级：
    1. ``FAMILY_SHARE_BASE_URL``（内网穿透的公网地址）—— 配了就直接用，
       手机不在同一 WiFi 也能扫开（跨网络场景）；
    2. 浏览器访问用的 localhost/127.0.0.1 对手机扫码无意义，替换成探测
       到的局域网 IP；
    3. 探测失败或本就是局域网地址则原样返回（家属通常就是通过局域网
       IP 访问的，此时 request 的 host 已经可用）。
    """
    base = getattr(settings, "FAMILY_SHARE_BASE_URL", "").rstrip("/")
    if base:
        return base + request.path
    url = request.build_absolute_uri()
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host not in ("localhost", "127.0.0.1"):
        return url
    lan = _lan_ip()
    if not lan:
        return url
    netloc = "%s:%d" % (lan, parsed.port) if parsed.port else lan
    return urlunparse(parsed._replace(netloc=netloc))


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

@require_auth
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
    return _render_page(request, "family/home.html", context)


@require_auth
def stats_page(request):
    """统计页：一周折线图 + 状态分布。"""
    return _render_page(request, "family/stats.html")


@require_auth
def report_page(request):
    """报告页：一周报告 + 饮食建议（微信扫码查看，替代打印导出）。"""
    conn = db.connect_ro()
    try:
        weather = report_lib.fetch_weather()
        if db.ensure_tables_or_none(conn):
            context = {"report": report_lib.weekly_report(conn, weather),
                       "share_url": _share_url(request),
                       "share_remote": bool(getattr(
                           settings, "FAMILY_SHARE_BASE_URL", ""))}
        else:
            context = {"report": None}
    finally:
        if conn is not None:
            conn.close()
    return _render_page(request, "family/report.html", context)


# --------------------------------------------------------------------------
# REST（前端 1.5s 轮询 api_live；其余按需）
# --------------------------------------------------------------------------

@require_auth
def api_live(request):
    """实时数据：最新体征 + 当前状态 + 今日概览 + 最新样本时间戳。

    ``data.server_ts`` 是**本响应生成时刻的服务端时间**（time.time()），
    前端离线判定改用它做 ``server_ts - latest_sample_ts`` 的差值 ——
    Django 与 backend_B 同机部署、共读同一个 SQLite，样本 ts 也是这台
    主机写的，所以这个差值与浏览器时钟**完全无关**（二期修复项：
    此前用浏览器本地时间对比，时钟不同步会误判离线）。
    """
    def handler(conn):
        data = {
            "vitals": stats_lib.latest_vitals(conn),
            "state": stats_lib.latest_state(conn),
            "today": stats_lib.today_overview(conn),
            "offline_threshold": db.OFFLINE_AFTER_SECONDS,
            "server_ts": round(time.time(), 3),
        }
        vitals = data["vitals"]
        data["latest_sample_ts"] = vitals["ts"] if vitals else None
        return _api(data)
    return _with_conn(handler)


@require_auth
def api_stats(request):
    """一周统计：逐日体征序列 + 状态分布（供统计页图表）。"""
    def handler(conn):
        return _api({
            "series": stats_lib.day_series(conn, days=7),
            "distribution": stats_lib.state_distribution(conn, days=7),
        })
    return _with_conn(handler)


@require_auth
def api_alerts(request):
    """预警列表（新→旧）。"""
    def handler(conn):
        return _api({"alerts": stats_lib.recent_alerts(conn, limit=20)})
    return _with_conn(handler)


@require_auth
def api_report(request):
    """一周报告（含饮食建议）。天气走 30 分钟缓存，失败自动降级。"""
    def handler(conn):
        weather = report_lib.fetch_weather()
        return _api(report_lib.weekly_report(conn, weather))
    return _with_conn(handler)
