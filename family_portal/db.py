# -*- coding: utf-8 -*-
"""家属库连接：**只读**，每请求临时开、用完即关，禁止全局长连接。

数据由 backend_B（core/vitals_log.py，写线程独占连接 + WAL）写入；
本模块只提供 ``mode=ro`` 的只读连接。WAL 模式下只读连接拿到的是快照，
不会阻塞写线程，写线程 commit 也不会挡它。

跨进程共享文件的规矩：本 app 对库**没有任何写操作**（连 PRAGMA 都只读），
B 没启动、文件不存在时 :func:`connect_ro` 返回 None，由上层诚实显示
"暂无数据"，不编造。
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

#: 状态分布统计的断档阈值（秒）：相邻两条 state_events 间隔超过它视为
#: "机器人没在运行"（程序重启 / 长时间关机），该段时长不计入任何状态。
STATE_GAP_SECONDS = 120.0

#: 离线判定阈值（秒）：最新体征样本距现在的间隔超过它，前端判定机器人
#: 离线（前端拿浏览器时间对比；时钟同步局限见方案文档 v3 §离线判定）。
OFFLINE_AFTER_SECONDS = 30.0


def db_path() -> Path:
    """家属库路径。测试用 ``override_settings(FAMILY_DB_PATH=...)`` 覆盖。"""
    from django.conf import settings

    override = getattr(settings, "FAMILY_DB_PATH", None)
    if override:
        return Path(override)
    # 默认：backend_B/data/family_data.sqlite3（config §17 的默认落点）
    return Path(__file__).resolve().parent.parent / "backend_B" / "data" / "family_data.sqlite3"


def connect_ro(path: Path | None = None):
    """打开只读连接。库文件不存在返回 None（调用方按"暂无数据"处理）。

    ⚠️ 用完必须 close —— 每个请求临时开一个，**禁止**挂在模块级或缓存在
    长生命周期的对象上（写线程一直在 commit，长连接会让 WAL 无限增长）。
    只读连接自己不跨线程使用，``check_same_thread=False`` 只是留个余地。
    """
    path = path or db_path()
    if not Path(path).exists():
        return None
    uri = f"file:{Path(path).as_posix()}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=2.0)
    except sqlite3.Error:
        return None


def ensure_tables_or_none(conn):
    """防御：连接打开但表还没建（B 刚启动、写线程还没跑完建表）。

    返回 True 表示三张表齐全；False 表示库还是空壳，上层按"暂无数据"处理。
    """
    if conn is None:
        return False
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except sqlite3.Error:
        return False
    return {"vitals_samples", "state_events", "alerts"}.issubset(names)
