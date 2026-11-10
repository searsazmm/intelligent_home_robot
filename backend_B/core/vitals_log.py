# -*- coding: utf-8 -*-
"""家属数据落盘 —— family_portal（8020 家属 Web 门户）的数据源。

B 收到的三类事件写进一个 SQLite（``config.FAMILY_DB_PATH``），家属 Web 端
用只读连接读它。三方契约：

    vitals_samples(id, ts, hr, rr, sqi)   hr/rr 为 NULL = 无测量值（A 端窗口未满 / SNR 不足）
    state_events(id, ts, state, reason)   仅状态**切换**时写一条（同状态心跳不重复）
    alerts(id, ts, kind, text)            只存 kind/text 两个字段映射，不序列化
                                          build_alert_payload 整包 —— emergency
                                          模块与家属 Web 侧就此解耦

⚠️ 本模块有一条**压倒一切的纪律**：B 的 TCP / 状态机主循环绝不能因为
落盘受任何影响。所有公开方法都只是**非阻塞入队**（``put_nowait``，队列满
丢弃本条 + 告警日志，绝不等待），真正的写库在单一线程里串行做，任何异常
就地吞掉只打日志。宁可丢一个数据点，不可卡状态机一拍。

线程模型：公开方法（任意线程调）→ ``queue.Queue(maxsize=32)`` → 单写线程
→ sqlite 连接。写连接**独占**在写线程里，所以不需要
``check_same_thread=False`` —— 多线程写坏库 / 锁死的问题从结构上不存在。
Django 只读侧（每请求临时 ``mode=ro`` 连接，用完即关）靠 WAL 快照读，
写线程 commit 不会挡它。

降采样规则：每满 ``sample_interval`` 秒取当前最新快照写一条（本版**不做
窗口均值**，后续扩展在 ``maybe_write_vitals`` 一处改）。时间戳全程
``time.time()``（REAL Unix 秒），不混 datetime —— 见 config §17。
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

#: 入队队列上限。满了就丢数据点 + 告警日志 —— 落盘是旁路，不是关键路径。
QUEUE_MAXSIZE = 32

#: 同一种 kind 的告警防抖窗口（秒）：60 秒内不重复入库，
#: 防止持续触发同一风险时 alerts 表被刷屏。
ALERT_DEDUP_SEC = 60.0

#: 数据保留天数默认值（二期数据老化清理，config §17 可用环境变量覆盖）。
#: 方案【范围与说明】：演示版只适合短期保留约 7 天数据。
DEFAULT_RETENTION_DAYS = 7.0

#: 清理动作的执行间隔（秒）：清理本身是轻量 DELETE（有 ts 索引），
#: 但没必要每次空转都跑 —— 默认 6 小时一次足够。
CLEANUP_INTERVAL_SEC = 6.0 * 3600.0

#: 写线程空转时的唤醒粒度（秒）：get 超时后顺手看一眼"清理闹钟"。
CLEANUP_WAKE_SEC = 1.0

#: 建表语句（幂等）。家属 Web 端直接按这些列名读。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS vitals_samples (
    id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts  REAL NOT NULL,
    hr  INTEGER,
    rr  REAL,
    sqi REAL
);
CREATE INDEX IF NOT EXISTS idx_vitals_ts ON vitals_samples(ts);
CREATE TABLE IF NOT EXISTS state_events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    state  TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_state_ts ON state_events(ts);
CREATE TABLE IF NOT EXISTS alerts (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts   REAL NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
"""

#: 写线程的退出哨兵（放队列里，保证退出前把积压任务写完）。
_STOP = object()


def delete_expired_vitals(conn: sqlite3.Connection, cutoff_ts: float) -> int:
    """删除 ``cutoff_ts`` 之前的 vitals_samples 旧样本，返回删除行数。

    模块级函数：写线程周期调用，也让单元测试不必碰线程就能测清理语义。
    **只清 vitals_samples**（方案约定的大头，15s 一条）；state_events /
    alerts 全量保留 —— 状态切换与预警本身量小，且是家属端"发生过什么"
    的唯一凭据，删了统计口径就不完整。

    WAL 下 DELETE 不会立刻缩小文件，但空间会被后续写入复用 —— 对
    "控制长期体积"的目标而言足够，不做 wal_checkpoint（清大库时那一下
    反而可能短暂阻塞）。
    """
    cur = conn.execute("DELETE FROM vitals_samples WHERE ts < ?", (cutoff_ts,))
    conn.commit()
    return cur.rowcount


class VitalsLogWriter:
    """家属数据落盘写手。见模块文档的线程模型与纪律。"""

    def __init__(self, path: str, sample_interval: float = 15.0,
                 retention_days: float = DEFAULT_RETENTION_DAYS,
                 cleanup_interval: float = CLEANUP_INTERVAL_SEC) -> None:
        self._path = path
        self._interval = max(0.0, float(sample_interval))
        # 保留天数：<= 0 = 关闭清理（本地调试想留全量历史时用）
        self._retention_days = float(retention_days)
        self._cleanup_interval = max(0.0, float(cleanup_interval))
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._closed = False
        # 只保护判定状态（_closed / _last_* 降采样与防抖记账），
        # 真正的写库在写线程里串行，不需要这把锁。
        self._state_lock = threading.Lock()
        self._last_vitals_ts = 0.0
        self._last_state: str | None = None
        self._last_alert_at: dict[str, float] = {}

        # 目录自动创建：首次启动 data 目录可能还不存在
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        # ⚠️ sqlite 连接**在写线程里创建**（见 _run）：连接由哪个线程创建，
        # 默认就只能被哪个线程使用。在 __init__ 里建好再传给写线程会得到
        # ProgrammingError（所有写入被异常兜底吞掉，等于永远落不了盘，
        # 连接还关不掉）。写连接独占在写线程里，所以不需要
        # check_same_thread=False —— 多线程写坏库 / 锁死从结构上不存在。
        self._conn: sqlite3.Connection | None = None
        self._ready = threading.Event()

        self._thread = threading.Thread(target=self._run, name="family-log", daemon=True)
        self._thread.start()
        logger.info("家属数据落盘已启动：%s（体征降采样 %.0fs）", path, self._interval)

    # ------------------------------------------------------------------
    # 公开方法：全部非阻塞，任意线程可调
    # ------------------------------------------------------------------

    def maybe_write_vitals(self, reading: object, now: float | None = None) -> bool:
        """按间隔降采样写入最新体征快照。返回是否真的入队。

        每满 ``sample_interval`` 秒取**当前**最新快照写一条 —— 不做窗口
        均值（本版简化）。``reading`` 为 None（还没收到过 rppg）直接跳过。
        hr/rr/sqi 缺哪个补 NULL（语义 = 无测量值，家属端据此显示"未检出"）。
        """
        now = time.time() if now is None else float(now)
        with self._state_lock:
            if self._closed:
                return False
            if now - self._last_vitals_ts < self._interval:
                return False
            self._last_vitals_ts = now
        if reading is None:
            return True   # 间隔到了但没有读数：本次跳过（间隔已记账）
        hr = getattr(reading, "hr", None)
        rr = getattr(reading, "rr", None)
        sqi = getattr(reading, "sqi", None)
        row = (
            now,
            None if hr is None else int(hr),
            None if rr is None else round(float(rr), 1),
            None if sqi is None else float(sqi),
        )
        self._enqueue(("vitals", row))
        return True

    def write_state_event(self, state: str, reason: str = "",
                          ts: float | None = None) -> bool:
        """状态**切换**才写一条（同状态心跳不重复）。返回是否真的入队。

        家属端统计状态时长靠"相邻两条的 ts 差"，重复写心跳会让每段时长
        全部归零 —— 这条去重是统计正确性的前提。程序重启后
        ``_last_state`` 清零，重启后的第一条必然落盘；时间轴上的断档
        （相邻记录间隔过大）由家属端统计逻辑按断档处理。
        """
        ts = time.time() if ts is None else float(ts)
        state = str(state or "")
        with self._state_lock:
            if self._closed:
                return False
            if state == self._last_state:
                return False
            self._last_state = state
        self._enqueue(("state", (ts, state, str(reason or ""))))
        return True

    def write_alert(self, kind: str, text: str = "",
                    ts: float | None = None) -> bool:
        """告警入库（只存 kind/text，60 秒同 kind 防抖）。返回是否真的入队。

        ⚠️ 只提取两个字段，**不序列化** :func:`core.emergency.build_alert_payload`
        的完整 payload —— emergency 模块的报文结构将来怎么改都不影响
        家属 Web 侧的表结构。
        """
        ts = time.time() if ts is None else float(ts)
        kind = str(kind or "")
        with self._state_lock:
            if self._closed:
                return False
            last = self._last_alert_at.get(kind)
            if last is not None and ts - last < ALERT_DEDUP_SEC:
                return False
            self._last_alert_at[kind] = ts
        self._enqueue(("alert", (ts, kind, str(text or "").strip())))
        return True

    def close(self, timeout: float = 5.0) -> None:
        """停止写线程（先写完积压任务）并关库。幂等，可重复调用。

        之后再调用任何写方法都是静默 no-op（返回 False），不抛异常 ——
        shutdown 流程里谁先谁后都不该炸。
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        # 哨兵也要走队列；队列满时挤掉一条待写任务给它腾位
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(_STOP)
            except (queue.Empty, queue.Full):
                pass
        self._thread.join(timeout=timeout)
        logger.info("家属数据落盘已停止")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _enqueue(self, item: object) -> None:
        if self._closed:
            return   # close 之后再写：静默跳过（config §17 硬约束）
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("家属数据落盘队列已满（%d），丢弃本条写入任务",
                           QUEUE_MAXSIZE)

    def _run(self) -> None:
        """写线程主循环：建库 → 串行消费队列 → 收尾关库。

        连接在这里创建（线程独占，见 __init__ 里的说明）；建库失败直接
        退出线程并打日志 —— 落盘整个不可用，主循环照常运行。
        """
        try:
            conn = sqlite3.connect(self._path)
            # WAL：家属 Web 端只读连接才能边写边读（见模块文档）
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(_SCHEMA)
            conn.commit()
        except Exception:
            logger.exception("家属数据落盘初始化失败，落盘功能停用（不影响 B 主循环）")
            self._ready.set()
            return
        self._conn = conn
        self._ready.set()

        # 清理闹钟：初始为 0 —— **启动先清一次**，上次运行留下的过期数据
        # 不用等满一个 cleanup_interval 才消失；每跑一次顺延一个周期。
        next_cleanup = 0.0
        while True:
            # 空转时每秒醒一次看清理闹钟；有任务则立即消费，两者都不误
            try:
                item = self._queue.get(timeout=CLEANUP_WAKE_SEC)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            if item is not None:
                try:
                    self._write(item)
                except Exception:
                    # 磁盘满 / 只读 / 库损坏……都只打日志。
                    # B 的主循环绝不能因为落盘失败被打断。
                    logger.exception("家属数据落盘失败（不影响 B 主循环）")
            now = time.monotonic()
            if (self._cleanup_interval > 0 and self._retention_days > 0
                    and now >= next_cleanup):
                try:
                    deleted = delete_expired_vitals(
                        conn, time.time() - self._retention_days * 86400.0)
                    if deleted:
                        logger.info("家属数据老化清理：删除 %.1f 天前旧样本 %d 条",
                                    self._retention_days, deleted)
                except Exception:
                    # 清理失败不影响写入，下个周期再试
                    logger.exception("家属数据老化清理失败（下个周期重试）")
                next_cleanup = now + self._cleanup_interval
        try:
            self._conn.close()
        except Exception:
            pass

    def _write(self, item: object) -> None:
        kind, row = item
        if kind == "vitals":
            self._conn.execute(
                "INSERT INTO vitals_samples(ts, hr, rr, sqi) VALUES(?, ?, ?, ?)", row)
        elif kind == "state":
            self._conn.execute(
                "INSERT INTO state_events(ts, state, reason) VALUES(?, ?, ?)", row)
        elif kind == "alert":
            self._conn.execute(
                "INSERT INTO alerts(ts, kind, text) VALUES(?, ?, ?)", row)
        self._conn.commit()


__all__ = ["ALERT_DEDUP_SEC", "CLEANUP_INTERVAL_SEC", "DEFAULT_RETENTION_DAYS",
           "QUEUE_MAXSIZE", "VitalsLogWriter", "delete_expired_vitals"]
