# -*- coding: utf-8 -*-
"""家属数据落盘（:mod:`core.vitals_log`）—— 家属 Web 门户的数据源。

这条测试钉住的是 config §17 里的每一条硬约束。它们的共同点是：
**出问题的代价不是"少一行数据"，而是"B 主循环被拖住 / 坏库 / 刷屏"** ——
所以测试比"能写进去"多走几步：

    降采样    15 秒间隔取最新快照（不缩采样就等于把 1Hz rppg 原样倒进库）
    状态去重  仅切换才写（重复写心跳会让家属端状态时长全部归零）
    告警防抖  同 kind 60 秒一条（持续风险会刷爆 alerts）
    溢出策略  队列满丢弃 + 告警日志，put_nowait 绝不阻塞
    异常兜底  写失败只打日志；close 之后写静默 no-op
    建库      目录自动创建、WAL 生效、幂等建表
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

_BACKEND_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_B)

import config
import core.vitals_log as vitals_log
from core.vitals_log import QUEUE_MAXSIZE, VitalsLogWriter


def make_reading(**overrides) -> types.SimpleNamespace:
    """一条"长得像 RppgReading"的读数（鸭子类型即可，vitals_log 不 import 它）。"""
    fields = dict(timestamp=1.0, hr=72, rr=16.0)
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def rows(db_path: str, table: str) -> list[tuple]:
    """另开一个连接读行数 —— 和写线程同一路径（WAL 模式下互不干扰）。"""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    finally:
        conn.close()


class VitalsLogTestCase(unittest.TestCase):
    """公共基建：每个用例一个临时目录 + 一个关干净的 writer。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "family_data.sqlite3")
        self._writers: list[VitalsLogWriter] = []

    def tearDown(self) -> None:
        for writer in self._writers:
            writer.close()
        self._tmp.cleanup()

    def make_writer(self, interval: float = 0.0, **kwargs) -> VitalsLogWriter:
        # interval 默认 0：单测里立刻放行，节流行为由专门用例验证。
        # 必须等写线程把库建好（schema 在线程里建），否则断言会撞上
        # "no such table" 的竞态。
        writer = VitalsLogWriter(self.db_path, sample_interval=interval, **kwargs)
        self.assertTrue(writer._ready.wait(timeout=5.0), "写线程 5s 内未完成建库")
        self._writers.append(writer)
        return writer


# ===========================================================================
# 建库：目录自动创建 / WAL / 幂等
# ===========================================================================

class TestSchemaBootstrap(VitalsLogTestCase):

    def test_creates_missing_directory(self):
        """首次启动 data 目录不存在也能建起来（config §17 追加项 4）。"""
        nested = os.path.join(self._tmp.name, "a", "b", "family_data.sqlite3")
        writer = VitalsLogWriter(nested, sample_interval=0.0)
        self._writers.append(writer)
        self.assertTrue(os.path.exists(os.path.dirname(nested)))

    def test_wal_mode_is_active(self):
        writer = self.make_writer()
        self.assertTrue(writer._ready.wait(timeout=5.0))
        # WAL 是库文件属性，另开一个连接也能读到（写连接独占在写线程里）
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("PRAGMA journal_mode;").fetchone()[0], "wal")
        finally:
            conn.close()

    def test_schema_idempotent_on_reopen(self):
        """同一个文件开两次 writer（模拟 B 重启），表不炸、旧数据还在。"""
        writer = self.make_writer()
        writer.write_state_event("sad", "第一段")
        writer.close()
        writer2 = VitalsLogWriter(self.db_path, sample_interval=0.0)
        self._writers.append(writer2)
        self.assertEqual(len(rows(self.db_path, "state_events")), 1)

    def test_all_three_tables_created(self):
        self.make_writer()
        for table in ("vitals_samples", "state_events", "alerts"):
            self.assertEqual(rows(self.db_path, table), [])


# ===========================================================================
# 体征降采样
# ===========================================================================

class TestVitalsDownsampling(VitalsLogTestCase):
    """这些用例用**假时钟**（now=100.0，即 1970 年）控制节流边界 ——
    必须关掉真实时钟的老化清理，否则刚写的行会被首轮清理当成旧数据删掉。"""

    def test_respects_interval_taking_latest_snapshot(self):
        """15 秒间隔内只落一条；间隔过了取**当时**的最新快照。"""
        writer = self.make_writer(interval=15.0, retention_days=0.0)
        self.assertTrue(writer.maybe_write_vitals(make_reading(hr=70), now=100.0))
        self.assertFalse(writer.maybe_write_vitals(make_reading(hr=75), now=110.0))
        self.assertTrue(writer.maybe_write_vitals(make_reading(hr=80), now=115.1))
        writer.close()
        self.assertEqual(
            [(row[1], row[2]) for row in rows(self.db_path, "vitals_samples")],
            [(100.0, 70), (115.1, 80)],
        )

    def test_missing_fields_become_null_not_zero(self):
        """hr/rr/sqi 缺哪个补 NULL —— "未检出"不是 0（vitals 同一条红线）。"""
        # 假时钟（now=1.0），见类说明：须关掉真实时钟的老化清理
        writer = self.make_writer(retention_days=0.0)
        writer.maybe_write_vitals(make_reading(hr=None, rr=None), now=1.0)
        writer.close()
        self.assertEqual(rows(self.db_path, "vitals_samples")[0][2:], (None, None, None))

    def test_no_reading_yet_does_not_write(self):
        writer = self.make_writer()
        writer.maybe_write_vitals(None, now=1.0)
        writer.close()
        self.assertEqual(rows(self.db_path, "vitals_samples"), [])


# ===========================================================================
# 状态切换去重
# ===========================================================================

class TestStateTransitions(VitalsLogTestCase):

    def test_writes_only_on_transition(self):
        writer = self.make_writer()
        self.assertTrue(writer.write_state_event("sad", "用户情绪低落", ts=1.0))
        self.assertFalse(writer.write_state_event("sad", "", ts=2.0))
        self.assertTrue(writer.write_state_event("normal", "", ts=3.0))
        writer.close()
        recorded = rows(self.db_path, "state_events")
        self.assertEqual([row[2] for row in recorded], ["sad", "normal"])
        self.assertEqual(recorded[0][3], "用户情绪低落")

    def test_first_write_after_restart_always_lands(self):
        """重启后 _last_state 清零：同状态也会记一条（家属端按断档处理）。"""
        writer = self.make_writer()
        writer.write_state_event("tired", "", ts=1.0)
        writer.close()
        writer2 = VitalsLogWriter(self.db_path, sample_interval=0.0)
        self._writers.append(writer2)
        self.assertTrue(writer2.write_state_event("tired", "", ts=100.0))
        writer2.close()
        self.assertEqual(len(rows(self.db_path, "state_events")), 2)


# ===========================================================================
# 告警防抖
# ===========================================================================

class TestAlertDedup(VitalsLogTestCase):

    def test_same_kind_within_window_is_suppressed(self):
        writer = self.make_writer()
        self.assertTrue(writer.write_alert("emergency", "救命", ts=1.0))
        self.assertFalse(writer.write_alert("emergency", "救命", ts=30.0))
        self.assertTrue(writer.write_alert("emergency", "救命", ts=61.1))
        writer.close()   # 先 flush 再读，避免与消费线程的 commit 竞态
        self.assertEqual(len(rows(self.db_path, "alerts")), 2)

    def test_different_kinds_do_not_interfere(self):
        writer = self.make_writer()
        self.assertTrue(writer.write_alert("emergency", "救命", ts=1.0))
        self.assertTrue(writer.write_alert("crisis", "不想活了", ts=1.5))
        writer.close()
        self.assertEqual(len(rows(self.db_path, "alerts")), 2)

    def test_stores_only_kind_and_text_mapping(self):
        """只存 kind/text 两个字段 —— 不序列化 build_alert_payload 整包。"""
        writer = self.make_writer()
        writer.write_alert("emergency", "  救命  ", ts=1.0)
        writer.close()
        row = rows(self.db_path, "alerts")[0]
        self.assertEqual(row[1:], (1.0, "emergency", "救命"))


# ===========================================================================
# 队列溢出与异常兜底
# ===========================================================================

class TestQueueOverflow(VitalsLogTestCase):

    def test_full_queue_drops_without_blocking(self):
        """队列塞满后 put_nowait 丢弃 + 告警日志，绝不阻塞调用线程。"""
        writer = VitalsLogWriter(self.db_path, sample_interval=0.0)
        self._writers.append(writer)
        # 写线程可能消费得很快，直接灌到队列满为止；每条状态都切换所以都不被去重
        with mock.patch.object(vitals_log.logger, "warning") as warn:
            for i in range(200):
                writer.write_state_event("normal" if i % 2 else "sad", "", ts=float(i))
            self.assertTrue(warn.call_args_list,
                            "灌了 200 条状态切换都没遇到队列满，maxsize 设置没生效")
        writer.close()
        # 丢弃了一些（具体几条随消费速度浮动），但写线程活到了收尾、
        # 落库数量在合理区间内
        landed = len(rows(self.db_path, "state_events"))
        self.assertLess(landed, 200)
        self.assertGreater(landed, 0)

    def test_write_thread_survives_io_exception(self):
        """写线程崩溃 = 数据从此全丢，这是最隐蔽的故障 —— 必须就地吞掉。"""
        writer = self.make_writer()
        real_write = vitals_log.VitalsLogWriter._write
        raised = threading.Event()

        def flaky_write(self_, item):
            if item[1][1] == "normal":
                raised.set()
                raise sqlite3.OperationalError("disk I/O error")
            return real_write(self_, item)

        writer.write_state_event("sad", "", ts=1.0)
        with mock.patch.object(vitals_log.VitalsLogWriter, "_write", flaky_write):
            writer.write_state_event("normal", "", ts=2.0)
            # 补丁必须等到"失败确实发生"才能撤 —— 写线程是异步的，
            # 主线程提前退出 with 的话，线程可能在补丁移除后才处理这一条
            self.assertTrue(raised.wait(timeout=5.0), "异常注入未生效")
        writer.write_state_event("tired", "", ts=3.0)        # 写线程必须还活着
        writer.close()
        states = [row[2] for row in rows(self.db_path, "state_events")]
        self.assertIn("tired", states, "写线程炸掉后不再消费队列")
        self.assertNotIn("normal", states, "失败的写入不应留下半条数据")


class TestCloseSemantics(VitalsLogTestCase):

    def test_close_flushes_pending_then_stops(self):
        writer = self.make_writer()
        # 20 条 < 队列容量 32：积压任务必须在 close 返回前全部落库
        for i in range(20):
            writer.write_state_event("normal" if i % 2 else "sad", "", ts=float(i))
        writer.close()
        self.assertEqual(len(rows(self.db_path, "state_events")), 20)

    def test_writes_after_close_are_silent_noop(self):
        writer = self.make_writer()
        writer.close()
        writer.close()   # 幂等
        self.assertFalse(writer.write_state_event("sad", "", ts=1.0))
        self.assertFalse(writer.write_alert("emergency", "救命", ts=1.0))
        self.assertFalse(writer.maybe_write_vitals(make_reading(hr=72), now=1.0))
        self.assertEqual(rows(self.db_path, "state_events"), [])


class TestConcurrentWriters(VitalsLogTestCase):

    def test_threads_serialize_into_queue(self):
        """四个线程同时写：串行落库、一条不丢、调用方零阻塞。

        每条之间稍作间隔，让消费速度跟上生产 —— 溢出丢弃是**预期策略**
        （专门用例已覆盖），这个用例验证的是"无溢出时一条不丢"。
        """
        writer = self.make_writer()
        errors: list[Exception] = []

        def hammer(seed: int) -> None:
            try:
                for i in range(25):
                    writer.write_state_event(f"t{seed}", "", ts=float(seed * 100 + i))
                    time.sleep(0.005)
            except Exception as exc:   # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        writer.close()
        self.assertEqual(len(rows(self.db_path, "state_events")), 100)


# ===========================================================================
# 数据老化清理（二期）：删 7 天前的 vitals_samples，写线程周期执行
# ===========================================================================

DAY = 86400.0


class TestDeleteExpiredVitalsFunction(VitalsLogTestCase):
    """清理 SQL 本体：不动线程，直接对函数断言语义。"""

    def _seed(self, sample_ages: list[float], states: int = 0, alerts: int = 0):
        """直接建库并插入指定"年龄"的数据，返回连接（调用方负责关）。"""
        conn = sqlite3.connect(self.db_path)
        conn.executescript(vitals_log._SCHEMA)
        now = time.time()
        for age in sample_ages:
            conn.execute("INSERT INTO vitals_samples(ts, hr, rr, sqi) VALUES(?, 70, 15, 0.9)",
                         (now - age,))
        for i in range(states):
            conn.execute("INSERT INTO state_events(ts, state, reason) VALUES(?, 'normal', '')",
                         (now - 9 * DAY + i,))
        for i in range(alerts):
            conn.execute("INSERT INTO alerts(ts, kind, text) VALUES(?, 'emergency', 'x')",
                         (now - 9 * DAY + i,))
        conn.commit()
        return conn

    def test_deletes_only_rows_older_than_cutoff(self):
        conn = self._seed([8 * DAY, 7.5 * DAY, 3 * DAY, 0.0])
        try:
            deleted = vitals_log.delete_expired_vitals(conn, time.time() - 7 * DAY)
        finally:
            conn.close()
        self.assertEqual(deleted, 2)
        kept = [r[1] for r in rows(self.db_path, "vitals_samples")]
        self.assertEqual(len(kept), 2)

    def test_state_and_alerts_are_never_touched(self):
        """只清 vitals_samples（方案约定）—— 状态与预警是"发生过什么"的唯一凭据。"""
        conn = self._seed([8 * DAY], states=2, alerts=1)
        try:
            vitals_log.delete_expired_vitals(conn, time.time() - 7 * DAY)
        finally:
            conn.close()
        self.assertEqual(len(rows(self.db_path, "vitals_samples")), 0)
        self.assertEqual(len(rows(self.db_path, "state_events")), 2)
        self.assertEqual(len(rows(self.db_path, "alerts")), 1)


class TestWriterRetentionCleanup(VitalsLogTestCase):
    """写线程里的清理：启动先清一次 + 周期执行 + 可关闭。"""

    def test_startup_cleanup_removes_stale_rows(self):
        """上次运行留下的过期样本，重启后不等满一个周期就该消失。"""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        conn.executescript(vitals_log._SCHEMA)
        for age in (8 * DAY, 0.0):
            conn.execute("INSERT INTO vitals_samples(ts) VALUES(?)", (now - age,))
        conn.commit()
        conn.close()

        writer = self.make_writer(retention_days=7.0, cleanup_interval=0.3)
        time.sleep(1.2)          # > 首轮闹钟（初始 0）+ 一拍唤醒
        writer.close()
        kept = [r[1] for r in rows(self.db_path, "vitals_samples")]
        self.assertEqual(len(kept), 1)          # 8 天前的没了，今天的还在

    def test_periodic_cleanup_runs_while_running(self):
        """运行中插入的旧数据（模拟时钟回拨/手工导入）也在下个周期被清。"""
        writer = self.make_writer(retention_days=7.0, cleanup_interval=0.3)
        time.sleep(0.2)          # 让首轮清理先跑完
        now = time.time()
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("INSERT INTO vitals_samples(ts) VALUES(?)", (now - 9 * DAY,))
        conn.commit()
        conn.close()
        time.sleep(1.2)          # > cleanup_interval + 唤醒粒度
        writer.close()
        self.assertEqual(len(rows(self.db_path, "vitals_samples")), 0)

    def test_retention_disabled_keeps_everything(self):
        """retention_days <= 0 = 关闭清理：想留全量历史的调试场景。"""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        conn.executescript(vitals_log._SCHEMA)
        conn.execute("INSERT INTO vitals_samples(ts) VALUES(?)", (now - 30 * DAY,))
        conn.commit()
        conn.close()

        writer = self.make_writer(retention_days=0.0, cleanup_interval=0.3)
        time.sleep(1.2)
        writer.close()
        self.assertEqual(len(rows(self.db_path, "vitals_samples")), 1)

    def test_writing_still_works_after_cleanup(self):
        """清理动的是同一张表 —— 清完之后新写入必须照常落盘。"""
        writer = self.make_writer(interval=0.0, retention_days=7.0,
                                  cleanup_interval=0.3)
        time.sleep(1.0)          # 首轮清理跑过
        self.assertTrue(writer.maybe_write_vitals(make_reading(hr=75), now=time.time()))
        time.sleep(0.5)
        writer.close()
        data = rows(self.db_path, "vitals_samples")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][2], 75)


if __name__ == "__main__":
    unittest.main()
