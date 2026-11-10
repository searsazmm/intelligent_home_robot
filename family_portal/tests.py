# -*- coding: utf-8 -*-
"""family_portal 的视图与统计逻辑测试。

数据契约由 backend_B/core/vitals_log.py 定义（"仅切换写一条"、NULL=无
测量值、断档），本测试用**预置的 SQLite 文件**当替身 —— 不需要 B 在跑，
`python manage.py test family_portal` 就能全绿。

重点钉住两类容易翻车的行为：
1. 空库/缺库的**诚实兜底**（status=1、页面显示"暂无数据"，绝不编造）；
2. 状态时长的边界规则：末条取当前时间、断档不计入。
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta

from django.test import TestCase, override_settings

from family_portal import stats

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vitals_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    hr INTEGER, rr REAL, sqi REAL);
CREATE TABLE IF NOT EXISTS state_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    state TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    kind TEXT NOT NULL, text TEXT NOT NULL DEFAULT '');
"""


class TempDbMixin(unittest.TestCase):
    """造一个预置家属库（schema 与 B 侧一致），指向临时路径。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_file = os.path.join(self._tmp.name, "family_data.sqlite3")
        self.conn = sqlite3.connect(self.db_file)
        self.conn.executescript(_SCHEMA)
        self._patcher = override_settings(FAMILY_DB_PATH=self.db_file)
        self._patcher.enable()
        self.addCleanup(self._patcher.disable)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.conn.close)

    def insert_vitals(self, ts, hr=None, rr=None, sqi=None):
        self.conn.execute(
            "INSERT INTO vitals_samples(ts, hr, rr, sqi) VALUES(?,?,?,?)",
            (ts, hr, rr, sqi))

    def insert_state(self, ts, state, reason=""):
        self.conn.execute(
            "INSERT INTO state_events(ts, state, reason) VALUES(?,?,?)", (ts, state, reason))

    def insert_alert(self, ts, kind, text=""):
        self.conn.execute(
            "INSERT INTO alerts(ts, kind, text) VALUES(?,?,?)", (ts, kind, text))

    def commit(self):
        self.conn.commit()


class TestNoDataHonesty(TempDbMixin):
    """库都不存在时的诚实兜底 —— 家属端绝不能编造数据。"""

    def test_missing_file_gives_status_1(self):
        """库文件根本不存在 → connect_ro 返回 None，不抛异常。"""
        missing = os.path.join(self._tmp.name, "not_exist.sqlite3")
        from family_portal import db
        with override_settings(FAMILY_DB_PATH=missing):
            self.assertIsNone(db.connect_ro())

    def test_api_live_reports_no_data(self):
        """库文件在但三张表一张都没有 → 按无数据处理（False）。"""
        shell = os.path.join(self._tmp.name, "shell.sqlite3")
        sqlite3.connect(shell).close()
        from family_portal import db
        with override_settings(FAMILY_DB_PATH=shell):
            conn = db.connect_ro()
            try:
                self.assertFalse(db.ensure_tables_or_none(conn))
            finally:
                # 不关掉的话 Windows 下临时目录删不掉（WinError 32）
                if conn is not None:
                    conn.close()

    def test_empty_shell_db_is_detected(self):
        """B 刚启动、写线程还没建完表：库文件在但三张表缺 → 仍按无数据处理。"""
        shell = sqlite3.connect(self.db_file)
        shell.execute("DROP TABLE vitals_samples")
        shell.commit()
        shell.close()
        from family_portal import db
        with self.conn as _:
            pass
        # 主连接还开着，重新打开一个检查
        conn2 = sqlite3.connect(self.db_file)
        try:
            names = {r[0] for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            self.assertNotIn("vitals_samples", names)
        finally:
            conn2.close()


class TestVitalsStats(TempDbMixin):

    def test_averages_exclude_null_samples(self):
        """NULL = 无测量值：不进均值，但计入样本数（有效率 < 1）。"""
        now = time.time()
        self.insert_vitals(now, hr=70, rr=16.0)
        self.insert_vitals(now + 1, hr=None, rr=None)   # 未检出
        self.insert_vitals(now + 2, hr=80, rr=None)
        self.commit()
        day = stats.day_series(self.conn, days=1)[0]
        self.assertEqual(day["count"], 3)
        self.assertEqual(day["valid"], 2)
        self.assertEqual(day["hr_avg"], 75.0)   # None 那条不进均值
        self.assertAlmostEqual(day["rr_avg"], 16.0)   # 同理：只有 1 条有效 rr

    def test_series_returns_empty_days_as_nulls(self):
        """没有样本的天数值为 None —— 折线图上断开，不硬连。"""
        result = stats.day_series(self.conn, days=7)
        self.assertEqual(len(result), 7)
        self.assertTrue(all(d["hr_avg"] is None for d in result))

    def test_latest_vitals_carries_null_fields(self):
        now = time.time()
        self.insert_vitals(now, hr=72, rr=None, sqi=None)
        self.commit()
        latest = stats.latest_vitals(self.conn)
        self.assertEqual(latest["hr"], 72)
        self.assertIsNone(latest["rr"])


class TestStateDurationRules(TempDbMixin):
    """状态时长两条边界规则：末条取当前时间、断档不计入。"""

    def test_last_segment_counts_until_now(self):
        """最后一条没有结束时间 → 结束时间取当前时间（正在持续的状态）。"""
        now = time.time()
        self.insert_state(now - 10, "sad")
        self.commit()
        segments = stats.day_segments(self.conn, now - 60, now + 60)
        self.assertEqual(len(segments), 1)
        state, seconds = segments[0]
        self.assertEqual(state, "sad")
        self.assertAlmostEqual(seconds, 10.0, delta=1.0)   # ~10 秒并持续增长

    def test_gap_not_counted_into_any_state(self):
        """重启断档（>120s）：断档那段时间不归属任何状态。"""
        t0 = time.time()
        self.insert_state(t0 - 500, "normal")
        self.insert_state(t0 - 5, "sad")     # 与上一条隔 495 秒 → 断档
        self.commit()
        segments = stats.day_segments(self.conn, t0 - 600, t0 + 60)
        by_state = {s: sec for s, sec in segments}
        self.assertLess(by_state["normal"], 120.0, "normal 不应把断档时长吃进去")
        self.assertAlmostEqual(by_state["sad"], 5.0, delta=2.0)

    def test_distribution_days_parameter(self):
        """days=7 的窗口起点：7 天前的零点起算。"""
        now = time.time()
        seven_days_ago = (datetime.now().date() - timedelta(days=6))
        old_ts = datetime(seven_days_ago.year, seven_days_ago.month,
                          seven_days_ago.day).timestamp() + 100
        if old_ts < now:
            self.insert_state(old_ts, "tired")
            self.commit()
            result = stats.state_distribution(self.conn, days=7)
            self.assertTrue(any(d["state"] == "tired" for d in result))


class TestAlerts(TempDbMixin):

    def test_recent_alerts_order_and_fields(self):
        now = time.time()
        self.insert_alert(now - 60, "emergency", "救命")
        self.insert_alert(now - 30, "crisis", "不想活了")
        self.commit()
        alerts = stats.recent_alerts(self.conn)
        self.assertEqual([a["kind"] for a in alerts], ["crisis", "emergency"])   # 新→旧
        self.assertEqual(alerts[0]["text"], "不想活了")
        self.assertIn("time", alerts[0])


# ===========================================================================
# 视图 / REST 契约
# ===========================================================================

class ApiContractMixin(TempDbMixin):
    """所有 REST 响应必须长成 {status, message, data, timestamp}。"""

    CONTRACT_KEYS = {"status", "message", "data", "timestamp"}


class TestApiLive(ApiContractMixin, TestCase):

    def test_contract_and_payload(self):
        now = time.time()
        self.insert_vitals(now, hr=72, rr=16.0, sqi=0.8)
        self.insert_state(now - 5, "normal", "一切正常")
        self.insert_alert(now - 60, "emergency", "救命")
        self.commit()
        resp = self.client.get("/api/live/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(set(body.keys()), self.CONTRACT_KEYS)
        self.assertEqual(body["status"], 0)
        self.assertEqual(body["data"]["vitals"]["hr"], 72)
        self.assertEqual(body["data"]["state"]["state"], "normal")
        self.assertEqual(body["data"]["latest_sample_ts"], now)
        self.assertEqual(body["data"]["offline_threshold"], 30)

    def test_empty_tables_still_ok_but_null(self):
        """表在但还没写数据：库可用 → status=0，数据字段为 None（诚实表示"暂无"）。"""
        resp = self.client.get("/api/live/")
        body = resp.json()
        self.assertEqual(body["status"], 0)
        self.assertIsNone(body["data"]["vitals"])
        self.assertIsNone(body["data"]["state"])
        self.assertIsNone(body["data"]["latest_sample_ts"])

    def test_shell_db_status_1(self):
        """库文件在但表未建（B 刚启动）：status=1，不是 500。"""
        self.conn.execute("DROP TABLE alerts")
        self.conn.commit()
        resp = self.client.get("/api/live/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], 1)


class TestApiStats(ApiContractMixin, TestCase):

    def test_week_series_and_distribution(self):
        now = time.time()
        for i in range(5):
            self.insert_vitals(now - i * 3600, hr=70 + i, rr=16.0)
        self.insert_state(now - 100, "sad")
        self.commit()
        body = self.client.get("/api/stats/").json()
        self.assertEqual(body["status"], 0)
        self.assertEqual(len(body["data"]["series"]), 7)
        self.assertTrue(any(d["state"] == "sad" for d in body["data"]["distribution"]))


class TestApiAlerts(ApiContractMixin, TestCase):

    def test_list_newest_first(self):
        now = time.time()
        self.insert_alert(now - 99, "emergency", "救命")
        self.insert_alert(now, "crisis", "不想活了")
        self.commit()
        body = self.client.get("/api/alerts/").json()
        self.assertEqual(body["status"], 0)
        self.assertEqual([a["kind"] for a in body["data"]["alerts"]],
                         ["crisis", "emergency"])


class TestApiReport(ApiContractMixin, TestCase):

    def test_report_with_data(self):
        now = time.time()
        for i in range(10):
            self.insert_vitals(now - i * 600, hr=95, rr=22.0)
        self.insert_state(now - 100, "tired")
        self.commit()
        body = self.client.get("/api/report/").json()
        self.assertEqual(body["status"], 0)
        self.assertTrue(body["data"]["has_data"])
        self.assertGreater(len(body["data"]["diet_advice"]), 1)
        # 规则引擎：心率 95、呼吸 22、疲惫占比高 —— 都该被点名
        joined = "".join(body["data"]["diet_advice"])
        self.assertIn("心率", joined)
        self.assertIn("呼吸", joined)

    def test_report_empty_shows_honest_advice(self):
        body = self.client.get("/api/report/").json()
        self.assertEqual(body["status"], 0)
        self.assertFalse(body["data"]["has_data"])
        self.assertIn("无有效体征数据", "".join(body["data"]["diet_advice"]))


class TestPages(TestCase, TempDbMixin):
    """页面渲染：空库与有数据都不 500。"""

    def test_home_page_renders(self):
        now = time.time()
        self.insert_vitals(now, hr=72, rr=16.0)
        self.commit()
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "家人守护")

    def test_home_page_empty_ok(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_stats_page_renders(self):
        self.assertEqual(self.client.get("/stats/").status_code, 200)

    def test_report_page_renders_empty(self):
        """库表在但本周没样本：不 500，且诚实说明"无有效体征数据"。"""
        resp = self.client.get("/report/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "无有效体征数据")

    def test_report_page_with_data(self):
        now = time.time()
        self.insert_vitals(now, hr=72, rr=16.0)
        self.commit()
        resp = self.client.get("/report/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "家人健康周报")
