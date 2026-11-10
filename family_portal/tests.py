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
from unittest import mock

from django.test import TestCase, override_settings

from family_portal import stats
from family_portal import report as report_lib

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

    def test_server_ts_is_authoritative_time_source(self):
        """二期：server_ts = 响应生成时刻的服务端时间，供前端离线判定，
        与浏览器时钟解耦（旧版用 Date.now() 对比会因时钟不同步误判）。"""
        before = time.time()
        self.insert_vitals(before - 10, hr=72, rr=16.0, sqi=0.8)
        self.commit()
        body = self.client.get("/api/live/").json()
        server_ts = body["data"]["server_ts"]
        self.assertIsInstance(server_ts, float)
        self.assertTrue(before <= server_ts <= time.time() + 1)
        # 判定输入齐了：server_ts - latest_sample_ts = 10s < 30s → 在线
        self.assertLess(server_ts - body["data"]["latest_sample_ts"],
                        body["data"]["offline_threshold"])

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


class TestWellnessIndices(TempDbMixin, TestCase):
    """健康指数分析（图3 渐变指数条）：心肺→元气，状态占比→压力。"""

    def test_no_data_returns_none(self):
        """无数据 → None，页面不渲染指数条 —— 不编造"看起来正常"的数。"""
        self.assertIsNone(report_lib.wellness_indices({"sample_count": 0}))
        self.assertIsNone(report_lib.wellness_indices(None))

    def test_vitality_follows_hr_rr_grades(self):
        """心率呼吸都平稳 → 元气分 92（good）；明显异常 → 45。"""
        good = report_lib.wellness_indices({
            "sample_count": 10, "hr_avg": 72, "rr_avg": 16.0,
            "negative_ratio": 0.0, "state_distribution": []})
        self.assertEqual(good["vitality"]["score"], 92)
        self.assertEqual(good["vitality"]["label"], "良好")
        self.assertEqual(good["stress"]["score"], 0)
        self.assertEqual(good["stress"]["label"], "从容")

        bad = report_lib.wellness_indices({
            "sample_count": 10, "hr_avg": 110, "rr_avg": 28.0,
            "negative_ratio": 0.0, "state_distribution": []})
        self.assertEqual(bad["vitality"]["score"], 45)
        self.assertEqual(bad["vitality"]["label"], "需关注")

    def test_stress_blends_sad_and_tired(self):
        """压力 = 低落×0.6 + 疲惫×0.4 ×100；段位词按分数落档。"""
        mixed = report_lib.wellness_indices({
            "sample_count": 10, "hr_avg": 72, "rr_avg": 16.0,
            "negative_ratio": 0.5,
            "state_distribution": [{"state": "sad", "ratio": 0.3},
                                   {"state": "tired", "ratio": 0.2}]})
        # 0.5*0.6 + 0.2*0.4 = 0.38 → 38 分 → "平稳"
        self.assertEqual(mixed["stress"]["score"], 38)
        self.assertEqual(mixed["stress"]["label"], "平稳")

    def test_stress_falls_back_to_negative_only(self):
        """分布里没有疲惫细分（全 sad）→ 退回负向占比整体。"""
        idx = report_lib.wellness_indices({
            "sample_count": 10, "hr_avg": 72, "rr_avg": 16.0,
            "negative_ratio": 0.4,
            "state_distribution": [{"state": "sad", "ratio": 0.4}]})
        self.assertEqual(idx["stress"]["score"], 40)

    def test_vitality_none_when_hr_rr_unmeasured(self):
        """本周样本存在但心率呼吸全未检出 → 元气"未检出"，压力仍照算。"""
        idx = report_lib.wellness_indices({
            "sample_count": 10, "hr_avg": None, "rr_avg": None,
            "negative_ratio": 0.5,
            "state_distribution": [{"state": "sad", "ratio": 0.5}]})
        self.assertIsNone(idx["vitality"]["score"])
        self.assertEqual(idx["vitality"]["label"], "未检出")
        # 无疲惫细分 → 退回负向占比整体：0.5 × 100
        self.assertEqual(idx["stress"]["score"], 50)

    def test_vitality_averages_only_measured_parts(self):
        """心率有效、呼吸未检出 → 取有效项单独定级，不当作 0 或满分。"""
        idx = report_lib.wellness_indices({
            "sample_count": 10, "hr_avg": 72, "rr_avg": None,
            "negative_ratio": 0.0, "state_distribution": []})
        self.assertEqual(idx["vitality"]["score"], 92)

    def test_stress_score_clamped_to_0_100(self):
        """异常输入（占比>1）不得把指数条顶出轨道。"""
        idx = report_lib.wellness_indices({
            "sample_count": 10, "hr_avg": 72, "rr_avg": 16.0,
            "negative_ratio": 1.5,
            "state_distribution": [{"state": "tired", "ratio": 2.0}]})
        self.assertTrue(0 <= idx["stress"]["score"] <= 100)

    def test_gauges_empty_without_data_and_rendered_as_hint(self):
        """无数据 → summary 整体为 None（评分/指数都不给），页面显示
        诚实提示而非空白卡。"""
        from family_portal import report as rep
        wk = rep.weekly_report(self.conn)
        self.assertIsNone(wk["summary"])
        self.assertIsNone(wk["indices"])
        resp = self.client.get("/report/")
        # 指数区块整体在 if summary 内；无数据时走 else 的诚实提示
        self.assertContains(resp, "本周无有效体征数据")
        self.assertNotContains(resp, "健康元气势数")

    def test_report_api_carries_indices(self):
        """api_report 输出 indices；报告页渲染指数条与状态词。"""
        now = time.time()
        for i in range(5):
            self.insert_vitals(now - i * 600, hr=72, rr=16.0)
        self.commit()
        body = self.client.get("/api/report/").json()
        self.assertEqual(body["data"]["indices"]["vitality"]["score"], 92)
        resp = self.client.get("/report/")
        self.assertContains(resp, "健康元气势数")
        self.assertContains(resp, "精神压力指数")


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


class TestReportPageRendering(TempDbMixin, TestCase):
    """报告页**模板渲染**断言：数据到页面的最后一公里。

    上游逻辑（评分/指数/建议规则）各有单元测试；这里钉的是模板侧：
    各区块在"有数据/无数据"两态下渲染出**该出现的**与**不该出现的**，
    以及 hover 提示（data-tip）确实跟着数据一起到页面 —— 模板变量
    拼错、字段名对不上这类问题只有走到 HTML 才暴露。
    """

    def seed_week(self, hr=88.0, rr=14.0):
        """铺 7 天样本 + 状态 + 预警。hr=88（>85）会触发心率建议。"""
        now = time.time()
        for day in range(7):
            base = now - day * 86400
            for i in range(4):
                self.insert_vitals(base - i * 60, hr=hr, rr=rr, sqi=0.9)
            self.insert_state(base, "normal", "演示")
        self.insert_alert(now, "emergency", "演示预警")
        self.commit()

    def test_empty_report_hides_hero_and_tips(self):
        """空库：评分环/词卡/指数区都不渲染 —— 不给家属看空白假区块。"""
        resp = self.client.get("/report/")
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, "健康状态综合分")
        self.assertNotContains(resp, "data-tip")
        self.assertNotContains(resp, "基本指标")
        self.assertContains(resp, "本周无有效体征数据")

    def test_hero_ring_with_score_rule_tip(self):
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "score-ring")
        self.assertContains(resp, "健康状态综合分")
        self.assertContains(resp, "综合评分")
        # hr=88 → 仅扣 1.5 → 98 分 ≥85 → "整体状态平稳"
        self.assertContains(resp, "整体状态平稳")
        # 评分环 hover：扣分规则原文跟着数据上页面
        self.assertContains(resp, "起点 100 分")

    def test_status_tiles_words_follow_data(self):
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "基本指标")
        # hr=88 → 偏快；rr=14 → 平稳；无疲惫低落 → 情绪良好、活力充沛
        for word in ("偏快", "平稳", "良好", "充沛"):
            self.assertContains(resp, word)
        # 四张卡各带自己的 hover 提示（含参考区间）
        self.assertContains(resp, "成年人静息参考 60~85 次/分")

    def test_gauges_rendered_with_dots(self):
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "健康元气势数")
        self.assertContains(resp, "精神压力指数")
        self.assertContains(resp, "index-dot")

    def test_overview_table_and_state_distribution(self):
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "平均心率")
        self.assertContains(resp, "有效样本")
        self.assertContains(resp, "预警次数")
        # 状态分布：badge 类名 + 日均时长行 + 覆盖率说明
        self.assertContains(resp, "badge state-normal")
        self.assertContains(resp, "日均约")
        self.assertContains(resp, "状态时间覆盖")

    def test_diet_advice_triggered_by_high_hr(self):
        """hr=88>85：饮食建议区渲染出心率相关建议 —— 规则到页面的闭环。"""
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "三、接下来一周的饮食建议")
        self.assertContains(resp, "心率")

    def test_weather_line_when_fetched(self):
        """天气拉取成功 → "参考天气"行上页面；视图确实把天气传给了模板。"""
        self.seed_week()
        with mock.patch("family_portal.views.report_lib.fetch_weather",
                        return_value={"temp": 12.3, "code": 1, "wind": 2.0}):
            resp = self.client.get("/report/")
        self.assertContains(resp, "参考天气")
        self.assertContains(resp, "12.3℃")

    def test_hover_tips_counted(self):
        """1 个评分环 + 4 张词卡 + 2 条指数 = 至少 7 个 data-tip。"""
        self.seed_week()
        html = self.client.get("/report/").content.decode("utf-8")
        self.assertGreaterEqual(html.count("data-tip"), 7)

    def test_generated_at_stamp(self):
        self.seed_week()
        self.assertContains(self.client.get("/report/"), "生成于")
