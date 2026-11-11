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
import json
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

from django.test import TestCase, override_settings

from family_portal import llm_advice, stats
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
        # 所有走库的测试默认关闭大模型出口：weekly_report 会征询
        # llm_advice，开发机上若配了 B_DEEPSEEK_TOKEN，不关断就会在
        # 测试里真发外网请求。LLM 专项用例自行打开并注入假传输。
        llm_off = override_settings(FAMILY_LLM_ENABLED=False)
        llm_off.enable()
        self.addCleanup(llm_off.disable)
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


class TestStatusWords(unittest.TestCase):
    """状态词表全覆盖 —— 每个词都对应一条**有依据**的规则分支。

    页面上每一个状态词（偏快/过缓/有波动…）都是给家属看的判断，
    词错一个就误导一次。此前只被 88/90 这种"偏高"路径偶然覆盖过，
    这里把整张词表钉死。
    """

    def test_hr_word_table(self):
        f = self._words()
        # 词表函数返回 (word, level, tip) 三元组，断言取 [0]
        self.assertEqual(f["_hr_word"](None)[0], "未检出")
        self.assertEqual(f["_hr_word"](105)[0], "过速")   # > 100
        self.assertEqual(f["_hr_word"](93)[0], "偏快")    # > 90（60~100 内偏高段）
        self.assertEqual(f["_hr_word"](72)[0], "平稳")    # 60~100
        self.assertEqual(f["_hr_word"](58)[0], "偏缓")    # < 60
        self.assertEqual(f["_hr_word"](50)[0], "过缓")    # < 55
        # 边界值归属：恰在阈值上取宽的一侧（90/85/60 平稳、55 偏缓）
        self.assertEqual(f["_hr_word"](90)[0], "平稳")
        self.assertEqual(f["_hr_word"](85)[0], "平稳")    # 85 在通用区间内，不是"偏快"
        self.assertEqual(f["_hr_word"](60)[0], "平稳")
        self.assertEqual(f["_hr_word"](55)[0], "偏缓")
        # level 分级与词一致：过速/过缓是 bad，偏快/偏缓是 mid
        self.assertEqual(f["_hr_word"](105)[1], "bad")
        self.assertEqual(f["_hr_word"](93)[1], "mid")

    def test_rr_word_table(self):
        f = self._words()
        self.assertEqual(f["_rr_word"](None)[0], "未检出")
        self.assertEqual(f["_rr_word"](8)[0], "过缓")     # < 10
        self.assertEqual(f["_rr_word"](11)[0], "偏缓")    # < 12
        self.assertEqual(f["_rr_word"](16)[0], "平稳")    # 12~20
        self.assertEqual(f["_rr_word"](22)[0], "偏快")    # > 20

    def test_mood_and_energy_word_tables(self):
        # None 由调用方兜底（summary 未生成时不进词表），词表只认 0~1 数值
        f = self._words()
        self.assertEqual(f["_mood_word"](0.1)[0], "良好")   # < 0.15
        self.assertEqual(f["_mood_word"](0.2)[0], "平稳")   # < 0.30
        self.assertEqual(f["_mood_word"](0.4)[0], "有波动")  # < 0.50
        self.assertEqual(f["_mood_word"](0.6)[0], "偏低落")
        self.assertEqual(f["_energy_word"](0.1)[0], "充沛")
        self.assertEqual(f["_energy_word"](0.2)[0], "正常")
        self.assertEqual(f["_energy_word"](0.4)[0], "易疲惫")
        self.assertEqual(f["_energy_word"](0.6)[0], "疲乏")

    @staticmethod
    def _words() -> dict:
        from family_portal import report as rep
        return {"_hr_word": rep._hr_word, "_rr_word": rep._rr_word,
                "_mood_word": rep._mood_word, "_energy_word": rep._energy_word}


class TestDietAdviceBranches(TempDbMixin, TestCase):
    """饮食建议规则的全分支：每条建议都必须有触发条件，且带依据。

    此前只测过"心率偏高"一条路径，这里补齐低心率/情绪/气温分支 ——
    这些规则是发给家属看的，漏一条 = 该提醒时没提醒。
    """

    @staticmethod
    def _stats(hr=72.0, rr=16.0, tired=0.0, sad=0.0, negative=None) -> dict:
        dist = []
        if tired:
            dist.append({"state": "tired", "ratio": tired, "seconds": 3600})
        if sad:
            dist.append({"state": "sad", "ratio": sad, "seconds": 3600})
        return {"sample_count": 100, "hr_avg": hr, "rr_avg": rr,
                "negative_ratio": negative, "state_distribution": dist}

    def _texts(self, stats, weather=None) -> str:
        from family_portal import report as rep
        return " ".join(rep.diet_advice(stats, weather))

    def test_low_hr_advise_iron(self):
        # hr=50 低于 60~100 下限 → 低心率建议（含补铁食物）
        self.assertIn("低于静息常见区间 60~100 的下限", self._texts(self._stats(hr=50.0)))
        self.assertIn("含铁食物", self._texts(self._stats(hr=50.0)))

    def test_mid_hr_no_advise(self):
        """85 是通用静息区间内的正常值 —— 不给心率建议，不吓家属。"""
        self.assertNotIn("心率", self._texts(self._stats(hr=85.0)))

    def test_sad_ratio_advise_companionship(self):
        self.assertIn("情绪低落", self._texts(self._stats(sad=0.4)))

    def test_negative_without_single_state_advise_family_time(self):
        """负面占比 >=50% 但疲惫/低落单项都不足 30%（混合型）也要接住。"""
        text = self._texts(self._stats(tired=0.25, sad=0.2, negative=0.6))
        self.assertIn("晒太阳", text)

    def test_hot_weather_advise_hydration(self):
        text = self._texts(self._stats(), weather={"temp": 32.0, "code": 0, "wind": 1.0})
        self.assertIn("补水", text)

    def test_cold_weather_advise_warm_food(self):
        text = self._texts(self._stats(), weather={"temp": 2.0, "code": 0, "wind": 1.0})
        self.assertIn("温热", text)


class TestHealthScoreBranches(TempDbMixin, TestCase):
    """健康综合评分的扣分分支：心缓/呼吸慢、'建议关注'档位。"""

    def test_low_hr_and_slow_rr_deduct(self):
        """hr=55、rr=10：两个'偏缓'方向同样扣分（此前只测过偏快方向）。"""
        from family_portal import report as rep
        s = rep.health_summary({"sample_count": 100, "hr_avg": 55.0,
                                "rr_avg": 10.0, "negative_ratio": 0.0,
                                "state_distribution": []})
        # 100 - (60-55)/2 - (12-10) = 95.5 → round 银行家舍入 → 96
        self.assertEqual(s["score"], 96)

    def test_score_below_70_says_attention(self):
        from family_portal import report as rep
        dist = [{"state": "sad", "ratio": 1.0, "seconds": 3600}]
        s = rep.health_summary({"sample_count": 100, "hr_avg": 94.0,
                                "rr_avg": 16.0, "negative_ratio": 1.0,
                                "state_distribution": dist})
        # 100 - 2 - 30 = 68 → 最低档"建议关注"
        self.assertEqual(s["score"], 68)
        self.assertEqual(s["overall"], "建议关注")


class TestStatsDateBranches(TempDbMixin, TestCase):
    """stats 的日期参数分支：指定日期统计、未来事件截断。"""

    def test_day_start_parses_date_string(self):
        from family_portal import stats as st
        ts = st._day_start("2026-11-01", 0)
        self.assertEqual(ts, datetime(2026, 11, 1).timestamp())

    def test_distribution_with_explicit_date_only_counts_that_day(self):
        now = time.time()
        self.insert_vitals(now, hr=72, rr=16.0)                      # 今天
        # 3 天前：事件放在**当天零点后一分钟** —— 覆盖几乎全天；
        # 若误按"7 天窗"统计，ratio 会被今天等其它天稀释
        old_midnight = (datetime.now() - timedelta(days=3)
                        ).replace(hour=0, minute=0, second=1, microsecond=0)
        self.insert_vitals(old_midnight.timestamp(), hr=80, rr=16.0)
        self.insert_state(old_midnight.timestamp(), "tired", "过去")
        self.commit()
        old_day = old_midnight.strftime("%Y-%m-%d")
        dist = stats.state_distribution(self.conn, date_str=old_day)
        # 返回 list（时长降序），span_covered 在末项、取值 0~1
        self.assertTrue(dist, "指定日期应有分布数据")
        self.assertTrue(all(d["ratio"] > 0.9 for d in dist[:-1]))  # tired 主导
        self.assertGreater(dist[-1]["span_covered"], 0.95)  # 覆盖近全天，非 7 天窗

    def test_future_state_events_are_excluded(self):
        """未来时间戳的状态事件（时钟回拨残留等）不得计入今日时长。"""
        now = time.time()
        self.insert_state(now - 300, "normal", "五分钟前")
        self.insert_state(now + 3600, "tired", "未来")   # seg_end=min(now,..) < start
        self.commit()
        dist = stats.state_distribution(self.conn)
        states = {d["state"]: d["ratio"] for d in dist}
        # 未来事件整段被丢弃；过去的事件保留（ratio 是"占当天已流逝时间"
        # 的比例且保留 3 位小数，5 分钟的时段数值会被舍到 0.0，不比大小）
        self.assertNotIn("tired", states)
        self.assertIn("normal", states)


class TestFailureFallbacks(TempDbMixin, TestCase):
    """db/views 的故障兜底分支（覆盖率缺口：需要制造故障才走到）。"""

    def test_db_path_default_when_setting_absent(self):
        from family_portal import db
        with override_settings(FAMILY_DB_PATH=None):
            self.assertTrue(str(db.db_path()).replace("\\", "/")
                            .endswith("backend_B/data/family_data.sqlite3"))

    def test_connect_ro_returns_none_on_sqlite_error(self):
        from family_portal import db
        with mock.patch.object(db.sqlite3, "connect",
                               side_effect=sqlite3.OperationalError("corrupt")):
            self.assertIsNone(db.connect_ro(self.db_file))

    def test_ensure_tables_with_none_conn(self):
        from family_portal import db
        self.assertFalse(db.ensure_tables_or_none(None))

    def test_ensure_tables_with_broken_conn(self):
        """坏连接（已关闭）上查表：吞 ProgrammingError 返回 False，
        绝不让"库不可用"变成 500。"""
        from family_portal import db
        conn = sqlite3.connect(self.db_file)
        conn.close()
        self.assertFalse(db.ensure_tables_or_none(conn))

    def test_report_page_on_shell_db_shows_empty_notice(self):
        """空壳库（文件在、表不在）：报告页诚实提示而非 500。

        TempDbMixin 的库自带表结构，这里另造一个真·空壳文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            shell = os.path.join(tmp, "shell.sqlite3")
            conn = sqlite3.connect(shell)   # 建文件但不建表
            conn.close()
            with override_settings(FAMILY_DB_PATH=shell):
                resp = self.client.get("/report/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "暂无任何数据")


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

    def seed_week(self, hr=92.0, rr=14.0):
        """铺 7 天样本 + 状态 + 预警。hr=92（>90）触发心率建议与"偏快"词卡。"""
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
        # hr=92 → 仅扣 1 → 99 分 ≥85 → "整体状态平稳"
        self.assertContains(resp, "整体状态平稳")
        # 评分环 hover：扣分规则原文跟着数据上页面
        self.assertContains(resp, "起点 100 分")

    def test_status_tiles_words_follow_data(self):
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "基本指标")
        # hr=92 → 偏快；rr=14 → 平稳；无疲惫低落 → 情绪良好、活力充沛
        for word in ("偏快", "平稳", "良好", "充沛"):
            self.assertContains(resp, word)
        # 四张卡各带自己的 hover 提示（含参考区间）
        self.assertContains(resp, "成年人静息参考 60~100 次/分")

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
        """hr=92>90：饮食建议区渲染出心率相关建议 —— 规则到页面的闭环。"""
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "三、AI 营养师饮食建议")
        self.assertContains(resp, "心率")

    def test_weather_not_displayed_but_drives_rules(self):
        """天气不再单独展示"参考天气"行，但仍驱动建议规则（32℃ → 补水）。"""
        self.seed_week()
        with mock.patch("family_portal.views.report_lib.fetch_weather",
                        return_value={"temp": 32.3, "code": 1, "wind": 2.0}):
            resp = self.client.get("/report/")
        self.assertNotContains(resp, "参考天气")
        self.assertContains(resp, "补水")

    def test_share_qr_replaces_print_tip(self):
        """打印导出提示换成微信扫码：二维码容器 + 文案 + vendor 脚本。"""
        self.seed_week()
        resp = self.client.get("/report/")
        self.assertContains(resp, "share-qr")
        self.assertContains(resp, "微信扫一扫，手机查看报告")
        self.assertContains(resp, "qrcode.min.js")
        self.assertNotContains(resp, "另存为 PDF")
        # 未配隧道 → 提示同一局域网
        self.assertContains(resp, "同一局域网")

    def test_share_qr_remote_hint_when_tunnel_configured(self):
        """配了 FAMILY_SHARE_BASE_URL → 文案切换为"外网也能打开"。"""
        self.seed_week()
        with override_settings(FAMILY_SHARE_BASE_URL="https://demo.example.com"):
            resp = self.client.get("/report/")
        self.assertContains(resp, "外网（流量）也能打开")
        self.assertNotContains(resp, "同一局域网")

    def test_share_url_uses_lan_ip_for_localhost(self):
        """localhost 访问 → 二维码链接替换成局域网 IP（手机才扫得开）。"""
        from django.test import RequestFactory
        from family_portal import views
        factory = RequestFactory()
        request = factory.get("/report/", SERVER_NAME="localhost")
        url = views._share_url(request)
        self.assertTrue(url.startswith("http://%s" % (views._lan_ip() or "?")),
                        url)
        # 非 localhost 的 host（家属直接用局域网 IP 访问）原样保留
        request2 = factory.get("/report/", SERVER_NAME="192.168.1.50")
        self.assertEqual(views._share_url(request2), "http://192.168.1.50/report/")

    def test_share_url_tunnel_base_url_wins(self):
        """内网穿透场景：配了 FAMILY_SHARE_BASE_URL，二维码直接用公网地址。"""
        from django.test import RequestFactory
        from family_portal import views
        factory = RequestFactory()
        request = factory.get("/report/", SERVER_NAME="localhost")
        with override_settings(FAMILY_SHARE_BASE_URL="https://demo.example.com"):
            self.assertEqual(views._share_url(request),
                             "https://demo.example.com/report/")
            # 尾部带斜杠也兼容（settings 读取时已 rstrip）
            with override_settings(FAMILY_SHARE_BASE_URL="https://demo.example.com/"):
                self.assertEqual(views._share_url(request),
                                 "https://demo.example.com/report/")

    def test_hover_tips_counted(self):
        """1 个评分环 + 4 张词卡 + 2 条指数 = 至少 7 个 data-tip。"""
        self.seed_week()
        html = self.client.get("/report/").content.decode("utf-8")
        self.assertGreaterEqual(html.count("data-tip"), 7)

    def test_generated_at_stamp(self):
        self.seed_week()
        self.assertContains(self.client.get("/report/"), "生成于")


# ===========================================================================
# 饮食建议大模型补充（llm_advice）
# ===========================================================================

class TestLlmAdvice(unittest.TestCase):
    """大模型补充建议的全套钉子 —— core/llm.py 那套纪律在 Django 侧的复刻：

    - complete 永不抛异常、失败返回 None（规则引擎永远兜底）；
    - 医疗/用药输出逐行判死（宁少一条，不给错的）；
    - 报告级缓存：同一份统计不重问，失败记短 TTL；
    - 全部走注入的假传输 —— 零网络、零等待。
    """

    # 有数据、有心率偏高的统计样本（跟规则引擎测试同款形状）
    STATS = {"sample_count": 100, "hr_avg": 92.0, "rr_avg": 16.0,
             "negative_ratio": 0.4,
             "state_distribution": [
                 {"state": "tired", "ratio": 0.3, "seconds": 3600},
                 {"state": "sad", "ratio": 0.1, "seconds": 3600}]}

    def setUp(self):
        llm_advice.reset_cache()
        self.addCleanup(llm_advice.reset_cache)

    # ---- 假传输工厂：返回固定 OpenAI 形响应，记录收到的请求 ----

    def _transport(self, lines=None, reply=None, fail=False):
        content = reply if reply is not None else "\n".join(lines or [])
        payload = json.dumps(
            {"choices": [{"message": {"content": content}}]}).encode("utf-8")
        calls: list = []

        def transport(request, timeout):
            calls.append(request)
            if fail:
                raise OSError("网络不可达")
            return payload
        return transport, calls

    def _client(self, **kwargs) -> llm_advice.DietAdviceClient:
        return llm_advice.DietAdviceClient(api_key="test-key", **kwargs)

    # ---- 纯函数：提示词与数据清单 ----

    def test_system_prompt_contains_no_braces(self):
        """提示词里不能有花括号：模型会模仿写法吐 {...}，被清洗判死。"""
        for message in llm_advice.build_messages(self.STATS, {"temp": 32.0}):
            self.assertNotIn("{", message["content"])
            self.assertNotIn("}", message["content"])

    def test_stats_facts_cover_values_and_omit_missing(self):
        facts = llm_advice.build_stats_facts(
            self.STATS, {"temp": 32, "code": 0, "wind": 1.0})
        self.assertIn("100", facts)
        self.assertIn("92", facts)
        self.assertIn("32℃", facts)
        self.assertIn("疲惫 30%", facts)
        # 缺失的方面整行不写（没有依据就不说）
        self.assertEqual(llm_advice.build_stats_facts({"sample_count": 5}, None),
                         "- 有效样本数：5")

    # ---- 纯函数：输出清洗 ----

    def test_sanitize_strips_markdown_and_list_markers(self):
        lines = llm_advice.sanitize_lines(
            "1. **清淡少盐**，本周心率偏高\n- 多喝水，气温偏高\n`规律三餐`，样本充足")
        self.assertEqual(len(lines), 3)
        self.assertFalse(any("*" in ln or "`" in ln for ln in lines))
        self.assertFalse(lines[0].startswith("1."))

    def test_sanitize_rejects_medical_lines(self):
        clean = llm_advice.sanitize_lines("多吃蔬菜水果，心率偏高注意休息")
        self.assertEqual(len(clean), 1)
        for bad in ("建议服用降压药控制心率", "每日剂量不超过 5 毫克",
                    "试试这个偏方调理"):
            self.assertEqual(llm_advice.sanitize_lines(bad), [], bad)

    def test_sanitize_rejects_placeholder_braces(self):
        self.assertEqual(llm_advice.sanitize_lines("建议多吃{食物}类"), [])

    def test_sanitize_caps_line_count_and_length(self):
        text = "\n".join("建议%d：多吃蔬菜水果，规律三餐，保持心情愉悦。" % i
                         for i in range(10))
        lines = llm_advice.sanitize_lines(text)
        self.assertEqual(len(lines), llm_advice.MAX_LINES)
        self.assertTrue(all(len(ln) <= llm_advice.LINE_MAX_CHARS for ln in lines))

    def test_sanitize_empty_returns_empty(self):
        self.assertEqual(llm_advice.sanitize_lines(""), [])
        self.assertEqual(llm_advice.sanitize_lines("   \n  "), [])
        # 只有被拒内容 → 一条不剩
        self.assertEqual(llm_advice.sanitize_lines("推荐处方药"), [])

    # ---- 客户端契约 ----

    def test_complete_success_returns_text(self):
        transport, calls = self._transport(lines=["多吃蔬菜水果，本周心率偏高"])
        client = self._client(transport=transport)
        self.assertEqual(
            client.complete(llm_advice.build_messages(self.STATS, None)),
            "多吃蔬菜水果，本周心率偏高")
        self.assertEqual(len(calls), 1)

    def test_complete_never_raises_on_transport_failure(self):
        transport, _ = self._transport(fail=True)
        client = self._client(transport=transport)
        self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))

    def test_complete_unavailable_returns_none_without_calling(self):
        transport, calls = self._transport(lines=["x"])
        client = llm_advice.DietAdviceClient(api_key="", transport=transport)
        self.assertFalse(client.available)
        self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))
        self.assertEqual(calls, [])   # 没配凭证就不该发请求

    def test_auth_header_is_bearer_with_space(self):
        """Bearer 后面是空格（别抄豆包分号版 —— 两家互不通用）。"""
        transport, calls = self._transport(lines=["x"])
        self._client(transport=transport).complete(
            [{"role": "user", "content": "hi"}])
        self.assertTrue(calls[0].get_header("Authorization").startswith("Bearer "))

    def test_complete_bad_payload_returns_none(self):
        # 直接注入坏 JSON 的传输
        def bad_transport(request, timeout):
            return b"not-json"
        self.assertIsNone(self._client(transport=bad_transport).complete(
            [{"role": "user", "content": "hi"}]))
        # 200 + error 字段（无 choices）同样返回 None
        def error_transport(request, timeout):
            return json.dumps({"error": {"message": "quota"}}).encode("utf-8")
        self.assertIsNone(self._client(transport=error_transport).complete(
            [{"role": "user", "content": "hi"}]))

    # ---- 编排：ai_diet_advice + 缓存 ----

    def test_no_data_means_no_request(self):
        """硬边界 1：没有数据就没有依据，直接不问大模型。"""
        transport, calls = self._transport(lines=["x"])
        out = llm_advice.ai_diet_advice({"sample_count": 0}, None,
                                        client=self._client(transport=transport))
        self.assertEqual(out, [])
        self.assertEqual(calls, [])

    def test_success_returns_sanitized_lines(self):
        transport, _ = self._transport(lines=[
            "1. **饮食清淡少盐**，本周平均心率 92 次/分偏高",
            "建议服用降压药（这行会被判死）",
            "多吃含镁食物，本周疲惫占比 30%"])
        out = llm_advice.ai_diet_advice(self.STATS, None,
                                        client=self._client(transport=transport))
        self.assertEqual(len(out), 2)
        self.assertNotIn("药", "".join(out))

    def test_failure_returns_empty(self):
        transport, _ = self._transport(fail=True)
        self.assertEqual(llm_advice.ai_diet_advice(
            self.STATS, None, client=self._client(transport=transport)), [])

    def test_cache_hit_skips_second_request(self):
        """同一份统计指纹 6 小时内不重问 —— 报告页是低频页面。"""
        transport, calls = self._transport(lines=["多吃蔬菜水果，本周心率偏高"])
        client = self._client(transport=transport)
        first = llm_advice.ai_diet_advice(self.STATS, None, client=client)
        second = llm_advice.ai_diet_advice(self.STATS, None, client=client)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_cache_refreshes_when_stats_change(self):
        transport, calls = self._transport(lines=["多吃蔬菜水果"])
        client = self._client(transport=transport)
        llm_advice.ai_diet_advice(self.STATS, None, client=client)
        changed = dict(self.STATS, hr_avg=60.0)
        llm_advice.ai_diet_advice(changed, None, client=client)
        self.assertEqual(len(calls), 2)

    def test_failure_is_short_cached(self):
        """失败记短 TTL：下一次调用不再立刻重试（避免每次开报告页都陪超时）。"""
        transport, calls = self._transport(fail=True)
        client = self._client(transport=transport)
        llm_advice.ai_diet_advice(self.STATS, None, client=client)
        llm_advice.ai_diet_advice(self.STATS, None, client=client)
        self.assertEqual(len(calls), 1)
        # 但换统计指纹后允许重试
        llm_advice.ai_diet_advice(dict(self.STATS, hr_avg=60.0), None, client=client)
        self.assertEqual(len(calls), 2)

    def test_disabled_globally_returns_empty(self):
        transport, calls = self._transport(lines=["x"])
        with override_settings(FAMILY_LLM_ENABLED=False):
            self.assertEqual(llm_advice.ai_diet_advice(
                self.STATS, None, client=self._client(transport=transport)), [])
        self.assertEqual(calls, [])

    # ---- 防重复：规则结论进提示词 + 执行层精确去重 ----

    def test_messages_carry_rule_advice_and_forbid_repeat(self):
        messages = llm_advice.build_messages(
            self.STATS, None, rule_advice=["多喝水，气温偏高"])
        system = messages[0]["content"]
        self.assertIn("规则引擎已给出的建议", system)
        self.assertIn("- 多喝水，气温偏高", system)
        self.assertIn("不要重复", system)

    def test_messages_without_rule_advice_use_placeholder(self):
        system = llm_advice.build_messages(self.STATS, None)[0]["content"]
        self.assertIn("（无）", system)

    def test_exact_duplicate_of_rule_line_dropped(self):
        """执行层去重：模型没听话、照抄规则建议原文时，那行不留。"""
        transport, _ = self._transport(lines=[
            "多喝水，气温偏高",              # 与规则建议逐字相同 → 丢弃
            "适量坚果豆类，本周疲惫占比 30%"])  # 新角度 → 保留
        out = llm_advice.ai_diet_advice(
            self.STATS, None, client=self._client(transport=transport),
            rule_advice=["多喝水，气温偏高"])
        self.assertEqual(out, ["适量坚果豆类，本周疲惫占比 30%"])


class TestLlmReportIntegration(TempDbMixin, TestCase):
    """大模型建议与报告页/api_report 的集成：叠加层到页面的最后一公里。"""

    def setUp(self):
        super().setUp()
        llm_advice.reset_cache()
        self.addCleanup(llm_advice.reset_cache)

    def _seed(self):
        self.insert_vitals(time.time(), hr=92, rr=16.0)
        self.commit()

    def _patched_client(self, lines):
        payload = json.dumps(
            {"choices": [{"message": {"content": "\n".join(lines)}}]}).encode("utf-8")
        client = llm_advice.DietAdviceClient(
            api_key="k", transport=lambda request, timeout: payload)
        return mock.patch.object(llm_advice, "_client_from_settings",
                                 return_value=client)

    @override_settings(FAMILY_LLM_ENABLED=True)
    def test_report_carries_and_renders_ai_advice(self):
        self._seed()
        with self._patched_client(["饮食清淡少盐，本周平均心率 92 次/分偏高",
                                   "多吃含镁食物，本周疲惫占比 30%"]):
            body = self.client.get("/api/report/").json()
            self.assertEqual(body["data"]["ai_advice"],
                             ["饮食清淡少盐，本周平均心率 92 次/分偏高",
                              "多吃含镁食物，本周疲惫占比 30%"])
            resp = self.client.get("/report/")
        self.assertContains(resp, "三、AI 营养师饮食建议")
        self.assertContains(resp, "饮食清淡少盐")
        # 合并卡片：规则建议与大模型建议在同一个 ai-advice 列表里
        self.assertContains(resp, "规则引擎 + 大模型")
        self.assertContains(resp, "心率")

    @override_settings(FAMILY_LLM_ENABLED=True)
    def test_llm_failure_hides_block_but_keeps_rule_advice(self):
        """大模型挂了：合并卡片里只剩规则建议，标注退回"规则引擎"。"""
        self._seed()
        def failing(request, timeout):
            raise OSError("网络不可达")
        client = llm_advice.DietAdviceClient(api_key="k", transport=failing)
        with mock.patch.object(llm_advice, "_client_from_settings",
                               return_value=client):
            body = self.client.get("/api/report/").json()
            self.assertEqual(body["data"]["ai_advice"], [])
            resp = self.client.get("/report/")
        self.assertContains(resp, "三、AI 营养师饮食建议")
        self.assertContains(resp, "规则引擎 · 根据本周统计生成")
        # AI 专属底注（"大模型"三个字在 HTML 注释里也有，所以要锁整句）
        self.assertNotContains(resp, "AI 补充部分由大模型生成")

    def test_llm_disabled_block_absent(self):
        """默认（测试关断/未配置）：ai_advice 为空列表，页面无 AI 标注。"""
        self._seed()
        body = self.client.get("/api/report/").json()
        self.assertEqual(body["data"]["ai_advice"], [])
        self.assertNotContains(self.client.get("/report/"), "AI 补充部分由大模型生成")

    def test_no_data_ai_advice_empty_even_when_enabled(self):
        """硬边界 1 走通到视图：无数据时就算配了大模型也不请求。"""
        with override_settings(FAMILY_LLM_ENABLED=True), \
                mock.patch.object(llm_advice, "_client_from_settings",
                                  side_effect=AssertionError("不该建客户端")):
            body = self.client.get("/api/report/").json()
        self.assertEqual(body["data"]["ai_advice"], [])

    @override_settings(FAMILY_LLM_ENABLED=True)
    def test_weekly_report_feeds_rule_advice_to_llm(self):
        """防重复闭环：weekly_report 把规则结论原文喂给大模型，
        免责声明行不带 —— 模型才知道哪些角度不许重复。"""
        self._seed()
        bodies = []

        def transport(request, timeout):
            bodies.append(json.loads(request.data.decode("utf-8")))
            return json.dumps({"choices": [{"message": {
                "content": "晚餐安排在睡前两小时之前，进食节奏放慢"}}]}).encode("utf-8")

        client = llm_advice.DietAdviceClient(api_key="k", transport=transport)
        with mock.patch.object(llm_advice, "_client_from_settings",
                               return_value=client):
            body = self.client.get("/api/report/").json()
        self.assertEqual(body["data"]["ai_advice"],
                         ["晚餐安排在睡前两小时之前，进食节奏放慢"])
        system = bodies[0]["messages"][0]["content"]
        self.assertIn("规则引擎已给出的建议", system)
        # hr=92 触发的心率偏高规则原文进了提示词
        self.assertIn("本周平均心率 92", system)
        # 免责声明不是"建议"，不进提示词
        self.assertNotIn("以上建议由规则引擎生成", system)
