# -*- coding: utf-8 -*-
"""天气投影与口语文案的单测（不联网；fetch_report 的失败兜底只验契约）。"""

import unittest

from core import weather


class TestParseReport(unittest.TestCase):

    def test_projects_openmeteo_current(self):
        report = weather.parse_report({
            "current": {
                "temperature_2m": 22.7,
                "weather_code": 2,
                "wind_speed_10m": 3.4,
            }
        })
        self.assertEqual(report, {"temp": 23, "code": 2, "wind": 3.4})

    def test_bad_shapes_return_none(self):
        self.assertIsNone(weather.parse_report(None))
        self.assertIsNone(weather.parse_report({}))
        self.assertIsNone(weather.parse_report({"current": "x"}))
        self.assertIsNone(weather.parse_report({"current": {"temperature_2m": 20}}))
        self.assertIsNone(
            weather.parse_report({"current": {"temperature_2m": "热",
                                              "weather_code": 0}}))

    def test_missing_wind_defaults_zero(self):
        report = weather.parse_report(
            {"current": {"temperature_2m": 10, "weather_code": 3}})
        self.assertEqual(report["wind"], 0.0)


class TestWeatherText(unittest.TestCase):

    def test_known_codes(self):
        self.assertEqual(weather.weather_text(0), "晴")
        self.assertEqual(weather.weather_text(61), "小雨")
        self.assertEqual(weather.weather_text(95), "雷阵雨")

    def test_unknown_and_garbage(self):
        self.assertEqual(weather.weather_text(999), "天气看不太准")
        self.assertEqual(weather.weather_text("x"), "天气看不太准")


class TestDescribe(unittest.TestCase):
    """天气全文**只报天气**：类型 + 温度 + 感受，不给出行/锻炼建议 ——
    那些只在活动类问句（describe_activity）里回答。"""

    def test_none_is_empty_string(self):
        # 空串是调用方落回"看不到天气"诚实模板的信号
        self.assertEqual(weather.describe_weather(None), "")

    def test_sunny_mild_sentence(self):
        """code 0 temp 23 → 晴（微凉）：城市名 + 晴 + 23度 + 体感宜人。"""
        text = weather.describe_weather({"temp": 23, "code": 0, "wind": 1.0})
        self.assertIn("青城山", text)
        self.assertIn("晴（微凉）", text)
        self.assertIn("23度", text)
        self.assertIn("体感宜人", text)
        self.assertNotIn("帽子", text)      # 建议不进天气回答
        self.assertNotIn("出门", text)

    def test_sunny_hot_sentence(self):
        """code 0 temp 35 → 晴（高温）+ 天挺热；正午建议留给活动问句。"""
        text = weather.describe_weather({"temp": 35, "code": 0, "wind": 1.0})
        self.assertIn("晴（高温）", text)
        self.assertIn("35度", text)
        self.assertIn("天挺热", text)
        self.assertNotIn("避开正午", text)

    def test_overcast_windy(self):
        """code 3 wind 20 → 阴天有风。"""
        text = weather.describe_weather({"temp": 18, "code": 3, "wind": 20.0})
        self.assertIn("阴天有风", text)
        self.assertIn("18度", text)
        self.assertNotIn("山口", text)      # 风的建议在活动回答里

    def test_overcast_calm(self):
        """code 3 wind 5 → 阴天。"""
        text = weather.describe_weather({"temp": 18, "code": 3, "wind": 5.0})
        self.assertIn("阴天", text)
        self.assertNotIn("有风", text)

    def test_heavy_rain(self):
        """code 63 → 大雨 + 凉爽感受；出门建议在活动回答里。"""
        text = weather.describe_weather({"temp": 18, "code": 63, "wind": 0.0})
        self.assertIn("大雨", text)
        self.assertIn("18度", text)
        self.assertNotIn("出门", text)
        self.assertNotIn("室内", text)

    def test_light_rain(self):
        """code 61 → 小雨 + 凉爽感受。"""
        text = weather.describe_weather({"temp": 18, "code": 61, "wind": 0.0})
        self.assertIn("小雨", text)
        self.assertIn("凉爽", text)
        self.assertNotIn("走远路", text)
        self.assertNotIn("拉伸", text)

    def test_fog(self):
        """code 45 → 雾天。"""
        text = weather.describe_weather({"temp": 17, "code": 45, "wind": 0.0})
        self.assertIn("雾天", text)
        self.assertIn("17度", text)
        self.assertNotIn("能见度", text)    # 雾的建议在活动回答里

    def test_cloudy(self):
        """code 2 → 多云 + 凉爽。"""
        text = weather.describe_weather({"temp": 21, "code": 2, "wind": 0.0})
        self.assertIn("多云", text)
        self.assertIn("凉爽", text)
        self.assertNotIn("散步", text)

    def test_no_advice_in_any_type(self):
        """**每种类型**的天气全文都不含出行/锻炼建议 —— 建议只归活动问句。"""
        for code in [0, 1, 2, 3, 45, 61, 63, 95]:
            text = weather.describe_weather(
                {"temp": 20, "code": code, "wind": 5.0})
            for banned in ("记得戴好帽子", "出门", "锻炼", "拉伸", "散步"):
                self.assertNotIn(banned, text,
                                 msg=f"code={code} 混进了建议词「{banned}」")
            self.assertTrue(text.endswith("。"), msg=f"code={code} 句式不完整")


class TestActivityDescribe(unittest.TestCase):
    """活动类问句的专属回答：只讲能不能出门、怎么锻炼。"""

    def test_light_rain_activity(self):
        text = weather.describe_activity({"temp": 18, "code": 61, "wind": 0.0})
        self.assertIn("小雨", text)
        self.assertIn("地滑", text)
        self.assertIn("活动", text)

    def test_cloudy_activity(self):
        text = weather.describe_activity({"temp": 21, "code": 2, "wind": 0.0})
        self.assertIn("多云", text)
        self.assertIn("出门", text)

    def test_activity_and_weather_never_share_advice(self):
        """天气全文与活动回答不共享任何建议措辞 —— 听感上必须一听就是两个回答。"""
        report = {"temp": 20, "code": 61, "wind": 0.0}
        plain, activity = weather.describe_weather(report), weather.describe_activity(report)
        self.assertNotEqual(plain, activity)
        self.assertNotIn("帽子", activity)
        self.assertNotIn("气温", activity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
