# -*- coding: utf-8 -*-
"""天气查询：免 key 的 open-meteo + WMO 天气码中文映射。

为什么自己查而不是走大模型：模型没有实时数据，问「今天天气怎么样」
它只会编一个晴天出来（见 dialogue.py 里 weather 意图的注释）。和
festival/vitals 同一类问题 —— **确定性现答**。

为什么是 open-meteo：
* 免费、不要 key、不要注册 —— 仓库里不会出现任何凭据；
* 返回 JSON、标准库 urllib 直接能拿（B 端运行期零第三方依赖）；
* 用经纬度查询，默认坐标写青城山（config.WEATHER_LAT/LON）。

失败纪律和各处网络调用一致：**查不到就返回 None，调用方落回
"我看不到天气"的诚实模板**，绝不编温度、编阴晴。
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_API_URL = "https://api.open-meteo.com/v1/forecast"

#: WMO 天气解释码 → 中文说法。只列 open-meteo 实际会返回的码，
#: 见 https://open-meteo.com/en/docs（WMO Weather interpretation codes）。
_WMO_TEXT = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "有雾", 48: "有雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "阵雪",
    80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷雨夹冰雹", 99: "雷雨夹冰雹",
}


def weather_text(code) -> str:
    """WMO 码 → 中文；未知码老实说"天气码X"而不是瞎安一个。"""
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return "天气看不太准"
    return _WMO_TEXT.get(code_int, "天气看不太准")


def parse_report(payload: dict) -> Optional[dict]:
    """把 open-meteo 的响应投影成 ``{temp, code, wind}``；坏形态返回 None。

    纯函数、不联网 —— 测试直接喂字典。
    """
    if not isinstance(payload, dict):
        return None
    current = payload.get("current")
    if not isinstance(current, dict):
        return None
    temp = current.get("temperature_2m")
    code = current.get("weather_code")
    if temp is None or code is None:
        return None
    try:
        return {
            "temp": round(float(temp)),
            "code": int(code),
            "wind": float(current.get("wind_speed_10m") or 0.0),
        }
    except (TypeError, ValueError):
        return None


def fetch_report(lat: float, lon: float, timeout: float = 4.0) -> Optional[dict]:
    """查一次实时天气。任何失败（断网 / 超时 / 坏响应）都返回 None。"""
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "timezone": "Asia/Shanghai",
    })
    url = "%s?%s" % (_API_URL, query)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read()
        return parse_report(json.loads(raw.decode("utf-8")))
    except Exception:
        logger.warning("天气查询失败，按'看不到天气'处理", exc_info=True)
        return None


def _advice(report: dict) -> str:
    """按天气给一句**非医疗**的出门提醒。宁可朴素，别下指令。"""
    code = report["code"]
    temp = report["temp"]
    if code in (95, 96, 99):
        return "外头在打雷，尽量别出门"
    if 61 <= code <= 67 or code in (80, 81, 82):
        return "有雨，出门带伞，路滑慢些走"
    if code in (51, 53, 55, 56, 57):
        return "飘着细雨，出门带伞稳妥些"
    if (71 <= code <= 77) or code in (85, 86):
        return "外头有雪，多穿点，留神脚下打滑"
    if code in (45, 48):
        return "雾大能见度低，出门慢些走"
    if temp <= 5:
        return "天冷，出门多添件衣裳"
    if temp >= 32:
        return "天热，多喝水，别在日头底下久待"
    if code == 0 or code == 1:
        return "天气不错，出门正合适"
    return "温度还行，出门正合适"


#: 活动类问句的**专属**话术（8 种类型 → (出门话术, 锻炼话术)）。
#: 天气问句（:func:`describe_weather`）只报天气本身；"能不能出门、
#: 怎么锻炼"全部由这里回答 —— 老人先问"天气怎么样"、再问"适合出门
#: 散步吗"，两次回答一个报天气、一个答能不能去，互不重复。每条措辞
#: 口语化、针对散步场景。
_ACTIVITY_TEMPLATES = {
    "多云":       ("外头挺舒服的，正适合出门溜达", "您出去走一圈，回来准松快"),
    "阴天":       ("可以出门走走，别走太远", "慢悠悠遛个弯就挺好"),
    "小雨":       ("这会儿地滑，别出去走远", "您就在屋里慢慢活动活动，等雨停了再出去"),
    "大雨":       ("雨下着呢，千万别出门", "在屋里甩甩胳膊、原地踏踏步就行"),
    "阴天有风":   ("风不小，出门多穿一件", "就在近处短短遛一小段吧"),
    "晴（高温）": ("日头正毒，这会儿别出门", "等傍晚凉快了再出门遛弯"),
    "雾天":       ("雾大看不清路，出门不安全", "今天就在家歇着，改天再锻炼"),
    "晴（微凉）": ("外头凉丝丝的，出门正舒坦", "您出去慢慢走一走、打套太极都行"),
}


def _temp_feel(temp: int) -> str:
    """根据实际温度给一句感受描述 —— 温度感受是「温度决定语义」，
    和天气类型无关：21 度不管晴阴都算"凉爽"，5 度不管晴阴都算"天挺冷"。

    阈值参考体感常识（非医疗标准）：

    * ≤ 5°C：天挺冷
    * 6-10°C：偏冷
    * 11-15°C：偏凉
    * 16-22°C：凉爽
    * 23-26°C：体感宜人
    * 27-30°C：偏热
    * > 30°C：天挺热
    """
    if temp <= 5:
        return "天挺冷"
    if temp <= 10:
        return "偏冷"
    if temp <= 15:
        return "偏凉"
    if temp <= 22:
        return "凉爽"
    if temp <= 26:
        return "体感宜人"
    if temp <= 30:
        return "偏热"
    return "天挺热"


def _classify(code: int, temp: int, wind: float) -> str:
    """WMO 天气码 + 温度 + 风速 → 8 种类型之一。

    映射规则（覆盖 open-meteo 实际会返回的码）：

    * **晴/多云**（0,1,2）：code 0/1 看温度分「晴（高温）」≥30°C / 「晴（微凉）」<30°C；code 2 → 多云。
    * **阴天**（3）：看风速 ≥15 km/h 分「阴天有风」/ 「阴天」。
    * **雾天**（45,48）：直接归「雾天」。
    * **小雨**（51-57,61）：毛毛雨、小雨。
    * **大雨**（63,65,66,67,80,81,82,95,96,99）：中雨、大雨、阵雨、暴雨、雷阵雨。
    * **雪**（71-77,85,86）：8 种里没有"雪天"，归到「雾天」（最接近的"不适合出门"语义）。
    * 未知码 → 多云（兜底，不瞎安一个）。
    """
    if code in (45, 48):
        return "雾天"
    if code in (51, 53, 55, 56, 57, 61):
        return "小雨"
    if code in (63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        return "大雨"
    if code in (71, 73, 75, 77, 85, 86):
        return "雾天"
    if code in (0, 1):
        return "晴（高温）" if temp >= 30 else "晴（微凉）"
    if code == 2:
        return "多云"
    if code == 3:
        return "阴天有风" if wind >= 15.0 else "阴天"
    return "多云"


def describe_weather(report: Optional[dict], city: str = "青城山") -> str:
    """把投影后的读数翻成一句能直接念的话；``report`` 为 None 返回空串，
    调用方据此落回诚实兜底模板。

    输出格式（8 种类型之一，温度用实时数据）::

        青城山现在是[类型]，气温[实时温度]度，[温度感受]。

    **只报天气本身** —— 出行建议、锻炼方式一律不给：老人问"天气怎么样"
    听的是天气；"能不能出门、怎么锻炼"由 :func:`describe_activity` 在
    活动类问句（"适合出门散步吗"）里回答。两类问句两种回答，互不掺杂。

    类型由 :func:`_classify` 按 WMO 码 + 温度 + 风速判定，温度感受由
    :func:`_temp_feel` 根据**实际温度**动态生成（21 度说"凉爽"、5 度说
    "天挺冷"）。
    """
    if not report:
        return ""
    code = int(report["code"])
    temp = int(report["temp"])
    wind = float(report.get("wind", 0.0))
    kind = _classify(code, temp, wind)
    feel = _temp_feel(temp)
    return "%s现在是%s，气温%d度，%s。" % (city, kind, temp, feel)


def describe_activity(report: Optional[dict], city: str = "青城山") -> str:
    """活动类问句（"适合出门散步吗"）的**专属**回答；``report`` 为 None 返回空串。

    和 :func:`describe_weather` 的分工：天气问句只报天气本身（类型 +
    温度 + 感受），活动问句真正想听的是"能不能去"——这里只答出门话术
    + 锻炼话术，不报气温，措辞口语化（:data:`_ACTIVITY_TEMPLATES`）。
    两类问句两种回答，互不掺杂也互不重复。

    顺带降低了 TTS 断流风险：40+ 字长句流式合成中途超时的概率远高于
    20 字短句（实测两次 TimeoutError 都发生在天气全文上）。
    """
    if not report:
        return ""
    code = int(report["code"])
    temp = int(report["temp"])
    wind = float(report.get("wind", 0.0))
    kind = _classify(code, temp, wind)
    go_out, exercise = _ACTIVITY_TEMPLATES[kind]
    return "%s现在%s，%s，%s。" % (city, kind, go_out, exercise)
