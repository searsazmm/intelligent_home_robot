# -*- coding: utf-8 -*-
"""健康报告与饮食建议 —— **规则引擎**，不是医疗意见。

规则全部透明、可解释，输入是 :func:`stats.overall_stats` 的统计结果和
天气（30 分钟内存缓存）。没配网 / 没数据时逐条降级：

- 没有体征数据 → 明说"无法给出针对性建议"，不编造；
- 没有天气 → 跳过天气相关规则，其余照常；
- 页面固定展示免责声明。

二期可以在这里接 backend_B 的大模型（core/llm.py），把规则输出当
提示词骨架 —— 接口（输入 stats/weather，输出建议列表）已经留好。
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime

from . import stats

# --------------------------------------------------------------------------
# 天气：与 backend_B/core/weather.py 同一 open-meteo 端点、同一 {temp,code,wind}
# 契约（同形复用，刻意不跨模块 import —— 两个进程两套依赖边界）。
# --------------------------------------------------------------------------

#: 内存缓存有效期（秒）。家属端是低频访问，30 分钟足够新鲜，
#: 还能把外网请求压到一天几十次。
_WEATHER_CACHE_TTL = 30 * 60.0

#: 简单进程内缓存 {"ts": 上次成功时间, "data": {temp,code,wind}}
_weather_cache: dict = {"ts": 0.0, "data": None}

_API_URL = "https://api.open-meteo.com/v1/forecast"
_DEFAULT_LAT, _DEFAULT_LON = 30.8997, 103.5639   # 与 backend_B config §16 默认一致


def fetch_weather(timeout: float = 4.0, lat: float | None = None,
                  lon: float | None = None) -> dict | None:
    """取当前天气（带 30 分钟缓存）。失败返回 None，调用方走兜底。"""
    now = time.time()
    if _weather_cache["data"] is not None and now - _weather_cache["ts"] < _WEATHER_CACHE_TTL:
        return _weather_cache["data"]

    lat = _DEFAULT_LAT if lat is None else lat
    lon = _DEFAULT_LON if lon is None else lon
    url = (
        f"{_API_URL}?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weather_code,wind_speed_10m"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current = payload.get("current") or {}
        data = {
            "temp": round(float(current["temperature_2m"])),
            "code": int(current["weather_code"]),
            "wind": float(current.get("wind_speed_10m") or 0.0),
        }
    except Exception:
        # 断网 / 超时 / 坏形态：沿用旧缓存（哪怕过期）也比没有强
        return _weather_cache["data"]

    _weather_cache["ts"] = now
    _weather_cache["data"] = data
    return data


# --------------------------------------------------------------------------
# 饮食建议规则引擎
# --------------------------------------------------------------------------

#: 触发阈值与依据。数值取自公开的健康科普常识区间（成年人静息心率
#: 60~100、静息呼吸 12~20 次/分），**只作生活方式提示，不构成医疗意见**。
HR_HIGH = 85.0
HR_LOW = 55.0
RR_HIGH = 20.0

DISCLAIMER = "以上建议由规则引擎生成，仅作生活方式参考，不构成医疗意见；如有不适请就医。"


def diet_advice(stats_result: dict | None, weather: dict | None) -> list[str]:
    """根据一周统计 + 天气生成饮食建议（每条都自带"为什么"）。

    设计原则：**没有依据就不说**。某条规则的前提数据缺失时直接跳过，
    绝不为了凑条数编建议 —— 家属读到的每一条都有数据支撑。
    """
    advice: list[str] = []
    if stats_result is None or not stats_result.get("sample_count"):
        return ["本周无有效体征数据，无法给出针对性饮食建议 —— 机器人运行并采集到数据后，这里会自动更新。"]

    hr = stats_result.get("hr_avg")
    if hr is not None:
        if hr >= HR_HIGH:
            advice.append(
                f"本周平均心率 {hr} 次/分，偏高。建议饮食清淡少盐、少浓茶咖啡，"
                "规律三餐，避免饱餐后立即休息。")
        elif hr <= HR_LOW:
            advice.append(
                f"本周平均心率 {hr} 次/分，偏低。建议营养均衡、适量补充优质蛋白"
                "与含铁食物（瘦肉、菠菜），少量多餐。")

    rr = stats_result.get("rr_avg")
    if rr is not None and rr >= RR_HIGH:
        advice.append(
            f"本周平均呼吸 {rr} 次/分，偏快。建议避免过饱、细嚼慢咽，"
            "晚餐七八分饱，饭后散步片刻再休息。")

    negative = stats_result.get("negative_ratio") or 0.0
    distribution = stats_result.get("state_distribution") or []
    tired = next((d["ratio"] for d in distribution if d["state"] == "tired"), 0.0)
    sad = next((d["ratio"] for d in distribution if d["state"] == "sad"), 0.0)
    if tired >= 0.3:
        advice.append(
            f"本周约 {tired:.0%} 的时间呈疲惫状态。建议晚餐清淡、睡前 2 小时不再进食，"
            "适量补充含镁食物（坚果、豆类、深色蔬菜）。")
    if sad >= 0.3:
        advice.append(
            f"本周约 {sad:.0%} 的时间情绪低落。建议三餐规律，适量摄入富含"
            "Omega-3（深海鱼）与 B 族维生素（全谷物）的食物。")
    if negative >= 0.5 and tired < 0.3 and sad < 0.3:
        advice.append(
            f"本周疲惫与低落合计约 {negative:.0%} 的时间。建议规律三餐、少油少糖，"
            "白天适量晒太阳有助于状态改善。")

    temp = weather.get("temp") if weather else None
    if temp is not None:
        if temp >= 30:
            advice.append(f"近期气温 {temp}℃，偏热：注意补水，饮食清淡解暑，少食生冷。")
        elif temp <= 5:
            advice.append(f"近期气温 {temp}℃，偏冷：宜温热饮食，适量热汤热粥，避免空腹外出。")

    if not advice:
        advice.append("本周各项指标均在常见范围内，保持目前均衡的饮食习惯即可。")
    advice.append(DISCLAIMER)
    return advice


# --------------------------------------------------------------------------
# 健康摘要：整体评分 + 四张状态词卡 + 指数条（报告页 hero 区）。
# 规则透明可解释：起点 100 分，按偏离扣分，**没有数据就不评分**。
# 只作生活方式观察参考，不构成医疗意见。
# --------------------------------------------------------------------------

def health_summary(stats_result: dict | None) -> dict | None:
    """由一周统计派生健康摘要。无有效数据返回 None（页面走"暂无"兜底）。

    评分规则（0~100，起点 100）：
    - 平均心率偏离常见区间（60~85）：每偏 2 次/分扣 1 分，上限 15；
    - 平均呼吸偏离常见区间（12~20）：每偏 1 次/分扣 1 分，上限 15；
    - 疲惫 + 低落合计占比 ×30 直接扣分（状态是比体征更直接的信号）。
    扣完下限 40 —— 观察性评分不应 dramatic 地趋零吓到家属，页面上
    永远带着"仅供参考"的说明。
    """
    if stats_result is None or not stats_result.get("sample_count"):
        return None

    hr = stats_result.get("hr_avg")
    rr = stats_result.get("rr_avg")
    negative = stats_result.get("negative_ratio") or 0.0
    distribution = stats_result.get("state_distribution") or []
    tired = next((d["ratio"] for d in distribution if d["state"] == "tired"), 0.0)

    score = 100.0
    if hr is not None:
        if hr > 85:
            score -= min((hr - 85) / 2.0, 15.0)
        elif hr < 60:
            score -= min((60 - hr) / 2.0, 15.0)
    if rr is not None:
        if rr > 20:
            score -= min(rr - 20, 15.0)
        elif rr < 12:
            score -= min(12 - rr, 15.0)
    score -= negative * 30.0
    score = int(round(max(score, 40.0)))

    hr_word = _hr_word(hr)
    rr_word = _rr_word(rr)
    mood_word = _mood_word(negative)
    energy_word = _energy_word(tired)

    overall = "平稳"
    if score >= 85:
        overall = "平稳"
    elif score >= 70:
        overall = "需要留意"
    else:
        overall = "建议关注"

    return {
        "score": score,
        "overall": overall,
        "score_note": "健康状态综合分 · 仅供观察参考，不同于疾病诊断",
        # 评分环 hover 提示：把扣分规则原样告诉家属（透明可解释）
        "score_tip": (
            "评分规则：起点 100 分。"
            "平均心率偏离 60~85 次/分、平均呼吸偏离 12~20 次/分，"
            "按偏离程度扣分（各上限 15 分）；"
            "疲惫与低落合计占比 ×30 扣分；下限 40 分。仅供观察参考。"
        ),
        "tiles": [
            {"name": "心率状态", "word": hr_word[0], "level": hr_word[1], "tip": hr_word[2],
             "desc": "成年人静息参考 60~85 次/分"},
            {"name": "呼吸状态", "word": rr_word[0], "level": rr_word[1], "tip": rr_word[2],
             "desc": "成年人静息参考 12~20 次/分"},
            {"name": "情绪状态", "word": mood_word[0], "level": mood_word[1], "tip": mood_word[2],
             "desc": "由疲惫/低落状态占比综合评估"},
            {"name": "活力状态", "word": energy_word[0], "level": energy_word[1], "tip": energy_word[2],
             "desc": "由疲惫状态占比评估"},
        ],
        # 指数条：1 - 占比，0~1（前端模板直显百分数）
        "gauges": [
            {"name": "情绪元气指数", "value": round(max(0.0, 1.0 - negative), 3),
             "note": "低落/疲惫占比越低越充实"},
            {"name": "精神放松指数", "value": round(max(0.0, 1.0 - tired), 3),
             "note": "疲惫占比越低越放松"},
        ],
    }


def _hr_word(hr):
    """返回 (状态词, 等级, hover 提示)。提示带本周实际数值，比固定文案更有用。"""
    if hr is None:
        return ("未检出", "none", "本周没有可用的心率测量值 —— 机器人运行并在镜头前采集到数据后自动更新。")
    if hr > 100:
        return ("过速", "bad", f"本周平均心率 {hr} 次/分，明显高于静息常见上限。建议留意休息与情绪，持续偏高请就医。")
    if hr > 85:
        return ("偏快", "mid", f"本周平均心率 {hr} 次/分，高于静息常见区间 60~85。可能与活动、紧张或咖啡因有关。")
    if hr < 55:
        return ("过缓", "bad", f"本周平均心率 {hr} 次/分，明显低于静息常见下限。如伴头晕乏力请就医。")
    if hr < 60:
        return ("偏缓", "mid", f"本周平均心率 {hr} 次/分，略低于静息常见区间 60~85。经常锻炼者偏缓也可能正常。")
    return ("平稳", "good", f"本周平均心率 {hr} 次/分，处于静息常见区间 60~85 内。")


def _rr_word(rr):
    if rr is None:
        return ("未检出", "none", "本周没有可用的呼吸测量值 —— 机器人运行并在镜头前采集到数据后自动更新。")
    if rr > 24:
        return ("急促", "bad", f"本周平均呼吸 {rr} 次/分，明显高于静息常见上限。如伴胸闷气短请就医。")
    if rr > 20:
        return ("偏快", "mid", f"本周平均呼吸 {rr} 次/分，高于静息常见区间 12~20。建议放慢节奏、深呼吸练习。")
    if rr < 10:
        return ("过缓", "bad", f"本周平均呼吸 {rr} 次/分，明显低于静息常见下限。如伴嗜睡请就医。")
    if rr < 12:
        return ("偏缓", "mid", f"本周平均呼吸 {rr} 次/分，略低于静息常见区间 12~20。")
    return ("平稳", "good", f"本周平均呼吸 {rr} 次/分，处于静息常见区间 12~20 内。")


def _mood_word(negative):
    if negative < 0.15:
        return ("良好", "good", f"本周疲惫与低落合计仅占约 {negative:.0%} 的时间，情绪状态良好。")
    if negative < 0.30:
        return ("平稳", "good", f"本周疲惫与低落合计约 {negative:.0%} 的时间，整体平稳。")
    if negative < 0.50:
        return ("有波动", "mid", f"本周疲惫与低落合计约 {negative:.0%} 的时间，有一定波动。建议多陪伴聊天、规律作息。")
    return ("偏低落", "bad", f"本周疲惫与低落合计约 {negative:.0%} 的时间。建议增加陪伴与户外活动，持续低落请寻求专业帮助。")


def _energy_word(tired):
    if tired < 0.15:
        return ("充沛", "good", f"本周仅约 {tired:.0%} 的时间呈疲惫状态，精力充沛。")
    if tired < 0.30:
        return ("正常", "good", f"本周约 {tired:.0%} 的时间呈疲惫状态，属常见范围。")
    if tired < 0.50:
        return ("易疲惫", "mid", f"本周约 {tired:.0%} 的时间呈疲惫状态。建议关注睡眠质量与白天小憩安排。")
    return ("疲乏", "bad", f"本周约 {tired:.0%} 的时间呈疲惫状态。建议调整作息，持续疲乏请就医排查。")


def weekly_report(conn, weather: dict | None = None) -> dict:
    """组装一周报告：统计 + 建议 + 元信息。报告页（HTML）与 api_report 共用。"""
    stats_result = stats.overall_stats(conn, days=7)
    has_data = bool(stats_result.get("sample_count"))
    series = stats.day_series(conn, days=7)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "has_data": has_data,
        "stats": stats_result if has_data else None,
        "summary": health_summary(stats_result if has_data else None),
        "series": series if has_data else [],
        "weather": weather,
        "diet_advice": diet_advice(stats_result if has_data else None, weather),
    }
