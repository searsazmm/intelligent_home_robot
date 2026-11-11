# -*- coding: utf-8 -*-
"""饮食建议的大模型补充（DeepSeek）—— 叠加在规则引擎之上，不是替换。

--------------------------------------------------------------------------
与 backend_B/core/llm.py 的关系：同一套纪律，两份独立实现
--------------------------------------------------------------------------
对话引擎那边（core/llm.py）已经立过规矩：

    规则引擎 —— 透明可解释、零依赖、永远有话说，**永远是兜底**
    大模型    —— 生成更个性化的补充，挂了只是少一层

家属报告这边的 ``report.diet_advice`` 就是规则引擎，本模块是在它后面
追加一组"AI 营养师补充建议"。complete 契约与那边完全一致：**永不抛
异常、失败返回 None**，调用方拿到空列表就静默省略整个区块 —— 页面上
永远还有规则引擎的建议兜着，没有任何一处会因为大模型出问题而开天窗。

刻意**不跨模块 import** backend_B/core/llm.py：family_portal 是独立
的 Django 进程，与 backend_B 只有 SQLite 一个契约边界（天气模块
report.fetch_weather 同理，同形复用而不是共享代码）。凭证环境变量
B_DEEPSEEK_TOKEN 两边通用（同机部署，只配一次）。

--------------------------------------------------------------------------
报告级缓存
--------------------------------------------------------------------------
报告页与 api_report 共用 weekly_report，但它是低频页面：同一份周统计
下大模型的输出没必要每次重算。进程内缓存按"统计指纹"键控：

- 命中（TTL 默认 6 小时）→ 直接复用，零外网请求；
- 失败也记短 TTL（30 分钟）—— 否则每次打开报告页都要陪大模型超时
  一遍，家属端的打印体验会变得不可预期。

三条硬边界（提示词是劝导，清洗是执行）：
1. 无有效体征数据 → 不请求（没有依据就不编）；
2. 医疗/用药类输出 → 逐行判死（宁可少一条，不能给错）；
3. 输出必须逐行清洗（剥 Markdown/emoji/网址/序号），洗完一条不剩
   → 返回空列表，前端整个区块不渲染。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence

from django.conf import settings

logger = logging.getLogger(__name__)

# ==========================================================================
# 纯函数：请求构造与响应解析（与 core/llm.py 同一套拆法，便于零网络单测）
# ==========================================================================


def build_chat_request(
    messages: Sequence[Dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """构造 /chat/completions 请求体。stream 固定 False（报告页等整段）。"""
    return {
        "model": model,
        "messages": [dict(m) for m in messages],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }


def parse_chat_response(raw: bytes) -> str:
    """从响应体取回复文本；取不到抛 ValueError（调用方据此落回兜底）。

    DeepSeek 的错误也常走 200 + error 字段，所以"没有 choices"必须查。
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("响应不是合法 JSON") from exc
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise ValueError("响应里没有 choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise ValueError("choices[0] 里没有 message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("回复内容为空")
    return content.strip()


# ==========================================================================
# 纯函数：提示词与输出清洗
# ==========================================================================

#: 系统提示词。**整段不能出现花括号** —— 模型会模仿提示词的写法，
#: 吐出一个 ``{...}`` 就会被清洗判死，于是整条大模型路被静默废掉。
#: 所以这里用 % 格式化，不用 f-string。有测试钉这一条。
_PROMPT_TEMPLATE = """你是一名面向家属健康报告的「AI 营养师」助手，服务对象是独居老人的家属。

任务：根据下面给出的本周体征与状态统计，输出几条下周的饮食建议。

硬性边界（违反任何一条，整段输出作废）：
- 只给生活方式与日常饮食搭配建议。不诊断疾病、不推荐药物、不提剂量、
  不提偏方、不评价保健品疗效，全文不得出现"药"字。
- 只依据给出的统计数据，数据里没有的方面不写、不推测、不编造。
- 只输出 %s 条中文建议，每条独占一行，每行不超过 %s 字，行首不要序号或符号。
- 每条先给具体建议，再用一句话点出依据（对应哪个统计数据）。
- **规则建议已覆盖的角度不要重复**：只补充它没提到的饮食方面
  （比如食材搭配、饮水安排、进食时机、烹饪方式）。如果数据里没有
  别的值得说的，宁少输出几条，也不要换个说法复读规则建议。
- 不输出 Markdown、表情符号、网址、括号补充、引号或任何符号装饰。

本周统计数据：
%s

规则引擎已给出的建议（这些角度不要重复）：
%s
"""

#: 每条建议的最大字数。报告页是打印导向，一行太长会折得难看。
LINE_MAX_CHARS = 80

#: 最多保留几条。规则引擎已有兜底，大模型只做"画龙点睛"的补充。
MAX_LINES = 4

#: 请求时要求几条。多要两条给清洗损耗留余量。
REQUEST_LINES = 4

#: 医疗/用药建议。命中整行判死 —— 这是给家属参考的，宁可少说不能说错。
#: 与 core/llm.py 的 _MEDICAL_RE 同思路，词表按"饮食建议"场景调整。
_MEDICAL_RE = re.compile(
    r"药|剂量|毫克|mg|处方|偏方|就医前|替代医嘱|疗效",
    re.IGNORECASE)

#: emoji / 象形符号。范围避开 CJK，不会误伤汉字。
_EMOJI_RE = re.compile(
    "["
    "🀀-🫿"
    "☀-➿"
    "⬀-⯿"
    "©®™ℹ〰〽㊗㊙"
    "]+")

#: 行首列表符号 / Markdown 装饰。
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*·•]|\d+[.、)])\s+")
_MARKDOWN_RE = re.compile(r"```|~~~|`|\*\*|__|\*|~~|^#{1,6}\s*|^>\s*")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[。！？!?]")


def build_stats_facts(stats_result: dict, weather: dict | None) -> str:
    """把统计结果翻成提示词里的"人话数据清单"。缺失的方面整行不写。

    纯函数：输入输出都是 str/dict，零网络零设置，可直接单测。
    """
    facts: List[str] = []
    count = stats_result.get("sample_count")
    if count:
        facts.append(f"- 有效样本数：{count}")
    hr = stats_result.get("hr_avg")
    if hr is not None:
        facts.append(f"- 本周平均心率：{hr} 次/分（成年人静息常见区间 60~100，90 以上提示留意）")
    rr = stats_result.get("rr_avg")
    if rr is not None:
        facts.append(f"- 本周平均呼吸：{rr} 次/分（成年人静息常见区间 12~20）")
    negative = stats_result.get("negative_ratio")
    if negative is not None:
        distribution = stats_result.get("state_distribution") or []
        tired = next((d["ratio"] for d in distribution if d["state"] == "tired"), None)
        sad = next((d["ratio"] for d in distribution if d["state"] == "sad"), None)
        line = f"- 本周负向状态（疲惫+低落）合计占比：{negative:.0%}"
        if tired is not None:
            line += f"，其中疲惫 {tired:.0%}"
        if sad is not None:
            line += f"，低落 {sad:.0%}"
        facts.append(line)
    temp = weather.get("temp") if weather else None
    if temp is not None:
        facts.append(f"- 近期气温：{temp}℃")
    return "\n".join(facts)


def build_messages(stats_result: dict, weather: dict | None,
                   rule_advice: Sequence[str] | None = None) -> List[Dict[str, str]]:
    """拼 chat/completions 的 messages（system + user）。

    ``rule_advice`` 是规则引擎已生成的建议（不含免责声明的最后一行）。
    规则引擎说过的角度直接告诉模型"不要重复" —— 提示词是劝导，
    :func:`ai_diet_advice` 里还有一道精确去重的执行层。
    """
    rules_text = "\n".join(f"- {line}" for line in (rule_advice or [])) or "（无）"
    system = _PROMPT_TEMPLATE % (REQUEST_LINES, LINE_MAX_CHARS,
                                 build_stats_facts(stats_result, weather), rules_text)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "请根据以上数据输出饮食建议。"},
    ]


def _clip_line(line: str) -> str:
    """单行超长时截到最后一个完整句尾；太短就不留。"""
    if len(line) <= LINE_MAX_CHARS:
        return line
    cut = line[:LINE_MAX_CHARS]
    ends = list(_SENTENCE_END_RE.finditer(cut))
    if ends:
        return cut[:ends[-1].end()]
    return cut[:LINE_MAX_CHARS]


def sanitize_lines(text: str) -> List[str]:
    """把模型输出洗成干净的建议行。一条不剩 = 返回空列表。

    **返回空列表永远比返回脏行安全** —— 调用方拿到空列表就省略整个
    区块，页面上还有规则引擎的建议兜着。
    """
    if not text or not text.strip():
        return []
    lines: List[str] = []
    for raw in text.strip().splitlines():
        line = _LIST_MARKER_RE.sub("", raw)
        line = _MARKDOWN_RE.sub("", line)
        line = _EMOJI_RE.sub("", line)
        line = _URL_RE.sub("", line)
        line = line.strip().strip("；;。")
        if not line:
            continue
        # 硬性拒绝：医疗/用药、占位符、前缀套话
        if _MEDICAL_RE.search(line) or "{" in line or "}" in line:
            continue
        lines.append(_clip_line(line))
        if len(lines) >= MAX_LINES:
            break
    return lines


# ==========================================================================
# 客户端：永不抛异常，失败返回 None
# ==========================================================================

Transport = Callable[[urllib.request.Request, float], bytes]


def _urlopen_bytes(req: urllib.request.Request, timeout: float) -> bytes:
    """默认传输：urlopen 读全量响应体。抽出来是为了测试能注入假传输。"""
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


class DietAdviceClient:
    """DeepSeek 报告建议客户端。``complete`` 契约：永不抛异常，失败 None。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 8.0,
        max_tokens: int = 300,
        temperature: float = 0.7,
        transport: Optional[Transport] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._transport = transport or _urlopen_bytes

    @property
    def available(self) -> bool:
        """只回答"配好凭证没有"。缺 key 时调用方直接走兜底，不发请求。"""
        return bool(self.api_key)

    def complete(self, messages: Sequence[Dict[str, str]]) -> Optional[str]:
        """请求大模型。任何失败（网络/超时/坏响应）返回 None，绝不抛。"""
        if not self.available:
            return None
        body = json.dumps(build_chat_request(
            messages, self.model, self.max_tokens, self.temperature)).encode("utf-8")
        # ⚠️ Bearer 后面是空格（与 core/llm.py 同一条注意事项，别抄豆包分号版）
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
            },
            method="POST")
        try:
            raw = self._transport(request, self.timeout)
            return parse_chat_response(raw)
        except Exception as exc:
            logger.info("饮食建议大模型请求失败：%s", exc)
            return None


# ==========================================================================
# 编排：配置读取 + 报告级缓存
# ==========================================================================

#: 失败结果的短缓存（秒）。见模块注释"失败也记短 TTL"。
_FAIL_CACHE_TTL = 30 * 60.0

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"key": None, "ts": 0.0, "lines": None}


def reset_cache() -> None:
    """清空报告级缓存（测试与换周切换用）。"""
    with _cache_lock:
        _cache["key"] = None
        _cache["ts"] = 0.0
        _cache["lines"] = None


def _cache_key(stats_result: dict | None, weather: dict | None) -> tuple:
    """统计指纹：这些数变了才值得重新问一次大模型。"""
    if not stats_result:
        return ()
    distribution = stats_result.get("state_distribution") or []
    return (
        stats_result.get("sample_count"),
        stats_result.get("hr_avg"),
        stats_result.get("rr_avg"),
        round(stats_result.get("negative_ratio") or 0.0, 2),
        tuple(sorted((d["state"], round(d["ratio"], 2)) for d in distribution)),
        (weather or {}).get("temp"),
    )


def _client_from_settings() -> DietAdviceClient:
    """从 Django settings 读配置建客户端。每次现读 —— 测试的
    ``override_settings`` 才能即时生效（不缓存客户端实例）。"""
    cfg = settings
    return DietAdviceClient(
        api_key=getattr(cfg, "FAMILY_LLM_API_KEY", ""),
        base_url=getattr(cfg, "FAMILY_LLM_BASE_URL", "https://api.deepseek.com"),
        model=getattr(cfg, "FAMILY_LLM_MODEL", "deepseek-chat"),
        timeout=float(getattr(cfg, "FAMILY_LLM_TIMEOUT", 8.0)),
        max_tokens=int(getattr(cfg, "FAMILY_LLM_MAX_TOKENS", 300)),
        temperature=float(getattr(cfg, "FAMILY_LLM_TEMPERATURE", 0.7)),
    )


def ai_diet_advice(stats_result: dict | None, weather: dict | None,
                   client: DietAdviceClient | None = None,
                   rule_advice: Sequence[str] | None = None) -> List[str]:
    """大模型补充建议（带报告级缓存）。任何失败返回空列表。

    ``rule_advice``：规则引擎已给出的建议（weekly_report 传入时会先
    滤掉免责声明行）。用途有二：提示词里明说"这些角度不要重复"；
    返回值再过一道**精确去重**（与规则建议逐字相同的行直接丢弃）——
    提示词是劝导，去重是执行。

    调用方（weekly_report）把结果放进 ``report["ai_advice"]``，模板在
    非空时渲染"AI 营养师补充建议"区块；空列表 = 区块整体不出现，
    规则引擎的建议照常展示 —— 大模型挂了对页面零影响。
    """
    # 硬边界 1：没有数据就没有依据，直接不问。
    if not stats_result or not stats_result.get("sample_count"):
        return []
    enabled = bool(getattr(settings, "FAMILY_LLM_ENABLED", True))
    key = _cache_key(stats_result, weather)
    now = time.time()
    with _cache_lock:
        cached_key, cached_ts, cached_lines = _cache["key"], _cache["ts"], _cache["lines"]
    if cached_key == key and cached_lines is not None and enabled:
        ttl = (_FAIL_CACHE_TTL if cached_lines == [] else
               float(getattr(settings, "FAMILY_LLM_CACHE_TTL", 6 * 3600.0)))
        if now - cached_ts < ttl:
            return list(cached_lines)
    if not enabled:
        return []

    client = client or _client_from_settings()
    if not client.available:
        # 没配凭证：记一次"空"结果，避免每次开报告页都白走一遍判断。
        with _cache_lock:
            _cache.update(key=key, ts=now, lines=[])
        return []

    reply = client.complete(build_messages(stats_result, weather, rule_advice))
    lines = sanitize_lines(reply) if reply else []
    # 执行层去重：与规则建议逐字相同的行不留（模型没听话时兜住）。
    if lines and rule_advice:
        seen = {line.strip() for line in rule_advice}
        lines = [line for line in lines if line not in seen]
    if lines:
        logger.info("饮食建议大模型补充生成 %d 条", len(lines))
    with _cache_lock:
        _cache.update(key=key, ts=now, lines=list(lines))
    return list(lines)
