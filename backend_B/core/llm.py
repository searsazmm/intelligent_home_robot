# -*- coding: utf-8 -*-
"""大模型对话客户端（DeepSeek），给 :mod:`core.dialogue` 当"接不住的话"的补充。

--------------------------------------------------------------------------
为什么需要它
--------------------------------------------------------------------------
``core/dialogue.py`` 是纯关键词规则：任何不含那约 60 个关键词的话，兜底都掉进
3 句通用套话里。用户听完的评价是「回答太过生硬、硬板，没有应对其他话语的能力」。
这个模块负责把那类话接住。

--------------------------------------------------------------------------
它和规则模板的关系：叠加，不是替换
--------------------------------------------------------------------------
    规则模板  ——  可预测、零依赖、永远有话说，**永远是兜底**
    大模型    ——  负责接住没预设过的话，挂了只是少一层

所以 ``complete()`` 的契约是**永不抛异常、失败就返回 None**，调用方（dialogue）
拿到 None 就落回模板。整条链路里没有任何一处会因为大模型出问题而让机器人变哑巴。

--------------------------------------------------------------------------
两条硬边界（在 dialogue 里执行，这里只提供零件）
--------------------------------------------------------------------------
1. **危机语句和身体不适不走大模型**，走确定性文案 —— 见 ``dialogue.llm_eligible``
   和 ``text_emotion.TextEmotion.crisis``。
2. **模型输出必须过** :func:`sanitize_reply` 才算数。提示词是劝导，清洗是执行；
   洗不干净就返回 None、落回模板 —— 模板永远是手写的、领域内的、安全的一句话。

--------------------------------------------------------------------------
为什么用标准库 urllib 而不是 requests / openai SDK
--------------------------------------------------------------------------
模块 B 的「运行期零第三方依赖」是被测试守着的（tests/test_no_third_party_imports.py
扫描顶层 import，tests/test_voice_engines.py 的子进程守卫检查 sys.modules）。
DeepSeek 是 OpenAI 兼容的纯 HTTP 接口，标准库足够，不该为它破例。

⚠️ 鉴权是 ``Authorization: Bearer <key>``（**空格分隔**）——
   和豆包 v1 那套 ``Bearer;<token>``（分号）不是一回事，别抄混。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional, Protocol, Sequence

import config

logger = logging.getLogger(__name__)

#: DeepSeek 的 OpenAI 兼容端点。请求拼成 ``{BASE_URL}/chat/completions``。
#: 官方文档里 ``https://api.deepseek.com`` 和 ``https://api.deepseek.com/v1``
#: 两种都收，所以这里不写死 /v1 —— 用户填哪个都行（拼之前会 rstrip('/')）。
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"

#: 选型时用来试探的短句。越短越好 —— 它花的是一次真实的往返。
DEFAULT_PROBE_MESSAGE = "你好"

#: 日志里一句话最多显示多少个字。和 ``core/voice/tts.py`` 的 ``LOG_TEXT_CHARS``
#: 同一个取舍（整句打出来淹日志，截太短又对不上是哪句话）。不跨模块复用它：
#: llm 是被 dialogue 用的，反过来 import tts 会拧成环。
LOG_TEXT_CHARS = 30


def _clip(text: str) -> str:
    """把一句话压成适合进日志的短形式。"""
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= LOG_TEXT_CHARS:
        return text
    return text[:LOG_TEXT_CHARS] + "…"


# ==========================================================================
# 纯函数：请求构造与响应解析
#
# 拆成模块级纯函数是为了能直接单测 —— 不用联网、不用假服务器。
# 和 ``tts.build_doubao_headers`` / ``build_doubao_request`` / ``parse_doubao_stream``
# 是同一套做法，理由也一样。
# ==========================================================================

def build_chat_headers(api_key: str) -> dict:
    """构造鉴权头。

    ⚠️ ``Bearer`` 后面是**空格**。本项目里另有一处 ``Authorization: Bearer;<token>``
    （豆包 v1，分号），那套已经废弃且两个厂商互不通用 —— 抄混了会得到一个
    语义完全不同的头，而且服务端只会回你一句"鉴权失败"。
    """
    return {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }


def build_chat_request(
    messages: Sequence[Dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
) -> dict:
    """构造 /chat/completions 的请求体。

    ``stream`` 固定 False：这一版走的是"等整句生成完再合成"的简单路径。
    流式（边说边合成）要改播放链路，留给后续按实测延迟决定。
    """
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": bool(stream),
    }


def parse_chat_response(raw: bytes) -> str:
    """从响应体里取出回复文本。取不到就抛 ``ValueError``。

    抛异常而不是返回空串：调用方据此区分"模型说了话"和"这次没成"，
    后者要落回模板。返回空串会让一句空回复覆盖掉本来能说的模板句。
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("响应不是合法 JSON：%s"
                         % _clip(raw.decode("utf-8", "replace"))) from exc

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        # 服务端的错误信息通常也走 200 + 一个 error 字段，所以这行很重要。
        raise ValueError("响应里没有 choices：%s" % _clip(json.dumps(
            payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)))

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

#: 四个状态给模型看的"人话"。不要把裸枚举（normal/sad/...）丢给模型 ——
#: 它没法从中推断该用什么语气，只会把它当成一个无意义的标签抄进回复里。
STATE_HINTS: Dict[str, str] = {
    config.STATE_NORMAL: "状态平稳",
    config.STATE_SAD: "情绪低落",
    config.STATE_TIRED: "看着很疲惫",
    config.STATE_ABSENT: "刚回到画面里",
}

#: 意图给模型看的"人话"。
INTENT_HINTS: Dict[str, str] = {
    "chat": "想跟您闲聊",
    "venting": "在跟您倾诉心事",
    "question": "在问您一件事",
}

#: 系统提示词。**整段不能出现花括号** —— 模型会模仿提示词的写法，
#: 吐出一个 ``{...}`` 就会被 sanitize_reply 判死，于是大模型这条路被静默废掉。
#: 所以这里用 % 格式化，不用 f-string（f-string 的源码里全是花括号，容易看漏）。
#: 有测试钉这一条。
_PROMPT_TEMPLATE = """你是「小陪」，一个陪伴独居老人的居家机器人。你是机器，不是人。

说话方式：
- 只用中文口语，1~2 句，整段不超过 40 字 —— 这句话是要念出来给老人听的。
- 先共情，再回应。不说教、不命令、不追问。
- 不给医疗建议：不诊断、不推荐用药、不说剂量、不提偏方。老人提到身体不舒服时，
  只表达关心，并建议告诉家里人。
- 不劝老人做任何有风险的事（登高、独自出门、搬重物、自己加减药）。
- 不要问"您还记得吗""您听懂了吗"这类话 —— 那会让老人觉得自己被考。
- 不输出 Markdown、表情符号、括号里的动作描写、网址、列表、换行。
- 只输出要念出来的那句话本身，不要任何解释、前后缀、引号或标注。

此刻的情况：
- 老人的状态：%s
- 他这句话的情绪：%s
- 他这句话想干什么：%s
"""


def build_system_prompt(state: str, intent: str, emotion_label: str) -> str:
    """拼系统提示词。状态/意图/情绪都翻成人话再给模型。"""
    return _PROMPT_TEMPLATE % (
        STATE_HINTS.get(state, "状态平稳"),
        _EMOTION_HINTS.get(emotion_label, "看不太出来"),
        INTENT_HINTS.get(intent, "想跟您说说话"),
    )


_EMOTION_HINTS: Dict[str, str] = {
    "negative": "明显消极",
    "neutral": "比较平静",
    "positive": "挺高兴",
}

# ---- 清洗用的正则 ----

#: emoji / 象形符号 / 变体选择符 / 零宽连接符。范围全部避开 CJK（U+4E00 起），
#: 所以不会误伤汉字。用转义写而不是字面量 —— 这些字符大多不可见，
#: 直接写在源码里没人看得出边界在哪，改的时候也没法确认自己删掉了什么。
_EMOJI_RE = re.compile(
    "["
    "🀀-🫿"   # 表情、象形、交通、补充符号（含国旗的区域指示符）
    "☀-➿"   # 杂项符号、装饰符号（☀ ✅ ✨ …）
    "⬀-⯿"   # 杂项符号与箭头
    "︀-️"   # 变体选择符（❤️ 后面那个不可见的 ️）
    "‍"                   # 零宽连接符（👨👩👧 用它拼起来）
    "⃣"                   # 组合包围键帽（1️⃣ 的那个）
    "©®™ℹ〰〽㊗㊙"   # © ® ™ ℹ 〰 〽 ㊗ ㊙
    "]+")

#: 括号里的**动作描写**。刻意用显式词表而不是"去掉所有括号" ——
#: 「（血压高）」是内容不是描写，去掉就丢信息了。
#: 词必须是括号内**最后**的内容，所以「（笑一笑就好了）」不会被误删。
_STAGE_DIRECTION_RE = re.compile(
    r"[（(\[【]\s*[^（）()\[\]【】]{0,6}?"
    r"(?:笑|笑了|叹气|叹息|停顿|停了一下|沉默|点头|摇头|轻声|小声|犹豫|想了想|温柔地)"
    r"\s*[）)\]】]")

#: 网址。念出来是噪音。
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

#: Markdown 的装饰性字符（此时已按行处理）。
_MARKDOWN_RE = re.compile(r"```|~~~|`|\*\*|__|\*|~~|^#{1,6}\s*|^>\s*")

#: 行首列表符号。
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*·•]|\d+[.、)])\s+")

#: 模型爱加的前缀。用显式词表而不是"第一个冒号之前都砍掉" ——
#: 后者会把「您说得对：……」这种正常句子砍坏。
_PREFIX_RE = re.compile(r"^(?:回复|回答|答|机器人|小陪|助手|AI|我)\s*[：:]\s*", re.IGNORECASE)

#: 句末标点。超长时用来截到一个完整的句子。
_SENTENCE_END_RE = re.compile(r"[。！？!?]")

#: 医疗/用药建议。命中直接判死 —— 这是念给老人听的，宁可不说也不能说错。
#: 会误伤「您别自己加药」这种**正确**的劝告（它也会被判死、落回模板）。
#: 这个方向是对的：误判的代价是少说一句好话，漏判的代价是给错医嘱。
_MEDICAL_RE = re.compile(
    r"(?:吃|服用|停|加|减|换|喝).{0,4}(?:药|片|粒|剂量)"
    r"|剂量|毫克|mg|处方|偏方|消炎药|止痛药|降压药",
    re.IGNORECASE)

#: 冒充人类。⚠️ ``(?!工)`` 那个前瞻不能省 ——
#: 朴素的「我是人」会把**正确答案**「我是人工智能」判死。
_HUMAN_CLAIM_RE = re.compile(
    r"我(?:是|就是|其实是)(?:个|一个)?(?:真|活)?人(?!工)"
    r"|我是人类"
    r"|我不是机器(?:人)?")

#: 超长时，截到这个位置之前就算"太短了不值得留"。
MIN_TRUNCATE_INDEX = 8


def sanitize_reply(text: str) -> Optional[str]:
    """把模型输出洗成"能直接念出来"的一句话。洗不干净返回 ``None``。

    **返回 None 永远比返回一句脏话安全** —— 调用方一定会落回模板，
    而模板是手写的、领域内的、审过的。

    顺序有意义：先剥结构（Markdown/换行/前缀）再量长度，否则一个带
    ``**`` 和列表符号的回复会因为"太长"被丢，而它本来洗干净是合格的。
    """
    if not text or not text.strip():
        return None

    # 1) 逐行剥掉 Markdown 结构和列表符号，再合成一行。
    #    列表符号必须先按行处理 —— 合成一行之后就找不到行首了。
    lines: List[str] = []
    for line in text.strip().splitlines():
        line = _LIST_MARKER_RE.sub("", line)
        line = _MARKDOWN_RE.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)
    if not lines:
        return None
    result = "".join(lines)

    # 2) 去掉"回复："这类前缀
    result = _PREFIX_RE.sub("", result, count=1)

    # 3) 剥掉不该念出来的东西
    result = _EMOJI_RE.sub("", result)
    result = _STAGE_DIRECTION_RE.sub("", result)
    result = _URL_RE.sub("", result)
    result = result.strip()

    # 4) 硬性拒绝
    #    占位符：{topic} 漏出来等于把模板的内部语法念给老人听，
    #    而且会破坏"回复里绝不含 {"这条既有不变量（test_core.py 有断言）。
    if "{" in result or "}" in result:
        logger.info("大模型回复含占位符，弃用：%s", _clip(result))
        return None
    if _MEDICAL_RE.search(result):
        logger.warning("大模型回复涉及医疗建议，弃用：%s", _clip(result))
        return None
    if _HUMAN_CLAIM_RE.search(result):
        logger.warning("大模型回复自称人类，弃用：%s", _clip(result))
        return None

    # 5) 长度。超了就截到第一个完整句子；截不出来就弃用 ——
    #    念半句话比念一句模板更糟。
    limit = max(4, int(getattr(config, "LLM_MAX_CHARS", 50)))
    if len(result) > limit:
        match = _SENTENCE_END_RE.search(result, MIN_TRUNCATE_INDEX)
        if match is None or match.end() > limit:
            logger.info("大模型回复过长且截不出整句（%d 字），弃用：%s",
                        len(result), _clip(result))
            return None
        result = result[:match.end()]

    # 6) 收尾
    result = result.strip().rstrip("，,、；;：:")
    if len(result) < 2:
        return None
    return result


# ==========================================================================
# 交通层
# ==========================================================================

def _urlopen_bytes(request, timeout: float) -> bytes:
    """默认的传输实现。urllib 在函数体内 import —— 见模块头的说明。"""
    import urllib.request

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# ==========================================================================
# 客户端
# ==========================================================================

class Chatter(Protocol):
    """对话模型的统一接口。:class:`DialogueEngine` 只依赖这几个成员。"""

    name: str
    requires_network: bool

    @property
    def available(self) -> bool: ...

    @property
    def reason(self) -> str: ...

    def complete(self, messages: Sequence[Dict[str, str]]) -> Optional[str]: ...

    def close(self) -> None: ...


class NullChatClient:
    """没有大模型时用的空实现。``available`` 恒为 False，``complete`` 恒为 None。"""

    name = "null"
    requires_network = False
    available = False

    def __init__(self, reason: str = "未启用大模型对话", **_kwargs) -> None:
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason

    def complete(self, messages: Sequence[Dict[str, str]]) -> Optional[str]:
        return None

    def close(self) -> None:
        return None


class DeepSeekClient:
    """DeepSeek 的 OpenAI 兼容客户端。

    形态上刻意和 :class:`core.voice.tts.DoubaoSynthesizer` 保持一致：
    缺凭证时在 ``__init__`` 里**点名缺的是哪个环境变量**（用户明确要求过
    「无凭证的时候，控制台明确打印缺失的变量名称」），``available`` 只回答
    "配好了没有"，``complete`` **任何失败都返回 None、永不抛异常**。
    """

    name = "deepseek"

    #: 见 :func:`probe_chatter`。要联网，所以选型时要真调一次才算通过。
    requires_network = True

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        transport=None,
    ) -> None:
        self.api_key = api_key or config.LLM_DEEPSEEK_API_KEY
        self.base_url = (base_url or config.LLM_DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or config.LLM_DEEPSEEK_MODEL
        self.timeout = float(
            config.LLM_TIMEOUT if timeout is None else timeout)
        self.max_tokens = int(
            config.LLM_MAX_TOKENS if max_tokens is None else max_tokens)
        self.temperature = float(
            config.LLM_TEMPERATURE if temperature is None else temperature)

        # 传输实现可注入。这是相对 DoubaoSynthesizer 的**一处刻意偏离**：
        # 超时 / HTTP 错误 / 坏 JSON / 空回复全都是 complete 的失败分支，
        # 没有这个缝就只能去 monkeypatch 函数体内的东西，很脆。
        # 纯函数拆分已经覆盖了构造与解析的测试，这个缝只承担错误路径。
        self._transport = transport or _urlopen_bytes

        self._usable = False
        self._reason = ""

        missing = [
            name for name, value in (
                ("B_DEEPSEEK_TOKEN", self.api_key),
                ("B_DEEPSEEK_BASE_URL", self.base_url),
                ("B_DEEPSEEK_MODEL", self.model),
            ) if not value
        ]
        if missing:
            self._reason = "未配置 %s" % "、".join(missing)
            if "B_DEEPSEEK_TOKEN" in missing:
                self._reason += "（API Key 在 platform.deepseek.com → API keys 页面）"
        else:
            self._usable = True
            logger.info("大模型对话已就绪（deepseek / %s）", self.model)

        if not self._usable:
            logger.warning("大模型对话不可用：%s", self._reason)

    @property
    def available(self) -> bool:
        return self._usable

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def endpoint(self) -> str:
        return self.base_url + CHAT_COMPLETIONS_PATH

    def complete(self, messages: Sequence[Dict[str, str]]) -> Optional[str]:
        """调一次模型，返回回复文本；任何失败都返回 ``None``。

        ⚠️ 超时用的是 ``config.LLM_TIMEOUT``（默认 4 秒），**不是**
        ``SYNTH_TIMEOUT`` 那样的 30 秒 —— 这是用户正等着回话的交互路径，
        宁可早失败落回模板，也不能让老人干等。
        """
        if not self._usable or not messages:
            return None

        import urllib.error
        import urllib.request

        started = time.monotonic()
        # ⚠️ 请求体的构造也必须在 try 里面。早先它在外面，于是
        #    `json.dumps(..., ensure_ascii=False).encode("utf-8")` 抛出的
        #    UnicodeEncodeError 会**从 complete 里逃出去**，而契约写的是
        #    "永不抛异常"。调用方虽然还有一层兜底，但契约不能是假的。
        try:
            body = build_chat_request(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=build_chat_headers(self.api_key),
                method="POST",
            )
            raw = self._transport(request, self.timeout)
            text = parse_chat_response(raw)
        except urllib.error.HTTPError as exc:
            # Key 无效、余额不足、模型名下线都走这里。
            # 把服务端的话原样带出来 —— 只剩一个状态码的话事后没法查。
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            logger.warning("大模型 HTTP 错误：%s %s", exc.code, detail)
            return None
        except UnicodeEncodeError:
            # 前面那句话里有 UTF-8 编不出去的字符 —— 孤立代理项
            # （lone surrogate），来源是上游某次失败的解码：Windows 控制台
            # 按错误的码页读中文、或对端发来的字节不是合法 UTF-8。
            #
            # 为什么是"落回模板"而不是"清洗掉再发"：含孤立代理项的句子
            # 本来就是乱码，发给模型只会得到一段关于乱码的回复 ——
            # 模板至少是手写的、领域内的一句话。
            #
            # ⚠️ 这个分支必须排在 ValueError **前面**：UnicodeEncodeError
            #    是 ValueError 的子类，顺序反了就会被当成"返回内容不合法"，
            #    日志指向错误的方向。
            logger.warning("大模型请求体含无法编码的字符（上游解码残留），"
                           "这一句回退规则模板")
            return None
        except ValueError as exc:
            # parse_chat_response 抛的（含服务端 200 + error 字段那种）。
            logger.warning("大模型返回内容不合法：%s", exc)
            return None
        except Exception:
            # 断网、超时、DNS 失败都是常态，不该让对话中断。
            logger.warning("大模型调用失败（断网或超时？），这一句回退规则模板")
            logger.debug("大模型调用异常详情", exc_info=True)
            return None

        logger.info("大模型回复用时 %.0fms：%s",
                    (time.monotonic() - started) * 1000, _clip(text))
        return text

    def close(self) -> None:
        return None


#: 可选的大模型。``null`` / ``none`` 都是"明确不要大模型"。
CHATTERS: Dict[str, type] = {
    "deepseek": DeepSeekClient,
    "null": NullChatClient,
    "none": NullChatClient,
}

#: ``auto`` 的候选顺序。目前只有一家，留着这个形状是为了以后加第二家时
#: 不用改调用方 —— 和 ``build_synthesizer`` 的候选链同一个思路。
AUTO_ORDER = ("deepseek",)


def probe_chatter(client, message: str = DEFAULT_PROBE_MESSAGE) -> bool:
    """**真的调一次模型**，看它到底行不行。

    为什么需要这一步：``available`` 只回答"**凭证填了没有**"，
    答不了"**Key 有效吗 / 模型名还在吗 / 端点通吗**"。这三件事的失败表现
    是一样的：每次对话都静默落回模板，而启动日志里一切正常。

    代价是一次真实往返（约 1 秒）。换来的是"启动日志里就能看出 Key 有没有效"。

    ⚠️ 用的是 ``complete`` 自己的超时（``LLM_TIMEOUT``，默认 4 秒），
    不是 ``SYNTH_TIMEOUT`` 的 30 秒 —— 这里搞错的话，一个填错的 Key
    会让启动卡满半分钟。豆包那次就是在这个地方踩过。
    """
    if not getattr(client, "available", False):
        return False
    try:
        return client.complete([{"role": "user", "content": message}]) is not None
    except Exception:
        # complete 的契约是"永不抛异常"，但探测不该因为某个实现违约
        # 而把整个选型流程带崩。
        logger.debug("探测大模型 %r 时抛了异常",
                     getattr(client, "name", "?"), exc_info=True)
        return False


def build_chatter(name: str = "auto", probe: Optional[bool] = None,
                  **kwargs) -> Chatter:
    """按名字构造大模型客户端。

    ``auto`` 的退让顺序是 ``deepseek`` 一家；没有凭证就返回
    :class:`NullChatClient`（对话完全走规则模板）——「没配凭证」不该表现为
    「机器人不说话了」，只是接不住没预设过的话而已。

    ``probe=None`` 时跟随 ``config.LLM_PROBE``。测试和离线开发传 ``False``。

    **任何情况下都不抛异常**，最差返回 :class:`NullChatClient`。
    """
    requested = (name or "auto").strip().lower()

    if requested in ("null", "none"):
        logger.info("大模型对话已按配置停用（--llm %s）", requested)
        return NullChatClient("已按配置停用（--llm %s）" % requested)

    if not config.LLM_ENABLED:
        logger.info("大模型对话已按配置停用（B_LLM=false）")
        return NullChatClient("已按配置停用（B_LLM=false）")

    if probe is None:
        probe = bool(config.LLM_PROBE)

    if requested == "auto":
        candidates = AUTO_ORDER
    elif requested in CHATTERS:
        # 名字认得出但没就绪（缺凭证）时退到候选链，而不是就此静默 ——
        # 和 `--tts doubao` 忘了填凭证那条路一个道理：命令行上看起来
        # 一切正常，实际上功能没了，最难查。
        candidates = (requested,) + tuple(
            item for item in AUTO_ORDER if item != requested)
    else:
        # 名字不认识是另一回事：拼写错误或过时配置。悄悄换成别的会让人
        # 以为 --llm 生效了，所以只记日志、不兜底。
        candidates = (requested,)

    # 每个候选失败的具体原因。最后要**带进 NullChatClient**，因为
    # main.py 的启动横幅直接打它的 reason —— 「未配置 B_DEEPSEEK_TOKEN」
    # 和「没有可用的大模型」对操作者是两条完全不同的信息：前者他立刻知道
    # 去设哪个环境变量，后者他还得往上翻日志。
    failures: List[str] = []

    for candidate in candidates:
        factory = CHATTERS.get(candidate)
        if factory is None:
            logger.warning("未知的大模型 %r（可选：%s）",
                           candidate, ", ".join(CHATTERS))
            failures.append("%s 不是已知的引擎" % candidate)
            continue

        try:
            client = factory(**kwargs)
        except Exception:
            logger.exception("初始化大模型客户端 %r 失败", candidate)
            failures.append("%s 初始化失败" % candidate)
            continue

        # 客户端自己判断能不能用（缺凭证 / 总开关关掉）
        if not getattr(client, "available", True):
            logger.warning("大模型 %r 不可用，尝试下一个", candidate)
            failures.append(getattr(client, "reason", "") or "%s 不可用" % candidate)
            continue

        # 配好了 ≠ 能用。要联网的客户端必须真调一次才算通过 ——
        # 见 probe_chatter 的说明。
        if probe and getattr(client, "requires_network", False):
            logger.info("正在试调一次 %r，确认凭证真的有效…", candidate)
            if not probe_chatter(client):
                logger.warning(
                    "大模型 %r 试调失败，跳过（Key 无效 / 断网 / 端点不通）；"
                    "对话将走规则模板", candidate)
                failures.append("%s 试调失败（Key 无效 / 断网 / 端点不通）" % candidate)
                try:
                    client.close()
                except Exception:
                    pass
                continue

        if candidate != candidates[0]:
            logger.warning("大模型已回退到 %r（原因见上一条）", candidate)
        return client

    logger.warning(
        "没有可用的大模型，对话将**完全走规则模板**。\n"
        "  原因：%s\n"
        "  这一条不影响程序运行 —— 只是接不住没预设过的话（答复会显得硬板）。\n"
        "  想启用：设 B_DEEPSEEK_TOKEN（API Key 在 platform.deepseek.com → API keys）。\n"
        "  详见 README.md §4.9「大模型对话怎么配」。",
        "；".join(failures) or "未知",
    )
    return NullChatClient("；".join(failures) or "没有可用的大模型")
