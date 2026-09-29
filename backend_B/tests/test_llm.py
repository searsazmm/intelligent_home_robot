# -*- coding: utf-8 -*-
"""大模型对话（core/llm.py）单元测试。

跑法（在 backend_B 目录下）：
    python -m pytest tests/test_llm.py -v
    python tests/test_llm.py            # 不装 pytest 也能跑，见文件末尾

**全程不碰网络。** 纯函数直接测；客户端注入一个假 transport；
对话集成的用例注入一个假客户端。这条纪律和 test_voice_engines.py 一致：
测试不该因为本机能不能连上某个域名而红或绿。

覆盖重点（都是"出错时很难查"的地方）：
    - 鉴权头抄错厂商（Bearer 空格 vs 分号、X-Api-Key）
    - 响应的各种坏形态（非 JSON / 空 choices / 空内容）
    - 模型输出的清洗：Markdown、emoji、动作描写、医疗建议、冒充人类
    - "洗不干净就返回 None"这条契约本身
    - 危机语句与身体不适**绝不进大模型**（断言客户端没被调用）
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import llm
from core.dialogue import DialogueEngine
from core.history_store import CsvHistoryStore
from core.llm import (DeepSeekClient, NullChatClient, build_chat_headers,
                      build_chat_request, build_chatter, build_system_prompt,
                      parse_chat_response, sanitize_reply)


def header_of(request, name: str):
    """按名字取请求头，**不区分大小写**。

    ``urllib`` 的 ``add_header`` 会把键存成 ``key.capitalize()``，
    于是 "Content-Type" 实际存成 "Content-type" —— 直接下标取会 KeyError，
    而报错信息只会说"没这个键"，很容易误判成"头没设上"。
    """
    for key, value in request.headers.items():
        if key.lower() == name.lower():
            return value
    return None


def chat_payload(content: str) -> bytes:
    """造一个正常响应体。"""
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]},
        ensure_ascii=False,
    ).encode("utf-8")


class FakeTransport:
    """假的 HTTP 传输：记下每次调用，返回预设结果或抛预设异常。"""

    def __init__(self, result=b"{}"):
        self.result = result
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# ==========================================================================
# 请求构造
# ==========================================================================

class TestBuildChatHeaders(unittest.TestCase):

    def test_uses_bearer_with_space(self):
        """DeepSeek（OpenAI 兼容）是 ``Bearer <key>``，**空格分隔**。

        本项目里另有一处 ``Authorization: Bearer;<token>``（豆包 v1，分号）。
        两个厂商的鉴权模型互不通用，抄混了服务端只会回一句"鉴权失败"，
        看不出是格式问题。这条测试就是钉这个。
        """
        headers = build_chat_headers("sk-abc123")
        self.assertEqual(headers["Authorization"], "Bearer sk-abc123")
        self.assertNotIn(";", headers["Authorization"])

    def test_does_not_leak_doubao_headers(self):
        """不能把豆包那套头带过来。

        X-Api-Key / X-Api-Resource-Id 是火山引擎的模型版本鉴权，
        DeepSeek 完全不认 —— 带上它们不会报错，只会让人以为配好了。
        """
        headers = build_chat_headers("sk-abc123")
        self.assertNotIn("X-Api-Key", headers)
        self.assertNotIn("X-Api-Resource-Id", headers)

    def test_content_type_is_json(self):
        self.assertEqual(build_chat_headers("k")["Content-Type"],
                         "application/json")


class TestBuildChatRequest(unittest.TestCase):

    def test_passes_fields_through(self):
        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": "u"}]
        body = build_chat_request(messages, "deepseek-chat", 128, 1.0)
        self.assertEqual(body["model"], "deepseek-chat")
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["max_tokens"], 128)
        self.assertEqual(body["temperature"], 1.0)

    def test_stream_is_off(self):
        """这一版刻意走非流式。

        流式要改播放链路（Speaker 现在是加锁阻塞的 say），
        收益是压缩首字延迟 —— 等实测延迟数据出来再决定。
        这条断言是提醒：改它的时候要连着改播放那边。
        """
        self.assertIs(build_chat_request([], "m", 1, 0.0)["stream"], False)

    def test_copies_messages(self):
        """不要持有调用方的列表 —— 之后有人往里 append 会污染历史。"""
        messages = [{"role": "user", "content": "u"}]
        body = build_chat_request(messages, "m", 1, 0.0)
        body["messages"].append({"role": "user", "content": "偷偷加的"})
        self.assertEqual(len(messages), 1)


class TestParseChatResponse(unittest.TestCase):

    def test_happy_path(self):
        self.assertEqual(parse_chat_response(chat_payload("您好呀。")), "您好呀。")

    def test_strips_whitespace(self):
        self.assertEqual(parse_chat_response(chat_payload("  您好呀。  ")),
                         "您好呀。")

    def test_rejects_bad_shapes(self):
        """取不到文本时**必须抛**，不能返回空串。

        返回空串会让一句空回复覆盖掉本来能说的模板句 ——
        表现出来就是"机器人突然不说话了"，而日志里一切正常。
        """
        bad_cases = {
            "不是 JSON": b"not json at all",
            "空 body": b"",
            "没有 choices": json.dumps({"error": "boom"}).encode(),
            "choices 为空": json.dumps({"choices": []}).encode(),
            "choices[0] 不是对象": json.dumps({"choices": ["x"]}).encode(),
            "没有 message": json.dumps({"choices": [{}]}).encode(),
            "message 不是对象": json.dumps(
                {"choices": [{"message": "x"}]}).encode(),
            "content 为空串": json.dumps(
                {"choices": [{"message": {"content": ""}}]}).encode(),
            "content 全是空白": json.dumps(
                {"choices": [{"message": {"content": "   "}}]}).encode(),
            "content 是数字": json.dumps(
                {"choices": [{"message": {"content": 42}}]}).encode(),
        }
        for label, raw in bad_cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    parse_chat_response(raw)


# ==========================================================================
# 输出清洗
# ==========================================================================

class TestSanitizeReply(unittest.TestCase):

    def test_strips_markdown(self):
        cases = {
            "**粗体**您好。": "粗体您好。",
            "__粗体__您好。": "粗体您好。",
            "*斜体*您好。": "斜体您好。",
            "`代码`您好。": "代码您好。",
            "```\n您好\n```": "您好",
            "~~~\n您好\n~~~": "您好",
            "## 您好": "您好",
            "> 您好": "您好",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_reply(raw), expected)

    def test_joins_multiple_lines(self):
        """多行合成一行 —— 每行单独念会变成一串短句，很怪。"""
        self.assertEqual(sanitize_reply("您好呀。\n我在这儿。"), "您好呀。我在这儿。")

    def test_strips_list_markers(self):
        self.assertEqual(sanitize_reply("- 您好呀。\n- 我在这儿。"),
                         "您好呀。我在这儿。")
        self.assertEqual(sanitize_reply("1. 您好呀。"), "您好呀。")
        self.assertEqual(sanitize_reply("· 您好呀。"), "您好呀。")

    def test_strips_prefix(self):
        for raw in ("回复：您好呀。", "机器人：您好呀。", "小陪：您好呀。",
                    "助手: 您好呀。"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_reply(raw), "您好呀。")

    def test_strips_emoji(self):
        cases = {
            "您好呀 😊": "您好呀",
            "您好呀 ✅": "您好呀",
            "❤️ 您好": "您好",
            "1️⃣ 您好": "1 您好",   # 数字本身不在 emoji 区段，剥不掉；不影响
            "👨‍👩‍👧 您好": "您好",       # ZWJ 拼起来的组合也要剥干净
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_reply(raw), expected)

    def test_keeps_chinese(self):
        """emoji 的区段必须避开 CJK —— 剥错就是把整句话吃掉了。"""
        text = "您今天气色不错，我看着高兴。"
        self.assertEqual(sanitize_reply(text), text)

    def test_strips_stage_directions(self):
        cases = {
            "（笑）您好呀。": "您好呀。",
            "(叹气) 您好呀。": "您好呀。",
            "[沉默] 您好呀。": "您好呀。",
            "【轻声】您好呀。": "您好呀。",
            "（慈祥地笑）您好呀。": "您好呀。",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_reply(raw), expected)

    def test_keeps_bracketed_content(self):
        """括号里是**内容**不是动作描写的时候必须留着。

        这条是"不要写成去掉所有括号"的回归测试 ——
        「（血压高）」去掉就等于把老人说的关键信息删了。
        """
        cases = [
            "（血压高）要当心。",
            "（笑一笑就好了）。",
            "（我不太舒服）您别担心。",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_reply(raw), raw)

    def test_strips_url(self):
        self.assertEqual(sanitize_reply("https://a.com 您看这个。"), "您看这个。")
        self.assertEqual(sanitize_reply("www.a.com 您看这个。"), "您看这个。")

    def test_rejects_placeholder(self):
        """占位符漏出来等于把模板的内部语法念给老人听。

        而且会破坏"回复里绝不含 {"这条既有不变量（test_core.py 有断言）。
        """
        for raw in ("您说的{topic}我记下了。", "结果}", "{开头"):
            with self.subTest(raw=raw):
                self.assertIsNone(sanitize_reply(raw))

    def test_rejects_medical_advice(self):
        """给老人念医嘱是这个项目里最不能出的错。"""
        for raw in ("该加药了，一天两片。", "您把降压药停了吧。",
                    "吃点消炎药就好了。", "剂量减半试试。",
                    "每天 500 毫克。", "有个偏方特别管用。",
                    "记得吃两片药。"):
            with self.subTest(raw=raw):
                self.assertIsNone(sanitize_reply(raw))

    def test_rejects_human_claim(self):
        """不能自称人类 —— 独居老人真的会信。"""
        for raw in ("我是真人，不是程序。", "我不是机器人。",
                    "我其实是个活人。", "我是人类，我懂您。"):
            with self.subTest(raw=raw):
                self.assertIsNone(sanitize_reply(raw))

    def test_keeps_honest_ai_self_description(self):
        """⚠️ 正确的自我描述不能被误杀。

        朴素的 ``"我是人" in text`` 会把「我是人工智能」判死 ——
        那条前瞻 ``(?!工)`` 就是为它留的。
        """
        for raw in ("我是人工智能，陪着您。", "我是您的陪伴助手。",
                    "我是机器人小陪。"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_reply(raw), raw)

    def test_rejects_empty(self):
        for raw in ("", "   ", "\n\n", "```\n```", "😊"):
            with self.subTest(raw=repr(raw)):
                self.assertIsNone(sanitize_reply(raw))

    def test_truncates_at_sentence_end(self):
        """超长时截到第一个完整句子，而不是砍在半句话上。"""
        text = "您今天气色真不错呀。" + "后面还有很多很多很多话" * 8 + "。"
        result = sanitize_reply(text)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), config.LLM_MAX_CHARS)
        self.assertTrue(result.endswith("。"))

    def test_rejects_overlong_without_sentence_end(self):
        """截不出整句就弃用 —— 念半句话比念一句模板更糟。"""
        # 开头凑满 8 个字以上，然后完全没有句末标点
        self.assertIsNone(sanitize_reply("您今天气色不错" + "啊" * 90))

    def test_keeps_short_reply_under_limit(self):
        """没超限的短句原样保留（截断只在超长时发生）。"""
        text = "您今天气色不错呀。"
        self.assertEqual(sanitize_reply(text), text)

    def test_strips_trailing_comma(self):
        self.assertEqual(sanitize_reply("您先歇着，"), "您先歇着")

    def test_rejects_too_short(self):
        for raw in ("好", "。", "，", "a"):
            with self.subTest(raw=raw):
                self.assertIsNone(sanitize_reply(raw))

    def test_is_idempotent(self):
        """洗过的句子再洗一遍不该变 —— 否则"重试一次"会越洗越短。"""
        for raw in ("**您好** 😊（笑）", "回复：您好呀。", "（血压高）要当心。"):
            with self.subTest(raw=raw):
                once = sanitize_reply(raw)
                if once is not None:
                    self.assertEqual(sanitize_reply(once), once)


# ==========================================================================
# 提示词
# ==========================================================================

class TestBuildSystemPrompt(unittest.TestCase):

    def test_contains_state_intent_emotion(self):
        prompt = build_system_prompt(config.STATE_TIRED, "chat", "negative")
        self.assertIn(llm.STATE_HINTS[config.STATE_TIRED], prompt)
        self.assertIn(llm.INTENT_HINTS["chat"], prompt)
        self.assertIn(llm._EMOTION_HINTS["negative"], prompt)

    def test_no_braces(self):
        """⚠️ 提示词里绝不能出现花括号。

        模型会模仿提示词的写法，吐出一个 ``{...}`` 就会被 sanitize_reply 判死；
        于是**大模型这条路被静默废掉** —— 每句话都是模板，而日志里看起来
        一切正常。这是最隐蔽的一类失败，所以单独钉一条。
        """
        for state in config.VALID_STATES:
            for intent in ("chat", "venting", "question"):
                for label in ("negative", "neutral", "positive"):
                    with self.subTest(state=state, intent=intent, label=label):
                        prompt = build_system_prompt(state, intent, label)
                        self.assertNotIn("{", prompt)
                        self.assertNotIn("}", prompt)

    def test_unknown_values_do_not_crash(self):
        prompt = build_system_prompt("不存在的状态", "不存在的意图", "不存在的情绪")
        self.assertTrue(prompt.strip())

    def test_forbids_medical_advice(self):
        """提示词层面也要写明 —— 清洗能挡住大部分，但那不该是唯一一道。"""
        prompt = build_system_prompt(config.STATE_NORMAL, "chat", "neutral")
        self.assertIn("医疗建议", prompt)
        self.assertIn("剂量", prompt)


# ==========================================================================
# 客户端（假 transport，不联网）
# ==========================================================================

class TestDeepSeekClient(unittest.TestCase):

    def make(self, transport=None, **kwargs):
        kwargs.setdefault("api_key", "sk-test")
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        kwargs.setdefault("model", "deepseek-chat")
        return DeepSeekClient(transport=transport or FakeTransport(
            chat_payload("您好呀。")), **kwargs)

    def test_endpoint(self):
        self.assertEqual(self.make().endpoint,
                         "https://api.deepseek.com/chat/completions")

    def test_trailing_slash_is_normalized(self):
        """用户填 https://api.deepseek.com/ 时不能拼出双斜杠。"""
        client = self.make(base_url="https://api.deepseek.com/")
        self.assertEqual(client.endpoint,
                         "https://api.deepseek.com/chat/completions")

    def test_accepts_v1_base_url(self):
        """带不带 /v1 都认 —— DeepSeek 两种都收。"""
        client = self.make(base_url="https://api.deepseek.com/v1")
        self.assertEqual(client.endpoint,
                         "https://api.deepseek.com/v1/chat/completions")

    def test_successful_call(self):
        transport = FakeTransport(chat_payload("您好呀。"))
        client = self.make(transport=transport)
        self.assertEqual(client.complete([{"role": "user", "content": "hi"}]),
                         "您好呀。")
        self.assertEqual(len(transport.calls), 1)

    def test_request_shape(self):
        transport = FakeTransport(chat_payload("好"))
        client = self.make(transport=transport, timeout=4.0)
        client.complete([{"role": "user", "content": "hi"}])
        request, timeout = transport.calls[0]
        self.assertEqual(request.full_url,
                         "https://api.deepseek.com/chat/completions")
        self.assertEqual(header_of(request, "Authorization"), "Bearer sk-test")
        self.assertEqual(header_of(request, "Content-Type"), "application/json")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(timeout, 4.0)
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "deepseek-chat")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_http_error_returns_none(self):
        """Key 无效 / 余额不足走这条路。返回 None，不抛。"""
        error = urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions", 401, "Unauthorized",
            {}, io.BytesIO(b'{"error":{"message":"Authentication Fails"}}'))
        client = self.make(transport=FakeTransport(error))
        with self.assertLogs("core.llm", level="WARNING") as logs:
            self.assertIsNone(
                client.complete([{"role": "user", "content": "hi"}]))
        # 服务端原话要带进日志 —— 只剩一个 401 的话事后没法查
        self.assertIn("401", "\n".join(logs.output))
        self.assertIn("Authentication Fails", "\n".join(logs.output))

    def test_timeout_returns_none(self):
        client = self.make(transport=FakeTransport(TimeoutError("timed out")))
        self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))

    def test_network_error_returns_none(self):
        client = self.make(transport=FakeTransport(OSError("网络不可达")))
        self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))

    def test_bad_json_returns_none(self):
        client = self.make(transport=FakeTransport(b"not json"))
        self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))

    def test_empty_content_returns_none(self):
        raw = json.dumps({"choices": [{"message": {"content": "  "}}]}).encode()
        client = self.make(transport=FakeTransport(raw))
        self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))

    def test_lone_surrogate_returns_none_instead_of_raising(self):
        """⚠️ 契约回归：请求体构造抛的异常也必须被 complete 吃掉。

        孤立代理项（``\\udcab``）来自上游一次失败的解码 —— Windows 控制台
        按错误的码页读中文，或对端发来的字节不是合法 UTF-8。
        ``json.dumps(..., ensure_ascii=False).encode("utf-8")`` 遇到它会抛
        ``UnicodeEncodeError``，而这段构造**早先写在 try 外面**，异常直接从
        ``complete`` 里逃了出去 —— 契约上白纸黑字写着"永不抛异常"。

        实际后果：那一轮对话静默落回模板（调用方还有一层兜底），
        但契约是假的，而契约是**调用方敢不敢省掉兜底**的依据。
        """
        transport = FakeTransport(chat_payload("好"))
        client = self.make(transport=transport)
        with self.assertLogs("core.llm", level="WARNING") as logs:
            self.assertIsNone(client.complete(
                [{"role": "user", "content": "猫又来了\udcab"}]))
        self.assertEqual(transport.calls, [], "编不出去的请求不该发出去")
        # 日志要指向"输入文本有问题"，不能混进"返回内容不合法"
        self.assertIn("无法编码", "\n".join(logs.output))

    def test_surrogate_error_is_not_reported_as_bad_response(self):
        """UnicodeEncodeError 是 ValueError 的子类 —— except 的顺序必须对。

        顺序反了的话它会落进"大模型返回内容不合法"那个分支，
        日志把人指向服务端，而问题其实在本地这句话上。
        """
        self.assertTrue(issubclass(UnicodeEncodeError, ValueError))
        client = self.make(transport=FakeTransport(chat_payload("好")))
        with self.assertLogs("core.llm", level="WARNING") as logs:
            client.complete([{"role": "user", "content": "\ud800"}])
        self.assertNotIn("返回内容不合法", "\n".join(logs.output))

    def test_empty_messages_skips_the_call(self):
        transport = FakeTransport(chat_payload("好"))
        client = self.make(transport=transport)
        self.assertIsNone(client.complete([]))
        self.assertEqual(transport.calls, [])

    def test_no_credentials_names_the_variable(self):
        """缺凭证时**点名缺的是哪个变量** —— 用户明确要求过这一条。"""
        client = DeepSeekClient(api_key="", base_url="https://x", model="m")
        self.assertFalse(client.available)
        self.assertIn("B_DEEPSEEK_TOKEN", client.reason)

    def test_unavailable_client_never_calls_transport(self):
        """没配好就一次网络请求都不该发。"""
        transport = FakeTransport(chat_payload("好"))
        client = DeepSeekClient(api_key="", base_url="", model="",
                                transport=transport)
        self.assertFalse(client.available)
        self.assertIsNone(client.complete([{"role": "user", "content": "hi"}]))
        self.assertEqual(transport.calls, [])


class TestNullChatClient(unittest.TestCase):

    def test_is_never_available(self):
        client = NullChatClient()
        self.assertFalse(client.available)
        self.assertFalse(client.requires_network)

    def test_complete_returns_none(self):
        self.assertIsNone(NullChatClient().complete(
            [{"role": "user", "content": "hi"}]))

    def test_tolerates_extra_kwargs(self):
        """build_chatter 会把构造参数透传进来，不能因此炸掉。"""
        self.assertFalse(NullChatClient(api_key="x", 无关="y").available)


class TestProbeChatter(unittest.TestCase):

    def test_probe_uses_a_real_call(self):
        """探测必须真调一次 —— "配好了"和"能用"是两件事。"""
        transport = FakeTransport(chat_payload("你好"))
        client = TestDeepSeekClient().make(transport=transport)
        self.assertTrue(llm.probe_chatter(client))
        self.assertEqual(len(transport.calls), 1)

    def test_probe_fails_on_error(self):
        transport = FakeTransport(OSError("断网"))
        client = TestDeepSeekClient().make(transport=transport)
        self.assertFalse(llm.probe_chatter(client))

    def test_probe_uses_the_short_timeout(self):
        """⚠️ 探测用的必须是 LLM_TIMEOUT（默认 4 秒），不是 SYNTH_TIMEOUT 的 30 秒。

        搞错的话，一个填错的 Key 会让**启动**卡满半分钟 ——
        豆包那次已经在同一个地方踩过一次。
        """
        transport = FakeTransport(chat_payload("你好"))
        client = TestDeepSeekClient().make(transport=transport)
        llm.probe_chatter(client)
        self.assertEqual(transport.calls[0][1], config.LLM_TIMEOUT)
        self.assertLess(config.LLM_TIMEOUT, 10.0)

    def test_unavailable_client_is_not_probed(self):
        transport = FakeTransport(chat_payload("你好"))
        client = DeepSeekClient(api_key="", base_url="", model="",
                                transport=transport)
        self.assertFalse(llm.probe_chatter(client))
        self.assertEqual(transport.calls, [])


class TestBuildChatter(unittest.TestCase):

    def test_none_disables(self):
        for name in ("none", "null", "NONE", " none "):
            with self.subTest(name=name):
                client = build_chatter(name, probe=False, api_key="sk-test")
                self.assertFalse(client.available)
                self.assertIsInstance(client, NullChatClient)

    def test_empty_name_means_auto(self):
        """空名字等同 auto（和 build_synthesizer 的 ``(name or "auto")`` 一致）。

        这一条容易想当然地写成"空 = 关掉" —— 那会让 `--llm ""` 静默变成
        "不接大模型"，而命令行上看起来一切正常。
        """
        client = build_chatter("", probe=False, api_key="sk-test")
        self.assertTrue(client.available)

    def test_missing_credentials_degrade_to_null(self):
        """没配凭证 → NullChatClient（对话走模板），**不抛异常、不阻止启动**。"""
        client = build_chatter("auto", probe=False, api_key="", base_url="",
                               model="")
        self.assertFalse(client.available)
        self.assertIsInstance(client, NullChatClient)

    def test_reason_names_the_missing_variable(self):
        """降级原因的终点是启动横幅那一行，**必须点名缺哪个环境变量**。

        用户的原话是「无凭证的时候，控制台明确打印缺失的变量名称」。
        只在中间某个 WARNING 里打印是不够的 —— 横幅是操作者唯一
        一定会看的那一屏，写「没有可用的大模型」等于让他自己往上翻日志。
        """
        client = build_chatter("auto", probe=False, api_key="", base_url="",
                               model="")
        self.assertIn("B_DEEPSEEK_TOKEN", client.reason)

    def test_reason_carries_probe_failure(self):
        """试调失败也要能追溯到 —— 「Key 填了但无效」和「没填 Key」
        对操作者是两件事，横幅不能把两者说成同一句话。"""
        client = build_chatter("auto", probe=True, api_key="sk-test",
                               transport=FakeTransport(
                                   urllib.error.HTTPError(
                                       "u", 401, "Unauthorized", {},
                                       io.BytesIO(b'{"error":"bad key"}'))))
        self.assertFalse(client.available)
        self.assertIn("deepseek", client.reason)
        self.assertIn("试调", client.reason)

    def test_auto_picks_deepseek_when_configured(self):
        client = build_chatter("auto", probe=False, api_key="sk-test")
        self.assertTrue(client.available)
        self.assertEqual(client.name, "deepseek")

    def test_explicit_name_without_credentials_does_not_silently_disable(self):
        """``--llm deepseek`` 忘了填 Key 时，不能就此静默。

        和 `--tts doubao` 那条路一个道理：命令行上看起来一切正常、
        功能却没了，是最难查的一类。
        """
        client = build_chatter("deepseek", probe=False, api_key="",
                               base_url="", model="")
        self.assertFalse(client.available)
        self.assertIsInstance(client, NullChatClient)

    def test_unknown_name_does_not_fall_back_silently(self):
        """名字拼错是另一回事：不能悄悄换成别的模型。"""
        client = build_chatter("deepsek", probe=False, api_key="sk-test")
        self.assertIsInstance(client, NullChatClient)

    def test_probe_failure_degrades(self):
        """凭证填了但无效（探测失败）→ 降级到模板，不是每句话都失败。"""
        client = build_chatter("auto", probe=True, api_key="sk-test",
                               transport=FakeTransport(
                                   urllib.error.HTTPError(
                                       "u", 401, "Unauthorized", {},
                                       io.BytesIO(b'{"error":"bad key"}'))))
        self.assertIsInstance(client, NullChatClient)

    def test_never_raises(self):
        """任何情况下都不抛异常 —— 最差返回一个哑客户端。"""
        for name in ("auto", "deepseek", "none", "null", "乱写的",
                     "deepseek "):
            with self.subTest(name=name):
                client = build_chatter(name, probe=False)
                self.assertTrue(hasattr(client, "complete"))


# ==========================================================================
# 对话集成：规则判定不许被大模型改动，危机语句绝不外发
# ==========================================================================

class FakeLlm:
    """假的大模型客户端：返回预设文本，并记下每一次调用。

    ``calls`` 里存的是发给模型的 messages —— 安全断言要断言的正是
    "它有没有被调用"，而不是"回复长得像不像模板"。
    """

    name = "fake"
    requires_network = True
    available = True

    def __init__(self, reply="这是我编的一句回复。"):
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        if callable(self.reply):
            return self.reply(messages)
        return self.reply

    def close(self):
        return None


class TestLlmRouting(unittest.TestCase):
    """哪些话能进大模型。"""

    def setUp(self):
        self.fake = FakeLlm()
        self.engine = DialogueEngine(llm=self.fake)

    def test_chat_uses_the_llm(self):
        reply = self.engine.respond("我年轻时候在东北待过")
        self.assertEqual(reply.reply, self.fake.reply)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIn("来源=llm", reply.reason)

    def test_crisis_never_reaches_llm(self):
        """⚠️ 本次改动里最重要的一条安全断言。

        求救信号必须走确定性文案，而且**绝不能把这句话发给第三方**。
        断言的是 calls（客户端有没有被调用），不是回复内容 ——
        后者只能说明"这次看起来对"，前者才是保证。
        """
        for text in ("我不想活了", "活着没意思", "不如死了", "想死",
                     "没人在乎我", "走了算了"):
            with self.subTest(text=text):
                self.fake.calls.clear()
                reply = self.engine.respond(text)
                self.assertEqual(self.fake.calls, [],
                                 "危机语句被发给了大模型：%r" % text)
                self.assertTrue(reply.reply.strip())
                self.assertIn("来源=template", reply.reason)

    def test_crisis_with_whitespace_evasion_never_reaches_llm(self):
        """插空格不能绕过 —— 证明是分析器的归一化在兜底。

        如果哪天有人在 dialogue 里拿原文自己匹配 CRISIS_PATTERNS，
        这条会红，而上面那条仍然绿。
        """
        self.fake.calls.clear()
        self.engine.respond("我 不 想 活 了")
        self.assertEqual(self.fake.calls, [])

    def test_discomfort_never_reaches_llm(self):
        """身体不适只有 _candidates 那条分支会叫家属，不能让模型自由发挥。"""
        for text in ("我胸口疼", "我血压有点高", "头疼得厉害", "肚子不舒服"):
            with self.subTest(text=text):
                self.fake.calls.clear()
                self.engine.respond(text)
                self.assertEqual(self.fake.calls, [],
                                 "身体不适被发给了大模型：%r" % text)

    def test_identity_and_time_never_reach_llm(self):
        from core.dialogue import INTENT_FALLBACK
        for text, intent in (("你是谁？", "identity"), ("现在几点了", "time")):
            with self.subTest(text=text):
                self.fake.calls.clear()
                reply = self.engine.respond(text)
                self.assertEqual(self.fake.calls, [])
                self.assertEqual(reply.intent, intent)
                if intent == "identity":
                    # 既有规格（test_core.py 也钉着同一条）
                    self.assertIn(reply.reply, INTENT_FALLBACK["identity"])

    def test_greeting_farewell_thanks_never_reach_llm(self):
        """这几个是延迟敏感回合，模板也最强 —— 不进大模型。

        「你好」是用户说的第一句话，让它等 5 秒正是预合成缓存要消灭的
        那个失败模式。
        """
        for text in ("你好呀", "再见", "谢谢你", "你真棒"):
            with self.subTest(text=text):
                self.fake.calls.clear()
                self.engine.respond(text)
                self.assertEqual(self.fake.calls, [],
                                 "问候/告别/道谢被发给了大模型：%r" % text)

    def test_filler_never_reaches_llm(self):
        """「嗯」「哦」没有可回应的内容，模型只会编一个话题出来。"""
        for text in ("嗯", "哦", "好"):
            with self.subTest(text=text):
                self.fake.calls.clear()
                self.engine.respond(text)
                self.assertEqual(self.fake.calls, [])

    def test_proactive_never_calls_llm(self):
        """主动开口 100% 保持模板：摄像头触发、逐句审过、跨重启去重。"""
        self.fake.calls.clear()
        for kind in ("care_sad", "care_tired", "greeting"):
            self.engine.proactive_reply(kind)
        self.assertEqual(self.fake.calls, [])

    def test_missing_llm_uses_templates(self):
        """没有大模型时一切照旧 —— 这是离线路径的核心保证。"""
        engine = DialogueEngine()
        reply = engine.respond("我年轻时候在东北待过")
        self.assertTrue(reply.reply.strip())
        self.assertIn("来源=template", reply.reason)

    def test_unavailable_llm_uses_templates(self):
        """available=False 的客户端等价于"没有大模型"。"""
        class Dead(FakeLlm):
            available = False
        dead = Dead()
        reply = DialogueEngine(llm=dead).respond("我年轻时候在东北待过")
        self.assertEqual(dead.calls, [])
        self.assertIn("来源=template", reply.reason)


class TestLlmDoesNotChangeRuleJudgements(unittest.TestCase):
    """大模型只产出文本，规则判定的结果一个都不许变。"""

    def test_state_intent_emotion_are_unchanged(self):
        """⚠️ 8001 状态机和 CSV 列的守卫。

        模型返回一句跟用户状态完全无关的话，state / intent / emotion_* /
        discomfort 仍必须等于"纯规则"的结果 —— 否则前端 C 显示的灯和
        历史记录里的列都会跟着模型跑。
        """
        from core.vision_state import VisionState

        for state in config.VALID_STATES:
            for text in ("我年轻时候在东北待过", "你说人这一辈子图个啥"):
                with self.subTest(state=state, text=text):
                    rule_engine = DialogueEngine()
                    llm_engine = DialogueEngine(
                        llm=FakeLlm("随便一句完全无关的话。"))
                    vision = VisionState(state=state)
                    expected = rule_engine.respond(text, vision)
                    actual = llm_engine.respond(text, vision)
                    self.assertEqual(actual.state, expected.state)
                    self.assertEqual(actual.intent, expected.intent)
                    self.assertEqual(actual.emotion_label, expected.emotion_label)
                    self.assertEqual(actual.emotion_score, expected.emotion_score)
                    self.assertEqual(actual.emotion_detail, expected.emotion_detail)
                    self.assertEqual(actual.discomfort, expected.discomfort)

    def test_reply_invariants_hold_with_hostile_llm(self):
        """把 test_core.py 那张矩阵换成**故意捣乱**的模型输出再跑一遍。

        大模型路径必须受制于和模板路径**完全相同**的契约：
        非空、不含 {、state 合法。
        """
        from core.vision_state import VisionState

        hostile = ["**嗯**{topic}", "", "a" * 300, "😀", "（笑）", "我是真人。",
                   "该吃两片药了。", "\n\n", "好" * 200 + "。"]
        for bad in hostile:
            with self.subTest(reply=bad[:12]):
                engine = DialogueEngine(llm=FakeLlm(bad))
                for text in ("我年轻时候在东北待过", "你说人这一辈子图个啥", "嗯"):
                    for state in config.VALID_STATES:
                        reply = engine.respond(text, VisionState(state=state))
                        self.assertTrue(reply.reply.strip(),
                                        "%r/%s 回复为空" % (bad, state))
                        self.assertNotIn("{", reply.reply)
                        self.assertIn(reply.state, config.VALID_STATES)

    def test_llm_reply_is_sanitized(self):
        engine = DialogueEngine(llm=FakeLlm("**您好**，我在这儿呢 😊"))
        reply = engine.respond("我年轻时候在东北待过")
        self.assertNotIn("*", reply.reply)
        self.assertNotIn("😊", reply.reply)
        self.assertIn("来源=llm", reply.reason)

    def test_llm_failure_falls_back_to_template(self):
        """模型返回 None（断网/超时/坏 JSON 都归到这里）→ 落回模板。"""
        class Failing(FakeLlm):
            def complete(self, messages):
                self.calls.append(messages)
                return None
        failing = Failing()
        reply = DialogueEngine(llm=failing).respond("我年轻时候在东北待过")
        self.assertTrue(reply.reply.strip())
        self.assertIn("来源=template", reply.reason)

    def test_llm_exception_falls_back_to_template(self):
        """客户端违约抛异常时也不能把对话带崩。"""
        engine = DialogueEngine(llm=FakeLlm(OSError("炸了")))
        reply = engine.respond("我年轻时候在东北待过")
        self.assertTrue(reply.reply.strip())
        self.assertIn("来源=template", reply.reason)

    def test_llm_overlong_falls_back_to_template(self):
        engine = DialogueEngine(llm=FakeLlm("这句话特别长长长长长" * 20))
        reply = engine.respond("我年轻时候在东北待过")
        self.assertLessEqual(len(reply.reply), config.LLM_MAX_CHARS)
        self.assertIn("来源=template", reply.reason)

    def test_llm_repeat_is_rejected(self):
        """模型重复上一句时改用模板 —— 复用既有的历史去重。

        陪伴场景里"同一句话连说两遍"比说得不好更伤人。
        """
        repeated = "您今天气色不错呀。"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "history.csv")
            store = CsvHistoryStore(path, session_id="repeat-test")
            store.append_turn("你好", repeated, config.STATE_NORMAL,
                              "neutral", 0.0, "chat")
            engine = DialogueEngine(history=store, llm=FakeLlm(repeated))
            reply = engine.respond("我年轻时候在东北待过")
            self.assertNotEqual(reply.reply, repeated)
            self.assertIn("来源=template", reply.reason)

    def test_dialogue_engine_default_has_no_llm(self):
        """默认构造**绝不能**带网络行为 —— 几十处测试都这么构造它。"""
        engine = DialogueEngine()
        self.assertIsNone(engine._llm)
        eligible, why = engine.llm_eligible(
            "chat", engine.analyzer.analyze("随便说点什么"), "随便说点什么")
        self.assertFalse(eligible)
        self.assertIn("未启用", why)


class TestLlmContext(unittest.TestCase):
    """发给模型的消息数组怎么拼。"""

    def build(self, history=None):
        fake = FakeLlm()
        return DialogueEngine(history=history, llm=fake), fake

    def test_message_roles_and_order(self):
        engine, fake = self.build()
        engine.respond("我年轻时候在东北待过")
        messages = fake.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "我年轻时候在东北待过")

    def test_current_utterance_appears_exactly_once(self):
        """⚠️ 当前这句话必须只出现一次。

        respond 跑在 main.handle_chat 写 CSV **之前**，所以历史里还没有它。
        如果哪天有人"顺手"在 respond 里先落库，这句就会被喂两遍。
        """
        with tempfile.TemporaryDirectory() as directory:
            store = CsvHistoryStore(os.path.join(directory, "history.csv"),
                                    session_id="ctx")
            store.append_turn("你好呀", "您好呀。", config.STATE_NORMAL,
                              "neutral", 0.0, "greeting")
            engine, fake = self.build(history=store)
            engine.respond("我年轻时候在东北待过")
            messages = fake.calls[0]
            occurrences = sum(1 for m in messages
                              if m["content"] == "我年轻时候在东北待过")
            self.assertEqual(occurrences, 1)

    def test_history_roles_are_mapped(self):
        """CSV 里是 robot/user，接口要 assistant/user。

        不做映射的话请求会被拒，或者更糟：把机器人的话当成用户说的。
        """
        with tempfile.TemporaryDirectory() as directory:
            store = CsvHistoryStore(os.path.join(directory, "history.csv"),
                                    session_id="roles")
            store.append_turn("我今天有点闷", "我陪着您呢。",
                              config.STATE_SAD, "negative", -1.0, "venting")
            engine, fake = self.build(history=store)
            engine.respond("我年轻时候在东北待过")
            roles = [m["role"] for m in fake.calls[0]]
            self.assertIn("assistant", roles)
            self.assertNotIn("robot", roles)
            self.assertTrue(all(r in ("system", "user", "assistant")
                                for r in roles))

    def test_other_sessions_are_excluded(self):
        """只喂本次会话的历史。

        history.csv 会跨多次演示累积，不过滤的话新会话第一句话就会把
        上次排练的尾巴（还包括别人的话）喂给模型。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "history.csv")
            other = CsvHistoryStore(path, session_id="上次排练")
            other.append_turn("这是上次排练说的话", "上次的回复",
                              config.STATE_NORMAL, "neutral", 0.0, "chat")
            mine = CsvHistoryStore(path, session_id="本次")
            mine.append_turn("我今天有点闷", "我陪着您呢。",
                             config.STATE_SAD, "negative", -1.0, "venting")
            engine, fake = self.build(history=mine)
            engine.respond("我年轻时候在东北待过")
            joined = json.dumps(fake.calls[0], ensure_ascii=False)
            self.assertNotIn("上次排练说的话", joined)
            self.assertIn("我今天有点闷", joined)

    def test_no_history_is_fine(self):
        engine, fake = self.build(history=None)
        engine.respond("我年轻时候在东北待过")
        self.assertEqual(len(fake.calls[0]), 2)   # system + 当前这句

    def test_broken_history_does_not_break_the_reply(self):
        class Broken:
            def recent_dialogue(self, **kwargs):
                raise OSError("历史文件坏了")
        engine, fake = self.build(history=Broken())
        reply = engine.respond("我年轻时候在东北待过")
        self.assertTrue(reply.reply.strip())
        self.assertEqual(len(fake.calls[0]), 2)

    def test_system_prompt_carries_the_state(self):
        """状态要进提示词 —— 不然模型的话和视觉判定的状态对不上。"""
        from core.vision_state import VisionState
        engine, fake = self.build()
        engine.respond("我年轻时候在东北待过",
                       VisionState(state=config.STATE_TIRED))
        system = fake.calls[0][0]["content"]
        self.assertIn(llm.STATE_HINTS[config.STATE_TIRED], system)


class TestRecentDialogueSessionFilter(unittest.TestCase):
    """history_store.recent_dialogue 的 session_only 开关。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "history.csv")
        self.other = CsvHistoryStore(self.path, session_id="other")
        self.other.append_turn("别的会话", "别的回复", config.STATE_NORMAL,
                               "neutral", 0.0, "chat")
        self.mine = CsvHistoryStore(self.path, session_id="mine")
        self.mine.append_turn("我的会话", "我的回复", config.STATE_NORMAL,
                              "neutral", 0.0, "chat")

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_is_a_pass_through(self):
        """默认 False 必须保持原来的语义 —— 这个方法的既有调用方不该受影响。"""
        records = self.mine.recent_dialogue(turns=10)
        texts = [r["text"] for r in records]
        self.assertIn("别的会话", texts)
        self.assertIn("我的会话", texts)

    def test_session_only_excludes_others(self):
        records = self.mine.recent_dialogue(turns=10, session_only=True)
        texts = [r["text"] for r in records]
        self.assertIn("我的会话", texts)
        self.assertNotIn("别的会话", texts)

    def test_zero_turns_returns_empty(self):
        self.assertEqual(self.mine.recent_dialogue(turns=0), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
