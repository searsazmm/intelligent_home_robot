# -*- coding: utf-8 -*-
"""对话管理：根据「视觉状态 + 用户文本情绪 + 对话内容」生成回复。

任务要求 3。**默认不接大模型**；配了 DeepSeek 凭证才接，而且规则模板永远是兜底。
两层是叠加关系，不是替换：

    规则模板  ——  可预测、可调试、零第三方依赖、**永远有话说**
    大模型    ——  接住没预设过的话（模板只有 8 个意图 + 3 句通用兜底，
                  这才是"回答生硬、接不住话"的根因）

规则引擎的好处没有丢：每条回复为什么被选中，仍然能指着代码说清楚 ——
联调时看日志里的 (intent, state, emotion, 来源) 就知道该去改哪条模板。

一次 respond() 做四件事：
    1. 识别意图        recognize_intent()
    2. 融合状态        fuse_state()  —— 视觉状态 与 文本情绪 合并成 4 态之一
    3. 生成回复        大模型（若可用且适用）否则挑模板，避开最近说过的
    4. 落库/给上下文    交给外部（main.py）写历史，这里只返回结果

⚠️ 第 3 步里，**大模型只负责产出「回复文本」这一件事**。
   state / intent / emotion_* / discomfort 全部保持规则判定，一行都不动 ——
   它们驱动 8001 状态机、CSV 的列、以及前端 C 的显示，不能让模型改写。

融合状态会同时用于两处：
    - 决定回复的语气和内容（也会写进系统提示词，让模型的话对得上状态）
    - 作为 B→C 推送的状态（api_doc §4.2 只允许 normal/sad/tired/absent）
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
from core.llm import build_system_prompt, sanitize_reply
from core.text_emotion import TextEmotion, TextEmotionAnalyzer
from core.vision_state import VisionState

logger = logging.getLogger(__name__)

# 用户发了消息后，多久之内认为"人在"，可以推翻视觉的 absent 判定（秒）
# 场景：用户低头打字/摄像头被挡，视觉判 absent，但人明明在讲话。
TEXT_PRESENCE_WINDOW = 30.0

# 回答内容与用户状态无关的意图：不管用户是累还是低落，问"你是谁"都得答身份。
# 这些意图的模板优先级要高于"状态通用兜底"。
STATE_INDEPENDENT_INTENTS = frozenset({"identity", "time", "question"})

# 回声模板里最少要有多长的"话题"才值得回显。
# 用户只发"嗯""哦""好"这种语气词时，回一句"嗯嗯，嗯——然后呢？"很傻，
# 所以话题太短时直接弃用带 {topic} 的模板。
MIN_TOPIC_LENGTH = 3

# 只有这几个意图交给大模型。
#
# ⚠️ 这是**允许表，不是拒绝表** —— 默认拒绝的那一侧才是安全的那一侧：
#     新加一个意图却忘了更新这张表 → 后果是"少用一次大模型"（照常回复，只是用了模板）
#     新加一个意图却忘了更新黑名单 → 后果是"一个本该确定性的回答被模型自由发挥"
#
# 被排除的，以及为什么：
#   identity / time   问身份必须答身份（test_core.py 有断言钉着）；问时间更糟 ——
#                     模型没有时钟，只会瞎编，模板"具体时间我这边看得不太准"才是诚实答案
#   discomfort        健康问题只有一个地方会说"要不要我帮您叫家里人"，就是 _candidates
#                     的那条分支；模型拿"多喝热水"回胸口疼，比模板更差
#   greeting/farewell/thanks/praise
#                     这几个是**延迟敏感**回合：「你好」是用户说的第一句话，
#                     让它等 5 秒正是预合成缓存要消灭的那个失败模式。
#                     而且这几组模板是全仓最强的（(tired, greeting)、(sad, greeting)
#                     都手工调过），状态条件化在这里最有用。零收益、真回退。
#
# 另有两条**前置检查**不在这个集合里，见 llm_eligible：危机语句、身体不适。
LLM_ELIGIBLE_INTENTS = frozenset({"chat", "venting", "question"})

# history.csv 里的角色名 → OpenAI 兼容接口的角色名。
# CSV 写的是 "robot"/"user"，接口要的是 "assistant"/"user"。不做这个映射的话
# 请求会被服务端拒掉（或者更糟：把机器人的话当成用户说的）。
HISTORY_ROLE_TO_CHAT_ROLE = {"robot": "assistant", "user": "user"}


# --------------------------------------------------------------------------
# 意图识别词表
# 顺序有意义：越靠前的越优先匹配（"你好，谢谢" 应识别为 greeting）
# --------------------------------------------------------------------------

#: 这几个关键词**只有整个输入就是它自己**时才算命中（允许跟一个标点）。
#:
#: 「早」是中文里最自然的打招呼方式之一，但作为**子串**它出现在大量
#: 完全无关的话里：「我们家那口子走得早」「我起得早」「你早点睡」。
#: 后果不只是分类错 —— 这些句子会被当成问候、回一句热情的招呼，
#: 而「我们家那口子走得早」是在说老伴去世，答非所问之外还伤人。
#: 词表里其他关键词都是两字以上，只有「早」有这个毛病。
STANDALONE_KEYWORDS = frozenset({"早"})

#: 判断"整句就是这个词"时，允许挂在后面的字符（标点与空白）。
#: 「早」「早！」「早，」都算打招呼；「早，我吃过了」不算（后面还有话）。
_TRAILING_NOISE = " \t\r\n，。！？、；：,.!?;:~～…"

INTENT_RULES: List[tuple] = [
    ("farewell", (
        "再见", "拜拜", "走了", "我睡", "睡觉了", "不聊了", "下次聊", "先这样",
        "回头聊", "晚安", "挂了", "出门",
    )),
    ("greeting", (
        "你好", "您好", "早上好", "中午好", "下午好", "晚上好", "在吗", "在不在",
        "嗨", "哈喽", "hello", "hi", "早",
    )),
    ("thanks", ("谢谢", "感谢", "多谢", "辛苦了", "麻烦你")),
    ("identity", (
        "你是谁", "你叫什么", "你的名字", "你是什么", "介绍一下你", "你会什么", "你能做什么",
    )),
    ("time", ("几点", "现在时间", "几号", "今天星期", "今天几", "什么时候")),
    ("discomfort", (
        "不舒服", "头疼", "头晕", "胃疼", "肚子疼", "腰疼", "腿疼", "感冒", "发烧",
        "咳嗽", "没胃口", "吃不下", "血压", "心慌", "疼",
    )),
    ("praise", ("真好", "真棒", "你真", "喜欢你", "谢谢你陪", "有你真好")),
    ("question", ("怎么", "为什么", "是什么", "能不能", "可以吗", "有没有", "多少", "吗？", "呢？")),
]


@dataclass
class DialogueReply:
    """一次对话的完整结果。"""

    reply: str                                  # 要发给用户的回复文本
    state: str                                  # 融合后的四态之一，用于推送 C
    intent: str = "chat"                        # 识别到的意图
    emotion_label: str = "neutral"              # 文本情绪三分类
    emotion_score: float = 0.0
    emotion_detail: str = "neutral"             # 细粒度情绪
    discomfort: bool = False                    # 是否提到身体不适
    reason: str = ""                            # 为什么这么回（调试用）

    def to_dict(self) -> dict:
        """转为发给 C 端 8002 的 JSON 结构。"""
        return {
            "type": "reply",
            "text": self.reply,
            "state": self.state,
            "intent": self.intent,
            "emotion": {
                "label": self.emotion_label,
                "detail": self.emotion_detail,
                "score": round(self.emotion_score, 2),
            },
            "timestamp": time.time(),
        }


class DialogueEngine:
    """对话管理器：规则判定 + （可选）大模型生成回复文本。

    意图 / 状态 / 情绪永远是规则算出来的，大模型只借"回复文本"这一件事 ——
    见模块 docstring，以及 :meth:`llm_eligible` 的允许表。

    线程安全说明：会修改的只有 self._last_user_text_at，单次赋值是原子的。
    真正的共享状态（历史 CSV）由外部 store 负责加锁。
    """

    def __init__(self, history=None, rng: Optional[random.Random] = None,
                 llm=None) -> None:
        """``llm`` 是大模型客户端（见 :mod:`core.llm`），**默认 None**。

        ⚠️ 默认值必须是 None，不能是"从 config 建一个"：
        那样的话每个 ``DialogueEngine()`` 都会去打网络 —— 包括测试里
        那几十处构造。任何 ``available`` 为假的客户端都等价于"没有大模型"，
        所以传 NullChatClient 进来也是安全的。
        """
        self.analyzer = TextEmotionAnalyzer()
        self.history = history                  # CsvHistoryStore 或 None
        self._rng = rng or random.Random()
        self._last_user_text_at: float = 0.0
        self._llm = llm                         # Chatter 或 None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def respond(
        self,
        user_text: str,
        vision: Optional[VisionState] = None,
        *,
        from_user: bool = True,
    ) -> DialogueReply:
        """生成回复。user_text 为空时返回一句兜底提示，不抛异常。

        ``from_user=False`` 表示这段话虽然也走了同一套融合/生成逻辑，
        但**不是用户说的**（目前只有一种情况：机器人主动开口，见
        :mod:`core.proactive`）。此时不更新 ``_last_user_text_at``。

        为什么必须区分：``_last_user_text_at`` 是「用户最近说过话」的证据，
        而 :meth:`fuse_state` 拿它来推翻视觉的 absent 判定。如果机器人自己
        说的话也算进去，就会变成 —— 机器人对着空房间说一句"我陪着您"，
        系统立刻据此认定「人在」，`absent` 被改写成 `normal`。
        等于用自己的回声证明了房间里有人。
        """
        user_text = (user_text or "").strip()
        vision = vision or VisionState()

        if not user_text:
            return DialogueReply(
                reply="我在这里呢，您慢慢说。",
                state=vision.state,
                reason="空输入兜底",
            )

        if from_user:
            self._last_user_text_at = time.monotonic()

        emotion = self.analyzer.analyze(user_text)
        intent = self.recognize_intent(user_text, emotion)
        state = self.fuse_state(vision, emotion)

        # 大模型只产出「回复文本」。上面那三行（意图 / 状态 / 情绪）已经是
        # 最终结果，不会因为走了大模型而变 —— 见模块 docstring 的说明。
        source = "llm"
        reply_text = self._online_reply(state, intent, emotion, user_text)
        if reply_text is None:
            # 不适用（危机 / 不适 / 问候这类）或调用失败，一律落回模板。
            source = "template"
            reply_text = self._compose_reply(state, intent, emotion, user_text, vision)

        return DialogueReply(
            reply=reply_text,
            state=state,
            intent=intent,
            emotion_label=emotion.label,
            emotion_score=emotion.score,
            emotion_detail=emotion.emotion,
            discomfort=emotion.discomfort,
            reason=f"视觉={vision.state}({vision.reason}) 文本={emotion.label}/{emotion.emotion} 意图={intent} 来源={source}",
        )

    # ------------------------------------------------------------------
    # 主动开口（机器人先说话，用户什么都没说）
    # ------------------------------------------------------------------

    def proactive_reply(self, kind: str, vision: Optional[VisionState] = None) -> str:
        """挑一句主动开口的话。``kind`` 由 :mod:`core.proactive` 的策略决定。

        文案放在这里、而不是新建一个文案模块，理由有两个：

        1. 去重逻辑只有一处。``_recent_replies()`` 读的是历史 CSV 里
           ``role=robot`` 的行，所以**跨重启**仍然记得说过什么 ——
           另起一套文案就享受不到这个。
        2. 中文回复模板全仓只有这一个文件，改文案的人不会漏掉这里的。

        注意这里**不调用** :meth:`respond`：那条路会更新 ``_last_user_text_at``
        （见 :meth:`respond` 的 ``from_user`` 参数说明）。主动开口是「机器人
        说给用户听」，不是「用户说了话」，两者的副作用必须分开。
        """
        candidates = list(PROACTIVE_TEMPLATES.get(kind) or PROACTIVE_TEMPLATES["greeting"])

        # 和普通应答共用同一套「避开最近说过的」逻辑，靠历史 CSV 去重
        recent = self._recent_replies()
        fresh = [text for text in candidates if text not in recent]
        pool = fresh or candidates

        return self._rng.choice(pool)

    # ------------------------------------------------------------------
    # 1) 意图识别
    # ------------------------------------------------------------------

    def recognize_intent(self, text: str, emotion: TextEmotion) -> str:
        """关键词匹配。命中多条时取 INTENT_RULES 里靠前的那个。

        单字关键词（见 :data:`STANDALONE_KEYWORDS`）要求整句就是它本身 ——
        否则「走得早」这种会把一句悼念识别成打招呼。
        """
        stripped = (text or "").strip()
        for intent, keywords in INTENT_RULES:
            for keyword in keywords:
                if keyword in STANDALONE_KEYWORDS:
                    if stripped.rstrip(_TRAILING_NOISE) == keyword:
                        return intent
                    continue
                if keyword in text:
                    return intent

        # 没命中关键词时，用情绪兜底：明显消极的话语按倾诉处理
        if emotion.label == "negative":
            return "venting"
        return "chat"

    # ------------------------------------------------------------------
    # 2) 状态融合
    # ------------------------------------------------------------------

    def fuse_state(self, vision: VisionState, emotion: TextEmotion) -> str:
        """把视觉状态和文本情绪合并成一个状态。

        优先级（高 → 低）：absent > tired > sad > normal
        理由：视觉比文本"更硬" —— 用户嘴上说"我没事"，但眼睛已经睁不开了，
        这时报 tired 比报 normal 更有用。反之文字里的强烈消极情绪也能把
        normal 拉低成 sad，因为视觉特征可能只是没捕捉到。
        """
        # 用户刚刚还在打字/说话，说明人在，推翻视觉的 absent
        if vision.state == config.STATE_ABSENT and self._text_recently_active():
            vision = VisionState(
                state=config.STATE_NORMAL,
                reason="用户刚刚发过消息，视为在场",
                has_face=vision.has_face,
                stale=False,
                confidence=0.5,
            )

        if vision.state == config.STATE_ABSENT:
            return config.STATE_ABSENT

        # 疲劳：视觉判疲劳，或文字里明确表达疲惫
        if vision.state == config.STATE_TIRED or emotion.emotion == "tired":
            return config.STATE_TIRED

        # 低落：视觉判低落，或文字明显消极（含身体不适）
        if vision.state == config.STATE_SAD or emotion.is_negative():
            return config.STATE_SAD

        # 积极文本 + 视觉正常 → normal
        return config.STATE_NORMAL

    @property
    def last_user_text_at(self) -> float:
        """用户最近一次说话的时刻（单调时钟）；0 表示本次运行还没说过。

        给 :mod:`core.proactive` 用 —— 主动关怀要靠它判断「用户刚说完话，
        先别插嘴」。之所以做成属性而不是让 proactive 直接读私有字段：
        这个值是**唯一**的「用户在场证据」，麦克风 / 8002 / ``--stdin``
        三条输入路径全部经由 :meth:`respond` 汇到这里，坏一处三条全坏，
        所以只留一个出口。
        """
        return self._last_user_text_at

    def _text_recently_active(self) -> bool:
        """最近 TEXT_PRESENCE_WINDOW 秒内用户是否说过话。"""
        if self._last_user_text_at <= 0:
            return False
        return (time.monotonic() - self._last_user_text_at) <= TEXT_PRESENCE_WINDOW

    # ------------------------------------------------------------------
    # 3) 回复生成
    # ------------------------------------------------------------------

    def _compose_reply(
        self,
        state: str,
        intent: str,
        emotion: TextEmotion,
        user_text: str,
        vision: VisionState,
    ) -> str:
        """按优先级挑模板。

        优先级：(状态, 意图) 精确匹配 → (状态) 通用 → 全局兜底
        这样既能对"疲劳 + 告别"说"那您早点休息"，也能对没预设过的组合兜底。
        """
        candidates = self._candidates(state, intent, emotion)

        # 用户只说了"嗯"这种语气词时，别回显，换成不带 {topic} 的模板
        if len(self._topic_snippet(user_text)) < MIN_TOPIC_LENGTH:
            without_topic = [text for text in candidates if "{topic}" not in text]
            if without_topic:
                candidates = without_topic

        # 避开最近说过的句子，让陪伴感不那么机械
        recent = self._recent_replies()
        fresh = [text for text in candidates if text not in recent]
        pool = fresh or candidates

        template = self._rng.choice(pool)
        return self._fill(template, user_text, vision, emotion)

    def _candidates(self, state: str, intent: str, emotion: TextEmotion) -> List[str]:
        """收集候选回复。"""
        # 危机信号优先级最高，排在身体不适**之前** —— 这两件事会同时出现
        # （"我疼得不想活了"），而这时候该回应的是"不想活"，不是"疼"。
        #
        # 这几句刻意写成**确定性的**、逐句审过的：
        #   - 不说教、不追问原因（追问会让对方更孤立）
        #   - 只做两件事：接住情绪 + 把人往"身边有人"上引
        #   - 不用"您别想不开"这类否定式劝阻（研究上无效，还像在敷衍）
        # 和主动关怀的文案不同，这几句**允许**用问句 —— 那句"现在身边有人吗"
        # 是在确认安全，不是在索取信息。
        if emotion.crisis:
            return list(CRISIS_REPLIES)

        # 身体不适其次 —— 健康问题不能只回一句"多喝热水"式的套话
        if emotion.discomfort or intent == "discomfort":
            return list(DISCOMFORT_REPLIES)

        # (状态, 意图) 精确匹配
        key = (state, intent)
        if key in REPLY_TEMPLATES:
            return list(REPLY_TEMPLATES[key])

        # 有些意图的回答内容和用户状态无关，问"你是谁"时不管他累不累，
        # 都得答出"我是谁"。这类必须排在状态兜底之前，否则会被通用套话盖掉。
        if intent in STATE_INDEPENDENT_INTENTS and intent in INTENT_FALLBACK:
            return list(INTENT_FALLBACK[intent])

        # 状态通用
        if state in STATE_FALLBACK:
            return list(STATE_FALLBACK[state])

        # 意图通用
        if intent in INTENT_FALLBACK:
            return list(INTENT_FALLBACK[intent])

        return list(DEFAULT_REPLIES)

    def _recent_replies(self, count: int = 5) -> List[str]:
        """取最近几条机器人回复，用于去重。没有历史存储时返回空。"""
        if self.history is None:
            return []
        try:
            return self.history.last_robot_replies(count)
        except Exception:
            # 历史读取失败不能影响回复生成
            return []

    @staticmethod
    def _fill(template: str, user_text: str, vision: VisionState, emotion: TextEmotion) -> str:
        """把模板里的占位符替换成实际内容。"""
        topic = DialogueEngine._topic_snippet(user_text)
        return (
            template
            .replace("{topic}", topic)
            .replace("{state_reason}", vision.reason or "")
            .replace("{len}", str(len(user_text)))
        )

    @staticmethod
    def _topic_snippet(text: str, max_len: int = 12) -> str:
        """从用户话里截一小段作为"话题"，让回复有回声感。

        只取第一句，太长就截断 —— 避免把用户整段话复述回去显得啰嗦。
        """
        stripped = text.strip()
        for separator in ("。", "！", "？", "，", ",", ".", "!", "?", "\n"):
            index = stripped.find(separator)
            if index > 0:
                stripped = stripped[:index]
                break
        if len(stripped) > max_len:
            stripped = stripped[:max_len]
        return stripped

    # ------------------------------------------------------------------
    # 3b) 大模型这条支路
    #
    # 只产出「回复文本」。判断"能不能用"和"洗得干不干净"都在这边，
    # 客户端（core/llm.py）只负责发请求和拿文本。
    # ------------------------------------------------------------------

    def llm_eligible(self, intent: str, emotion: TextEmotion,
                     user_text: str) -> Tuple[bool, str]:
        """这句话能不能交给大模型。返回 ``(可以?, 不可以的原因)``。

        刻意做成**纯函数、不碰网络**：于是"危机语句绝不进大模型"这条安全断言
        可以直接单测 —— 不需要真的调一次模型来证明它没被调。

        检查顺序有意义，从最不能商量的开始。
        """
        if self._llm is None or not getattr(self._llm, "available", False):
            return False, "大模型未启用"

        # 最高优先级。求救信号必须是确定性的、逐句审过的回复 ——
        # 而且不能把这句话发给第三方。
        if emotion.crisis:
            return False, "危机语句，必须走确定性回复"

        # 兜住"我血压有点高"这种意图判成 chat、但命中了身体不适词典的情况。
        if emotion.discomfort or intent == "discomfort":
            return False, "身体不适，必须走确定性回复"

        if intent not in LLM_ELIGIBLE_INTENTS:
            return False, f"意图 {intent} 保持确定性回复"

        # 「嗯」「哦」没有可回应的内容，交给模型只会得到废话，
        # 或者更糟：它自己编一个话题出来。
        if len(self._topic_snippet(user_text)) < MIN_TOPIC_LENGTH:
            return False, "只说了语气词，没有可回应的内容"

        return True, ""

    def _online_reply(self, state: str, intent: str, emotion: TextEmotion,
                      user_text: str) -> Optional[str]:
        """让大模型生成一句回复。返回 ``None`` 表示"不适用或失败，请落回模板"。

        契约：**只可能返回一句洗干净的、非空的文本，或者 None**。
        绝不抛异常 —— 调用方拿到 None 就落回模板，那条路永远是好的。
        """
        eligible, why = self.llm_eligible(intent, emotion, user_text)
        if not eligible:
            logger.debug("这句不走大模型：%s", why)
            return None

        try:
            raw = self._llm.complete(
                self._llm_messages(state, intent, emotion, user_text))
        except Exception:
            # complete 的契约是"永不抛异常"，但不能指望每个实现都守约。
            logger.warning("大模型调用抛了异常，这一句回退规则模板", exc_info=True)
            return None
        if raw is None:
            return None

        text = sanitize_reply(raw)
        if text is None:
            return None

        # 和模板共用同一套"避开最近说过的"逻辑：模型偶尔会连着两次用同一个
        # 套路（"您今天气色不错"说两遍），这里统一治掉。
        if text in self._recent_replies():
            logger.info("大模型这句最近说过，改用模板：%s", text)
            return None

        return text

    def _llm_messages(self, state: str, intent: str,
                      emotion: TextEmotion, user_text: str) -> List[Dict[str, str]]:
        """拼出发给大模型的消息数组。"""
        messages: List[Dict[str, str]] = [{
            "role": "system",
            "content": build_system_prompt(state, intent, emotion.label),
        }]
        messages.extend(self._history_messages())
        messages.append({"role": "user", "content": user_text})
        return messages

    def _history_messages(self, turns: Optional[int] = None) -> List[Dict[str, str]]:
        """最近几轮对话，转成接口要的 role 词表。

        ⚠️ 当前这句话**不在这里**：``respond`` 跑在 ``main.handle_chat`` 写 CSV
        之前，所以历史里只有之前说过的话。别"修"这一点 —— 修了就会把用户
        这句话喂两遍（一次在历史里，一次在最后那条 user 消息里）。

        只取**本次会话**：``data/history.csv`` 会跨多次演示累积，不过滤的话
        新会话第一句话就会把上次排练的尾巴（还包括别人的话）喂给模型。
        """
        if self.history is None:
            return []
        try:
            records = self.history.recent_dialogue(
                turns=turns if turns is not None else config.LLM_CONTEXT_TURNS,
                session_only=True,
            )
        except Exception:
            # 历史读不到不能影响回复生成（和 _recent_replies 一个道理）
            logger.debug("读取对话历史失败", exc_info=True)
            return []

        messages: List[Dict[str, str]] = []
        for record in records:
            text = (record.get("text") or "").strip()
            if not text:
                continue
            # 未知角色兜底成 user —— 一个野角色不该让整个请求被拒
            role = HISTORY_ROLE_TO_CHAT_ROLE.get(record.get("role"), "user")
            messages.append({"role": role, "content": text})
        return messages


# --------------------------------------------------------------------------
# 回复模板库
# 按 (状态, 意图) 组织。写模板的原则：
#   - 一句话，别超过 40 字（老人听不了长句）
#   - 不说教、不命令，先共情再建议
#   - 需要时用 {topic} 回应用户说的内容
# --------------------------------------------------------------------------

REPLY_TEMPLATES: Dict[tuple, List[str]] = {
    # ---- 疲劳 tired ----
    (config.STATE_TIRED, "greeting"): [
        "您好呀。看您好像有点乏，先坐下歇一会吧。",
        "来啦。您今天精神看着不太足，要不要先闭眼养养神？",
    ],
    (config.STATE_TIRED, "farewell"): [
        "那您早点休息，睡够了精神就好了。晚安。",
        "好，去歇着吧，别硬撑。我在这里守着。",
    ],
    (config.STATE_TIRED, "venting"): [
        "听着就累。您先别想那些事了，躺一会最要紧。",
        "累到心里去了吧。先放下，缓一缓再说。",
    ],
    (config.STATE_TIRED, "thanks"): [
        "别客气，您先歇着，这比什么都强。",
        "不用谢我。您把自己照顾好，我就放心了。",
    ],

    # ---- 低落 sad ----
    (config.STATE_SAD, "greeting"): [
        "您好。今天看着心情不太高，愿意跟我说说吗？",
        "来啦。您要是不想说也没事，我陪着您坐一会。",
    ],
    (config.STATE_SAD, "farewell"): [
        "好，您先歇着。心里要是堵得慌，随时喊我。",
        "那您早点睡，明天说不定就好了。我在的。",
    ],
    (config.STATE_SAD, "venting"): [
        "您说的我听着呢，{topic}这事确实让人心里不好受。",
        "心里憋着难受吧。说出来就好些了，我一直在听。",
        "我懂您的意思。这种感觉搁谁身上都不好受。",
    ],
    (config.STATE_SAD, "thanks"): [
        "不用谢。您愿意跟我说话，我就挺高兴的。",
        "别跟我客气。您心情好一点，我就踏实了。",
    ],
    (config.STATE_SAD, "chat"): [
        "嗯，我听着呢。您接着说。",
        "这事您别一个人扛着，说出来轻快点。",
    ],

    # ---- 走神/无人 absent ----
    (config.STATE_ABSENT, "greeting"): [
        "哎，我在呢。刚才没看见您，您去哪里了？",
        "您好呀，可算听见您说话了。",
    ],
    (config.STATE_ABSENT, "chat"): [
        "我在听，您说。",
        "您说话我就听得见，不用管我看没看见您。",
    ],

    # ---- 正常 normal ----
    (config.STATE_NORMAL, "greeting"): [
        "您好呀！今天感觉怎么样？",
        "来啦，看您气色不错。现在想聊点什么？",
        "您好，我在呢。今天过得顺心吗？",
    ],
    (config.STATE_NORMAL, "farewell"): [
        "好嘞，那您慢点，有空再聊。",
        "行，您去忙吧。记得按时吃饭。",
        "那咱们回头聊，您照顾好自己。",
    ],
    (config.STATE_NORMAL, "thanks"): [
        "不客气，这是我该做的。",
        "跟我还客气什么呀，您高兴就好。",
    ],
    (config.STATE_NORMAL, "praise"): [
        "您这么一说我都不好意思了。陪着您我也开心。",
        "谢谢您夸奖，我就想让您每天舒舒服服的。",
    ],
    (config.STATE_NORMAL, "venting"): [
        "嗯，{topic}这事听着就让人不痛快。您接着说。",
        "我明白。您有什么想法都可以跟我讲。",
    ],
    (config.STATE_NORMAL, "chat"): [
        "嗯嗯，{topic}——然后呢？",
        "这事挺有意思的，您多说说。",
        "我听着呢，您慢慢讲。",
    ],
}

# 状态通用兜底（没想到的 状态×意图 组合落到这里）
STATE_FALLBACK: Dict[str, List[str]] = {
    config.STATE_TIRED: [
        "您看着有点累了，先歇一会吧。",
        "别太勉强自己，身体要紧。",
    ],
    config.STATE_SAD: [
        "您要是心里不痛快，跟我说说也行。",
        "我在这里陪着您呢，别一个人闷着。",
    ],
    config.STATE_ABSENT: [
        "我在呢。您有什么事尽管说。",
        "您说话我就能听见。",
    ],
    config.STATE_NORMAL: [
        "嗯，我听着呢，您说。",
        "好呀，然后呢？",
    ],
}

# 意图通用兜底
INTENT_FALLBACK: Dict[str, List[str]] = {
    "identity": [
        "我是您的居家陪伴助手，能陪您聊天，也能看看您累不累、心情好不好。",
        "我叫小陪，是这个家里陪着您的那一个。您想聊什么都可以。",
    ],
    "time": [
        "具体时间我这边看得不太准，您看墙上那个钟更靠谱。",
        "这个我还真说不好，别耽误了您的事。",
    ],
    "question": [
        "这个问题我得想想。您先说说您是怎么想的？",
        "您问得挺好，不过这个我拿不太准，咱们一起琢磨琢磨。",
    ],
    "discomfort": [
        "身体不舒服可别硬扛，要不要我帮您叫家里人？",
        "那您先坐下歇歇，难受得厉害一定要说。",
    ],
}

DEFAULT_REPLIES: List[str] = [
    "嗯，我在听呢，您接着说。",
    "好，我记下了。",
    "您说的这个我听着呢。",
]

# --------------------------------------------------------------------------
# 两处**最高优先级**的回复，不按 (状态, 意图) 组织
#
# 提成模块级常量、而不是内联在 _candidates 里返回，是为了能被
# static_replies() 收进预合成缓存 —— 这两类话恰恰是最不能等 3 秒的：
# 老人说"不舒服"和说"不想活了"，都不该让机器先沉默一下再开口。
# --------------------------------------------------------------------------

# 身体不适
DISCOMFORT_REPLIES: List[str] = [
    "身体不舒服可别硬扛着，要不要我帮您叫家里人？",
    "您说的这个我得记一下。要是难受得厉害，咱们先歇一歇，别撑着。",
    "不舒服的时候先坐下缓缓，喝口温水。严重的话一定要告诉家人。",
]

# 危机信号（求救）。写法上和别处不同，是刻意为之：
#   - 不追问原因（追问会让对方更孤立），只接住情绪 + 把人往"身边有人"上引
#   - 不用"您别想不开"这类否定式劝阻
#   - **允许用问句**（"您现在身边有人吗"）：那是在确认安全，不是索取信息。
#     只有 PROACTIVE_TEMPLATES 受"不含问号"那条测试约束，这里不受。
CRISIS_REPLIES: List[str] = [
    "您说的这句我记在心上了，我很担心您。咱们给家里人打个电话好不好？",
    "我听见了，您别一个人扛着。您现在身边有人吗？我帮您把家里人叫过来。",
    "您对我很重要。这种时候我想让家里人陪在您身边，我这就帮您联系。",
]

# --------------------------------------------------------------------------
# 主动开口的文案（api_doc §5 V1.2 的 proactive 报文用它）
#
# 和上面的应答模板有三条不同的写作约束，都来自「老人没有向你提问」这个前提：
#
#   1. **不用问句。** 追问「您怎么了」「要不要跟我说说」是一种索取 ——
#      老人本来情绪就低，还要组织语言回答你。用陈述句，把选择权留给对方。
#      （api_doc §5.4.3 的 V1.2 修订同样是这个意思：不主动要求用户应答。）
#   2. **不评价、不说教。** 「您应该多出去走走」是责备。只描述观察 + 表达陪伴。
#   3. **给台阶。** 每句话都留一个「不理我也没关系」的余地。
#
# kind 与状态的对应关系由 core/proactive.py 决定，这里保持扁平。
# --------------------------------------------------------------------------

PROACTIVE_TEMPLATES: Dict[str, List[str]] = {
    # 视觉判 sad 持续一段时间
    "care_sad": [
        "我看您今天心情不太好。不用说什么，我就在这里陪着您。",
        "您要是心里闷得慌，我一直在旁边，想什么时候说都行。",
        "人总有提不起劲的时候。您先别为难自己，坐着歇一会。",
        "我瞧着您今天话少。没事，不想说话就不说，我陪着。",
    ],

    # 视觉判 tired 持续一段时间
    "care_tired": [
        "您看着有点乏了。先靠一会，别急着起身。",
        "累了就把手里的事放一放，喝口温水，歇歇眼睛。",
        "我看您眼睛都快睁不开了。要不先眯一会，我守着。",
        "这些事不急，您先缓缓。身体比什么都当紧。",
    ],

    # 从 absent 回到在场（离开过一段时间又回来了）
    "greeting": [
        "哎，您回来啦。",
        "您回来就好，我刚还惦记着。",
        "回来啦，我一直在呢。",
        "可算见着您了。坐着歇一会吧。",
    ],
}


def static_replies() -> List[str]:
    """所有**不需要填充占位符**的固定回复，去重后返回。

    用途只有一个：给在线 TTS 做预合成缓存。豆包/edge-tts 合成一句要 3 秒左右
    （SAPI 只要 37ms），而这里面全是启动时就已知的句子 —— 提前合成好，
    播放时直接读文件，就把「先沉默三秒再开口」抹掉了。

    带 ``{topic}`` 的模板**故意排除在外**：它要拼进用户刚说的话，无法预测，
    预合成了也是白占地方。排除之后剩下的恰好是主动关怀的全部文案
    和大部分兜底句 —— 也就是演示时最常听到的那几句。
    """
    collected: List[str] = []

    def add(text: str) -> None:
        # 含占位符的句子里 {topic} 未知，合成出来是死文本，直接跳过。
        if "{" in text:
            return
        stripped = text.strip()
        if stripped and stripped not in collected:
            collected.append(stripped)

    for templates in (REPLY_TEMPLATES, STATE_FALLBACK,
                      INTENT_FALLBACK, PROACTIVE_TEMPLATES):
        for entries in templates.values():
            for text in entries:
                add(text)
    for text in DEFAULT_REPLIES:
        add(text)
    # 这两组不按 (状态, 意图) 组织，所以上面的循环扫不到它们。
    # 早先它们是内联在 _candidates 里的字面量，于是**从来没被预合成过** ——
    # 表现为"老人说不舒服，机器先沉默三四秒"。危机那组更不能等。
    for text in DISCOMFORT_REPLIES:
        add(text)
    for text in CRISIS_REPLIES:
        add(text)

    return collected
