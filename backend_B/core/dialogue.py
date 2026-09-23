# -*- coding: utf-8 -*-
"""对话管理：根据「视觉状态 + 用户文本情绪 + 对话内容」生成回复。

任务要求 3。这里刻意不接大模型 —— 原型阶段要的是**可预测、可调试、零依赖**。
规则引擎的好处是：每条回复为什么被选中，都能指着代码说清楚；联调时
只要看日志里的 (intent, state, emotion) 就知道该去改哪条模板。

一次 respond() 做四件事：
    1. 识别意图        recognize_intent()
    2. 融合状态        fuse_state()  —— 视觉状态 与 文本情绪 合并成 4 态之一
    3. 挑回复模板      按 (状态, 意图) 选一组候选，避开最近说过的
    4. 落库/给上下文    交给外部（main.py）写历史，这里只返回结果

融合状态会同时用于两处：
    - 决定回复的语气和内容
    - 作为 B→C 推送的状态（api_doc §4.2 只允许 normal/sad/tired/absent）
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import config
from core.text_emotion import TextEmotion, TextEmotionAnalyzer
from core.vision_state import VisionState

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


# --------------------------------------------------------------------------
# 意图识别词表
# 顺序有意义：越靠前的越优先匹配（"你好，谢谢" 应识别为 greeting）
# --------------------------------------------------------------------------

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
    """规则式对话管理器。

    线程安全说明：会修改的只有 self._last_user_text_at，单次赋值是原子的。
    真正的共享状态（历史 CSV）由外部 store 负责加锁。
    """

    def __init__(self, history=None, rng: Optional[random.Random] = None) -> None:
        self.analyzer = TextEmotionAnalyzer()
        self.history = history                  # CsvHistoryStore 或 None
        self._rng = rng or random.Random()
        self._last_user_text_at: float = 0.0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def respond(self, user_text: str, vision: Optional[VisionState] = None) -> DialogueReply:
        """生成回复。user_text 为空时返回一句兜底提示，不抛异常。"""
        user_text = (user_text or "").strip()
        vision = vision or VisionState()

        if not user_text:
            return DialogueReply(
                reply="我在这儿呢，您慢慢说。",
                state=vision.state,
                reason="空输入兜底",
            )

        self._last_user_text_at = time.monotonic()

        emotion = self.analyzer.analyze(user_text)
        intent = self.recognize_intent(user_text, emotion)
        state = self.fuse_state(vision, emotion)

        reply_text = self._compose_reply(state, intent, emotion, user_text, vision)

        return DialogueReply(
            reply=reply_text,
            state=state,
            intent=intent,
            emotion_label=emotion.label,
            emotion_score=emotion.score,
            emotion_detail=emotion.emotion,
            discomfort=emotion.discomfort,
            reason=f"视觉={vision.state}({vision.reason}) 文本={emotion.label}/{emotion.emotion} 意图={intent}",
        )

    # ------------------------------------------------------------------
    # 1) 意图识别
    # ------------------------------------------------------------------

    def recognize_intent(self, text: str, emotion: TextEmotion) -> str:
        """关键词匹配。命中多条时取 INTENT_RULES 里靠前的那个。"""
        for intent, keywords in INTENT_RULES:
            for keyword in keywords:
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
        # 身体不适优先级最高 —— 健康问题不能只回一句"多喝热水"式的套话
        if emotion.discomfort or intent == "discomfort":
            return [
                "身体不舒服可别硬扛着，要不要我帮您叫家里人？",
                "您说的这个我得记一下。要是难受得厉害，咱们先歇一歇，别撑着。",
                "不舒服的时候先坐下缓缓，喝口温水。严重的话一定要告诉家人。",
            ]

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
        "您好呀。看您好像有点乏，先坐下歇会儿吧。",
        "来啦。您今天精神看着不太足，要不要先闭眼养养神？",
    ],
    (config.STATE_TIRED, "farewell"): [
        "那您早点休息，睡够了精神就好了。晚安。",
        "好，去歇着吧，别硬撑。我在这儿守着。",
    ],
    (config.STATE_TIRED, "venting"): [
        "听着就累。您先别想那些事了，躺一会儿最要紧。",
        "累到心里去了吧。先放下，缓一缓再说。",
    ],
    (config.STATE_TIRED, "thanks"): [
        "别客气，您先歇着，这比什么都强。",
        "不用谢我。您把自己照顾好，我就放心了。",
    ],

    # ---- 低落 sad ----
    (config.STATE_SAD, "greeting"): [
        "您好。今天看着心情不太高，愿意跟我说说吗？",
        "来啦。您要是不想说也没事，我陪着您坐会儿。",
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
        "哎，我在呢。刚才没看见您，您去哪儿了？",
        "您好呀，可算听见您说话了。",
    ],
    (config.STATE_ABSENT, "chat"): [
        "我在听，您说。",
        "您说话我就听得见，不用管我看没看见您。",
    ],

    # ---- 正常 normal ----
    (config.STATE_NORMAL, "greeting"): [
        "您好呀！今天感觉怎么样？",
        "来啦，看您气色不错。这会儿想聊点什么？",
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
        "嗯，{topic}这事儿听着就让人不痛快。您接着说。",
        "我明白。您有什么想法都可以跟我讲。",
    ],
    (config.STATE_NORMAL, "chat"): [
        "嗯嗯，{topic}——然后呢？",
        "这事儿挺有意思的，您多说说。",
        "我听着呢，您慢慢讲。",
    ],
}

# 状态通用兜底（没想到的 状态×意图 组合落到这里）
STATE_FALLBACK: Dict[str, List[str]] = {
    config.STATE_TIRED: [
        "您看着有点累了，先歇会儿吧。",
        "别太勉强自己，身体要紧。",
    ],
    config.STATE_SAD: [
        "您要是心里不痛快，跟我说说也行。",
        "我在这儿陪着您呢，别一个人闷着。",
    ],
    config.STATE_ABSENT: [
        "我在呢。您有什么事儿尽管说。",
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
        "这个我还真说不好，别耽误了您的事儿。",
    ],
    "question": [
        "这个问题我得想想。您先说说您是怎么想的？",
        "您问得挺好，不过这个我拿不太准，咱们一块儿琢磨琢磨。",
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
