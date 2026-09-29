# -*- coding: utf-8 -*-
"""文本情绪识别（简易实现）。

任务要求"简易实现，分析用户对话文本的情绪"，因此这里用**情感词典 + 规则**
的方案，不引入任何模型、不下载任何权重 —— 零依赖、可离线、可解释、可随时改词表。

做法（经典的中文情感分析三件套）：
    1. 情感词打分    ：命中词典里的词，按词权重累加
    2. 程度副词放大  ："很累"比"累"更消极，"太"、"特别"、"非常" 按倍数放大
    3. 否定词翻转    ："不开心"要翻成正/负向的反面

再叠加两个小规则：感叹号/问号加重语气，以及一些整体性短语（"好累啊"这类词表
里已有的不用管，但"不想活了"这种要多给权重）。

输出：
    label  —— negative / neutral / positive 三分类，供对话管理使用
    emotion—— 更细的标签 tired / sad / angry / anxious / happy / neutral，
              用于挑选更贴切的回复模板
    score  —— 连续得分，负=消极 正=积极，方便前端画曲线或后续调阈值
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# 情感词典
# 键是词，值是 (细粒度情绪, 权重)。权重范围 0.5~3.0：
#   1.0 普通情绪词，2.0 强烈情绪词，3.0 极端/危险表达
# 词表故意做得短小，够用即可；后续要扩充直接往这里加，不动逻辑。
# --------------------------------------------------------------------------

EMOTION_LEXICON: Dict[str, Tuple[str, float]] = {
    # ---- 疲惫 tired ----
    "累": ("tired", 1.2), "好累": ("tired", 1.8), "太累": ("tired", 2.0),
    "疲惫": ("tired", 1.8), "疲劳": ("tired", 1.8), "困": ("tired", 1.2),
    "好困": ("tired", 1.8), "想睡": ("tired", 1.5), "没精神": ("tired", 1.6),
    "没力气": ("tired", 1.6), "乏力": ("tired", 1.5), "睁不开眼": ("tired", 1.8),
    "熬不住": ("tired", 1.8), "撑不住": ("tired", 1.8), "干不动": ("tired", 1.6),

    # ---- 悲伤 sad ----
    "难过": ("sad", 1.8), "伤心": ("sad", 2.0), "想哭": ("sad", 2.0),
    "哭": ("sad", 1.5), "低落": ("sad", 1.6), "郁闷": ("sad", 1.6),
    "失落": ("sad", 1.5), "孤独": ("sad", 1.8), "寂寞": ("sad", 1.6),
    "没人陪": ("sad", 1.6), "想家": ("sad", 1.5), "委屈": ("sad", 1.8),
    "没意思": ("sad", 1.2), "不开心": ("sad", 1.5), "心情不好": ("sad", 1.8),
    # 注意：词典是子串匹配的，"心情不好" 匹配不到 "心情不太好"（中间插了个"太"），
    # 所以这类常用的插入式变体必须单独列出来，不能指望一条覆盖全部说法。
    "心情不太好": ("sad", 1.8), "心情很差": ("sad", 2.0), "心情不好受": ("sad", 1.8),
    "心里难受": ("sad", 1.8), "心里堵": ("sad", 1.6), "想不开": ("sad", 2.2),
    "难受": ("sad", 1.5), "想念": ("sad", 1.2), "舍不得": ("sad", 1.2),

    # ---- 生气 angry ----
    "生气": ("angry", 1.8), "烦": ("angry", 1.4), "好烦": ("angry", 1.8),
    "讨厌": ("angry", 1.6), "气死": ("angry", 2.2), "恼火": ("angry", 1.8),
    "火大": ("angry", 1.8), "不爽": ("angry", 1.4), "讨厌死": ("angry", 2.0),
    "别烦我": ("angry", 2.0),

    # ---- 焦虑 anxious ----
    "担心": ("anxious", 1.6), "害怕": ("anxious", 1.8), "紧张": ("anxious", 1.6),
    "焦虑": ("anxious", 1.8), "发愁": ("anxious", 1.6), "不安": ("anxious", 1.5),
    "怎么办": ("anxious", 1.4), "睡不着": ("anxious", 1.8), "心慌": ("anxious", 1.8),

    # ---- 积极 happy ----
    "开心": ("happy", 1.8), "高兴": ("happy", 1.8), "快乐": ("happy", 1.8),
    "舒服": ("happy", 1.4), "不错": ("happy", 1.0), "好多了": ("happy", 1.6),
    "谢谢": ("happy", 1.2), "感谢": ("happy", 1.2), "喜欢": ("happy", 1.4),
    "太好了": ("happy", 1.8), "棒": ("happy", 1.2), "满足": ("happy", 1.4),
    "踏实": ("happy", 1.2), "放心": ("happy", 1.2), "还好": ("happy", 0.8),
    "挺好": ("happy", 1.4), "有意思": ("happy", 1.2), "精神好": ("happy", 1.6),
}

# 身体不适也算消极信号，单独一张表，命中时给出健康关怀类回复
PHYSICAL_DISCOMFORT = {
    "头疼": 1.8, "头晕": 1.8, "胃疼": 1.8, "肚子疼": 1.8, "腰疼": 1.6,
    "腿疼": 1.6, "不舒服": 1.6, "感冒": 1.5, "发烧": 2.0, "咳嗽": 1.2,
    "吃药": 1.0, "没胃口": 1.5, "吃不下": 1.5, "血压": 1.2,
}

# 程度副词：倍率
INTENSIFIERS: Dict[str, float] = {
    "非常": 1.6, "特别": 1.6, "十分": 1.5, "相当": 1.4, "很": 1.4,
    "太": 1.5, "好": 1.3, "真的": 1.3, "真是": 1.3, "超级": 1.8,
    "有点": 0.7, "稍微": 0.6, "一点点": 0.6, "还算": 0.8, "比较": 0.9,
}

# 否定词：命中后把后一个情感词的极性翻转
NEGATIONS = ("不", "没", "别", "无", "非", "未", "莫", "勿", "不要", "不太", "没有")

# 否定词往后看多少个字符去找情感词（中文里"不怎么开心"的间隔很短）
NEGATION_WINDOW = 4

# 危险信号：命中就置 TextEmotion.crisis 并直接给最强消极权重。
# 对话管理（core/dialogue.py）靠这个标记走**专门的危机关怀分支**，并且
# **不把这句话交给大模型** —— 见 dialogue.llm_eligible。
#
# 这里要穷举常见说法 —— 漏掉一种说法就可能把一个求救信号当成普通抱怨，
# 值得多写几个变体。匹配是在 _normalize() 之后做的，所以「想 死」「不 想 活」
# 这种插空格的写法也能命中；下游不要再拿原文自己匹配一遍（会漏）。
CRISIS_PATTERNS = (
    "不想活", "活不下去", "想死", "自杀", "想不开",
    "活着没意思", "没意思活着", "不如死了", "死了算了", "活着没劲",
    "没人在乎我", "走了算了",
)

# 标点权重
EXCLAMATION_BONUS = 0.4   # 每个 ！ 最多加 0.4
QUESTION_PENALTY = 0.2    # 连续 ？？ 往往伴随焦虑


@dataclass
class TextEmotion:
    """文本情绪识别结果。"""

    label: str = "neutral"        # negative / neutral / positive
    emotion: str = "neutral"      # tired / sad / angry / anxious / happy / neutral
    score: float = 0.0            # 连续得分，负=消极
    hits: List[str] = None        # 命中的情感词，便于联调时解释结果
    discomfort: bool = False      # 是否提到身体不适（头疼 / 不舒服 …）
    crisis: bool = False          # 是否命中 CRISIS_PATTERNS（求救信号，见下）

    def __post_init__(self) -> None:
        if self.hits is None:
            self.hits = []

    def is_negative(self) -> bool:
        return self.label == "negative"

    def is_positive(self) -> bool:
        return self.label == "positive"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "emotion": self.emotion,
            "score": round(self.score, 2),
            "discomfort": self.discomfort,
            "crisis": self.crisis,
            "hits": self.hits,
        }


# 判定 negative / positive 的分界。取 0.8 而不是 0，是为了让"还好""不错"
# 这类弱信号落在 neutral，避免把客套话当成真的开心。
POSITIVE_THRESHOLD = 0.8
NEGATIVE_THRESHOLD = -0.8


class TextEmotionAnalyzer:
    """基于词典的文本情绪分析器。

    无状态（词典都是模块级常量），可以多线程共用同一个实例。
    """

    def __init__(self) -> None:
        # 预编译：把词典按词长倒序排列，保证"好累"先于"累"被匹配到
        self._keys_by_length = sorted(EMOTION_LEXICON.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> TextEmotion:
        """分析一句用户文本的情绪。空文本返回 neutral。"""
        if not text or not text.strip():
            return TextEmotion()

        normalized = self._normalize(text)
        score = 0.0
        hits: List[str] = []
        # 记录每种细粒度情绪累计到的"强度"，最后取最大的那个当主情绪
        emotion_weights: Dict[str, float] = {}
        consumed = [False] * len(normalized)

        # ---- 1) 主情感词典 ----
        # 按词长倒序匹配，保证"好累"先于"累"命中，且同一段文字不重复计分
        for word in self._keys_by_length:
            for start in self._find_all(normalized, word, consumed):
                end = start + len(word)
                for i in range(start, end):
                    consumed[i] = True

                emotion, weight = EMOTION_LEXICON[word]
                magnitude = weight * self._intensifier_factor(normalized, start)
                polarity = 1.0 if emotion == "happy" else -1.0

                if self._is_negated(normalized, start):
                    polarity = -polarity      # "不开心" → 由正转负
                    magnitude *= 0.8          # "不难过" 的正面程度不如直接说"开心"

                score += polarity * magnitude
                hits.append(word)
                bucket = emotion if polarity < 0 else "happy"
                emotion_weights[bucket] = emotion_weights.get(bucket, 0.0) + magnitude

        # ---- 2) 身体不适：算消极，触发健康关怀而非心理安慰 ----
        for word, weight in PHYSICAL_DISCOMFORT.items():
            for start in self._find_all(normalized, word, consumed):
                for i in range(start, start + len(word)):
                    consumed[i] = True
                magnitude = weight * self._intensifier_factor(normalized, start)
                if self._is_negated(normalized, start):
                    continue  # "肚子不疼了" 是好事，不作为主情绪
                score -= magnitude
                hits.append(word)
                emotion_weights["sad"] = emotion_weights.get("sad", 0.0) + magnitude
                # discomfort 单独标记，对话管理据此走"身体关怀"分支
                emotion_weights["discomfort"] = emotion_weights.get("discomfort", 0.0) + magnitude

        # ---- 3) 危险信号：一票否决，直接拉到最强消极 ----
        # crisis 不只是"分很低"，它是一个**独立信号**：对话管理据此走专门的关怀
        # 分支，并且**绝不允许交给大模型自由发挥**（见 core/dialogue.py 的
        # llm_eligible）。所以这里必须留下显式标记，不能让下游靠 score 去猜 ——
        # 别的极端消极句子也能凑到 -5 分，但那些不需要走危机处理。
        crisis = False
        for pattern in CRISIS_PATTERNS:
            if pattern in normalized:
                score -= 5.0
                hits.append(pattern)
                emotion_weights["sad"] = emotion_weights.get("sad", 0.0) + 5.0
                crisis = True
                break

        # ---- 4) 标点微调 ----
        score += self._punctuation_delta(normalized)

        label = self._to_label(score)
        emotion = self._dominant_emotion(emotion_weights, label)

        return TextEmotion(
            label=label,
            emotion=emotion,
            score=score,
            hits=hits[:12],
            discomfort="discomfort" in emotion_weights,
            crisis=crisis,
        )

    def label_of(self, text: str) -> str:
        """只取三分类标签的便捷方法。"""
        return self.analyze(text).label

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化：去空白、全角转半角、英文小写。

        中文之间插入空格会破坏词语匹配（"好 累"），所以把空白直接删掉。
        """
        text = text.strip().lower()
        text = re.sub(r"\s+", "", text)
        # 全角字母数字 → 半角（全角标点保留，后面标点规则要用）
        result = []
        for ch in text:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E and ch not in "！？，。；：、（）《》""''":
                result.append(chr(code - 0xFEE0))
            else:
                result.append(ch)
        return "".join(result)

    @staticmethod
    def _find_all(text: str, word: str, consumed: List[bool]) -> List[int]:
        """找出 word 在 text 中所有未被占用的出现位置。

        用 consumed 标记避免重复计分：一旦匹配到"好累"，就把这几个字标记掉，
        否则后面扫到"累"会再算一次。
        """
        positions: List[int] = []
        start = text.find(word)
        while start != -1:
            end = start + len(word)
            if not any(consumed[start:end]):
                positions.append(start)
            start = text.find(word, start + 1)
        return positions

    @staticmethod
    def _intensifier_factor(text: str, word_start: int) -> float:
        """程度副词倍率：往前 3 个字符内找副词，找到就乘倍率。"""
        prefix = text[max(0, word_start - 3):word_start]
        # 长词优先，"十分"要优先于"十"这类前缀重叠
        for adverb, factor in sorted(INTENSIFIERS.items(), key=lambda kv: -len(kv[0])):
            if prefix.endswith(adverb):
                return factor
        return 1.0

    @staticmethod
    def _is_negated(text: str, word_start: int) -> bool:
        """判断情感词前 NEGATION_WINDOW 个字符内是否出现否定词。"""
        prefix = text[max(0, word_start - NEGATION_WINDOW):word_start]
        if not prefix:
            return False
        # 长否定词优先（"不要" 要优先于 "不"，否则"不要开心"会被判成否定+开心=消极）
        for negation in sorted(NEGATIONS, key=len, reverse=True):
            if prefix.endswith(negation):
                return True
        return False

    @staticmethod
    def _punctuation_delta(text: str) -> float:
        """标点对情绪的微调。"""
        delta = 0.0
        if "！" in text or "!" in text:
            count = text.count("！") + text.count("!")
            # 感叹号只放大"已有情绪"，没有情感词时不该凭空造出情绪，
            # 所以这里只是很小的增量，不足以单独跨过阈值。
            delta += min(count, 3) * EXCLAMATION_BONUS
        if "？？" in text or "??" in text:
            delta -= QUESTION_PENALTY
        if "。。。" in text or "..." in text:
            delta -= QUESTION_PENALTY
        return delta

    @staticmethod
    def _to_label(score: float) -> str:
        if score >= POSITIVE_THRESHOLD:
            return "positive"
        if score <= NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    @staticmethod
    def _dominant_emotion(emotion_weights: Dict[str, float], label: str) -> str:
        """取权重最大的细粒度情绪。

        正向文本里如果混进了消极词（"累是累，但是挺开心的"），
        以整体 label 为准，避免选出和整体情绪相反的标签。
        """
        if not emotion_weights:
            return "happy" if label == "positive" else "neutral"
        if label == "positive" and "happy" in emotion_weights:
            return "happy"
        if label == "neutral":
            return "neutral"
        # discomfort 只是辅助标记，主情绪仍取 sad
        filtered = {k: v for k, v in emotion_weights.items() if k not in ("discomfort", "happy")}
        if not filtered:
            return "sad"
        return max(filtered.items(), key=lambda kv: kv[1])[0]
