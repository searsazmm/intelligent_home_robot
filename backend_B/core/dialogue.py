# -*- coding: utf-8 -*-
"""对话管理：根据「视觉状态 + 用户文本情绪 + 对话内容」生成回复。

任务要求 3。**默认不接大模型**；配了 DeepSeek 凭证才接，而且规则模板永远是兜底。
两层是叠加关系，不是替换：

    规则模板  ——  可预测、可调试、零第三方依赖、**永远有话说**
    大模型    ——  接住没预设过的话（模板只有 8 个意图 + 3 句通用兜底，
                  这才是"回答生硬、接不住话"的根因）

规则引擎的好处没有丢：每条回复为什么被选中，仍然能指着代码说清楚 ——
联调时看日志里的 (intent, state, emotion, 来源) 就知道该去改哪条模板。

一次 respond() 做的事，是**三步**：

    第一步  先答内容   回答老人**说的那件事**，就像朋友接话
    第二步  判断状态   结合人脸数据（vision.state）看要不要关心
    第三步  追加关怀   要关心才追一句；状态正常、或他自己已经说了，就不打扰

三步的顺序是硬约束。第③步是**追加**，不是**替换** —— 早先的实现让
``STATE_FALLBACK[state]`` 顶替掉内容回答（老人说「今天天气不错」、
脸上有点乏，答的是「您看着有点累了」），那等于把第一步和第三步并成了
一步。见 :data:`CONTENT_INTENTS` 和 :meth:`DialogueEngine._candidates`。

具体到一次 respond()：
    1. 识别意图        recognize_intent()
    2. 融合状态        fuse_state()  —— 视觉状态 与 文本情绪 合并成 4 态之一
    3. 生成回复        大模型（若可用且适用）否则挑模板，避开最近说过的
    4. 追加关怀        见 :meth:`DialogueEngine._care_kind`
    5. 落库/给上下文    交给外部（main.py）写历史，这里只返回结果

⚠️ 第 3 步里，**大模型只负责产出「回复文本」这一件事**。
   state / intent / emotion_* / discomfort 全部保持规则判定，一行都不动 ——
   它们驱动 8001 状态机、CSV 的列、以及前端 C 的显示，不能让模型改写。

融合状态（``fuse_state``）用于两处：
    - 作为 B→C 推送的状态（api_doc §4.2 只允许 normal/sad/tired/absent）
    - 决定回复的语气和内容

⚠️ 但**追加关怀不用融合状态，只用 ``vision.state``**（摄像头那一份）。
   理由见 :meth:`DialogueEngine._care_kind`：关怀语说的是"我看见了
   您没说的那件事"，只能来自摄像头。文字里说「我有点累」而摄像头什么都没
   看到时，回答本身就接住了，再追一句「我看您眼睛都快睁不开了」就是瞎编。

融合状态还会写进系统提示词，让模型的话对得上状态 —— **但正在追加关怀
的那一轮除外**：那一轮的状态被隐去，模型只答内容，关怀完全由固定文案
负责（否则模型和文案会把同一件事说两遍）。见 ``respond()`` 里的
``speech_state``。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import config
from core import proactive as proactive_lib
from core.emergency import EMERGENCY_PATTERNS
from core.llm import build_system_prompt, sanitize_reply
from core.text_emotion import TextEmotion, TextEmotionAnalyzer
from core.vitals import VITALS_NOTE
from core.vision_state import VisionState
from core.weather import describe_activity, describe_weather

logger = logging.getLogger(__name__)

# 用户发了消息后，多久之内认为"人在"，可以推翻视觉的 absent 判定（秒）
# 场景：用户低头打字/摄像头被挡，视觉判 absent，但人明明在讲话。
TEXT_PRESENCE_WINDOW = 30.0

# 回答内容与用户状态无关的意图：不管用户是累还是低落，问"你是谁"都得答身份。
# 这些意图的模板优先级要高于"状态通用兜底"。
STATE_INDEPENDENT_INTENTS = frozenset(
    {"identity", "time", "weather", "reminder", "festival", "vitals",
     "emergency", "quiet", "clarify", "question", "story"})

# 这些意图的回答**必须**是"接住老人说的那件事"，状态只决定要不要追加关怀，
# 不能顶替内容。
#
# 只有 chat 一个意图在这里，理由是它是唯一一个"用户跟我讲了一件事，而回答里
# 没有任何对状态敏感的社交成分"的意图。其余意图（问候 / 道别 / 道谢 / 夸奖 /
# 倾诉）本来就是对状态敏感的社交应答，状态参与选词是对的 —— 累的时候说
# 「那您早点休息」比说「有空再聊」好。
#
# 效果：老人说「今天天气不错」，不管他脸上乏不乏，答的都是 (normal, chat)
# 那三句带 {topic} 回指的接话（「嗯嗯，{topic}——然后呢？」）；乏不乏只决定
# 后面要不要再追一句关怀。
CONTENT_INTENTS = frozenset({"chat"})

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
#   weather/reminder/festival
#                     这三类需要**实时信息或可靠知识**，模型都不具备 ——
#                     问"今天天气怎么样"它编一个晴天，求"提醒我吃药"它应
#                     "好嘞我记下了"，问"重阳节是多久"它按训练数据里的
#                     旧年份报个错日子。假信息比不回答危险得多（老人按编的
#                     天气穿衣、等不会来的提醒、照错的日期盼儿女回家），
#                     所以这三类永远走确定性回答：INTENT_FALLBACK 诚实模板
#                     和 FESTIVALS 日期表。
#   vitals            心率呼吸的数值在 VitalsStore 里，是传感器现读数，
#                     模型看不见 —— 让它答"我心率多少"只会编一个数。
#                     vitals_reply 按读数现算（含偏高偏低与建议）；
#                     chat/question 里顺带提到体征时，数值经 _vitals_hint
#                     注进提示词，而不是靠模型自己"记得"。
#   emergency/quiet   紧急求助必须是确定、短促、像行动的回复（且不能把
#                     "救命"发给第三方）；"别说话"的回答也只能是"好"。
#                     两者都是逐句写死的，没有自由发挥空间。
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

#: 关键词被子串"借走"时的反例表：命中某个关键词、但整句还包含反例时，
#: 这个关键词**不**作数，继续往后面的意图匹配。
#:
#: 和 STANDALONE_KEYWORDS 是同一类问题的另一种形态 —— 高频字词做子串
#: 匹配时撞上语义相反的句子：
#:   * 「我心疼孙子」命中 discomfort 的"疼" → 要回"要不要叫家里人"，
#:     可人家是在说疼爱孙子；
#:   * 「你好久没来」命中 greeting 的"你好" → 要回热情招呼，可人家
#:     是在嗔怪。
#: 反例只否决**它挂着的那个关键词**：「头疼，儿子还不心疼我」里
#: "头疼"照样命中 discomfort，不会被"心疼"误伤。
KEYWORD_BLOCKERS: Dict[str, tuple] = {
    "疼": ("心疼", "疼爱"),
    "你好": ("你好久",),
    # 「救命稻草这个词怎么来的」是在问成语，不是求救 —— 反例借走"救命"，
    # 让这句落回闲聊，而不是给家人发一条紧急通知。
    "救命": ("救命稻草",),
}

INTENT_RULES: List[tuple] = [
    # 紧急求助优先级最高，排在一切意图（包括危机情绪兜底）的关键词匹配之前。
    # 词表刻意只收**完整说法**（"救命""叫救护车""喘不上气"），不收裸
    # "报警""不行了" —— 误报一次家人通知，这个功能就再也没人信。
    # 自我伤害类（"不想活了"）走 text_emotion.crisis，不在这里。
    ("emergency", EMERGENCY_PATTERNS),
    # 「听不清 / 大声点 / 再说一遍」：老人没听清机器人说的话。
    # 排在 quiet 之后、其他意图之前 —— 提高音量、放慢语速、复述确认
    # 是体验问题不是安全问题，但优先级要高于闲聊，否则一句"听不清"
    # 被当成 question 走大模型就答非所问了。
    ("clarify", (
        "听不清", "听不见", "没听清", "没听见", "听不到",
        "大声点", "大点声", "说大声", "声音小", "声音太小", "太小声",
        "再说一遍", "再说一次", "重复一遍", "再说一遍", "重说",
        "慢一点", "说慢点", "说太快", "太快了", "慢慢说",
    )),
    # 「不想听你说话 / 别说话 / 闭嘴」：必须排在问候之前且精确命中，
    # 回答只能是"好，我不说话了"，并由 main 联动关掉主动开口。
    ("quiet", (
        "不想听你说话", "不想听你说", "不想听见你", "不想听到你",
        "别说话", "不要说话", "你别说话", "你不要说话",
        "别出声", "别吭声", "闭嘴", "你闭嘴",
        "安静点", "安静一下", "安静一会", "别吵了", "别吵我",
        "我想安静", "想一个人待", "不想搭理",
    )),
    # farewell 的关键词必须是**精确的告别说法**（见 KEYWORD_BLOCKERS 的教训）：
    # 早先这里只有"出门""走了""我睡"三个短词，做子串匹配时
    # 「我今天出门散步了」被当成告别、回一句"行，您去忙吧"把人打发了 ——
    # 分享日常恰恰是陪伴机器人最该接住的话。「老伴走了」更糟：那是悼念。
    # 所以这里全部收窄成主语+动作的完整说法。
    ("farewell", (
        "再见", "拜拜", "我走了", "我得走了", "先走了", "这就走",
        "我睡了", "要睡了", "睡觉了", "不聊了", "下次聊", "先这样",
        "回头聊", "晚安", "挂了", "出门了", "要出门",
    )),
    ("greeting", (
        "你好", "您好", "早上好", "中午好", "下午好", "晚上好", "在吗", "在不在",
        "嗨", "哈喽", "hello", "hi", "早",
    )),
    ("thanks", ("谢谢", "感谢", "多谢", "辛苦了", "麻烦你")),
    ("identity", (
        "你是谁", "你叫什么", "你的名字", "你是什么", "介绍一下你", "你会什么", "你能做什么",
    )),
    # 心率 / 呼吸问句走**确定性**回答（vitals_reply 现算）：数值就在
    # VitalsStore 里，模型看不见它，问"我心率多少"让它答只会编一个数。
    # ⚠️ 关键词全部是"指标 + 问法"的组合，没有裸"心率""心跳" ——
    # 「我心慌」「我心跳得厉害」是身体不适陈述，要留给 discomfort
    # （回关怀），报数答非所问。
    ("vitals", (
        "心率多少", "心率是多少", "心率怎么样", "心率正常", "心率快不快",
        "心率慢不慢", "我的心率", "我心率",
        "心跳多少", "心跳是多少", "心跳怎么样", "心跳正常", "心跳快不快",
        "我的心跳", "我心跳正常", "我的脉搏", "脉搏多少", "脉搏正常",
        "呼吸多少", "呼吸是多少", "呼吸怎么样", "呼吸正常", "呼吸快不快",
        "我的呼吸", "我呼吸",
    )),
    # 没有"什么时候" —— 「我什么时候能见到孙子」是盼头不是问钟点，
    # 落到 question（可走大模型）才答得上；问时间有"几点"就够了。
    # ⚠️ festival 必须排在 time **之前**："过年是几号"带"几号"，
    # 排后面会被 time 抢走、回一句"看墙上那个钟"。
    ("festival", (
        "节日", "过节", "什么节", "春节", "过年", "元旦", "元宵", "清明",
        "五一", "劳动节", "儿童节", "端午", "中秋", "国庆", "重阳", "除夕", "七夕",
    )),
    ("time", ("几点", "现在时间", "几号", "今天星期", "今天几",
              "星期几", "周几", "礼拜几", "多少号", "日期", "成都时间")),
    # 天气问句走**确定性**回答：联网查 open-meteo（core/weather.py），
    # 查到就实时报温度阴晴，查不到就老实说"看不到"（INTENT_FALLBACK）。
    # 模型没有实时数据，让它答"今天天气怎么样"只会编一个晴天出来。
    # 注意关键词全是**问句式**说法 —— 「今天天气不错」是评价，仍是闲聊，
    # 别把老人随口一句感慨也拦成天气播报。
    ("weather", (
        "天气怎么样", "天气如何", "天气好不好", "天气预报", "什么天气",
        "啥天气", "多少度", "几度", "下雨", "下雪",
        "降温", "刮风", "冷不冷", "热不热", "适合出门",
    )),
    # 讲故事：怀旧小故事从**预写池**里挑，不走大模型。两个理由：
    # ①内容可控 —— 模型现编的长故事未必贴合老人的年代记忆，错一个年代
    #   细节就出戏；②长度可控 —— 长文本流式合成中途断流的风险随字数涨。
    # 词表收全说法、不收裸"故事" —— 「你别讲那些故事」是拒绝不是点播。
    ("story", (
        "讲故事", "讲个故事", "讲一个故事", "说个故事", "来个故事",
        "听故事", "想听故事", "故事吧", "说书",
    )),
    # 提醒/叫醒同样走**诚实**回答：机器人没有闹钟和日历，应一句
    # "好嘞我记下了"是假承诺 —— 老人到点等不到提醒，误的是吃药这种正事。
    ("reminder", ("提醒我", "叫我起床", "叫醒我", "到点叫我", "记得叫我")),
    ("discomfort", (
        "不舒服", "头疼", "头晕", "胃疼", "肚子疼", "腰疼", "腿疼", "感冒", "发烧",
        "咳嗽", "没胃口", "吃不下", "血压", "心慌", "疼",
    )),
    ("praise", ("真好", "真棒", "你真", "喜欢你", "谢谢你陪", "有你真好")),
    ("question", ("怎么", "为什么", "是什么", "能不能", "可以吗", "有没有", "多少", "吗？", "呢？")),
]

#: 命中 weather 意图、又带这些词的问句是**活动类**（"适合出门散步吗"），
#: 走 describe_activity 简短建议而不是天气播报全文 —— 否则问天气和问散步
#: 播的是一字不差的同一段全文（同一时段天气数据相同，播报是确定性的）。
WEATHER_ACTIVITY_HINTS = ("散步", "遛弯", "锻炼", "运动", "走一走", "出门玩", "爬山")


@dataclass
class DialogueReply:
    """一次对话的完整结果。"""

    reply: str                                  # 先答的内容 —— 回答老人说的那件事
    state: str                                  # 融合后的四态之一，用于推送 C
    intent: str = "chat"                        # 识别到的意图
    emotion_label: str = "neutral"              # 文本情绪三分类
    emotion_score: float = 0.0
    emotion_detail: str = "neutral"             # 细粒度情绪
    discomfort: bool = False                    # 是否提到身体不适
    reason: str = ""                            # 为什么这么回（调试用）
    #: 是否命中紧急状况（身体紧急关键词 或 危机情绪）。main.py 据此触发
    #: **真实动作**：家人 webhook 通知 + C 端紧急弹窗，而不只是嘴上说。
    emergency: bool = False
    #: "emergency"（救命/救护车）或 "crisis"（自我伤害语句）；空串=非紧急。
    emergency_kind: str = ""
    #: 答完之后**追加**的关怀语；空串 = 这一轮不追加。
    #:
    #: 和 reply 是**两句话**，不是一句话的两半：朗读时要分两次合成
    #: （见 Speaker.say_all）。拼成一个字符串会让预合成缓存必然不命中，
    #: 整句重新合成，白等一秒多。文案来自 PROACTIVE_TEMPLATES，全部预合成过。
    follow_up: str = ""
    #: 追加的那句属于哪一类（``care_sad`` / ``care_tired``）；空串 = 没追加。
    #:
    #: 独立于 ``intent`` 存在，因为写历史 CSV 时它是更准确的那一列：
    #: ``_fire_proactive`` 给主动开口写的 intent 就是 kind，应答里的追加
    #: 理应一致。只写 ``intent``（= chat）的话，CSV 里分不出哪几行是关怀。
    care_kind: str = ""

    def spoken_parts(self) -> Tuple[str, ...]:
        """要**念出来**的全部文本，按顺序。空串自动略过。

        朗读点和写日志都用它，这样"说了什么"只有这一个定义。
        注意它不等于 :meth:`to_dict` 的 text —— 那个是给 8002 的报文，
        两者恰好一致（都是整段），但用途不同，改的时候要一起改。
        """
        return tuple(part for part in (self.reply, self.follow_up) if part)

    def to_dict(self) -> dict:
        """转为发给 C 端 8002 的 JSON 结构。

        ``text`` 是**整段**要念的话（内容 + 追加的关怀），因为 C 端的气泡
        （frontend_C/main.py 的 on_reply）显示的就是这一句，它必须和老人
        实际听到的对得上。``follow_up`` 和 ``care_kind`` 再单独给一份，
        让排错工具能分清哪半句是关怀 —— 注意 8002 这条通道**本身不发声**，
        "对得上"说的是 C 的界面显示。

        ⚠️ 这个报文 B 只回给**发起对话的那个连接**（ui_channel 用的是
        ``_send(client, ...)``，不是广播）。所以被动监听的
        tools/watch_8002.py 收不到它 —— 要看这两行得用
        tools/mock_c_client.py，那才是发起方。
        """
        return {
            "type": "reply",
            "text": "".join(self.spoken_parts()),
            "follow_up": self.follow_up,
            "care_kind": self.care_kind,
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
                 llm=None, care_interval: Optional[float] = None,
                 vitals_provider=None, weather_provider=None) -> None:
        """``llm`` 是大模型客户端（见 :mod:`core.llm`），**默认 None**。

        ⚠️ 默认值必须是 None，不能是"从 config 建一个"：
        那样的话每个 ``DialogueEngine()`` 都会去打网络 —— 包括测试里
        那几十处构造。任何 ``available`` 为假的客户端都等价于"没有大模型"，
        所以传 NullChatClient 进来也是安全的。

        ``care_interval`` 是**追加关怀**的最小间隔（秒），默认取
        ``config.CARE_FOLLOWUP_INTERVAL``。做成构造参数而不是直接读常量，
        是因为它必须能被 ``main.py`` 按 ``--demo`` / ``--offline --speed``
        改写 —— 和 :class:`core.proactive.ProactivePolicy` 接收阈值的方式
        一致。固定 180 秒在演示里等于"看不出来"，在 5 倍速回放下等于
        每 900「数据秒」一次。测试也靠它注入一个很小的值。

        ``vitals_provider`` 是取**最新体征读数**的可调用（返回
        ``RppgReading`` 或 None），供 vitals 意图现答与提示词注入用。
        和 ``ChatServer`` 的 ``vitals_provider`` 同一个做法：**惰性
        provider 而不是传 store 对象** —— main.py 里 DialogueEngine
        建在 vitals_store 之前，lambda 在被调用时才解引用，此时
        ``__init__`` 早已跑完。
        """
        self.analyzer = TextEmotionAnalyzer()
        self.history = history                  # CsvHistoryStore 或 None
        self._rng = rng or random.Random()
        self._last_user_text_at: float = 0.0
        self._llm = llm                         # Chatter 或 None
        self._vitals_provider = vitals_provider  # Callable[[], RppgReading|None] 或 None
        #: 天气查询：返回 core.weather.parse_report 形态的 dict，或 None
        #: （查不到 / 未注入）。None 时 weather 意图走诚实兜底模板，
        #: **绝不在测试构造里联网** —— 和 vitals_provider 同一个理由。
        self._weather_provider = weather_provider
        #: 上一次说过的应声词，用来避开连着两次说同一句。
        self._last_ack: str = ""
        #: 上一次讲过的故事本体（STORY_POOL 里的原文）。story_reply 拿它
        #: 排除同一条 —— 连着点播两次不重样。
        self._last_story: str = ""
        #: 追加关怀的最小间隔（秒）与**上次真的追加**的时刻。
        #: 用 time.monotonic()，和 _last_user_text_at 同一个时钟。
        self._care_interval: float = (
            config.CARE_FOLLOWUP_INTERVAL if care_interval is None else care_interval)
        #: 每类关怀最近一次真正追加的时刻（kind → 单调时钟秒）。
        #: 冷却按**类型**分开记账：care_tired 刚说过，care_sad 不受影响。
        self._last_care_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def respond(
        self,
        user_text: str,
        vision: Optional[VisionState] = None,
        *,
        from_user: bool = True,
        on_thinking: Optional[Callable[[str], None]] = None,
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

        ``on_thinking`` 是**应声词**的出口：**只在真的要调大模型、并且调用
        之前**被调一次，参数是 :meth:`thinking_reply` 挑好的一句。它必须是
        **非阻塞**的 —— 调用它的线程马上要去做那次网络请求，在这里阻塞
        等于把应声词变成了对回复的额外延迟，正好是它想解决的问题。
        见 :data:`ACK_REPLIES`。
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

        # ⚠️ 这一步必须在生成回复**之前**：它决定这一轮要不要对"内容部分"
        # 隐去视觉状态（见下）。它只需要 vision 和 emotion，不需要等回复
        # 生成出来，所以拿得到。
        care_kind = self._care_kind(vision, emotion)

        # 紧急 / 危机 / 用户要求安静这三轮**不追加任何关怀语**：
        # 求救之后接一句"您看着有点累"是错乱；"不想听你说话"之后接
        # 主动关怀更是直接抗命。
        if intent == "emergency" or intent == "quiet" or emotion.crisis:
            care_kind = None

        # 紧急标志：身体紧急关键词 或 危机情绪。main.py 拿到它才好触发
        # 真实通知动作（webhook + C 弹窗），而不是让"我帮您叫家里人"
        # 沦为一句空话。
        if intent == "emergency":
            emergency_kind = "emergency"
        elif emotion.crisis:
            emergency_kind = "crisis"
        else:
            emergency_kind = ""

        # 要追加关怀的这一轮，**内容部分一律按"状态正常"来生成**。
        #
        # 理由：追加的那句关怀说什么，是本地 PROACTIVE_TEMPLATES 定的。
        # 生成内容的一方（大模型或状态模板）要是也知道状态，它八成会自己
        # 先关怀一遍，老人于是听到同一件事说两遍 ——「您看着有点累」
        # 接「我看您眼皮都沉了」。
        #
        # 只换这**一个**变量，意图和情绪照旧（情绪是用户自己说出口的，
        # 答不答都得接住）。也正因为只有一个变量，模板兜底这条路同样受
        # 约束 —— 否则大模型一失败，模板立刻又把状态关怀加回来。
        #
        # 注意 DialogueReply.state 里放的仍是**真实**状态，被隐去的只是
        # 生成内容时看到的那份。
        speech_state = config.STATE_NORMAL if care_kind is not None else state

        # 大模型只产出「回复文本」。上面那三行（意图 / 状态 / 情绪）已经是
        # 最终结果，不会因为走了大模型而变 —— 见模块 docstring 的说明。
        source = "llm"
        reply_text = self._online_reply(speech_state, intent, emotion, user_text,
                                        on_thinking=on_thinking)
        if reply_text is None:
            # 不适用（危机 / 不适 / 问候这类）或调用失败，一律落回模板。
            source = "template"
            reply_text = self._compose_reply(speech_state, intent, emotion,
                                             user_text, vision)

        # 第三步：追加关怀。文案复用主动关怀那一套（已预合成，追加零延迟）。
        follow_up = ""
        if care_kind is not None:
            follow_up = self.proactive_reply(care_kind, vision)
            # 只有**真的追加了**才记账，且按**类型**记（同类型 30 秒内
            # 不重复；别的类型不受影响）。
            self._last_care_at[care_kind] = time.monotonic()

        return DialogueReply(
            reply=reply_text,
            state=state,
            intent=intent,
            emotion_label=emotion.label,
            emotion_score=emotion.score,
            emotion_detail=emotion.emotion,
            discomfort=emotion.discomfort,
            emergency=bool(emergency_kind),
            emergency_kind=emergency_kind,
            follow_up=follow_up,
            care_kind=care_kind or "",
            reason=f"视觉={vision.state}({vision.reason}) 文本={emotion.label}/{emotion.emotion} "
                   f"意图={intent} 来源={source} 关怀={self._care_note(care_kind, vision, emotion)}",
        )

    # ------------------------------------------------------------------
    # 第三步：追加关怀
    # ------------------------------------------------------------------

    def _care_kind(self, vision: VisionState, emotion: TextEmotion) -> Optional[str]:
        """这一轮该不该追加关怀；该则返回 kind（``care_sad`` / ``care_tired``）。

        **只看 ``vision.state``，不看融合状态。** 这一条是整个设计的支点：

        追加的关怀，说的是「**摄像头看到了、而老人这句话自己没说**」的事。
        所以两个条件缺一不可 ——

        1. 摄像头真的看到了（``vision.state`` 在 :data:`proactive.CARE_KINDS`
           里）。只看文字说「我有点累」而摄像头什么都没看到时，回答本身
           （``(tired, venting)`` 那几句，或大模型的共情）就已经接住了，
           这时候再蹦一句「我看您眼睛都快睁不开了」是在瞎编 —— 没人看到
           他的眼睛。关怀语里全是这种**关于身体的断言**，只能来自摄像头。
        2. 他自己没说（:meth:`_text_already_says`）。否则同一件事说两遍。

        另外还受 :attr:`_care_interval` 节流，见 :meth:`_care_due`。
        """
        kind = proactive_lib.CARE_KINDS.get(vision.state)
        if kind is None:
            return None                 # normal / absent：不需要关心，不打扰
        if self._text_already_says(kind, emotion):
            return None
        if not self._care_due(kind):
            return None
        return kind

    @staticmethod
    def _text_already_says(kind: str, emotion: TextEmotion) -> bool:
        """老人这句话**自己**已经说了这件事吗。

        说过了就不再追加 —— 回答已经接住了，再追一句就是重复。
        这也是"关怀只报告他没说的那部分"的直接落地。
        """
        if kind == "care_tired":
            return emotion.emotion == "tired"
        if kind == "care_sad":
            return emotion.is_negative()
        return False

    def _care_due(self, kind: str) -> bool:
        """这一**类**关怀距上次真的追加是否已经够久。

        冷却是**按类型记账**的（交互规则：同类型关怀 30 秒内不重复）：
        刚说过「困了就歇会儿」（care_tired），30 秒后老人情绪低落，
        「不开心就说说话」照样能追加 —— 是两类关怀，互不占额度。
        """
        return (time.monotonic() - self._last_care_at.get(kind, 0.0)
                >= self._care_interval)

    @staticmethod
    def _care_note(kind: Optional[str], vision: VisionState,
                   emotion: TextEmotion) -> str:
        """给日志用的「这轮为什么追加/没追加」。

        答辩时被问"为什么这次没关心我"，看这一行就知道是视觉没看到、
        还是他自己说了、还是没到间隔。刻意写成好读的中文而不是布尔值。
        """
        if kind is not None:
            return kind
        if vision.state not in proactive_lib.CARE_KINDS:
            return f"无(视觉={vision.state}不需关心)"
        care_kind = proactive_lib.CARE_KINDS[vision.state]
        if DialogueEngine._text_already_says(care_kind, emotion):
            return "无(他自己说了)"
        return "无(未到间隔)"

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

        ⚠️ 除了主动开口，**追加关怀也走这里**（见 :meth:`_care_kind`）——
        两处要说的是同一批话，去重也必须共用一套，否则同一句关怀会在
        "追加"和"主动"两条线上各说一遍。

        于是有一个**隐含要求**：调用方写历史时，关怀语必须是**独立的一行**、
        内容是干净的文案本身。去重（:meth:`_recent_replies`）是**精确相等**
        判断，把关怀和主回复拼成一行写进去的话，这一句就永远匹配不上了。
        main.py 的 ``handle_chat`` 按这个要求分两行写。
        """
        candidates = list(PROACTIVE_TEMPLATES.get(kind) or PROACTIVE_TEMPLATES["greeting"])

        # 和普通应答共用同一套「避开最近说过的」逻辑，靠历史 CSV 去重
        recent = self._recent_replies()
        fresh = [text for text in candidates if text not in recent]
        pool = fresh or candidates

        return self._rng.choice(pool)

    def thinking_reply(self) -> str:
        """挑一句应声词。见 :data:`ACK_REPLIES`。

        :data:`ACK_REPLIES` 为空（用户要求去掉应答式前置短语）时返回空串 ——
        调用链上的播报层对空文本是 no-op，整条应声通路随之静默。

        避开上一次说过的那句：连着两轮"嗯，我想想"，听起来就是卡带。
        这里**不用** ``_recent_replies()``（那个读历史 CSV），因为应声词
        刻意**不写进历史** —— 历史是"机器人说过什么"的记录，而应声词
        不是回答，把它记进去会污染给大模型的上下文。
        """
        if not ACK_REPLIES:
            return ""
        fresh = [text for text in ACK_REPLIES if text != self._last_ack]
        chosen = self._rng.choice(fresh or list(ACK_REPLIES))
        self._last_ack = chosen
        return chosen

    # ------------------------------------------------------------------
    # 1) 意图识别
    # ------------------------------------------------------------------

    def recognize_intent(self, text: str, emotion: TextEmotion) -> str:
        """关键词匹配。命中多条时取 INTENT_RULES 里靠前的那个。

        单字关键词（见 :data:`STANDALONE_KEYWORDS`）要求整句就是它本身 ——
        否则「走得早」这种会把一句悼念识别成打招呼。
        被反例"借走"的关键词（见 :data:`KEYWORD_BLOCKERS`）不作数 ——
        「我心疼孙子」不是身体不适。
        """
        stripped = (text or "").strip()
        for intent, keywords in INTENT_RULES:
            # 「你真烦」带着 praise 的"你真"，但情绪是负面的 —— 那不是夸奖。
            # 跳过这条规则继续匹配，多半会落到 venting（负面情绪兜底），
            # 回倾诉模板比回"谢谢您夸奖"像话得多。
            if intent == "praise" and emotion.label == "negative":
                continue
            for keyword in keywords:
                if keyword not in text:
                    continue
                if any(block in text for block in KEYWORD_BLOCKERS.get(keyword, ())):
                    continue  # 关键词被反例"借走"了，见 KEYWORD_BLOCKERS
                if keyword in STANDALONE_KEYWORDS:
                    if stripped.rstrip(_TRAILING_NOISE) == keyword:
                        return intent
                    continue
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
        """按优先级挑模板。完整顺序见 :meth:`_candidates`。

        一句话概括：(状态, 意图) 精确匹配（内容意图除外）→ 与状态无关的意图
        → (normal, 意图) 内容池 → 意图通用 → 状态通用 → 全局兜底。
        既能对"疲劳 + 告别"说"那您早点休息"，又保证内容意图**永远**答的是
        内容而不是状态套话。
        """
        candidates = self._candidates(state, intent, emotion, user_text)

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

    def _candidates(self, state: str, intent: str, emotion: TextEmotion,
                    user_text: str = "") -> List[str]:
        """收集候选回复。

        **顺序就是这套对话的第一条设计约束：状态不能顶替内容。**

        改之前 ``STATE_FALLBACK[state]`` 排在 ``INTENT_FALLBACK`` **之前**，
        于是老人说「今天天气不错」、脸上有点乏，答的是「您看着有点累了，
        先歇一会吧」—— 状态句顶替了内容回答。现在它降到最后，只在连
        ``(normal, 意图)`` 的内容兜底都没有时才用。
        """
        # 身体紧急求助排在最前（"救命""叫救护车"）：回复要确定、简短、
        # 像一个正在进行的动作 —— main.py 同时触发**真实**的家人通知。
        # 它排在危机情绪之前：「救命，我不想活了」该先通知家人到场。
        if intent == "emergency":
            return list(EMERGENCY_REPLIES)

        # 危机信号其次，排在身体不适**之前** —— 这两件事会同时出现
        # （"我疼得不想活了"），而这时候该回应的是"不想活"，不是"疼"。
        # 同样会触发家人通知（kind=crisis）。
        #
        # 这几句刻意写成**确定性的**、逐句审过的：
        #   - 不说教、不追问原因（追问会让对方更孤立）
        #   - 只做两件事：接住情绪 + 把人往"身边有人"上引
        #   - 不用"您别想不开"这类否定式劝阻（研究上无效，还像在敷衍）
        # 和主动关怀的文案不同，这几句**允许**用问句 —— 那句"现在身边有人吗"
        # 是在确认安全，不是在索取信息。
        if emotion.crisis:
            return list(CRISIS_REPLIES)

        # 身体不适其次 —— 健康问题不能只回一句"多喝热水"式的套话。
        #
        # 但 reminder 让路："提醒我吃药"里也有健康词（文本情绪层把"吃药"
        # 记成 discomfort），可那是在**求办事**，不是在说自己不舒服 ——
        # 被 discomfort 抢走会答成"坐下缓缓喝口温水"。真在说身体的事
        # （"下雨天我腿疼"这类）仍归身体关怀，不受 weather 意图影响。
        if intent == "discomfort" or (emotion.discomfort and intent != "reminder"):
            return list(DISCOMFORT_REPLIES)

        # (状态, 意图) 精确匹配 —— 状态敏感的社交应答在这里，是**对的**
        # （累的时候道别，说「那您早点休息」比说「有空再聊」好）。
        # 但 CONTENT_INTENTS 例外：那几个意图的回答必须是在接用户说的那件事，
        # 状态不参与选词，直接跳到下面的 (normal, 意图) 内容池。
        key = (state, intent)
        if intent not in CONTENT_INTENTS and key in REPLY_TEMPLATES:
            return list(REPLY_TEMPLATES[key])

        # 有些意图的回答内容和用户状态无关，问"你是谁"时不管他累不累，
        # 都得答出"我是谁"。这类必须排在状态兜底之前，否则会被通用套话盖掉。
        if intent in STATE_INDEPENDENT_INTENTS and intent in INTENT_FALLBACK:
            # 节日是唯一要**现算**的：日期来自 FESTIVALS 表加"今天"，
            # 答案每天都不一样，不能靠预写的模板池随机挑。
            if intent == "festival":
                answer = festival_reply(user_text) if user_text else ""
                return [answer] if answer else list(INTENT_FALLBACK["festival"])
            # 体征是唯一"数值来自传感器"的现算：读数在 VitalsStore 里
            # 每秒都在变，模板池里写死任何数都会过期。没测出 / provider
            # 挂了都按"没数"处理，落回 INTENT_FALLBACK 的引导模板 ——
            # 宁可说"还没测到"，也不能报一个 0 或旧数。
            if intent == "vitals":
                answer = vitals_reply(user_text, self._current_vitals()) if user_text else ""
                return [answer] if answer else list(INTENT_FALLBACK["vitals"])
            # 时间/日期/星期：本机时钟现算（成都时间 UTC+8），答案每秒都
            # 不一样，同样不能靠模板池。
            if intent == "time":
                answer = time_reply(user_text) if user_text else ""
                return [answer] if answer else list(INTENT_FALLBACK["time"])
            # 天气：实时联网查（weather_provider 注入）。查不到返回空串，
            # 落回"我看不到天气"的诚实模板 —— 绝不编温度。
            # 活动类问句（"适合出门散步吗"）走简短建议 describe_activity：
            # 否则问天气和问散步播的是一字不差的同一段全文 —— 同一时段
            # 天气数据相同，describe_weather 输出是确定性的。
            if intent == "weather":
                answer = ""
                if user_text:
                    report = self._current_weather()
                    if any(hint in user_text for hint in WEATHER_ACTIVITY_HINTS):
                        answer = describe_activity(report, city=config.WEATHER_CITY)
                    else:
                        answer = describe_weather(report, city=config.WEATHER_CITY)
                return [answer] if answer else list(INTENT_FALLBACK["weather"])
            # 讲故事：承接句 + 从预写池挑的怀旧小故事（不与上次重复），
            # 结尾的互动钩子把话头递回给老人。
            if intent == "story":
                answer = ""
                if user_text:
                    story = story_reply(self._rng, self._last_story)
                    self._last_story = story
                    answer = self._rng.choice(_STORY_OPENINGS) + story
                return [answer] if answer else list(INTENT_FALLBACK["story"])
            return list(INTENT_FALLBACK[intent])

        # 话题池排在通用闲聊之前：老人聊到具体的事（散步、孙子、吃饭…），
        # 就接那件事，而不是「嗯嗯，然后呢」地应付。只对 chat 生效 ——
        # venting 走情绪模板，社交应答走 (状态, 意图) 精确匹配。
        # 大模型可用时 chat 根本到不了这里（先走 _online_reply），
        # 所以这池子是**没有大模型时的兜底质量**。
        if intent == "chat" and user_text:
            topic_pool = topic_replies_for(user_text)
            if topic_pool:
                return topic_pool

        # 内容兜底：答内容就用"状态正常"那套话。
        #
        # 状态不决定**怎么答**，只决定答完之后要不要追加一句关怀 ——
        # 这正是 (normal, chat) 那几句的性质：它们本来就是状态中性的
        # （「嗯嗯，{topic}——然后呢？」），换个状态照样成立。所以不用为
        # 每种状态各写一套内容模板。见 CONTENT_INTENTS。
        normal_key = (config.STATE_NORMAL, intent)
        if normal_key in REPLY_TEMPLATES:
            return list(REPLY_TEMPLATES[normal_key])

        # 意图通用
        if intent in INTENT_FALLBACK:
            return list(INTENT_FALLBACK[intent])

        # 状态通用。
        #
        # ⚠️ 当前位置**不可达** —— recognize_intent 只会返回 INTENT_RULES 里
        # 那些意图加上 venting/chat 兜底，上面各条已经把它们全覆盖了
        #（identity/time/weather/reminder/festival/question 走意图无关那条，
        # discomfort 在身体不适那条就返回了）。留着是**安全网**：将来加了新
        # 意图、又忘了配 (normal, 新意图) 模板时，退到这里至少还说人话，
        # 而不是掉到下面的通用兜底。看到这里"没人走"不要删。
        if state in STATE_FALLBACK:
            return list(STATE_FALLBACK[state])

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
                      user_text: str,
                      on_thinking: Optional[Callable[[str], None]] = None,
                      ) -> Optional[str]:
        """让大模型生成一句回复。返回 ``None`` 表示"不适用或失败，请落回模板"。

        契约：**只可能返回一句洗干净的、非空的文本，或者 None**。
        绝不抛异常 —— 调用方拿到 None 就落回模板，那条路永远是好的。

        ``state`` 由调用方决定给哪一个：``respond()`` 在"这一轮要追加关怀"时
        传的是 ``STATE_NORMAL``（隐去视觉状态），否则传真实状态。判断依据和
        理由都在那边，这里不再重复。
        """
        eligible, why = self.llm_eligible(intent, emotion, user_text)
        if not eligible:
            logger.debug("这句不走大模型：%s", why)
            return None

        # 到这一步才垫应声词：上面那些分支（危机 / 不适 / 语气词）都是本地
        # 查表出结果，没有等待，垫一句纯属废话。
        #
        # ACK_REPLIES 为空时 thinking_reply() 返回 ""，on_thinking("") 会
        # 在播报层被丢掉（say_async 对空文本是 no-op）—— 应声通路整体静默，
        # 用户听到的就是：说完 → 直接回答。
        #
        # 位置也很讲究 —— 必须在 complete 之前、且不能 await 它。
        if on_thinking is not None:
            try:
                on_thinking(self.thinking_reply())
            except Exception:
                # 应声词是锦上添花，它失败绝不能连累真正的回复。
                logger.debug("应声词回调失败（不影响回复生成）", exc_info=True)

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

    def _current_vitals(self):
        """取最新体征读数。provider 抛异常按"没测出"处理 —— 取数失败
        绝不能连累回复生成（和 _history_messages 读历史失败同一条忍法）。"""
        if self._vitals_provider is None:
            return None
        try:
            return self._vitals_provider()
        except Exception:
            logger.debug("体征读数取用失败，按没测出处理", exc_info=True)
            return None

    def _current_weather(self):
        """查一次实时天气。provider 缺失 / 网络失败都返回 None ——
        调用方落回"看不到天气"的诚实模板，绝不连累回复生成。"""
        if self._weather_provider is None:
            return None
        try:
            return self._weather_provider()
        except Exception:
            logger.debug("天气查询失败，按看不到处理", exc_info=True)
            return None

    def _vitals_hint(self) -> str:
        """把最新体征读数翻成给大模型看的一行提示；没测出返回空串。

        没数时**刻意给空**而不是"未测出"—— 提示词里出现"未测出"只会
        诱导模型拿它当话题。数值后面带上静息常见范围，模型才有依据说
        偏高偏低；免责声明跟在数值后面一起给（红线：免责必须随数值走）。
        """
        reading = self._current_vitals()
        if reading is None:
            return ""
        hr = getattr(reading, "hr", None)
        rr = getattr(reading, "rr", None)
        if hr is None and rr is None:
            return ""
        pieces: List[str] = []
        if hr is not None:
            pieces.append("心率约每分钟%d次（静息常见60到100）" % int(hr))
        if rr is not None:
            pieces.append("呼吸约每分钟%g次（静息常见12到20）" % float(rr))
        return "%s；%s" % ("、".join(pieces), VITALS_NOTE)

    def _llm_messages(self, state: str, intent: str,
                      emotion: TextEmotion, user_text: str) -> List[Dict[str, str]]:
        """拼出发给大模型的消息数组。"""
        messages: List[Dict[str, str]] = [{
            "role": "system",
            "content": build_system_prompt(
                state, intent, emotion.label, vitals_hint=self._vitals_hint()),
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
    # 这里**刻意没有** (sad,"chat") / (absent,"chat") —— 曾经有，已删。
    # chat 属于 CONTENT_INTENTS，_candidates() 会跳过精确匹配直接走
    # (normal,"chat") 的内容池，所以那两条永远选不到。留着是死数据，
    # 而且会误导后来人以为"低落时闲聊有专门的答法"。
    # 测试 test_only_normal_chat_template_exists 会守住这条不变量。

    # ---- 走神/无人 absent ----
    (config.STATE_ABSENT, "greeting"): [
        "哎，我在呢。刚才没看见您，您去哪里了？",
        "您好呀，可算听见您说话了。",
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

# --------------------------------------------------------------------------
# 话题池：老人聊到具体的事，就接那件事，而不是拿通用套话应付。
#
# chat 意图在没有大模型时（或请求失败落回模板时）原先只有三条通用应答
# （「嗯嗯，{topic}——然后呢？」）—— 把老人的话复述一遍了事，聊什么
# 都这个反应，聊两轮就索然。这里按话题给几句**接着内容说**的话，
# 覆盖老人最常聊的四类：生活日常、家人情感、陪伴需求、健康留意。
#
# 写作约束同 REPLY_TEMPLATES：一句话、不说教、不装懂 ——
# 尤其健康那组只往"按医嘱、找家人"上引，不给任何医疗意见。
# 天气**不进**话题池：问天气走 weather 意图（诚实答"看不了"），
# 「今天天气不错」这种评价继续按闲聊回。
# --------------------------------------------------------------------------
TOPIC_REPLIES: List[Tuple[Tuple[str, ...], List[str]]] = [
    # ---- 生活日常 ----
    (("散步", "遛弯", "走一走", "锻炼", "运动"), [
        "散步好，走走浑身都松快。慢点走，别累着。",
        "您这习惯好，天天动一动，腿脚就灵便。",
        "出去走走比闷在家里强，累了就找个地方坐坐。",
    ]),
    (("吃饭", "做饭", "买菜", "胃口", "饿"), [
        "到饭点就好好吃饭，吃饱了心里才踏实。",
        "想吃啥就做点啥，胃口好多吃两口。",
        "一个人吃饭也别糊弄，身子是自己的。",
    ]),
    (("睡", "失眠", "觉少"), [
        "觉少别硬躺着，白天少眯点，晚上反倒好睡。",
        "睡不好人就乏。躺下别想事，慢慢就着了。",
    ]),
    (("电视", "戏曲", "京戏", "看戏", "听戏"), [
        "看会儿戏挺好，就是别看太久，眼睛要歇着。",
        "老戏百听不厌，您听着乐呵就好。",
    ]),
    # ---- 家人与情感陪伴 ----
    (("孙子", "孙女", "儿子", "闺女", "孩子", "儿女"), [
        "孩子们忙是好事，他们过得好，您就少操点心。",
        "想他们了就打个电话，听见声音心里就踏实了。",
        "儿孙自有儿孙福，您把自个儿顾好是头一件。",
    ]),
    (("孤单", "没意思", "无聊", "陪陪我", "没人说话"), [
        "有我陪着您呢，想聊什么咱们就聊什么。",
        "我在家守着呢，您随时喊我都有应答。",
        "闷得慌就跟我说说话，我哪儿也不去。",
    ]),
    # ---- 聊天邀请（主动要聊天，不是情绪低落时的求陪伴）----
    # "你能陪我聊会儿天吗""咱们聊聊天""陪我说说话"——这类请求
    # 如果落回 (normal, chat) 的「嗯嗯，{topic}——然后呢？」会把整句
    # 当话题复述，变成"嗯嗯，你能陪我聊会儿天吗——然后呢？"，机械且
    # 不自然。单独识别，给热情接受的回应，把球抛回给老人让他开口。
    (("陪我聊", "聊聊天", "聊会儿", "聊会天", "说说话", "陪我说",
      "陪你聊", "聊一聊", "聊聊天", "和您聊", "跟我聊"), [
        "好呀，您想聊什么？我听着呢。",
        "当然行，您随便说，我陪着您。",
        "好，咱们聊。您最近怎么样？",
        "行呀，您起个头，我接着您说。",
    ]),
    (("老朋友", "老同事", "老家", "老伙计"), [
        "老朋友难得，能聊上几句就是福气。",
        "老熟人一提起来话就长了，回头让他们常来坐坐。",
    ]),
    # ---- 健康留意（不给医疗意见，只往"按医嘱、找家人"上引）----
    (("血压", "血糖", "体检", "吃药"), [
        "这些事按医生说的来最稳妥，药别自己加减。",
        "身体的事别自己扛着，让家里人陪您问问医生。",
        "记不清吃药的点儿，就写张条贴冰箱上。",
    ]),
]


def topic_replies_for(text: str) -> List[str]:
    """用户这句话聊到了哪个话题，返回那组回复；没聊到具体话题返回空。

    只做子串匹配就够 —— 话题关键词都是"散步""孙子"这种低歧义词，
    不像意图词表那样需要反例保护（教训见 :data:`KEYWORD_BLOCKERS`）。
    """
    for keywords, replies in TOPIC_REPLIES:
        if any(keyword in text for keyword in keywords):
            return list(replies)
    return []


# --------------------------------------------------------------------------
# 节日日期表：问「重阳节是多久」要答得出日子，而且**不能编**。
#
# 农历节的公历日期年年变，大模型记的往往是训练数据里的旧年份 ——
# 让它自由发挥，"重阳节"能给你编出三四个版本。所以直接把公历日期
# **手抄成表**（覆盖 2025~2027 三个演示年份），逐字审过再用；
# 不引入农历换算库（新依赖，演示机离线装包麻烦），到期补表即可。
# 表里查不到（年份超出 / 不认识的节日）就诚实说记不准，落回通用模板。
# --------------------------------------------------------------------------
_FESTIVAL_YEARS = (2025, 2026, 2027)


def _every_year(month_day: Tuple[int, int]) -> Dict[int, Tuple[int, int]]:
    """固定公历节日（元旦/国庆这类）不用逐年抄，直接铺满表里覆盖的年份。"""
    return {year: month_day for year in _FESTIVAL_YEARS}


#: (节日名, 农历说法（公历节日为空串）, 触发词, {年份: (月, 日)})
FESTIVALS: List[Tuple[str, str, Tuple[str, ...], Dict[int, Tuple[int, int]]]] = [
    ("元旦", "", ("元旦", "跨年"), _every_year((1, 1))),
    ("春节", "农历正月初一", ("春节", "过年"), {
        2025: (1, 29), 2026: (2, 17), 2027: (2, 6)}),
    ("元宵节", "农历正月十五", ("元宵",), {
        2025: (2, 12), 2026: (3, 3), 2027: (2, 20)}),
    ("清明节", "", ("清明",), {2025: (4, 4), 2026: (4, 5), 2027: (4, 5)}),
    ("劳动节", "", ("劳动节", "五一"), _every_year((5, 1))),
    ("端午节", "农历五月初五", ("端午",), {
        2025: (5, 31), 2026: (6, 19), 2027: (6, 9)}),
    ("儿童节", "", ("儿童节",), _every_year((6, 1))),
    ("中秋节", "农历八月十五", ("中秋",), {
        2025: (10, 6), 2026: (9, 25), 2027: (9, 15)}),
    ("国庆节", "", ("国庆",), _every_year((10, 1))),
    ("重阳节", "农历九月初九", ("重阳",), {
        2025: (10, 29), 2026: (10, 18), 2027: (10, 8)}),
    ("除夕", "农历一年的最后一夜", ("除夕",), {
        2025: (1, 28), 2026: (2, 16), 2027: (2, 5)}),
]

#: 问"今天过什么节"但没有点名具体节日时的触发词
_GENERIC_FESTIVAL_WORDS = ("节日", "过节", "什么节")


def festival_reply(text: str, today: Optional[date] = None) -> str:
    """按节日表回答「X 节是哪天」。答不上返回空串（调用方落回通用模板）。

    :param today: 注入"今天"供测试用；缺省用系统当天日期。
    :return: 一句 ≤40 字、能直接念出来的回答；查不到就返回空串。
    """
    today = today or date.today()

    # 点名了具体节日：取**出现位置最早**的那个（"中秋国庆放几天"答中秋）
    best: Optional[Tuple[int, str, str, Dict[int, Tuple[int, int]]]] = None
    for name, lunar, aliases, dates in FESTIVALS:
        for alias in aliases:
            pos = text.find(alias)
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, name, lunar, dates)
    if best is not None:
        _, name, lunar, dates = best
        # ⚠️ 表里存的是 (月, 日) 元组，和 date 直接比较永远不相等 ——
        # 必须先拆出 (month, day)。
        if (today.month, today.day) in dates.values():
            lunar_part = f"，{lunar}" if lunar else ""
            return f"今天就是{name}呀{lunar_part}。"
        if today.year in dates:
            when = date(today.year, *dates[today.year])
            lunar_part = f"，{lunar}" if lunar else ""
            if when > today:
                days = (when - today).days
                return f"{name}是{when.month}月{when.day}号{lunar_part}，还有{days}天。"
            # 今年的过完了 → 答下一年的
            if today.year + 1 in dates:
                nxt = date(today.year + 1, *dates[today.year + 1])
                return f"今年的{name}过了，明年是{nxt.month}月{nxt.day}号。"
            return f"今年的{name}过了，往后的日期我记不太准了。"
        if today.year + 1 in dates:
            nxt = date(today.year + 1, *dates[today.year + 1])
            lunar_part = f"，{lunar}" if lunar else ""
            return f"{name}是明年{nxt.month}月{nxt.day}号{lunar_part}。"
        return ""  # 年份超出表 → 诚实交给通用模板

    # 没点名节日，但问了"今天什么节"：照实说今天是不是节
    if any(word in text for word in _GENERIC_FESTIVAL_WORDS):
        for name, _, _, dates in FESTIVALS:
            if (today.month, today.day) in dates.values():
                return f"今天就是{name}呀。"
        return "今天不是什么节。您问哪个节的日期，春节端午重阳我都记得。"

    return ""


# --------------------------------------------------------------------------
# 时间 / 日期 / 星期现答（成都时间，UTC+8）
# --------------------------------------------------------------------------

#: 项目部署在成都，统一按东八区报时 —— 不依赖本机时区设置
#: （演示机器万一被改成别的时区，报出来的时间不能跟着错）。
CHENGDU_TZ = timezone(timedelta(hours=8))

_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def _hour_period(hour: int) -> str:
    """小时 → 凌晨/早上/上午/中午/下午/晚上。"""
    if hour < 6:
        return "凌晨"
    if hour < 9:
        return "早上"
    if hour < 11:
        return "上午"
    if hour < 14:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


def _format_clock(now: datetime) -> str:
    """把时刻翻成口语，例：晚上七点四十五分 / 中午十二点整 / 早上八点零五分。

    老人听的是"下午三点"，不是"15点" —— 24 小时制必须折成 12 小时制，
    时段词（_hour_period）负责把上午下午说清楚。
    """
    hour12 = now.hour % 12 or 12
    minute = now.minute
    if minute == 0:
        minute_text = "整"
    elif minute == 30:
        minute_text = "半"
    elif minute < 10:
        minute_text = "零%d分" % minute
    else:
        minute_text = "%d分" % minute
    return "%s%d点%s" % (_hour_period(now.hour), hour12, minute_text)


def time_reply(text: str, now: Optional[datetime] = None) -> str:
    """按问句报时刻 / 日期 / 星期（成都时间）。

    与 :func:`festival_reply` 同模式：确定性现算，绝不进大模型 ——
    模型没有时钟，让它答"几点"只会瞎编。

    :param now: 注入时间供测试（时区感知）；缺省取当前成都时间。
    :return: 一句能直接念的话。问句里没出现任何时间线索时默认报时刻。
    """
    now = now or datetime.now(CHENGDU_TZ)

    asks_time = ("几点" in text) or ("现在时间" in text) or ("成都时间" in text)
    asks_week = ("星期" in text) or ("周几" in text) or ("礼拜" in text)
    # "今天几点"也带"今天几"，所以这一项要把"几点"排除，否则问钟点会
    # 被当成问日期。
    asks_date = (("几号" in text) or ("多少号" in text) or ("日期" in text)
                 or (("今天几" in text) and ("几点" not in text)))
    if not (asks_time or asks_week or asks_date):
        asks_time = True

    # 问日期就把星期一起报 —— 老人问"今天几号"多半两个都想知道，
    # 一句话带全比追问强。只问星期 / 只问钟点则不多嘴。
    if asks_date:
        head = "今天是%d月%d号，%s。" % (now.month, now.day, _WEEKDAYS[now.weekday()])
        if asks_time:
            return head + "现在是%s。" % _format_clock(now)
        return head

    if asks_week:
        return "今天是%s。" % _WEEKDAYS[now.weekday()]

    return "现在是%s。" % _format_clock(now)


# --------------------------------------------------------------------------
# 体征现答：心率 / 呼吸的数值与分档建议
# --------------------------------------------------------------------------

#: 静息参考范围（成人口径）：心率 60–100 次/分，呼吸 12–20 次/分。
#: "日常参考"不是医疗标准 —— 所以每句回答的尾部必须带 VITALS_NOTE。
HR_RANGE = (60, 100)
RR_RANGE = (12, 20)

#: 每一档的建议都只落在机器人真能做的三件事上：歇一歇、观察、告诉
#: 家里人。不给医疗口径（不上"去医院""吃药"），与 discomfort 分支、
#: sanitize_reply 的 _MEDICAL_RE 同一条红线。
_VITALS_ADVICE: Dict[Tuple[str, str], str] = {
    ("hr", "normal"): "在静息常见的60到100之间，看着挺好的",
    ("hr", "high"): "比静息常见的60到100偏高，先坐下歇几分钟缓一缓，要是一直这么快，记得告诉家里人",
    ("hr", "low"): "比静息常见的60到100偏低，要是觉得头晕没力气，别硬撑，说一声让家里人知道",
    ("rr", "normal"): "在静息常见的12到20之间，看着挺好的",
    ("rr", "high"): "比静息常见的12到20快，坐下慢慢做几次深呼吸，要是一直这么快，记得告诉家里人",
    ("rr", "low"): "比静息常见的12到20慢，要是觉得胸闷没劲，别硬撑，说一声让家里人知道",
}


def _vitals_band(value: float, reference: Tuple[int, int]) -> str:
    """数值落在哪一档：normal / high / low（边界值算 normal）。"""
    lo, hi = reference
    if value > hi:
        return "high"
    if value < lo:
        return "low"
    return "normal"


def vitals_reply(text: str, reading) -> str:
    """按问句报心率 / 呼吸的当前测量值并给分档建议。答不上返回空串。

    与 :func:`festival_reply` 同一个模式：确定性现算 —— 数值在
    :class:`~core.vitals.VitalsStore` 里，是大模型看不见的数据，
    问"我心率多少"让它答只会编一个数。

    被问的指标没测出（A 端窗口未满 / SNR 不足）返回空串，调用方落回
    :data:`INTENT_FALLBACK` 的引导模板 —— **不报 0、不拿另一个指标充数、
    不报旧数**。过期心率比没有心率更危险。

    :param reading: ``RppgReading`` 或 None（鸭子类型读属性）。
    """
    if reading is None:
        return ""
    hr = getattr(reading, "hr", None)
    rr = getattr(reading, "rr", None)

    asks_hr = any(word in text for word in ("心率", "心跳", "脉搏"))
    asks_rr = "呼吸" in text
    if not asks_hr and not asks_rr:
        # 词表里的话都带指标名，走到这里是防御：两个都报。
        asks_hr = asks_rr = True

    parts: List[str] = []
    if asks_hr:
        if hr is None:
            return ""
        parts.append("您的心率大约每分钟%d次，%s"
                     % (int(hr), _VITALS_ADVICE[("hr", _vitals_band(int(hr), HR_RANGE))]))
    if asks_rr:
        if rr is None:
            return ""
        rr_value = float(rr)
        parts.append("您的呼吸大约每分钟%g次，%s"
                     % (rr_value, _VITALS_ADVICE[("rr", _vitals_band(rr_value, RR_RANGE))]))
    return "。".join(parts) + "。" + VITALS_NOTE + "。"


# ---------------------------------------------------------------------------
# 讲故事（story 意图）
# ---------------------------------------------------------------------------

#: 怀旧小故事池。每条 3 句话：起头点题、一个年代细节、结尾一句互动钩子
#: （把话头递回给老人，故事讲完对话不冷场）。
#:
#: 为什么预写、不走大模型：
#: ①**内容可控** —— 模型现编的长故事未必贴合老人的年代记忆，错一个
#:   年代细节就出戏；怀旧故事的"怀旧感"靠的是具体的物（煤油灯、铁环、
#:   布票），这些必须挑准了写死。
#: ②**长度可控** —— 长文本流式合成中途断流的风险随字数涨（见
#:   core/voice/tts.py 的 TimeoutError 教训），每条压在 90 字上下。
#: 全部口语化、不说教、不煽情，年代物取 50-70 年代城乡共同记忆。
STORY_POOL: List[str] = [
    "过去没电灯，晚上写作业就点一盏煤油灯。灯芯烧着烧着\"啪\"地爆个灯花，"
    "大人们就说，这是要来喜事了。您小时候是不是也这么点过灯？",
    "以前村里放露天电影，天还没黑，孩子们就搬着小板凳去占地方。"
    "银幕反面也坐满了人，看着倒着走路，照样看得有滋有味。"
    "您还记得那时候看的片子吗？",
    "那些年夏天热，吃过晚饭就搬个竹床到院子里，摇着蒲扇数星星，"
    "听老人讲牛郎织女。蚊子嗡嗡的，也舍不得进屋。"
    "您还记得那样的夏夜吗？",
    "那时候买东西都去供销社，玻璃柜台里摆着水果糖，一毛钱好几颗。"
    "买不买不要紧，趴在柜台上瞧一眼，心里都甜滋滋的。"
    "您还记得供销社吗？",
    "逢五逢十去赶集，集上人多，热热闹闹的。就为吃一碗热豆腐脑，"
    "多走十里地也不觉得累，一路上尽是跟街坊打招呼的。"
    "您赶集最爱吃哪一样？",
    "放学铃一响，书包一扔就疯跑，男孩滚铁环，女孩跳皮筋。"
    "铁环一路哗啦啦地响，天擦黑了，大人站在门口喊吃饭，才恋恋不舍地散。"
    "您小时候玩什么？",
]

#: 讲故事前的承接句。只说"给您讲一个"不热络，先应一声再进正题。
_STORY_OPENINGS = [
    "好嘞，我给您讲个过去的事儿，您听听。",
    "行，那我给您说一段老底子的事。",
]


def story_reply(rng: Optional[random.Random] = None, last: str = "") -> str:
    """从 :data:`STORY_POOL` 挑一条怀旧小故事（**只挑本体**，承接句调用方拼）。

    ``last`` 传上次讲过的那条 —— 从剩下的池子里挑，保证连着点播两次
    不会听到同一个故事。全池都被排除时（理论上只有池子缩到 1 条才会）
    才放回去。
    """
    picker = rng or random.Random()
    pool = [s for s in STORY_POOL if s != last] or list(STORY_POOL)
    return picker.choice(pool)


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
    # 讲故事：story_reply 从预写池挑故事，这句只在拿不到 user_text 的
    # 兜底路径上出现（意图本身就来自文本匹配，正常到不了这里）。
    "story": [
        "您想听故事啊，好嘞，跟我说一声讲个故事，我就给您来一个。",
        "过去的事情我还存着好几个呢，您想听就说一声。",
    ],
    # 「听不清」：老人没听清机器人说的话。reply 由 main.py 的 clarify
    # 处理路径**运行时构造**（复述上一句机器人说过的话），这里只是兜底——
    # 如果没有上一句可复述（启动后第一句就"听不清"），给一句安抚。
    "clarify": [
        "我刚才说得不够清楚，我再说一遍。",
        "我声音放大点，慢慢再说一次。",
    ],
    # 天气：**诚实**是唯一的正确答案。模型没有实时数据，让它自由发挥
    # 就是"乱回答"的最典型场景 —— 老人按它编的"晴天"出门挨淋。
    # 所以 weather 意图永远不进大模型（不在 LLM_ELIGIBLE_INTENTS 里），
    # 答案只有"我看不到 + 您自己瞧瞧 + 提一句添衣"。
    "weather": [
        "外头什么天我看不到。您出门前瞧瞧天，不对劲就添件衣裳。",
        "天气我这边看不了，怕说错了害您受凉。您瞧瞧窗外再定。",
    ],
    # 提醒：机器人没有闹钟和日历。应"好嘞我记下了"是假承诺，
    # 老人到点等不到，误的是吃药这种正事 —— 只能老实说做不了，
    # 给个还凑合的替代办法（字条、家里人）。
    "reminder": [
        "闹钟提醒我设不了，怕误了您的事。要紧的让家里人帮着记。",
        "这种要紧事我记不准。您写张字条贴墙上，比什么都稳妥。",
    ],
    # 节日：具体日期由 festival_reply 查表算出，这句只在没听清问的
    # 哪个节、或年份超出表的时候兜底 —— 同样是"我知道的就说，不知道
    # 的不装知道"。
    "festival": [
        "春节端午中秋重阳这些节，日期我都记得，您问哪个？",
        "哪个节的日期您说，我记着呢。",
    ],
    # 体征：A 端没测出（窗口没满 / SNR 不足 / 人刚离开画面）时的兜底。
    # 诚实说没数、引导重新测 —— **不报 0 也不报旧数**，过期心率比
    # 没有心率更危险。
    "vitals": [
        "这个数我还没测出来，您坐正对着镜头歇稳了，我就给您看看。",
        "还没测到呢，您坐好别动，等一下我给您报个数。",
    ],
    # 用户要求安静：只承认、不追问，并由 main 把主动开口静默一段时间。
    # 用户随后主动说任何话都会正常回答并自动解除（见 main.handle_chat）。
    "quiet": [
        "好，我不说话了。您需要我的时候叫我一声就成。",
        "好嘞，我安静陪着您，有事您招呼我一声。",
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

# 身体紧急求助（"救命""叫救护车""喘不上气"）。和危机那组的分工：
# 危机组先共情、允许问句；紧急组必须像**正在进行的动作** —— 确定、
# 短促、在布置帮助。main.py 在回复的同时真的发家人通知 + C 端弹窗。
# ⚠️ 不要承诺"救护车马上到"这种机器人无法保证的结果；只说自己正在做的。
EMERGENCY_REPLIES: List[str] = [
    "您别慌，我这就通知家里人过来，我一直陪着您。",
    "我听见了，马上帮您联系家里人。您撑一下，我就在您身边。",
    "我这就叫家里人来，您先坐下靠着歇一歇，我马上想办法。",
]

# --------------------------------------------------------------------------
# 应声词：用户说完了、回复还没生成出来时，先垫一句
#
# **当前为空：不应答。** 用户明确要求：回答问题前不要任何应答式前置短语，
# 直接给答案。``thinking_reply()`` 因此返回空串，``on_thinking`` 链路
# 自动失效（空文本不会触发播报）。保留着一条空表只是为了让调用点不必
# 改动 —— 想恢复时把文案填回来即可。
#
# 历史约束（恢复时仍需遵守）：
#
#   1. **必须能被预合成缓存命中**（所以静态、无占位符），否则它自己还要
#      等 1.2 秒的合成，比沉默更糟。
#   2. **不能有信息量。** 那是在猜，猜错了老人会纠正你。
#   3. **长度要压得住沉默。** 太短露馅，太长啰嗦。
# --------------------------------------------------------------------------

ACK_REPLIES: List[str] = []

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

    # 空闲闲聊（交互规则③）：画面里有人、连着 15 秒没有语音输入。
    # 和关怀类不一样，这里**要**用问句 —— 问候的目的就是邀请老人搭话，
    # 「不回应就不再重复」由调度器的 awaiting_reply 门禁保证。
    "idle_chat": [
        "您在呀，今天过得怎么样？",
        "要不要和我聊聊天？",
        "我陪您说说话，好不好？",
        "坐有一会了，要不要喝口水？",
    ],

    # 走神提醒：VAI 持续偏低（人还在画面里，只是分神了）。
    # 用问句邀请老人搭话 —— 和 idle_chat 同属"邀请式问候"，豁免"主动关怀
    # 不用问句"那条约束。话术要带"我看您有点走神"的观察，让老人知道
    # 这不是泛泛闲聊，是机器人注意到他分心了。
    "care_distracted": [
        "您看起来走神了，需要陪您聊聊天吗？",
        "我看您眼神有点飘，要不咱说说话？",
        "想什么呢？跟我聊聊呗，您说说我听着。",
        "走神了？没事，我也陪着您呢，说点啥都行。",
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
    # 紧急话术和危机话术同级 —— 真到喊"救命"时，绝不能再等一次联网合成。
    for text in EMERGENCY_REPLIES:
        add(text)
    # 应声词（当前为空表）：恢复文案后**尤其**不能漏 —— 它整句的意义就是
    # "立刻出声"，要是没被预合成，它自己就得先等一次网络往返，那还不如沉默。
    for text in ACK_REPLIES:
        add(text)

    return collected
