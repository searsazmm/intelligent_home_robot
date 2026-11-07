# -*- coding: utf-8 -*-
"""语音循环：听 → 识别 → 交给对话逻辑 → 朗读回复。

--------------------------------------------------------------------------
关键设计：识别出的文本走**和 8002、--stdin 完全相同的那一个漏斗**
--------------------------------------------------------------------------
``VoiceLoop`` 不自己处理对话，它把识别结果交给 ``on_text`` 回调
（由 main.py 注入 ``BackendB.handle_chat``）。于是麦克风输入、模块 C 的
文本输入、以及键盘输入三条路径共享同一套历史记录、状态覆盖和日志，
不会出现"语音走一套行为、打字走另一套"的分裂。

--------------------------------------------------------------------------
半双工：为什么不做回声消除
--------------------------------------------------------------------------
同一台机器的扬声器和麦克风之间必然有耦合，机器人播报的话会被自己听见，
识别成"用户说话"，再生成回复…… 形成自激回路，表现为机器人自言自语停不下来。

真正的解法是 AEC（声学回声消除），它需要参考信号和自适应滤波器，
超出本项目的范围。这里用两个便宜但有效的办法：

1. **半双工**：播报期间把麦克风数据丢掉（``Speaker`` 的播放回调里做）。
   代价是不能插话 —— 但对话轮次本来就是一问一答，影响很小。
   戴耳机时可以用 ``--barge-in`` 关掉它实现插话。
2. **相似度去重**：识别文本与上一句机器人回复高度相似时直接丢弃。
   这是兜底 —— 万一半双工没盖住（比如用户外放），还有一道防线。
"""

from __future__ import annotations

import difflib
import logging
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from core.voice.base import NullRecognizer, NullSynthesizer, Recognizer, Synthesizer
from core.voice.vad import Endpointer, SpeechSegment, VadConfig

logger = logging.getLogger(__name__)

#: 识别文本与上一条机器人回复的相似度超过这个值，判定为自己声音的残留。
#: 0.6 是实测出来的经验值：同一句话被回声识别回来通常能到 0.8 以上，
#: 而正常人附和一句"对""是"的相似度远低于此。
ECHO_SIMILARITY = 0.6

#: 连续出错多少次后放弃语音功能（而不是无限重试刷日志）。
MAX_CONSECUTIVE_ERRORS = 5

#: 出错后的退避（秒）
ERROR_BACKOFF = 1.0

#: 日志里一句话最多显示多少个字。整句都打出来会把日志淹掉，
#: 但截得太短又对不上是哪句话 —— 30 个字够看清开头的语气。
LOG_TEXT_CHARS = 30


def _brief(text: str) -> str:
    """把一句话压成适合进日志的短形式。"""
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= LOG_TEXT_CHARS:
        return repr(text)
    return repr(text[:LOG_TEXT_CHARS] + "…")


def _wav_seconds(path: str) -> float:
    """读 WAV 时长（秒）。读不出来返回 -1，**绝不抛异常**。

    只用来给日志提供对照值（"播了多久" vs "音频多长"），
    所以失败时返回哨兵值即可，不能让日志本身把播报搞崩。
    """
    try:
        with wave.open(path, "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return -1.0
            return handle.getnframes() / float(rate)
    except (OSError, wave.Error, ZeroDivisionError):
        return -1.0


def looks_like_echo(text: str, last_reply: str, threshold: float = ECHO_SIMILARITY) -> bool:
    """判断 text 是不是机器人自己刚说过的话被麦克风收了回去。

    用 ``difflib.SequenceMatcher``（标准库）做字符级相似度。
    它不需要任何依赖，对中文按字符比较也合适 —— 中文里"字"就是最小的
    有意义的单位，不像英文那样需要先分词。

    ``last_reply`` 可能是**几段拼起来的**（见 :meth:`Speaker.say_all`：
    回复 + 追加关怀，中间用空格分隔）。拿整段去比会让每一段都被稀释 ——
    一段 15 字、一段 20 字拼成 36 字，只回声了前一段时相似度只有
    2×15/(15+36)≈0.59，**刚好掉到 0.6 的门槛以下**，于是没认出来。
    所以按空格拆开，取其中最像的那一段来比。
    """
    if not text or not last_reply:
        return False
    segments = last_reply.split() or [last_reply]
    return any(
        difflib.SequenceMatcher(None, text, segment).ratio() >= threshold
        for segment in segments
    )


def is_meaningful(text: str) -> bool:
    """识别结果里有没有实际内容。

    STT 在噪声上经常吐出单个标点或一个"嗯"，这类结果拿去对话
    只会得到一句莫名其妙的回复，不如丢掉。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    # 去掉标点和空白后还剩至少一个字符才算有话
    body = "".join(ch for ch in stripped if ch.isalnum() or "一" <= ch <= "鿿")
    return len(body) >= 2


# 完整问句的**结尾**特征（配合"去句尾语气词"使用，见
# _looks_like_complete_question）。刻意只收疑问结构，不收裸动词 ——
# 判错一个故事片段为问句，长句合并就前功尽弃。
_QUESTION_TAILS: tuple = (
    "几点", "几号", "多少号", "星期几", "周几", "礼拜几", "几岁",
    "多少", "什么", "怎么样", "怎么了", "为什么", "为啥",
    "是不是", "好不好", "行不行", "对不对", "有没有", "能不能", "要不要",
    "哪个", "哪些", "哪里", "哪儿", "谁",
)
# 单字疑问语气词直接看原文结尾（它们不该被"去语气词"那步剥掉）。
_QUESTION_PARTICLES: tuple = ("吗", "呢")
# 剥掉之后再看疑问结构的句尾语气词。
_TRAILING_MOODS: str = "了呀啊啦哦吧嘛呢哟"


def _looks_like_complete_question(text: str) -> bool:
    """投机识别的文本是不是一句**独立的完整问句**。

    用于问句早放（不等续句窗口）：老人问「现在几点了」「天气怎么样」
    「吃饭了吗」，说完就是在等答案，窗口等待纯属浪费。

    判定：剥掉句尾语气词（"几点**了**""吃了**吗**"）后以疑问结构收尾，
    或原文以"吗/呢"收尾。讲故事式的片段（"半夜醒了好几次""头有点昏"）
    不带疑问尾巴 → 返回 False → 继续走续句合并。
    """
    text = (text or "").strip()
    if len(text) < 2:
        return False
    if text.endswith(_QUESTION_PARTICLES):
        return True
    core = text.rstrip(_TRAILING_MOODS)
    return any(core.endswith(tail) for tail in _QUESTION_TAILS)


# ---------------------------------------------------------------------------
# 回声时间窗与杂音冷静（"机器人自己说个不停"的最后一道防线）
# ---------------------------------------------------------------------------
#
# 现场实测（19:47 前后）机器人会连着自答好几轮：
#   语音识别：'那个肥满我刚从啊' → 回复
#   语音识别：'实话说一句不行行不行' → 回复
#   语音识别：'害别人' → 回复
# 这类文本是**远场失真的回声/环境杂音**被 STT 幻觉出来的 —— 半双工在
# 播报中静音了麦克风，但**播完解除静音的瞬间**，房间里的混响尾会被
# VAD 拾成一段"语音"；低头（视觉 absent）时的桌面杂音同理。文本相似度
# （looks_like_echo）抓不住它们 —— 失真后的文本和原句根本不像。

#: 播报结束后的混响尾（秒）：解除静音瞬间的余音还在房间里。
ECHO_REVERB_TAIL = 0.5
#: 播报开始前的提前量（秒）：VAD 分段起点可能比第一声略早。
ECHO_LEAD = 0.3

#: ChatterGuard 的**内容型**疑问词表。刻意不收「行不行 / 好不好 /
#: 是不是」这类 A-不-A 重叠尾 —— 现场杂音幻觉里它们高频出现
#: （'实话说一句不行行不行'），而真人问 A-不-A 通常是长句，
#: 会被 ChatterGuard 的长度线自然放行。
_GUARD_QUESTION_WORDS = (
    "什么", "怎么", "多少", "几点", "几号", "几岁", "星期几",
    "为什么", "为啥", "哪个", "哪些", "哪里", "哪儿", "谁", "吗", "呢",
)


def _guard_text_is_question(text: str) -> bool:
    """ChatterGuard 的疑问判定：文本里有没有**内容型**疑问词（子串包含）。"""
    return any(word in text for word in _GUARD_QUESTION_WORDS)


def window_overlaps_playback(seg_start: float, seg_end: float,
                             play_window: Optional[Tuple[float, float]]) -> bool:
    """一段音频（起止 monotonic 秒）是否落在最近一次播报的窗口里。

    重叠 ⇒ 这段"语音"只可能是外放声被收了回去（或与它混叠），
    无论识别出了什么都不可信。
    """
    if play_window is None:
        return False
    play_start, play_end = play_window
    return seg_start <= play_end + ECHO_REVERB_TAIL and seg_end >= play_start - ECHO_LEAD


class ChatterGuard:
    """连续杂音应答后的**冷静闸** —— 治"机器人自己说个不停"。

    现场链路：杂音 → STT 幻觉出短文本 → 走 LLM 得到一句澄清
    （"您这句话我没听清"）→ 外放 → 混响尾/下一阵杂音 → 再幻觉……
    一轮三轮自答，老人听到的是机器人对着发呆的自己说个不停。

    规则（刻意保守，只拦"几乎不可能是真人对话"的输入）：

    - 只有**短**（≤ ``max_chars`` 字）且**不带内容型疑问词**的识别才记账；
      疑问句（:func:`_guard_text_is_question`）和长句都是真实对话的证据
      → 清账放行。
    - ``window`` 秒内累计 ``threshold`` 次这样的应答 → 进入 ``cooldown``
      秒冷静期：期间短且无疑问的识别直接丢弃；疑问句/长句照常放行，
      且疑问句应答会**立即解除**冷静 —— 出现了真问句说明对面是真人在对话。
    """

    def __init__(self, window: float = 60.0, threshold: int = 2,
                 cooldown: float = 60.0, max_chars: int = 10) -> None:
        self.window = window
        self.threshold = threshold
        self.cooldown = cooldown
        self.max_chars = max_chars
        self._replied_times: list = []
        self._quiet_until = 0.0

    def on_reply(self, text: str, now: float) -> None:
        """一次应答真的发生了（回复已生成，即将播）。"""
        text = (text or "").strip()
        if _guard_text_is_question(text) or len(text) > self.max_chars:
            # 真实对话的证据：清账；冷静期一并解除（见类文档）。
            self._replied_times.clear()
            self._quiet_until = 0.0
            return
        self._replied_times.append(now)

    def should_drop(self, text: str, now: float) -> bool:
        """这条识别要不要静默丢弃（只拦短且无疑问的）。"""
        text = (text or "").strip()
        if _guard_text_is_question(text) or len(text) > self.max_chars:
            return False
        if now < self._quiet_until:
            return True
        recent = [t for t in self._replied_times if now - t <= self.window]
        if len(recent) >= self.threshold:
            self._quiet_until = now + self.cooldown
            self._replied_times.clear()
            return True
        return False


@dataclass
class LoopConfig:
    """语音循环的可调参数。"""

    #: 播报期间丢麦克风数据（半双工）。--barge-in 会把它关掉。
    half_duplex: bool = True
    #: 识别置信度/内容过滤（见 is_meaningful）
    filter_noise: bool = True
    #: 队列空转时的等待时长（秒）。0.25 让续句窗口的到期检查与
    #: 问句早放检查足够勤 —— 空转一拍最多 0.25 秒，体感上就是"说完就答"。
    read_timeout: float = 0.25
    #: **续句等待**：VAD 判出一句结束后，再多等这么久看老人有没有接着说。
    #:
    #: 背景：``end_silence_ms=700`` 时，老人说长句中间停顿换气（例如
    #: 「半夜醒了好几次……头有点昏……手还发麻」）会被切成两三个片段，
    #: 旧逻辑每个片段都立刻识别+应答，表现为**回答有间歇、一句话答两遍**。
    #: 现在先攒着：窗口内又来了新片段就拼成一段 PCM 统一识别、统一应答。
    #:
    #: 延迟权衡（实测被用户抱怨过"回答不迅速"）：
    #:   - 窗口 1.5s + 串行识别 ≈ 每句多等 2 秒 —— 太钝；
    #:   - 现在 1.2s，并且识别**在等待期间就并行开跑**（见
    #:     ``_start_speculation``），窗口到期时文本多半已就绪，
    #:     实际只比"不等待"慢约半秒 —— 这是覆盖 ~1.2s 换气停顿
    #:     （恰好是当初被切成两段的那种停顿）的最低代价。
    continuation_wait: float = 1.2
    #: 续句合并的音频总长上限（秒）：连续絮叨时攒够这么长就立刻送出，
    #: 不能无限拼接把识别拖死。
    continuation_max_total: float = 12.0


class Speaker:
    """串行化的播报器：合成 + 播放，同一时刻只说一句。

    独立出来是因为**有三个地方要说话**：语音循环的应答、主动关怀、
    以及启动问候。没有这一层，两句话会同时播出去叠在一起。

    ``on_busy_change`` 用来同步"机器人正在说话"这个状态给
    :mod:`core.proactive`（它据此不插嘴）和 :class:`Recorder`（半双工静音）。
    """

    def __init__(
        self,
        synthesizer: Optional[Synthesizer] = None,
        player=None,
        on_busy_change: Optional[Callable[[bool], None]] = None,
        half_duplex: bool = True,
        cache=None,
        cache_allow: Optional[Iterable[str]] = None,
        clarify_factory: Optional[Callable[[], "Synthesizer"]] = None,
    ) -> None:
        self.synthesizer = synthesizer or NullSynthesizer()
        self.player = player
        self.on_busy_change = on_busy_change
        self.half_duplex = half_duplex
        #: 预合成缓存（core.voice.speech_cache.SpeechCache）。留 None = 不缓存。
        #: 在线引擎每句要合成 3 秒，命中缓存就没有这段等待。
        self.cache = cache
        #: 允许写进缓存的文本白名单（通常是 ``dialogue.static_replies()``）。
        #: 留 None = 不设限、全都缓存（向后兼容）。
        #:
        #: 为什么要这道闸：缓存只有**会被反复说起**的固定句才划算 ——
        #: ``SpeechCache.prune()`` 的保留集就是 ``static_replies()``，所以
        #: 存了别的文本也只是等着被 prune 删掉，纯粹涨磁盘、命中率 0。
        #: 接上大模型之后每条回复都是模型现编的、**必然唯一**，这个问题会从
        #: "偶尔几个孤儿文件"变成"每句话都留一份"。与其指望调用方每处都记得，
        #: 不如把策略变成一个机械保证。（``{topic}`` 填充出来的动态模板句同理。）
        self._cache_allow = None if cache_allow is None else frozenset(
            text.strip() for text in cache_allow)

        self._lock = threading.Lock()
        self._busy = threading.Event()
        #: 异步播报（应声词，见 :meth:`say_async`）的簿记。
        #: **单独一把锁**，绝不能用 ``_lock`` —— 理由见 :meth:`_join_async`。
        self._async_lock = threading.Lock()
        self._async_busy = False
        self._async_thread: Optional[threading.Thread] = None
        self._last_text = ""
        #: 最近一次**真的出了声**的播报窗口 (start, end)（monotonic 秒）。
        #: 时间窗回声过滤（见 :data:`ECHO_REVERB_TAIL`）用它对齐"机器人
        #: 什么时候在说话"和"麦克风这段音频是什么时候拾的"。
        self._last_play_window: Optional[Tuple[float, float]] = None
        self._tempdir = tempfile.mkdtemp(prefix="b_voice_")
        self._counter = 0

        # ---- 「听不清」clarify 模式 ----
        # 老人说"听不清""大声点""再说一遍"时，提高音量 + 放慢语速 +
        # 复述上一句。三项由 main.py 的 handle_chat 在 clarify 意图命中时触发。
        #
        # 音量增益直接作用在 PCM 上（Player.play_wav 的 volume 参数），
        # 对所有引擎都生效，零额外开销。
        #
        # 语速放慢需要一个**更慢的合成器实例**——各引擎的 speed 在构造时
        # 固定（SAPI 的 Rate 写进 PowerShell 脚本，豆包/edge-tts 的 speed
        # 是 __init__ 参数），运行时改不掉。所以 clarify_factory 惰性
        # 构造第二个实例，只在第一次触发 clarify 时付一次启动代价。
        # 没传 factory = 不支持语速放慢（只提音量）。
        self._clarify_factory = clarify_factory
        self._clarify_synth: Optional[Synthesizer] = None
        self._clarify_volume: float = 1.0          # 1.0 = 原始音量
        self._clarify_active: bool = False

    # ------------------------------------------------------------------
    # 「听不清」clarify 模式
    # ------------------------------------------------------------------

    def set_clarify(self, volume_boost: float = 0.0,
                    speed_cut: float = 0.0) -> None:
        """进入 clarify 模式：音量放大、语速放慢。

        ``volume_boost`` 是**增量**（0.20 = +20%），内部转成乘数 1.0+boost。
        ``speed_cut`` 是**降幅**（0.30 = -30%），仅用于决定要不要惰性构造
        # 慢速合成器——实际语速由 factory 构造时用更小的 speed 参数决定。

        惰性构造：第一次调用时才花钱建第二个合成器。建不出来（factory
        没传 / 引擎不支持）就只提音量，不报错 —— 提音量本身就是"听不清"
        时最直接的改善。
        """
        self._clarify_volume = max(1.0, 1.0 + max(0.0, volume_boost))
        self._clarify_active = True

        if speed_cut > 0.0 and self._clarify_synth is None and self._clarify_factory is not None:
            try:
                self._clarify_synth = self._clarify_factory()
                logger.info("已构造慢速合成器用于 clarify 模式（引擎 %s）",
                            getattr(self._clarify_synth, "name", "?"))
            except Exception:
                logger.exception("构造慢速合成器失败，clarify 模式只提音量")
                self._clarify_synth = None

    def clear_clarify(self) -> None:
        """退出 clarify 模式，恢复正常音量与语速。"""
        self._clarify_volume = 1.0
        self._clarify_active = False

    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    @property
    def engine_name(self) -> str:
        return getattr(self.synthesizer, "name", "?")

    @property
    def last_text(self) -> str:
        """最近一次播报的内容。回声去重要用它。"""
        return self._last_text

    @property
    def last_play_window(self) -> Optional[Tuple[float, float]]:
        """最近一次播报的起止时刻（monotonic 秒）；从未播过返回 None。"""
        return self._last_play_window

    def say(self, text: str) -> bool:
        """合成并播放一句话。**阻塞到说完**，返回是否真的发出了声音。

        阻塞是刻意的：调用方需要知道"说完了"才能恢复采集。
        失败返回 False（比如没装 TTS），但**不抛异常** ——
        文字回复和表情照常工作。

        进门时如果上一句**应声词**还在播，会先等它说完再排队 ——
        见 :meth:`say_async`。
        """
        return self._say(text, join_async=True)

    def say_all(self, parts) -> bool:
        """按顺序念几段话，但**只开一个忙碌窗口**。

        用于"回复 + 追加关怀"这种一句话分两段合成的场景。

        **为什么不直接连着调两次 say()：**

        1. 每调一次 ``say()`` 都会 set/clear 一次 ``_busy`` 并各触发一次
           ``on_busy_change``。两段之间那一瞬间，主动关怀的 ``speaking``
           门禁和半双工静音都是**开**的 —— 正好是机器人刚说完"……然后呢？"
           到"我看您今天心情不太好"之间那道缝，机器会在这时候跑去听自己。
        2. ``_last_text`` 会被第二段覆盖，回声去重就认不出第一段的回声了
           （第二段播完的瞬间，环境里还飘着第一段的声音）。
           所以最后存的是**整段**。

        **为什么是分两次合成、而不是拼成一句：** 拼起来会让
        :class:`SpeechCache` 必然不命中 —— 那句关怀语本来是预合成好的，
        拼进去就得连回复一起重新联网合成（本机实测 +1.2 秒），
        正好把预合成想省的时间全花回去。所以这里宁可多进一次
        :meth:`_speak_locked`，也不要拼。

        ``parts`` 里的空串会被跳过（``follow_up`` 为空是常态）。
        返回是否至少真的发出了**一段**声音。
        """
        texts = [p.strip() for p in (parts or ()) if isinstance(p, str) and p.strip()]
        if not texts:
            return False

        # 和 say() 一样，先等应声词说完再排队（必须在取 _lock 之前）。
        self._join_async()

        with self._lock:
            # 用空格拼 —— looks_like_echo 靠这个分隔符把两段拆回去比。
            # 中文回复本身不含空格，所以这个分隔符是安全的。
            self._last_text = " ".join(texts)
            self._busy.set()
            if self.on_busy_change:
                self.on_busy_change(True)

            spoke = False
            try:
                # 逐段走同一套缓存/合成/流式判断。一段失败不影响后面几段 ——
                # 回复出声了、关怀没出声，也远比整句吞掉好。
                for text in texts:
                    try:
                        if self._speak_locked(text):
                            spoke = True
                    except Exception:
                        logger.exception("播报「%s」失败，继续念下一段", _brief(text))
                return spoke
            finally:
                self._busy.clear()
                if self.on_busy_change:
                    self.on_busy_change(False)

    def say_async(self, text: str) -> bool:
        """异步说一句：立刻返回，播报在后台线程里做。

        用途只有一个 —— **应声词**（当前 :data:`core.dialogue.ACK_REPLIES`
        为空，这条通路整体静默，函数仍保留以供恢复；空文本直接返回 False）。
        设计意图：从用户说完到机器人给出正式回复，大模型那一段本机实测
        0.9~1.0 秒；干等着就是"说完之后机器沉默"，先垫一句体感上完全不同。

        两个关键性质：

        1. **和 :meth:`say` 排在同一条队上，应声词一定在前。** 这一步是
           自动的：正式回复走 :meth:`say`，它进门先 :meth:`_join_async`。
           **不能靠"两个线程抢同一把锁"来排序** —— ``threading.Lock``
           不保证先来先得，正式回复完全可能抢先拿到锁，于是老人听到的是
           "回答……嗯，我听着呢"。
        2. **同一时刻只允许一句异步播报。** 上一句还没说完又来一句时直接
           丢掉这一次，否则应声词会连成一片："嗯，我想想，嗯，我想想"。

        返回 False 表示"没排上队"（空文本 / 上一句还在说），**不是失败**。
        """
        text = (text or "").strip()
        if not text:
            return False

        with self._async_lock:
            if self._async_busy:
                logger.debug("上一句异步播报还没说完，丢掉这次的：%s", _brief(text))
                return False
            self._async_busy = True
            thread = threading.Thread(
                target=self._say_async_worker, args=(text,),
                name="voice-say-async", daemon=True)
            self._async_thread = thread
            thread.start()
        return True

    def _say_async_worker(self, text: str) -> None:
        try:
            # join_async=False：worker 本身就是被等的那一方，再等自己会死锁。
            self._say(text, join_async=False)
        except Exception:
            logger.exception("异步播报失败")
        finally:
            with self._async_lock:
                self._async_busy = False
                self._async_thread = None

    def _join_async(self) -> None:
        """等上一句异步播报说完。**必须在取 ``_lock`` 之前调用。**

        顺序反了就是死锁：worker 里也要拿 ``_lock``，先取 ``_lock`` 再等
        worker，两边就永远等下去。同理 ``_async_lock`` 只用来读一个引用，
        读完立刻放 —— 拿着它去 join 会卡住 worker 的收尾（worker 的
        finally 也要拿这把锁）。
        """
        with self._async_lock:
            thread = self._async_thread
        if thread is not None:
            thread.join()

    def _say(self, text: str, *, join_async: bool) -> bool:
        text = (text or "").strip()
        if not text:
            return False

        if join_async:
            self._join_async()

        with self._lock:                      # 串行化：同一时刻只能说一句
            self._last_text = text
            self._busy.set()
            if self.on_busy_change:
                self.on_busy_change(True)

            try:
                return self._speak_locked(text)
            finally:
                self._busy.clear()
                if self.on_busy_change:
                    self.on_busy_change(False)

    def _speak_locked(self, text: str) -> bool:
        wav_path = None
        try:
            # clarify 模式下跳过缓存：clarify 的回复是运行时构造的确认句，
            # 永远不会重复，缓存它只是涨磁盘。跳过它也免去了"从缓存拿出
            # 正常音量音频再想提音量"的麻烦——缓存里存的是原始音量。
            if self._clarify_active or self.cache is not None:
                if not self._clarify_active and self.cache is not None:
                    # 先查预合成缓存。命中就直接播，一步网络都不用等 ——
                    # 这是把「先沉默三秒再开口」抹掉的地方。
                    cached = self.cache.lookup(text)
                    if cached is not None:
                        return self._play_locked(text, cached, from_cache=True)

            # clarify 模式跳过流式：流式播报无法对 PCM 做音量增益（边收
            # 边播没有"拿到整段再缩放"的机会），也无法用慢速合成器（流式
            # 接口挂在主合成器上）。clarify 优先清晰度，牺牲这一句的延迟。
            streamed = self._try_stream_locked(text)
            if streamed is not None:
                return streamed

            self._counter += 1
            wav_path = os.path.join(self._tempdir, f"say_{self._counter}.wav")

            # 每次用**独立**的临时文件路径：固定路径会让并发/连续调用
            # 互相截断（后一次覆盖前一次的文件，而前一次还在播）。
            synth = self._clarify_synth if (self._clarify_active and self._clarify_synth is not None) \
                else self.synthesizer
            t0 = time.monotonic()
            if not synth.synthesize_wav(text, wav_path):
                # ⚠️ 这条必须是 WARNING，不能是 DEBUG。
                #    合成失败意味着这句话**一个字都没说出去**，而 DEBUG 在默认的
                #    INFO 级别下完全不可见 —— 表现出来就是"机器人突然哑了，
                #    但日志里什么都没有"，是最难查的一类故障（曾经就是这个样子）。
                #    clarify 模式下慢速合成器可能没配好 → 回退到主合成器再试一次。
                if self._clarify_active and self._clarify_synth is not None:
                    logger.debug("慢速合成器失败，回退到主合成器重试：%s", _brief(text))
                    if not self.synthesizer.synthesize_wav(text, wav_path):
                        logger.warning("语音合成失败，这句话只进了文字通道、没有声音：%r",
                                       _brief(text))
                        return False
                else:
                    logger.warning(
                        "语音合成失败，这句话只进了文字通道、没有声音：%r", _brief(text))
                    return False
            synth_ms = (time.monotonic() - t0) * 1000

            # ⚠️ **兜底产出的那句不能存。** 缓存键记的是主引擎（name|voice|speed），
            #    把 SAPI 的机械音存进"豆包"的键下，等网络恢复后这句话就永远是
            #    错的音色了 —— 而且是**永久**的，因为缓存命中不会再走合成。
            #    宁可下次重合成，也不要缓存一个错音色。
            if (self.cache is not None and self._cache_allowed(text)
                    and not self._clarify_active
                    and not getattr(synth, "used_fallback", False)):
                self.cache.store(text, wav_path)

            return self._play_locked(text, wav_path, synth_ms=synth_ms)
        except Exception:
            logger.exception("播报失败")
            return False
        finally:
            # 只删自己合成的临时文件 —— 缓存文件是共享的，删了下次还得重合成。
            if wav_path is not None:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _cache_allowed(self, text: str) -> bool:
        """这句话值不值得缓存。没设白名单（None）时一律缓存。"""
        if self._cache_allow is None:
            return True
        return text.strip() in self._cache_allow

    # ------------------------------------------------------------------
    # 边收边播（流式合成）
    # ------------------------------------------------------------------

    def _try_stream_locked(self, text: str) -> Optional[bool]:
        """试一次边收边播。``None`` = "条件不成立 / 一个字都没出声，请落回整句合成"。

        四种情况返回 None，每一种都对应一条**本来就走不通**的路：

          1. 没有播放设备（``--no-play``）—— 没有播放端就无所谓"边收边播"。
             注意这条走 None 之后会落到 ``_play_locked`` 的"已合成但未播放"
             分支，那正是 ``--no-play`` 期望的行为。
          2. 引擎不支持流式（SAPI 是本地 37ms 合成，根本不需要；edge-tts 与
             豆包的"落盘"路径也没实现）。
          3. 设备采样率和引擎能给的档位没有交集 —— 流式不能重采样（见
             ``Player.play_pcm_stream``），谈不拢就只能走整句合成。
          4. 试了但**一块音频都没吐出来**就失败 —— 这是干净的失败，落回
             整句合成那条路，它还带着本地 SAPI 的逐句兜底。

        对第 4 种情况的**反面**特别要紧：一旦已经出声，就**绝不返回 None**。
        返回 None 会让调用方把同一句话再合成一遍、再念一遍。

        clarify 模式也返回 None：流式路径既不能做 PCM 音量增益（拿到的是
        逐块、不是整段），又用的是主合成器（不是慢速的 clarify 合成器）。
        clarify 优先清晰度，走整句合成路径才能同时提音量 + 放慢语速。
        """
        if self._clarify_active:
            return None
        if self.player is None:
            return None
        if not getattr(self.synthesizer, "supports_streaming", False):
            return None
        stream_fn = getattr(self.synthesizer, "synthesize_pcm_stream", None)
        if stream_fn is None:
            return None

        rates = getattr(self.synthesizer, "stream_rates", ())
        if not rates:
            return None
        rate = self.player.stream_rate(rates)
        if rate is None:
            logger.debug("输出设备与合成引擎的采样率没有交集，这句走整句合成")
            return None

        return self._stream_locked(text, stream_fn, rate)

    def _stream_locked(self, text: str, stream_fn, rate: int) -> Optional[bool]:
        """真的去边收边播。调用方已持有锁。"""
        if self.half_duplex:
            try:
                self.player.stop()        # 打断上一次可能的播放
            except Exception:
                pass

        # 用一个盒子把"出没出过声""为什么断的"从生成器里带出来 ——
        # Player.play_pcm_stream 会把异常吃掉并返回 False，光看返回值
        # 分不清"设备没开起来"和"服务端一块都没给"。
        state = {"started": False, "error": None}

        def chunks():
            try:
                for piece in stream_fn(text, rate):
                    state["started"] = True
                    yield piece
            except Exception as exc:       # noqa: BLE001 —— 原样往上抛，只顺手记一笔
                state["error"] = exc
                raise

        t0 = time.monotonic()
        ok = bool(self.player.play_pcm_stream(chunks(), rate))
        elapsed = (time.monotonic() - t0) * 1000
        if ok:
            self._last_play_window = (t0, time.monotonic())

        if not state["started"]:
            logger.warning("流式合成一块音频都没出来就失败，改用整句合成：%s（%s）",
                           _brief(text), state["error"] or "无音频")
            return None

        if state["error"] is not None:
            # 中途断了（实测是豆包服务端长句合成中途停发音频块，等满
            # 30 秒 socket 超时）。返回 None 落回**整句合成重念一遍**：
            # 听到一半戛然而止比重听一遍糟糕得多 —— 老人只会觉得机器人
            # 坏了，不会意识到该要求"再说一遍"。短句断流概率极低，
            # 真断了重念一次的成本可接受。
            logger.warning("流式合成中途断了，整句重念一遍：%s（%s）",
                           _brief(text), state["error"])
            return None
        logger.info("流式播报：%s（首个字节到播完共 %.0fms，%dHz）",
                    _brief(text), elapsed, rate)
        return ok

    def _play_locked(self, text: str, wav_path: str,
                     synth_ms: float = 0.0, from_cache: bool = False) -> bool:
        """把一份已有的 WAV 播出去。调用方已持有锁。"""
        if self.player is None:
            # 有 WAV 但没播放设备（--no-play）。合成成功 ≠ 说了话，
            # 日志要能区分"没播"和"播了没人听见"。
            logger.info(
                "已合成但未播放（无播放设备）：%s（%.0fms 合成）", _brief(text), synth_ms)
            return True

        if self.half_duplex:
            try:
                self.player.stop()        # 打断上一次可能的播放
            except Exception:
                pass

        # clarify 模式下提音量：volume=1.0 时直通，>1.0 增益（钳位在 int16）。
        volume = self._clarify_volume if self._clarify_active else 1.0
        t1 = time.monotonic()
        ok = bool(self.player.play_wav(wav_path, volume=volume))
        play_ms = (time.monotonic() - t1) * 1000
        if ok:
            self._last_play_window = (t1, time.monotonic())
        if from_cache:
            logger.info("命中预合成缓存，无需合成：%s", _brief(text))
        if self._clarify_active and volume > 1.0:
            logger.info("clarify 模式：音量 ×%.2f，%s", volume, _brief(text))
        self._log_playback(text, wav_path, ok, synth_ms, play_ms)
        return ok

    @staticmethod
    def _log_playback(text: str, wav_path: str, ok: bool,
                      synth_ms: float, play_ms: float) -> None:
        """记一条能判断"到底出没出声"的日志。

        为什么要记**播放耗时**而不是只记一句"播放成功"：
        ``play_wav`` 是同步的，正常出声时它会阻塞到放完，于是 ``play_ms``
        约等于音频本身的长度；而两种"假成功"都会让这个数字露馅 ——

          * 声音进了虚拟声卡（本机装着 ToDesk Virtual Audio）：
            数据被丢进黑洞，调用**立刻**返回；
          * 设备以远高于实际的速率"播"完。

        两者都表现为"播放成功但听不见"，且都**不报错**。
        只记"成功"的话，这两种情况和真的出声在日志里长得一模一样。
        """
        seconds = _wav_seconds(wav_path)
        if not ok:
            logger.warning(
                "播报失败：%s（合成 %.0fms，音频长 %.1fs）—— "
                "用 --list-audio 看设备，默认输出可能是虚拟声卡",
                _brief(text), synth_ms, seconds)
            return

        # 播放耗时明显短于音频长度 => 大概率没真的出声（虚拟声卡/设备不对）。
        if seconds >= 0.3 and play_ms < seconds * 1000 * 0.5:
            logger.warning(
                "已播报但耗时异常：%s（音频长 %.1fs，却只用了 %.0fms）—— "
                "数据可能进了虚拟声卡，人耳听不到。用 --audio-out <名字片段> 指定真音箱",
                _brief(text), seconds, play_ms)
        else:
            logger.info(
                "已播报：%s（音频 %.1fs，合成 %.0fms，播放 %.0fms）",
                _brief(text), seconds, synth_ms, play_ms)

    def stop(self) -> None:
        """打断当前播报。从别的线程调用。"""
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        try:
            self.synthesizer.close()
        except Exception:
            pass
        # clarify 模式的慢速合成器也要关：SAPI 是常驻 PowerShell 子进程，
        # 不关会变成孤儿进程；edge-tts / 豆包也该释放连接。
        if self._clarify_synth is not None and self._clarify_synth is not self.synthesizer:
            try:
                self._clarify_synth.close()
            except Exception:
                pass
        try:
            for name in os.listdir(self._tempdir):
                try:
                    os.unlink(os.path.join(self._tempdir, name))
                except OSError:
                    pass
            os.rmdir(self._tempdir)
        except OSError:
            pass


class VoiceLoop:
    """后台线程：不断从麦克风读数据、切句、识别、交给对话逻辑。"""

    def __init__(
        self,
        on_text: Callable[[str], object],
        speaker: Speaker,
        recognizer: Optional[Recognizer] = None,
        recorder=None,
        vad: Optional[VadConfig] = None,
        config: Optional[LoopConfig] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.on_text = on_text
        self.speaker = speaker
        self.recognizer = recognizer or NullRecognizer()
        self.recorder = recorder
        self.vad = vad or VadConfig()
        self.config = config or LoopConfig()
        #: 单调时钟，默认 time.monotonic；测试可注入假时钟驱动续句等待。
        self._clock = clock or time.monotonic

        self._endpointer = Endpointer(self.vad)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._errors = 0
        #: 语音功能是否还活着。连续出错太多会置假，main.py 据此提示降级。
        self._alive = False

        # ---- 续句合并的缓冲（见 LoopConfig.continuation_wait）----
        #: 已切出、但还在"可能有下半句"窗口里的 PCM 片段。
        self._pending_pcm: list = []
        self._pending_rate: int = 0
        #: 到点（单调时钟）就把缓冲拼成一段送出；新片段到来会续期。
        self._pending_deadline: float = 0.0

        # ---- 投机识别：等待窗口里就把 STT 跑掉，别让用户白等 ----
        # 窗口本来就在等"有没有下半句"，这段时间 CPU 是闲的 —— 让识别
        # 提前在后台线程进行，到期直接用现成的文本。延迟构成从
        # "窗口 + 识别" 变成 "max(窗口, 识别)"。
        self._spec_lock = threading.Lock()
        self._spec_thread: Optional[threading.Thread] = None
        #: (线程开始时缓冲的字节数, 识别文本)；字节数对不上=过期。
        self._spec_result: Optional[tuple] = None

        # ---- 杂音冷静闸：连续杂音应答后短暂闭嘴（见 ChatterGuard）----
        #: 用可注入的时钟，测试能用假时钟驱动冷静期。
        self._chatter = ChatterGuard()

    # ------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._alive

    def start(self) -> bool:
        """启动后台线程。没有麦克风或识别引擎时返回 False。

        返回 False 不是错误 —— main.py 会照常运行，只是没有语音输入。
        """
        if self.recorder is None:
            logger.warning("没有可用的录音设备，语音输入未启动")
            return False

        if not self.recorder.start():
            logger.warning(
                "打开麦克风失败，语音输入未启动。\n"
                "  排查：python main.py --list-audio 看设备列表，\n"
                "        再用 --audio-in \"设备名片段\" 显式指定（不要用设备序号）。"
            )
            return False

        self._alive = True
        self._thread = threading.Thread(target=self._run, name="voice-loop", daemon=True)
        self._thread.start()
        logger.info("语音输入已启动（识别引擎 %s，VAD 帧 %dms）",
                    getattr(self.recognizer, "name", "?"), self.vad.frame_ms)
        return True

    def stop(self) -> None:
        """停止循环并关闭录音。必须幂等，且要能被 join 住。"""
        self._stop.set()
        self._alive = False
        if self.recorder is not None:
            self.recorder.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    # ------------------------------------------------------------------

    def _run(self) -> None:
        """线程主体。带退避重试，连续失败多次就整体停用。"""
        while not self._stop.is_set():
            try:
                self._loop_once()
                self._errors = 0
            except Exception:
                self._errors += 1
                logger.exception("语音循环出错（第 %d 次）", self._errors)
                if self._errors >= MAX_CONSECUTIVE_ERRORS:
                    # 明确地把"语音死了"喊出来。
                    # 以前这里是静默吞异常，表现为"用了三分钟之后语音就不灵了，
                    # 日志里只有一行看不懂的堆栈"。
                    self._alive = False
                    logger.error(
                        "语音功能已停用（连续 %d 次出错），回退到键盘 / 8002 文本通道。"
                        "对话本身不受影响。", self._errors,
                    )
                    return
                time.sleep(ERROR_BACKOFF * self._errors)

    def _loop_once(self) -> None:
        """读一块音频 → 切句 → 续句合并 → 到点统一处理。"""
        block = self.recorder.read(timeout=self.config.read_timeout)
        if block is not None:
            for segment in self._endpointer.push(block):
                if self._stop.is_set():
                    return
                self._enqueue_segment(segment)

        # 这一拍即使没读到新数据（超时 None）也要检查缓冲：
        # 句子可能正在"续句窗口"里等到期。没有这句，flush 会被推迟到
        # 下一块带数据的音频到来 —— 安静环境里就是好几秒甚至永远。
        if self._pending_pcm:
            if self._clock() >= self._pending_deadline:
                self._flush_pending()
            else:
                # 窗口还没满，但投机识别若已断定"这是句完整的问句"，
                # 就别让用户多等 —— 问句要的是立刻回答（见
                # _looks_like_complete_question）。
                self._maybe_early_flush()

    def _maybe_early_flush(self) -> None:
        """投机结果像完整问句 → 立刻送出，不等满续句窗口。

        「现在几点了」「天气怎么样」这类话，老人说完就是在等答案，
        多等一秒都是迟钝；而讲故事的片段（"头有点昏"）不带疑问尾巴，
        会继续等下文、走合并。投机结果与当前缓冲字节数对不上（窗口内
        又来了新片段）时不早放 —— 那说明用户确实还在接着说。
        """
        if not self._pending_pcm:
            return
        with self._spec_lock:
            spec = self._spec_result
        if not spec or spec[0] != sum(len(p) for p in self._pending_pcm):
            return
        if _looks_like_complete_question(spec[1]):
            logger.info("投机文本像完整问句，不等续句窗口直接送出：%r", spec[1])
            self._flush_pending()

    def _enqueue_segment(self, segment: SpeechSegment) -> None:
        """把切出的一句话先放进续句缓冲，而不是立刻识别。

        ``continuation_wait`` 秒内若又有新片段（老人停顿换气后接着说），
        两段会拼成一段 PCM 只识别、只应答一次 —— 这是修"长句被切成
        好几段、回答两遍"的地方。
        """
        # 同一台录音设备采样率不可能变；真变了说明设备重连，旧缓冲不能拼。
        if self._pending_pcm and segment.sample_rate != self._pending_rate:
            logger.warning("录音采样率变化（%s→%s），先送出旧缓冲",
                           self._pending_rate, segment.sample_rate)
            self._flush_pending()

        self._pending_pcm.append(segment.pcm)
        self._pending_rate = segment.sample_rate
        self._pending_deadline = self._clock() + self.config.continuation_wait
        self._start_speculation()

        # 总长封顶：持续说不停时，攒够上限立刻送，不无限拼接。
        if self._pending_duration_ms() >= self.config.continuation_max_total * 1000:
            logger.debug("续句合并达到 %.0fs 上限，提前送出",
                         self.config.continuation_max_total)
            self._flush_pending()

    def _start_speculation(self) -> None:
        """后台线程里把**当前缓冲**的识别先跑起来（投机识别）。

        规则：上一个投机线程还活着就不再起 —— 两个线程同时调同一个
        识别引擎是不允许的。此时缓冲已经变了，旧结果必然过期，
        flush 时会同步重识一次（概率很小：识别一秒音频远快于 1.2s 窗口）。
        """
        if self._spec_thread is not None and self._spec_thread.is_alive():
            return
        pcm = b"".join(self._pending_pcm)
        if not pcm or self._pending_rate <= 0:
            return

        def run(pcm: bytes = pcm, rate: int = self._pending_rate,
                expect: int = len(pcm)) -> None:
            try:
                text = self.recognizer.transcribe(pcm, rate)
            except Exception:
                # 投机失败不算错：flush 时会同步重识一遍，结果照常出。
                logger.debug("投机识别失败，flush 时同步重试", exc_info=True)
                return
            with self._spec_lock:
                self._spec_result = (expect, text)

        with self._spec_lock:
            self._spec_result = None
        self._spec_thread = threading.Thread(
            target=run, name="voice-spec-stt", daemon=True)
        self._spec_thread.start()

    def _pending_duration_ms(self) -> float:
        """缓冲音频总时长（16bit 单声道：每采样 2 字节）。"""
        if not self._pending_pcm or not self._pending_rate:
            return 0.0
        total_bytes = sum(len(piece) for piece in self._pending_pcm)
        return total_bytes / 2.0 / self._pending_rate * 1000.0

    def _flush_pending(self) -> bool:
        """把缓冲里的一段或多段拼成连续 PCM，统一识别一次。

        :return: 是否真的送出了。投机线程还没收工时会**推迟这一拍**
                 （返回 False、缓冲原样保留，下个节拍重试）—— 识别引擎
                 不允许两个线程同时用，宁可晚半拍也不能撞车。
        """
        if not self._pending_pcm:
            return False

        # 先等投机线程收工（串行化引擎访问）。等待窗口通常早已覆盖
        # 识别耗时，这里多半是瞬时返回；引擎自己的超时保证不会永久挂住。
        thread = self._spec_thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("投机识别迟迟未返回，本轮送出推迟一拍")
                return False

        with self._spec_lock:
            spec, self._spec_result = self._spec_result, None
        self._spec_thread = None

        pcm = b"".join(self._pending_pcm)
        rate = self._pending_rate
        pieces = len(self._pending_pcm)
        duration_ms = len(pcm) / 2.0 / rate * 1000.0

        self._pending_pcm = []
        self._pending_rate = 0
        self._pending_deadline = 0.0

        if pieces > 1:
            logger.info("续句合并：%d 个停顿片段拼成 %.0fms 统一识别，避免重复应答",
                        pieces, duration_ms)

        # 投机结果对得上当前缓冲就直接用（省掉一次串行识别）；对不上
        # （窗口内又来了新片段且线程当时还活着）则同步重识。
        text = spec[1] if spec and spec[0] == len(pcm) and spec[1] else None
        self._process_pcm(pcm, rate, duration_ms, text=text)
        return True

    def _handle_segment(self, segment: SpeechSegment) -> None:
        """直接处理**单个**片段（绕过续句缓冲）。

        主循环已经统一走 :meth:`_enqueue_segment`；保留这个入口是给
        flush 收尾与外部调用用的。
        """
        self._process_pcm(segment.pcm, segment.sample_rate, segment.duration_ms)

    def _process_pcm(self, pcm: bytes, sample_rate: int, duration_ms: float,
                     text: Optional[str] = None) -> None:
        """一段最终音频：识别 → 过滤 → 对话 → 播报。整段只走一次。

        ``text`` 是投机识别已经算好的现成结果；缺省时现场识别。
        """
        started = time.monotonic()
        if text is None:
            text = self.recognizer.transcribe(pcm, sample_rate)
        elapsed = int((time.monotonic() - started) * 1000)

        if not text:
            # INFO 而不是 debug：老人明明说了话却没人理，是必须能在日志里
            # 一眼看到的现场 —— 否则"说了话机器人不回答"永远查不出在哪层丢的。
            logger.info("识别为空（音频 %.0fms，耗时 %dms）",
                        duration_ms, elapsed)
            return

        # 过滤 1：噪声上 STT 常吐单个标点/语气词
        if self.config.filter_noise and not is_meaningful(text):
            logger.info("识别结果无实际内容，丢弃：%r", text)
            return

        # 过滤 2：兜底的回声去重（半双工没盖住时生效）
        if looks_like_echo(text, self.speaker.last_text):
            logger.info("判定为自身回声，丢弃：%r", text)
            return

        # 过滤 3：时间窗回声 —— 只在**全双工**（--barge-in）时启用。
        # 全双工播报不静音麦克风，回声段必然与播报窗口重叠，时间对齐
        # 是唯一可靠的判据（远场失真后的文本和原句完全不像）。
        # 半双工（默认）播报中收音本来就静着，回声只剩解除静音瞬间的
        # 混响尾 —— 而"播完 1 秒内的段"也可能是老人最快的接话，
        # 时间窗区分不了这两者，误杀真接话的代价更高，交给过滤 4 兜底。
        if (not self.config.half_duplex and window_overlaps_playback(
                started - duration_ms / 1000.0, started,
                self.speaker.last_play_window)):
            logger.info("识别音频段与播报窗口重叠，判为回声丢弃：%r", text)
            return

        # 过滤 4：连续杂音冷静 —— 60 秒内两轮"短且无疑问"的应答后，
        # 再来的短杂音不再应答（治"机器人对着发呆的自己说个不停"）。
        # 疑问句 / 长句是真实对话的证据，照常放行并解除冷静。
        if self._chatter.should_drop(text, self._clock()):
            logger.info("嘈杂冷静期内丢弃疑似环境杂音：%r", text)
            return

        logger.info("语音识别：%r（音频 %.0fms，耗时 %dms）",
                    text, duration_ms, elapsed)

        try:
            reply = self.on_text(text)
        except Exception:
            logger.exception("对话处理失败")
            return

        self._chatter.on_reply(text, self._clock())
        self._speak_reply(reply)

    def _speak_reply(self, reply) -> None:
        """把 ``on_text`` 返回的东西念出来：内容一段，追加的关怀一段。

        ⚠️ ``on_text`` 的类型契约是 ``object`` —— 测试里传的是
        ``mock.Mock(reply="…")``。**Mock 会自动造出任何属性**，所以
        ``getattr(reply, "follow_up", "")`` 拿到的不是默认值，而是一个新的
        Mock 对象，于是会多念一段、``len(player.played) == 1`` 这类断言就挂。
        必须用 ``isinstance(..., str)`` 卡死。
        """
        parts = []
        for attr in ("reply", "follow_up"):
            value = getattr(reply, attr, None)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        if parts:
            self.speaker.say_all(parts)

    # ------------------------------------------------------------------

    def say(self, text: str) -> bool:
        """让机器人说话。供主动关怀等外部逻辑调用。"""
        return self.speaker.say(text)

    def interrupt(self) -> None:
        """打断当前播报（barge-in 模式下由语音循环调用）。"""
        self.speaker.stop()
