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
    """
    if not text or not last_reply:
        return False
    ratio = difflib.SequenceMatcher(None, text, last_reply).ratio()
    return ratio >= threshold


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


@dataclass
class LoopConfig:
    """语音循环的可调参数。"""

    #: 播报期间丢麦克风数据（半双工）。--barge-in 会把它关掉。
    half_duplex: bool = True
    #: 识别置信度/内容过滤（见 is_meaningful）
    filter_noise: bool = True
    #: 队列空转时的等待时长（秒）
    read_timeout: float = 0.5


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
        self._last_text = ""
        self._tempdir = tempfile.mkdtemp(prefix="b_voice_")
        self._counter = 0

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

    def say(self, text: str) -> bool:
        """合成并播放一句话。**阻塞到说完**，返回是否真的发出了声音。

        阻塞是刻意的：调用方需要知道"说完了"才能恢复采集。
        失败返回 False（比如没装 TTS），但**不抛异常** ——
        文字回复和表情照常工作。
        """
        text = (text or "").strip()
        if not text:
            return False

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
            # 先查预合成缓存。命中就直接播，一步网络都不用等 ——
            # 这是把「先沉默三秒再开口」抹掉的地方。
            if self.cache is not None:
                cached = self.cache.lookup(text)
                if cached is not None:
                    return self._play_locked(text, cached, from_cache=True)

            self._counter += 1
            wav_path = os.path.join(self._tempdir, f"say_{self._counter}.wav")

            # 每次用**独立**的临时文件路径：固定路径会让并发/连续调用
            # 互相截断（后一次覆盖前一次的文件，而前一次还在播）。
            t0 = time.monotonic()
            if not self.synthesizer.synthesize_wav(text, wav_path):
                # ⚠️ 这条必须是 WARNING，不能是 DEBUG。
                #    合成失败意味着这句话**一个字都没说出去**，而 DEBUG 在默认的
                #    INFO 级别下完全不可见 —— 表现出来就是"机器人突然哑了，
                #    但日志里什么都没有"，是最难查的一类故障（曾经就是这个样子）。
                logger.warning(
                    "语音合成失败，这句话只进了文字通道、没有声音：%r", _brief(text))
                return False
            synth_ms = (time.monotonic() - t0) * 1000

            # ⚠️ **兜底产出的那句不能存。** 缓存键记的是主引擎（name|voice|speed），
            #    把 SAPI 的机械音存进"豆包"的键下，等网络恢复后这句话就永远是
            #    错的音色了 —— 而且是**永久**的，因为缓存命中不会再走合成。
            #    宁可下次重合成，也不要缓存一个错音色。
            if self.cache is not None and self._cache_allowed(text) and not getattr(
                    self.synthesizer, "used_fallback", False):
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

        t1 = time.monotonic()
        ok = bool(self.player.play_wav(wav_path))
        play_ms = (time.monotonic() - t1) * 1000
        if from_cache:
            logger.info("命中预合成缓存，无需合成：%s", _brief(text))
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
    ) -> None:
        self.on_text = on_text
        self.speaker = speaker
        self.recognizer = recognizer or NullRecognizer()
        self.recorder = recorder
        self.vad = vad or VadConfig()
        self.config = config or LoopConfig()

        self._endpointer = Endpointer(self.vad)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._errors = 0
        #: 语音功能是否还活着。连续出错太多会置假，main.py 据此提示降级。
        self._alive = False

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
        """读一块音频 → 切句 → 逐句处理。"""
        block = self.recorder.read(timeout=self.config.read_timeout)
        if block is None:
            return                            # 超时，回头检查停止标志

        for segment in self._endpointer.push(block):
            if self._stop.is_set():
                return
            self._handle_segment(segment)

    def _handle_segment(self, segment: SpeechSegment) -> None:
        """一句检测出来的话：识别 → 过滤 → 对话 → 播报。"""
        started = time.monotonic()
        text = self.recognizer.transcribe(segment.pcm, segment.sample_rate)
        elapsed = int((time.monotonic() - started) * 1000)

        if not text:
            logger.debug("识别为空（音频 %.0fms，耗时 %dms）",
                         segment.duration_ms, elapsed)
            return

        # 过滤 1：噪声上 STT 常吐单个标点/语气词
        if self.config.filter_noise and not is_meaningful(text):
            logger.debug("识别结果无实际内容，丢弃：%r", text)
            return

        # 过滤 2：兜底的回声去重（半双工没盖住时生效）
        if looks_like_echo(text, self.speaker.last_text):
            logger.info("判定为自身回声，丢弃：%r", text)
            return

        logger.info("语音识别：%r（音频 %.0fms，耗时 %dms）",
                    text, segment.duration_ms, elapsed)

        try:
            reply = self.on_text(text)
        except Exception:
            logger.exception("对话处理失败")
            return

        reply_text = getattr(reply, "reply", None)
        if reply_text:
            self.speaker.say(reply_text)

    # ------------------------------------------------------------------

    def say(self, text: str) -> bool:
        """让机器人说话。供主动关怀等外部逻辑调用。"""
        return self.speaker.say(text)

    def interrupt(self) -> None:
        """打断当前播报（barge-in 模式下由语音循环调用）。"""
        self.speaker.stop()
