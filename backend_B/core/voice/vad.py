# -*- coding: utf-8 -*-
"""语音端点检测（VAD）：把连续的 PCM 流切成一句一句。

--------------------------------------------------------------------------
为什么这个文件是整个语音层里最该被单独拿出来的一块
--------------------------------------------------------------------------
语音链路里最容易出错的不是"识别得准不准"，而是**哪里算一句话的开头、
哪里算结尾**。切错了，STT 拿到的是半句话，识别结果自然一塌糊涂，
而你从日志上完全看不出问题出在哪。

所以这里刻意做成**只依赖标准库的纯函数**：
    - 不 import sounddevice（碰设备就没法在 CI 上跑）
    - 不 import numpy（B 模块运行期零第三方依赖，这是硬约束）
    - 输入是 bytes，输出是 bytes，没有任何全局状态

于是 tests/test_voice_vad.py 可以合成 PCM 直接喂进来断言，
不需要麦克风、不需要等待真实时间流逝。

--------------------------------------------------------------------------
为什么不用固定阈值
--------------------------------------------------------------------------
固定阈值在安静的书房和开着风扇的客厅里表现天差地别 ——
前者把翻书声当说话，后者对正常说话没反应。

这里用**最小值统计**（minimum statistics）估计噪声底：
维护一个 2 秒的滑动窗口，取窗口内**最小**的帧能量作为当前噪声水平。
理由是「人在说话时一定是断续的」—— 任何 2 秒窗口里总有停顿，
最小值就落在停顿上；而稳态的嗡声（风扇、空调）没有停顿，
最小值就等于嗡声本身，于是它被学成噪声底，不再被当成说话。

这比"只在静音时更新噪声底"的做法好在一个地方：
后者遇到稳态嗡声会死锁 —— 因为嗡声一上来就被判成"说话"，
于是永远不进静音分支，噪声底永远学不上去。
"""

from __future__ import annotations

import array
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: int16 的满量程。能量一律归一化到 0~1，这样阈值数字是可读的
#: （0.05 比 1638 直观得多）。
INT16_FULL_SCALE = 32768.0


@dataclass(frozen=True)
class VadConfig:
    """端点检测参数。全部用**毫秒**，因为调参时想的是"停顿多久算说完"。"""

    sample_rate: int = 16000
    frame_ms: int = 20

    #: 判定「开始说话」需要连续多少毫秒超阈值。
    #: 太短会把咳嗽、键盘声触发；太长会把第一个字吃掉。
    start_trigger_ms: int = 150

    #: 判定「说完了」需要连续多少毫秒低于阈值。
    #: **这个值必须给得大。** 老人说话中间停顿多，尤其是想词的时候；
    #: 给 300ms 就会把一句话切成好几段，每段都是残缺的。
    end_silence_ms: int = 700

    #: 触发点**之前**额外保留多少毫秒。
    #: 没有它，「你好」的「你」这个声母会落在触发帧之前被丢掉 ——
    #: 而中文声母丢了，STT 基本就废了。这是端点的经典坑。
    pre_roll_ms: int = 300

    #: 句子尾部额外保留多少毫秒的静音。
    #: 上面 700ms 的静音判定是"确认说完了"用的，不该全部收进语句 ——
    #: 全收进去等于每句话都拖长 0.7 秒，白白喂给 STT 一堆静音。
    tail_pad_ms: int = 200

    #: **真正发声**的时长短于这个值就丢弃（咳嗽、关门、椅子响）。
    #:
    #: 注意是"发声时长"，不是"片段长度"。片段里还含 pre_roll 的静音
    #: 和尾部保留，用它来判会让一声 150ms 的咳嗽也凑够 400ms 而被放行。
    #: 这个区别是实测出来的，不是洁癖。
    min_speech_ms: int = 400

    #: 超过这个时长强制切断。
    #: 宁可切两句让 STT 分别识别，也不能无限缓冲 —— 老人絮叨起来
    #: 可以连续说很久，缓冲区涨爆之前得有个头。
    max_speech_ms: int = 8000

    #: 噪声底统计窗口长度（见模块开头的最小值统计说明）。
    noise_window_ms: int = 2000

    #: 超过噪声底这么多倍才算说话。
    threshold_ratio: float = 3.0

    #: 绝对下限。安静房间里噪声底接近 0，此时阈值会低到任何抖动都算说话，
    #: 必须有个地板兜住。
    abs_min_level: float = 0.006

    #: 绝对上限。噪声底被学得太高时（比如有人一直在旁边大声说话），
    #: 阈值会高到听不见正常说话 —— 这个上限防止它失控。
    abs_max_level: float = 0.20

    def frames(self, milliseconds: float) -> int:
        """毫秒 → 帧数，至少 1 帧。"""
        return max(1, int(round(milliseconds / self.frame_ms)))

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000.0)

    @property
    def frame_bytes(self) -> int:
        """每帧的字节数（16bit 单声道）。"""
        return self.frame_samples * 2


@dataclass(frozen=True)
class SpeechSegment:
    """一句检测出来的话。"""

    pcm: bytes            #: 16bit 单声道 PCM
    sample_rate: int
    #: 结束原因：``silence`` = 正常说完；``max_length`` = 说太长被切断；
    #: ``flush`` = 外部主动收尾（比如程序退出）。调试时很有用。
    reason: str

    @property
    def duration_ms(self) -> float:
        return len(self.pcm) / 2.0 / self.sample_rate * 1000.0

    def __len__(self) -> int:
        return len(self.pcm)


def frame_level(frame: bytes) -> float:
    """一帧的归一化 RMS（0~1）。

    为什么手算而不用 ``audioop``：**``audioop`` 在 Python 3.13 被移除了**，
    本机是 3.14，import 会直接失败。这是"照抄网上例子"最容易踩的坑。

    ``array`` 的 ``frombytes`` 按**本机字节序**解释，而 PCM 约定是小端，
    所以在小端机器上（x86/ARM 都是）无需转换；大端机器上必须 byteswap。
    加这个判断是为了让代码在纸面上就是对的，而不是"碰巧在我机器上能跑"。
    """
    if not frame:
        return 0.0

    usable = len(frame) - (len(frame) % 2)      # 奇数字节是半个采样，丢掉
    if usable <= 0:
        return 0.0

    samples = array.array("h")
    samples.frombytes(frame[:usable])
    if sys.byteorder == "big":
        samples.byteswap()

    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples)) / INT16_FULL_SCALE


class Endpointer:
    """把任意长度的 PCM 喂进来，吐出完整的语句。

    用法::

        ep = Endpointer(VadConfig())
        for segment in ep.push(chunk):     # chunk 可以任意长、任意切分
            handle(segment)
        last = ep.flush()                  # 收尾时把半句交出来

    线程安全：不保证。约定只有识别线程调用它。
    """

    def __init__(self, config: Optional[VadConfig] = None) -> None:
        self.config = config or VadConfig()
        self._frame_bytes = self.config.frame_bytes
        self._pending = bytearray()                # 还没凑够一帧的零头

        # 触发点之前的音频 + 它是否发声，用来补上被吃掉的声母
        self._pre_roll: Deque[Tuple[bytes, bool]] = deque(
            maxlen=self.config.frames(self.config.pre_roll_ms)
        )
        # 噪声底的最小值统计窗口
        self._levels: Deque[float] = deque(
            maxlen=self.config.frames(self.config.noise_window_ms)
        )

        self._segment: List[bytes] = []
        self._segment_ms = 0.0
        self._voiced_ms = 0.0                      # 真正发声的时长
        self._trailing_silence = 0                 # 句尾连续静音帧数
        self._in_speech = False
        self._above_count = 0

    # ------------------------------------------------------------------

    @property
    def noise_floor(self) -> float:
        """当前估计的噪声底。联调时打印出来看阈值合不合理。"""
        return min(self._levels) if self._levels else 0.0

    @property
    def threshold(self) -> float:
        """当前判定阈值 = clip(噪声底 × 倍数, 下限, 上限)。"""
        raw = self.noise_floor * self.config.threshold_ratio
        return max(self.config.abs_min_level,
                   min(raw, self.config.abs_max_level))

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    # ------------------------------------------------------------------

    def push(self, pcm: bytes) -> List[SpeechSegment]:
        """喂入任意长度的 PCM，返回其中切出来的完整语句（0 条或多条）。

        内部按固定帧长切分，所以调用方不必关心自己拿到多少字节 ——
        音频回调给多少都行。
        """
        self._pending.extend(pcm)
        out: List[SpeechSegment] = []

        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[:self._frame_bytes])
            del self._pending[:self._frame_bytes]
            segment = self._feed_frame(frame)
            if segment is not None:
                out.append(segment)
        return out

    def flush(self) -> Optional[SpeechSegment]:
        """收尾：把正在收集的半句话交出来。

        程序退出、或者音频流结束时调用。
        """
        if not self._in_speech:
            return None
        return self._close("flush")

    def reset(self) -> None:
        """清空全部状态。音频设备重连后调用。"""
        self._pending.clear()
        self._pre_roll.clear()
        self._levels.clear()
        self._segment = []
        self._segment_ms = 0.0
        self._voiced_ms = 0.0
        self._trailing_silence = 0
        self._in_speech = False
        self._above_count = 0

    # ------------------------------------------------------------------

    def _feed_frame(self, frame: bytes) -> Optional[SpeechSegment]:
        level = frame_level(frame)
        self._levels.append(level)                  # 噪声统计与说话无关，一直更新
        is_loud = level > self.threshold

        if not self._in_speech:
            return self._feed_idle(frame, is_loud)
        return self._feed_speech(frame, is_loud)

    def _feed_idle(self, frame: bytes, is_loud: bool) -> Optional[SpeechSegment]:
        """等待说话开始的阶段。"""
        self._pre_roll.append((frame, is_loud))

        if not is_loud:
            self._above_count = 0
            return None

        self._above_count += 1
        if self._above_count < self.config.frames(self.config.start_trigger_ms):
            return None

        # 确认开始说话：把触发点**之前**缓存的声音一起算进去，
        # 否则"你好"的声母就丢了。
        self._in_speech = True
        self._segment = [chunk for chunk, _ in self._pre_roll]
        self._segment_ms = len(self._segment) * self.config.frame_ms
        self._voiced_ms = sum(
            self.config.frame_ms for _, loud in self._pre_roll if loud
        )
        self._trailing_silence = 0
        self._above_count = 0
        self._pre_roll.clear()
        logger.debug("说话开始（噪声底 %.4f，阈值 %.4f）", self.noise_floor, self.threshold)
        return None

    def _feed_speech(self, frame: bytes, is_loud: bool) -> Optional[SpeechSegment]:
        """正在收集一句话的阶段。"""
        self._segment.append(frame)
        self._segment_ms += self.config.frame_ms

        if is_loud:
            self._voiced_ms += self.config.frame_ms
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

        # 说太长了，强制切开。宁可切两句分别识别，也不能无限缓冲。
        if self._segment_ms >= self.config.max_speech_ms:
            logger.debug("说话超过 %dms，强制切断", self.config.max_speech_ms)
            return self._close("max_length")

        if self._trailing_silence >= self.config.frames(self.config.end_silence_ms):
            return self._close("silence")

        return None

    def _close(self, reason: str) -> Optional[SpeechSegment]:
        """结束当前语句。发声太短的丢弃，返回 None。"""
        frames = self._segment
        voiced_ms = self._voiced_ms
        trailing = self._trailing_silence

        # 强制切断时，把尾部留给下一段当 pre_roll ——
        # 否则接缝处会丢掉约 pre_roll_ms 的声音，连续说话时后半句开头就残缺。
        if reason == "max_length" and frames:
            keep = self.config.frames(self.config.pre_roll_ms)
            self._pre_roll = deque(
                ((chunk, True) for chunk in frames[-keep:]),
                maxlen=keep,
            )
        else:
            self._pre_roll.clear()

        self._segment = []
        self._segment_ms = 0.0
        self._voiced_ms = 0.0
        self._trailing_silence = 0
        self._in_speech = False
        self._above_count = 0

        # 尾部只留 tail_pad_ms：700ms 的静音判定是"确认说完了"用的，
        # 全收进去等于每句话都拖长 0.7 秒。
        keep_tail = self.config.frames(self.config.tail_pad_ms)
        drop = trailing - keep_tail
        if drop > 0:
            frames = frames[:-drop] if drop < len(frames) else []

        pcm = b"".join(frames)

        # 用**发声**时长而不是片段长度来筛 —— 片段里还含着 pre_roll 的静音，
        # 拿它来判，一声 150ms 的咳嗽配 300ms 前置缓冲也能凑够 400ms 被放行。
        if not pcm or voiced_ms < self.config.min_speech_ms:
            logger.debug("丢弃过短片段（发声 %.0fms < %dms，%s）",
                         voiced_ms, self.config.min_speech_ms, reason)
            return None

        logger.debug("语句结束（%s，片段 %.0fms / 发声 %.0fms）",
                     reason, len(pcm) / 2.0 / self.config.sample_rate * 1000.0, voiced_ms)
        return SpeechSegment(pcm=pcm, sample_rate=self.config.sample_rate, reason=reason)
