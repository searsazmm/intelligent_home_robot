# -*- coding: utf-8 -*-
"""端点检测的测试：合成 PCM，不碰麦克风、不等真实时间。

这是语音层里唯一能完全确定性地测的部分，所以测得细一点 ——
端点切错了，后面 STT 再准也没用，而那种错在真机上极难定位。
"""

import math
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.voice.vad import (                                    # noqa: E402
    Endpointer,
    SpeechSegment,
    VadConfig,
    frame_level,
)

RATE = 16000


# --------------------------------------------------------------------------
# 合成音频的工具
# --------------------------------------------------------------------------

def silence(ms: float, rate: int = RATE) -> bytes:
    """纯静音。int16 零值。"""
    return b"\x00\x00" * int(rate * ms / 1000.0)


def tone(ms: float, amplitude: float = 0.3, freq: float = 220.0,
         rate: int = RATE) -> bytes:
    """一段正弦音，模拟"有人在说话"。

    amplitude 是归一化到 0~1 的峰值，对应 RMS = amplitude / √2。
    """
    count = int(rate * ms / 1000.0)
    peak = int(amplitude * 32767)
    return b"".join(
        struct.pack("<h", int(peak * math.sin(2.0 * math.pi * freq * i / rate)))
        for i in range(count)
    )


class TestFrameLevel(unittest.TestCase):
    """底层能量计算。它是所有判定的基础，先钉死。"""

    def test_silence_is_zero(self):
        self.assertEqual(frame_level(silence(20)), 0.0)

    def test_empty_frame_is_zero(self):
        """空输入不能抛异常 —— 音频设备偶尔会给空回调。"""
        self.assertEqual(frame_level(b""), 0.0)

    def test_odd_byte_count_does_not_crash(self):
        """奇数字节是半个采样。真实网卡/USB 麦克风偶尔会这样切。"""
        self.assertEqual(frame_level(b"\x01\x02\x03"), frame_level(b"\x01\x02"))

    def test_full_scale_square_wave_is_near_one(self):
        pcm = struct.pack("<h", 32767) * 160
        self.assertAlmostEqual(frame_level(pcm), 1.0, places=4)

    def test_sine_rms_matches_theory(self):
        """幅度 0.5 的正弦，RMS 应当是 0.5/√2。

        频率取 200Hz 而不是默认的 220Hz：20ms 帧在 16kHz 下是 320 个采样，
        200Hz 恰好是整数个周期（4 个）。220Hz 会留下 0.4 个周期的零头，
        半个周期内的能量不匀，实测 RMS 会偏高约 0.8%（0.3563 而不是 0.3536）。
        所以**要断言绝对能量就选整数周期**；其余测试一律用比值断言，
        正是为了不受这个影响。
        """
        measured = frame_level(tone(20, amplitude=0.5, freq=200.0))
        self.assertAlmostEqual(measured, 0.5 / math.sqrt(2), places=3)

    def test_level_scales_with_amplitude(self):
        quiet = frame_level(tone(20, amplitude=0.1))
        loud = frame_level(tone(20, amplitude=0.4))
        self.assertAlmostEqual(loud / quiet, 4.0, places=2)

    def test_negative_samples_count_too(self):
        """只算正半周的话能量会减半 —— 这个错误会让阈值整体偏高一倍。"""
        pcm = struct.pack("<h", -32767) * 160
        self.assertAlmostEqual(frame_level(pcm), 1.0, places=4)


class TestSingleUtterance(unittest.TestCase):
    """最基本的情形：说一句，切出一句。"""

    def setUp(self):
        self.config = VadConfig()
        self.ep = Endpointer(self.config)

    def test_one_utterance_yields_exactly_one_segment(self):
        pcm = silence(500) + tone(1000) + silence(1500)
        segments = self.ep.push(pcm)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].reason, "silence")
        # 说话内容应该都在里面（1 秒语音 + 前后缓冲）
        self.assertGreater(segments[0].duration_ms, 1000)
        self.assertLess(segments[0].duration_ms, 2000)

    def test_pre_roll_keeps_the_onset(self):
        """触发点之前的声音不能丢 —— 丢了的话「你好」就变成「好」。

        构造：一段较长的前导语音，触发发生在其中。断言切出来的片段
        **比从触发起算更长**，说明 pre_roll 确实补进来了。
        """
        pcm = silence(300) + tone(1200) + silence(1500)
        segments = self.ep.push(pcm)
        self.assertEqual(len(segments), 1)

        # 触发需要 150ms，若没有 pre_roll，片段会短于 1200ms
        self.assertGreater(segments[0].duration_ms, 1200)

    def test_trailing_silence_is_trimmed(self):
        """尾部不能把 700ms 的静音判定全部收进去。

        收了的话每句话都白拖长 0.7 秒；断言片段明显短于
        「语音 + 全部静音判定」。
        """
        pcm = silence(300) + tone(1000) + silence(2000)
        segments = self.ep.push(pcm)
        self.assertEqual(len(segments), 1)
        # 语音 1000ms + 触发前最多 300ms + 尾部保留 200ms ≈ 1500ms 封顶
        self.assertLess(segments[0].duration_ms, 1600)

    def test_short_pause_does_not_split_a_sentence(self):
        """200ms 的换气停顿必须合并，不能切成两句。

        老人说话中间停顿多，这里切错的话每句都是半截。
        """
        pcm = silence(300) + tone(500) + silence(200) + tone(500) + silence(1500)
        segments = self.ep.push(pcm)
        self.assertEqual(len(segments), 1, "200ms 停顿被误判成句尾")
        self.assertGreater(segments[0].duration_ms, 1000)

    def test_long_pause_does_split(self):
        """超过 700ms 的停顿确实是句尾。这是上一条的对照组。"""
        pcm = silence(300) + tone(500) + silence(1200) + tone(500) + silence(1500)
        segments = self.ep.push(pcm)
        self.assertEqual(len(segments), 2)

    def test_two_clear_utterances_yield_two_segments(self):
        pcm = (silence(300) + tone(600) + silence(1500)
               + tone(600) + silence(1500))
        segments = self.ep.push(pcm)
        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertGreater(segment.duration_ms, 600)

    def test_nothing_is_emitted_for_pure_silence(self):
        self.assertEqual(self.ep.push(silence(5000)), [])
        self.assertIsNone(self.ep.flush())

    def test_frame_level_content_is_16bit_mono(self):
        """输出的采样率必须与配置一致 —— 写错的话 STT 会拿到变速的音频。"""
        pcm = silence(300) + tone(800) + silence(1500)
        segments = self.ep.push(pcm)
        self.assertEqual(segments[0].sample_rate, RATE)
        self.assertAlmostEqual(
            segments[0].duration_ms,
            len(segments[0].pcm) / 2.0 / RATE * 1000.0,
            places=6,
        )


class TestBlipRejection(unittest.TestCase):
    """短促的噪声不该被当成一句话。"""

    def setUp(self):
        self.ep = Endpointer(VadConfig())

    def test_150ms_blip_is_rejected(self):
        """150ms 毛刺 —— 关门声、咳嗽、碰桌子。

        它被挡下的位置和「250ms 毛刺」不同：这里连触发都够不上
        （start_trigger 要求 150ms，取整后是 8 帧 = 160ms），
        属于第一道防线。两条测试都保留，是为了区分是哪道防线在起作用。
        """
        pcm = silence(500) + tone(150, amplitude=0.5) + silence(1500)
        self.assertEqual(self.ep.push(pcm), [])

    def test_250ms_blip_triggers_but_min_speech_rejects_it(self):
        """250ms 毛刺够得着触发，但**发声时长**不足 400ms，被第二道防线丢弃。

        这是 min_speech 用「发声时长」而非「片段长度」的原因：
        片段长度里含着 300ms 前置缓冲，用它来判的话这个毛刺会被放行。
        """
        pcm = silence(500) + tone(250, amplitude=0.5) + silence(1500)
        segments = self.ep.push(pcm)
        self.assertEqual(segments, [], "短促噪声被当成了说话")

    def test_real_speech_after_a_blip_still_works(self):
        """毛刺之后紧接真话，不能因为毛刺把状态卡住。"""
        pcm = (silence(500) + tone(250, amplitude=0.5) + silence(800)
               + tone(1200) + silence(1500))
        segments = self.ep.push(pcm)
        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0].duration_ms, 1200)


class TestMaxLengthCut(unittest.TestCase):
    """说太久要强制切断，不能无限缓冲。"""

    def test_long_monologue_is_cut(self):
        config = VadConfig(max_speech_ms=4000)      # 调小以便测试跑得快
        ep = Endpointer(config)

        # 20 秒不间断的声音，期间没有任何停顿
        segments = ep.push(tone(20000))

        self.assertGreaterEqual(len(segments), 4)
        for index, segment in enumerate(segments):
            with self.subTest(index=index):
                # 允许一点点余量：切点落在帧边界上
                self.assertLessEqual(segment.duration_ms, 4000 + config.frame_ms * 2)

    def test_cut_segments_are_marked(self):
        config = VadConfig(max_speech_ms=2000)
        ep = Endpointer(config)
        segments = ep.push(tone(10000))
        self.assertTrue(any(s.reason == "max_length" for s in segments))

    def test_no_audio_is_lost_across_the_cut(self):
        """强制切断时接缝处不能丢声音。

        做法：切完的那一段尾部会被留给下一段当 pre_roll。
        断言累计输出时长与输入基本相当（不断裂成缺斤少两的片段）。
        """
        config = VadConfig(max_speech_ms=2000)
        ep = Endpointer(config)
        total_in = 10000
        segments = ep.push(tone(total_in))
        total_out = sum(s.duration_ms for s in segments)
        # 允许重叠（pre_roll 会让相邻片段有少量重复），但不该明显少于输入
        self.assertGreater(total_out, total_in * 0.9)

    def test_flush_returns_the_partial_utterance(self):
        """收尾时半句话也要交出来，不能默默丢掉。"""
        ep = Endpointer(VadConfig())
        ep.push(silence(300) + tone(1200))
        self.assertTrue(ep.in_speech, "断言前提：此时确实还在收集")

        segment = ep.flush()
        self.assertIsNotNone(segment)
        self.assertEqual(segment.reason, "flush")
        self.assertFalse(ep.in_speech)

    def test_flush_is_idempotent(self):
        ep = Endpointer(VadConfig())
        ep.push(silence(300) + tone(1200))
        self.assertIsNotNone(ep.flush())
        self.assertIsNone(ep.flush())

    def test_flush_discards_a_too_short_tail(self):
        """退出时缓冲区里往往只有几个残帧，不该当一句话交出去。"""
        ep = Endpointer(VadConfig())
        ep.push(silence(300) + tone(1200) + silence(300))
        ep.flush()                                  # 先说完一句
        ep.push(silence(300) + tone(200))           # 刚起个头就退出
        self.assertIsNone(ep.flush())


class TestNoiseFloorAdaptation(unittest.TestCase):
    """噪声底自适应 —— 这是不用固定阈值的全部理由。"""

    def test_steady_hum_is_not_speech(self):
        """0.05 幅度的持续嗡声（风扇/空调）不该被当成说话。

        机制：最小值统计取窗口内最小能量，稳态嗡声没有停顿，
        最小值就等于它自己，于是阈值升到它的 3 倍以上，嗡声被判为静音。
        """
        ep = Endpointer(VadConfig())
        hum = tone(6000, amplitude=0.05)
        self.assertEqual(ep.push(hum), [], "稳态嗡声被当成了说话")

    def test_hum_raises_the_estimated_floor(self):
        """确认上一条是靠"学上去"实现的，而不是碰巧没触发。"""
        ep = Endpointer(VadConfig())
        ep.push(tone(3000, amplitude=0.05))
        # 0.05 幅值正弦的 RMS = 0.05/√2 ≈ 0.0354
        self.assertAlmostEqual(ep.noise_floor, 0.0354, places=3)
        self.assertGreater(ep.threshold, 0.0354)

    def test_speech_over_a_hum_is_still_detected(self):
        """**关键场景**：客厅开着风扇，人在上面说话。

        用固定阈值的话这里必然失败 —— 要么阈值定低了嗡声触发，
        要么定高了人声进不来。自适应才能两头兼顾。
        """
        ep = Endpointer(VadConfig())
        ep.push(tone(3000, amplitude=0.05))                 # 先让噪声底学上去
        segments = ep.push(tone(1200, amplitude=0.4) + silence(1500))
        self.assertEqual(len(segments), 1, "风扇声之上的说话没能检测到")

    def test_threshold_never_drops_below_absolute_floor(self):
        """数字静音下噪声底为 0，阈值必须被绝对下限兜住。

        否则任何一点抖动都会被判成说话。
        """
        ep = Endpointer(VadConfig())
        ep.push(silence(3000))
        self.assertEqual(ep.noise_floor, 0.0)
        self.assertEqual(ep.threshold, VadConfig().abs_min_level)

    def test_threshold_never_exceeds_absolute_ceiling(self):
        """噪声底被学得过高时阈值要有上限，否则正常人声听不见。"""
        ep = Endpointer(VadConfig())
        ep.push(tone(3000, amplitude=0.9))                  # 极响的持续噪声
        self.assertLessEqual(ep.threshold, VadConfig().abs_max_level)

    def test_noise_floor_is_the_minimum_not_the_average(self):
        """必须是**最小值**统计。

        用平均值的话，一段"安静-说话-安静"里平均值会被说话拉高，
        反而听不见紧接着的小声说话。这条把实现选择钉住。
        """
        ep = Endpointer(VadConfig())
        ep.push(tone(1000, amplitude=0.3))                  # 说了一句
        floor_after_speech = ep.noise_floor

        ep.push(silence(2500))                              # 然后安静下来
        self.assertLess(ep.noise_floor, floor_after_speech, "噪声底没有回落")
        self.assertAlmostEqual(ep.noise_floor, 0.0, places=6)


class TestStreamingBehaviour(unittest.TestCase):
    """真实音频回调给的数据是任意切分的，不能假设它对齐到帧。"""

    def test_arbitrary_chunk_sizes_give_the_same_result(self):
        """同一段音频，按 37 字节一块喂 vs 一次性喂，结果必须相同。

        音频回调的块大小取决于设备（本机 MME 与 WASAPI 就不同），
        如果结果依赖切分方式，那就等于"换台机器行为就变了"。
        """
        pcm = silence(300) + tone(1000) + silence(1500)

        whole = Endpointer(VadConfig()).push(pcm)

        chopped_ep = Endpointer(VadConfig())
        chopped = []
        step = 37                                    # 刻意取奇数，且不是帧长的约数
        for start in range(0, len(pcm), step):
            chopped.extend(chopped_ep.push(pcm[start:start + step]))

        self.assertEqual(len(whole), len(chopped))
        for a, b in zip(whole, chopped):
            self.assertEqual(a.duration_ms, b.duration_ms)

    def test_partial_frame_is_held_not_dropped(self):
        """不足一帧的零头要留住，等下一块拼上。"""
        ep = Endpointer(VadConfig())
        frame_bytes = VadConfig().frame_bytes
        pcm = silence(300) + tone(1000) + silence(1500)

        out = []
        for start in range(0, len(pcm), frame_bytes + 7):    # 每次多喂 7 字节
            out.extend(ep.push(pcm[start:start + frame_bytes + 7]))
        self.assertEqual(len(out), 1)

    def test_reset_clears_everything(self):
        ep = Endpointer(VadConfig())
        ep.push(silence(300) + tone(1200))
        self.assertTrue(ep.in_speech)

        ep.reset()
        self.assertFalse(ep.in_speech)
        self.assertEqual(ep.noise_floor, 0.0)
        self.assertIsNone(ep.flush())

        # 重置之后还能正常工作
        segments = ep.push(silence(300) + tone(1000) + silence(1500))
        self.assertEqual(len(segments), 1)

    def test_in_speech_reflects_state(self):
        ep = Endpointer(VadConfig())
        self.assertFalse(ep.in_speech)
        ep.push(silence(300) + tone(600))
        self.assertTrue(ep.in_speech)
        ep.push(silence(1500))
        self.assertFalse(ep.in_speech)


class TestConfigDerivations(unittest.TestCase):
    def test_frame_geometry(self):
        config = VadConfig()
        self.assertEqual(config.frame_samples, 320)      # 16000 * 20ms
        self.assertEqual(config.frame_bytes, 640)

    def test_frames_helper_never_returns_zero(self):
        """毫秒数很小的时候不能算出 0 帧 —— 那会让判定条件永远不成立。"""
        config = VadConfig()
        self.assertEqual(config.frames(0), 1)
        self.assertEqual(config.frames(5), 1)

    def test_frames_helper_matches_expected_counts(self):
        config = VadConfig()
        self.assertEqual(config.frames(150), 8)          # 开始触发
        self.assertEqual(config.frames(700), 35)         # 结束静音
        self.assertEqual(config.frames(300), 15)         # pre_roll
        self.assertEqual(config.frames(400), 20)         # min_speech

    def test_custom_sample_rate_scales_frame_size(self):
        """换成 48kHz（WASAPI 的原生采样率）时帧长要跟着变。"""
        config = VadConfig(sample_rate=48000)
        self.assertEqual(config.frame_samples, 960)
        self.assertEqual(config.frame_bytes, 1920)


if __name__ == "__main__":
    unittest.main(verbosity=2)
