# -*- coding: utf-8 -*-
"""音频层的纯函数测试：重采样、下混、设备挑选。

这些都不碰真实声卡 —— 设备清单用假的注入，
所以没有麦克风/音箱的机器上也能跑。
"""

import array
import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.voice import audio as audio_mod                            # noqa: E402
from core.voice.audio import (                                       # noqa: E402
    DeviceInfo,
    _downmix,
    find_device,
    native_sample_rate,
    resample_pcm,
    wav_info,
)


def make_pcm(values) -> bytes:
    return b"".join(struct.pack("<h", int(v)) for v in values)


def read_pcm(pcm: bytes):
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    return list(samples)


class TestResample(unittest.TestCase):
    """重采样是播放能否出声的关键 —— 见 audio.resample_pcm 的实测说明。"""

    def test_upsample_length(self):
        """16k → 48k，帧数应当变成 3 倍。"""
        pcm = make_pcm([0] * 100)
        out = resample_pcm(pcm, 16000, 48000)
        self.assertEqual(len(out) // 2, 300)

    def test_downsample_length(self):
        pcm = make_pcm([0] * 480)
        out = resample_pcm(pcm, 48000, 16000)
        self.assertEqual(len(out) // 2, 160)

    def test_same_rate_is_identity(self):
        """采样率相同时必须原样返回 —— 不做没必要的计算，也不引入失真。"""
        pcm = make_pcm([1, -2, 3, -4])
        self.assertEqual(resample_pcm(pcm, 16000, 16000), pcm)

    def test_empty_input(self):
        self.assertEqual(resample_pcm(b"", 16000, 48000), b"")

    def test_invalid_rates_are_passed_through(self):
        """采样率为 0 或负数时不能除零崩溃。"""
        pcm = make_pcm([1, 2, 3])
        self.assertEqual(resample_pcm(pcm, 0, 48000), pcm)
        self.assertEqual(resample_pcm(pcm, 16000, 0), pcm)

    def test_constant_signal_stays_constant(self):
        """直流信号重采样后应当还是同一个值（线性插值的正确性）。"""
        pcm = make_pcm([1000] * 160)
        out = read_pcm(resample_pcm(pcm, 16000, 48000))
        self.assertEqual(len(out), 480)
        for value in out:
            self.assertAlmostEqual(value, 1000, delta=1)

    def test_amplitude_is_preserved(self):
        """正弦波重采样后峰值不应衰减太多。

        插值实现写错（比如索引偏移）会让输出整体变小或变形 ——
        这条能抓到那类错误。
        """
        source = [int(20000 * math.sin(2 * math.pi * 200 * i / 16000))
                  for i in range(1600)]
        out = read_pcm(resample_pcm(make_pcm(source), 16000, 48000))
        self.assertAlmostEqual(max(out), max(source), delta=200)
        self.assertAlmostEqual(min(out), min(source), delta=200)

    def test_resampling_preserves_duration(self):
        """重采样不能改变时长 —— 变了的话语速就错了。"""
        pcm = make_pcm([0] * 16000)                  # 1 秒 @16k
        out = resample_pcm(pcm, 16000, 48000)
        self.assertAlmostEqual(len(out) / 2 / 48000, 1.0, places=3)

    def test_odd_byte_count_does_not_crash(self):
        out = resample_pcm(b"\x01\x02\x03", 16000, 48000)
        self.assertIsInstance(out, bytes)

    def test_no_numpy_path_also_works(self):
        """没装 numpy 时退回纯 Python 循环，结果要一致。

        这条模拟"客户机上没有 numpy"的情况 —— 语音层不该因此失效。
        """
        pcm = make_pcm([1000] * 160)
        with_numpy = resample_pcm(pcm, 16000, 48000)

        real_import = __import__

        def blocked(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("模拟没有 numpy")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked):
            without_numpy = resample_pcm(pcm, 16000, 48000)

        self.assertEqual(len(with_numpy), len(without_numpy))
        # 两条路径都是线性插值，结果应当完全一致
        self.assertEqual(read_pcm(with_numpy), read_pcm(without_numpy))


class TestDownmix(unittest.TestCase):
    def test_stereo_to_mono_takes_first_channel(self):
        # 交错：[L,R] = [100,200], [300,400]
        pcm = make_pcm([100, 200, 300, 400])
        self.assertEqual(read_pcm(_downmix(pcm, 2)), [100, 300])

    def test_eight_channel(self):
        frame = [10, 20, 30, 40, 50, 60, 70, 80]
        pcm = make_pcm(frame * 3)
        self.assertEqual(read_pcm(_downmix(pcm, 8)), [10, 10, 10])

    def test_partial_frame_is_dropped(self):
        """不足一个完整帧的尾部要丢掉，不能把半个采样当数据。"""
        pcm = make_pcm([1, 2, 3, 4, 5])          # 5 个采样、2 声道 → 只剩 2 帧
        self.assertEqual(read_pcm(_downmix(pcm, 2)), [1, 3])


class TestFindDevice(unittest.TestCase):
    """设备挑选。用假设备清单，**不依赖本机装了什么声卡**。"""

    #: 复刻本机实测的坑：同名设备里有 8 声道的环绕端点排在前面
    FAKE_DEVICES = [
        DeviceInfo(3, "Microsoft 声音映射器 - Output", 2, 44100, "output"),
        DeviceInfo(4, "扬声器 (ToDesk Virtual Audio)", 2, 44100, "output"),
        DeviceInfo(5, "扬声器 (Realtek(R) Audio)", 8, 44100, "output"),
        DeviceInfo(7, "麦克风阵列 (Realtek(R) Audio)", 2, 44100, "input"),
        DeviceInfo(12, "扬声器 (Realtek(R) Audio)", 2, 48000, "output"),
        DeviceInfo(15, "麦克风阵列 (Realtek(R) Audio)", 2, 48000, "input"),
    ]

    def setUp(self):
        patcher = mock.patch.object(audio_mod, "list_devices",
                                    return_value=self.FAKE_DEVICES)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_prefers_stereo_over_surround(self):
        """**这是本机实测出来的坑。**

        "Realtek" 的第一个匹配是 [5] 的 8 声道环绕端点。
        往 8 声道送单声道语音，声音去哪几个音箱完全不受控；
        应该选 [12] 那个 2 声道端点。
        """
        self.assertEqual(find_device("Realtek", "output"), 12)

    def test_kind_is_respected(self):
        """输入和输出不能串 —— 名字里都带 Realtek。"""
        self.assertEqual(find_device("Realtek", "input"), 7)

    def test_case_insensitive_substring(self):
        self.assertEqual(find_device("realtek", "output"), 12)
        self.assertEqual(find_device("REALTEK", "output"), 12)

    def test_partial_name_matches(self):
        self.assertEqual(find_device("ToDesk", "output"), 4)
        self.assertEqual(find_device("麦克风阵列", "input"), 7)

    def test_unknown_name_returns_none(self):
        """找不到时返回 None 让调用方退回系统默认，而不是抛异常。"""
        self.assertIsNone(find_device("没有这个设备", "output"))

    def test_empty_name_returns_none(self):
        self.assertIsNone(find_device("", "output"))

    def test_falls_back_to_multichannel_when_it_is_the_only_match(self):
        """只有环绕端点可用时也得给一个（总比直接失败强）。"""
        self.assertEqual(find_device("声音映射器", "output"), 3)


class TestDeviceQueries(unittest.TestCase):

    def test_describe_devices_handles_no_devices(self):
        with mock.patch.object(audio_mod, "list_devices", return_value=[]):
            self.assertIn("没有检测到", audio_mod.describe_devices())

    def test_describe_devices_lists_entries(self):
        devices = [DeviceInfo(1, "测试设备", 2, 48000, "output")]
        with mock.patch.object(audio_mod, "list_devices", return_value=devices):
            text = audio_mod.describe_devices()
        self.assertIn("测试设备", text)
        self.assertIn("48000", text)

    def test_native_sample_rate_returns_fallback_on_error(self):
        """查不到设备时必须给个兜底值，不能返回 None 让上层去解包。"""
        with mock.patch.object(audio_mod, "_sounddevice",
                               side_effect=Exception("没有 PortAudio")):
            self.assertEqual(native_sample_rate(None, 16000), 16000)


class TestWavInfo(unittest.TestCase):

    def test_reads_header(self):
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        try:
            with wave.open(handle.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 800)
            rate, channels, width, frames = wav_info(handle.name)
            self.assertEqual((rate, channels, width), (16000, 1, 2))
            self.assertEqual(frames, 800)
        finally:
            os.unlink(handle.name)


class TestAvailabilityProbe(unittest.TestCase):

    def test_is_available_returns_false_without_sounddevice(self):
        with mock.patch("core.voice.base.resolve_optional", return_value=None):
            self.assertFalse(audio_mod.is_available())

    def test_is_available_survives_device_query_failure(self):
        """PortAudio 装了但没有可用设备（无声卡服务器）时不能抛异常。"""
        def boom():
            raise Exception("PortAudio 没有设备")

        with mock.patch.object(audio_mod, "_sounddevice", side_effect=boom), \
                mock.patch("core.voice.base.resolve_optional", return_value="sounddevice"):
            self.assertFalse(audio_mod.is_available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
