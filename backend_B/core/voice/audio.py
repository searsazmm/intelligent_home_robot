# -*- coding: utf-8 -*-
"""音频设备封装（sounddevice）。

--------------------------------------------------------------------------
三条用血换来的规则
--------------------------------------------------------------------------
1. **设备一律按名字子串解析，绝不按索引。**
   索引会随虚拟设备的增减而漂移 —— 本机就装着 ToDesk 的远程虚拟声卡，
   而这类设备列表每次远程连接都可能变。误选到虚拟声卡的表现是
   「播放成功但听不见」：**完全不报错**，只有人站在机器前才发现。
   （更正一条旧记录：PortAudio 的 MME 默认输出是 Realtek 扬声器、
   不是 ToDesk，``sd.default.device`` 为 ``[1, 4]``。
   但"按名字选"这条规则与默认是谁无关，照样成立。）

2. **采样率取设备原生值，绝不硬编码 16000。**
   本机 MME 报 44100、WASAPI 报 48000。直接开 16000 的流会抛
   ``PortAudioError: Invalid sample rate``。

3. **音频回调里绝不做重活。**
   回调有硬实时期限（几十毫秒），在里面做 STT 或写文件必然
   丢帧或爆音。回调只把数据塞进队列，其余在 worker 线程做。

``sounddevice`` 的所有 import 都在函数体内部 —— 本模块可以在没装
sounddevice 的机器上被 import，只是功能不可用。
"""

from __future__ import annotations

import array
import logging
import queue
import sys
import threading
import wave
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_rate: float
    kind: str          # "input" / "output"


def _sounddevice():
    """惰性 import。没装时抛 ImportError，由调用方转成友好提示。"""
    import sounddevice

    return sounddevice


def is_available() -> bool:
    """sounddevice 能不能用（装了 **且** PortAudio 有设备）。"""
    from core.voice.base import resolve_optional

    if resolve_optional("sounddevice") is None:
        return False
    try:
        _sounddevice().query_devices()
        return True
    except Exception:
        # PortAudio 装了但没有可用设备（无声卡的服务器、虚拟机）
        return False


def list_devices() -> List[DeviceInfo]:
    """列出所有设备。``--list-audio`` 用它，也用于排错。"""
    try:
        devices = _sounddevice().query_devices()
    except Exception:
        logger.exception("枚举音频设备失败")
        return []

    out: List[DeviceInfo] = []
    for index, device in enumerate(devices):
        channels_in = int(device.get("max_input_channels") or 0)
        channels_out = int(device.get("max_output_channels") or 0)
        if channels_in > 0:
            out.append(DeviceInfo(index, device["name"], channels_in,
                                  float(device.get("default_samplerate") or 0), "input"))
        if channels_out > 0:
            out.append(DeviceInfo(index, device["name"], channels_out,
                                  float(device.get("default_samplerate") or 0), "output"))
    return out


def describe_devices() -> str:
    """人类可读的设备清单，给启动日志和 ``--list-audio`` 用。"""
    devices = list_devices()
    if not devices:
        return "（没有检测到任何音频设备）"

    try:
        defaults = _sounddevice().default.device
        default_in, default_out = defaults[0], defaults[1]
    except Exception:
        default_in = default_out = None

    lines = []
    for device in devices:
        mark = ""
        if device.kind == "input" and device.index == default_in:
            mark = "  ← 系统默认输入"
        elif device.kind == "output" and device.index == default_out:
            mark = "  ← 系统默认输出"
        lines.append(
            f"  [{device.index:>2}] {device.kind:<6} {device.name} "
            f"（{device.channels} 声道, {device.default_rate:.0f}Hz）{mark}"
        )
    return "\n".join(lines)


def find_device(name: str, kind: str, prefer_max_channels: int = 2) -> Optional[int]:
    """按名字**子串**查找设备索引。找不到返回 None。

    大小写不敏感的子串匹配：用户记不住完整设备名，
    而且 Windows 上的名字里常有一堆后缀。

    **同名设备常常有好几个，不能只取第一个。** 本机实测：
    搜 ``"Realtek"`` 的输出设备会依次命中
        [5]  扬声器 (Realtek(R) Audio)                    8 声道 44100Hz
        [12] 扬声器 (Realtek(R) Audio)                    2 声道 48000Hz
        [21] Speakers 1 (Realtek HD Audio output with SST) 2 声道 48000Hz
    第一个是**环绕声端点**（8 声道），我们的语音是单声道，
    往它上面送等于把一句话摊到 8 个声道去，很可能听不见或者走错音箱。

    所以策略是：先按「声道数不超过 prefer_max_channels」筛一遍，
    再取索引最小的。筛选后为空才退回全部匹配里的第一个。
    """
    if not name:
        return None

    needle = name.strip().lower()
    matches = [d for d in list_devices()
               if d.kind == kind and needle in d.name.lower()]

    if not matches:
        logger.warning("没找到名字包含 %r 的%s设备，改用系统默认", name,
                       "输入" if kind == "input" else "输出")
        return None

    simple = [d for d in matches if d.channels <= prefer_max_channels]
    pool = simple or matches
    chosen = min(pool, key=lambda d: d.index)

    logger.info("按名字 %r 选中%s设备 [%d] %s（%d 声道, %.0fHz）",
                name, "输入" if kind == "input" else "输出",
                chosen.index, chosen.name, chosen.channels, chosen.default_rate)

    if len(matches) > 1:
        # 把其它候选也列出来 —— 选错了要能一眼看出该改用什么名字
        alternatives = ", ".join(f"[{d.index}]{d.name}({d.channels}ch)"
                                 for d in matches if d.index != chosen.index)
        logger.debug("同名设备还有：%s", alternatives)
    return chosen.index


def resolve_input_device(name: str = "") -> Optional[int]:
    """解析输入设备。给了名字就按名字找，没给就用系统默认。"""
    if name:
        return find_device(name, "input")
    try:
        return _sounddevice().default.device[0]
    except Exception:
        return None


def resolve_output_device(name: str = "") -> Optional[int]:
    """解析输出设备。"""
    if name:
        return find_device(name, "output")
    try:
        return _sounddevice().default.device[1]
    except Exception:
        return None


def native_sample_rate(device: Optional[int], fallback: int = 16000) -> int:
    """设备的原生采样率。

    **不要硬编码 16000**：本机 MME 是 44100、WASAPI 是 48000，
    用不匹配的采样率开流会直接抛 ``PortAudioError: Invalid sample rate``。
    """
    try:
        info = _sounddevice().query_devices(device)
        rate = int(info.get("default_samplerate") or fallback)
        return rate if rate > 0 else fallback
    except Exception:
        return fallback


def wav_info(path: str):
    """读 WAV 头，返回 ``(rate, channels, width, frames)``。"""
    with wave.open(path, "rb") as handle:
        return (handle.getframerate(), handle.getnchannels(),
                handle.getsampwidth(), handle.getnframes())


def resample_pcm(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """16bit 单声道 PCM 线性插值重采样。

    **为什么必须有它**：Windows 上 PortAudio 不做重采样，声卡只接受自己的
    原生采样率。本机实测（``check_output_settings``）：

        [5]  Realtek 8 声道   mono @16000 OK  @44100 OK  @48000 OK   ← 8 声道环绕端点，什么都收
        [12] Realtek 2 声道   mono @16000 失败  @44100 失败  @48000 OK
        [21] Realtek 2 声道   mono @16000 失败  @44100 失败  @48000 OK

    也就是说 SAPI 合成出来的 16kHz WAV 往真正的音箱上播**会直接抛
    ``PortAudioError: Invalid sample rate``**。而 8 声道那个端点之所以
    "能播"，只是因为它内部替我们重采样了 —— 声音到底去了哪几个声道没人知道。
    所以不能靠"随便挑个能开的设备"，必须自己把采样率对上。

    实现上优先用 numpy（快几十倍），没装 numpy 时退回纯 Python 的
    ``array`` 循环 —— 后者慢，但保证 B 的单声道依赖不被打破。
    """
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate or not pcm:
        return pcm

    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return b""

    try:
        import numpy

        source = numpy.frombuffer(pcm[:usable], dtype="<i2")
        count = int(len(source) * target_rate / source_rate)
        if count <= 0:
            return b""
        positions = numpy.arange(count) * (source_rate / float(target_rate))
        base = positions.astype("int64")
        numpy.clip(base, 0, len(source) - 1, out=base)
        upper = numpy.clip(base + 1, 0, len(source) - 1)
        frac = positions - base
        mixed = source[base] + (source[upper] - source[base]) * frac
        return mixed.astype("<i2").tobytes()
    except ImportError:
        pass

    source_array = array.array("h")
    source_array.frombytes(pcm[:usable])
    if sys.byteorder == "big":
        source_array.byteswap()

    count = int(len(source_array) * target_rate / source_rate)
    if count <= 0:
        return b""

    ratio = source_rate / float(target_rate)
    last = len(source_array) - 1
    out = array.array("h", bytes(2 * count))
    for index in range(count):
        position = index * ratio
        low = int(position)
        if low >= last:
            out[index] = source_array[last]
            continue
        frac = position - low
        start = source_array[low]
        out[index] = int(start + (source_array[low + 1] - start) * frac)
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def supported_output_rate(device: Optional[int], preferred: int,
                          fallbacks=(48000, 44100, 16000)) -> int:
    """挑一个该输出设备**真的能开**的采样率。

    优先用设备报告的原生采样率，不行再按 fallbacks 依次探测。
    探测用 ``check_output_settings``，它只做校验、不真的打开流。
    """
    try:
        sounddevice = _sounddevice()
    except ImportError:
        return preferred

    candidates = [preferred] if preferred else []
    try:
        native = int(sounddevice.query_devices(device).get("default_samplerate") or 0)
        if native > 0 and native not in candidates:
            candidates.append(native)
    except Exception:
        pass
    candidates.extend(rate for rate in fallbacks if rate not in candidates)

    for rate in candidates:
        try:
            sounddevice.check_output_settings(device=device, samplerate=rate, channels=1)
            return rate
        except Exception:
            continue

    logger.warning("没找到可用的输出采样率（设备 %r），仍按 %d Hz 尝试", device, preferred)
    return preferred


def _downmix(pcm: bytes, channels: int) -> bytes:
    """多声道交错 PCM 取第一声道。"""
    import array as _array

    usable = len(pcm) - (len(pcm) % (2 * channels))
    samples = _array.array("h")
    samples.frombytes(pcm[:usable])
    if sys.byteorder == "big":
        samples.byteswap()
    mono = samples[0::channels]
    if sys.byteorder == "big":
        mono.byteswap()
    return mono.tobytes()


class Player:
    """播放 WAV。用 RawOutputStream 逐块写，**不依赖 numpy**。"""

    def __init__(self, device_name: str = "") -> None:
        self.device_name = device_name
        self._device: Optional[int] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------

    def play_wav(self, path: str, block_ms: int = 100) -> bool:
        """同步播放一个 WAV。播放期间调用 ``stop()`` 可以打断。

        同步是刻意的：调用方（语音循环）需要知道"说完了"才能恢复采集，
        否则会把自己的声音录进去。

        采样率会对齐到设备真正支持的值（见 :func:`resample_pcm`）——
        SAPI 给的是 16kHz，而本机真正出声的 2 声道端点只吃 48kHz，
        不做这一步会直接抛 ``PortAudioError: Invalid sample rate``。
        """
        try:
            wav_rate, channels, width, frames = wav_info(path)
        except (OSError, wave.Error):
            logger.warning("无法读取 WAV：%s", path)
            return False

        if frames <= 0:
            logger.warning("WAV 没有音频数据：%s", path)
            return False

        if width != 2:
            logger.warning("只支持 16bit WAV，当前 %d 字节/采样：%s", width, path)
            return False

        if self._device is None:
            self._device = resolve_output_device(self.device_name)

        try:
            sounddevice = _sounddevice()
        except ImportError:
            logger.warning("未安装 sounddevice，无法播放语音（pip install sounddevice）")
            return False

        try:
            with wave.open(path, "rb") as handle:
                raw = handle.readframes(handle.getnframes())
        except (OSError, wave.Error):
            logger.warning("读取 WAV 数据失败：%s", path)
            return False

        # 多声道广播成单声道：我们的内容本来就是单声道，
        # 这里只是防御一个意外带声道的 WAV。
        if channels > 1:
            raw = _downmix(raw, channels)

        play_rate = supported_output_rate(self._device, wav_rate)
        if play_rate != wav_rate:
            logger.debug("重采样 %dHz → %dHz 以匹配输出设备", wav_rate, play_rate)
            raw = resample_pcm(raw, wav_rate, play_rate)

        self._stop.clear()
        block_bytes = max(2, int(play_rate * block_ms / 1000.0) * 2)

        try:
            with sounddevice.RawOutputStream(
                samplerate=play_rate,
                channels=1,
                dtype="int16",
                device=self._device,
            ) as stream:
                for start in range(0, len(raw), block_bytes):
                    if self._stop.is_set():
                        stream.abort()      # 被打断：立刻停，不等缓冲区放完
                        return True
                    stream.write(raw[start:start + block_bytes])
            return True
        except Exception:
            # 播放失败不该中断对话 —— 文字回复和表情照常
            logger.exception("播放音频失败（设备 %r）", self.device_name or "默认")
            return False

    def stop(self) -> None:
        """打断正在进行的播放。从别的线程调用。"""
        self._stop.set()

    def close(self) -> None:
        self.stop()
        try:
            _sounddevice().stop()               # 兜底：停掉 sounddevice 全局播放
        except Exception:
            pass


class Recorder:
    """采集音频，把数据块推进队列。回调里只做入队这一件事。

    ``mute_while_playing`` 是半双工开关：机器人说话期间丢弃麦克风数据。
    理由见 :mod:`core.voice.loop` 的说明 —— 不做回声消除的话，
    机器人一定会听见自己，形成"自己回答自己"的自激回路。
    戴耳机时可以把 ``--barge-in`` 打开（即 mute 关闭）实现插话。
    """

    def __init__(
        self,
        device_name: str = "",
        sample_rate: Optional[int] = None,
        blocksize_ms: int = 20,
    ) -> None:
        self.device_name = device_name
        self.requested_rate = sample_rate
        self.blocksize_ms = blocksize_ms

        #: 留 None 表示"打开时问设备要原生值"（见 start()）。
        #: ⚠️ 这里**不能**写 `sample_rate or 16000` —— 那样 start() 里
        #: `if self.sample_rate is None` 永远不成立，"取设备原生采样率"
        #: 这条规则就成了写着好看的死代码，而丢给 PortAudio 的 16000
        #: 在只吃 44100/48000 的设备上会直接抛 Invalid sample rate。
        self.sample_rate: Optional[int] = sample_rate
        self._device: Optional[int] = None
        self._stream = None
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._muted = threading.Event()
        self._dropped = 0
        self._running = False

    # ------------------------------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        """PortAudio 回调。**这里只能做入队**。

        status 非零意味着丢帧/溢出，值得记一笔 —— 它通常说明
        队列消费太慢或机器太忙，是"识别偶尔失灵"的根因。
        """
        if status:
            logger.debug("音频回调状态异常：%s", status)

        if self._muted.is_set():
            self._dropped += 1
            return

        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            # 队列满了就丢最旧的：宁可丢一点旧音频，
            # 也不能在回调里阻塞（阻塞会直接爆音）。
            self._dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(bytes(indata))
            except queue.Empty:
                pass

    def start(self) -> bool:
        """打开输入流。失败返回 False（没有麦克风时照常跑文本模式）。"""
        try:
            sounddevice = _sounddevice()
        except ImportError:
            logger.warning("未安装 sounddevice，语音输入不可用（pip install sounddevice）")
            return False

        self._device = resolve_input_device(self.device_name)
        if self.sample_rate is None:
            # 用设备原生采样率，不做硬编码
            self.sample_rate = native_sample_rate(self._device, 16000)

        block = max(1, int(self.sample_rate * self.blocksize_ms / 1000.0))
        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=block,
                device=self._device,
                channels=1,
                dtype="int16",
                callback=self._callback,
            )
            self._stream.start()
            self._running = True
            logger.info("麦克风已打开（设备 %s，%dHz，每块 %dms）",
                        self.device_name or "系统默认", self.sample_rate, self.blocksize_ms)
            return True
        except Exception:
            logger.exception("打开麦克风失败（设备 %r）", self.device_name or "系统默认")
            self._stream = None
            return False

    def read(self, timeout: float = 0.5) -> Optional[bytes]:
        """取一块音频。超时返回 None（**不是空字节**）。

        超时返回 None 让调用方有机会检查停止标志 ——
        否则关不掉这个循环。
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_muted(self, muted: bool) -> None:
        """半双工开关。播放期间置真。"""
        if muted:
            self._muted.set()
        else:
            self._muted.clear()

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    @property
    def dropped_blocks(self) -> int:
        return self._dropped

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """关闭输入流。必须幂等。

        先 ``stop()`` 再 ``close()``：直接把流对象丢掉会让 PortAudio
        在后台继续回调，进程退出时表现为"卡住几秒"。
        """
        self._running = False
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
