# -*- coding: utf-8 -*-
"""语音合成（TTS）适配器。

默认引擎是 Windows 自带的 SAPI —— **零安装、离线可用**，这对答辩现场很重要。
edge-tts 作为可选的音色增强，装了就用。

--------------------------------------------------------------------------
SAPI 这条路为什么要这么绕
--------------------------------------------------------------------------
调用 Windows 的语音合成有几条路，都踩过坑：

1. ``win32com`` / ``comtypes`` —— 本机没装；而且 COM 的 ``SAPI.SpVoice.Speak()``
   可能**异步提前返回**，导致 WAV 被截断（文件存在但是半截，最难查的一类 bug）。

2. ``pyttsx3`` —— 本机没装。

3. 走 PowerShell，但把脚本写在命令行上 —— 中文的引号、``$``、反引号
   会和 PowerShell 的转义规则打架。

所以走的是 ``powershell -EncodedCommand <base64>``：
    - 脚本用 **UTF-16LE** 编码后 base64，命令行上只有纯 ASCII，
      任何中文/特殊字符都不可能破坏命令行解析（这是微软给的标准做法）。
    - 待朗读的文本走 **stdin**，同样 base64（UTF-8）编码，管道上也是纯 ASCII，
      彻底绕开控制台代码页的问题（Python 写 UTF-8、PowerShell 按 GBK 读是经典乱码现场）。
    - ``creationflags=CREATE_NO_WINDOW`` —— 注意 ``-WindowStyle Hidden``
      **挡不住**控制台闪窗，必须用这个标志位。

4. **常驻子进程**：实测每次 spawn + ``Add-Type`` 要 602ms 的死寂。
   一次启动、逐句喂行，把这 602ms 摊到整个会话上。

5. 用同步的 ``Speak()`` 而不是 ``SpeakAsync``，并强制 WAV 为
   16000Hz/16bit/单声道 —— 默认输出可能是 22.05kHz 或 ``WAVE_FORMAT_EXTENSIBLE``，
   而标准库 ``wave`` 可能拒收。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional, Tuple

import config
from core.voice.base import NullSynthesizer, Synthesizer, resolve_optional

logger = logging.getLogger(__name__)

#: 合成出来的 WAV 统一格式。16k 单声道是语音的常规选择，
#: 也让标准库 ``wave`` 一定能读。
WAV_RATE = 16000
WAV_CHANNELS = 1
WAV_SAMPLE_WIDTH = 2

#: 常驻 PowerShell 启动与单次合成的超时（秒）。
STARTUP_TIMEOUT = 20.0
SYNTH_TIMEOUT = 30.0

#: 选型时用来试探在线引擎的短句。见 :func:`probe_synthesizer`。
#: 越短越好 —— 它花的是一次真实的合成往返。
PROBE_TEXT = "你好"

#: stdout 上用 ASCII 标记协议状态，避免任何编码问题
READY = "READY"
OK = "OK"
ERR = "ERR"


def speed_to_sapi_rate(speed: float) -> int:
    """语速倍数 → SAPI 的 ``Rate`` 整数。

    约定对齐 ``backend_A/shared/actions.py`` 的 ``Speak.speed``（浮点倍数，
    ``0.85`` 表示比正常慢一点）。SAPI 的 Rate 是 -10~10 的整数，0 为正常。

    这里用线性映射 ``(speed - 1) * 10``，简单、可预测、可测试。
    实际听感上 SAPI 的 Rate 并非严格线性，但对 0.8~1.2 这个我们真正会用的
    区间来说够准；真要精细调，改这一个函数即可。
    """
    try:
        value = float(speed)
    except (TypeError, ValueError):
        return 0
    return max(-10, min(10, int(round((value - 1.0) * 10.0))))


def build_sapi_script(rate: int = 0, voice_hint: str = "") -> str:
    """生成常驻 PowerShell 脚本的正文。**纯函数，可直接单测。**

    协议：
        启动     → 打印 ``READY``
        每行请求 → ``<wav路径>\\t<文本的base64(UTF-8)>``
        每次应答 → ``OK`` / ``ERR``
    """
    return r"""
$ErrorActionPreference = 'Stop'
try { Add-Type -AssemblyName System.Speech } catch { [Console]::Out.WriteLine('FATAL'); exit 1 }

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# 选中文音色：**按 Culture 前缀挑，不要硬编码名字**。
# 'Microsoft Huihui Desktop' 在这台机器上有，换一台就未必 ——
# 硬编码会让程序在演示机上直接抛异常。
$voiceHint = '__VOICE_HINT__'
if ($voiceHint -ne '') {
    try { $synth.SelectVoice($voiceHint) } catch { }
} else {
    $voice = $synth.GetInstalledVoices() |
        Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'zh*' } |
        Select-Object -First 1
    if ($voice -ne $null) { $synth.SelectVoice($voice.VoiceInfo.Name) }
}

$synth.Rate = __RATE__

# 强制 WAV 格式：默认输出可能是 22.05kHz 或 WAVE_FORMAT_EXTENSIBLE，
# 标准库 wave 读不了，后续播放也会踩采样率的坑。
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

[Console]::Out.WriteLine('READY')
[Console]::Out.Flush()

# 逐行处理，直到 stdin 关闭。常驻进程就是为了省掉每次 600ms 的启动开销。
while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }
    if ($line.Trim() -eq '') { continue }

    $parts = $line.Split([char]9, 2)
    if ($parts.Length -lt 2) { [Console]::Out.WriteLine('ERR'); [Console]::Out.Flush(); continue }

    $path = $parts[0]
    $text = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($parts[1]))

    $status = 'OK'
    try {
        $synth.SetOutputToWaveFile($path, $fmt)
        # 同步 Speak：SpeakAsync 可能在写盘完成前就返回，得到被截断的 WAV
        $synth.Speak($text)
    } catch {
        $status = 'ERR'
    } finally {
        # 无论如何都要把输出重定向收回来，否则下一次 SetOutputToWaveFile 会失败
        try { $synth.SetOutputToNull() } catch { }
    }
    [Console]::Out.WriteLine($status)
    [Console]::Out.Flush()
}
$synth.Dispose()
""".replace("__RATE__", str(int(rate))).replace("__VOICE_HINT__", voice_hint.replace("'", "''"))


def encode_ps_command(script: str) -> str:
    """把脚本编码成 ``-EncodedCommand`` 需要的 base64（UTF-16LE）。

    PowerShell 只认 UTF-16LE，用 UTF-8 会得到一串乱码命令。
    """
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def decode_ps_command(encoded: str) -> str:
    """``encode_ps_command`` 的逆运算。测试与排查用。"""
    return base64.b64decode(encoded.encode("ascii")).decode("utf-16-le")


def build_sapi_argv(script: str) -> list:
    """构造 PowerShell 的命令行参数。**纯函数，可直接单测。**

    任何情况下都只用 ``-EncodedCommand``，**绝不把脚本正文放到命令行上**。
    这是整个 SAPI 适配器最关键的一条：脚本里含中文音色名、引号、``$``、
    反引号，直接拼进命令行会被 PowerShell 的转义规则撕碎，
    而且报错信息完全指不出真正的原因。
    """
    return [
        "powershell",
        "-NoProfile",          # 不加载用户 profile：别人的机器上有奇怪 profile 会拖慢/报错
        "-NonInteractive",     # 不等待任何交互输入
        "-EncodedCommand", encode_ps_command(script),
    ]


def format_speak_request(wav_path: str, text: str) -> str:
    """构造一行 stdin 请求：``<路径>\\t<文本base64>``。

    文本第二次 base64（这次是 UTF-8）是为了让**管道上全是 ASCII**：
    Python 默认按 UTF-8 写管道，而 PowerShell 读控制台输入时用的是
    系统 OEM 代码页（本机是 GBK）—— 直接写中文必然是乱码。
    """
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"{wav_path}\t{payload}"


def parse_speak_request(line: str) -> Tuple[str, str]:
    """``format_speak_request`` 的逆运算。测试用。"""
    path, _, payload = line.partition("\t")
    return path, base64.b64decode(payload.encode("ascii")).decode("utf-8")


class SapiSynthesizer:
    """Windows 自带语音合成。默认引擎，零安装。"""

    name = "sapi"

    def __init__(
        self,
        rate: int = 0,
        voice: str = "",
        startup_timeout: float = STARTUP_TIMEOUT,
    ) -> None:
        self.rate = rate
        self.voice = voice

        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._usable = False
        #: 是不是我们主动关掉了子进程（而不是它自己崩了）。
        #: 用来把"退出时的正常中断"和"真的有故障"分开 —— 见 synthesize_wav。
        self._closing = False
        self._start(startup_timeout)

    # ------------------------------------------------------------------

    def _start(self, timeout: float) -> None:
        """启动常驻 PowerShell 并等它报告 READY。"""
        # 重新启动（_start 也可能被 synthesize_wav 的"进程已退出"分支调用），
        # 于是我们不再是"正在关闭"的状态。
        self._closing = False

        if sys.platform != "win32":
            logger.warning("SAPI 只在 Windows 上可用，当前平台 %s，语音合成停用", sys.platform)
            return

        script = build_sapi_script(rate=self.rate, voice_hint=self.voice)
        argv = build_sapi_argv(script)

        # CREATE_NO_WINDOW 是唯一能真正不闪控制台的开关：
        # -WindowStyle Hidden 只管 PowerShell 自己的窗口，
        # 控制台宿主窗口照样会闪一下。
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
                bufsize=0,                      # 无缓冲，保证逐行即时互通
            )
        except OSError:
            logger.exception("启动 PowerShell 失败，语音合成停用")
            return

        self._reader = threading.Thread(
            target=self._read_loop, args=(self._process,),
            name="sapi-reader", daemon=True,
        )
        self._reader.start()

        try:
            first = self._lines.get(timeout=timeout)
        except queue.Empty:
            logger.error("PowerShell 语音进程 %v 秒内没有就绪，语音合成停用", timeout)
            self._terminate()
            return

        if first == READY:
            self._usable = True
            logger.info("SAPI 语音合成已就绪（Rate=%d%s）",
                        self.rate, f"，音色 {self.voice}" if self.voice else "")
        else:
            logger.error("PowerShell 语音进程启动异常（收到 %r），语音合成停用", first)
            self._terminate()

    def _read_loop(self, process: subprocess.Popen) -> None:
        """把子进程的 stdout 逐行推进队列。

        必须单开线程：``readline()`` 会一直阻塞，而我们需要给每一次合成
        设超时。子进程若卡死，没有这层就会把整个语音线程拖住。
        """
        try:
            for raw in process.stdout:                    # type: ignore[union-attr]
                self._lines.put(raw.decode("ascii", "replace").strip())
        except (ValueError, OSError):
            pass                                          # 进程被关闭，正常
        finally:
            self._lines.put(None)                         # 哨兵：告诉等待方进程没了

    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._usable

    def synthesize_wav(self, text: str, path: str) -> bool:
        """合成到 path。失败返回 False，不抛异常。"""
        if not self._usable or not text.strip():
            return False

        with self._lock:
            if self._process is None or self._process.poll() is not None:
                logger.warning("PowerShell 语音进程已退出，尝试重启")
                self._usable = False
                self._start(STARTUP_TIMEOUT)
                if not self._usable:
                    return False

            try:
                request = format_speak_request(path, text)
                self._process.stdin.write(request.encode("ascii") + b"\n")   # type: ignore[union-attr]
                self._process.stdin.flush()                                   # type: ignore[union-attr]
            except (OSError, ValueError):
                logger.exception("向语音进程写入失败")
                return False

            try:
                status = self._lines.get(timeout=SYNTH_TIMEOUT)
            except queue.Empty:
                logger.error("语音合成超时（%v 秒）", SYNTH_TIMEOUT)
                return False

            if status is None and self._closing:
                # 子进程是我们自己关掉的（退出流程），不是故障。
                # 这里打 WARNING 是"喊狼来了"：Ctrl+C 时正好有句话在合成，
                # 就会看到一条"语音合成失败"——而其实什么都没坏。
                # 误导性的告警会训练人忽略告警，所以这条必须是 INFO。
                logger.info("语音进程已关闭，放弃这次合成（程序正在退出）")
                return False

            if status != OK:
                logger.warning("语音合成失败（子进程返回 %r）", status)
                return False

        # 子进程说 OK 不代表文件可用（磁盘满、路径非法……），落盘再确认一次。
        # 这一步同时让「WAV 存在且非空」成为可断言的测试点。
        if not os.path.isfile(path) or os.path.getsize(path) <= 44:
            # 44 字节是 WAV 头的长度：只有头没有数据，等于空文件
            logger.warning("语音合成返回成功但文件不可用：%s", path)
            return False
        return True

    def close(self) -> None:
        """关掉常驻进程。必须幂等。"""
        self._terminate()

    def _terminate(self) -> None:
        # 先置标志再动进程：读取线程会在进程结束时往队列里推哨兵，
        # 而此刻可能正有一句话在等它 —— 标志先立起来，等待方才知道
        # 这是"我们关的"而不是"它崩了"。
        self._closing = True
        process, self._process = self._process, None
        self._usable = False
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()        # 关 stdin → 脚本的 ReadLine 返回 null → 自己退出
        except OSError:
            pass
        try:
            process.wait(timeout=3.0)
        except (subprocess.TimeoutExpired, OSError):
            try:
                process.kill()
            except OSError:
                pass


class EdgeTtsSynthesizer:
    """edge-tts 在线合成，音色比 SAPI 自然得多。

    **代价**：要联网；而且它只输出 MP3（edge-tts 7.x 把输出格式写死成
    ``audio-24khz-48kbitrate-mono-mp3``，没有 PCM 选项）。
    标准库解不了 MP3，所以这里用 ``soundfile``（内置的 libsndfile ≥1.1
    原生支持 MP3 解码）把 mp3 转成 WAV 落盘。

    转换后**刻意重采样到 16k 单声道**，让下游（播放、测试）面对统一的格式。
    """

    name = "edge_tts"

    #: 这个引擎要联网。``build_synthesizer`` 选中它之前会先**试合成一句**
    #: （见 :func:`probe_synthesizer`）。SAPI 是离线的，没有这个属性，
    #: 于是不会被试 —— 它本来就不会因为网络而失败。
    requires_network = True

    #: 默认音色。晓晓是最常用的中文女声，语速偏慢，适合老人。
    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

    def __init__(self, voice: str = "", speed: float = 1.0) -> None:
        self.voice = voice or os.environ.get("B_TTS_VOICE", self.DEFAULT_VOICE)
        self.speed = speed
        self._usable = False
        self._reason = ""

        if resolve_optional("edge_tts") is None:
            self._reason = "未安装 edge-tts（pip install edge-tts）"
        elif resolve_optional("soundfile") is None:
            # 没有 soundfile 就没法把 mp3 转成 WAV，等于装了个不能用的引擎。
            # 这种情况要**在启动时就说清楚**，而不是每次合成时才失败。
            self._reason = ("edge-tts 输出的是 MP3，需要 soundfile 解码"
                            "（pip install soundfile）")
        else:
            self._usable = True
            logger.info("edge-tts 语音合成已就绪（音色 %s）", self.voice)

        if not self._usable:
            logger.warning("edge-tts 不可用：%s", self._reason)

    @property
    def available(self) -> bool:
        return self._usable

    def synthesize_wav(self, text: str, path: str) -> bool:
        if not self._usable or not text.strip():
            return False

        import asyncio
        import tempfile

        import edge_tts
        import soundfile

        # edge-tts 的语速用百分比字符串（+20% / -10%）
        percent = int(round((self.speed - 1.0) * 100))
        rate = f"{percent:+d}%"

        mp3_path = ""
        try:
            handle, mp3_path = tempfile.mkstemp(suffix=".mp3", prefix="b_tts_")
            os.close(handle)

            async def _run() -> None:
                communicate = edge_tts.Communicate(text, self.voice, rate=rate)
                await communicate.save(mp3_path)

            asyncio.run(_run())

            if not os.path.isfile(mp3_path) or os.path.getsize(mp3_path) == 0:
                logger.warning("edge-tts 没有产出音频（可能是网络问题）")
                return False

            # mp3 → PCM → 统一格式的 WAV
            samples, source_rate = soundfile.read(mp3_path, dtype="int16", always_2d=True)
            mono = samples[:, 0] if samples.shape[1] > 1 else samples.reshape(-1)

            if source_rate != WAV_RATE:
                mono = _resample_nearest(mono, source_rate, WAV_RATE)

            import wave
            with wave.open(path, "wb") as wav:
                wav.setnchannels(WAV_CHANNELS)
                wav.setsampwidth(WAV_SAMPLE_WIDTH)
                wav.setframerate(WAV_RATE)
                wav.writeframes(mono.astype("int16").tobytes())
            return True

        except Exception:
            # 网络失败是常态（断网、被墙、服务变更），不该让对话中断
            logger.exception("edge-tts 合成失败")
            return False
        finally:
            if mp3_path:
                try:
                    os.unlink(mp3_path)
                except OSError:
                    pass

    def close(self) -> None:
        return None


def _resample_nearest(samples, source_rate: int, target_rate: int):
    """最近邻重采样。

    **刻意用最近邻而不是插值**：这里只是把 24k 降到 16k 好让下游格式统一，
    语音的清晰度由引擎决定，插值带来的那点差别听不出来，
    却要引入 numpy 的插值实现或额外依赖。简单可靠优先。

    真正需要高质量重采样时（比如要做声纹），再换成 scipy.signal.resample_poly。
    """
    import numpy

    if source_rate == target_rate or len(samples) == 0:
        return samples
    ratio = target_rate / float(source_rate)
    count = int(len(samples) * ratio)
    if count <= 0:
        return samples[:0]
    positions = numpy.arange(count) / ratio
    positions = numpy.clip(positions.astype("int64"), 0, len(samples) - 1)
    return samples[positions]


# --------------------------------------------------------------------------
# 豆包（火山引擎）语音合成 —— 语音合成大模型 2.0（SeedTTS 2.0）
#
# 走 v3 的 HTTP Chunked 单向流式接口。请求的构造和**响应流的解析**都被拆成
# 模块级纯函数（build_doubao_headers / build_doubao_request /
# parse_doubao_stream），理由和 SAPI 那边的 build_sapi_argv 一样：
# 网络没法在单测里跑，但"请求拼得对不对、响应解没解对"必须能测。
#
# ⚠️ 这一整套是**把 v1 换掉**来的，原因是实测（2026-09-29）：v1 那套
#    「appid + access_token + cluster」在本项目的凭证下**永远**返回
#    401 "load grant: requested grant not found in SaaS storage" ——
#    换集群、换音色、换 appid 形态（字符串/数字）、甚至喂**故意的垃圾凭证**，
#    服务端回的都是同一句话，说明它压根没走到核对凭证那步。
#    换成下面这套「API Key + ResourceId」之后，音频立刻就出来了。
#
#    两套模型的区别是结构性的，不是参数没调对：
#
#      v1（旧）：appid + access_token + cluster，鉴权头 Authorization: Bearer;<token>
#      v3（本文件）：API Key，鉴权头 X-Api-Key；模型版本走 X-Api-Resource-Id
#
#    v3 里**没有 appid 这个东西**，ResourceId 也不是账号相关的值，
#    而是"要调哪个模型"的常量（见 DOUBAO_RESOURCE_ID）。
# --------------------------------------------------------------------------

#: 语音合成大模型 2.0 的 HTTP Chunked 单向流式端点。
#: （SSE 版本是同路径加 ``/sse``，两者协议一致，只是 SSE 每行带 ``data:`` 前缀，
#:   parse_doubao_stream 两种都认。）
DOUBAO_ENDPOINT = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

#: 模型版本。**这是常量，不是账号里的值** ——
#: ``seed-tts-2.0`` = 语音合成大模型 2.0（2.0 音色以 ``*_uranus_bigtts`` 结尾）；
#: ``seed-tts-1.0`` = 1.0（兼容 ``BV*_streaming`` 音色）。
#: 1.0 和 2.0 的音色**不能混用**：用 2.0 的模型名配 1.0 的音色会被拒。
DOUBAO_RESOURCE_ID = "seed-tts-2.0"

#: 流的正常结束码。收到它就说明这一句合成完整了。
DOUBAO_DONE_CODE = 20000000

#: 采样率。24k 是 SeedTTS 的常规值，最后统一重采样到 WAV_RATE。
DOUBAO_SAMPLE_RATE = 24000


def probe_synthesizer(engine, text: str = PROBE_TEXT) -> bool:
    """**真的合成一句话**，看它到底行不行。

    为什么需要这一步：``available`` 只回答"**库装好没有**"，
    答不了"**能不能出声**" —— 而这两件事对用户是同一个结果：机器一声不吭。

    为什么不只探端口：本机的实际故障正是「TCP 连得上、TLS 被重置」——
    ``socket.create_connection("speech.platform.bing.com", 443)`` 成功，
    可紧接着的 WebSocket 请求被中间设备 RST 掉
    （``ClientConnectorError ... 指定的网络名不再可用``）。
    只探端口会得出"网络正常"的结论，然后每一句都在运行期失败，
    而候选链后面那个**离线可用**的 SAPI 永远轮不到。

    所以这里走一次完整合成：它和之后要发生的事一模一样，
    没有比这更准的判据了。代价是一次真实往返（本机实测 edge-tts 约 1.7s
    —— 失败时更短，连接被重置得很快）。

    ⚠️ 超时用的是引擎自己的 ``SYNTH_TIMEOUT``（30s）：这是**启动路径**上的
    阻塞调用，最坏情况下会让启动慢半分钟。但这只发生在"网络坏得
    连都连不上"的时候 —— 那时候慢一点也比哑巴强。
    """
    import tempfile

    directory = tempfile.mkdtemp(prefix="b_probe_")
    try:
        return bool(engine.synthesize_wav(text, os.path.join(directory, "p.wav")))
    except Exception:
        # synthesize_wav 的契约是"永不抛异常"，但探测不该因为
        # 某个引擎违约而把整个选型流程带崩。
        logger.debug("探测 %r 时抛了异常", getattr(engine, "name", "?"),
                     exc_info=True)
        return False
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def build_doubao_headers(api_key: str, resource_id: str, reqid: str) -> dict:
    """构造 v3 接口的三个鉴权头。

    ⚠️ 这里**没有 Authorization**，也**没有 appid**。v3 的鉴权是：

        X-Api-Key          API Key（控制台「语音技术 → API Key 管理」拿的那个）
        X-Api-Resource-Id  要调哪个模型，见 DOUBAO_RESOURCE_ID
        X-Api-Request-Id   本次请求的 id，接口要求必填

    别把 v1 的 ``Authorization: Bearer;{token}`` 混进来 —— v3 不认它，
    实测会回 ``no token or access_key was found from the header or query``，
    一个和真实原因（头放错了）毫无关系的报错。所以单独写测试钉住这组头。
    """
    return {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": reqid,
        "Content-Type": "application/json",
    }


def speech_rate_from_speed(speed: float) -> int:
    """把"语速倍数"换算成 SeedTTS 的 ``speech_rate``。

    两套单位不一样，这里是唯一的换算点，所以单独拎出来测：

        倍数 1.0（正常）→ 0
        倍数 0.9（慢 10%）→ -10
        倍数 1.2（快 20%）→ +20

    服务端接受的范围是 [-50, 100]，越界会被拒，所以**必须夹住** ——
    配置里写错一个数量级（比如填了 90 而不是 0.9）不该让整句话合成失败。
    """
    rate = int(round((float(speed) - 1.0) * 100))
    return max(-50, min(100, rate))


def build_doubao_request(
    text: str,
    speaker: str,
    speed: float = 1.0,
) -> dict:
    """构造 v3 的请求体。

    结构比 v1 扁：没有 ``app`` 段（鉴权全在头里），合成参数集中在 ``req_params``。

    ``speed`` 是**倍数**（和 config.VOICE_SPEED 同一单位），
    这里换算成服务端的 ``speech_rate``，见 :func:`speech_rate_from_speed`。

    ⚠️ **这里没有 reqid。** 请求 id 只走 ``X-Api-Request-Id`` 头
    （见 :func:`build_doubao_headers`）—— v3 的 body 里不带这个字段，
    塞进来是照搬 v1 的习惯，服务端不认。
    """
    return {
        "user": {
            # 用户标识，服务端只用来做统计/隔离，给个固定值即可。
            "uid": "intelligent_home_robot",
        },
        "req_params": {
            "text": text,
            "speaker": speaker,
            "audio_params": {
                # 拿 MP3 而不是 PCM：MP3 的解码路径
                # （soundfile → 重采样到 16k）已经在 edge-tts 上验证过了。
                "format": "mp3",
                "sample_rate": DOUBAO_SAMPLE_RATE,
                "speech_rate": speech_rate_from_speed(speed),
            },
        },
    }


def parse_doubao_stream(raw: bytes) -> bytes:
    """把接口返回的 JSON 行流拼成 MP3 字节。

    协议：**一行一个 JSON 对象**，音频块在 ``data`` 字段（base64）。

        {"code":0,        "data":"<base64>", ...}   ← 音频块，一行一块，十几块
        {"code":20000000, "message":"OK"}           ← 结束

    ⚠️ 出错时服务端也走 HTTP 200 + 同样的流，错误信息只在 ``message`` 里 ——
    所以**不能只看 HTTP 状态码**，否则鉴权失败会被当成"合成成功但没有声音"。

    也认 SSE 版本：每行前面的 ``data:`` 前缀会被剥掉，``event:`` 行会被忽略。

    返回拼好的 MP3；一块音频都没解出来就抛 ``ValueError``（带上服务端的原话，
    便于排查）。抛而不返回 None，是为了让"失败原因"没法被调用方顺手丢掉。
    """
    chunks = []
    done = False
    failure = ""

    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line or line.startswith("event:"):
            continue
        if line.startswith("data:"):          # SSE 端点带这个前缀
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            event = json.loads(line)
        except ValueError:
            # 不是 JSON 就跳过：流的边界上偶尔会有零散字符，
            # 为它整句失败不值得。
            continue

        code = event.get("code")
        if code == DOUBAO_DONE_CODE:
            done = True
            continue
        if code not in (0, None):
            failure = "code=%s message=%s" % (code, event.get("message"))
            continue
        if event.get("data"):
            chunks.append(event["data"])

    if not chunks:
        raise ValueError(failure or "服务端没有返回任何音频数据")
    if failure:
        # 有音频但也有错误：宁可当成失败，也不要把半句话播出去。
        raise ValueError(failure)
    if not done:
        logger.debug("豆包流没有结束码（可能被截断），但已拿到 %d 块音频", len(chunks))

    return base64.b64decode("".join(chunks))


class DoubaoSynthesizer:
    """豆包（火山引擎）在线合成，音色比 SAPI 自然得多。

    走**语音合成大模型 2.0（SeedTTS 2.0）**，鉴权只要一个 API Key。
    **代价**：要联网。没配 API Key 就判为不可用，由 ``build_synthesizer``
    退到下一个引擎 ——「凭证没填」不该表现为「机器人突然不说话了」。

    走标准库 ``urllib`` 而不是 ``requests``：模块 B 的运行期零第三方依赖
    是被测试守着的（见 tests/test_no_third_party_imports.py），
    能用标准库解决就不该为它破例。
    """

    name = "doubao"

    #: 见 :func:`probe_synthesizer`。豆包的域名在国内可直连，
    #: 这正是它比 edge-tts 更适合这个项目的原因之一。
    requires_network = True

    def __init__(
        self,
        voice: str = "",
        speed: float = 1.0,
        api_key: str = "",
        resource_id: str = "",
    ) -> None:
        self.voice = voice or config.TTS_DOUBAO_VOICE
        self.speed = speed
        self.api_key = api_key or config.TTS_DOUBAO_API_KEY
        self.resource_id = resource_id or config.TTS_DOUBAO_RESOURCE_ID

        self._usable = False
        self._reason = ""

        # 缺哪个就报哪个，别只说一句"不可用"让人去猜 —— 这是用户明确要求的：
        # 「无凭证的时候，控制台明确打印缺失的变量名称」。
        # 只有 API Key 和音色两项：v3 不需要 appid（见文件头那段说明）。
        missing = [
            name for name, value in (
                ("B_DOUBAO_TOKEN", self.api_key),
                ("B_DOUBAO_VOICE", self.voice),
            ) if not value
        ]
        if missing:
            self._reason = "未配置 %s" % "、".join(missing)
            if "B_DOUBAO_TOKEN" in missing:
                self._reason += ("（API Key 在控制台「语音技术 → API Key 管理」页面）")
        elif resolve_optional("soundfile") is None:
            # 接口返回 MP3，没有 soundfile 就解不开，等于装了个不能用的引擎。
            # 和 edge-tts 一样，要在启动时就说清楚，而不是每次合成时才失败。
            self._reason = ("豆包返回的是 MP3，需要 soundfile 解码"
                            "（pip install soundfile）")
        else:
            self._usable = True
            logger.info("豆包语音合成已就绪（SeedTTS 2.0，音色 %s）", self.voice)

        if not self._usable:
            logger.warning("豆包语音合成不可用：%s", self._reason)

    @property
    def available(self) -> bool:
        return self._usable

    @property
    def reason(self) -> str:
        return self._reason

    def synthesize_wav(self, text: str, path: str) -> bool:
        if not self._usable or not text.strip():
            return False

        import io
        import uuid
        import urllib.error
        import urllib.request

        import soundfile

        reqid = uuid.uuid4().hex
        body = build_doubao_request(
            text=text,
            speaker=self.voice,
            speed=self.speed,
        )
        request = urllib.request.Request(
            DOUBAO_ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers=build_doubao_headers(self.api_key, self.resource_id, reqid),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=SYNTH_TIMEOUT) as response:
                raw = response.read()

            # 失败也可能走 200，所以真正的判据在流里面 —— 见 parse_doubao_stream。
            mp3 = parse_doubao_stream(raw)

            # MP3 → PCM → 统一的 16k 单声道 WAV。
            # 这段和 EdgeTtsSynthesizer 里那段是同一套做法，故意保持一致：
            # 一条已经在本机验证过的解码路径，不值得为省几行代码再赌一次。
            samples, source_rate = soundfile.read(
                io.BytesIO(mp3), dtype="int16", always_2d=True)
            mono = samples[:, 0] if samples.shape[1] > 1 else samples.reshape(-1)
            if source_rate != WAV_RATE:
                mono = _resample_nearest(mono, source_rate, WAV_RATE)

            import wave
            with wave.open(path, "wb") as wav:
                wav.setnchannels(WAV_CHANNELS)
                wav.setsampwidth(WAV_SAMPLE_WIDTH)
                wav.setframerate(WAV_RATE)
                wav.writeframes(mono.astype("int16").tobytes())
            return True

        except urllib.error.HTTPError as exc:
            # API Key 无效、音色与模型版本不匹配都会走到这里。
            # 把服务端的话原样带出来，否则排查时只剩下一个数字状态码。
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            logger.warning("豆包合成 HTTP 错误：%s %s", exc.code, detail)
            return False
        except ValueError as exc:
            # parse_doubao_stream 抛的：服务端明确报了错（200 也照样出错）。
            logger.warning("豆包合成失败：%s", exc)
            return False
        except Exception:
            # 断网、超时、解码失败都是常态，不该让对话中断。
            logger.exception("豆包合成失败")
            return False

    def close(self) -> None:
        return None


#: 日志里一句话最多显示多少个字。整句都打出来会把日志淹掉，
#: 截得太短又对不上是哪句话 —— 和 loop.py 里那个 ``_brief`` 是同一个取舍。
#: 不跨模块复用它：tts 是被 loop 的上层装配用的，反过来 import loop 会拧成环。
LOG_TEXT_CHARS = 30


def _clip(text: str) -> str:
    """把一句话压成适合进日志的短形式。"""
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= LOG_TEXT_CHARS:
        return text
    return text[:LOG_TEXT_CHARS] + "…"


class FallbackSynthesizer:
    """主引擎失败时，**逐句**改用本地引擎兜底。

    为什么需要它：``build_synthesizer`` 只在**启动时**挑一次引擎。挑中豆包之后，
    运行中途网络抖一下，那一句就彻底没声音了 —— 用户看到的是"机器人突然
    不吭声"，而本地明明有个完全离线的 SAPI 闲着，只是没人叫它。

    为什么不能只靠启动时的探测：本机实测的网络故障是「TCP 连得上、TLS 被重置」，
    这种故障是**间歇性**的。探测通过只说明"那一刻能通"，不保证之后每句都能通。
    演示时最怕的正是这种偶发。

    备用引擎用**工厂**而不是实例：``SapiSynthesizer.__init__`` 会真的拉起一个
    PowerShell 进程，提前建一个就等于每次启动都白起一个进程、还拖慢启动。
    所以工厂只在**第一次真失败**时调用一次，之后复用。
    """

    #: 主引擎要联网，兜底才有可能被用到。
    requires_network = True

    def __init__(self, primary: Synthesizer, factory) -> None:
        self.primary = primary
        self._factory = factory
        self._backup: Optional[Synthesizer] = None
        self._backup_tried = False
        #: 上一次合成是不是兜底产出的。``Speaker`` 靠它决定**要不要写缓存** ——
        #: 见下面 property 的说明。
        self.used_fallback = False

    @property
    def name(self) -> str:
        """对外报**主引擎**的名字。

        这不是为了好看：缓存键是按 ``name|voice|speed`` 算的，而兜底产出的
        音频**不进缓存**（见 :attr:`used_fallback`）。所以这里保持主引擎身份，
        缓存里存的永远是主引擎的音色。
        """
        return getattr(self.primary, "name", "fallback")

    @property
    def voice(self) -> str:
        return getattr(self.primary, "voice", "")

    @property
    def speed(self) -> float:
        return getattr(self.primary, "speed", 1.0)

    @property
    def available(self) -> bool:
        return bool(getattr(self.primary, "available", True))

    @property
    def reason(self) -> str:
        return getattr(self.primary, "reason", "")

    def _backup_engine(self) -> Optional[Synthesizer]:
        """懒建备用引擎。**只试一次**：建不出来就返回 None。

        为什么失败后不重试：这是在"网络已经坏了"的路径上，
        每次都重试会把一句失败变成一串异常日志。也不能退回
        ``NullSynthesizer`` —— 它没有 ``available``，会被当成"能用"，
        然后日志里出现一句 "改用本地引擎 'null' 兜底"，看着像成功了，
        实际什么都没发出去。
        """
        if not self._backup_tried:
            self._backup_tried = True
            try:
                self._backup = self._factory()
            except Exception:
                logger.exception("兜底语音引擎创建失败")
                self._backup = None
        return self._backup

    def synthesize_wav(self, text: str, path: str) -> bool:
        self.used_fallback = False
        try:
            if self.primary.synthesize_wav(text, path):
                return True
        except Exception:
            logger.exception("主语音引擎抛异常，转兜底")

        backup = self._backup_engine()
        if backup is None or not getattr(backup, "available", True):
            logger.warning("主语音引擎不可用，且没有可用的本地引擎可兜底")
            return False

        logger.warning("在线合成失败，改用本地引擎 %r 兜底这句：%r",
                       backup.name, _clip(text))
        try:
            ok = bool(backup.synthesize_wav(text, path))
        except Exception:
            logger.exception("兜底引擎也失败了")
            return False
        self.used_fallback = ok
        return ok

    def close(self) -> None:
        for engine in (self.primary, self._backup):
            if engine is None:
                continue
            try:
                engine.close()
            except Exception:
                pass


#: 引擎名 → 构造函数
ENGINES = {
    "sapi": SapiSynthesizer,
    "doubao": DoubaoSynthesizer,
    "volcengine": DoubaoSynthesizer,
    "edge_tts": EdgeTtsSynthesizer,
    "edge": EdgeTtsSynthesizer,
    "null": NullSynthesizer,
    "none": NullSynthesizer,
}


def build_synthesizer(
    name: str = "auto",
    speed: float = 1.0,
    voice: str = "",
    probe: bool = True,
    local_fallback: bool = True,
) -> Synthesizer:
    """按名字构造合成引擎。

    ``auto`` 的退让顺序是 **豆包 → edge-tts → SAPI**：音色从好到差，
    最后落到完全离线的 SAPI。于是「默认用最好的音色」和
    「断网 / 没配好也不会变哑巴」同时成立。

    ``probe=False`` 跳过对联网引擎的试合成。测试用它来避免联网 ——
    探测本身是 :func:`probe_synthesizer`，生产路径上不要关掉它。

    ``local_fallback=True``（默认）给**联网**引擎再套一层运行期兜底：
    启动时探测通过、之后某一句真的失败了，就临时改用本地 SAPI 把
    **那一句**发出来，而不是让它静默消失。见 :class:`FallbackSynthesizer`。
    离线引擎（SAPI）本来就不需要兜底，套了只是白套。

    **任何情况下都不抛异常**，最差返回 :class:`NullSynthesizer`。
    """
    requested = (name or "auto").strip().lower()
    rate = speed_to_sapi_rate(speed)

    if requested in ("null", "none"):
        logger.info("语音合成已按配置停用（--tts none）")
        return NullSynthesizer()

    if requested == "auto":
        # 音色从好到差依次退让，最后落到完全离线的 SAPI：
        #   豆包（要凭证、要联网）→ edge-tts（要联网）→ SAPI（离线，机械音）
        # 这样「默认用最好的音色」和「断网/没配好也不会变哑巴」同时成立。
        candidates = ("doubao", "edge_tts", "sapi")
    elif requested in ENGINES:
        # 名字**认得出**、但那个引擎没就绪（缺凭证 / 缺库 / 断网）时不能就此罢手：
        # 原先是 candidates=(requested,) 然后直接掉到 NullSynthesizer，
        # 于是 `--tts doubao` 忘了填凭证 = 机器人一声不吭，
        # 而命令行上看起来一切正常。退到 auto 链，音色差一点总比不发声强。
        candidates = (requested,) + tuple(
            name for name in ("doubao", "edge_tts", "sapi")
            if name != requested
        )
    else:
        # 名字不认识是另一回事：那是拼写错误或过时的配置。
        # 悄悄换成别的音色只会让人以为 --tts 生效了，所以这里只记日志。
        # （想彻底静音请用 --tts none，那条路在前面就返回了。）
        candidates = (requested,)

    for candidate in candidates:
        factory = ENGINES.get(candidate)
        if factory is None:
            logger.warning("未知的语音合成引擎 %r（可选：%s）",
                           candidate, ", ".join(ENGINES))
            continue
        try:
            if candidate == "sapi":
                engine = factory(rate=rate, voice=voice)
            elif candidate in ("edge_tts", "doubao"):
                engine = factory(voice=voice, speed=speed)
            else:
                engine = factory()
        except Exception:
            logger.exception("初始化语音合成引擎 %r 失败", candidate)
            continue

        # 引擎自己判断能不能用（缺库、缺凭证、缺模型、平台不对）
        if not getattr(engine, "available", True):
            logger.warning("语音合成引擎 %r 不可用，尝试下一个", candidate)
            continue

        # 库装好了 ≠ 能出声。走网络的引擎还要**真的合成一次**才算通过 ——
        # 见 probe_synthesizer 的说明：不试这一步的话，「网不通」会让 auto
        # 停在第一个"看起来可用"的在线引擎上，而它每句都失败，
        # 于是机器人彻底哑掉，明明候选链后面还有离线的 SAPI 可用。
        if probe and getattr(engine, "requires_network", False):
            logger.info("正在试合成一句话，确认 %r 真的能用…", candidate)
            if not probe_synthesizer(engine):
                logger.warning(
                    "语音合成引擎 %r 试合成失败，跳过（多半是断网或网络受限）；"
                    "后面还有离线引擎可以兜底", candidate)
                try:
                    engine.close()
                except Exception:
                    pass
                continue

        if candidate != candidates[0]:
            logger.warning("已回退到 %r 发声（原因见上一条）。"
                           "音色会差一些，但不会不发声。", candidate)

        # 联网引擎再套一层运行期兜底：启动时探测通过 ≠ 之后每句都通过。
        # 挑中的已经是 SAPI 就不用套了 —— 它本来就在本地。
        if local_fallback and getattr(engine, "requires_network", False):
            return FallbackSynthesizer(
                engine,
                # 工厂，不是实例：SapiSynthesizer 一建就起一个 PowerShell 进程，
                # 提前建等于每次启动都白起一个。这里只传"怎么建"。
                lambda: SapiSynthesizer(rate=rate, voice=voice),
            )
        return engine

    logger.warning(
        "没有可用的语音合成引擎，机器人将**只说话不发声**（文字仍会显示 / 记入历史）。\n"
        "  本机可用的方案：Windows 自带 SAPI 应为默认可用；\n"
        "  若想用更好的音色：pip install edge-tts soundfile 然后 --tts edge；\n"
        "  豆包（SeedTTS 2.0）需要 B_DOUBAO_TOKEN（API Key）和 B_DOUBAO_VOICE 两个环境变量，\n"
        "  不需要 appid。详见 README.md §4.8「豆包凭证怎么配」。"
    )
    return NullSynthesizer()
