# -*- coding: utf-8 -*-
"""语音识别（STT）适配器。

--------------------------------------------------------------------------
为什么是"注册表 + 适配器"而不是直接选一个库
--------------------------------------------------------------------------
本机 Python 3.14，语音识别库的可用性很不确定（vosk 的 wheel 未必跟上，
speech_recognition 要另外接后端）。与其赌一个库能装上，不如把识别做成可插拔的：
装了哪个就用哪个，一个都没装就降级为键盘 / 8002 文本通道。

于是一个关键性质：**``build_recognizer()`` 在什么都没装时返回空识别器，绝不抛异常。**
B 模块必须能在没有任何语音依赖的机器上启动 —— 这是可演示性的一部分。

所有第三方 import 都在方法内部，模块顶层只有标准库。
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from core.voice.base import NullRecognizer, Recognizer, resolve_optional

logger = logging.getLogger(__name__)

#: 各引擎的依赖包名，用于可用性探测与提示信息
ENGINE_REQUIREMENTS = {
    "vosk": ("vosk",),
    "speechrecognition": ("speech_recognition",),
    "dashscope": ("dashscope",),
}


#: 默认的模型存放目录：``backend_B/models/``。
#: 把解压出来的模型目录丢进去就能用，**不必再设环境变量** ——
#: 环境变量仍然优先，方便把模型放在别处或在多份模型间切换。
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)

#: 备用模型目录：``%LOCALAPPDATA%\\vosk``（拿不到就是用户主目录下的 vosk/）。
#:
#: **存在的唯一理由是「项目路径含中文」**：vosk 的 C++ 层在 Windows 上打不开
#: 非 ASCII 路径下的模型文件（实测见 find_model_dir 的说明）。而本仓库的路径
#: 恰恰是 ``D:\\Virtually C\\成都东软学院下期\\...``，于是 ``backend_B/models/``
#: 这条路在这台机器上**永远不可能成功**。把模型放到用户目录下绕开中文路径。
LOCAL_MODELS_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "vosk")


def is_ascii_path(path: str) -> bool:
    """路径里有没有非 ASCII 字符。

    vosk 对此**没有任何容错**：含中文的路径会报
    ``Folder '...' does not contain model files``，
    而那个目录里明明躺着完整的模型 —— 报错信息完全是误导，
    会让人去反复检查文件在不在，而真正的原因是路径编码。
    """
    return all(ord(ch) < 128 for ch in path)


def _looks_like_model(path: str) -> bool:
    """是不是一个 vosk 模型目录。

    判据是 vosk 模型必备的 ``am/`` 与 ``conf/`` 两个子目录，而不是"随便一个目录"：
    没解压的 zip、误建的空目录如果被当成模型，vosk 会抛一个难懂的异常，
    比"找不到模型"难查得多。
    """
    return (os.path.isdir(os.path.join(path, "am"))
            and os.path.isdir(os.path.join(path, "conf")))


def find_model_dir() -> str:
    """找一个 **vosk 真的能加载** 的模型目录，找不到返回空串。

    查找顺序：
      1. 环境变量 ``B_STT_MODEL``（显式指定，最高优先级）
      2. ``backend_B/models/`` 下第一个像模型的子目录
      3. ``%LOCALAPPDATA%\\vosk`` 下第一个像模型的子目录

    为什么要自动找：vosk 的模型是必须单独下载的 40MB 压缩包，
    只 ``pip install vosk`` 完全没用。如果还要用户自己设环境变量，
    多出来的这一步足以让人卡在「装是装了，机器人还是听不见」。

    ⚠️ **含非 ASCII 字符的路径会被跳过**并记一条 warning。
    这不是洁癖，是实测结论：把模型放在 ``D:\\...\\成都东软学院下期\\...`` 下时，
    ``vosk.Model()`` 必定失败，且报的是"目录里没有模型文件"这种误导信息。
    与其把这种路径返回出去让 vosk 抛一个看不懂的异常，
    不如在这里就说清楚原因。
    """
    env = os.environ.get("B_STT_MODEL", "").strip()
    if env:
        # 显式指定时不做 ASCII 检查 —— 尊重用户的显式选择，
        # 但如果不是 ASCII 就先提醒一句，免得他对着 vosk 的误导报错排查半天。
        if os.path.isdir(env) and not is_ascii_path(env):
            _warn_non_ascii(env, "B_STT_MODEL")
        return env if os.path.isdir(env) else ""

    for base in (DEFAULT_MODELS_DIR, LOCAL_MODELS_DIR):
        if not os.path.isdir(base):
            continue
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            continue
        for name in entries:
            candidate = os.path.join(base, name)
            if not os.path.isdir(candidate) or not _looks_like_model(candidate):
                continue
            if not is_ascii_path(candidate):
                _warn_non_ascii(candidate, "自动查找")
                continue          # 这个加载不了，继续找下一个
            return candidate
    return ""


def _warn_non_ascii(path: str, source: str) -> None:
    """路径含中文时给一条能照着做的提示，而不是让 vosk 去报误导性错误。"""
    logger.warning(
        "找到的语音模型在含中文的路径下，vosk 无法加载，已跳过：%s\n"
        "  原因：vosk 的 C++ 层在 Windows 上打不开非 ASCII 路径，"
        "报错会是「目录里没有模型文件」（具有误导性，文件其实是好的）。\n"
        "  办法：把模型目录移到一个纯英文路径，然后设 B_STT_MODEL 指过去，例如：\n"
        "      set B_STT_MODEL=C:\\vosk\\vosk-model-small-cn-0.22\n"
        "  或者直接放到 %s\\<模型目录名>\\ 下（本程序会自动找）。\n"
        "  本次来源：%s",
        path, LOCAL_MODELS_DIR, source,
    )


def _vosk_ready() -> bool:
    """vosk 光装库没用，还要有模型目录（约 40MB，得另外下）。"""
    return bool(find_model_dir())


def _dashscope_ready() -> bool:
    """dashscope 是云端 API，没有 key 时调用必然失败。"""
    return bool(os.environ.get("DASHSCOPE_API_KEY", "").strip())


#: 除"依赖包装了没"之外，还需要**配置好了**才算可用的引擎。
#:
#: 为什么必须区分这两件事：只看 ``find_spec`` 的话，"装了库但没配好"会被算成可用，
#: 于是 ``auto`` 挑中它，而失败发生在**每次识别**的时候 —— dashscope 无 key 时
#: ``transcribe`` 捕获异常返回空串，说话的人看到的是"机器人毫无反应"，
#: 完全没有报错指向真正的原因。这和"什么都没装"是两种完全不同的故障，
#: 提示信息也该不一样。
#:
#: 没有额外前提的引擎写 ``None``（装了就能用）。
READINESS = {
    "vosk": _vosk_ready,
    "speechrecognition": None,
    "dashscope": _dashscope_ready,
}

#: 引擎优先级。识别质量与离线能力的综合考虑：
#: vosk 可离线、中文效果好；speech_recognition 需要联网或额外后端；
#: dashscope 是云端 API。
AUTO_ORDER = ("vosk", "speechrecognition", "dashscope")


class VoskRecognizer:
    """vosk 离线识别。

    需要额外下载中文模型（约 40MB）。搜索顺序见 :func:`find_model_dir`：
    显式传入的 ``model_path`` → ``B_STT_MODEL`` → ``backend_B/models/``。

    没找到模型时**记一条明确的指引再降级**，而不是抛异常 ——
    "装了库但忘了下模型"是很容易发生的情况，提示要能直接照做。
    """

    name = "vosk"

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000) -> None:
        import json                                   # noqa: F401  (供 transcribe 用)
        import vosk

        self._json = json
        self.sample_rate = sample_rate
        self._model = None
        self._available = False

        path = model_path or find_model_dir()
        if not path:
            logger.warning(
                "vosk 已安装但没有可用的中文模型，语音识别不可用。\n"
                "  请从 https://alphacephei.com/vosk/models 下载 vosk-model-small-cn-0.22，\n"
                "  解压后把整个目录放进下面任一个位置：\n"
                "      %s\n"
                "      %s   （推荐：纯英文路径，不受项目路径影响）\n"
                "  或者设环境变量 B_STT_MODEL 指向它。\n"
                "  ⚠️ 模型目录的路径**不能含中文**，否则 vosk 加载不了（见 find_model_dir）。\n"
                "  在此之前请用键盘输入或 8002 文本通道测试。",
                DEFAULT_MODELS_DIR, LOCAL_MODELS_DIR,
            )
            return

        try:
            vosk.SetLogLevel(-1)                      # 关掉 vosk 自己的刷屏日志
            self._model = vosk.Model(path)
            self._available = True
            logger.info("vosk 识别已就绪（模型 %s）", path)
        except Exception:
            logger.exception("加载 vosk 模型失败，语音识别降级为不可用")

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not self._available or not pcm:
            return ""
        import vosk

        try:
            # 每句话新建一个识别器：句与句之间是独立的，
            # 复用会把上一句的上下文带进来。
            recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
            recognizer.AcceptWaveform(pcm)
            result = self._json.loads(recognizer.FinalResult())
            return str(result.get("text") or "").replace(" ", "").strip()
        except Exception:
            logger.exception("vosk 识别失败")
            return ""

    def close(self) -> None:
        self._model = None
        self._available = False


class SpeechRecognitionRecognizer:
    """``speech_recognition`` 适配器。

    ``recognize_google`` 走网络（但免费、中文可用），``recognize_sphinx``
    是离线的但中文支持很差。默认用后者作为离线兜底，可用 ``B_STT_SR_ENGINE``
    切到 ``google``。

    音频格式上有个坑：``speech_recognition`` 要的是 WAV 容器，
    不是裸 PCM。这里用标准库 ``wave`` 现包一个，不引入 soundfile。
    """

    name = "speechrecognition"

    def __init__(self, engine: Optional[str] = None) -> None:
        import speech_recognition

        self._sr = speech_recognition
        self.engine = (engine or os.environ.get("B_STT_SR_ENGINE", "sphinx")).lower()

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""

        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm)
        buffer.seek(0)

        recognizer = self._sr.Recognizer()
        try:
            with self._sr.AudioFile(buffer) as source:
                audio = recognizer.record(source)
            if self.engine == "google":
                return recognizer.recognize_google(audio, language="zh-CN").strip()
            return recognizer.recognize_sphinx(audio).strip()
        except self._sr.UnknownValueError:
            return ""                                  # 没听懂，正常情况
        except Exception:
            logger.exception("speech_recognition 识别失败")
            return ""

    def close(self) -> None:
        return None


class DashScopeRecognizer:
    """阿里云 dashscope 识别（云端，需要 API key）。"""

    name = "dashscope"

    def __init__(self, api_key: Optional[str] = None) -> None:
        import dashscope

        self._dashscope = dashscope
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if key:
            dashscope.api_key = key

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""
        import base64
        import tempfile
        import wave

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(handle.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm)
            with open(handle.name, "rb") as source:
                encoded = base64.b64encode(source.read()).decode("ascii")

            response = self._dashscope.MultiModalConversation.call(
                model="qwen-audio-asr",
                messages=[{"role": "user", "content": [
                    {"audio": f"data:audio/wav;base64,{encoded}"}
                ]}],
            )
            content = response.output.choices[0].message.content
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) for part in content)
            return str(content).strip()
        except Exception:
            logger.exception("dashscope 识别失败")
            return ""
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

    def close(self) -> None:
        return None


#: 引擎名 → 构造函数
ENGINES = {
    "vosk": VoskRecognizer,
    "speechrecognition": SpeechRecognitionRecognizer,
    "dashscope": DashScopeRecognizer,
    "null": NullRecognizer,
    "none": NullRecognizer,
}


def installed_engines() -> List[str]:
    """依赖包在的引擎名（不管配没配好）。用 ``find_spec`` 探测，**不 import**。"""
    return [
        name for name in AUTO_ORDER
        if all(resolve_optional(package) for package in ENGINE_REQUIREMENTS[name])
    ]


def unavailable_reasons() -> List[str]:
    """"装了但没配好"的引擎及原因，用来打一条能直接照做的提示。

    这一条是为了让 ``auto`` 的失败**可见**：挑不到引擎时用户至少知道
    "dashscope 在，但缺 API key"，而不是面对一句笼统的"没有可用引擎"。
    """
    reasons = []
    for name in installed_engines():
        ready = READINESS.get(name)
        if ready is None or ready():
            continue
        if name == "dashscope":
            reasons.append(
                "dashscope 已安装，但没有 DASHSCOPE_API_KEY（云端识别需要密钥）"
            )
        elif name == "vosk":
            reasons.append(
                "vosk 已安装，但没有可用的中文模型（解压到 %s 或 %s，"
                "或设 B_STT_MODEL；路径不能含中文）"
                % (DEFAULT_MODELS_DIR, LOCAL_MODELS_DIR)
            )
        else:
            reasons.append(f"{name} 已安装但没有配置好")
    return reasons


def available_engines() -> List[str]:
    """当前环境**实际可用**的引擎名（按 AUTO_ORDER 排序）。

    判据是"装了 **且** 配好了"两件事，缺一不可 —— 只看包在不在的话，
    ``auto`` 会挑中一个每次识别都失败的引擎（见 READINESS 的说明）。
    用 ``find_spec`` 探测，**不 import** —— 见 base.resolve_optional 的说明。
    """
    return [
        name for name in installed_engines()
        if READINESS.get(name) is None or READINESS[name]()
    ]


def build_recognizer(name: str = "auto", **kwargs) -> Recognizer:
    """按名字构造识别引擎；``auto`` 取第一个可用的。

    **任何情况下都不抛异常。** 引擎缺失、构造失败、名字写错，
    一律记日志并退回 :class:`NullRecognizer`。理由见模块开头：
    B 必须能在没有语音依赖的机器上启动。
    """
    requested = (name or "auto").strip().lower()

    if requested in ("null", "none"):
        logger.info("语音识别已按配置停用（--stt none）")
        return NullRecognizer()

    if requested == "auto":
        candidates = available_engines()
        if not candidates:
            # 把"装了但没配好"和"根本没装"分开说 —— 前者用户只需要补一步配置，
            # 笼统地说"没有可用引擎"会让人以为要重新装包。
            reasons = unavailable_reasons()
            skipped = "\n  已跳过：".join(reasons) if reasons else ""
            # ⚠️ 用命名占位符，不要用 % 拼接多个字面量：
            #    `"a%sb" % x + "c%sd" % y` 会被解析成 ((a%x) + (c%y))，
            #    看着没错，但只要某一段没有占位符就会在运行期抛
            #    "not all arguments converted"，而那正是启动路径上最不该炸的地方。
            logger.warning(
                "没有检测到可用的语音识别引擎，语音输入不可用。%(skipped)s\n"
                "  可选方案（任选其一，配置好重启 B 即可自动启用）：\n"
                "      pip install vosk           然后把中文模型解压到 %(models)s\n"
                "                                 （或 %(local)s，路径不能含中文）\n"
                "      pip install SpeechRecognition\n"
                "      pip install dashscope      并设置 DASHSCOPE_API_KEY\n"
                "  在此之前：用键盘输入（--stdin），或从模块 C 的 8002 文本通道输入。",
                {"skipped": "\n  已跳过：" + skipped if skipped else "",
                 "models": DEFAULT_MODELS_DIR,
                 "local": LOCAL_MODELS_DIR},
            )
            return NullRecognizer()
        requested = candidates[0]
        logger.info("自动选择语音识别引擎：%s（可选：%s）",
                    requested, ", ".join(candidates))

    factory = ENGINES.get(requested)
    if factory is None:
        logger.warning("未知的语音识别引擎 %r，回退为空识别器（可选：%s）",
                       name, ", ".join(ENGINES))
        return NullRecognizer()

    try:
        return factory(**kwargs)
    except Exception:
        # 最典型的是"库装了但模型/密钥没配好"
        logger.exception("初始化语音识别引擎 %r 失败，回退为空识别器", requested)
        return NullRecognizer()
