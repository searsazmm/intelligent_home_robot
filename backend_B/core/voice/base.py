# -*- coding: utf-8 -*-
"""语音引擎的接口定义。

用 ``typing.Protocol`` 而不是抽象基类，理由和 ``history_store.HistoryProvider``
一样：适配器不必 import 本模块就能满足接口，测试里塞一个假引擎也不用继承任何东西。
结构对上了就是实现了。

**这里不 import 任何第三方库**，也不 import sounddevice —— 接口文件被到处 import，
一旦它带来了 PortAudio，就再也没有"无音频设备也能跑测试"这回事了。
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Recognizer(Protocol):
    """语音识别引擎：PCM 进，文本出。"""

    #: 引擎名，进日志用（"sapi" / "vosk" / "null" …）
    name: str

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """把一段 16bit 单声道 PCM 转成文本。

        契约：
            - 识别不出/输入为空时返回**空字符串**，不返回 None、不抛异常。
              调用方约定用 ``if text:`` 判断，破约会静默产生一堆
              "机器人对着一片空白回答"的诡异日志。
            - 引擎内部错误应自己吞掉并记日志 —— 一句话识别失败，
              不该让整个语音循环退出。
        """
        ...

    def close(self) -> None:
        """释放资源。必须幂等：关闭路径可能被调用多次。"""
        ...


@runtime_checkable
class Synthesizer(Protocol):
    """语音合成引擎：文本进，WAV 落盘。

    刻意做成"落盘"而不是"播放"：合成与播放解耦之后，
    「WAV 是否存在且非空」就成了一个可断言的测试点，
    而播放设备的问题（音量被调低、误选到虚拟声卡之类）不会污染合成逻辑。
    """

    name: str

    def synthesize_wav(self, text: str, path: str) -> bool:
        """把 text 合成到 path。成功返回 True。

        失败要返回 False 并记日志，不要抛 —— 语音播报失败不该中断对话。
        """
        ...

    def close(self) -> None:
        """释放资源（比如杀掉常驻的 PowerShell 子进程）。必须幂等。"""
        ...


class NullRecognizer:
    """什么都没装时的兜底：永远识别不出内容。

    存在的意义是**让 B 在没有语音依赖时照常启动** ——
    缺库时降级为文本/键盘通道，而不是启动失败。
    """

    name = "null"

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        return ""

    def close(self) -> None:
        return None


class NullSynthesizer:
    """什么都没装时的兜底：不发声，只记日志。"""

    name = "null"

    def synthesize_wav(self, text: str, path: str) -> bool:
        return False

    def close(self) -> None:
        return None


def describe(engines: List[object]) -> str:
    """把一组引擎名拼成一行，给启动日志用。"""
    return ", ".join(getattr(engine, "name", "?") for engine in engines) or "（无）"


def resolve_optional(name: str) -> Optional[str]:
    """探测某个可选依赖是否可用。**不执行模块**。

    用 ``find_spec`` 而不是 ``import``：``import sounddevice`` 会初始化
    PortAudio 并枚举设备（几百毫秒），仅仅是"检查装没装"不该有这个代价。
    """
    import importlib.util

    try:
        return name if importlib.util.find_spec(name) is not None else None
    except (ImportError, ValueError):
        return None
