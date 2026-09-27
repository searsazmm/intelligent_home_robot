"""动作（Action）——B 核心与外界交互的唯一方式。

B 的判定核心是**纯同步、无 I/O** 的：它不碰 socket、不碰时钟、不碰数据库，
只接收输入、返回一个 :data:`Action` 列表。真正的副作用由
``module_b_agent.runtime`` 统一执行。

这样做的收益是决定性的：

* 全部安全逻辑可以用构造好的 ``WindowState`` 驱动，微秒级跑完，无需网络、
  无需摄像头、无需大模型，也无需 ``sleep``——把十分钟的推送场景压进一次
  单元测试。
* "取消待发的 60 秒无应答定时器"这类操作，在核心层只是一个
  :class:`CancelTimer` 动作，不必在业务逻辑里操纵 asyncio 任务。

新增动作类型时，记得同步 ``runtime`` 里的分发分支。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import Level, Scene, Tone


class Action:
    """动作基类。所有动作为不可变值对象。"""

    __slots__ = ()


# ---------------------------------------------------------------- 对老人

@dataclass(frozen=True, slots=True)
class Speak(Action):
    """让 C 播报一句话。"""

    text: str
    tone: Tone = Tone.WARM
    #: 语速倍数。协议层用浮点，由 TTS 引擎自行映射（SSML prosody 或 SAPI Rate）。
    speed: float = 0.85
    expect_reply: bool = True
    #: 等待老人回应的秒数；0 表示不等待。由 **B 负责计时**，C 不做超时判定。
    wait_reply_sec: int = 30
    scene: Scene = Scene.NONE
    trace_id: str = ""


@dataclass(frozen=True, slots=True)
class SetDisplay(Action):
    """更新 C 的状态展示。兼容接口文档 V1 的四种状态字符串。"""

    state: str  # normal | sad | tired | absent


@dataclass(frozen=True, slots=True)
class StartListen(Action):
    """让 C 开始采集老人语音。"""

    timeout_sec: int = 10


@dataclass(frozen=True, slots=True)
class StopListen(Action):
    """让 C 停止采集。"""


# ---------------------------------------------------------------- 对家属

@dataclass(frozen=True, slots=True)
class PushFamily(Action):
    """推送家属。

    ⚠️ **本动作只应由规则闸门的推送仲裁产生。** 决策 Agent（LLM）无权
    生成它，也无权否决它——让大模型掌握"要不要通知家属"的否决权，
    等于把安全底线交给一个概率模型。
    """

    category: str
    level: Level
    summary: str
    suggested_action: str
    dedup_key: str
    receivers: tuple[str, ...] = ()
    #: 事件明细，供家属端展示"为什么推送"。
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- 内部

@dataclass(frozen=True, slots=True)
class ScheduleRetry(Action):
    """安排一次延后动作（如无应答后的二次呼叫）。"""

    after_sec: float
    reason: str
    trace_id: str = ""


@dataclass(frozen=True, slots=True)
class CancelTimer(Action):
    """取消待发的延后动作。

    老人一开口就立刻取消"无应答升级"定时器——这是抢占规则的关键一步。
    """

    reason: str


@dataclass(frozen=True, slots=True)
class RecordEvent(Action):
    """把一次事件写入持久化存储（用于三振计数、冷却、审计）。"""

    category: str
    level: Level
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Log(Action):
    """记录一条内部日志。``intent`` 一类的内部推理不应播给老人，但需要留痕。"""

    message: str
    level: str = "info"
