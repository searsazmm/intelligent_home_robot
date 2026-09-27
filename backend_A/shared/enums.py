"""跨模块共用的枚举定义。

严格对齐《系统设计方案》§3.4（视觉枚举）与 §4.4（触发等级）。

⚠️ 枚举的**值**就是线路上传输的字符串。《系统总接口文档》§2.2 规定
"接口字段、传输格式一经确定，禁止私自修改"——改动这里的字符串等同于
破坏模块间协议，必须走全员确认流程。
"""

from __future__ import annotations

from enum import StrEnum


# ---------------------------------------------------------------- 视觉观测

class Emotion(StrEnum):
    """面部表情，对齐设计方案 §3.4。"""

    NORMAL = "normal"   # 正常
    TIRED = "tired"     # 疲惫
    SAD = "sad"         # 难过
    UPSET = "upset"     # 烦闷


class HeadPose(StrEnum):
    """头部姿态。"""

    UPRIGHT = "upright"  # 头部端正
    BOWED = "bowed"      # 低头
    TILTED = "tilted"    # 头部歪倒


class Attention(StrEnum):
    """注意力状态。"""

    FOCUSED = "focused"  # 专注
    ABSENT = "absent"    # 发呆失神


class FatigueLevel(StrEnum):
    """疲劳等级。"""

    NONE = "none"        # 无疲劳
    MILD = "mild"        # 轻度疲劳
    SEVERE = "severe"    # 重度疲劳


class EyeState(StrEnum):
    """眼部状态。"""

    NORMAL_BLINK = "normal_blink"  # 正常眨眼
    HALF_CLOSED = "half_closed"    # 半闭眼
    CLOSED = "closed"              # 持续闭眼


# ---------------------------------------------------------------- 画面质量

class Illumination(StrEnum):
    NORMAL = "normal"
    LOW = "low"
    OVEREXPOSED = "overexposed"


class Blur(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Occlusion(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    SEVERE = "severe"


class Glasses(StrEnum):
    NONE = "none"
    READING = "reading"        # 老花镜
    SUNGLASSES = "sunglasses"  # 墨镜，眼部关键点不可信


# ---------------------------------------------------------------- 调度决策

class Level(StrEnum):
    """触发等级，对齐设计方案 §4.4。

    L3+（呼唤无应答）不是独立等级，而是 L3 的升级态，
    由 L3 决策中的 retry_policy.escalate_on_no_reply 表达。
    """

    L0 = "L0"  # 无异常
    L1 = "L1"  # 情绪关怀
    L2 = "L2"  # 休息提醒
    L3 = "L3"  # 身体疑似异常
    L4 = "L4"  # 数据断流（设备类）


#: 等级高低排序。升级放行、取最高等级等逻辑都依赖它。
LEVEL_ORDER: dict[Level, int] = {
    Level.L0: 0,
    Level.L1: 1,
    Level.L2: 2,
    Level.L3: 3,
    Level.L4: 4,
}


def is_higher(a: Level, b: Level) -> bool:
    """a 是否严格高于 b。"""
    return LEVEL_ORDER[a] > LEVEL_ORDER[b]


class L4Reason(StrEnum):
    """L4（数据断流）的成因。

    两种成因的处置方式完全不同，**不应混为一谈**：

    * ``LINK_LOST``——与 A 的连接断了。查进程、查端口。
    * ``VISION_UNUSABLE``——连接还在，但画面长期不可用（遮挡、全黑）。
      查摄像头、查遮挡。此时 A 仍在发心跳，只有靠 ``usable`` 才能发现。

    若共用一个名字，会出现"心跳还在却报断流"或"拔了网线却报遮挡"这类
    误导性告警，家属和运维都无从下手。
    """

    LINK_LOST = "link_lost"
    VISION_UNUSABLE = "vision_unusable"


class Scene(StrEnum):
    """交互场景。"""

    GREETING = "greeting"                # 定时问候
    COMFORT = "comfort"                  # 情绪关怀
    REST_REMINDER = "rest_reminder"      # 休息提醒
    BODY_CHECK = "body_check"            # 身体状态关心
    NONE = "none"                        # 不动作


class Tone(StrEnum):
    WARM = "warm"          # 日常亲切
    GENTLE = "gentle"      # 柔和
    CALM = "calm"          # 平静稳定
    CHEERFUL = "cheerful"  # 轻快


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Period(StrEnum):
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class SessionStatus(StrEnum):
    """会话状态机，对齐设计方案 §4.6。"""

    IDLE = "idle"                # 空闲
    ACTIVE = "active"            # 老人正在说话（最高优先级）
    AWAIT_REPLY = "await_reply"  # 已询问，等待回应
    ESCALATED = "escalated"      # 无应答，已升级


class EventCategory(StrEnum):
    """家属推送的事件类别，用于推送仲裁与去重。"""

    BODY_ABNORMAL = "suspected_body_abnormal"
    NO_REPLY = "no_reply"
    DEVICE_OFFLINE = "device_offline"
    LOW_MOOD = "low_mood"
