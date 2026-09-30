# -*- coding: utf-8 -*-
"""api_doc §3.5 的类型化报文：识别、校验、分流。

A→B 的 v1 流上现在有**三种**东西（api_doc §3.5.1）：

    视觉帧      无 ``type``          §3.2 的平铺 8 字段
    rppg        ``"type":"rppg"``    §3.5.2 体征
    focus       ``"type":"focus"``   §3.5.3 视线原始量

判别规则只有一句：**有 ``type`` 就查白名单，没有就是帧。**

为什么这件事必须显式做，而不能"让它自己落到帧分支"
----------------------------------------------------

:meth:`VisionSample.from_payload` 对**未知键静默忽略**，对**缺失键回退安全默认**
（``has_face=False``）。所以一条 ``{"type":"rppg",...}`` 如果没人拦，
会被兜成一个"看不见人"的样本推进判定器 —— 而 A 每秒钟都在发一条。
表现是**周期性伪造"老人不在"**，把状态往 ``absent`` 拖。

这不是假想的：A 在 v1 模式下**刻意关掉了心跳**（``backend_A`` 的 ``server.serve``）
就是为了躲开同一个失效 —— 心跳报文没有那 8 个字段，进来同样是幽灵帧。
新开的门不能把这套失效再放进来一次。

所以：``type`` 存在但不在白名单里 → **丢弃 + 计数**，绝不落进帧分支。
宁可少一帧（B 有失联判定兜着，会显式地降级），也不要一帧假的"看不见人"。

**本模块是唯一的判别点。** :class:`VisionClient`（实时）与
:class:`~core.vision_client.OfflineVisionFeeder`（离线回放）各自持有一个
:class:`TypedMessageRouter`，而不是各写一份 if —— 两条路对同一份报文的处理
必须逐字相同，否则"实时对、回放错"这类差异只能在演示当天被发现。

接收方策略（与 A 侧**刻意不同**）
----------------------------------

A 侧 ``assert_*_contract`` 要求**键集合精确相等**（它是生产方，要保证自己只发
规定的东西）。B 侧是接收方，按 Postel 定律**容忍多余的键**：将来 A 给 ``rppg``
加一个有意义的字段时，旧版 B 应当继续工作，而不是整条报文被拒。
所以下面只校验"必须有的在不在、类型对不对、范围合不合理"。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

#: api_doc §3.5 已确认的报文类型。**改这里等于改协议**，要先动 api_doc。
MSG_RPPG = "rppg"
MSG_FOCUS = "focus"
TYPED_MESSAGE_TYPES = (MSG_RPPG, MSG_FOCUS)

#: 与 api_doc §3.5.2 / §3.5.3 一致的取值区间。
HR_RANGE = (30.0, 240.0)
RR_RANGE = (4.0, 40.0)
IBI_RANGE = (200.0, 3000.0)
MAX_IBI_COUNT = 64

#: 未知类型最多打几条 WARNING —— A 一旦发错，每秒一条会刷屏刷掉别的日志。
_UNKNOWN_LOG_LIMIT = 3


class TypedMessageError(ValueError):
    """类型化报文不符合 api_doc §3.5。调用方应丢弃该条并计数，不要中断连接。"""


@dataclass(frozen=True, slots=True)
class RppgReading:
    """一条 §3.5.2 体征报文。"""

    timestamp: float
    hr: Optional[int]
    rr: Optional[float]
    ibi_ms: tuple

    @property
    def has_pulse(self) -> bool:
        """算法有没有算出心率。

        ``False`` 是**正常态**：A 的窗口未满 8 秒、SNR<1.5、或有效帧率<5Hz 时
        都会置灰。任何"收到 rppg ⇒ hr 一定是数字"的假设都会在服务刚起来时误报。
        """
        return self.hr is not None


@dataclass(frozen=True, slots=True)
class FocusReading:
    """一条 §3.5.3 视线报文。"""

    timestamp: float
    gaze: Optional[float]
    gaze_quality: float

    @property
    def usable(self) -> bool:
        """这一帧的视线能不能用。

        **判据是"有没有 ``gaze``"，不是 ``gaze_quality`` 的数值。**
        ``gaze_quality`` 是 A 给自己的估计打的分，B 拿它去开自己的门控等于
        让被测方给自己的考试打分。质量分的唯一用途是排查问题时看一眼。
        """
        return self.gaze is not None


# ---------------------------------------------------------------------------
# 校验辅助
# ---------------------------------------------------------------------------

def _as_float(value: Any) -> Optional[float]:
    """把可能是数字、也可能是数字**字符串**的值转成 float；转不了返回 ``None``。

    为什么要认字符串：报文有**两条**来路。实时 TCP 上是 JSON，数字就是数字；
    而 ``--offline`` 读的 CSV / 抓包 dump 里，每个格子都是字符串
    （``"timestamp": "12.50"``）。:meth:`VisionSample.from_payload` 早就
    两样都认（``float(value)`` 包在 try 里），这里保持一致 —— 否则同一份数据
    实时跑得通、回放全被丢掉。

    ``bool`` 必须显式排除：``float(True)`` 是 ``1.0``，于是 ``"hr": true``
    会变成心率 1、"``gaze``": true`` 会变成 1.0（= 完全对正，专注度满分）。

    **``nan`` / ``inf`` 也一律拒绝**，理由不是洁癖：``float("nan")`` 是能成功的，
    它会一路走到后面的 ``int(value)`` 才炸，而那时抛的是**裸的** ``ValueError``
    / ``OverflowError`` —— 不是 :class:`TypedMessageError`，路由器的 ``except``
    接不住。那一下会穿过 ``_read_loop``（它只接 ``ProtocolError``）和 ``run()``
    （只接 ``OSError`` 家族），**直接把视觉读取线程打死**：状态停在上一次的值
    不再更新，而日志里只有一条栈回溯，看起来像崩溃而不是"某一行数据脏了"。
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _require_timestamp(payload: Dict[str, Any]) -> float:
    value = _as_float(payload.get("timestamp"))
    if value is None:
        raise TypedMessageError(f"timestamp 必须是数字，实际是 {payload.get('timestamp')!r}")
    return value


def _optional_number(
    payload: Dict[str, Any], key: str, lo: float, hi: float
) -> Optional[float]:
    """取一个可为 ``null`` 的数字，并把范围检查一起做掉。

    ``null``、空串与"键缺失"都返回 ``None`` —— 对接收方来说它们没有区别。
    """
    raw = payload.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    value = _as_float(raw)
    if value is None:
        raise TypedMessageError(f"{key} 必须是数字或 null，实际是 {raw!r}")
    if not (lo <= value <= hi):
        raise TypedMessageError(f"{key}={value} 越界，应在 [{lo}, {hi}]")
    return value


def _parse_rppg(payload: Dict[str, Any]) -> RppgReading:
    timestamp = _require_timestamp(payload)
    hr = _optional_number(payload, "hr", *HR_RANGE)
    rr = _optional_number(payload, "rr", *RR_RANGE)

    raw_ibi = payload.get("ibi_ms")
    if raw_ibi is None or (isinstance(raw_ibi, str) and not raw_ibi.strip()):
        raw_ibi = []
    if isinstance(raw_ibi, str):
        # CSV / dump 里数组被写成了字符串（``"[780, 780]"``）。这不是数字
        # 强转能解决的，得按 JSON 解一次；解不出来就是真的坏了，丢弃该条。
        try:
            raw_ibi = json.loads(raw_ibi)
        except (ValueError, TypeError) as exc:
            raise TypedMessageError(f"ibi_ms 不是合法数组：{raw_ibi!r}（{exc}）") from exc
    if not isinstance(raw_ibi, (list, tuple)):
        raise TypedMessageError(f"ibi_ms 必须是数组，实际是 {type(raw_ibi).__name__}")
    if len(raw_ibi) > MAX_IBI_COUNT:
        raise TypedMessageError(f"ibi_ms 有 {len(raw_ibi)} 条，超过上限 {MAX_IBI_COUNT}")
    ibi = []
    for item in raw_ibi:
        value = _as_float(item)
        if value is None or value != int(value):
            raise TypedMessageError(f"ibi_ms 里必须都是整数毫秒，实际有 {item!r}")
        if not (IBI_RANGE[0] <= value <= IBI_RANGE[1]):
            raise TypedMessageError(f"ibi_ms={item} 越界，应在 [{IBI_RANGE[0]}, {IBI_RANGE[1]}]")
        ibi.append(int(value))

    return RppgReading(
        timestamp=timestamp,
        hr=None if hr is None else int(hr),
        rr=rr,
        ibi_ms=tuple(ibi),
    )


def _parse_focus(payload: Dict[str, Any]) -> FocusReading:
    timestamp = _require_timestamp(payload)
    gaze = _optional_number(payload, "gaze", 0.0, 1.0)

    raw_quality = payload.get("gaze_quality")
    if raw_quality is None or (isinstance(raw_quality, str) and not raw_quality.strip()):
        raw_quality = 0.0
    quality = _as_float(raw_quality)
    if quality is None:
        raise TypedMessageError(f"gaze_quality 必须是数字，实际是 {raw_quality!r}")
    if not (0.0 <= quality <= 1.0):
        raise TypedMessageError(f"gaze_quality={quality} 越界，应在 [0, 1]")

    # api_doc §3.5.3 的不变式，**两个方向都查**。
    #
    # 这一条不是在替 A 做检查，而是在堵一个具体的坑：A 侧在**没有虹膜关键点**
    # 的模型上估不出视线，而"估不出来"与"完全对正"在数值上都是 0.0。
    # 一旦有人把 None 改写回 0.0，专注度就会把"看不见眼睛"读成"专注满分" ——
    # 一个方向是假的满分，另一个方向是"有值却说没质量"，都不能放过。
    if (gaze is None) != (quality == 0.0):
        raise TypedMessageError(
            f"gaze={gaze!r} 与 gaze_quality={quality} 不自洽："
            f"gaze 为 null 当且仅当 gaze_quality == 0（api_doc §3.5.3）"
        )

    return FocusReading(timestamp=timestamp, gaze=gaze, gaze_quality=quality)


_PARSERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    MSG_RPPG: _parse_rppg,
    MSG_FOCUS: _parse_focus,
}


def parse_typed(payload: Dict[str, Any]) -> Any:
    """把一条类型化报文解析成对应的 Reading。不合规抛 :class:`TypedMessageError`。"""
    kind = payload.get("type")
    parser = _PARSERS.get(kind)
    if parser is None:
        raise TypedMessageError(f"未知报文类型 {kind!r}")
    return parser(payload)


# ---------------------------------------------------------------------------
# 路由器
# ---------------------------------------------------------------------------

class TypedMessageRouter:
    """把带 ``type`` 的报文从帧流里摘出来。

    调用方只有一件事要做：每条解码后的报文都先问一次
    :meth:`handle`；返回 ``True`` 表示**这条已经被消费掉了**，不要再当帧处理。
    """

    def __init__(self, on_reading: Optional[Callable[[str, Any], None]] = None) -> None:
        #: ``(kind, reading)`` 回调。③b 的 FocusTracker 从这里拿 gaze。
        #: 默认 ``None`` —— 也就是**没有订阅者时，报文被计数后丢弃**，
        #: 而不是将就着当帧用。
        self.on_reading = on_reading
        #: 各类型累计收到多少条。
        self.counts: Dict[str, int] = {}
        #: 白名单外的 ``type``：丢弃 + 计数。
        self.unknown = 0
        #: 在白名单内但字段不合规：丢弃 + 计数。
        self.malformed = 0
        #: **我们自己的解析器抛了非预期异常**的次数。与 ``malformed`` 分开：
        #: 非零说明这里有个 bug 要修，而不是"对端数据脏"。—— 联调时这两个数
        #: 指向完全不同的排查方向。
        self.errors = 0
        self._unknown_logged = 0

    def handle(self, payload: Dict[str, Any]) -> bool:
        """``True`` = 这是一条类型化报文（无论收下还是丢弃），调用方别再当帧。"""
        if "type" not in payload:
            return False

        kind = payload.get("type")
        if kind not in TYPED_MESSAGE_TYPES:
            self.unknown += 1
            if self._unknown_logged < _UNKNOWN_LOG_LIMIT:
                self._unknown_logged += 1
                logger.warning(
                    "A 发来未知报文类型 %r，已丢弃（api_doc §3.5 只认 %s）。"
                    "注意：**不会**把它当帧处理 —— 那会伪造一帧 has_face=false",
                    kind, "、".join(TYPED_MESSAGE_TYPES),
                )
            return True

        try:
            reading = parse_typed(payload)
        except TypedMessageError as exc:
            self.malformed += 1
            logger.warning("丢弃不符合 api_doc §3.5 的 %s 报文：%s", kind, exc)
            return True
        except Exception:
            # **兜底这一层是有意的，不是偷懒。** 解析器里任何一个没预料到的
            # 异常（比如某个字段上冒出来的 ``OverflowError``）如果放出去，
            # 会穿过 ``_read_loop``（只接 ProtocolError）和 ``run()``（只接
            # OSError 家族），**把视觉读取线程打死** —— 状态停在上一次的值
            # 不再更新，日志里只有一条栈回溯，看起来像崩溃而不是"某一行数据脏了"。
            #
            # 单独计进 ``errors`` 而不是 ``malformed``：前者是**我们代码的 bug**
            # （该修），后者是对端发了坏数据（正常容忍）。合成一个数的话，
            # "malformed 非零"这句诊断就同时指向两个完全不同的排查方向。
            self.errors += 1
            logger.exception("解析 §3.5 %s 报文时抛出非预期异常，已丢弃该条", kind)
            return True

        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.counts[kind] == 1:
            logger.info("收到首条 §3.5 %s 报文：%s", kind, reading)

        if self.on_reading is not None:
            try:
                self.on_reading(kind, reading)
            except Exception:
                # 订阅方是上层的事，它崩了不该带崩读取循环
                logger.exception("on_reading 回调抛出异常，已忽略")
        return True

    def summary(self) -> str:
        """一行摘要，联调时看这个就够了。"""
        if not (self.counts or self.unknown or self.malformed or self.errors):
            return "未收到任何 §3.5 类型化报文"
        parts = [f"{kind}×{count}" for kind, count in sorted(self.counts.items())]
        if self.unknown:
            parts.append(f"未知类型×{self.unknown}（已丢弃）")
        if self.malformed:
            parts.append(f"不合规×{self.malformed}（已丢弃）")
        if self.errors:
            parts.append(f"解析器异常×{self.errors}（**这是 B 的 bug**）")
        return " §3.5 类型化报文：" + "，".join(parts)


__all__ = [
    "FocusReading",
    "HR_RANGE",
    "IBI_RANGE",
    "MSG_FOCUS",
    "MSG_RPPG",
    "MAX_IBI_COUNT",
    "RR_RANGE",
    "RppgReading",
    "TYPED_MESSAGE_TYPES",
    "TypedMessageError",
    "TypedMessageRouter",
    "parse_typed",
]
