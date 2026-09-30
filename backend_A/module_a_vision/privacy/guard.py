"""出站隐私闸门——A 模块"绝不外传像素"这条红线的**可执行**版本。

《系统设计方案》把"原始帧不出模块"写成了要求，但要求只是要求；
:func:`assert_clean` 把它变成一道**会在测试里失败的断言**。B 侧另有
:class:`shared.schema.PrivacyAttestation` 的自证字段做二次校验，两侧合起来
才算把这条红线钉死。

三道防线
--------

* :func:`assert_clean`——出站前递归扫描报文，发现敏感字段立即抛
  :class:`PrivacyViolationError`。
* :func:`scrub_frame`——帧用完后就地清零，实现"用完即毁"。
* :func:`redact_repr`——需要打日志时用它，它会把敏感字段替换成占位符。

展示用例外（V1.5 新增，**不影响本文件的任何一个字符**）
-----------------------------------------------------

前端右栏要显示摄像头画面，而画面只能来自 A。于是 A 多了一条
**展示用**的 MJPEG 流（:mod:`module_a_vision.stream`）。这里必须写清楚它
与本文件的关系，否则下一个人会以为"红线放宽了"：

* **本文件一个字符都没有改。** :data:`SENSITIVE_TOKENS` 仍然拦着
  ``bbox`` / ``pixel`` / ``jpeg`` / ``frame`` / ``video``，
  :data:`SAFE_KEYS` 仍然只有那七条**声明与计数**，:func:`assert_clean`
  仍然逐条扫过每一位出站报文。
* 展示流走的是 **HTTP 到 127.0.0.1**，与 :func:`assert_clean` 守的那条
  **TCP 8000 报文链路**是两条路。往 8000 的报文里塞像素，行为与从前
  一模一样：被拦下、被计数、被丢弃。
* 例外本身收在 :mod:`module_a_vision.stream` 的模块文档里，四条边界
  （只绑回环 / 默认关 / 不落盘 / 用完即毁）。**要放宽容忍度前先读那一段** ——
  那里写的是这个例外为什么仍然是安全的，而不是"因为要用所以放宽了"。
* :func:`scrub_frame` 从 V1.5 起**第一次真的被调用**（在展示流的编码之后，
  见 :meth:`VisionServer._publish_frame`）。此前它全仓没有任何调用点 ——
  一条写着"用完即毁"却从没执行过的代码，与没有这条代码是一样的。

与 ``PrivacyAttestation`` 的冲突（**这是一处必须处理的现实矛盾**）
-----------------------------------------------------------------

按最朴素的写法（"键名里出现 frame/image/raw 就报警"），A 自己的干净报文
会被判违规：

* ``privacy.raw_frame_uploaded``——布尔自证字段，报的是"**没有**上传原始帧"；
* ``privacy.face_image_uploaded``——同理；
* ``privacy.frame_retention``——字符串 "memory_only"；
* ``window.frames_total`` / ``frames_valid``——帧**计数**，不是帧；
* ``observations.*.frames_ratio``——"**占比**"（N-of-M 投票用的帧比例），
  既不是帧也不是计数。

最后一条是跑真实报文才发现的：它在三处观测里各出现一次，是最容易漏掉、
也最容易被"顺手把断言删掉"解决的一类误报。

因此匹配规则是"**分词 + 显式豁免**"：

1. 键名先规范化（camelCase 拆开、转小写、按非字母数字切词），
   逐**词**与敏感词表比对，而不是整串做子串匹配。
   ``frames_ratio`` 里的 "frames" 是敏感词，但整键在豁免表里，跳过。
2. :data:`SAFE_KEYS` 里的键是"名字带敏感词、语义却是声明或计数"的字段。
   它们**不含**任何像素与几何，放行是安全的；把它们一起拦掉只会逼着
   后来的人删掉这条断言。

这条豁免表是**唯一的**例外，加新条目需要同时确认该字段确实只是标志位
或计数——不要为了让某个报错消失而往里加东西。
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import Any

# ================================================================ 词表

#: 敏感词（按**词**匹配，不是子串）。命中即认为该字段可能携带像素或可还原
#: 人像的几何。复数形式显式列出——不做英语单复数推断，"bboxes" 这类词
#: 靠规则推是推不准的。
SENSITIVE_TOKENS: frozenset[str] = frozenset({
    # —— 任务规定的核心词表 ——
    "bbox", "bboxes",
    "landmark", "landmarks",
    "image", "images",
    "frame", "frames",
    "jpeg", "jpg",
    "png",
    "raw",
    "crop", "crops",
    "face_image",
    # —— 同类风险的补充词 ——
    "pixel", "pixels",
    "photo", "photos",
    "snapshot", "snapshots",
    "thumbnail", "thumbnails",
    "video", "videos",
})

#: 名字里带敏感词、但语义是"声明 / 计数"的字段，按**规范化后的整键**豁免。
#: 每一条都对应 :mod:`shared.schema` 里一个真实字段，理由见模块文档。
SAFE_KEYS: frozenset[str] = frozenset({
    "raw_frame_uploaded",     # PrivacyAttestation：布尔自证，恒为 False 才是对的
    "face_image_uploaded",    # PrivacyAttestation：同上
    "frame_retention",        # PrivacyAttestation："memory_only"
    "frames_total",           # WindowMeta：帧计数
    "frames_valid",           # WindowMeta：帧计数
    "frames_ratio",           # 三项观测里的"占比"（EmotionObs / HeadPoseObs / AttentionObs）
    "frame_index",            # 采集源自报的帧序号（日志字段，非数据）
})

#: 值本身是二进制/数组就一定违规——这条与键名无关，是**兜底**。
#: 有人给字段起个 "data" 之类的名字时，只有它能拦住。
_BLOB_TYPES: tuple[type, ...] = (bytes, bytearray, memoryview)

#: 递归深度上限。超过它说明报文异常深或存在循环引用，
#: 两者都不能被 JSON 序列化，直接拦住比栈溢出好。
MAX_DEPTH = 16

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD_RE = re.compile(r"[^0-9A-Za-z]+")


class PrivacyViolationError(RuntimeError):
    """出站报文含有可还原人像的字段。

    消息里必须带上**字段路径**：只说"报文违规"会让排查的人从几十个字段里
    一个个试，而这类问题通常出现在赶联调的深夜。
    """


# ================================================================ 键名判定

def normalize_key(key: str) -> str:
    """把键名规范化成小写下划线形式（``faceBBox`` → ``face_bbox``）。"""
    return _NON_WORD_RE.sub("_", _CAMEL_RE.sub("_", str(key))).strip("_").lower()


def key_tokens(key: str) -> list[str]:
    """把键名切成词。``frames_total`` → ``["frames", "total"]``。"""
    return [t for t in normalize_key(key).split("_") if t]


def is_sensitive_key(key: str) -> bool:
    """这个键名是否可能携带像素或可还原人像的几何。

    :func:`assert_clean` 与 :func:`redact_repr` 共用同一份判定，
    避免"能拦住但打印时漏了"或反过来的错位。
    """
    normalized = normalize_key(key)
    if normalized in SAFE_KEYS:
        return False
    return any(token in SENSITIVE_TOKENS for token in key_tokens(key))


def is_blob(value: Any) -> bool:
    """是否是二进制/数组一类"一看就是像素"的值。"""
    if isinstance(value, _BLOB_TYPES):
        return True
    # 不 import numpy：用鸭子类型判定，guard 保持零第三方依赖。
    return hasattr(value, "shape") and hasattr(value, "dtype")


# ================================================================ 出站校验

def find_violations(payload: Any) -> list[str]:
    """返回所有违规字段的路径。空列表表示干净。

    与 :func:`assert_clean` 用同一套规则，供测试与排查工具直接列出问题，
    不必去捕获异常再从消息里解析。
    """
    found: list[str] = []
    _scan(payload, path="", depth=0, found=found)
    return found


def assert_clean(payload: Any) -> None:
    """出站前校验：报文里不得有像素或可还原人像的几何。

    :param payload: 通常是 ``WindowState.to_dict()`` 的结果；
        也接受 dataclass 实例（如 ``WindowState`` 本身）——**这一点是刻意的**，
        调用方图省事直接传对象时不应该绕过检查。

    :raises PrivacyViolationError: 命中敏感字段（消息含字段路径），
        或嵌套过深/存在循环引用。

    ::

        assert_clean(window_state.to_dict())   # 干净 → 返回 None
        # 若报文里混进了 face_bbox → 抛 PrivacyViolationError，
        # 消息形如：出站报文含敏感字段：observations.face_bbox（"bbox"）
    """
    violations = find_violations(payload)
    if violations:
        raise PrivacyViolationError(
            "出站报文含敏感字段（A 模块不得外传像素或可还原人像的几何）：\n  "
            + "\n  ".join(violations)
            + "\n若这些字段确实只是标志位或计数，请把它加进 "
            "module_a_vision.privacy.guard.SAFE_KEYS 并说明理由。"
        )


def _scan(value: Any, *, path: str, depth: int, found: list[str]) -> None:
    """递归扫描。:param found: 收集违规路径（就地追加）。"""
    if depth > MAX_DEPTH:
        raise PrivacyViolationError(
            f"出站报文嵌套超过 {MAX_DEPTH} 层（或存在循环引用），位于 {path or '<根>'}。\n"
            "这类报文无法被 JSON 序列化，先修结构再谈隐私校验。"
        )

    if is_blob(value):
        found.append(f"{path or '<根>'}：值是二进制/数组（{type(value).__name__}）")
        return

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            child = f"{path}.{f.name}" if path else f.name
            if is_sensitive_key(f.name):
                found.append(f'{child}：{".".join(key_tokens(f.name))} 命中敏感词')
                continue
            _scan(getattr(value, f.name, None), path=child, depth=depth + 1, found=found)
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if is_sensitive_key(str(key)):
                found.append(f'{child}：命中敏感词 "{_hit_token(str(key))}"')
                continue
            _scan(item, path=child, depth=depth + 1, found=found)
        return

    # 字符串/字节之外的序列才递归；str 不是像素容器，别把它当列表拆。
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for i, item in enumerate(value):
            _scan(item, path=f"{path}[{i}]", depth=depth + 1, found=found)
        return

    if isinstance(value, (set, frozenset)):
        for i, item in enumerate(value):
            _scan(item, path=f"{path}[{i}]", depth=depth + 1, found=found)


def _hit_token(key: str) -> str:
    """命中的敏感词，用于报错消息。"""
    for token in key_tokens(key):
        if token in SENSITIVE_TOKENS:
            return token
    return normalize_key(key)


# ================================================================ 用完即毁

def scrub_frame(frame: Any) -> None:
    """就地把一帧的像素缓冲清零，实现"用完即毁"。

    :param frame: 可写的 numpy 数组；传 ``None`` 时静默返回
        （便于在 ``finally`` 里无脑调用，不必再判一次空）。

    :raises TypeError: 不是数组。
    :raises ValueError: 数组只读，清不掉——**这种情况必须报错**，
        因为"以为清了其实没清"比崩溃危险得多。

    ⚠️ 局限：只清这一块缓冲。若上游或别处已经持有同一帧的副本
    （``cv2`` 的某些后端会复用内部缓冲，转成 numpy 视图后清零是有效的，
    但显式 ``copy()`` 过的不会），那些副本不受影响。所以真正的保证来自
    "不复制、不出模块"，而不是这个函数。
    """
    if frame is None:
        return
    if not (hasattr(frame, "shape") and hasattr(frame, "fill")):
        raise TypeError(
            f"scrub_frame 需要一个 numpy 数组或 None，收到 {type(frame).__name__}。"
        )
    flags = getattr(frame, "flags", None)
    if flags is not None and not flags.writeable:
        raise ValueError(
            "该数组是只读的，无法就地清零。请确保帧在采集时就拿到了可写缓冲——"
            "“以为清零了其实没有”比报错危险得多。"
        )
    frame.fill(0)


# ================================================================ 安全打印

#: 单个字符串值的最大打印长度。
_MAX_STR = 120
#: 容器最多打印多少项，超出部分折叠。
_MAX_ITEMS = 12
#: dataclass 最多打印多少个字段。
#:
#: 比容器宽一些是刻意的：``FrameFeatures`` 有 16 个字段，其中两个是敏感字段
#: （人脸框、478 个关键点）。用 12 截断会把它们一起吞掉——虽然"不打印"也算
#: 安全，但日志里连"这里有个字段被脱敏了"都看不到，排查的人就无从判断
#: 这个对象到底带了什么。宁可多打几个字段名。
_MAX_FIELDS = 24
#: 最终字符串的最大长度。
_MAX_REPR = 2000
#: 脱敏的最大递归深度。
_MAX_REDACT_DEPTH = 6
#: 占位符。
REDACTED = "<已脱敏>"


def redact_repr(obj: Any) -> str:
    """安全的 repr：用于日志，敏感字段会被替换成 :data:`REDACTED`。

    与直接 ``repr()`` 的区别就是那些字段——``FrameFeatures`` 与
    ``WindowState`` 里都有"打印一次就等于泄漏一次"的内容（人脸框、
    478 个关键点），而日志恰恰是最容易被顺手贴进聊天窗口的东西。

    dataclass 按字段逐个脱敏后按 ``类名(字段=值)`` 的形式还原，
    保持与 dataclass 自带 repr 一致的观感；超长内容会被折叠。
    """
    try:
        text = _redact(obj, depth=0)
    except Exception as exc:  # noqa: BLE001 - 打日志绝不能因为打日志而崩
        return f"<{type(obj).__name__} 脱敏失败：{type(exc).__name__}>"
    if len(text) > _MAX_REPR:
        text = f"{text[:_MAX_REPR]}…（已截断，完整长度 {len(text)} 字符）"
    return text


def _redact(value: Any, *, depth: int) -> str:
    if depth > _MAX_REDACT_DEPTH:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if is_blob(value):
        n = getattr(value, "size", None) or getattr(value, "nbytes", None) or len(value)  # type: ignore[arg-type]
        return f"<二进制 {n} 字节>"
    if isinstance(value, str):
        return repr(value if len(value) <= _MAX_STR else f"{value[:_MAX_STR]}…")

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        parts: list[str] = []
        fields = dataclasses.fields(value)
        for f in fields[:_MAX_FIELDS]:
            if is_sensitive_key(f.name):
                parts.append(f"{f.name}={REDACTED}")
            else:
                parts.append(f"{f.name}={_redact(getattr(value, f.name, None), depth=depth + 1)}")
        if len(fields) > _MAX_FIELDS:
            parts.append(f"…共 {len(fields)} 个字段")
        return f"{type(value).__name__}({', '.join(parts)})"

    if isinstance(value, Mapping):
        parts = []
        items = list(value.items())
        for key, item in items[:_MAX_ITEMS]:
            if is_sensitive_key(str(key)):
                parts.append(f"{key!r}: {REDACTED}")
            else:
                parts.append(f"{key!r}: {_redact(item, depth=depth + 1)}")
        if len(items) > _MAX_ITEMS:
            parts.append(f"…共 {len(items)} 项")
        return "{" + ", ".join(parts) + "}"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_redact(item, depth=depth + 1) for item in list(value)[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            parts.append(f"…共 {len(value)} 项")
        return "[" + ", ".join(parts) + "]"

    if isinstance(value, (set, frozenset)):
        parts = [_redact(item, depth=depth + 1) for item in list(value)[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            parts.append(f"…共 {len(value)} 项")
        return "{" + ", ".join(parts) + "}"

    return repr(value)[:_MAX_STR]


__all__ = [
    "MAX_DEPTH",
    "REDACTED",
    "SAFE_KEYS",
    "SENSITIVE_TOKENS",
    "PrivacyViolationError",
    "assert_clean",
    "find_violations",
    "is_blob",
    "is_sensitive_key",
    "key_tokens",
    "normalize_key",
    "redact_repr",
    "scrub_frame",
]
