# -*- coding: utf-8 -*-
"""右栏（画面 / 识别结果）里那些文字 —— 纯函数，**不 import Qt**。

和 ``c_core/expressions.py`` 同一条约定：文案与渲染是两件事。分开之后，
"专注度该怎么显示"这件事可以脱离图形环境单测（``tests/test_display_text.py``），
而改文案不必动窗口代码。

--------------------------------------------------------------------------
这一层存在的真正理由：**index 是 None 不等于 0**
--------------------------------------------------------------------------
B 的 ``vai`` 报文里 ``index`` 允许是 ``None`` —— 表示"没有分数"（校准还没锁定、
有效观察时长不够、或者门控把某几路证据拦了），而**不是**"专注度很低"。
这两个在界面上必须长得不一样，否则老人看到的是系统在指责他走神。

所以 :func:`format_index` 对 ``None`` 返回破折号而不是 ``0.0``，
并且把 B 给的 ``index_status``（"有效观察时长不足"这类）原样显示出来 ——
它是**唯一**能解释"为什么没有分数"的东西。

--------------------------------------------------------------------------
还有一条：这里一个字都不许改写 ``note``
--------------------------------------------------------------------------
``note`` 里那两半（「研究趋势（非认知专注）」+「日常参考，非医疗结论」）
是这套指数的安全声明，由 B 侧 ``ui_channel.VAI_NOTE`` 定稿。C 只负责显示，
包括**缺字段时的兜底值**也照抄同一句 —— 兜底时自己编一句更顺口的说法
（比如"专注度良好"）恰恰是最糟的做法。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------
# 文案常量
# --------------------------------------------------------------------------

#: 没有指数时的占位。**不是 ``0.0``**，理由见模块文档。
NO_INDEX = "——"

#: 一直没收到 ``vai`` 报文时的说法。分两种原因，见 :func:`format_vai_lines`。
VAI_NOT_DELIVERED = "未提供 —— B 未下发（可能没开专注度展示，或 B 是旧版本）"

#: ``vai`` 报文里缺少 ``note`` 字段时的兜底（含旧版 B 的情况）。
#: 与 ``backend_B/core/ui_channel.py`` 的 ``VAI_NOTE`` 逐字相同 ——
#: C 不许自己发明一句更顺口的免责声明。
FALLBACK_NOTE = "研究趋势（非认知专注）；日常参考，非医疗结论"

#: 模态键 → 中文。措辞与 B 侧 ``core/focus.py`` 的 ``_MODALITY_CONFIG_MAP``
#: 保持一致（凝视/头姿/睁眼），这样左栏日志与右栏界面说的是同一套话。
MODALITY_LABELS: Dict[str, str] = {
    "gaze": "凝视",
    "pose": "头姿",
    "head_pose": "头姿",
    "eye_open": "睁眼",
    "ear": "睁眼",
}

#: 没有画面时的占位文字
STREAM_OFFLINE = "画面未接入"
STREAM_HINT = "启动 A 时加 --stream，并给 C 加 --stream-url"
#: 画面通道连上了但还没收到第一帧（步骤②的 MJPEG 拉流会用到）
STREAM_WAITING = "画面通道已连接，等待第一帧"

#: 进度条字符（实心 / 空心）。U+25A0 / U+25A1 在 Microsoft YaHei UI 里有字形，
#: 但**必须**在演示机上用 ``--screenshot`` 肉眼过一遍（字体回退会让缺字形静默变方块）。
BAR_FILLED = "■"
BAR_EMPTY = "□"

#: 进度条格子数
BAR_CELLS = 10


# --------------------------------------------------------------------------
# 指数
# --------------------------------------------------------------------------

def format_index(index: Optional[float]) -> str:
    """指数 → 显示串。``None`` → 破折号（**不是 0.0**）。"""
    if index is None:
        return NO_INDEX
    try:
        return f"{float(index):.1f}"
    except (TypeError, ValueError):
        return NO_INDEX


def format_index_bar(index: Optional[float], cells: int = BAR_CELLS) -> str:
    """把 0–100 画成一条纯文本进度条。``None`` 时返回对应长度的空心条。

    用文本而不是 Qt 画的矩形：一是这一层不许 import Qt，二是投影仪上
    这种字符条比一条细线看得清。没有分数时**整条都是空的** ——
    空条加破折号，和"低分"在视觉上不会混。
    """
    cells = max(1, int(cells))
    if index is None:
        return BAR_EMPTY * cells
    try:
        value = float(index)
    except (TypeError, ValueError):
        return BAR_EMPTY * cells
    filled = int(round(max(0.0, min(100.0, value)) / 100.0 * cells))
    return BAR_FILLED * filled + BAR_EMPTY * (cells - filled)


# --------------------------------------------------------------------------
# 模态与专度文本
# --------------------------------------------------------------------------

def format_modalities(keys: Optional[Iterable[str]]) -> str:
    """把模态键列表翻成中文。认不出的键**原样保留**（宁可显示 ``foo`` 也别吞掉）。

    传进来的是**单个字符串**时按一个键处理：直接 ``for`` 一个字符串会逐字符
    迭代，``"gaze"`` 会被翻成 ``g+a+z+e`` —— 一个字段类型不对就把整行弄成乱码，
    而这行恰恰是用来解释"这个分数是怎么来的"。
    """
    if isinstance(keys, str):
        keys = [keys] if keys else []
    if not keys:
        return ""
    return "+".join(MODALITY_LABELS.get(str(k), str(k)) for k in keys)


def format_vai_headline(vai: Optional[dict]) -> str:
    """第一行：「专注度 VAI 63.5」/「专注度 VAI ——」。"""
    if not isinstance(vai, dict):
        return f"专注度 VAI {NO_INDEX}"
    return f"专注度 VAI {format_index(vai.get('index'))}"


def format_vai_detail(vai: Optional[dict]) -> str:
    """第二行：测量状态 + 模态 + 已测时长。没有分数时改成解释为什么。"""
    if not isinstance(vai, dict):
        return VAI_NOT_DELIVERED

    status = str(vai.get("status") or "").strip()
    config = str(vai.get("modality_config_id") or "").strip()
    modalities = config or format_modalities(vai.get("modalities"))
    seconds = vai.get("valid_seconds")

    if vai.get("index") is None:
        # 没有分数时，**唯一**该显示的解释来自 B 的 index_status。
        why = str(vai.get("index_status") or "").strip() or "原因未知"
        parts = [why]
    else:
        parts = [status] if status else []

    if modalities:
        parts.append(f"模态 {modalities}")
    try:
        if seconds:
            parts.append(f"已测 {float(seconds):.0f}s")
    except (TypeError, ValueError):
        pass
    return " · ".join(parts)


def format_vai_lines(vai: Optional[dict]) -> List[str]:
    """整个专注度区块的若干行（不含免责声明，那行由 :func:`format_note` 给）。"""
    return [
        format_vai_headline(vai),
        format_index_bar((vai or {}).get("index") if isinstance(vai, dict) else None),
        format_vai_detail(vai),
    ]


def format_note(vai: Optional[dict]) -> str:
    """免责声明那一行。用 B 给的原文；缺字段（旧版 B）时用逐字相同的兜底。"""
    if isinstance(vai, dict):
        note = str(vai.get("note") or "").strip()
        if note:
            return note
    return FALLBACK_NOTE


# --------------------------------------------------------------------------
# 状态与通道
# --------------------------------------------------------------------------

def format_state_line(state: str, reason: str = "") -> str:
    """状态那一行。``state_label`` 由调用方传入或由本模块的默认实现给。

    这里 import ``c_core.expressions`` 是安全的：它同样不依赖 Qt。
    """
    from c_core.expressions import state_label

    line = f"状态 {state_label(state)}"
    if reason:
        line += f"（{reason}）"
    return line


#: 通道名 → 中文。与 ``ui/window.py`` 的 ``LINK_LABELS`` 是**两张表**：
#: 那张给标题栏与 HUD 用（"对话通道:通"），这张给右栏的文本块用。
#: 键必须一致，新增通道时两边都要加。
LINK_LABELS: Dict[str, str] = {
    "chat": "对话通道",
    "status": "状态通道",
    "stream": "画面通道",
}


def format_link_line(source: str, connected: bool) -> str:
    return f"{LINK_LABELS.get(source, source)} {'通' if connected else '断'}"


def format_link_block(links: Dict[str, bool]) -> str:
    """把通道状态拼成一行。顺序固定（chat → stream），别让它在界面上跳来跳去。"""
    order: Sequence[str] = ("chat", "status", "stream")
    known = [s for s in order if s in links]
    extra = [s for s in sorted(links) if s not in order]
    return " · ".join(format_link_line(s, links[s]) for s in list(known) + list(extra))


def format_stream_text(connected: Optional[bool], stream_url: Optional[str] = None) -> str:
    """占位区里的那两行文字。

    ``connected is None`` = 本次运行压根没打算拉流（没给 ``--stream-url``），
    与"给了地址但还没连上/已断开"是**两件事**，不能都写成"未接入"。
    """
    if connected is None:
        return f"{STREAM_OFFLINE}\n{STREAM_HINT}"
    if connected:
        return STREAM_WAITING
    url = stream_url or "A 的展示流"
    return f"{STREAM_OFFLINE}（{url} 未连接，正在重连）"
