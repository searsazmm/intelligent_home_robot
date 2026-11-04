# -*- coding: utf-8 -*-
"""心率 / 呼吸（rppg）的 B 端持有与展示投影。

A 端每秒发一条 ``{"type":"rppg", ...}``（api_doc §3.5.2），经
``core/typed_messages.py`` 解析成 :class:`~core.typed_messages.RppgReading`。
在此之前 B 把它收下、计数、然后**丢弃** —— 没有消费者。现在
:class:`VitalsStore` 接住它，:func:`build_vitals_message` 把它投影成
给 C 看的 ``vitals`` 展示报文（B→C，走 8002 对话通道，与 ``vai`` 同路）。

⚠️ 与 ``vai`` 同一条红线：**vitals 报文刻意不带 ``state`` 字段。**
带了它，迟早有人把它接到表情上 —— 体征就变成第二个状态源，去和
8001/8002 的 state 打架。心率呼吸只该是右栏的**另一行数字**，
不参与任何判定（不进 evaluator、不碰静默、不碰主动关怀）。

``hr`` / ``rr`` 允许是 ``None``：A 的窗口未满 8 秒、SNR 不足时都会置灰。
**那不是"心率 0"**，是没有测量值 —— 前端必须分开显示。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

#: 体征展示报文里的免责声明。这东西是**摄像头看脸估出来的**，
#: 不是医疗器械 —— 这句话必须跟着数值走，改顺口了等于替它下医疗结论。
VITALS_NOTE = "视觉测量，仅供参考，非医疗结论"


class VitalsStore:
    """线程安全地持有最新一条 rppg 读数。

    写入方是视觉读取线程（``VisionClient`` / ``OfflineVisionFeeder`` 的
    ``on_typed`` 回调，约 1Hz），读取方是状态发布节拍（0.2 秒）与客户端
    线程（新连接补发）—— 一个写方、两个读方，谁都不该看到半更新的数据。

    与 :class:`FocusTracker` 的 ``snapshot()`` 同一条纪律：交出去的是
    不可变读数（``RppgReading`` 是 frozen dataclass），持有方不用加锁去读。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[object] = None

    def update(self, reading: object) -> None:
        """存下最新一条读数（直接覆盖 —— 界面只要"当下"，不要历史）。"""
        with self._lock:
            self._latest = reading

    def snapshot(self) -> Optional[object]:
        """最新一条读数；一条都没收到过时返回 ``None``。"""
        with self._lock:
            return self._latest


def build_vitals_message(reading: object, timestamp: Optional[float] = None) -> Optional[dict]:
    """把 :class:`~core.typed_messages.RppgReading` 投影成给 C 看的展示报文。

    **逐字段原样转发，不做任何加工**（不平滑、不取整、不补测）。
    ``reading`` 为 ``None``（还没收到过 rppg）时返回 ``None`` —— 上游据此
    跳过发送，而不是发一条 ``hr=null`` 的空报文把 C 的"未检出"语义冲掉。

    ⚠️ **刻意不带 ``state`` 字段** —— 见模块文档的红线。
    这里用鸭子类型读属性而不是 import ``typed_messages``：本模块是纯粹的
    对 C 传输层，不该认识报文解析那个领域对象；传给它的只要是个
    "长得像读数"的东西。
    """
    if reading is None:
        return None
    hr = getattr(reading, "hr", None)
    rr = getattr(reading, "rr", None)
    return {
        "type": "vitals",
        "hr": None if hr is None else int(hr),
        "rr": None if rr is None else round(float(rr), 1),
        "ibi_ms": list(getattr(reading, "ibi_ms", ()) or ()),
        "note": VITALS_NOTE,
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }


__all__ = ["VITALS_NOTE", "VitalsStore", "build_vitals_message"]
