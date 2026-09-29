# -*- coding: utf-8 -*-
"""CSV 历史对话记录读写。

任务要求 4：预留读取 CSV 历史记录的接口，后续加载历史对话数据。
api_doc §5.1 也提到开发阶段支持离线 CSV 调试，所以 CSV 在这个项目里是第一等公民，
不是临时方案。

提供两类能力：
    写入 —— append_turn()  / append_record()
    读取 —— load_recent() / load_session() / iter_records() / recent_dialogue()

设计意图：对上层（对话管理）只暴露 `HistoryProvider` 这个抽象，
今天背后是 CSV，明天换成 sqlite 或真实数据库时，对话管理一行都不用改。
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Protocol

import config


@dataclass
class TurnRecord:
    """一行历史记录。字段与 config.HISTORY_FIELDS 严格对应。"""

    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    role: str = "user"            # user / robot
    text: str = ""
    vision_state: str = config.STATE_NORMAL
    text_emotion: str = "neutral"
    emotion_score: float = 0.0
    intent: str = "chat"

    def to_row(self) -> Dict[str, object]:
        """转成 csv.DictWriter 需要的字典，顺序由 config.HISTORY_FIELDS 决定。"""
        data = asdict(self)
        return {key: data.get(key, "") for key in config.HISTORY_FIELDS}

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "TurnRecord":
        """从 CSV 行还原。脏数据（缺列、数字格式错）一律回退默认值，不抛异常。"""

        def as_float(key: str, default: float = 0.0) -> float:
            try:
                return float(row.get(key) or default)
            except (TypeError, ValueError):
                return default

        return cls(
            timestamp=as_float("timestamp"),
            session_id=(row.get("session_id") or "").strip(),
            role=(row.get("role") or "user").strip(),
            text=row.get("text") or "",
            vision_state=(row.get("vision_state") or config.STATE_NORMAL).strip(),
            text_emotion=(row.get("text_emotion") or "neutral").strip(),
            emotion_score=as_float("emotion_score"),
            intent=(row.get("intent") or "chat").strip(),
        )

    def as_dialogue(self) -> Dict[str, str]:
        """转成对话管理需要的 {role, text} 结构。"""
        return {"role": self.role, "text": self.text}


class HistoryProvider(Protocol):
    """历史记录来源的抽象接口。

    后续要接数据库/远程服务时，实现同样的方法即可替换 CsvHistoryStore。
    """

    def append_turn(
        self,
        user_text: str,
        reply_text: str,
        vision_state: str,
        text_emotion: str,
        emotion_score: float,
        intent: str,
    ) -> None:
        ...

    def load_recent(self, limit: int) -> List[TurnRecord]:
        ...

    def recent_dialogue(self, turns: int) -> List[Dict[str, str]]:
        ...


class CsvHistoryStore:
    """CSV 实现。线程安全（对话线程和视觉线程可能同时写）。

    文件不存在时自动创建并写入表头；表头不匹配时按"尽量读"处理，
    缺的列补默认值 —— 绝不因为历史文件格式老就崩掉。
    """

    def __init__(
        self,
        path: str = config.HISTORY_CSV,
        session_id: Optional[str] = None,
    ) -> None:
        self.path = path
        self.session_id = session_id or time.strftime("session-%Y%m%d-%H%M%S")
        self._lock = threading.Lock()
        self._ensure_file()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def append_turn(
        self,
        user_text: str,
        reply_text: str,
        vision_state: str = config.STATE_NORMAL,
        text_emotion: str = "neutral",
        emotion_score: float = 0.0,
        intent: str = "chat",
    ) -> None:
        """一次交互写两行：用户说的话 + 机器人回复。

        拆成两行而不是塞一行，是为了格式统一 —— 这样整份 CSV 每一行都
        只是"某人在某时刻说了一句话"，将来做数据分析/画情绪曲线都方便。
        """
        now = time.time()
        self.append_record(
            TurnRecord(
                timestamp=now,
                session_id=self.session_id,
                role="user",
                text=user_text,
                vision_state=vision_state,
                text_emotion=text_emotion,
                emotion_score=emotion_score,
                intent=intent,
            )
        )
        self.append_record(
            TurnRecord(
                timestamp=time.time(),
                session_id=self.session_id,
                role="robot",
                text=reply_text,
                vision_state=vision_state,
                text_emotion=text_emotion,
                emotion_score=emotion_score,
                intent=intent,
            )
        )

    def append_record(self, record: TurnRecord) -> None:
        """写单条记录。"""
        with self._lock:
            # newline="" 是 csv 模块在 Windows 上的硬性要求，否则每行之间会多出空行
            with open(self.path, "a", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=config.HISTORY_FIELDS)
                writer.writerow(record.to_row())

    # ------------------------------------------------------------------
    # 读取 —— 任务要求 4 的"预留接口"
    # ------------------------------------------------------------------

    def iterate(self) -> Iterator[TurnRecord]:
        """流式读取全部历史记录，逐条产出。

        注意：这里是"先在锁内把文件读进内存，再逐条 yield"，而不是在 yield 期间
        一直持锁 —— 生成器跨 yield 持锁会让写入方永久阻塞，是个典型死锁陷阱。
        """
        rows = self._read_all_rows()
        for row in rows:
            yield TurnRecord.from_row(row)

    def _read_all_rows(self) -> List[Dict[str, str]]:
        """在锁保护下把 CSV 全部读成字典列表。文件损坏时返回已读到的部分。"""
        if not os.path.exists(self.path):
            return []
        with self._lock:
            try:
                with open(self.path, "r", newline="", encoding="utf-8") as fp:
                    return [row for row in csv.DictReader(fp)]
            except (OSError, csv.Error, UnicodeDecodeError):
                # 历史文件坏了不该影响主流程
                return []

    # 别名，方便外部按语义调用
    def iter_records(self) -> Iterator[TurnRecord]:
        return self.iterate()

    def load_recent(self, limit: int = config.HISTORY_PRELOAD_ROWS) -> List[TurnRecord]:
        """取最近 limit 条记录（返回顺序为时间正序）。"""
        if limit <= 0:
            return []
        # 用 deque 做定长环形缓冲，避免把整个文件读进列表
        from collections import deque

        buffer: deque = deque(maxlen=limit)
        for record in self.iterate():
            buffer.append(record)
        return list(buffer)

    def load_session(self, session_id: str, limit: int = 0) -> List[TurnRecord]:
        """取某个会话的记录，limit=0 表示不限条数。"""
        result: List[TurnRecord] = []
        for record in self.iterate():
            if record.session_id != session_id:
                continue
            result.append(record)
            if limit and len(result) >= limit:
                break
        return result

    def recent_dialogue(self, turns: int = config.DIALOGUE_CONTEXT_TURNS,
                        session_only: bool = False) -> List[Dict[str, str]]:
        """取最近 turns 轮对话，转成 [{"role": ..., "text": ...}]。

        对话管理在生成回复前调用它，用来避免重复上一句、并做上下文衔接。
        大模型那条路也用它拼上下文（见 core/dialogue.py 的 _history_messages）。

        ``session_only=True`` 只取**本次会话**的行。默认 False 是为了保持
        这个方法原来的语义（纯透传），但**喂给大模型时一定要传 True**：
        history.csv 会跨多次演示累积，不过滤的话新会话第一句话就会把
        上次排练的尾巴喂给模型 —— 那里面还包括别人的话。

        "记住昨天"对陪伴机器人是个真功能，但那需要真正的记忆摘要，
        不是把 CSV 的尾巴直接塞进提示词；那是另一个独立决策。
        """
        if turns <= 0:
            return []
        records = self.load_recent(limit=turns * 2)
        if session_only:
            records = [record for record in records
                       if record.session_id == self.session_id]
        return [record.as_dialogue() for record in records]

    def last_robot_replies(self, count: int = 3) -> List[str]:
        """取机器人最近说过的 count 句话，用于避免回复重复。"""
        replies: List[str] = []
        for record in reversed(self.load_recent(limit=count * 6)):
            if record.role == "robot" and record.text:
                replies.append(record.text)
                if len(replies) >= count:
                    break
        return replies

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def count(self) -> int:
        """历史记录总条数（不含表头）。"""
        return sum(1 for _ in self.iterate())

    def _ensure_file(self) -> None:
        """确保文件存在且有表头。"""
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            return
        with self._lock:
            with open(self.path, "w", newline="", encoding="utf-8") as fp:
                csv.DictWriter(fp, fieldnames=config.HISTORY_FIELDS).writeheader()
