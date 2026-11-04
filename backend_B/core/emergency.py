# -*- coding: utf-8 -*-
"""紧急求助：把「我帮您叫家里人」从一句话变成**真的动作**。

以前危机 / 不适分支只会说「我帮您把家里人叫过来」，说完什么都没发生 ——
老人真出事时，这句话比沉默更糟（给了一个不会兑现的承诺）。本模块补的
是**通知**这一段：

1. :data:`EMERGENCY_PATTERNS` 命中「救命 / 叫救护车 / 喘不上气」这类
   **身体紧急**语句；自我伤害类危机（「不想活了」）由
   :mod:`core.text_emotion` 的 ``crisis`` 标记识别。两条线在 dialogue
   里都置 ``DialogueReply.emergency``。
2. :class:`FamilyAlerter` 向配置的 webhook（企业微信 / 钉钉 / 飞书机器人
   或自建服务，``B_FAMILY_ALERT_URL``，分号分隔多个）POST 一条 JSON。
   标准库 urllib 实现 —— B 端运行期零第三方依赖是测试守着的硬约束。

三条安全纪律：

* **永不抛异常**。通知失败只能记日志，绝不能连累回复生成 —— 那句
  「您别慌，我这就通知家里人」本身必须先稳稳定定说出口。
* **冷却去重**。同一句话被端点检测切成两段、或老人慌乱中重复喊，
  冷却窗口内只真正通知一次（对话回复不受影响）。
* **同步方法里不做阻塞网络调用** —— :meth:`FamilyAlerter.alert` 在
  daemon 线程里发请求，立刻返回；调用方（对话线程 / 语音线程）
  不该为一次外网往返卡住应答。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Callable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: 身体紧急语句。和 text_emotion.CRISIS_PATTERNS 分开：
#: 那组管「自我伤害」，这组管「立刻需要人到场」。
#:
#: 选词原则：**完整说法，不收短词**。裸「报警」会误伤「电视里在报警」，
#: 裸「不行了」会误伤「这把年纪不行了」这类感慨 —— 误报一次，家人和
#: 老人对这个功能的信任就没了。宁可漏一句边缘说法，由 crisis/discomfort
#: 分支兜底接住，也不狼来了。
EMERGENCY_PATTERNS: Tuple[str, ...] = (
    "救命", "救救我", "快来人",
    "快报警", "帮我报警", "赶紧报警",
    "打120", "快打120", "帮我打120", "叫救护车", "快叫救护车", "喊救护车",
    "喘不上气", "喘不过气", "上不来气", "呼吸困难",
    "起不来了", "站不起来", "不能动了", "动不了了",
    "摔了一跤", "摔倒了", "摔着了", "我摔了",
    "胸口疼", "心口疼", "胸痛",
)

#: 通知类型
KIND_EMERGENCY = "emergency"   # 身体紧急（关键词）
KIND_CRISIS = "crisis"         # 自我伤害危机（情绪层）


def is_emergency_text(text: str) -> bool:
    """这句话是不是明确的身体紧急求助。"""
    if not text:
        return False
    return any(pattern in text for pattern in EMERGENCY_PATTERNS)


def build_alert_payload(kind: str, text: str,
                        timestamp: Optional[float] = None) -> dict:
    """构造发给家人 webhook 的 JSON 体。

    结构保持扁平、字段名自解释 —— 对端可能是一个十几行的云函数，
    不该要求它懂机器人内部的报文协议。
    """
    return {
        "type": "home_robot_emergency",
        "kind": kind,
        "text": (text or "").strip(),
        "message": {
            KIND_EMERGENCY: "家里老人发出紧急求助，请立刻联系或回家查看",
            KIND_CRISIS: "家里老人说了危险的话，请立刻联系陪伴",
        }.get(kind, "家里老人需要帮助，请尽快联系"),
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }


#: 发送函数签名：url + payload dict + 超时秒；失败抛异常。
#: 抽成可注入参数纯粹是为了测试不联网。
Sender = Callable[[str, dict, float], None]


def _default_send(url: str, payload: dict, timeout: float) -> None:
    """用标准库把 JSON POST 出去。非 2xx 抛 ``URLError``。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 0)
        if status and not 200 <= status < 300:
            raise OSError("家人通知 webhook 返回 %s" % status)


class FamilyAlerter:
    """把紧急求助真正发出去：webhook 通知 + 冷却去重 + 失败只记日志。

    线程安全：对话线程调 :meth:`alert`，网络请求在 daemon 线程里跑；
    冷却记账用一把锁保护。
    """

    def __init__(
        self,
        urls: Sequence[str] = (),
        timeout: float = 5.0,
        cooldown: float = 300.0,
        sender: Optional[Sender] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._urls = tuple(urls)
        self._timeout = float(timeout)
        self._cooldown = float(cooldown)
        self._sender = sender or _default_send
        self._clock = clock
        self._lock = threading.Lock()
        #: 上一次**真正发出**通知的时刻（按 kind 分别记账）。
        self._last_sent: dict[str, float] = {}
        #: 已启动但还没回来的通知线程，测试 join 用；运行期不依赖它。
        self._threads: list[threading.Thread] = []

    @property
    def configured(self) -> bool:
        """有没有配置任何通知地址。没配置时 alert 只记一条醒目日志。"""
        return bool(self._urls)

    def alert(self, kind: str, text: str, *, force: bool = False) -> bool:
        """触发一次家人通知。**立刻返回**（网络在后台线程），不抛异常。

        :return: 是否真的排上了一次发送（冷却中 / 无 URL 时为 False）。
        """
        now = self._clock()
        with self._lock:
            # 没发过的 kind 视作"无限久以前" —— 用 -inf 而不是 0：
            # 冷却窗是相对量，0 在时钟起点附近（或测试的假时钟下）
            # 会被误判成"刚发过"，第一声救命就被吞了。
            last = self._last_sent.get(kind, float("-inf"))
            if not force and now - last < self._cooldown:
                logger.warning(
                    "紧急求助在 %.0f 秒冷却窗内，不再重复通知家人：%r",
                    self._cooldown, text[:30])
                return False
            self._last_sent[kind] = now

        payload = build_alert_payload(kind, text)

        if not self._urls:
            # 没配 webhook：这是部署配置问题，不是运行故障 —— 用 WARNING
            # 喊清楚（C 端弹窗和语音安慰仍然生效，见 main._raise_emergency）。
            logger.warning(
                "【紧急求助】未配置 B_FAMILY_ALERT_URL，无法通知家人，"
                "仅本机提示：kind=%s text=%r", kind, text[:50])
            return False

        thread = threading.Thread(
            target=self._send_all, args=(payload,),
            name="family-alert", daemon=True)
        thread.start()
        self._threads = [t for t in self._threads if t.is_alive()]
        self._threads.append(thread)
        return True

    def _send_all(self, payload: dict) -> None:
        """逐个 URL 发；某一个失败不影响其余的，全程不抛。"""
        for url in self._urls:
            try:
                self._sender(url, payload, self._timeout)
                logger.warning("【紧急求助】已通知家人：%s kind=%s",
                               url, payload.get("kind"))
            except Exception:
                logger.exception(
                    "【紧急求助】通知家人失败（%s）：%r",
                    url, str(payload.get("text", ""))[:50])

    def join(self, timeout: Optional[float] = None) -> None:
        """等后台通知发完。**仅供测试**用。"""
        for thread in list(self._threads):
            thread.join(timeout=timeout)
