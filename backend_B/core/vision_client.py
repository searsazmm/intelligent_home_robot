# -*- coding: utf-8 -*-
"""连模块 A 收视觉数据。

api_doc §3.1：**服务端是模块 A，客户端是模块 B**。所以这里是 B 主动去 connect
127.0.0.1:8000，而不是 B 监听 8000 —— 别写反了。

两个数据来源，对外接口一样，通过 run() / run_offline() 二选一：
    VisionClient   实时 TCP，带断线重连
    OfflineVisionFeeder 读 CSV 回放（api_doc §5.1 的开发阶段方案）

断线重连是这里的重点，规则：
    - 连不上：指数退避重试，1s → 2s → 4s → 8s → 10s 封顶，永远不停
    - 读到 EOF / 报错：立刻重连，退避重置
    - 重连期间：B 的其它部分（对 C 的两个服务）照常工作，绝不阻塞或崩溃
    - 断线期间：视觉状态降级为 absent（看代码 vision_state.mark_disconnected）
"""

from __future__ import annotations

import csv
import logging
import socket
import threading
from typing import Callable, Optional

import config
from core.protocol import LineBuffer, ProtocolError, decode_json_line
from core.vision_state import VisionSample, VisionStateEvaluator

logger = logging.getLogger(__name__)


class VisionClient:
    """A→B 的 TCP 客户端，收到一帧就回调一次。"""

    def __init__(
        self,
        evaluator: VisionStateEvaluator,
        on_sample: Optional[Callable[[VisionSample], None]] = None,
        host: str = config.VISION_HOST,
        port: int = config.VISION_PORT,
    ) -> None:
        self.host = host
        self.port = port
        self.evaluator = evaluator
        self.on_sample = on_sample

        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._connected = threading.Event()

        # 统计信息，联调时看日志用
        self.frames_received = 0
        self.bad_lines = 0
        self.reconnect_count = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def run(self) -> None:
        """主循环：连接 → 读 → 断了就重连。阻塞，直到 stop() 被调用。"""
        backoff = config.RECONNECT_BACKOFF_INITIAL

        while not self._stop.is_set():
            try:
                self._connect()
            except (ConnectionRefusedError, socket.timeout, OSError) as exc:
                # A 还没启动是常态（联调顺序是先 A 后 B），不是错误，用 WARNING 就够
                logger.warning(
                    "连接模块 A %s:%s 失败（%s），%.1fs 后重试",
                    self.host, self.port, exc.__class__.__name__, backoff,
                )
                self.evaluator.mark_disconnected()
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * config.RECONNECT_BACKOFF_FACTOR, config.RECONNECT_BACKOFF_MAX)
                continue

            # 连上了，退避重置
            backoff = config.RECONNECT_BACKOFF_INITIAL
            try:
                self._read_loop()
            except (ConnectionResetError, socket.timeout, OSError) as exc:
                logger.warning("与模块 A 的连接中断（%s），准备重连", exc.__class__.__name__)
            finally:
                self._close_socket()
                self.evaluator.mark_disconnected()

            if self._stop.is_set():
                break

            self.reconnect_count += 1
            logger.info("第 %d 次重连，%.1fs 后开始", self.reconnect_count, backoff)
            if self._stop.wait(backoff):
                break

        logger.info("视觉客户端已停止（共收到 %d 帧，脏数据 %d 条）",
                    self.frames_received, self.bad_lines)

    def stop(self) -> None:
        """请求停止并唤醒可能正在等待的循环。"""
        self._stop.set()
        self._close_socket()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """建立连接。"""
        logger.info("正在连接模块 A %s:%s ...", self.host, self.port)
        sock = socket.create_connection((self.host, self.port), timeout=config.CONNECT_TIMEOUT)
        # 连上之后切成短超时：这样 recv 会定期抛 timeout，
        # 让我们有机会检查 _stop 标志并及时退出（否则 Ctrl+C 要等半天）。
        sock.settimeout(config.SOCKET_TIMEOUT)
        # 关掉 Nagle：视觉数据是小包高频，攒包会平白增加延迟
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._connected.set()
        logger.info("已连接模块 A %s:%s", self.host, self.port)

    def _read_loop(self) -> None:
        """持续 recv 并按 \\n 分帧。"""
        assert self._sock is not None
        buffer = LineBuffer()

        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                # 短超时到了但没有数据，属于正常情况：回去检查 _stop
                continue

            if not chunk:
                # 对端正常关闭（recv 返回空字节）
                logger.info("模块 A 主动关闭了连接")
                return

            try:
                for line in buffer.feed(chunk):
                    self._handle_line(line)
            except ProtocolError as exc:
                # 超长行等协议级错误：丢弃缓冲区继续，不断连接
                logger.error("协议错误：%s", exc)
                self.bad_lines += 1
                buffer.clear()

    def _handle_line(self, line: str) -> None:
        """处理一条完整报文。单条脏数据只记日志，不影响后续。"""
        try:
            payload = decode_json_line(line)
        except ProtocolError as exc:
            self.bad_lines += 1
            logger.warning("丢弃非法报文：%s", exc)
            return

        self._warn_missing_fields(payload)

        sample = VisionSample.from_payload(payload)
        self.frames_received += 1

        self.evaluator.push(sample)

        if self.on_sample is not None:
            try:
                self.on_sample(sample)
            except Exception:
                # 回调是上层的事，它崩了不该带崩读取循环
                logger.exception("on_sample 回调抛出异常，已忽略")

    def _warn_missing_fields(self, payload: dict) -> None:
        """api_doc §3.2 规定的字段缺了就提醒一次（只在开头几帧提醒，避免刷屏）。"""
        if self.frames_received > 3:
            return
        required = ("timestamp", "has_face", "ear", "blink_cnt",
                    "pitch", "yaw", "roll", "emo_feature")
        missing = [name for name in required if name not in payload]
        if missing:
            logger.warning(
                "A 发来的报文缺少 api_doc 规定的字段：%s（本条按默认值处理）",
                "、".join(missing),
            )

    def _close_socket(self) -> None:
        self._connected.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# --------------------------------------------------------------------------
# 离线回放（api_doc §5.1：开发阶段模块 A 导出 CSV，B 读文件并行开发）
# --------------------------------------------------------------------------

class OfflineVisionFeeder:
    """读 CSV 逐行喂给判定器，模拟模块 A 的实时数据流。

    用途：模块 A 还没写好时，B 单方面就能跑通全链路。
    用法：
        python main.py --offline data/sample_vision.csv --speed 5
    """

    def __init__(
        self,
        csv_path: str,
        evaluator: VisionStateEvaluator,
        on_sample: Optional[Callable[[VisionSample], None]] = None,
        speed: float = 1.0,
        loop: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.evaluator = evaluator
        self.on_sample = on_sample
        self.speed = max(speed, 0.01)   # 0 会导致除零
        self.loop = loop
        self._stop = threading.Event()
        self.frames_received = 0

    def run(self) -> None:
        """阻塞回放，直到 stop() 或文件读完（loop=False 时）。"""
        while not self._stop.is_set():
            try:
                played = self._play_once()
            except FileNotFoundError:
                logger.error("离线 CSV 不存在：%s", self.csv_path)
                return
            except OSError as exc:
                logger.error("读取离线 CSV 失败：%s", exc)
                return

            if played == 0:
                logger.warning("离线 CSV 没有任何有效数据行：%s", self.csv_path)
                return

            if not self.loop:
                logger.info("离线回放结束，共 %d 帧", self.frames_received)
                return
            logger.info("一轮回放结束（%d 帧），重新开始", played)

    def stop(self) -> None:
        self._stop.set()

    def _play_once(self) -> int:
        """播一遍文件，返回播放的帧数。

        节奏按 CSV 里 timestamp 的相邻差值来，而不是固定间隔 —— 这样回放出来的
        时间尺度跟真实采集一致，疲劳判定里那些"持续 N 秒"的阈值才有意义。
        speed 作为倍速乘数（speed=5 表示 5 倍速快放）。
        """
        played = 0
        previous_ts: Optional[float] = None

        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as fp:
            # utf-8-sig：兼容用 Excel 另存过的 CSV（会带 BOM）
            reader = csv.DictReader(fp)
            for row in reader:
                if self._stop.is_set():
                    break

                timestamp = float(row.get("timestamp") or 0.0)

                if previous_ts is not None:
                    delta = timestamp - previous_ts
                    # 差值异常（负数/超大）时退回一个默认间隔，避免卡住或瞬移
                    if not (0 < delta <= 5.0):
                        delta = 0.1
                    if self._stop.wait(delta / self.speed):
                        break
                previous_ts = timestamp

                # 先 sleep 再构造样本：这样 received_at 记的是"喂给判定器的那一刻"，
                # 防抖/窗口的时长才和数据里的时间轴对得上。
                sample = VisionSample.from_payload(row)
                self.evaluator.push(sample)
                self.frames_received += 1
                played += 1

                if self.on_sample is not None:
                    try:
                        self.on_sample(sample)
                    except Exception:
                        logger.exception("on_sample 回调抛出异常，已忽略")

        return played
