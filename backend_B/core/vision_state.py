# -*- coding: utf-8 -*-
"""视觉状态判定：把模块 A 的原始特征换算成 4 种合法状态。

api_doc §3.3.2 明确写了"模块 A 仅负责视觉特征采集，不做最终用户状态判定"，
所以 normal / sad / tired / absent 的判定责任在 B 端，也就是本模块。

输入字段（api_doc §3.2，禁止改名）：
    timestamp    float  程序运行时间戳（秒）
    has_face     bool   是否检测到人脸
    ear          float  眼睑开合度
    blink_cnt    int    累计眨眼次数
    pitch        float  头部俯仰
    yaw          float  头部左右偏转
    roll         float  头部倾斜
    emo_feature  str    normal / tired / sad / blank（正式枚举，api_doc §3.2 V1.1）
                        low（= sad 的兼容别名，A-包仍发这个值）

输出：4 种状态字符串之一。

设计要点（防止前端状态乱跳）：
  1. 滑动窗口 —— 只看最近 WINDOW_SECONDS 秒的样本，单帧异常不影响结果。
  2. 防抖     —— 新状态要连续稳定 STATE_MIN_HOLD 秒才生效。
  3. 迟滞     —— 从 absent 恢复比其他切换更慢，避免"闪一下脸"就切回 normal。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import config


# ---------------------------------------------------------------------------
# emo_feature 的两套取值（api_doc §3.2）
# ---------------------------------------------------------------------------
# 仓库里并存两套模块 A，出站报文格式一样但 emo_feature 枚举不一样：
#
#   A-包（backend_A/module_a_vision/）  normal / low   / tired
#   A-单文件（backend_A/vision_a.py）   normal / tired / sad / blank
#
# api_doc V1.1 起把后者记为**正式枚举**，low 降为 sad 的兼容别名，
# 并要求接收方同时接受两套。**别只认一套**：只认 low 会静默忽略 sad，
# 于是老人难过时界面显示正常；只认 sad 则反过来漏掉 A-包。两边单测
# 各自全绿，这种漏判在单侧测不出来。
EMO_LOW_ALIASES = ("low", "sad")   # 同义：都判 STATE_SAD
EMO_BLANK = "blank"                # 发呆/失神：双眼睁开但视线长时间无位移
# tired / normal 两套同名，直接按字面量比较即可，不需要常量。


@dataclass
class VisionSample:
    """一帧视觉特征。字段名与 api_doc §3.2 一一对应。"""

    timestamp: float = 0.0
    has_face: bool = False
    ear: float = 0.0
    blink_cnt: int = 0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    emo_feature: str = "normal"

    # 本机收到该帧的时间，用于判断数据是否过期（A 的 timestamp 是"程序运行时间"，
    # 可能是从 0 开始计数的相对时间，不能直接和本机时钟比）。
    #
    # 用 monotonic 而不是 time.time()：后者会被系统对时/夏令时往回拨，
    # 而这里只需要"过了多久"，单调时钟才是对的。
    received_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_payload(cls, payload: dict) -> "VisionSample":
        """从 A 发来的 JSON 字典构造样本。

        对端字段缺失/类型不对时一律回退到安全默认值，绝不抛异常中断连接 ——
        接口约束里 A 在无人脸时也要发包，字段可能填 0 或 null。
        """

        def num(key, default=0.0):
            try:
                value = payload.get(key, default)
                return default if value is None else float(value)
            except (TypeError, ValueError):
                return default

        def boolean(key, default=False):
            value = payload.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "y")
            return bool(value)

        return cls(
            timestamp=num("timestamp", 0.0),
            has_face=boolean("has_face", False),
            ear=num("ear", 0.0),
            blink_cnt=int(num("blink_cnt", 0)),
            pitch=num("pitch", 0.0),
            yaw=num("yaw", 0.0),
            roll=num("roll", 0.0),
            emo_feature=str(payload.get("emo_feature") or "normal").strip().lower(),
        )


@dataclass
class VisionState:
    """判定结果快照，供对话管理和前端推送使用。"""

    state: str = config.STATE_ABSENT          # 四态之一
    reason: str = "尚未收到视觉数据"            # 人类可读的判定理由，方便联调排查
    has_face: bool = False
    stale: bool = True                        # 数据是否已过期/失联
    confidence: float = 0.0                   # 0~1，粗略可信度

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "has_face": self.has_face,
            "stale": self.stale,
            "confidence": round(self.confidence, 3),
        }


class VisionStateEvaluator:
    """带滑动窗口 + 防抖 + 迟滞的视觉状态判定器。

    线程安全性：本类自身不加锁。约定只由视觉读取线程调用 push()，
    其它线程通过 get_state() 读取；get_state() 返回的是不可变快照对象，
    且 Python 的属性赋值是原子的，因此这里不加锁不会读到半个对象。
    """

    def __init__(self, clock=time.monotonic) -> None:
        # 取"现在"的函数。默认单调时钟；测试可以注入假时钟来验证"失联超时"。
        self._clock = clock
        # 主滑动窗口（config.WINDOW_SECONDS），用于常规状态判定
        self._samples: Deque[VisionSample] = deque()
        # 眨眼统计专用的长窗口（config.BLINK_WINDOW_SECONDS）
        self._blinks: Deque[VisionSample] = deque()
        # 当前生效的状态
        self._current = VisionState(state=config.STATE_ABSENT, reason="尚未收到视觉数据")
        # 候选状态（还没稳定到可以生效的那个）
        self._candidate: Optional[str] = None
        self._candidate_since: float = 0.0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def push(self, sample: VisionSample) -> VisionState:
        """喂入一帧，返回最新判定结果。

        注意这里用 `sample.received_at` 当作"现在"，而不是再取一次本机时间。
        这样做有两个好处：
          1. 防抖和滑动窗口的时间尺度跟数据本身走，离线回放（含 --speed 倍速）
             和实时接收的行为完全一致；
          2. 测试里只要给样本设好 received_at，就能在毫秒内"快进"两分钟，
             不用真的 sleep 两分钟。
        """
        now = sample.received_at
        self._samples.append(sample)
        self._blinks.append(sample)
        self._trim(now)

        raw_state, reason, confidence = self._evaluate_window(now)
        return self._apply_debounce(raw_state, reason, confidence, now)

    def mark_disconnected(self) -> VisionState:
        """A 端断开时调用：立刻降级为 absent（无人/失联）。

        A 是视觉来源，断了就等于"看不见人"，继续维持 normal 是危险的误判。
        """
        self._samples.clear()
        self._blinks.clear()
        self._candidate = None
        self._current = VisionState(
            state=config.STATE_ABSENT,
            reason="与模块 A 的连接已断开，视觉数据不可用",
            has_face=False,
            stale=True,
            confidence=1.0,
        )
        return self._current

    def get_state(self) -> VisionState:
        """读取当前状态；若数据已过期，返回降级后的 absent 快照。

        这里用墙钟（self._clock）算"多久没收到数据了"—— 这是真实的流逝时间，
        和样本自带的 received_at 是同一个时钟源（都默认 monotonic）。
        """
        now = self._clock()
        if self._samples:
            age = now - self._samples[-1].received_at
            if age > config.VISION_STALE_SECONDS:
                return VisionState(
                    state=config.STATE_ABSENT,
                    reason=f"已 {age:.1f}s 未收到视觉数据（超过 {config.VISION_STALE_SECONDS:.0f}s）",
                    has_face=False,
                    stale=True,
                    confidence=1.0,
                )
        return self._current

    # ------------------------------------------------------------------
    # 内部：窗口评估
    # ------------------------------------------------------------------

    def _trim(self, now: float) -> None:
        """丢弃两个窗口各自范围外的老样本。"""
        cutoff = now - config.WINDOW_SECONDS
        while self._samples and self._samples[0].received_at < cutoff:
            self._samples.popleft()

        blink_cutoff = now - config.BLINK_WINDOW_SECONDS
        while self._blinks and self._blinks[0].received_at < blink_cutoff:
            self._blinks.popleft()

    def _evaluate_window(self, now: float):
        """综合窗口内所有样本，得出"原始判定"（尚未防抖）。"""
        samples = list(self._samples)
        if not samples:
            return config.STATE_ABSENT, "窗口内无样本", 0.0

        face_samples = [s for s in samples if s.has_face]

        # --- 1) 无人脸：看"最后一张人脸之后过了多久"，而不是"窗口里还有没有人脸" ---
        # 注意不能用 `if not face_samples` 判断：窗口有 10 秒长，人走进走出后
        # 旧的人脸样本还会在窗口里待 10 秒，那样会晚 8 秒才发现人已经离开。
        last_face_at = self._last_face_time()
        if last_face_at is None:
            return config.STATE_ABSENT, "窗口内始终未检测到人脸", 1.0

        lost_seconds = now - last_face_at
        if lost_seconds >= config.FACE_LOST_GRACE:
            return (
                config.STATE_ABSENT,
                f"连续 {lost_seconds:.1f}s 未检测到人脸",
                1.0,
            )

        if not face_samples:
            return self._current.state, "短暂丢帧，维持原状态", 0.4

        # 人脸"时不时闪一下"但整体占比很低（扭头出画、只露半张脸），也倾向 absent。
        #
        # 这里加了 `not samples[-1].has_face` 这个前置条件很关键：人脸刚回到画面时，
        # 10 秒窗口里还残留着大量无人脸的旧样本，占比必然很低。如果不过滤这种情况，
        # 人回来了还要额外等 4 秒才能恢复 normal，白白拖慢恢复速度。
        face_ratio = len(face_samples) / len(samples)
        if not samples[-1].has_face and face_ratio < 0.4:
            return (
                config.STATE_ABSENT,
                f"人脸仅在窗口内闪现已 {face_ratio:.0%} 的帧，疑似人已离开",
                0.6,
            )

        latest_feature = face_samples[-1].emo_feature

        # --- 1.5) 发呆/失神：A-单文件的 blank ---
        # 判成 absent 而不是新加一种状态：C 端 absent 的显示标签本来就是
        # 「走神/无人」（frontend_C/c_core/expressions.py），语义正好对上。
        #
        # ⚠️ 副作用（必须知道）：absent 是主动关怀「人回来了」那条边沿判定的
        # 输入（core/proactive.py 的 _observe_state）。发呆被判成 absent 后恢复
        # normal，会触发一次问候：
        #   - 默认 PROACTIVE_GREETING_ABSENT = 60s，而 blank 只要静止 3s 就成立
        #     → **默认配置下不会误触发**（要发呆满 60 秒才算"离开过"）；
        #   - 但 --demo 把它降到 5s → **演示时会误触发**。
        # 放在疲劳判定之前：blank 的成立条件是"双眼睁开且视线不动"，
        # 与"困得睁不开眼"是互斥的两种情形，先判走神更贴近语义。
        if latest_feature == EMO_BLANK:
            return config.STATE_ABSENT, "A 端 emo_feature=blank（发呆/失神）", 0.6

        # --- 2) 疲劳：A 端特征 / EAR 持续偏低 / 眨眼过频 / 长时间低头 ---
        tired_reasons = []

        if latest_feature == "tired":
            tired_reasons.append("A 端 emo_feature=tired")

        low_ear_seconds = self._low_ear_seconds()
        if low_ear_seconds >= config.EAR_TIRED_DURATION:
            tired_reasons.append(f"EAR 低于 {config.EAR_TIRED_THRESHOLD} 持续 {low_ear_seconds:.1f}s")

        blink_rate = self._blink_rate_per_minute()
        if blink_rate is not None and blink_rate > config.BLINK_RATE_TIRED:
            tired_reasons.append(f"眨眼频率 {blink_rate:.0f} 次/分偏高")

        if self._sustained_pitch_down(face_samples):
            tired_reasons.append(f"持续低头超过 {config.PITCH_DOWN_THRESHOLD}°")

        if tired_reasons:
            return config.STATE_TIRED, "；".join(tired_reasons), min(0.5 + 0.15 * len(tired_reasons), 1.0)

        # --- 3) 低落：A 端特征 low / sad（同义），或长时间频繁偏头（坐立不安的表现之一）---
        # 理由串写实际收到的值，别写死 "low" —— 排查日志时能一眼看出是哪套 A 发的。
        if latest_feature in EMO_LOW_ALIASES:
            return config.STATE_SAD, f"A 端 emo_feature={latest_feature}", 0.7

        # --- 4) 正常 ---
        return config.STATE_NORMAL, "人脸在位且各项特征正常", 0.8

    # ------------------------------------------------------------------
    # 内部：单项特征计算
    # ------------------------------------------------------------------

    def _last_face_time(self) -> Optional[float]:
        for sample in reversed(self._samples):
            if sample.has_face:
                return sample.received_at
        return None

    def _low_ear_seconds(self) -> float:
        """最近一段"连续低 EAR"持续的秒数（从窗口尾部往前数）。"""
        tail: list[VisionSample] = []
        for sample in reversed(self._samples):
            if not sample.has_face:
                break
            if sample.ear < config.EAR_TIRED_THRESHOLD:
                tail.append(sample)
            else:
                break
        if len(tail) < 2:
            return 0.0
        # tail 是倒序的，首尾相减得到持续时长
        return tail[0].received_at - tail[-1].received_at

    # 眨眼频率的估算门槛。
    # blink_cnt 是整数，眨眼又是稀疏事件 —— 10 秒窗口里正常波动就能算出 40 次/分，
    # 把好好的人判成疲劳。所以这里用独立的、长得多的时间窗（见 config.BLINK_WINDOW_SECONDS），
    # 并要求统计期内确实累计到足够多的眨眼次数。
    # 宁可漏判（EAR 和低头还有两条路兜底），也不要误判。
    MIN_BLINK_SPAN_SECONDS = 20.0
    MIN_BLINK_DELTA = 6

    def _blink_rate_per_minute(self) -> Optional[float]:
        """用长窗口内 blink_cnt 的增量估算眨眼频率。样本不足/不可靠时返回 None。

        时间轴用的是 **A 端给的 timestamp（数据时间）**，不是本机收到数据的时刻。
        原因是"次数/分钟"是个速率：离线回放用 --speed 3 快放时，真实 1 秒里会
        灌进 3 秒的数据，按墙钟算出来就是 3 倍频率（实测把 21 次/分算成 64 次/分，
        直接把正常人误判成疲劳）。用数据自带的时间轴算才与回放倍速无关。
        """
        face_samples = [s for s in self._blinks if s.has_face]
        if len(face_samples) < 2:
            return None

        first, last = face_samples[0], face_samples[-1]
        span = last.timestamp - first.timestamp
        if span < self.MIN_BLINK_SPAN_SECONDS:
            return None

        # blink_cnt 是 A 端的累计值，可能因为重连/重置而回退
        delta = last.blink_cnt - first.blink_cnt
        if delta < self.MIN_BLINK_DELTA:
            return None

        # 统计期内不能出现累计值倒退（A 端重连后 blink_cnt 会从头计数），
        # 否则 delta 是负数或偏小，算出来的频率没有意义。
        # 同时校验 timestamp 单调递增 —— 时间轴倒退同样说明数据源重置过。
        for previous, current in zip(face_samples, face_samples[1:]):
            if current.blink_cnt < previous.blink_cnt:
                return None
            if current.timestamp < previous.timestamp:
                return None

        return delta / span * 60.0

    def _sustained_pitch_down(self, face_samples: list[VisionSample]) -> bool:
        """低头是否持续了足够久（单帧低头不算，可能只是捡东西）。"""
        tail: list[VisionSample] = []
        for sample in reversed(face_samples):
            if sample.pitch > config.PITCH_DOWN_THRESHOLD:
                tail.append(sample)
            else:
                break
        if len(tail) < 2:
            return False
        return (tail[0].received_at - tail[-1].received_at) >= config.EAR_TIRED_DURATION

    # ------------------------------------------------------------------
    # 内部：防抖 + 迟滞
    # ------------------------------------------------------------------

    def _apply_debounce(
        self, raw_state: str, reason: str, confidence: float, now: float
    ) -> VisionState:
        """候选状态必须稳定足够久才允许生效。"""
        if raw_state == self._current.state:
            # 已经是当前状态，重置候选，并顺便刷新理由（理由可能变了）
            self._candidate = None
            self._current = VisionState(
                state=self._current.state,
                reason=reason,
                has_face=self._latest_has_face(),
                stale=False,
                confidence=confidence,
            )
            return self._current

        # 出现了不同的状态
        if raw_state != self._candidate:
            self._candidate = raw_state
            self._candidate_since = now
            return self._current  # 还没开始计时，先维持原状态

        # 从 absent 恢复需要更长的稳定时间（迟滞）
        required = (
            config.ABSENT_RECOVER_HOLD
            if self._current.state == config.STATE_ABSENT
            else config.STATE_MIN_HOLD
        )

        if now - self._candidate_since >= required:
            self._current = VisionState(
                state=raw_state,
                reason=reason,
                has_face=self._latest_has_face(),
                stale=False,
                confidence=confidence,
            )
            self._candidate = None

        return self._current

    def _latest_has_face(self) -> bool:
        return bool(self._samples and self._samples[-1].has_face)
