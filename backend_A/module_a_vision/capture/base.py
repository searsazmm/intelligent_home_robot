"""采集源契约——A 模块唯一接触原始像素的地方。

四种源可以互换，上层（``main`` / ``server``）因此不必知道帧从哪来：

* :class:`FrameSource`——产出**原始帧**（``np.ndarray``，BGR 三通道）。
  摄像头与视频文件走这条。真实管线必须保持这个形状。
* :class:`FeatureSource`——直接产出 :class:`~shared.frame_features.FrameFeatures`。
  合成源与 CSV 回放走这条。

为什么接缝有两条
----------------

接缝被刻意抬到 ``FrameFeatures`` 这一层（见 :mod:`shared.frame_features` 的
说明）：合成 478 个地标是在测试几何代码，而不是在测试策略。所以合成源
**不必**假装自己是一路摄像头——它直接声明"这一帧：闭眼、头右倾 34 度"，
下游的聚合、投票、规则判定全部被真实覆盖。

但像素路径不能因此消失：它是真机上唯一存在的路径，把它删掉会让
"管线形状"和"真机形状"悄悄分叉。所以两个协议都保留，谁用哪条由
具体源决定，由 :func:`is_frame_source` / :func:`is_feature_source` 判定。

``read()`` 返回 ``None`` 的两种含义
-----------------------------------

* **本帧没取到**：摄像头偶发丢帧、视频解码失败一帧。源仍然活着，继续读。
* **数据源已耗尽**：视频不循环且播完了、``max_frames`` 到了上限。

两者处置方式完全不同（前者继续、后者退出），所以需要区分时读
:attr:`BaseCapture.exhausted`，不要靠 ``None`` 猜。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from shared.frame_features import FrameFeatures


class CaptureError(RuntimeError):
    """采集源无法建立或无法继续。

    消息面向运维人员，一律用中文写清楚**下一步该怎么做**——
    "打开失败" 这种话在无人值守的设备上等于没说。
    """


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """采集参数。四种源共用，各源只取自己用得到的那几项。"""

    #: 目标帧率。既是合成/回放的时间轴步长，也是向摄像头请求的采集帧率。
    fps: float = 15.0
    width: int = 640
    height: int = 480
    #: 最多产出多少帧；``None`` 表示不限。
    max_frames: int | None = None
    #: 数据耗尽（视频播完、剧本走完）后是否从头再来。
    loop: bool = True
    #: 是否按时间轴真实等待。默认 ``False``：离线回放与测试要的是快，
    #: 十秒的剧本不该真跑十秒。只有实时演示才需要打开。
    real_time: bool = False
    #: 回放倍速，仅在 ``real_time=True`` 时生效（2.0 = 两倍速）。
    speed: float = 1.0

    @property
    def frame_interval_sec(self) -> float:
        """单帧时间步长。"""
        return 1.0 / self.fps if self.fps > 0 else 0.0


@runtime_checkable
class FrameSource(Protocol):
    """原始帧源。

    实现者必须保证：``open()`` 之后 ``read()`` 可反复调用；
    ``read()`` 返回 ``None`` 表示本帧不可用（见模块文档）；
    ``close()`` 可重复调用且不抛异常。
    """

    def open(self) -> None: ...

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...

    def describe(self) -> str: ...


@runtime_checkable
class FeatureSource(Protocol):
    """帧特征源。

    与 :class:`FrameSource` 的区别只在于 ``read_features``——
    它直接给出 :class:`~shared.frame_features.FrameFeatures`。
    """

    def open(self) -> None: ...

    def read_features(self) -> FrameFeatures | None: ...

    def close(self) -> None: ...

    def describe(self) -> str: ...


class BaseCapture(ABC):
    """采集源公共实现：生命周期、时间轴推进、回放节奏。

    子类只需实现 :meth:`open` 与 :meth:`describe`，并负责在读到数据后
    调用 :meth:`_tick`（或自行设置 ``_last_ts``）。
    """

    def __init__(self, cfg: CaptureConfig | None = None) -> None:
        self.cfg = cfg or CaptureConfig()
        #: 已产出的帧数。既用于计数，也是时间轴的步进量。
        self._index = 0
        self._opened = False
        #: 数据源已耗尽（不是"本帧失败"）。
        self._exhausted = False
        #: 最近一帧的单调时间戳（秒）。
        self._last_ts = 0.0
        #: 回放节奏对齐用的墙钟基准。
        self._last_wall: float | None = None
        #: 上一次做过节奏对齐的时间轴位置。与 ``_last_ts`` 分开维护——
        #: ``_last_ts`` 已经被 :meth:`_tick` 更新成了本帧的时刻，拿它当
        #: "上一帧"会得到恒为真的回绕判断，回放节奏就永远对不齐。
        self._pace_ts = 0.0

    # ------------------------------------------------------------ 状态

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def exhausted(self) -> bool:
        """数据源是否已耗尽。真机上据此决定退出还是重连。"""
        return self._exhausted

    @property
    def frame_index(self) -> int:
        """已产出的帧数。"""
        return self._index

    @property
    def last_ts(self) -> float:
        """最近一帧的单调时间戳（秒，自采集开始）。

        原始帧源没有地方携带时间戳，而 :class:`~shared.frame_features.FrameFeatures`
        的 ``ts`` 又必须由采集侧给出——这个属性就是那个缺口。
        特征源不需要它（``FrameFeatures.ts`` 自带）。
        """
        return self._last_ts

    # ------------------------------------------------------------ 生命周期

    @abstractmethod
    def open(self) -> None:
        """建立连接/加载数据。失败必须抛 :class:`CaptureError` 并说明原因。"""

    @abstractmethod
    def describe(self) -> str:
        """一句话说明自己是什么，供启动日志使用。"""

    def close(self) -> None:
        """释放资源。幂等：未打开或已关闭时调用不报错。"""
        self._opened = False

    def __enter__(self) -> "BaseCapture":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ 时间轴

    def _tick(self) -> float:
        """按 ``cfg.fps`` 推进一帧，返回该帧的单调时间戳（秒）。

        ``ts`` 用"第几帧 ÷ 帧率"而不是墙钟——它必须单调、不受系统对时影响，
        因为窗口切分与持续时长判定全部建立在它上面（见 ``FrameFeatures.ts``）。
        """
        ts = self._index / self.cfg.fps if self.cfg.fps > 0 else float(self._index)
        self._index += 1
        self._last_ts = ts
        return ts

    def _over_budget(self) -> bool:
        """是否已达 ``max_frames`` 上限。"""
        return self.cfg.max_frames is not None and self._index >= self.cfg.max_frames

    def _pace(self, ts: float) -> None:
        """按 ``real_time`` / ``speed`` 对齐回放节奏。

        用**相邻两帧的时间差**算等待量，而不是"帧号 × 帧率"——后者在
        数据源循环回绕（时间轴跳回 0）时会算出巨大的负等待，表现为
        "回放突然全速冲刺"。差值法对回绕天然正确。
        """
        if not self.cfg.real_time:
            self._pace_ts = ts
            return
        now = time.monotonic()
        if self._last_wall is None or ts <= self._pace_ts:
            # 首帧，或时间轴回绕：重置基准，本帧不等待。
            self._last_wall = now
            self._pace_ts = ts
            return
        want = (ts - self._pace_ts) / max(self.cfg.speed, 1e-6)
        delay = want - (now - self._last_wall)
        if delay > 0:
            time.sleep(delay)
            now = time.monotonic()
        self._last_wall = now
        self._pace_ts = ts


class BaseFeatureSource(BaseCapture):
    """帧特征源的公共实现。

    只有一条额外要求：``read_features()``。
    """

    @abstractmethod
    def read_features(self) -> FrameFeatures | None:
        """产出下一帧特征；``None`` 表示本帧无数据或已耗尽（见 :attr:`exhausted`）。"""


# ================================================================ 判定与统一读取

def is_frame_source(obj: object) -> bool:
    """对象是否满足 :class:`FrameSource`（有 ``read()``）。

    只检查方法是否存在，不检查签名——协议是给人和静态检查看的，
    运行时能做的最多是"有没有这个方法"。
    """
    return isinstance(obj, FrameSource)


def is_feature_source(obj: object) -> bool:
    """对象是否满足 :class:`FeatureSource`（有 ``read_features()``）。"""
    return isinstance(obj, FeatureSource)


def read_any(src: object) -> np.ndarray | FrameFeatures | None:
    """从任意采集源读一帧，返回类型取决于源的种类。

    ⚠️ 返回值类型是**联合类型**，调用方必须先判定源的种类再取值：

    ::

        if is_frame_source(src):
            frame = src.read()                 # np.ndarray | None
        elif is_feature_source(src):
            feats = src.read_features()        # FrameFeatures | None

    这个函数只是为了让"通用循环"少写一次分支，它**不**做类型伪装：
    拿到 ``FrameFeatures`` 就不要当成像素去喂人脸后端。
    """
    if is_frame_source(src):
        return src.read()  # type: ignore[attr-defined]
    if is_feature_source(src):
        return src.read_features()  # type: ignore[attr-defined]
    raise CaptureError(
        "该对象既不是帧源也不是特征源：它没有 read() 或 read_features()。"
        "请传入 module_a_vision.capture 里的四种源之一。"
    )
