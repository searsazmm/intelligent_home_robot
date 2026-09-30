"""CSV 回放源——读模块 A 自己导出的 CSV。

存在的理由写在《系统总接口文档》§5.1 里：**开发阶段 A 导出标准 CSV，
B 开启离线模式读文件并行开发**，不必依赖实时摄像头与 Socket。这个类就是
那条离线通路的入口，所以它必须能吃下 A 真实写出的文件，而不是"理论上能吃下"。

列格式（V1，按《系统总接口文档》§3.2）
----------------------------------------

::

    timestamp,has_face,ear,blink_cnt,pitch,yaw,roll,emo_feature

V1 只有 8 列，这里有两个**必须显式处理的缺口**（不处理会静默产生错误的指标）：

1. **没有 blendshape 列**。闭眼程度本应由
   :func:`shared.geometry.fuse_closure` 融合 EAR 与 ``eyeBlink`` 两个通道，
   但 V1 只有 EAR。如果只填 EAR、让 blendshape 缺省为 0，融合结果会被
   0.6 的权重压到 **0.6 以下**，而"闭眼"的门槛是 **0.80**——于是回放数据
   永远判不出 ``closed``，疲劳链条整体失效，且**不报任何错**。
   这里的做法是把 EAR 反推的闭眼度同时写进 ``eyeBlink`` 通道，使
   ``fuse_closure`` 的两个通道一致（0.6c + 0.4c = c）。这是刻意的补全，
   不是伪造：V1 数据本来就只有 EAR 一个眼部信号。
2. **没有视线列**。``gaze_off_ratio`` 只能是 ``None``（"估不出来"），
   因此**回放数据的注意力判定不可信**（会一律判成"专注"）。CSV 回放只覆盖
   头姿、眼部、疲劳、表情四条链路，注意力必须在真机联调阶段验证。

   这里**刻意不填 ``0.0``**。``0.0`` 的意思是"视线完全对正前方"，也就是
   **专注满分** —— 拿它去顶替"这一列不存在"，就是在伪造一个"老人一直很专注"
   的结论，而且没有任何东西会报错。填 ``None`` 至少让这件事显式：
   :attr:`~module_a_vision.metrics.attention.AttentionMetrics.gaze_frames`
   会等于 0，看到 ``label=FOCUSED`` 配 ``gaze_frames=0`` 就该读成
   "这一窗没有测量"，而不是"很专注"。

``emo_feature`` 的取值
----------------------

V1 是 ``normal`` / ``low`` / ``tired``，V2 扩容为 ``normal`` / ``tired`` /
``sad`` / ``upset``（《系统设计方案》§3.5 规定旧数据 ``low → sad``）。
这里按标签映射到一组能复现该标签的 blendshape——CSV 里只有标签没有系数，
下游的表情分类器需要系数才能工作，两者靠这张映射表对齐。

``blink_cnt`` 是**累计值**，不能直接映射成 ``blink_rate_per_min``
（疲劳公式要的是频率，见 :mod:`module_a_vision.metrics.fatigue`）。
默认在计数增加的那一帧注入一次闭合尖峰，让
:class:`~module_a_vision.metrics.eye.EyeTracker` 自己数出眨眼——
这样回放数据的眨眼频率与真机同源。注入是"补一刀"，不是原始数据，
所以它是可关闭的（``synthesize_blinks=False``）。
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from shared.frame_features import EXPECTED_BLENDSHAPES, FrameFeatures, FrameQuality
from shared.geometry import fuse_closure, closure_from_ear

from .base import BaseFeatureSource, CaptureConfig, CaptureError
from .synthetic import STATE_SPECS

#: V1 必需的列。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp", "has_face", "ear", "blink_cnt", "pitch", "yaw", "roll", "emo_feature",
)

#: 可选列。V2 导出若带上它们就优先采用；缺了也不影响回放。
OPTIONAL_COLUMNS: tuple[str, ...] = ("closure_ratio", "gaze_off_ratio")

#: ``emo_feature`` → 合成状态名。状态名对应的 blendshape 直接复用
#: :data:`module_a_vision.capture.synthetic.STATE_SPECS`，避免同一套系数
#: 在两个文件里各调一遍、慢慢漂移。
EMO_FEATURE_TO_STATE: dict[str, str] = {
    "normal": "normal",
    "low": "sad",      # 设计方案 §3.5：V1 的 low 迁移为 sad
    "tired": "tired",
    "sad": "sad",
    "upset": "upset",
}

#: 注入眨眼时用的 EAR（很小的值 → 闭眼程度接近 1）。
BLINK_EAR = 0.10

#: EAR 的合理上限。超过它说明导出方写进去的可能不是 EAR，而是闭眼度
#: （0–1 的比值），两者混用会让疲劳指标整体错位，所以必须察觉。
EAR_SANE_MAX = 0.60


class _RowError(Exception):
    """单行解析失败。整文件不因此中断——离线回放要能容忍中间几行脏数据。"""


@dataclass(frozen=True, slots=True)
class _ParsedRow:
    ts: float
    has_face: bool
    ear: float
    blink_cnt: int
    pitch: float
    yaw: float
    roll: float
    emo_feature: str
    closure_ratio: float | None
    gaze_off_ratio: float | None


class CsvReplaySource(BaseFeatureSource):
    """按行回放 A 导出的 CSV，产出 :class:`FrameFeatures`。

    只实现特征路径：CSV 里没有像素，硬造一个 ``read()`` 只会让上游
    以为"回放也是像素源"。需要像素请用 :class:`VideoFileSource`。
    """

    def __init__(
        self,
        path: str | Path,
        cfg: CaptureConfig | None = None,
        rebase_ts: bool = True,
        synthesize_blinks: bool = True,
        delimiter: str = ",",
        encoding: str = "utf-8-sig",
    ) -> None:
        """
        :param rebase_ts: 把首行时间戳平移到 0。A 导出的 ``timestamp`` 是
            "程序运行时间戳"，可能从任意值开始；窗口切分要的是**相对**秒数。
            平移只改原点，不改变行与行之间的间隔，因此丢帧造成的空隙被保留。
        :param synthesize_blinks: 见模块文档关于 ``blink_cnt`` 的说明。
        :param encoding: 默认 ``utf-8-sig``——Windows 上的表格工具很容易
            在文件头写一个 BOM，用 ``utf-8`` 读会把第一列列名读成
            ``\\ufefftimestamp``，然后报"缺少 timestamp 列"，极难排查。
        """
        super().__init__(cfg)
        self._path = Path(path)
        self._rebase = rebase_ts
        self._synthesize_blinks = synthesize_blinks
        self._delimiter = delimiter
        self._encoding = encoding

        self._reader: Iterator[dict[str, str]] | None = None
        self._file = None
        self._fields: tuple[str, ...] = ()
        #: 数据行总数（不含表头）。
        self.rows_total: int = 0
        #: 解析失败被跳过的行数。
        self.skipped_rows: int = 0
        #: 因时间戳倒退被强制推平的行数（倒退会让窗口判定错乱）。
        self.non_monotonic_rows: int = 0
        #: ``emo_feature`` 出现未知取值的行数。
        self.unknown_emo_rows: int = 0
        #: 疑似"把闭眼度当成 EAR 写"的行数。
        self.suspect_ear_rows: int = 0

        self._first_ts: float | None = None
        self._prev_blink_cnt: int | None = None

        #: 启动自检采集到的问题描述，供上层在日志里原样打印。
        self.self_check_notes: list[str] = []

    # ------------------------------------------------------------ 生命周期

    def open(self) -> None:
        if not self._path.exists():
            raise CaptureError(
                f"CSV 文件不存在：{self._path.resolve()}\n"
                "请确认路径；相对路径是相对于**进程的工作目录**，不是本文件所在目录。"
            )
        self._open_file()
        header = self._peek_header()
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise CaptureError(
                f"CSV 缺少必需的列：{'、'.join(missing)}\n"
                f"  文件：{self._path.resolve()}\n"
                f"  实际列：{'、'.join(header) or '（空）'}\n"
                f"  期望列：{'、'.join(REQUIRED_COLUMNS)}（见《系统总接口文档》§3.2）"
            )
        self._fields = header
        self.rows_total = self._count_data_rows()

        self._index = 0
        self._exhausted = False
        self._last_ts = 0.0
        self._pace_ts = 0.0
        self._last_wall = None
        self._first_ts = None
        self._prev_blink_cnt = None
        self.skipped_rows = 0
        self.non_monotonic_rows = 0
        self.unknown_emo_rows = 0
        self.suspect_ear_rows = 0
        self._opened = True

    def _open_file(self):
        """打开文件并定位到第一行数据。"""
        self._close_file()
        try:
            self._file = self._path.open("r", encoding=self._encoding, newline="")
        except OSError as exc:
            raise CaptureError(f"无法读取 CSV 文件：{self._path.resolve()}（{exc}）") from exc
        self._reader = csv.DictReader(self._file, delimiter=self._delimiter)

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._reader = None

    def _peek_header(self) -> tuple[str, ...]:
        """读表头并做列名规范化（去空白、去 BOM、统一小写）。

        列名是跨模块约定，但导出脚本多一个空格、少一个下划线是常事，
        为此让整个回放失败不值得；规范化后仍缺列才是真问题。
        """
        try:
            with self._path.open("r", encoding=self._encoding, newline="") as f:
                reader = csv.reader(f, delimiter=self._delimiter)
                raw = next(reader, [])
        except OSError as exc:
            raise CaptureError(f"无法读取 CSV 文件：{self._path.resolve()}（{exc}）") from exc
        except StopIteration:  # pragma: no cover - next(reader, []) 已兜住
            raw = []
        except UnicodeDecodeError as exc:
            raise CaptureError(
                f"CSV 编码无法识别：{self._path.resolve()}\n"
                f"默认按 utf-8-sig 读取，失败可显式传 encoding='gbk'。（{exc}）"
            ) from exc
        return tuple(_norm_column(c) for c in raw)

    def _count_data_rows(self) -> int:
        """数一遍数据行数，只为 :meth:`describe` 能报出规模。"""
        try:
            with self._path.open("r", encoding=self._encoding, newline="") as f:
                return max(0, sum(1 for _ in f) - 1)
        except (OSError, UnicodeDecodeError):
            return 0

    def close(self) -> None:
        self._close_file()
        super().close()

    def describe(self) -> str:
        rows = self.rows_total or "?"
        return (
            f"CSV 回放：{self._path.name}（{rows} 行，"
            f"{'重定基于 0' if self._rebase else '沿用原始时间戳'}，"
            f"{'循环' if self.cfg.loop else '不循环'}，"
            f"{'注入眨眼' if self._synthesize_blinks else '不注入眨眼'}）"
        )

    # ------------------------------------------------------------ 读取

    def read_features(self) -> FrameFeatures | None:
        """产出下一行对应的特征；``None`` 表示已到文件末尾（``loop=False``）或达上限。"""
        if not self._opened or self._reader is None:
            raise CaptureError("CSV 源尚未 open()。请先调用 open()。")

        while True:
            if self._over_budget():
                self._exhausted = True
                return None

            row = next(self._reader, None)
            if row is None:
                if not self.cfg.loop:
                    self._exhausted = True
                    return None
                # 循环：重新打开文件从头来。首行时间戳的基准要一并复位，
                # 否则新一轮的 ts 会接着上一轮继续涨（时间轴被拉长）。
                self._open_file()
                self._first_ts = None
                continue

            try:
                parsed = self._parse_row(row)
            except _RowError:
                # 单行脏数据不该让整段回放中断——离线开发时 csv 常被手工改过。
                self.skipped_rows += 1
                continue

            return self._to_features(parsed)

    # ------------------------------------------------------------ 解析

    def _parse_row(self, row: dict[str, str]) -> _ParsedRow:
        def num(key: str, default: float = 0.0) -> float:
            raw = _get(row, key)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise _RowError(f"{key}={raw!r} 不是数字") from exc

        raw_has_face = _get(row, "has_face")
        if raw_has_face is None:
            raise _RowError("缺少 has_face 列")

        raw_blink = _get(row, "blink_cnt")
        blink_cnt = 0
        if raw_blink not in (None, ""):
            try:
                blink_cnt = int(float(raw_blink))
            except ValueError as exc:
                raise _RowError(f"blink_cnt={raw_blink!r} 不是整数") from exc

        closure = None
        raw_closure = _get(row, "closure_ratio")
        if raw_closure not in (None, ""):
            try:
                closure = float(raw_closure)
            except ValueError as exc:
                raise _RowError(f"closure_ratio={raw_closure!r} 不是数字") from exc

        gaze = None
        raw_gaze = _get(row, "gaze_off_ratio")
        if raw_gaze not in (None, ""):
            try:
                gaze = float(raw_gaze)
            except ValueError as exc:
                raise _RowError(f"gaze_off_ratio={raw_gaze!r} 不是数字") from exc

        return _ParsedRow(
            ts=num("timestamp"),
            has_face=_parse_bool(raw_has_face),
            ear=num("ear"),
            blink_cnt=blink_cnt,
            pitch=num("pitch"),
            yaw=num("yaw"),
            roll=num("roll"),
            emo_feature=(_get(row, "emo_feature") or "normal").strip().lower(),
            closure_ratio=closure,
            gaze_off_ratio=gaze,
        )

    def _to_features(self, r: _ParsedRow) -> FrameFeatures:
        # ---- 时间戳：重定基 + 强制单调 ----
        if self._first_ts is None:
            self._first_ts = r.ts
        ts = r.ts - self._first_ts if self._rebase else r.ts
        if ts < self._last_ts:
            # 时间戳倒退会让"持续时长"变成负数、窗口永远关不上。
            # 与其静默错乱，不如推平并计数——行数会出现在 describe() 里。
            self.non_monotonic_rows += 1
            ts = self._last_ts + 1e-6
        self._last_ts = ts
        self._index += 1
        self._pace(ts)

        wall_ts = time.time()

        if not r.has_face:
            # 无人脸：按《系统总接口文档》§3.3，其余字段填默认值。
            self._prev_blink_cnt = r.blink_cnt
            return FrameFeatures(
                ts=ts,
                wall_ts=wall_ts,
                has_face=False,
                quality=self._quality(),
            )

        # ---- 眼部 ----
        ear = r.ear
        if ear > EAR_SANE_MAX or ear < 0.0:
            self.suspect_ear_rows += 1
            if len(self.self_check_notes) < 8:
                self.self_check_notes.append(
                    f"第 {self._index} 行的 ear={ear:g} 超出合理范围 [0,{EAR_SANE_MAX}]，"
                    "已夹紧。这通常意味着导出时写进去的是闭眼度（0–1）而不是 EAR，"
                    "或者列顺序对不上。"
                )
            ear = min(max(ear, 0.0), EAR_SANE_MAX)

        blink_injected = False
        if self._synthesize_blinks and self._prev_blink_cnt is not None:
            if r.blink_cnt > self._prev_blink_cnt:
                blink_injected = True
        self._prev_blink_cnt = r.blink_cnt
        if blink_injected:
            ear = BLINK_EAR

        # ---- 闭眼程度 ----
        if r.closure_ratio is not None and not blink_injected:
            closure = _clamp01(r.closure_ratio)
        else:
            # V1 没有 blendshape 列：把 EAR 反推的闭眼度同时写进 eyeBlink 通道，
            # 使 fuse_closure 的两个通道一致（0.6c + 0.4c = c）。理由见模块文档。
            closure = fuse_closure(ear, closure_from_ear(ear))

        # ---- 表情 ----
        bl = self._blendshapes(r)
        # 眼部系数与 closure 保持一致，下游若改用 blendshape 通道重算也不会打架。
        bl["eyeBlinkLeft"] = closure
        bl["eyeBlinkRight"] = closure

        return FrameFeatures(
            ts=ts,
            wall_ts=wall_ts,
            has_face=True,
            ear_left=ear,
            ear_right=ear,
            closure_ratio=closure,
            pitch_deg=r.pitch,
            yaw_deg=r.yaw,
            roll_deg=r.roll,
            blendshapes=bl,
            # V1 CSV **没有**视线列（见模块文档第 2 条）：那就如实给 `None`。
            #
            # ⚠️ 这里原先写的是 `_clamp01(r.gaze_off_ratio or 0.0)`。
            # `or 0.0` 会把刚在 §7 建立起来的 `None` 又抹平成 `0.0`，
            # 而 `0.0` 在这条链路上的意思是**视线完全对正**（专注满分）——
            # 于是"没有视线这一列"被读成"一直很专注"，且全程无提示。
            # 现在 `None` 一路传到 `metrics/attention.py`，由那里的
            # `gaze_frames` 把"这一窗没有视线信息"显式暴露出来。
            gaze_off_ratio=(
                None if r.gaze_off_ratio is None else _clamp01(r.gaze_off_ratio)
            ),
            quality=self._quality(),
        )

    def _blendshapes(self, r: _ParsedRow) -> dict[str, float]:
        """按 ``emo_feature`` 取一组系数。未知名回落到 ``normal`` 并计数。"""
        state = EMO_FEATURE_TO_STATE.get(r.emo_feature)
        if state is None:
            self.unknown_emo_rows += 1
            if len(self.self_check_notes) < 8:
                self.self_check_notes.append(
                    f"emo_feature={r.emo_feature!r} 不是已知取值"
                    f"（{'、'.join(EMO_FEATURE_TO_STATE)}），已按 normal 处理。"
                )
            state = "normal"
        preset = STATE_SPECS[state].blendshapes
        # 深拷贝一份：FrameFeatures 的 blendshapes 是可变的，共享同一字典
        # 会让任何一处就地修改污染后续所有帧。
        full = {name: 0.0 for name in EXPECTED_BLENDSHAPES}
        full.update(preset)
        return full

    def _quality(self) -> FrameQuality:
        """CSV 里没有画质字段，只能给"正常"。

        刻意**不**去猜画质：猜错会让"画面不可用"这条 L4 链路在回放模式下
        要么永远不触发、要么永远触发，比不猜更危险。画质链路必须在真机联调
        阶段验证，CSV 回放不覆盖它。
        """
        return _REPLAY_QUALITY


# ================================================================ 工具

#: 回放数据的画质：一律"正常且有效"（原因见 :meth:`CsvReplaySource._quality`）。
#: :class:`FrameQuality` 是冻结的，可以安全共享同一个实例。
_REPLAY_QUALITY = FrameQuality()

#: UTF-8 BOM 的码位。写成 chr() 而不是字面量，避免源码里出现看不见的字符。
_BOM = chr(0xFEFF)


def _norm_column(name: str) -> str:
    """列名规范化：去 BOM、去首尾空白、转小写。"""
    return name.replace(_BOM, "").strip().lower()


def _get(row: dict[str, str], key: str) -> str | None:
    """按规范化后的列名取值（容忍导出脚本多写的空格与大小写）。"""
    for k, v in row.items():
        if k is None:
            continue
        if _norm_column(k) == key:
            return v
    # DictReader 在行尾多出字段时会把它们塞进 None 键，此处忽略。
    return None


def _parse_bool(raw: str) -> bool:
    """宽松解析布尔列。A 用 Python 的 ``True/False``，表格工具可能写出别的。"""
    v = raw.strip().lower()
    if v in ("true", "1", "yes", "y", "t", "是"):
        return True
    if v in ("false", "0", "no", "n", "f", "", "否"):
        return False
    return bool(v)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
