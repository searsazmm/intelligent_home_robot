"""帧级特征——A 模块内部的统一接缝。

这是视觉部分**唯一的**插件接缝。两种后端产出同一种结构：

* :class:`~module_a_vision.face.synthetic.SyntheticFaceBackend`
  按脚本直接产出特征。纯 Python、零依赖、完全确定性。
  它是本机（无摄像头）的主路径，也是全部回归测试的数据来源。
* :class:`~module_a_vision.face.mediapipe_backend.MediaPipeFaceBackend`
  把像素转换成特征。这是唯一的真实后端。

**为什么接缝设在这里，而不是"合成 478 个地标"**：合成地标是在测试特征
提取器，而不是在测试策略——那需要写一大堆几何代码去伪造一个可信的人脸，
而它恰好会把真正的风险（聚合、投票、规则判定）漏掉。把接缝上移一层，
合成后端只需直接声明"这一帧：闭眼、头部右倾 30 度"，下游全部能被真实覆盖。

⚠️ 本结构包含可还原人像的几何信息（``face_bbox`` / ``landmarks``），
**生命周期不得超出 A 模块进程**——不得落盘、不得打印、不得跨模块传输。
出站校验见 :mod:`module_a_vision.privacy.guard`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import Blur, Glasses, Illumination, Occlusion


@dataclass(frozen=True, slots=True)
class FrameQuality:
    """单帧画面质量。"""

    illumination: Illumination = Illumination.NORMAL
    blur: Blur = Blur.LOW
    occlusion: Occlusion = Occlusion.NONE
    glasses: Glasses = Glasses.NONE
    #: 本帧是否可用于统计。``False`` 的帧会计入 ``frames_total`` 但不计入
    #: ``frames_valid``，从而拉低 ``frames_ratio``。
    valid: bool = True


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """单帧的特征集合。

    这是 A 模块内部流转的**唯一**数据结构，也是视觉后端的返回值类型。

    所有角度单位为度，所有比值单位为 [0,1]。
    """

    # ---- 时间 ----
    #: 自采集开始起的单调秒数。用于窗口切分与持续性计算，不受系统对时影响。
    ts: float = 0.0
    #: 墙钟时间戳（Unix 秒），仅用于日志与报文展示。
    wall_ts: float = 0.0

    # ---- 人脸 ----
    has_face: bool = False
    #: 人脸框 [x1, y1, x2, y2]。**敏感：禁止出模块**。
    face_bbox: tuple[int, int, int, int] | None = None

    # ---- 眼部 ----
    #: 眼睑开合比（Eye Aspect Ratio），越小越闭合。
    ear_left: float = 0.0
    ear_right: float = 0.0
    #: 归一化闭眼程度，0=完全睁开，1=完全闭合。由 EAR 与 blendshape 融合得出。
    closure_ratio: float = 0.0

    # ---- 头姿 ----
    pitch_deg: float = 0.0   # 正值为低头
    yaw_deg: float = 0.0     # 正值为向右偏
    roll_deg: float = 0.0    # 正值为向右倾（歪头）

    # ---- 表情 ----
    #: ARKit 命名的 blendshape 系数，如 ``{"mouthFrownLeft": 0.4, ...}``。
    blendshapes: dict[str, float] = field(default_factory=dict)

    # ---- 注意力 ----
    #: 视线偏离正前方的程度 ``[0,1]``，越大越偏离。``None`` = **这一帧估不出来**。
    #:
    #: ``None`` 与 ``0.0`` 语义完全不同，**别把两者揉成一个**：``0.0`` 是
    #: "正对着镜头"（最专注），``None`` 是"不知道"。虹膜关键点缺失（468 点
    #: 模型）、画面质量不合格时都会走到 ``None``。
    #:
    #: 这个区分是**下游正确性的前提**：判"失神"时分母若把"不知道"的帧也算
    #: 进去，偏离占比会被系统性压低 → 结论整体偏向 FOCUSED，不报错、不告警。
    #: 见 :meth:`~module_a_vision.metrics.attention.AttentionTracker.snapshot`。
    gaze_off_ratio: float | None = None

    # ---- 质量 ----
    quality: FrameQuality = field(default_factory=FrameQuality)

    # ---- 调试用（禁止外传）----
    #: MediaPipe 输出的 478 个归一化地标。**敏感：禁止出模块**。
    landmarks: tuple[tuple[float, float, float], ...] | None = None

    def __repr__(self) -> str:  # pragma: no cover - 仅为避免误打印敏感字段
        """刻意不打印 bbox 与 landmarks，防止它们经日志外泄。

        ``gaze_off`` 要单独判一次 ``None``：格式说明符 ``:.2f`` 会直接对
        ``None`` 调 ``__format__`` 并抛 ``TypeError`` —— 而 ``repr()`` 是
        排错路径，**在排错时崩掉**比打不出这个字段糟得多。
        """
        gaze = "None" if self.gaze_off_ratio is None else f"{self.gaze_off_ratio:.2f}"
        return (
            f"FrameFeatures(ts={self.ts:.2f}, has_face={self.has_face}, "
            f"closure={self.closure_ratio:.2f}, "
            f"pose=({self.pitch_deg:.1f},{self.yaw_deg:.1f},{self.roll_deg:.1f}), "
            f"gaze_off={gaze}, valid={self.quality.valid})"
        )

    def blend(self, name: str, default: float = 0.0) -> float:
        """读取一个 blendshape 系数，缺失时返回默认值。

        按**名字**而非索引读取是刻意的——MediaPipe 的枚举顺序与
        Apple ARKit 的文档顺序不一致，硬编码索引迟早出错。
        """
        return self.blendshapes.get(name, default)

    def blend_pair(self, base: str) -> float:
        """读左右成对的 blendshape 并取均值，如 ``blend_pair("eyeBlink")``。"""
        return (self.blend(f"{base}Left") + self.blend(f"{base}Right")) / 2.0


#: 表情识别期望用到的 blendshape 名称。启动时断言其存在性，
#: 缺名立即告警——静默缺失会让表情永远判成"正常"，是极难发现的故障。
EXPECTED_BLENDSHAPES: tuple[str, ...] = (
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawOpen",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker",
    "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
)


def missing_blendshapes(available: dict[str, Any] | set[str]) -> list[str]:
    """返回期望但缺失的 blendshape 名称。供启动自检使用。"""
    have = set(available)
    return [n for n in EXPECTED_BLENDSHAPES if n not in have]
