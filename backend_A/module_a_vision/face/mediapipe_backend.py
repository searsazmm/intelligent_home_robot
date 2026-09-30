"""MediaPipe Tasks 后端——把像素变成 :class:`FrameFeatures`。

真机标定记录（2026-09，Windows + 640×480 摄像头）
================================================

本文件曾长期带着"**未在本机验证过**"的警告——当时开发机没有摄像头。现在
已在真机上跑通并逐项标定。**下面每条都是实测结论，不要再凭印象改回去。**

已标定、可以信
--------------

* **pitch 符号（``facial_transformation_matrixes`` 路径）**：低头为正、
  抬头为负、平视接近 0。实测低头 +17.5°、抬头 -9.5°、平视 -4.6°，
  大角度时低头可达 +22°、抬头可达 -57°。
  ⚠️ 这里**不需要**任何符号翻转——``normalize_angles`` 曾经翻过一次，
  那个负号是错的，详见该函数的文档。
* **人脸检出率**：正常光照、正常坐姿下 100%（62/62 帧）。
* **roll 量级**：左右歪头实测 -30.4° / +40.8°，**够得到 R6 的 25° 阈值**，
  R6 在真机上不是死规则。
* **roll 的符号**与下文旧记录说的相反（向左歪为负），但
  :mod:`shared.geometry` 全程用 ``abs()``，**不承重**，因此未做修正。

仍未验证 / 已知有问题
---------------------

1. ⚠️ **``solvePnP`` 回退路径不可用**。强制走该路径实测：pitch **恒为
   ±90°**（钳位饱和），685 帧无一例外——模型点用的是 Y 轴朝上的经典头部
   模型，而 OpenCV 相机系 Y 轴朝下，两者差一个绕 X 轴的 180°。
   该路径目前不会触发（真机上 ``facial_transformation_matrixes`` 一直都在，
   实测 ``used_pnp_fallback = False``），但**别把它当成可用的退路**：
   一旦 MediaPipe 不再返回矩阵，它会安静地把每一帧都变成俯仰 90°。
2. **pitch 与 roll 的耦合**：纯歪头会把 pitch 一并抬到 +23°~+35°。因此
   一个中等幅度的歪头（roll < 25°，不触发 ``is_tilted``）**可能**误触发
   ``is_bowed``。目前 R1–R6 没有任何规则消费 ``bowed``，所以不产生误报，
   但将来若要接入必须先在真机上解决这个耦合。
3. **大角度姿态下人脸会丢失**：持续歪头/深低头时实测有 8~12 秒完全检不到
   人脸。"歪倒 + 持续闭眼"这条 R6 判据要求的姿态，恰好是最容易丢脸的姿态。
4. **画面质量阈值**（照度、模糊、眼镜）仍是经验值，未在真实居家光照下标定，
   摄像头分辨率/白平衡一变就要重标。见下方各常量的注释。
5. **眼镜启发式**只有粗略的亮度对比，没有标注样本可验证。
6. **视线估计**由虹膜关键点相对眼角的位置粗略换算，未经标定。
7. **左右眼编号**（:data:`EYE_IDX_RIGHT` / :data:`EYE_IDX_LEFT`）遵循
   MediaPipe 的编号习惯（33 为**本人右眼**外角）。真机上若发现左右颠倒，
   交换这两个常量即可，不影响后续融合。

已核对的硬事实（踩过的坑，**不要改回去**）
------------------------------------------

* MediaPipe 1.0.1 **删除了** ``mp.solutions``。必须走 Tasks API：
  ``from mediapipe.tasks import python as mp_python`` 与
  ``from mediapipe.tasks.python import vision``。
* ``FaceLandmarker`` 必须**只创建一次**并复用。逐帧重建会泄漏 delegate 与
  线程池，几分钟就能把进程拖死；本模块用一个带引用计数的模块级缓存
  强制这一点（:data:`_LANDMARKER_CACHE`）。
* ``detect_for_video`` 的时间戳必须**严格单调递增**，重复即抛异常。
* **人脸关键点不带 ``presence``**。Tasks API 的 ``NormalizedLandmark`` 有
  这个字段但**恒为 ``None``**（那是姿态模型的量，人脸模型不填）。任何
  "用 presence 判遮挡"的逻辑都必须把 ``None`` 与 ``0.0`` 分开处理——
  把前者当成后者会让每一帧都判"严重遮挡"，A 在真实摄像头上一个可用窗口
  都发不出来。详见 :meth:`MediaPipeFaceBackend._refine_quality`。
* blendshape 必须**按名字**取，绝不能按下标——MediaPipe 的枚举顺序与
  Apple ARKit 文档顺序不一致，按索引取会在某次版本升级后静默错位。
* VIDEO 模式是**非确定性**的：同一段视频两次跑出的系数会有差异。因此它
  **不能**作为回归测试的数据来源——这正是合成后端存在的理由。
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shared.enums import Blur, Glasses, Illumination, Occlusion
from shared.frame_features import (
    EXPECTED_BLENDSHAPES,
    FrameFeatures,
    FrameQuality,
    missing_blendshapes,
)
from shared.geometry import fuse_closure

from ..metrics.head_pose import euler_from_matrix, normalize_angles
from .base import BaseFaceBackend, FaceBackendError

try:  # 没装 opencv 的其他路径不受影响
    import cv2
except ImportError:  # pragma: no cover - 本机已装
    cv2 = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

# ================================================================ 模型

#: 指定模型文件路径的环境变量，便于部署时把模型放在别处。
MODEL_ENV_VAR = "FACE_LANDMARKER_MODEL"

#: 默认模型位置：``<项目根>/models/face_landmarker.task``。
DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"

#: 官方模型下载地址（写进报错里，省得部署的人到处找）。
MODEL_DOWNLOAD_HINT = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ================================================================ 关键点编号

#: 本人**右**眼（画面左侧）的 6 个点，顺序为 EAR 公式的 p1..p6：
#: p1/p4 是左右眼角，p2/p6 与 p3/p5 是上下眼睑的两对。下同。
EYE_IDX_RIGHT: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
#: 本人**左**眼（画面右侧）。
EYE_IDX_LEFT: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

#: 虹膜关键点（478 点模型的 468–477）。用于粗略估计视线。
IRIS_IDX_RIGHT: tuple[int, ...] = (468, 469, 470, 471, 472)
IRIS_IDX_LEFT: tuple[int, ...] = (473, 474, 475, 476, 477)

#: ``solvePnP`` 回退用的 3D 参照点（毫米，原点在鼻尖）。
#: 注意：本路径的**符号约定未经验证**，见模块文档第 1 条。
MODEL_POINTS_3D: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),        # 鼻尖
    (0.0, -63.6, -12.5),    # 下巴
    (-43.3, 32.7, -26.0),   # 右眼外角（画面左）
    (43.3, 32.7, -26.0),    # 左眼外角（画面右）
    (-28.9, -28.9, -24.1),  # 右嘴角（画面左）
    (28.9, -28.9, -24.1),   # 左嘴角（画面右）
)
#: 与 :data:`MODEL_POINTS_3D` 一一对应的关键点下标。
LANDMARK_IDX_PNP: tuple[int, ...] = (1, 152, 33, 263, 61, 291)

#: 判定"关键点被挡住"用的下标集合（双眼 + 嘴 + 鼻）。
KEY_LANDMARKS_FOR_OCCLUSION: tuple[int, ...] = (
    33, 133, 362, 263, 61, 291, 1, 13, 14, 152,
)

# ================================================================ 质量阈值
# ⚠️ 以下阈值均**未在真实居家光照下标定**，且模糊度是在归一化到
#    320 宽之后计算的（见 _blur_score），换分辨率不必重标，换镜头要重标。

#: 灰度均值低于/高于即判过暗/过曝。
LUMA_LOW = 50.0
LUMA_HIGH = 210.0

#: 拉普拉斯方差：低于前者判"严重模糊"，低于后者判"轻度模糊"。
LAPLACIAN_BLURRY = 60.0
LAPLACIAN_SHARP = 150.0

#: 模糊度分析前统一缩放到这个宽度，避免阈值随摄像头分辨率漂移。
BLUR_ANALYSIS_WIDTH = 320

#: 关键点 presence 低于此值判"被遮挡"。
OCCLUSION_PRESENCE_PARTIAL = 0.60
OCCLUSION_PRESENCE_SEVERE = 0.30

#: 眼部区域亮度 / 面颊区域亮度 低于此比值，怀疑墨镜（镜片挡光）。
SUNGLASSES_LUMA_RATIO = 0.55

#: 眼部近饱和像素占比高于此值（且明显高于整幅画面），怀疑镜片反光 → 老花镜。
READING_GLASSES_SPECULAR_RATIO = 0.06

#: 视线偏移的放大系数：虹膜偏移 / 眼宽 达到 1/4 即认为完全偏离。
GAZE_OFFSET_GAIN = 4.0


# ================================================================ 单例缓存

@dataclass
class _CachedLandmarker:
    """缓存条目。``refs`` 记录还有几个后端在用它。"""

    landmarker: Any
    refs: int = 0


#: 模型路径 → landmarker。**这是"只创建一次"这条硬性要求的落点**：
#: 多个后端实例（例如重连后新建）共享同一个 landmarker，引用计数归零才真正关闭。
_LANDMARKER_CACHE: dict[str, _CachedLandmarker] = {}
_LANDMARKER_LOCK = threading.Lock()


def _load_mediapipe():
    """延迟导入 mediapipe。

    刻意不在模块顶层导入：合成路径（本机主路径）与 CSV 回放都不需要
    mediapipe，requirements.txt 里也写明了"无摄像头场景不需要它"。
    顶层导入会让 ``import module_a_vision.face`` 直接失败。
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - 本机已装
        raise FaceBackendError(
            "未安装 mediapipe，无法使用真实人脸后端。\n"
            "请安装：pip install mediapipe\n"
            "或在无摄像头场景改用 SyntheticFaceBackend（纯直通，零依赖）。"
        ) from exc
    return mp, mp_python, vision


# ================================================================ 后端

class MediaPipeFaceBackend(BaseFaceBackend):
    """基于 MediaPipe FaceLandmarker（Tasks API，VIDEO 模式）的人脸后端。

    ⚠️ 未在本机验证，见模块文档。
    """

    name = "mediapipe-face-landmarker"

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        super().__init__()
        self._model_path = Path(model_path) if model_path is not None else None
        self._num_faces = num_faces
        self._det_conf = min_face_detection_confidence
        self._pres_conf = min_face_presence_confidence
        self._track_conf = min_tracking_confidence

        self._landmarker: Any = None
        self._model_key: str = ""
        self._mp: Any = None
        #: detect_for_video 用的毫秒时间戳，必须严格单调递增。
        self._last_ts_ms = 0
        #: 产出给下游的单调秒数。
        self._last_frame_ts = 0.0
        self._mono_start = time.monotonic()
        self._blendshape_checked = False
        #: 上一次检测到的 blendshape 名字（供排查用）。
        self.blendshape_names: tuple[str, ...] = ()
        #: 头姿是否走了 solvePnP 回退（真机排查时很有用）。
        self.used_pnp_fallback = False

    # ------------------------------------------------------------ 生命周期

    def open(self) -> None:
        """加载模型并创建 landmarker（幂等；已创建则直接复用）。"""
        if self._landmarker is not None:
            self._opened = True
            self._closed = False
            return

        if cv2 is None:
            raise FaceBackendError(
                "未安装 opencv，无法做像素预处理。请安装 opencv-contrib-python。"
            )
        _mp, _mp_python, _vision = _load_mediapipe()

        path = self._resolve_model_path()
        key = str(path)
        with _LANDMARKER_LOCK:
            hit = _LANDMARKER_CACHE.get(key)
            if hit is not None:
                hit.refs += 1
                self._landmarker = hit.landmarker
                self._model_key = key
                _log.info("复用已存在的 FaceLandmarker（refs=%d）：%s", hit.refs, path)
            else:
                landmarker = self._create_landmarker(_mp_python, _vision, path)
                _LANDMARKER_CACHE[key] = _CachedLandmarker(landmarker=landmarker, refs=1)
                self._landmarker = landmarker
                self._model_key = key
                _log.info("已创建 FaceLandmarker：%s", path)

        self._mp = _mp
        self._mono_start = time.monotonic()
        self._last_ts_ms = 0
        self._last_frame_ts = 0.0
        self._blendshape_checked = False
        self._opened = True
        self._closed = False

    def _resolve_model_path(self) -> Path:
        """定位模型文件。顺序：显式参数 → 环境变量 → 默认路径。"""
        if self._model_path is not None:
            path = self._model_path
        elif os.environ.get(MODEL_ENV_VAR):
            path = Path(os.environ[MODEL_ENV_VAR])
        else:
            path = DEFAULT_MODEL_PATH

        if not path.exists():
            raise FaceBackendError(
                f"人脸模型文件不存在：{path}\n"
                "本仓库不随附模型文件（体积与授权原因），需要手动下载一次：\n"
                f"  下载地址：{MODEL_DOWNLOAD_HINT}\n"
                f"  放置位置：{DEFAULT_MODEL_PATH}\n"
                f"  或用环境变量指定：{MODEL_ENV_VAR}=D:\\\\path\\\\to\\\\face_landmarker.task\n"
                "无摄像头/离线开发场景不需要它——请改用 SyntheticFaceBackend。"
            )
        return path

    def _create_landmarker(self, mp_python, vision, path: Path):
        """按 Tasks API 创建 landmarker。

        参数含义见模块文档；三个置信度阈值保持 MediaPipe 默认值，
        调整它们等于调整"什么时候认为有人脸"，属于需要真机标定的参数。
        """
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self._num_faces,
            min_face_detection_confidence=self._det_conf,
            min_face_presence_confidence=self._pres_conf,
            min_tracking_confidence=self._track_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """释放引用。引用计数归零时才真正关闭 landmarker。"""
        if self._landmarker is not None and self._model_key:
            with _LANDMARKER_LOCK:
                hit = _LANDMARKER_CACHE.get(self._model_key)
                if hit is not None:
                    hit.refs -= 1
                    if hit.refs <= 0:
                        try:
                            hit.landmarker.close()
                        finally:
                            _LANDMARKER_CACHE.pop(self._model_key, None)
        self._landmarker = None
        self._model_key = ""
        super().close()

    def describe(self) -> str:
        state = "已打开" if self._opened else "未打开"
        model = self._model_path or DEFAULT_MODEL_PATH
        check = "已自检" if self._blendshape_checked else "未自检"
        return (
            f"MediaPipe FaceLandmarker（Tasks/VIDEO，num_faces={self._num_faces}，"
            f"{state}，blendshape {check}）：{model.name}"
        )

    # ------------------------------------------------------------ 主流程

    def process(
        self,
        frame_or_features: np.ndarray | FrameFeatures,
        ts: float | None = None,
        wall_ts: float | None = None,
    ) -> FrameFeatures:
        """把一帧 BGR 图转成 :class:`FrameFeatures`。

        :param ts: 该帧自采集开始起的秒数；不传则用内部单调时钟。
            **必须递增**——MediaPipe 的 VIDEO 模式要求严格单调的时间戳，
            内部会强制保证（同一毫秒内的连续两帧会被推开 1ms）。
        :param wall_ts: 墙钟时间戳（Unix 秒），仅用于日志与报文展示。

        入参已经是 :class:`FrameFeatures` 时原样返回，便于混合管线。
        """
        if isinstance(frame_or_features, FrameFeatures):
            return frame_or_features

        if frame_or_features is None:
            raise FaceBackendError(
                "收到 None 帧。采集源在读不到帧时会返回 None（摄像头掉线、"
                "视频解码失败），调用方必须先判空再送进来——否则这里只会"
                "把它当成一次莫名其妙的崩溃。"
            )
        if not self._opened:
            # 懒打开：调用方少写一行，也不会因为漏了 open() 拿到一堆空特征。
            self.open()

        frame = frame_or_features
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise FaceBackendError(
                f"需要三通道 BGR 图（H×W×3），收到 shape={frame.shape}。"
                "若上游给的是灰度图或 RGBA，请先转成 BGR。"
            )
        frame = np.ascontiguousarray(frame)

        # 灰度图只转一次：照度、模糊、眼镜三处都要用它。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality = self._assess_quality(gray)

        ts_ms = self._next_ts_ms(ts)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        frame_ts = ts if ts is not None else ts_ms / 1000.0
        if frame_ts <= self._last_frame_ts:
            frame_ts = self._last_frame_ts + 1e-3
        self._last_frame_ts = frame_ts
        wall = wall_ts if wall_ts is not None else time.time()

        if not result.face_landmarks:
            return FrameFeatures(ts=frame_ts, wall_ts=wall, has_face=False, quality=quality)

        landmarks = result.face_landmarks[0]
        blendshapes = self._extract_blendshapes(result)
        h, w = frame.shape[:2]

        ear_right = _ear(landmarks, EYE_IDX_RIGHT)
        ear_left = _ear(landmarks, EYE_IDX_LEFT)
        # 与 metrics.eye.fuse_or_raw 用的是同一个融合式：EAR 占 0.6、
        # blendshape 占 0.4（未做个体标定时的临时权重，见 shared.geometry）。
        closure = fuse_closure(
            (ear_left + ear_right) / 2.0,
            (blendshapes.get("eyeBlinkLeft", 0.0) + blendshapes.get("eyeBlinkRight", 0.0)) / 2.0,
        )

        pitch, yaw, roll = self._head_pose(result, landmarks, w, h)
        quality = self._refine_quality(gray, landmarks, quality)

        return FrameFeatures(
            ts=frame_ts,
            wall_ts=wall,
            has_face=True,
            face_bbox=_bbox(landmarks, w, h),
            ear_left=ear_left,
            ear_right=ear_right,
            closure_ratio=closure,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            blendshapes=blendshapes,
            gaze_off_ratio=self._gaze_off(landmarks),
            quality=quality,
            # ⚠️ 478 个归一化地标：**敏感**，仅供本模块内调试，
            #    禁止落盘、禁止打印、禁止出模块（见 privacy.guard）。
            landmarks=tuple((lm.x, lm.y, lm.z) for lm in landmarks),
        )

    # ------------------------------------------------------------ 时间戳

    def _next_ts_ms(self, ts: float | None) -> int:
        """给出下一个严格递增的毫秒时间戳。

        MediaPipe 的 VIDEO 模式对时间戳的要求是**严格**递增：相同即抛异常。
        同一毫秒内连读两帧（回放时很常见）必须被人为推开 1ms。代价是时间戳
        与真实时间有最多几毫秒的漂移，对特征提取没有影响。
        """
        if ts is not None:
            now_ms = int(ts * 1000.0)
        else:
            now_ms = int((time.monotonic() - self._mono_start) * 1000.0)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms
        return now_ms

    # ------------------------------------------------------------ blendshape

    def _extract_blendshapes(self, result) -> dict[str, float]:
        """按**名字**取 blendshape 系数。

        绝不按下标取：MediaPipe 的枚举顺序与 Apple ARKit 文档顺序不一致，
        按索引取会在某次版本升级后静默错位，表现为"表情判定的结果整体
        换了一副面孔"——极难定位。名字对不上时保留原名（不丢数据）并告警。
        """
        raw: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                name = category.category_name or category.display_name
                if name:
                    raw[name] = float(category.score)
        if not raw:
            return {}

        # 名字规范化：大小写不一致（如 eyeBlinkLeft vs eyeblinkleft）时对齐到
        # 期望写法，使下游的 blend("eyeBlinkLeft") 一定能命中。
        canonical = {n.lower(): n for n in EXPECTED_BLENDSHAPES}
        out: dict[str, float] = {}
        for name, score in raw.items():
            std = canonical.get(name.lower())
            if std is None:
                out[name] = score
            else:
                if std != name:
                    self._note(f"blendshape 名字大小写不一致：{name} → 已对齐为 {std}")
                out[std] = score

        self.blendshape_names = tuple(out)
        self._self_check_blendshapes(out)
        return out

    def _self_check_blendshapes(self, available: dict[str, float]) -> None:
        """启动自检：期望的名字缺了任何一个就**大声**告警。

        静默缺失会让表情永远判成 normal（缺失的名字读出来就是 0），
        这是最容易被忽略、又最难定位的一类故障（见 shared.frame_features
        里 EXPECTED_BLENDSHAPES 的注释）。因此只告警一次，但要足够显眼。
        """
        if self._blendshape_checked:
            return
        self._blendshape_checked = True

        missing = missing_blendshapes(available)
        if not missing:
            _log.info("blendshape 自检通过：%d 个期望名字全部存在", len(available))
            return

        message = (
            f"blendshape 自检未通过：缺少 {len(missing)} 个期望的名字——"
            f"{'、'.join(missing)}。\n"
            "后果：表情分类器会把缺失项读成 0，表情判定将整体偏向 normal，"
            "而且**不会报任何错**。\n"
            "常见原因：MediaPipe 版本升级后改了命名，或模型不是 face_landmarker。\n"
            f"实际拿到的名字（{len(available)} 个）：{'、'.join(sorted(available))}"
        )
        self._note(message)
        _log.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------ 头姿

    def _head_pose(self, result, landmarks, w: int, h: int) -> tuple[float, float, float]:
        """``(pitch, yaw, roll)``，单位为度。

        优先用 ``facial_transformation_matrixes``——MediaPipe 专门为头部姿态
        输出的 4×4 齐次矩阵，含公制平移，比 solvePnP 更稳、更省算力。
        :func:`euler_from_matrix` 与 :func:`normalize_angles` 与角度的符号
        约定统一在 :mod:`module_a_vision.metrics.head_pose` 里，A/B 共用。
        """
        matrices = result.facial_transformation_matrixes
        if matrices is not None and len(matrices) > 0:
            flat = np.asarray(matrices[0], dtype=float).reshape(-1).tolist()
            if len(flat) in (9, 16):
                self.used_pnp_fallback = False
                return normalize_angles(*euler_from_matrix(flat))

        self.used_pnp_fallback = True
        self._note(
            "本次未拿到 facial_transformation_matrixes，已回退 solvePnP。"
            "回退路径的符号约定**未经验证**，请核对“向左歪头时 roll 应为正”。"
        )
        pose = _head_pose_from_pnp(landmarks, w, h)
        if pose is None:
            return 0.0, 0.0, 0.0
        return normalize_angles(*pose)

    # ------------------------------------------------------------ 视线

    def _gaze_off(self, landmarks) -> float | None:
        """粗略的视线偏离度 ``[0,1]``；**估不出来时返回 ``None``**。

        做法：虹膜中心相对于两眼内/外角中点的水平偏移 ÷ 眼宽。
        单目、无标定，**只做趋势用**（方案里注意力判定本来也要求
        "偏离持续 6 秒以上"，单帧噪声会被时间维度滤掉）。

        ⚠️ **"估不出来"必须是 ``None``，不能是 ``0.0``。**
        ``0.0`` 在这条链路上是"视线完全对正前方"，也就是**专注度满分**。
        早先这里在两处返回 ``0.0``（468 点模型没有虹膜点、两眼都量不出眼宽），
        于是一条"根本看不见眼睛"的帧会被读成"非常专注"。
        两个方向的后果都很坏，而且都不报错：

        * 对判"失神"而言，这些帧还留在分母里 → 偏离占比被系统性压低 →
          结论偏向 FOCUSED（见 ``metrics/attention.py`` 的 ``snapshot``）；
        * 对 ③b 的 VAI 而言，相当于给"不知道"打满分。

        出站那一层不受影响：``wire.to_v1_sample`` 用的是 :data:`_NO_FACE`
        兜的 8 字段报文，本来就不带视线；视线只走 §3.5 的 ``focus`` 报文，
        那里 ``None`` 是一等公民。
        """
        offs = []
        for eye_idx, iris_idx in (
            (EYE_IDX_RIGHT, IRIS_IDX_RIGHT),
            (EYE_IDX_LEFT, IRIS_IDX_LEFT),
        ):
            if len(landmarks) <= max(iris_idx):
                # 478 点模型才有虹膜；468 点模型上直接放弃估计。
                return None
            p1 = landmarks[eye_idx[0]]
            p4 = landmarks[eye_idx[3]]
            width = abs(p4.x - p1.x)
            if width <= 1e-9:
                continue
            iris_x = sum(landmarks[i].x for i in iris_idx) / len(iris_idx)
            offs.append(abs(iris_x - (p1.x + p4.x) / 2.0) / width)
        if not offs:
            # 两眼都没有可用眼宽 —— 同样是"估不出来"，不是"很专注"。
            return None
        return _clamp01(max(offs) * GAZE_OFFSET_GAIN)

    # ------------------------------------------------------------ 质量

    def _assess_quality(self, gray: np.ndarray) -> FrameQuality:
        """照度 + 模糊。与是否有人脸无关，先算出来。"""
        luma = float(gray.mean())

        illumination = Illumination.NORMAL
        if luma < LUMA_LOW:
            illumination = Illumination.LOW
        elif luma > LUMA_HIGH:
            illumination = Illumination.OVEREXPOSED

        lap = _blur_score(gray)
        blur = Blur.LOW
        if lap < LAPLACIAN_BLURRY:
            blur = Blur.HIGH
        elif lap < LAPLACIAN_SHARP:
            blur = Blur.MEDIUM

        # 没人脸时无法用关键点判遮挡，只能靠画面本身的证据：
        # 又糊又暗 → 判"严重遮挡"（对应 L4 的 vision_unusable），
        # 否则判"没人"（两种成因的处置完全不同，不能混为一谈）。
        occluded = blur is Blur.HIGH or illumination is Illumination.LOW
        return FrameQuality(
            illumination=illumination,
            blur=blur,
            occlusion=Occlusion.SEVERE if occluded else Occlusion.NONE,
            glasses=Glasses.NONE,
            valid=not occluded,
        )

    def _refine_quality(
        self,
        gray: np.ndarray,
        landmarks,
        quality: FrameQuality,
    ) -> FrameQuality:
        """有人脸时：用关键点 presence 判遮挡，再跑一次眼镜启发式。"""
        # ``presence`` 在 Tasks API 的人脸关键点上**存在但恒为 ``None``**
        # （那是姿态模型的字段，人脸模型不填）。必须把"没给"与"给了 0"分开：
        # 原来写成 ``getattr(..., 1.0) or 0.0``，那个默认值 1.0 **永远轮不到**
        # （字段存在，只是值为 None），而 ``None or 0.0`` 又把"不知道"翻译成了
        # "完全不可见"——于是每一帧都判严重遮挡、每个窗口都不可用，
        # 真实摄像头下 A **一个可用窗口都发不出来**（真机实测 81/81 帧 SEVERE）。
        presence = []
        for i in KEY_LANDMARKS_FOR_OCCLUSION:
            if i >= len(landmarks):
                continue
            value = getattr(landmarks[i], "presence", None)
            if value is not None:
                presence.append(float(value))

        if presence:
            avg = sum(presence) / len(presence)
        else:
            # 拿不到输入就**不判遮挡**（fail-open）。判 SEVERE 会让整条链路
            # 失效，判 NONE 只是退回"这件事做不了"——两害相权，取后者。
            avg = 1.0
            self._note(
                "关键点未提供 presence（Tasks API 的已知行为，实测恒为 None），"
                "遮挡判定不生效（fail-open）：无法用关键点判断面部是否被遮挡。"
            )

        occlusion = Occlusion.NONE
        if avg < OCCLUSION_PRESENCE_SEVERE:
            occlusion = Occlusion.SEVERE
        elif avg < OCCLUSION_PRESENCE_PARTIAL:
            occlusion = Occlusion.PARTIAL

        glasses = _glasses_heuristic(gray, landmarks)

        return FrameQuality(
            illumination=quality.illumination,
            blur=quality.blur,
            occlusion=occlusion,
            glasses=glasses,
            valid=(
                quality.illumination is not Illumination.LOW
                and occlusion is not Occlusion.SEVERE
            ),
        )


# ================================================================ 纯函数

def _ear(landmarks, idx: tuple[int, ...]) -> float:
    """Eye Aspect Ratio。睁眼约 0.25–0.35，闭眼约 0.05–0.15。

    用归一化坐标即可——EAR 是比值，与像素尺度无关。
    """
    if len(landmarks) <= max(idx):
        return 0.0
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in idx)
    horizontal = math.hypot(p1.x - p4.x, p1.y - p4.y)
    if horizontal <= 1e-9:
        return 0.0
    vertical = math.hypot(p2.x - p6.x, p2.y - p6.y) + math.hypot(p3.x - p5.x, p3.y - p5.y)
    return vertical / (2.0 * horizontal)


def _bbox(landmarks, w: int, h: int) -> tuple[int, int, int, int]:
    """由关键点包围盒（略作外扩）得到人脸框。

    ⚠️ **敏感字段**：可还原人像位置，禁止出模块、禁止落盘（见 privacy.guard）。
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * 0.05
    pad_y = (y2 - y1) * 0.05
    return (
        int(max(0, (x1 - pad_x) * w)),
        int(max(0, (y1 - pad_y) * h)),
        int(min(w, (x2 + pad_x) * w)),
        int(min(h, (y2 + pad_y) * h)),
    )


def _blur_score(gray: np.ndarray) -> float:
    """模糊度 = 拉普拉斯方差。

    先缩放到固定宽度再算，让阈值不随摄像头分辨率漂移（1080p 与 480p
    的拉普拉斯方差能差好几倍，不归一化就得为每种分辨率各配一套阈值）。
    """
    h, w = gray.shape[:2]
    if w > BLUR_ANALYSIS_WIDTH:
        scale = BLUR_ANALYSIS_WIDTH / float(w)
        gray = cv2.resize(gray, (BLUR_ANALYSIS_WIDTH, max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _head_pose_from_pnp(landmarks, w: int, h: int) -> tuple[float, float, float] | None:
    """``solvePnP`` 回退：仅在变换矩阵缺失时使用。

    ⚠️ 本函数的**符号约定未经验证**（模块文档第 1 条）。相机内参是粗略假设
    （焦距 = 图像宽度、主点在图像中心、零畸变），因为没有做棋盘格标定。
    真机上若发现角度系统性偏大/偏小，先怀疑这里。
    """
    if len(landmarks) <= max(LANDMARK_IDX_PNP):
        return None

    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDX_PNP],
        dtype=np.float64,
    )
    model_points = np.array(MODEL_POINTS_3D, dtype=np.float64)
    focal = float(w)
    camera_matrix = np.array(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return euler_from_matrix(np.asarray(rotation, dtype=float).reshape(-1).tolist())


def _glasses_heuristic(gray: np.ndarray, landmarks) -> Glasses:
    """眼镜启发式。

    ⚠️ **未经验证**，只有两条粗略的亮度证据，没有任何标注样本支撑：

    * **墨镜**：镜片挡光，眼部条带明显比面颊条带暗。
    * **老花镜**：镜片反光，眼部区域出现一批近饱和像素，占比明显高于整幅画面。

    两条都不成立就判"没戴"。误判的代价不对称：把老花镜判成墨镜会白白
    扣掉置信度（见 AggregatorConfig.glasses_penalty），所以阈值取得偏保守。
    """
    if len(landmarks) < 300:
        return Glasses.NONE
    h, w = gray.shape[:2]

    left_corner, right_corner = landmarks[33], landmarks[263]
    eye_y = (left_corner.y + right_corner.y) / 2.0
    x1 = int(max(0, min(left_corner.x, right_corner.x) * w))
    x2 = int(min(w, max(left_corner.x, right_corner.x) * w))
    if x2 - x1 < 4:
        return Glasses.NONE

    eye_h = max(2, int((x2 - x1) * 0.35))
    y1 = int(max(0, eye_y * h - eye_h / 2))
    y2 = int(min(h, eye_y * h + eye_h / 2))
    eye_band = gray[y1:y2, x1:x2]
    if eye_band.size == 0:
        return Glasses.NONE

    # 面颊条带：眼部下方同样宽度的一条，作为"不戴镜片"的参照亮度。
    cy1 = min(h, y2 + eye_h)
    cy2 = min(h, cy1 + eye_h)
    cheek_band = gray[cy1:cy2, x1:x2]

    if cheek_band.size > 0:
        eye_luma = float(eye_band.mean())
        cheek_luma = float(cheek_band.mean())
        if cheek_luma > 1e-6 and eye_luma < cheek_luma * SUNGLASSES_LUMA_RATIO:
            return Glasses.SUNGLASSES

    eye_specular = float((eye_band > 235).mean())
    frame_specular = float((gray > 235).mean())
    if (
        eye_specular > READING_GLASSES_SPECULAR_RATIO
        and eye_specular > 2.0 * max(frame_specular, 1e-6)
    ):
        return Glasses.READING

    return Glasses.NONE


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


__all__ = [
    "DEFAULT_MODEL_PATH",
    "MODEL_ENV_VAR",
    "MediaPipeFaceBackend",
]
