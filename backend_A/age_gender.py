"""
模块 A：性别/年龄估计（需求文档 §12.5 #1/#2，P2 演示项）
==========================================================
模型：62x62 人脸性别年龄 ONNX（retail-0013 系，本仓库 models/age_gender.onnx 为
bluefoxcreation/gender_age 转换版，NHWC 布局；NCHW 布局的 retail-0013 原版同样兼容）
输出：gender（female/male + 置信度）、age（岁，"预测"口径，MAE ±5-7 年，禁止当真实信息用）

降级语义（对齐 A9 自检）：模型文件缺失或 onnxruntime 未安装时功能自动禁用，
只打印一行说明，不影响主流程任何指标。输入布局（NCHW/NHWC）与输出顺序
由模型签名自适应，换模型不需要改代码。
"""

from pathlib import Path

import cv2
import numpy as np


class AgeGender:
    def __init__(self, model_path: Path, enabled: bool = True):
        self.ok = False
        self.gender = None      # 'female' / 'male'
        self.conf = None        # 置信度 0-1
        self.age = None         # 预测年龄（岁）
        self._sess = None
        self._in = None
        self._nhwc = False      # True=输入 (1,H,W,3)，False=输入 (1,3,H,W)
        self._size = (62, 62)
        if not enabled:
            print("[A] 性别/年龄估计：按配置禁用（age_gender_enabled=false）")
            return
        if not Path(model_path).exists():
            print(f"[A] 性别/年龄估计禁用：未找到 {Path(model_path).name}（一次性下载见 README）")
            return
        try:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            inp = self._sess.get_inputs()[0]
            self._in = inp.name
            shp = [d if isinstance(d, int) else 62 for d in inp.shape]
            if len(shp) == 4 and shp[1] == 3:          # NCHW (1,3,H,W)
                self._size = (shp[3], shp[2])
                self._nhwc = False
            elif len(shp) == 4 and shp[3] == 3:        # NHWC (1,H,W,3)
                self._size = (shp[2], shp[1])
                self._nhwc = True
            self.ok = True
            print(f"[A] 性别/年龄模型已加载（onnxruntime CPU，输入 {self._size[0]}x{self._size[1]} "
                  f"{'NHWC' if self._nhwc else 'NCHW'}）")
        except Exception as e:
            print(f"[A] ⚠️ 性别/年龄模型加载失败：{e}")

    def infer(self, frame_bgr, bbox) -> None:
        """bbox=(x1,y1,x2,y2) 人脸像素框；结果存 self.gender/self.age，不返回值。"""
        if not self.ok:
            return
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (max(0, int(bbox[0])), max(0, int(bbox[1])),
                          min(w, int(bbox[2])), min(h, int(bbox[3])))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return
        face = cv2.resize(frame_bgr[y1:y2, x1:x2], self._size)
        blob = face[None].astype(np.float32) if self._nhwc else \
            face.transpose(2, 0, 1)[None].astype(np.float32)
        try:
            outs = [np.asarray(o).reshape(-1) for o in self._sess.run(None, {self._in: blob})]
        except Exception:
            return
        # 按展平后长度识别输出：长度 2=性别概率，长度 1=年龄。
        # 通道序经 lena.jpg 实测校准：本转换版 [male, female]（与 retail-0013 原版相反）
        gender_v = next((o for o in outs if o.size == 2), None)
        age_v = next((o for o in outs if o.size == 1), None)
        if gender_v is None or age_v is None:
            return
        female_p = float(gender_v[1])
        self.gender = "female" if female_p >= 0.5 else "male"
        self.conf = round(abs(female_p - 0.5) * 2.0, 2)
        raw = float(age_v[0])
        self.age = int(round(raw * 100.0)) if raw < 2.0 else int(round(raw))   # ×100 编码保护

    def label(self) -> str:
        """预览窗覆盖行文本；未出结果返回 '--'。"""
        if self.gender is None:
            return "--"
        tag = "M" if self.gender == "male" else "F"
        age = f"{self.age}y" if self.age is not None else "?"
        return f"{tag}/{age}({self.conf:.2f})"
