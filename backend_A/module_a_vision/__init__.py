"""模块 A · 视觉感知——**全系统唯一接触原始画面的模块**。

这个唯一性不是架构上的巧合，而是隐私设计的支点：因为只有 A 碰得到
像素，所以"图像不上云"这条红线只需要在一处守住，也只需要在一处审计。
下游的 B / C / D / E 拿到的永远是结构化标量。

数据流::

    采集(capture) → 人脸(face) → 指标(metrics) → 聚合(aggregate) → server → B
                                        ↑
                                   privacy/guard 在出站前做最后一道断言

两种使用方式：

1. **端到端**：``main.py`` 起一个进程，采集→聚合→TCP 8000 推给 B。
2. **只用聚合**：``WindowAggregator`` 可以脱离摄像头与网络单独使用——
   喂 ``FrameFeatures`` 进去，吐 ``WindowState`` 出来。测试走的是这条路，
   因此 B 的判定逻辑可以在没有摄像头的机器上完整验证。

⚠️ **本机未验证的部分**：``face/mediapipe_backend.py` 的真实推理路径。
本机无摄像头，MediaPipe 只对着桩对象跑过。合成源与 CSV 回放是已实测
的主路径。详见该模块的 docstring。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
