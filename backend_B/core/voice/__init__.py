# -*- coding: utf-8 -*-
"""语音层：听（VAD + STT）与说（TTS）。

设计约束（**必须保持**）：模块 B 的运行期零第三方依赖是刻意设计。
本包内所有第三方 import（sounddevice / vosk / edge_tts …）一律写在函数体内部
惰性执行，探测可用性用 ``importlib.util.find_spec``（不执行模块，
避免只为探测就初始化 PortAudio）。缺库时记一行清晰指引并降级，绝不让 B 启动失败。

见 backend_B/requirements-voice.txt。
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []
