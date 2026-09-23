# -*- coding: utf-8 -*-
"""后端 B 全局配置。

约定：本模块只放"常量"，不放逻辑，方便小组其他成员一眼看到端口和阈值。

接口来源标注说明：
  [api_doc]  = api_doc.md 已规定的字段/端口，禁止私自修改
  [新增]     = 本模块为补齐 chat 通道而新增，已同步进 api_doc.md §7，需 C 端同学对齐

所有配置项都支持环境变量覆盖，方便联调时不改代码换端口，例如：
    set B_VISION_PORT=9000 && python main.py
"""

import os


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --------------------------------------------------------------------------
# 1. 网络地址 —— 三模块统一 127.0.0.1
# --------------------------------------------------------------------------

# [api_doc §3.1] A→B：服务端是模块 A，客户端是模块 B
#   注意方向！B 是"主动连接"的一方，不是监听的一方。
VISION_HOST = _env_str("B_VISION_HOST", "127.0.0.1")
VISION_PORT = _env_int("B_VISION_PORT", 8000)

# [api_doc §4.1] B→C：服务端是模块 B，只推 4 种状态字符串
STATUS_HOST = _env_str("B_STATUS_HOST", "127.0.0.1")
STATUS_PORT = _env_int("B_STATUS_PORT", 8001)

# [新增] C→B：C 需要把用户说的话发给 B，B 把回复发回 C。
# api_doc 原本只定义了 B→C 单向状态，没有"用户对话文本"的入口，故新增此端口。
# 详见 api_doc.md §7 与 backend_B/README.md。
CHAT_HOST = _env_str("B_CHAT_HOST", "127.0.0.1")
CHAT_PORT = _env_int("B_CHAT_PORT", 8002)

# socket 读写超时（秒）。设为 None 会永久阻塞，不利于 Ctrl+C 退出，故给一个值。
SOCKET_TIMEOUT = _env_float("B_SOCKET_TIMEOUT", 1.0)

# 监听队列长度
LISTEN_BACKLOG = 8

# --------------------------------------------------------------------------
# 2. A→B 断线重连（本模块重点要求的容错逻辑）
# --------------------------------------------------------------------------

# 第一次重连前等待秒数，之后指数退避：1 → 2 → 4 → 8 → 10 → 10 ...
RECONNECT_BACKOFF_INITIAL = _env_float("B_RECONNECT_INITIAL", 1.0)
# 退避上限，避免无限增长
RECONNECT_BACKOFF_MAX = _env_float("B_RECONNECT_MAX", 10.0)
# 退避乘数
RECONNECT_BACKOFF_FACTOR = _env_float("B_RECONNECT_FACTOR", 2.0)
# 单次 connect 的连接超时（秒）
CONNECT_TIMEOUT = _env_float("B_CONNECT_TIMEOUT", 3.0)

# 多久收不到 A 的包就认为视觉数据过期/失联，状态降级为 absent（秒）
VISION_STALE_SECONDS = _env_float("B_VISION_STALE", 5.0)

# --------------------------------------------------------------------------
# 3. 视觉状态判定阈值（模块 A 只给原始特征，最终判定由 B 负责 —— api_doc §3.3.2）
# --------------------------------------------------------------------------

# 眼睑开合度 EAR：睁眼一般 0.25~0.35，闭眼 < 0.20，因人而异，联调时可调
EAR_TIRED_THRESHOLD = _env_float("B_EAR_TIRED", 0.20)
# EAR 连续低于阈值多久判定为疲劳（秒）
EAR_TIRED_DURATION = _env_float("B_EAR_TIRED_SEC", 3.0)
# 眨眼频率（次/分钟）高于该值也判为疲劳（正常 15~20 次/分）
BLINK_RATE_TIRED = _env_float("B_BLINK_RATE_TIRED", 30.0)
# 估算眨眼频率专用的观察窗口。必须比 WINDOW_SECONDS 长很多 ——
# 眨眼次数是稀疏的泊松过程，10 秒窗口里"多眨两次"就能把频率算翻倍，
# 实测会把正常人误判成疲劳。用 60 秒才能把噪声压到可用范围。
BLINK_WINDOW_SECONDS = _env_float("B_BLINK_WINDOW_SECONDS", 60.0)

# 低头角度阈值（度），超过视为低头犯困
PITCH_DOWN_THRESHOLD = _env_float("B_PITCH_DOWN", 20.0)
# 头部左右偏转过大视为走神（度）
YAW_DISTRACT_THRESHOLD = _env_float("B_YAW_DISTRACT", 45.0)

# 人脸丢失多久判定为 absent（秒）。短暂丢帧不立刻切状态。
FACE_LOST_GRACE = _env_float("B_FACE_LOST_GRACE", 2.0)

# --------------------------------------------------------------------------
# 4. 状态防抖参数（避免状态在 normal/tired 之间来回跳，导致前端闪烁）
# --------------------------------------------------------------------------

# 一个新状态至少连续稳定这么久才真正生效（秒）
STATE_MIN_HOLD = _env_float("B_STATE_MIN_HOLD", 1.5)
# 从 absent 切回其他状态需要更长的恢复时间，防止假装有人（迟滞）
ABSENT_RECOVER_HOLD = _env_float("B_ABSENT_RECOVER", 2.5)
# 判定窗口：只保留最近 N 秒的视觉样本参与计算
WINDOW_SECONDS = _env_float("B_WINDOW_SECONDS", 10.0)

# --------------------------------------------------------------------------
# 5. 四种合法状态字符串（api_doc §4.2 只允许这 4 个，禁止自造）
# --------------------------------------------------------------------------

STATE_NORMAL = "normal"
STATE_SAD = "sad"
STATE_TIRED = "tired"
STATE_ABSENT = "absent"

VALID_STATES = (STATE_NORMAL, STATE_SAD, STATE_TIRED, STATE_ABSENT)

# --------------------------------------------------------------------------
# 6. 路径与持久化
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 离线调试用的视觉 CSV（模拟模块 A 导出的数据，api_doc §5.1）
SAMPLE_VISION_CSV = os.path.join(DATA_DIR, "sample_vision.csv")
# 历史对话记录 CSV（任务要求 4：预留读取 CSV 历史记录的接口）
HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")

# 历史对话文件表头，禁止随意改动（改了旧文件读不出来）
HISTORY_FIELDS = [
    "timestamp",      # float，写入时间（epoch 秒）
    "session_id",     # str，会话标识，一次程序运行一个
    "role",           # str，user / robot
    "text",           # str，说话内容
    "vision_state",   # str，该轮对话时的视觉状态
    "text_emotion",   # str，文本情绪标签
    "emotion_score",  # float，文本情绪得分，负=消极 正=积极
    "intent",         # str，识别到的意图
]

# --------------------------------------------------------------------------
# 7. 对话管理参数
# --------------------------------------------------------------------------

# 生成回复时最多参考多少轮历史（用于避免重复和上下文衔接）
DIALOGUE_CONTEXT_TURNS = _env_int("B_DIALOGUE_CONTEXT", 5)
# 从 CSV 冷启动时预加载多少条历史
HISTORY_PRELOAD_ROWS = _env_int("B_HISTORY_PRELOAD", 50)

# 状态推送心跳间隔（秒）：即使状态没变也定期重发，防止 C 端判定掉线
STATUS_HEARTBEAT_SECONDS = _env_float("B_STATUS_HEARTBEAT", 15.0)

# --------------------------------------------------------------------------
# 8. 日志
# --------------------------------------------------------------------------

LOG_LEVEL = _env_str("B_LOG_LEVEL", "INFO")
