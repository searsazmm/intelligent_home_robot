# -*- coding: utf-8 -*-
"""后端 B 全局配置。

约定：本模块只放"常量"，不放逻辑，方便小组其他成员一眼看到端口和阈值。

接口来源标注说明：
  [api_doc]  = api_doc.md 已规定的字段/端口，禁止私自修改
  [新增]     = 本模块为补齐 chat 通道而新增，已同步进 api_doc.md §5，需 C 端同学对齐

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


def _env_bool(name: str, default: bool) -> bool:
    """环境变量里的布尔值。接受 1/true/yes/on（大小写不限）。"""
    try:
        return os.environ[name].strip().lower() in ("1", "true", "yes", "y", "on")
    except KeyError:
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

# [新增] C↔B：C 需要把用户说的话发给 B，B 把回复发回 C。
# api_doc 原本只定义了 B→C 单向状态，没有"用户对话文本"的入口，故新增此端口。
# 详见 api_doc.md §5 与 backend_B/README.md。
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

# 离线调试用的视觉 CSV（模拟模块 A 导出的数据，api_doc §6.1「开发阶段」）
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

# --------------------------------------------------------------------------
# 9. 主动关怀 proactive（任务要求 3：检测到呆滞/低落时主动关心、日常主动问候）
#
# ⚠️ 这里的所有秒数都是**墙钟秒**，和 §3/§4 那些判定阈值同一量纲
#    （VisionStateEvaluator 的防抖与窗口用的也是墙上时钟）。
#
#    实时运行时墙钟 == 数据时间，两者没有区别；
#    但离线回放 --speed N 时，数据时间跑得比墙钟快 N 倍，
#    门槛必须除以 N 才能在「同一段剧情」上触发。
#    这个换算由 main.py 在装配时做（见 _scale_for_speed），
#    策略本身只认墙钟，保持干净。
# --------------------------------------------------------------------------

# 总开关
PROACTIVE_ENABLED = _env_bool("B_PROACTIVE", True)

# sad / tired 必须**连续持续**这么久才主动关心（墙钟秒）。
# 设成 20 是因为要明显长于判定器的防抖（STATE_MIN_HOLD 1.5s）
# 和滑动窗口（WINDOW_SECONDS 10s）——「刚皱了下眉就凑过来问」很吓人。
PROACTIVE_SUSTAIN = _env_float("B_PROACTIVE_SUSTAIN", 20.0)

# 两次主动开口之间的最小间隔（墙钟秒）。
# 这条是防骚扰的主力：抢话头比不说话更糟。
PROACTIVE_MIN_INTERVAL = _env_float("B_PROACTIVE_MIN_INTERVAL", 90.0)

# 一小时内最多主动开口几次。滑动窗口计数。
PROACTIVE_MAX_PER_HOUR = _env_int("B_PROACTIVE_MAX_PER_HOUR", 4)

# 用户说过话之后的静默期（墙钟秒）。
# **任何**真实用户输入（麦克风 / 8002 / --stdin）之后都闭嘴这么久 ——
# 用户正要往下说的时候被机器人抢话，是最糟的失败模式。
# 顺带保证了用 --stdin 打字自测时机器人不会插嘴。
PROACTIVE_USER_COOLDOWN = _env_float("B_PROACTIVE_USER_COOLDOWN", 60.0)

# 人离开超过这么久再回来，才值得说一句「您回来啦」（墙钟秒）。
# 短于这个时长的消失多半只是扭头/走开拿个东西，不值得开口。
PROACTIVE_GREETING_ABSENT = _env_float("B_PROACTIVE_GREETING_ABSENT", 60.0)

# 启动后的宽限期（墙钟秒）。刚开机就热情打招呼会吓人一跳，
# 而且启动初期视觉状态本来就在 absent→normal 之间抖。
PROACTIVE_STARTUP_GRACE = _env_float("B_PROACTIVE_STARTUP_GRACE", 15.0)

# 回答完之后**追加**关怀语的最小间隔（墙钟秒）。
#
# 和 PROACTIVE_MIN_INTERVAL 分开是刻意的：那个管「机器人主动开口」，
# 这个管「应答里附带一句」。两件事的合理密度不一样 —— 主动开口抢的是
# 话头（越少越好），应答里的追加只是把摄像头看到的事说一声。
#
# 180 秒 ≈ 连续聊天时每三分钟提一次。调小 = 更唠叨。
# 第一次对话必然追加（计数器从 0 起算）。
# 见 core/dialogue.py 的 DialogueEngine.respond()。
CARE_FOLLOWUP_INTERVAL = _env_float("B_CARE_FOLLOWUP_INTERVAL", 180.0)

# 静默时段：这段时间内**不主动开口**。
# ⚠️ 注意只静默"主动"，**绝不静默应答** —— 凌晨两点的「我不舒服」必须回答。
# 这是安全属性，不是体验偏好。
PROACTIVE_QUIET_ENABLED = _env_bool("B_PROACTIVE_QUIET", True)
PROACTIVE_QUIET_START = _env_int("B_PROACTIVE_QUIET_START", 22)   # 22:00
PROACTIVE_QUIET_END = _env_int("B_PROACTIVE_QUIET_END", 7)        # 次日 07:00

# --------------------------------------------------------------------------
# 10. 主动关怀的演示预设（main.py --demo）
#
# 为什么需要它：实测 data/sample_vision.csv 全长只有 119.9 数据秒，
# 其中 tired 段 31.9 秒、sad 段 20.0 秒。而上面的默认门槛是 20 秒 ——
# sad 段刚好卡在边界上，演示时"有时触发有时不触发"，最难看。
# 再叠加 --speed 5（sad 段只剩 4 个墙钟秒），默认门槛必然一次都不触发。
#
# 所以演示时改用这一组更宽松、但**仍然有意义**的阈值。
# 它们只在 --demo 下生效，正常运行时用上面那组。
# --------------------------------------------------------------------------

PROACTIVE_DEMO_SUSTAIN = _env_float("B_PROACTIVE_DEMO_SUSTAIN", 10.0)
PROACTIVE_DEMO_MIN_INTERVAL = _env_float("B_PROACTIVE_DEMO_MIN_INTERVAL", 20.0)
PROACTIVE_DEMO_MAX_PER_HOUR = _env_int("B_PROACTIVE_DEMO_MAX_PER_HOUR", 60)
PROACTIVE_DEMO_GREETING_ABSENT = _env_float("B_PROACTIVE_DEMO_GREETING_ABSENT", 5.0)
PROACTIVE_DEMO_STARTUP_GRACE = _env_float("B_PROACTIVE_DEMO_STARTUP_GRACE", 3.0)

# 追加关怀的间隔同样要压短。默认 180 秒在演示里等于「看不出来」——
# 而这一条正是演示时要展示的东西，比主动关怀更不能被门槛藏起来。
CARE_DEMO_FOLLOWUP_INTERVAL = _env_float("B_CARE_DEMO_FOLLOWUP_INTERVAL", 20.0)

# --------------------------------------------------------------------------
# 10b. 专注静默模式（VAI 专注度指数）
#
# 需求文档 §4 B2：``focus``（专注，**B 内部使用、不下发 C**）——
# 判定老人正专注在某件事上时，**暂不主动开口**。
# 指数本身在 core/focus.py 里算；这里只是把它接到主动关怀的决策上。
#
# ⚠️ 只压**主动**开口（问候 / 关怀）。**绝不压应答** ——
#    「老人先说了话」那条路（handle_chat）一个字都不受这里影响，
#    理由与静默时段完全相同：凌晨两点的「我不舒服」必须回答。
#
# ⚠️ 静默的判据分三段，见 core/focus.py 的 ``should_stay_silent``：
#    ① **可信**（**有指数**、**状态有效**、**数据够新**，三缺一都不行，
#       否则 A 掉线时 B 会拿着上一次的 90 分把关怀无限期关掉）；
#    ② **证据齐全**（视线与睁眼两路都在 —— 缺模态会权重重归一化，
#       只剩头姿也能算出 100 分）；
#    ③ **够专注**（指数 ≥ FOCUS_SILENT_MIN_INDEX）。
# --------------------------------------------------------------------------

# 专注静默的总开关。默认开 —— 关掉的话 core/focus.py 算出来的指数
# 没有任何去处，等于白算。
FOCUS_SILENT_ENABLED = _env_bool("B_FOCUS_SILENT", True)

# 「专注」这条证据要有多新才算数（墙钟秒）。
#
# 为什么不写成「2× 周期」：那个说法在"周期"有单位时才成立，而这里有两个
# 周期（A 的帧周期约 0.1s、B 的发布节拍 0.2s）。按前者取 0.2s，墙钟抖动
# 一次就会让抑制时开时关；按后者取 0.4s 也只是勉强。而指数本身就是秒级
# 平滑的东西，判新鲜度用秒级更贴合它的时间尺度。
#
# 这条是**安全阀**，不是调优旋钮：A 掉线或 focus 报文被吞时，
# FocusTracker 会一直返回上次那个分数（它不报错），
# 只有新鲜度能把它和"真的还在专注"分开。
FOCUS_STALE_SECONDS = _env_float("B_FOCUS_STALE", 2.0)

# 「够专注」的门槛（0-100，VAI 的百分制）。
#
# ⚠️ **这一条不是从参考实现抄来的 —— 参考实现没有任何阈值**，
#    它把自己的指数声明为"研究趋势（非认知专注）"，只出分、不下判断。
#    所以这是本项目自己定的**工程初值**，作用是**把"明显没在看"和
#    "明显正看着"分开**，不做细粒度判读。
#
# 为什么非有不可：只判"指数可信"是不够的 —— 脸在画面里、管线没坏，
# 指数就一直是有效值，扭头看窗外也能出 43 分。那等于"摄像头一通电就
# 再也不主动关心"，与本功能想做的事正好相反。
#
# 70 的依据（权重 0.55 视线 / 0.25 头姿 / 0.20 睁眼）：
#   正对镜头 + 头正 + 睁眼 ≈ 92；视线明显离开正前方（对正 0.1）时
#   光这一项就掉 0.55*0.9 ≈ 50 分，再叠头姿偏移会落到 40 上下。
#   70 落在两者中间，且要求三路证据里至少两路在线。
FOCUS_SILENT_MIN_INDEX = _env_float("B_FOCUS_MIN_INDEX", 70.0)

# 连续专注静默的总时长上限（墙钟秒）。到点放行一次主动关怀，并打日志。
#
# 为什么必须有：新鲜度只管"数据还活着"，管不了"人真的连续专注了 40 分钟"。
# 没有上限的话，一个一直低头看书的长者会得到**无限期**的静默，
# 而日志上看不出任何异常 —— 关掉一个功能而不留痕迹，是最难查的失效。
FOCUS_SUPPRESS_MAX_SECONDS = _env_float("B_FOCUS_SUPPRESS_MAX", 900.0)

# 演示预设：15 分钟等于「看不出来」，压到 30 秒才观察得到。
FOCUS_DEMO_SUPPRESS_MAX_SECONDS = _env_float("B_FOCUS_DEMO_SUPPRESS_MAX", 30.0)

# --------------------------------------------------------------------------
# 10c. VAI 展示报文（B→C，**只供界面显示**）
#
# 需求文档 §12.3 原先写「不下发 C —— 它只用于 B 内部的静默判决，界面上不展示」。
# 前端要展示专注度，这条规格已显式修订：指数经 8002 的 ``vai`` 报文下发 C。
#
# ⚠️ 修订的只是「展示」这一件事，判决仍然只在 B 内部：
#    下发的是**只读旁路**，不进 VisionStateEvaluator、不进 _publish_state、
#    不参与 should_stay_silent，也**不进 8001 状态流**（那条通道仍然只有
#    四个状态字符串，api_doc §4 不变）。
#
# ⚠️ 报文里**刻意不带 state 字段**：带了它迟早会被人接到表情上，
#    变成第二个状态源去和 8001/8002 的 state 打架。
# --------------------------------------------------------------------------

# 是否向 C 发 vai 展示报文。默认开。
VAI_DISPLAY_ENABLED = _env_bool("B_VAI_DISPLAY", True)

# 展示报文的（下限）发送间隔（秒）。
#
# 语义与 8001 的状态推送一致：**变化才发，没变化按这个周期补心跳**。
# 心跳的作用是让 C 能区分「专注度没变」与「B 挂了」—— 少了它，
# 一个静止的 92 分和一条断掉的链路在界面上长得一模一样。
#
# 节拍本身挂在 _state_publish_loop 的 0.2 秒循环上（不新增线程），
# 所以这个值只决定"最多多久必发一条"，不决定轮询频率。
VAI_HEARTBEAT_SECONDS = _env_float("B_VAI_HEARTBEAT", 5.0)

# --------------------------------------------------------------------------
# 11. 语音（听与说）
#
# ⚠️ 语音相关的库都是**可选的**：模块 B 的运行期零第三方依赖是刻意设计，
#    所有语音库只在函数内部惰性 import。缺库时降级到键盘 / 8002 文本通道，
#    绝不阻止 B 启动。安装指引见 requirements-voice.txt。
# --------------------------------------------------------------------------

# 语音总开关（auto 会在有设备时启用）
VOICE_ENABLED = _env_bool("B_VOICE", True)

# 识别引擎：auto / vosk / speechrecognition / dashscope / none
STT_ENGINE = _env_str("B_STT", "auto")
# 合成引擎：auto / doubao / edge / sapi / none
#   auto 按 doubao → edge_tts → sapi 的顺序挑第一个**真的能用**的：
#     有豆包凭证就用豆包（音色最好），没凭证退到 edge-tts（音色次之、要联网），
#     再不行退到 SAPI（机械音，但完全离线）。
#   这条链保证「默认用豆包」和「没配好也不能变哑巴」同时成立。
#   ⚠️ 联网引擎要**试合成一句**才算通过，不只看库装没装 —— 本机实测
#      到 edge-tts 服务器 TCP 连得上但 TLS 被重置，只查"装了没"会选中它，
#      然后每句都在运行期失败，机器人彻底哑掉。见 tts.probe_synthesizer。
TTS_ENGINE = _env_str("B_TTS", "auto")

# 音频设备：**按名字子串匹配，不要填序号**（序号会随虚拟设备增减漂移）。
# 留空 = 用系统默认。
#   本机注意：默认输出是 [4] 扬声器 (Realtek(R) Audio)，不是 ToDesk 虚拟声卡
#   （sd.default.device 为 [1, 4]）。听不见时先查**端点音量**，
#   再考虑显式指定 --audio-out Realtek。
AUDIO_INPUT_NAME = _env_str("B_AUDIO_IN", "")
AUDIO_OUTPUT_NAME = _env_str("B_AUDIO_OUT", "")

# 语速倍数（对齐 backend_A/shared/actions.py 的 Speak.speed）。
# 0.9 = 比正常慢 10%。**刻意不说快**：这是陪聊安慰场景，语速一快就显得
# 敷衍、像在赶时间，长者听感上也吃力。三个引擎都换算成各自的单位：
#     SAPI     → Rate=-1（见 speed_to_sapi_rate）
#     SeedTTS  → speech_rate=-10（见 speech_rate_from_speed）
#     edge-tts → rate="-10%"（见 EdgeTtsSynthesizer）
VOICE_SPEED = _env_float("B_VOICE_SPEED", 0.9)

# 音色：留空 = 自动挑第一个中文音色（按 Culture 前缀匹配，**不硬编码名字**）。
# 豆包引擎下这个值就是 voice_type，见下面的 B_DOUBAO_VOICE。
VOICE_TTS_VOICE = _env_str("B_VOICE_NAME", "")

# ---- 豆包（火山引擎）语音合成：语音合成大模型 2.0（SeedTTS 2.0）----
#
# 走 v3 的 HTTP Chunked 单向流式接口，用**标准库 urllib** 发请求，
# 不引入 requests —— 这样「运行期零第三方依赖」的约束不用为它破例
# （只有 MP3 解码需要 soundfile）。
#
# ⚠️ 缺任何一个，引擎就判为「不可用」，在启动日志里**点名缺的是哪个**，
#    然后 build_synthesizer 自动退到下一个引擎，**不会让机器人变哑巴**。
#
# ⚠️⚠️ **这套凭证模型和以前那套完全不同，别再往 appid 上想。**
#    实测（2026-09-29）：旧的 v1 接口（appid + access_token + cluster）
#    在本项目的凭证下**永远**返回
#        401 "load grant: requested grant not found in SaaS storage"
#    —— 换集群、换音色、换 appid 形态、甚至喂故意的垃圾凭证，
#    服务端回的都是同一句话，说明它压根没走到核对凭证那一步。
#    换成 v3 的「API Key + ResourceId」之后，音频立刻就有了：
#
#        v1（旧）：appid + access_token + cluster，头 Authorization: Bearer;<token>
#        v3（现在）：一个 API Key，头 X-Api-Key；模型版本走 X-Api-Resource-Id
#
#    **v3 里没有 appid 这个东西。** 所以本文件**刻意不读 B_DOUBAO_APPID** ——
#    读了也没用，只会让人以为少配了它才不发声。
TTS_DOUBAO_API_KEY = _env_str("B_DOUBAO_TOKEN", "")

# ResourceId：要调哪个**模型版本**。**这是常量，不是账号里的值。**
#   seed-tts-2.0 = 语音合成大模型 2.0（2.0 音色以 *_uranus_bigtts 结尾）
#   seed-tts-1.0 = 1.0（兼容 BV*_streaming 音色）
# 1.0 和 2.0 的音色**不能混用**：下面默认的 Vivi 2.0 是 2.0 音色，
# 所以这里必须是 seed-tts-2.0。留空会自动回到这个默认值。
TTS_DOUBAO_RESOURCE_ID = _env_str("B_DOUBAO_RESOURCE_ID", "seed-tts-2.0")

# 音色（speaker）。**必须和模型版本匹配、且控制台里已开通**，大小写敏感。
# 默认值 zh_female_vv_uranus_bigtts = Vivi 2.0 陪聊音色，适合长者陪伴场景。
# 想换音色时用 B_DOUBAO_VOICE 覆盖；填错的报错长这样（服务端原话会被带出来）：
#     豆包合成失败：code=45000010 message=invalid speaker
TTS_DOUBAO_VOICE = _env_str("B_DOUBAO_VOICE", "zh_female_vv_uranus_bigtts")

# 半双工：机器人播报期间丢掉麦克风数据。
# 不做回声消除的话，喇叭的声音会被自己听见 → 自激回路 → 自言自语停不下来。
# 戴耳机时可以用 --barge-in 关掉它实现插话。
VOICE_HALF_DUPLEX = _env_bool("B_VOICE_HALF_DUPLEX", True)

# 音频块大小（毫秒）。20ms 是语音处理的常规值：
# 足够细的端点检测粒度，又不会让回调调用得太频繁。
VOICE_BLOCK_MS = _env_int("B_VOICE_BLOCK_MS", 20)

# ---- 预合成缓存 ----
#
# 在线引擎（豆包 / edge-tts）实测合成一句 4.6 秒的话要 3.3 秒，而 SAPI 只要 37ms。
# 这个差距会变成「机器人先沉默三秒再开口」——聊天气氛上很致命。
# 对策：把**固定不变的**回复在启动时后台预合成成 WAV 存下来，
# 播放时直接命中缓存，零合成延迟；带 {topic} 的句子无法预测，不预合成。
VOICE_CACHE_DIR = os.path.join(DATA_DIR, "tts_cache")
VOICE_PREWARM = _env_bool("B_VOICE_PREWARM", True)

# --------------------------------------------------------------------------
# 12. 大模型对话（DeepSeek）
#
# 补的是 core/dialogue.py 的这块短板：它是纯关键词规则，任何不含那约 60 个
# 关键词的话，兜底都掉进 3 句通用套话里 —— 听起来就是"回答生硬、接不住话"。
#
# 规则模板与大模型是**叠加**关系，不是替换：
#     模板   —— 可预测、零依赖、永远有话说，**永远是兜底**
#     大模型 —— 负责接住没预设过的话，挂了只是少一层
#
# ⚠️ 缺凭证 / 断网 / 超时 / 返回体不合法，一律**优雅降级回规则模板**，
#    在启动日志里**点名缺的是哪个变量**，绝不让机器人变哑巴 ——
#    和豆包那套（§11）同一套约定。
#
# ⚠️ 默认开启（auto），但**没配 API Key 就等于没开**：不配也能跑，
#    所有现存测试构造的 DialogueEngine 都不带大模型（见 dialogue 的 llm 参数）。
#
# ⚠️⚠️ 致命信号和身体不适**永远不走大模型**，走确定性文案 ——
#    见 core/dialogue.py 的 llm_eligible 与 text_emotion 的 crisis 标记。
# --------------------------------------------------------------------------

# 总开关。false = 完全走规则模板，一次网络请求都不发。
LLM_ENABLED = _env_bool("B_LLM", True)

# 引擎：auto / deepseek / none
#   auto = 有凭证就用 deepseek，没有就静默降级（**不报错、不阻止启动**）
LLM_ENGINE = _env_str("B_LLM_ENGINE", "auto")

# API Key。控制台：platform.deepseek.com → API keys。
# 变量名是 B_DEEPSEEK_TOKEN（和 B_DOUBAO_TOKEN 一个路数：名字是凭证标识，
# 属性名另起），所以启动日志里点名的是**变量名**，不是这里的属性名。
LLM_DEEPSEEK_API_KEY = _env_str("B_DEEPSEEK_TOKEN", "")

# OpenAI 兼容端点。请求拼成 {BASE_URL}/chat/completions。
# 带不带 /v1 都认（DeepSeek 两种都收），代码会 rstrip('/') 之后再拼。
LLM_DEEPSEEK_BASE_URL = _env_str("B_DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 模型名。deepseek-chat = 通用对话模型。
LLM_DEEPSEEK_MODEL = _env_str("B_DEEPSEEK_MODEL", "deepseek-chat")

# 单次请求超时（秒）。**这里和 SYNTH_TIMEOUT=30 的取舍完全相反**：
# 合成超时在启动路径上，慢一点也比哑巴强；这里是**用户正等着回话**的交互路径，
# 超时越长，老人干等的时间越长。宁可早失败、早落回模板。
# 启动探针也复用这个值 —— 否则一个错的 Key 会让启动卡满 30 秒。
LLM_TIMEOUT = _env_float("B_LLM_TIMEOUT", 4.0)

# 回复长度上限（token）。128 大约够 60~80 个汉字，而模板句都 ≤40 字。
# 这是延迟的主要杠杆：输出 token 数直接决定等待时间。
LLM_MAX_TOKENS = _env_int("B_LLM_MAX_TOKENS", 128)

# 采样温度。DeepSeek 官方按场景给的建议是写代码 0.0 / 通用对话 1.3 / 创作 1.5，
# 不设默认 1.0。陪聊属于"通用对话"，但**这是念给老人听的**，太飘会冒出
# 莫名其妙的句子 —— 从 1.0 起步，按实际听感再调。
LLM_TEMPERATURE = _env_float("B_LLM_TEMPERATURE", 1.0)

# 回复字符上限（汉字数）。超了先试着截到第一个句号，截不出来就落回模板。
# 40 字 ≈ 9 秒语音（语速 0.9），和模板的写作约束一致（见 dialogue.py 的模板库注释）。
LLM_MAX_CHARS = _env_int("B_LLM_MAX_CHARS", 50)

# 喂给大模型的对话窗口（轮）。只取**本次会话**的历史 ——
# data/history.csv 会跨多次演示累积，不过滤的话新会话第一句话就会把
# 上次排练的尾巴（还包括别人的话）喂给模型。
# 见 core/history_store.recent_dialogue 的 session_only。
LLM_CONTEXT_TURNS = _env_int("B_LLM_CONTEXT_TURNS", 4)

# 启动时真调一次模型，验证「有 Key」≠「能用」。
# 代价是一次真实请求（约 1 秒），换来的是"启动日志里就能看出 Key 有没有效"。
# 详见 core/llm.py 的 probe_chatter。离线开发时用 --no-llm-probe 跳过。
LLM_PROBE = _env_bool("B_LLM_PROBE", True)
