# 模块 A —— 视觉感知

采集 → 人脸关键点 → 五项指标 → 上报。**图像不出本模块**，出站只有结构化统计量。

从 `xiangmu1/home_robot` 移植而来，并对齐团队 `api_doc.md` §3.2 的报文格式。

---

## 1. 快速开始

```bash
cd backend_A

# 无摄像头也能跑通（推荐第一次用这个）：合成源 + v1 报文，起 8000 服务
python main.py

# 不起服务，只打印报文 —— 顺带逐帧跑一遍数据契约检查
python main.py --dry-run --scenario sad --max-frames 300

# 换剧本
python main.py --scenario drowsy
python main.py --scenario no_face

# 看有哪些剧本
python main.py --list-scenarios
```

跑测试（40 个用例，标准库 unittest，**不需要装任何东西**）：

```bash
python -m unittest discover -s tests -v
```

> `tests/test_interop_with_b.py` 会起**真的** A 服务与**真的** B 客户端，
> 约 13 秒。`backend_B` 不在旁边时其中一半会自动跳过。

---

## 2. 端口与连接方向

| 链路 | 端口 | A 的角色 | 数据格式 |
| ---- | ---- | -------- | -------- |
| A → B | 127.0.0.1:8000 | **服务端**（listen） | 一行一条 JSON，`\n` 分隔 |

⚠️ **方向别写反**：A 是 `listen` 的一方，B 主动 `connect`。

```bash
python main.py --port 9000        # 换端口
python main.py --host 0.0.0.0     # 换监听地址（本机回环以外，需自行评估风险）
```

---

## 3. 目录结构

```
backend_A/
├── main.py                 入口垫片（真正的内容在 module_a_vision/main.py）
├── requirements.txt
├── shared/                 枚举 · schema · 分帧收发 · FrameFeatures
├── module_a_vision/
│   ├── capture/            camera · video · csv · synthetic（四源同接口）
│   ├── face/               MediaPipeFaceBackend（Tasks API）· SyntheticFaceBackend
│   ├── metrics/            head_pose · eye · expression · attention · fatigue
│   ├── aggregate/          10 秒窗口 + N-of-M 投票 + 跨窗口累计
│   ├── privacy/guard.py    出站断言：报文只许含聚合量
│   ├── wire.py             ★ 新增：api_doc §3.2 的逐帧投影与契约闸
│   └── server.py           8000 服务
└── tests/
    ├── test_v1_contract.py      27 个：投影、映射、契约闸、眨眼计数器
    └── test_interop_with_b.py   13 个：真 socket、真 A、真 B
```

---

## 4. 出站格式：`--emit v1`（默认）与 `v2`

A 现在能吐**两种**报文，用 `--emit` 选。

### 4.1 `v1` —— api_doc §3.2，逐帧平铺 8 字段

```json
{"timestamp": 12.3, "has_face": true, "ear": 0.29, "blink_cnt": 7,
 "pitch": 2.0, "yaw": -3.0, "roll": 1.5, "emo_feature": "normal"}
```

**这是团队仓库的默认，也是 `backend_B` 认的格式。**

### 4.2 `v2` —— 原生的 10 秒嵌套窗口

`WindowState`（13 个顶层键 + `observations` 五个子对象），另含
`heartbeat` 与 `vision_unusable` 两条附加报文。`xiangmu1` 侧的消费方用这个。

```bash
python main.py --emit v2
```

### 4.3 为什么 `v1` 模式下**不发**心跳

`backend_B` 的 `VisionSample.from_payload` 会把任何一条缺字段的报文
兜成一个 `has_face=false` 的样本**推进判定器**。一条 `heartbeat` 会被
读成"这一刻看不见老人"，周期性污染 B 的状态输入。

B 不需要心跳：A 死掉由它那个 5 秒失联判据兜住。这是"一种模式一种报文"的
必然结论，不是省事。

### 4.4 为什么是**逐帧**而不是逐窗口（一条真实踩过的坑）

api_doc §3.2 是逐帧流式协议。设计过程中曾经打算"在 A 关窗时（每 10 秒）
发一条 §3.2 报文"，**那是错的**，而且错得很隐蔽：

`backend_B/config.py` 的 `VISION_STALE_SECONDS` 默认 **5 秒**。
每 10 秒发一条的话，B 每 10 秒里有 **5 秒**判 `absent`，
状态周期性抖动、前端卡片跟着闪。

而这个错误**在单元测试里看不出来** —— 两个模块各自的逻辑都是对的，
错的是两者时间尺度的关系。所以 `tests/test_interop_with_b.py` 里有一条
`test_frame_rate_beats_the_stale_threshold` 专门钉住它：断言帧间隔
远小于 5 秒。把逐帧改回逐窗口，那条测试会立刻变红。

> 注意区分：B 的 `WINDOW_SECONDS = 10.0` 是 **B 在样本流上的滑动窗口**，
> 和 A 的 10 秒聚合窗口是两回事。这里踩过一次混为一谈的坑。

### 4.5 字段映射

8 个字段**全部来自同一帧**，没有一个需要编造：

| §3.2 字段 | 来源 |
| --- | --- |
| `timestamp` | `frame.ts`（采集起点的单调秒，**不是** Unix 时间） |
| `has_face` | `frame.has_face and frame.quality.valid` |
| `ear` | `(frame.ear_left + frame.ear_right) / 2` |
| `blink_cnt` | `EyeTracker.blink_total`，**进程内累计、只增不减** |
| `pitch` / `yaw` / `roll` | `frame.pitch_deg` / `yaw_deg` / `roll_deg` |
| `emo_feature` | 逐帧分类 → `normal` / `low` / `tired` |

两处刻意的判断，都写进了代码注释和测试：

- **画面不可用的帧压成 `has_face=false`。** §3.2 没有 quality 字段，
  无法表达"看见设备但看不清人"。若照字面报 `true` 而把 `ear` 填 0，
  B 会把它读成"EAR 持续低于阈值"，几秒后报出 `tired` ——
  一个纯粹由缺字段制造出来的**假疲劳**。
- **无人脸时 `blink_cnt` 不清零。** "这一帧没人脸"不等于"这个人没眨过眼"。
  它是事件计数器，不是测量量。

### 4.6 `emo_feature=tired` 代码可达、**数据不可达**

表情分类器对疲惫有刻意的阻尼（`TIRED_DAMPING = 0.6`），
所以表情标签在结构上**不会**变成 `tired`。B 判疲惫只能靠
`ear` / `blink_cnt` / `pitch` 三条路。

`EMOTION_TO_FEATURE` 里 `Emotion.TIRED → "tired"` 这一支有测试覆盖，
但不要因为它是绿的，就以为实跑中会出现 `emo_feature=tired`。

---

## 5. 命令行参数

```
python main.py [选项]

  --source {synthetic,camera,video,csv}   采集源（默认 synthetic）
  --scenario NAME        合成剧本名（--source synthetic）
  --video PATH           视频文件（--source video）
  --csv PATH             CSV 文件（--source csv）
  --camera-index N       摄像头序号

  --emit {v1,v2}         出站格式（默认 v1，见 §4）
  --host / --port        监听地址与端口（默认 127.0.0.1:8000）
  --fps FLOAT            目标帧率（默认 10）
  --elder-id / --device-id

  --dry-run              不起服务，只打印报文并跑契约检查
  --every N              仅 --dry-run：v1 下每 N 帧打印一行（默认 10）
  --max-frames N         读到 N 帧后停止
  --speed FLOAT          回放倍速
  --real-time / --no-real-time
  --loop                 剧本循环播放
  --list-scenarios       列出剧本后退出
```

---

## 6. 联调步骤

按 api_doc §6.2 的顺序：**先 A → 再 B**。顺序反了也不会坏，B 会指数退避重连。

```bash
# 终端1：模块 A（默认 v1，正是 B 要的格式）
cd backend_A && python main.py --scenario sad

# 终端2：后端 B
cd backend_B && python main.py --log-level DEBUG
```

三个非 `normal` 状态都有现成的合成剧本，**不需要摄像头、不需要造数据**：

| 剧本 | 时长 | 预期 B 的状态 |
| ---- | ---- | ------------- |
| `sad` | 205s（25s normal → 180s sad） | `normal` → **`sad`** |
| `drowsy` | 185s（25s normal → 120s drowsy → 40s normal） | `normal` → **`tired`** |
| `no_face` | 325s（25s normal → 300s no_face） | `normal` → **`absent`** |

⚠️ **联调不要加 `--speed`。** B 的防抖用墙上时钟算 `STATE_MIN_HOLD = 1.5s`，
快放会把时间尺度压掉、防抖永远不成立、B 会一直停在 `absent`。
这是 B 的既定行为，不是本次移植的问题 —— 但极容易被误判成移植失败。

**先分辨是 A 算错了还是 B 判错了**：`python main.py --dry-run` 看 A 吐的
报文本身对不对，再去看 B。绝大多数"系统好像看不见老人"最后都是字段没对上。

---

## 7. 验证状态（请如实阅读）

本节比其它任何一节都重要。**不要把没验证过的东西当成验证过的。**

### 已验证

| 内容 | 怎么验的 |
| --- | --- |
| 报文严格符合 §3.2 的 8 字段 | 27 个单元测试 + 对端测试逐条断言键集合**精确相等** |
| 真 A → 真 B 的链路与字段对齐 | `test_interop_with_b.py`：真 socket、真 A 服务线程、真 `VisionClient` + 真 `VisionStateEvaluator` |
| **B 能判成 `normal`**（= 字段名真的对上了） | 同上。字段错位时 B 会静默停在 `absent`，这是唯一能证明对齐的观测 |
| **无失联抖动** | 自动化断言：稳定运行期内 `state.stale` 一次都不为真；另有帧间隔 < 2.5s 的断言 |
| `blink_cnt` 只增不减 | 单元测试（跨 400 秒、远超 60 秒滚动窗）+ 对端测试（整条流单调） |
| 三个非 `normal` 状态的端到端轨迹 | 真时钟实跑，见下表 |
| `v2` 模式无回归 | `--dry-run --emit v2 --scenario drowsy` 输出与改动前逐字相同 |
| 契约闸拦得住脏报文 | 逐类断言：缺字段 / 多字段 / `blink_cnt=True` / NaN / 字符串数字 / 非法 `emo_feature` |

真时钟实跑记录（A 与 B 同时起，B 连上后观察 45 秒，`--log-level DEBUG`）：

| 剧本 | B 的状态轨迹 | 缺字段警告 | 状态抖动 |
| --- | --- | --- | --- |
| `sad` | `absent` → `normal`(3s) → **`sad`**(27s) | 0 | 无 |
| `drowsy` | `absent` → `normal`(2s) → **`tired`**(27s) | 0 | 无 |
| `no_face` | `absent` → `normal`(3s) → **`absent`**(26s) | 0 | 无 |

"缺字段警告 0"这一列值得单独说：B 的 `_warn_missing_fields` 会在头几条报文上
检查 8 个字段是否齐全。三次运行都是零 —— 比任何断言都更直接地说明契约对上了。

### 未验证 —— 请勿当作已完成

| 内容 | 为什么没验证 | 怎么补 |
| --- | --- | --- |
| **「像素 → 人脸关键点」这一步** | 开发机**没有摄像头**。合成源与 CSV 回放直接产出特征，**跳过了像素** | 接真实摄像头或真人视频：`python main.py --source camera` |
| MediaPipe 后端在真实画面上的表现 | 同上 | 同上 |
| **真实场景下的判定准确率** | 同上。验的是**判定逻辑**，不是**视觉精度** | 需要真人数据，并做误报/漏报统计 |
| **表情识别的精度** | 是 52 个 ARKit blendshape 的**规则启发式基线**，不是训练模型 | 采集真实老人表情数据后接 ONNX 模型，`ExpressionClassifier` 接口已留好 |
| 长时间运行稳定性 | 没有跑过数天量级的浸泡测试 | — |
| `--source video` | 没有可用的测试视频 | 自备一段 |

---

## 8. 已知问题

### ⚠️ 顶层 `requirements.txt` 的 `cv2` 冲突 —— 需要全员确认后才能改

仓库根目录的 `requirements.txt` 里写着：

```
opencv-python==4.13.0.90     ← 与下面冲突
mediapipe                    ← 未锁版本
pyaudio                      ← 未锁版本
```

`mediapipe` 依赖 `opencv-contrib-python`。这两个包**都提供 `cv2`**，
同时安装会让 `cv2` 的导入行为取决于安装顺序 —— 产生"昨天还好好的"
这一类极难排查的故障。`xiangmu1/home_robot/requirements.txt` 开头专门
警告过这一点。

**这是一个共享文件，已按流程挂起，没有单方面改动。** 本模块自己的
`backend_A/requirements.txt` 用的是 `opencv-contrib-python`。

建议的改法（**待确认，尚未执行**）：

1. `opencv-python==4.13.0.90` → `opencv-contrib-python>=4.10`
2. 删掉 `pyaudio`。目前全仓没有任何代码 import 它（团队 C 还没写）。
   顶层那条注释说"Windows 上 pip 直装常编译失败"，是对的 ——
   等真有消费方时，建议改用 `sounddevice`（wheel 自带 PortAudio）。
3. 锁住 `mediapipe` 版本。顶层注释担心的 `mediapipe.solutions.face_mesh`
   （旧 Solutions API）**与本模块无关**：A 用的是 **Tasks API**
   （`mediapipe.tasks.python`）。但那不等于可以不锁版本。

### A 在团队仓库与 xiangmu1 会各自演进（分叉）

见 §10。

---

## 9. 降级矩阵

| 缺什么 | 会怎样 | 怎么看出来 |
| --- | --- | --- |
| **摄像头** | 用合成源或 CSV 回放 | `--source synthetic` / `--source csv` |
| **mediapipe / opencv** | 合成源与 CSV 源照常工作（它们不经过像素） | `--source camera` 才会报错，且退出码是 2 而不是抛栈回溯 |
| **B 没启动** | A 照常采集，只是没人收；B 起来后自动接上 | 日志 `[A] B 已连接` |
| **B 断开** | A 继续跑，日志提示等待重连 | `[A] 所有 B 连接均已断开，等待重连` |
| **采集源耗尽** | 停止产出但进程不退出（"看不见了"本身是要上报的状态） | `[A] 采集源已耗尽` |

---

## 10. 与 `xiangmu1/home_robot` 的关系

本目录是一份**移植副本**，不是符号链接、不是子模块。

- **同步来源**：`xiangmu1/home_robot/` 的 `shared/` 与 `module_a_vision/`
- **上游指纹**（同步时对全部 `.py` 求 sha256 再汇总）：
  `d296fab3e2888354346cf899f3b132ac19a1e23386140b65876c74df343c3896`
- **本地相对上游只动了 5 个文件**：

  | 文件 | 改动 |
  | --- | --- |
  | `metrics/eye.py` | 加只增不减的 `blink_total` 累计计数器 |
  | `wire.py` | **新增**：§3.2 投影层与契约闸 |
  | `aggregate/window.py` | 加 `blink_total` 属性与 `classify_frame()` |
  | `server.py` | `emit_mode`、逐帧投影、契约闸、v1 下关心跳、真实端口入横幅 |
  | `main.py` | `--emit` / `--every`、v1 的 `--dry-run`、`ImportError` 兜底、行缓冲 |

  **其余 30 个文件与上游逐字节相同。**

改上游那 5 个文件里的任何一个时，请**两边一起改**，改完回来更新这张表 ——
不然两个仓库会静默地长成两个东西，而症状是"xiangmu1 那边好好的，
团队这边状态一直是 absent"。

`--emit v2` 保留上游的原生行为，所以消费 V2 报文的代码不会因为这次移植而失效。
