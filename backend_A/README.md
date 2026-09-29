# 模块 A —— 视觉感知（本目录并存两套实现）

采集 → 人脸关键点 → 指标 → 上报。**图像不出本模块**，出站只有结构化统计量。

本目录下并存**两套**模块 A。接口一致（都是 `api_doc.md` §3.2 的 8 字段报文、
都监听 `127.0.0.1:8000`），但实现、能力、依赖各不相同。此前两套各自独立演进、
互相看不见对方，2026-09-29 合并到同一分支，**刻意保留并存、不合成一套**。

> ⚠️ **同一时刻只能起一套** —— 都要占 8000 端口。

|  | A-包 | A-单文件 |
|---|---|---|
| 入口 | `python main.py` | `python vision_a.py` |
| 代码 | `module_a_vision/` 包（38 文件） | `vision_a.py` 及同目录扁平 9 文件 |
| 归属 | 本分支（移植自 `xiangmu1/home_robot`） | 队友 GYZ（原 `origin/main`） |
| 人脸 | MediaPipe **Tasks** `FaceLandmarker` | MediaPipe **FaceMesh**（solutions） |
| 独有能力 | 五项指标 + 10 秒聚合 + 隐私闸 + v1 契约闸 + **40 个测试** + 确定性**合成源**（无摄像头也能演示） | **CHROM rPPG**（心率/呼吸率）+ **ONNX 性别年龄** + 多人脸锁主脸 + 照度自检 + 挂机态 A8/A9/A11 |
| 依赖 | numpy / opencv-contrib / mediapipe（**Tasks 构建**） | mediapipe（**需带 `solutions` 的旧构建**，见下方警告）+ 可选 `onnxruntime` |
| `emo_feature` 取值 | `normal` / `low` / `tired` | `normal` / `tired` / `sad` / `blank` |

## 两套的 `emo_feature` 枚举不一样 —— 已确认按下面处理

`low` 与 `sad` 同义（低落），`blank` 是「双眼睁开但视线长时间无位移」（发呆/失神）。

**B 侧同时接受两套**（见 [backend_B/core/vision_state.py](../backend_B/core/vision_state.py)
的 `_evaluate_window`）：`low` 和 `sad` 都判 `sad`，`blank` 判 `absent`。
所以两套 A 接上去都能被正确理解，不需要谁改报文。

`api_doc.md` §3.2 把 `normal / tired / sad / blank` 记为正式枚举，
`low` 标为兼容别名（现存 A-包仍发 `low`）。

> ⚠️ 一处已知副作用：`blank` 判成 `absent` 后，如果发呆持续够久再「回来」，
> 会触发一次主动问候。默认阈值 60 秒，而 `blank` 只需 3 秒静止 → 默认配置下
> 不会误触发；但 **`--demo` 模式把它降到 5 秒，演示时会误触发**。

## ⚠️ 跑 A-单文件要单独一个环境（两件事：mediapipe 版本 + **中文路径**）

两套 A 代码可以并存、`git` 上没有冲突，但 `vision_a.py` 起不来有**两个独立原因**，
必须同时解决。先看两套要的东西不一样：

### 一、版本：0.10.35 是 Tasks-only，没有 `solutions`

| | 用的 API | 需要什么 |
|---|---|---|
| A-包 | `mediapipe.tasks.python`（Tasks API） | Tasks 构建即可。本仓库锁的 **0.10.35 是 Tasks-only**，`mediapipe.solutions` **不存在** |
| A-单文件 | `mediapipe.solutions.face_mesh`（旧 Solutions API） | 必须是**仍带 `solutions` 的旧构建**，GYZ 当时用的是 0.10.14 |

本机实测（mediapipe 0.10.35 + Python 3.14）：A-包正常起服务；
`vision_a.py` 在自检阶段就退出：

```
[A] 性别/年龄估计禁用：未找到 age_gender.onnx（一次性下载见 README）
[A] ❌ 自检失败：FaceMesh 模型加载失败（module 'mediapipe' has no attribute 'solutions'）
```

（性别/年龄那条是**正常的降级提示**，不是故障 —— `onnxruntime` 装了但模型文件
需要另外下载，见下文。真正拦住的是下面那条自检失败。）

**这不是合并引入的回归**：两套 A 在合并前就各自锁着不同的 mediapipe，
只是此前互相看不见、没人同时装过；合并到同一个仓库后才浮出来。

### 二、路径：**换对版本也还起不来** —— Solutions API 打不开含中文的路径

这条 2026-09-29 才挖出来，比版本问题隐蔽得多。本仓库路径含中文
（`D:\Virtually C\成都东软学院下期\...`），而 Solutions API 会把模型**文件路径**
（`face_landmark_front_cpu.binarypb` 等）交给 C++ 层，C++ 按 ANSI 代码页解释路径，
中文目录打不开。症状极具欺骗性：

```
FileNotFoundError: The path does not exist: ...\mediapipe\modules\face_landmark\face_landmark_front_cpu.binarypb
```

**文件确实在、`os.path.exists()` 也返回 True**，只有 C++ 那边看不见。

> 定位判据：把 mediapipe 包整个拷到纯 ASCII 路径并用 `PYTHONPATH` 前置，
> FaceMesh 立刻能实例化。**用目录 junction 骗不过去** —— venv 里
> `mediapipe.__file__` 会解析回中文原路径。

A-包不受影响：它用 **Tasks** API，模型字节在 Python 侧读出来再交给 C++，不传路径。

### 可用组合（2026-09-29 实测）

```bash
# 环境必须建在 **仓库外** 的纯 ASCII 路径；Python 用 3.12（原因见下）
python3.12 -m venv C:\venv-gyz-a
C:\venv-gyz-a\Scripts\pip install opencv-contrib-python==4.13.0.92 mediapipe==0.10.14 onnxruntime==1.23.2

cd backend_A
C:/venv-gyz-a/Scripts/python.exe vision_a.py --no-window --no-socket --seconds 10
```

实测（本机 `C:\Users\35920\venv-gyz-a`，Python 3.12.10）：打印
`[A] 性别/年龄模型已加载` → `[A] 摄像头已打开：index=0` → `[A] 自检通过`，退出码 0。

### 更正：两套 A **不互斥**，可以共用一套环境

本文档早先写过「两套 A 装不进同一个 Python 环境」，**这个结论是错的**。
2026-09-29 实测两项后更正：

| 实测 | 结果 |
|---|---|
| `mediapipe==0.10.14` 是否同时带两套 API | **是** —— `solutions` 与 `tasks` 都在 |
| A-包跑在 0.10.14 下 | **40 个测试全过**（`python -m unittest discover -s tests`） |

即 **0.10.14 是两套都满足的下限，一套环境就能同时跑两套 A**。真正的约束是另两条：

1. **Python ≤ 3.12** —— 0.10.14 **没有 cp314 wheel**
   （`pip download --python-version 3.14` 报 `No matching distribution found`）。
   本机的 3.12.10 可以，3.14 装不上。
2. **环境建在纯 ASCII 路径**（仓库外）—— 否则 A-单文件仍然起不来，见上。

> 早先本节建议 `python -m venv .venv-gyz-a`（建在仓库里），**那条路走不通**
> （中文路径，且 3.14 装不上 0.10.14），已按实测替换为上面的建法。
>
> **演示路径走的是 A-包**（合成源、不需要摄像头、40 个测试），
> 所以锁 0.10.35 不影响演示。

---

## 一、A-包（`module_a_vision/`）
采集 → 人脸关键点 → 五项指标 → 上报。**图像不出本模块**，出站只有结构化统计量。

从 `xiangmu1/home_robot` 移植而来，并对齐团队 `api_doc.md` §3.2 的报文格式。

---

### 1. 快速开始

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

### 2. 端口与连接方向

| 链路 | 端口 | A 的角色 | 数据格式 |
| ---- | ---- | -------- | -------- |
| A → B | 127.0.0.1:8000 | **服务端**（listen） | 一行一条 JSON，`\n` 分隔 |

⚠️ **方向别写反**：A 是 `listen` 的一方，B 主动 `connect`。

```bash
python main.py --port 9000        # 换端口
python main.py --host 0.0.0.0     # 换监听地址（本机回环以外，需自行评估风险）
```

---

### 3. 目录结构

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

### 4. 出站格式：`--emit v1`（默认）与 `v2`

A 现在能吐**两种**报文，用 `--emit` 选。

#### 4.1 `v1` —— api_doc §3.2，逐帧平铺 8 字段

```json
{"timestamp": 12.3, "has_face": true, "ear": 0.29, "blink_cnt": 7,
 "pitch": 2.0, "yaw": -3.0, "roll": 1.5, "emo_feature": "normal"}
```

**这是团队仓库的默认，也是 `backend_B` 认的格式。**

#### 4.2 `v2` —— 原生的 10 秒嵌套窗口

`WindowState`（13 个顶层键 + `observations` 五个子对象），另含
`heartbeat` 与 `vision_unusable` 两条附加报文。`xiangmu1` 侧的消费方用这个。

```bash
python main.py --emit v2
```

#### 4.3 为什么 `v1` 模式下**不发**心跳

`backend_B` 的 `VisionSample.from_payload` 会把任何一条缺字段的报文
兜成一个 `has_face=false` 的样本**推进判定器**。一条 `heartbeat` 会被
读成"这一刻看不见老人"，周期性污染 B 的状态输入。

B 不需要心跳：A 死掉由它那个 5 秒失联判据兜住。这是"一种模式一种报文"的
必然结论，不是省事。

#### 4.4 为什么是**逐帧**而不是逐窗口（一条真实踩过的坑）

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

#### 4.5 字段映射

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

#### 4.6 `emo_feature=tired` 代码可达、**数据不可达**

表情分类器对疲惫有刻意的阻尼（`TIRED_DAMPING = 0.6`），
所以表情标签在结构上**不会**变成 `tired`。B 判疲惫只能靠
`ear` / `blink_cnt` / `pitch` 三条路。

`EMOTION_TO_FEATURE` 里 `Emotion.TIRED → "tired"` 这一支有测试覆盖，
但不要因为它是绿的，就以为实跑中会出现 `emo_feature=tired`。

---

### 5. 命令行参数

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

### 6. 联调步骤

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

### 7. 验证状态（请如实阅读）

本节比其它任何一节都重要。**不要把没验证过的东西当成验证过的。**

#### 已验证

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

#### 未验证 —— 请勿当作已完成

| 内容 | 为什么没验证 | 怎么补 |
| --- | --- | --- |
| **「像素 → 人脸关键点」这一步** | 开发机**没有摄像头**。合成源与 CSV 回放直接产出特征，**跳过了像素** | 接真实摄像头或真人视频：`python main.py --source camera` |
| MediaPipe 后端在真实画面上的表现 | 同上 | 同上 |
| **真实场景下的判定准确率** | 同上。验的是**判定逻辑**，不是**视觉精度** | 需要真人数据，并做误报/漏报统计 |
| **表情识别的精度** | 是 52 个 ARKit blendshape 的**规则启发式基线**，不是训练模型 | 采集真实老人表情数据后接 ONNX 模型，`ExpressionClassifier` 接口已留好 |
| 长时间运行稳定性 | 没有跑过数天量级的浸泡测试 | — |
| `--source video` | 没有可用的测试视频 | 自备一段 |

---

### 8. 已知问题

#### ✅ 顶层 `requirements.txt` 的 `cv2` 冲突 —— 已解决（2026-09-29）

原来顶层写着 `opencv-python==4.13.0.90` + 未锁版本的 `mediapipe` / `pyaudio`，
与本模块用的 `opencv-contrib-python` 冲突：两个包**都提供 `cv2`**，
同时安装时谁生效取决于安装顺序，而且**卸载任一个都会删掉另一个的文件**
（删的是那个共享的 `cv2` 目录）—— 表现就是"昨天还好好的"这类极难排查的故障。

**现已按原建议全部改完**，两边一致：

1. 顶层只留 `opencv-contrib-python==4.13.0.92`。
2. 删掉 `pyaudio` / `requests` —— 全仓无人 import，录音一律走 `sounddevice`。
3. `mediapipe` 锁成 `0.10.35`。顶层注释担心的 `mediapipe.solutions.face_mesh`
   （旧 Solutions API）**与本模块无关**：A 用的是 **Tasks API**
   （`mediapipe.tasks.python`）；但版本仍然要锁 —— 写 `>=1.0` 会被解析到
   `1.0.1`，那是跨大版本的另一套东西。

修 `cv2` 请用下面这条，**单独 uninstall 再装回来是修不好的**：

```bash
pip install --force-reinstall --no-deps opencv-contrib-python==4.13.0.92
```

#### A 在团队仓库与 xiangmu1 会各自演进（分叉）

见 §10。

---

### 9. 降级矩阵

| 缺什么 | 会怎样 | 怎么看出来 |
| --- | --- | --- |
| **摄像头** | 用合成源或 CSV 回放 | `--source synthetic` / `--source csv` |
| **mediapipe / opencv** | 合成源与 CSV 源照常工作（它们不经过像素） | `--source camera` 才会报错，且退出码是 2 而不是抛栈回溯 |
| **B 没启动** | A 照常采集，只是没人收；B 起来后自动接上 | 日志 `[A] B 已连接` |
| **B 断开** | A 继续跑，日志提示等待重连 | `[A] 所有 B 连接均已断开，等待重连` |
| **采集源耗尽** | 停止产出但进程不退出（"看不见了"本身是要上报的状态） | `[A] 采集源已耗尽` |

---

### 10. 与 `xiangmu1/home_robot` 的关系

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

---

## 二、A-单文件（`vision_a.py`）

> 以下为 GYZ 在 `origin/main` 上的原文，标题层级已整体下沉一级，内容未改动。

摄像头 + MediaPipe FaceMesh，每帧计算指标，**双出口**：CSV 落盘 + TCP Socket 服务（api_doc §3）。
附加 rPPG 非接触脉搏波（心率/呼吸率/心跳间期，输出走独立波形 CSV，尚未进协议，见需求文档 §12.4）。

### 运行

```bash
cd backend_A
python vision_a.py                # 开预览窗口，q 退出，c 重新标定
python vision_a.py --seconds 10   # 采 10 秒自动退出（自测）
python vision_a.py --no-window    # 后台采数，不开窗口
python vision_a.py --no-socket    # 只写 CSV，不起 Socket
python vision_a.py --camera 1     # 指定摄像头编号（默认 0，失败自动回退 1）
python sim_b.py                   # 另开终端：模拟 B 连 8000 收流并按 api_doc §3 逐条校验
```

依赖：`opencv-contrib-python==4.13.0.90`、`mediapipe==0.10.14`、`onnxruntime==1.23.2`（性别/年龄用，可选）（见根目录 requirements.txt）。

### 输出

| 文件 | 内容 |
|---|---|
| `data/vis_data_时间戳.csv` | 协议数据：表头 `timestamp,has_face,ear,blink_cnt,pitch,yaw,roll,emo_feature`（api_doc §3.4，UTF-8 无 BOM，`\n` 换行，数值 2 位小数） |
| `data/pulse_wave_时间戳.csv` | 实验数据：`timestamp,r,g,b,a_lab,b_lab,mouth_open,hr,rr,ibi_ms,sqi`（rPPG 原始三通道/双颊 Lab 色值/口部开口度 + 派生指标 + 质量分，未进协议；`mouth_open` 为唇 13/14 开口度时序，供 B 对话状态机，协议字段见 api_doc §3.5 V1.2 草案）。**仓库内保留了一份真实波形样例** `data/pulse_wave_20260929_181230.csv`（1563 帧、`hr` 有值 93%、末次 HR=75bpm / SQI=1.0）作为格式实证，由 .gitignore 的 `!` 例外放行 |
| TCP 127.0.0.1:8000 | 每帧一行 JSON + `\n`（字段同 CSV）；**无人脸帧照发心跳** `has_face=false`（api_doc §3.3）——B 靠心跳区分"没人"与"掉线"，B 断开自动等待重连 |

**rPPG 成熟度（诚实标注）**：采用 **CHROM 色度法**（三通道抗运动伪影，优于裸绿通道）+ **SQI 质量门控**
（SNR=带内/带外功率比 <1.5 时 HR/IBI 置灰 `--`；IBI 还要求间隔变异系数 CV≤0.3）。HR 为**参考级**（安静场景可用，
动作多时误差大）；**RR 需 20 秒窗口才输出；IBI/RR 为实验性**。全部输出禁止作为医疗结论（需求文档 §12.1 T3 红线）。

**rPPG 验证（Bland-Altman，手环对照）**：课程答辩的精度证据链，`validate_rppg.py` 两步——
1) 戴手环安坐，终端 1 跑 `vision_a.py`，终端 2 跑 `python validate_rppg.py --record`，
   每 ≥30 秒看一眼手环输入读数，采 8-10 个点按 q；
2) `python validate_rppg.py --compare --sync data/rppg_sync_xxx.csv`（波形 CSV 自动取最新），
   输出 bias / 95% LoA / MAE / Pearson r，逐点表存 `data/rppg_validation_*.csv` 供报告画散点图。
结论口径只写"与手环读数一致性"（手环自身也有光电误差），不写精度绝对值。

### 性别/年龄估计（P2 演示项，可选）

`age_gender.py` 加载 `models/age_gender.onnx`（62x62 人脸输入，onnxruntime CPU ~3ms/次，1 秒节流），
结果只进预览窗与控制台摘要，**协议与 CSV 均不变**。年龄为**预测**口径（MAE ±5-7 年），禁止当真实信息用。
模型缺失或 `age_gender_enabled=false` 时自动禁用，不影响主流程（A9 降级语义）。

模型一次性下载（已加入 .gitignore，8.5MB）：

```bash
# 国内直连（hf-mirror，facefusion/insightface 生态的 gender_age 转换版）
curl -L -o backend_A/models/age_gender.onnx \
  "https://hf-mirror.com/bluefoxcreation/gender_age/resolve/main/gender_age.onnx"
```

性别通道序已用 OpenCV 示例图 lena.jpg 实测校准（`age_gender.py` 注释）；换其他 ONNX 版本若性别反向，
改 `infer()` 里 `gender_v[1]` 的索引即可。

- 标签小写英文：`normal / tired / sad / blank`（api_doc §3.2 V1.1 枚举）。
- **无人脸帧跳过 CSV 写入**，Socket 照常发心跳。
- ⚠️ 8000 端口与 Django runserver 默认端口冲突：起 Socket 时别同时 `python manage.py runserver`（脚本检测到占用会打印警告并只出 CSV）。

### 基线标定（重要）

首次运行自动采集前 25 个有效帧（约 2 秒）的**个人基线**并保存到 `baseline.json`；
之后启动**直接加载，无需重新标定**。pitch/yaw/roll/嘴角弧度输出的是相对基线的偏移量
（解决 pitch 数值整体偏大的问题）。预览窗口按 `c` 键随时重标定，结果覆盖保存。
若加载的旧基线与当前姿态持续失配（如换人/换机位，连续 90 帧极端偏移），程序自动重采——
**自动重标仅当次生效**，落盘仍需按 `c` 人工确认，防止跌倒等异常姿态被固化成基线。
`baseline.json` 属个人派生数据，仅存本地，已加入 .gitignore 不进版本库。
**首次标定（或按 c 重标）时保持正常坐姿，不要歪头做表情。**

**2026-09-29 实测（Python 3.12.10 + mediapipe 0.10.14，纯 ASCII 路径环境）**：
首跑打印 `[A] 基线标定完成：pitch=15.30 yaw=-4.73 roll=-0.92` 并写出 `baseline.json`；
再跑时 `Calibrator.load()` 返回 `True`、`feeding()` 返回 `False`（不再重新标定）、
`_loaded=True`（自动重标保护已激活）—— **A8 持久化确实生效**，不只是文档里写着。

⚠️ 标定出的 `baseline[3]`（嘴角弧度基准）**容易正好是 `0.0`**：若前 25 个有效帧的
弧度均值落在 ±0.0005 内，四舍五入后即为 0，此时 `sad` 判据退化成「原始弧度 >
`sad_curvature`」，不再相对个人基线，而默认 0.015 偏严 —— 实测 1538 帧、其中
刻意做了 20 秒难过表情的会话里，`sad` 也只触发 19 帧（1.2%）。想采到足量 `sad`
得真的皱眉、嘴角明显下压。**这与本仓库既有样例一致**（`vis_data_20260922_154601.csv`
610 帧里 `sad` 只有 2 帧），是阈值口径问题、不是故障。

### 调参

所有阈值集中在 `config.json`。个体差异大时优先调：`ear_tired`（疲劳）、`sad_curvature`（难过）、
`gaze_still_th`（发呆判定灵敏度）、`pitch_scale/yaw_scale`（头姿灵敏度）。
`low_light_th` 为照度自检阈值（画面均值 0–255，默认 45）：低于阈值时控制台与预览窗口告警
"has_face=false 可能是光线问题而非无人"，供 B 侧区分"黑屋/离开/掉线"参考。
`csv_retention_days` 为采集 CSV 保留天数（默认 7）：启动时自动清理过期文件，设 0 关闭清理。
`max_faces` 为同时跟踪的人脸数上限（默认 3）：多人入镜时按包围盒面积锁定主脸（通常离镜头最近者），
访客短暂入镜不会抢走主人指标；单脸场景 CPU 开销不变。

### 给 B 交付样例前的检查项

1. 程序退出时打印自测摘要：**四种标签都出现过**（对着镜头分别做正常/疲惫/难过/发呆各十几秒）。
   ⚠️ `sad` 和 `blank` 是最难采的两类：前者要真皱眉（阈值说明见「基线标定」一节），
   后者要求视线静止满 `gaze_still_sec`（默认 3 秒）—— 实测各留 20 秒也只采到
   19 / 18 帧。交付口径是**出现过**即可，不要求这两类占比高；
2. CSV 行数 = 有人脸帧数，表头与 api_doc §3.4 逐字一致；
3. Socket 用 `telnet 127.0.0.1 8000` 或 B 侧脚本连上能收到 JSON 流（含无人脸心跳）。
