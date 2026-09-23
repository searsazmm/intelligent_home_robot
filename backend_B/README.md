# 后端 B —— 业务后端

居家陪伴机器人的业务模块：接收视觉后端 A 的实时数据，识别用户文本情绪，
结合两者管理对话，并把用户状态推送给桌面前端 C。

不使用 Django，不涉及网页。**运行期零第三方依赖**（纯 Python 标准库）。

---

## 1. 快速开始

```bash
cd backend_B

# 方式一：完整联调（需要模块 A 已经在跑）
python main.py

# 方式二：离线自测，不需要模块 A 和 C（推荐第一次跑用这个）
python main.py --offline data/sample_vision.csv --stdin
#   然后直接在终端里打字，就当作是用户说的话

# 方式三：离线回放 + 模拟前端，看完整链路
python main.py --offline data/sample_vision.csv --loop --speed 5   # 终端1
python tools/mock_c_client.py --auto                              # 终端2
```

首次使用如果 `data/sample_vision.csv` 不存在，先执行：

```bash
python tools/make_sample_vision.py
```

跑单元测试（48 个用例，标准库 unittest，无需安装任何东西）：

```bash
python tests/test_core.py
# 或
python -m pytest tests/ -v
```

---

## 2. 端口与连接方向

> ⚠️ **注意方向别写反了。** 模块 B 在 A→B 这条链路里是**客户端**（主动 connect），
> 在 B→C 两条链路里都是**服务端**（被动 listen）。

| 链路 | 端口 | B 的角色 | 数据格式 | 来源 |
| ---- | ---- | -------- | -------- | ---- |
| A → B | 127.0.0.1:8000 | **客户端**（连 A） | JSON，`\n` 分隔 | api_doc §3 |
| B → C | 127.0.0.1:8001 | 服务端 | 纯文本状态串，`\n` 分隔 | api_doc §4 |
| C ↔ B | 127.0.0.1:8002 | 服务端 | JSON，`\n` 分隔 | api_doc §5（V1.1 新增） |

8002 是本次新增的：api_doc V1.0 只定义了 B→C 的单向状态，没有"用户说的话怎么进 B"
的通道，文本情绪识别拿不到输入。已按流程补进 api_doc §5，**C 端同学需要对齐这一节**。

所有端口都能用命令行参数或环境变量改，联调时不用改代码：

```bash
python main.py --vision-port 9000 --chat-port 9002
set B_VISION_PORT=9000 && python main.py
```

---

## 3. 目录结构

```
backend_B/
├── main.py                    入口：装配组件、管线程、处理退出
├── config.py                  全部常量：端口、阈值、路径（改参数只看这个文件）
├── requirements.txt           依赖清单（运行期零依赖）
├── core/
│   ├── protocol.py            \n 分帧 + JSON 编解码（防粘包/半包/中文截断）
│   ├── vision_client.py       连模块 A，带指数退避断线重连；另含离线 CSV 回放
│   ├── vision_state.py        视觉特征 → normal/sad/tired/absent（防抖 + 迟滞）
│   ├── text_emotion.py        文本情绪识别（情感词典 + 否定/程度副词规则）
│   ├── dialogue.py            对话管理：视觉 × 文本 × 意图 → 回复
│   ├── history_store.py       CSV 历史记录读写
│   └── ui_channel.py          对 C 的两个服务（8001 状态推送 / 8002 双向对话）
├── data/
│   ├── sample_vision.csv      离线调试用的视觉数据（由 tools 脚本生成）
│   ├── sample_chat.csv        模拟 C 的自动对话脚本
│   └── history.csv            历史对话记录（运行时生成，不进版本库）
├── tools/
│   ├── make_sample_vision.py  生成样例视觉 CSV
│   ├── mock_a_server.py       模拟模块 A（TCP 服务端），含断线演练开关
│   └── mock_c_client.py       模拟模块 C（命令行），连 B 的两个端口
└── tests/
    └── test_core.py           48 个单元测试
```

---

## 4. 各功能说明

### 4.1 接收模块 A 的视觉数据（含断线重连）

`core/vision_client.py` 是 TCP **客户端**，主动连 `127.0.0.1:8000`。

容错策略：

| 情况 | 行为 |
| ---- | ---- |
| A 还没启动 | 指数退避重试：1s → 2s → 4s → 8s → 10s 封顶，**永不放弃** |
| A 主动断开 / 网络中断 | 立即重连，退避重置回 1s |
| 收到半包 / 粘包 | 按 `\n` 缓存拼包，中文被 TCP 从中间切开也不会解错 |
| 收到非法 JSON | 记一条 WARNING 后跳过，**不断开连接** |
| 字段缺失或为 null | 用安全默认值兜底，不抛异常 |
| 连接断开期间 | 状态自动降级为 `absent`（看不见人就该报无人，不能假装正常） |

重连期间 8001 / 8002 两个服务照常工作，不会互相阻塞。

**演练重连**（模拟 A 发 100 帧后挂断）：

```bash
python tools/mock_a_server.py --loop --speed 10 --drop-after 100
```

### 4.2 视觉状态判定

api_doc §3.3.2 明确写了"A 仅负责采集，不做最终状态判定"，所以判定在 B 端：
`core/vision_state.py`。

判定优先级：`absent` > `tired` > `sad` > `normal`

- **absent** — 连续 2 秒没检测到人脸；或人脸只在窗口里零星闪现
- **tired** — A 端给 `emo_feature=tired`；或 EAR 低于阈值持续 3 秒；或眨眼频率偏高；或持续低头
- **sad** — A 端给 `emo_feature=low`
- **normal** — 以上都不是

三道防抖机制，避免前端状态乱跳：

1. **滑动窗口** — 只看最近 10 秒，单帧异常不影响结果
2. **防抖** — 新状态要连续稳定 1.5 秒才生效
3. **迟滞** — 从 absent 恢复要 2.5 秒（比进入慢），人一闪而过不会立刻切回 normal

所有阈值都在 `config.py` 里，联调时按实际摄像头表现调整。

### 4.3 文本情绪识别

`core/text_emotion.py`，情感词典 + 规则，不接模型、不下载权重、可离线、结果可解释。

经典三件套：

- **情感词打分** — 命中词典累加权重
- **程度副词放大** — "很累"比"累"更消极
- **否定词翻转** — "不开心"要翻成消极

另外单独标记**身体不适**（头疼、不舒服…），对话那边会走健康关怀分支；
**危险表达**（"活着没意思"等）会拿到最强消极权重。

输出三分类 `label`（negative/neutral/positive）、细粒度 `emotion`
（tired/sad/angry/anxious/happy/neutral）、连续 `score`、命中的词 `hits`（调试用）。

要扩充词表，直接改 `EMOTION_LEXICON` 即可，逻辑不用动。
注意词典是**子串匹配**的：`"心情不好"` 匹配不到 `"心情不太好"`，
插入式的常用变体必须单独列一条。

### 4.4 对话管理

`core/dialogue.py`。一次回复做四件事：

1. **识别意图** — 问候/告别/感谢/问身份/问时间/身体不适/倾诉/闲聊
2. **融合状态** — 把视觉状态和文本情绪合并成一个状态（这个状态也会推给 C）

   | 视觉 | 文本 | 融合结果 | 理由 |
   | ---- | ---- | -------- | ---- |
   | normal | 明显消极 | **sad** | 视觉没捕捉到，但文字不会骗人 |
   | tired | 中性 | **tired** | 视觉更"硬"，用户说"我没事"也不改 |
   | absent | 用户刚发过消息 | **normal** | 人在打字，说明人在，推翻视觉误判 |

3. **挑回复模板** — 按 `(状态, 意图)` 精确匹配 → 意图专属 → 状态兜底 → 通用兜底，
   并避开最近说过的句子，不让回复显得机械
4. **返回结果** — 由 `main.py` 写入历史 CSV

回复模板都在 `dialogue.py` 底部的 `REPLY_TEMPLATES` 里，改文案不用碰逻辑。
写模板的原则：一句话不超过 40 字、先说共情再给建议、不用命令句。

> 这里刻意**不接大模型**：原型阶段要的是可预测、可调试、零依赖。
> 每条回复为什么被选中，看日志里的 `意图/状态/情绪` 就能指着代码说清楚。

### 4.5 CSV 历史记录

`core/history_store.py`，满足"预留读取 CSV 历史记录的接口"。

一次交互写两行（用户一句 + 机器人一句），表头：

```
timestamp,session_id,role,text,vision_state,text_emotion,emotion_score,intent
```

拆两行而不是挤一行，是为了整份文件格式统一：每行都只是"某人在某时刻说了一句话"，
后续做数据分析或画情绪曲线都方便。

已提供的读取接口：

| 方法 | 用途 |
| ---- | ---- |
| `load_recent(n)` | 取最近 n 条（启动时预加载上下文） |
| `load_session(id, n)` | 取指定会话 |
| `iterate()` / `iter_records()` | 流式读取全部 |
| `recent_dialogue(turns)` | 转成 `[{role, text}]` 给对话管理用 |
| `last_robot_replies(n)` | 取机器人最近说过的话，用于避免重复 |

要换成数据库或远程服务时，照着 `HistoryProvider` 协议实现同样的方法，
对话管理一行都不用改。

历史文件损坏、缺列、数字格式不对，都不会抛异常影响主流程。

---

## 5. 命令行参数

```
python main.py [选项]

  --offline CSV        离线模式：回放该 CSV，不连模块 A
  --speed FLOAT        离线回放倍速（默认 1.0）
  --loop               离线回放循环播放
  --stdin              开启控制台输入，直接打字测试对话（输入 :q 退出）

  --vision-host/--vision-port    模块 A 地址与端口（默认 127.0.0.1:8000）
  --status-host/--status-port    状态推送监听（默认 127.0.0.1:8001）
  --chat-host/--chat-port        对话通道监听（默认 127.0.0.1:8002）

  --history PATH       历史记录 CSV 路径
  --no-history         不读写历史记录
  --log-level LEVEL    DEBUG / INFO / WARNING / ERROR（默认 INFO）
```

---

## 6. 联调步骤

按 api_doc §6.2 的顺序：**先 A → 再 B → 最后 C**。

```bash
# 终端1：模块 A
python backend_A/main.py

# 终端2：后端 B
cd backend_B && python main.py

# 终端3：模块 C
python frontend_C/main.py
```

模块 A 和 C 还没就绪时，用 tools/ 里的模拟程序顶上：

```bash
# 终端1：模拟 A
cd backend_B && python tools/mock_a_server.py --loop

# 终端2：后端 B
cd backend_B && python main.py

# 终端3：模拟 C
cd backend_B && python tools/mock_c_client.py --auto
```

### 排查问题

- **B 一直打印"连接模块 A 失败"** — 模块 A 没启动，或端口不是 8000。
  用 `netstat -ano | findstr :8000` 看到底谁在听。
- **C 连上了但收不到状态** — B 只在状态**变化**时推送，另外每 15 秒有一次心跳。
  想看实时变化，用 `--log-level DEBUG`。
- **状态一直是 absent** — 检查 A 发的 `has_face` 是不是一直是 false；
  B 收到的报文会记进日志。
- **端口被占用** — 用 `--status-port` / `--chat-port` 换端口，
  或 `netstat -ano | findstr :8001` 找到进程 PID 后 `taskkill /F /PID <PID>`。

---

## 7. 已知限制

写在明处，避免联调时误判成 bug：

1. **`--speed` 会压缩时间尺度。** 离线回放的倍速是按墙钟 sleep 实现的，
   所以"持续 N 秒"这类阈值在倍速下也按同样比例压缩（这是想要的效果，
   能更快看完状态变化）。但**短事件可能被漏掉**：样例数据里"无人"只有 8 秒，
   在 `--speed 5` 下只剩 1.6 秒，而"人脸丢失宽限 2 秒"是按墙钟算的，
   就来不及判定。要看完整状态序列请用 `--speed 1`。
   眨眼频率已经改成按数据里的 timestamp 计算，不受倍速影响。
2. **文本情绪是词典法，不是模型。** 反讽、隐喻、"累是累了点但挺开心的"
   这种混合情绪会落在 neutral。对陪伴机器人的原型够用，不要当成通用情感分析。
3. **对话是规则模板，没有真正的上下文记忆。** 历史 CSV 目前只用于
   避免回复重复，没有喂进语义理解。
4. **阈值是按样例数据调的**，`config.py` 里的 EAR、低头角度等需要按
   真实摄像头和真实用户重新标定。
