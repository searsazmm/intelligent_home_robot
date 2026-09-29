# 居家陪伴机器人

适老化居家陪伴机器人原型：**面对面人机交互，全程语音对话，不需要打字**。

三个模块通过 TCP 互联，各自可独立开发与测试：

```
   ┌───────────────┐   8000    ┌───────────────┐  8001  ┌───────────────┐
   │  模块 A        │ ────────► │   模块 B       │ ─────► │   模块 C       │
   │  视觉感知      │  JSON     │   交互决策     │ 状态串 │   前端界面     │
   │  摄像头+MediaPipe│          │   核心交互     │ ◄───── │   PyQt5       │
   └───────────────┘           └───────────────┘  8002  └───────────────┘
                                  A→B 是 B 去连 A      JSON    纯黑底 + 白色颜文字
```

| 模块 | 做什么 | 端口 | 详细文档 |
| ---- | ------ | ---- | -------- |
| **backend_A** 视觉感知 | 摄像头采集 + MediaPipe 人脸状态识别，判断眼神呆滞 / 情绪低落，实时发特征给 B | 8000 服务端 | [backend_A/README.md](backend_A/README.md) |
| **backend_B** 交互决策 | 接收视觉状态、麦克风语音识别、决策（主动关心 / 日常问候 / 正常应答）、语音合成、驱动前端表情 | 8000 客户端<br>8001 / 8002 服务端 | [backend_B/README.md](backend_B/README.md) |
| **frontend_C** 前端界面 | 纯黑背景窗口 + 白色颜文字表情，按 B 的指令动态切换 | 8002 客户端 | [frontend_C/README.md](frontend_C/README.md) |

接口规范（**改端口和字段前必读**）：[api_doc.md](api_doc.md)。
四个人机状态的取值固定为 `normal` / `sad` / `tired` / `absent`（api_doc §4.2，禁止自造）。

---

## 快速开始：三个终端，A → B → C

按 api_doc §6.2 的顺序启动：**先 A → 再 B → 最后 C**。

```bash
# 终端 1：模块 A（默认用合成源，不需要摄像头）
cd backend_A && python main.py --scenario sad

# 终端 2：模块 B（等 A 起来后再启动）
cd backend_B && python main.py

# 终端 3：模块 C（纯黑窗口，出现颜文字）
cd frontend_C && python main.py
```

顺序反了也不会坏：B 会指数退避重连 A（1s→10s 封顶，永不放弃），
C 也会自己重连 B，重连期间照常显示 `normal` 表情。

### 常用参数

```bash
cd backend_A && python main.py --list-scenarios    # 看有哪些合成剧本
cd backend_A && python main.py --source camera     # 用真实摄像头
cd frontend_C && python main.py --fullscreen       # 全屏无边框（答辩用，ESC 退出）
cd frontend_C && python main.py --dump-glyphs      # 字形自检：看不到方块即为正常
```

---

## 答辩演示路径（不用摄像头、不用模块 A）

没有摄像头、或者模块 A 起不来时，用 B 的离线回放顶上。
`--demo` 会放宽主动关怀的阈值，让它在演示的几十秒内真的触发：

```bash
# 终端 1：模块 B —— 离线回放 + 演示阈值 + 循环
cd backend_B && python main.py --offline data/sample_vision.csv --speed 5 --loop --demo

# 终端 2：模块 C —— 真实的纯黑颜文字界面
cd frontend_C && python main.py

# 终端 3（可选）：被动监听 8002，确认 B 真的发了 proactive 报文
cd backend_B && python tools/watch_8002.py --seconds 90
```

> `sample_vision.csv` 的时间线：`normal → tired → absent → normal → sad → normal`，
> 全长 119.9 数据秒，`--speed 5` 下约 24 墙钟秒跑完。
>
> ⚠️ **验证主动关怀不要用 `tools/mock_c_client.py`** —— 它每发一句话都会刷新
> B 的 60 秒 `user_cooldown`，主动关怀在此期间**按设计**不会触发。
> 用一句话都不发的 `watch_8002.py`。

### 一条命令自检

```bash
cd backend_B
bash tools/e2e_offline_check.sh    # 五步：状态判定 / 主动关怀 / 8002 收报文 / 对话回环 / 无异常
bash tools/fault_drill.sh          # 容错演练：杀掉 A 或 B，看另外两个模块自己长回来
```

两个脚本都不开摄像头、不碰声卡，任何机器上都能跑，**答辩前跑一遍**。

---

## 安装

```bash
pip install -r requirements.txt              # 视觉 + 前端 + 测试所需的全部依赖
pip install -r backend_B/requirements-voice.txt   # 可选：语音（不装也能跑，见下）
```

- `opencv-contrib-python` + `mediapipe` —— 模块 A 用。两者版本已锁定，
  MediaPipe 与 OpenCV 的版本不匹配是「昨天还好好的」类故障的主要来源。
- `PyQt5` —— 模块 C 用。装完请用下面这条**验证**（`import PyQt5` 在残缺安装时
  会假成功，不作数）：
  ```bash
  python -c "from PyQt5.QtWidgets import QApplication; print('OK')"
  ```
- **模块 B 运行期零第三方依赖**（纯标准库），这是刻意设计 —— 答辩现场少一个
  装包失败的可能就少一个坑。语音库全部是**可选**的，缺库时自动降级到键盘
  （`--stdin`）或 8002 文本通道，**绝不阻止 B 启动**。
  这条约束有测试守着（AST 扫描顶层 import）。

语音的完整说明（引擎选择、采样率、声卡陷阱）见
[backend_B/README.md §4.8](backend_B/README.md)。

---

## 测试

```bash
python -m pytest              # 仓库根跑全部三个模块：456 passed, 7 skipped
```

分模块跑：

```bash
cd backend_A && python -m unittest discover -s tests   # 40 个（标准库 unittest）
cd backend_B && python -m pytest tests/                # 340 个
cd frontend_C && python -m pytest tests/               # 76 个
```

### 关于那 7 个 skipped

它们全是前端窗口测试里的**字形断言**，跳过原因是**离屏平台的字体库是空的**
（`QFontDatabase().families()` 返回 0 个字体族，Qt 在该平台一个字都画不出来）。

这不是「测试写坏了」，而是刻意的：离屏下断言"脸上有白色像素"必然失败，
所以那些用例宁可**跳过并说明原因**，也不假装通过。想看它们真的执行：

```bash
QT_QPA_PLATFORM=windows python -m pytest        # 463 passed, 0 skipped
```

**答辩前建议在演示机上跑这一条** —— 它才是能发现「字体选错导致满屏豆腐块」
的那一档测试。前端 README §9 有完整说明。

---

## 常见问题

1. **`C 的表情不动`** —— 第一步永远是问「B 到底发了没有」：
   ```bash
   cd backend_B && python tools/watch_8002.py
   ```
   看 8002 上有没有报文，就能立刻区分是 B 没发、C 没收到、还是 C 收到了没画。

2. **端口被占用 / 两个 B 同时在跑** ——
   Windows 的 `SO_REUSEADDR` 与 Linux **语义不同**：它允许第二个进程绑定同一端口，
   于是第二个 B 不会报错，而是**静默分走一部分连接**。症状是「日志正常但界面上啥也没有」。
   ```bash
   netstat -ano | grep -E ":800[012].*LISTENING"   # 看 PID
   taskkill //F //PID <PID>
   ```

3. **有声音「播放成功」但听不见** —— 本机真出现过，原因是 **Realtek 渲染端点的
   音量停在 14%**：播放调用完全成功、耗时也正常，就是没声音。先查端点音量
   （不是应用音量）；仍不行再看列表显式指定设备：
   ```bash
   cd backend_B && python main.py --list-audio
   cd backend_B && python main.py --audio-out Realtek
   ```
   ⚠️ 「默认输出是 ToDesk 虚拟声卡、走默认设备必然听不见」这条旧记录**是错的** ——
   PortAudio 的 MME 默认输出就是 Realtek 扬声器（`sd.default.device` = `[1, 4]`），
   详见 [backend_B/README.md 故障排查](backend_B/README.md)。
   设备名一律填**名字里的一段**，不要填序号（序号会随虚拟设备增减漂移）。

4. **脚本 grep 中文匹配不上** —— 输出重定向到文件时，Python 用系统 ANSI 码页
   （本机 GBK），而脚本文件是 UTF-8。各入口的 `main()` 都会把 stdout 切到 UTF-8；
   自己写新脚本时记得照做，**别靠放宽断言来「修」这类失败**。

5. **A 连不上 / 状态一直 absent** —— 检查模块 A 是否在 8000 上监听，
   以及它发的 `has_face` 是不是一直为 false。B 收到的报文会记进日志
   （`--log-level DEBUG`）。

---

## 当前状态：哪些验证过，哪些没有

写在明处，避免把「没验证」当成「坏了」，也避免把「没验证」当成「好的」。

**已实测跑通：**

- 真实三模块链路（A 合成源 → B → C 真实 PyQt 窗口）：
  A 的 `sad` 剧本让 B 判定为 `sad`（`reason: A 端 emo_feature=low`），
  C 的表情依次切换 `normal → absent → sad`，全程无异常。
- 离线端到端自检 5/5 通过；主动关怀在 `--speed 5 --demo` 下按倍速缩放正确触发。
- 容错演练全部通过：杀掉 A → B 降级为 `absent` 并在 A 回来后自动重连；
  杀掉 B → C 自动重连，且**重连后 0 秒就拿到当前状态**（而不是干等 15 秒心跳）。
- **颜文字字形无误**：在真实平台导出 `glyphs.png` 逐个看过，32 帧全部正常，
  **没有豆腐块**；字体正确解析到 `Microsoft YaHei UI`
  （网上常见的写法 `Microsoft YaHei` 在本机不存在，会静默回退）。
  另有一条单元测试用"缺字样板"逐字比对渲染结果，把这件事钉住。
- **语音识别链路跑通（用合成语音）**：本机装了 vosk 0.3.45 +
  `vosk-model-small-cn-0.22`，喂进去的中文语音 → 16kHz PCM → vosk → 文本，
  三句话全部识别正确。
  ⚠️ 模型必须放在**不含中文的路径**下（本仓库路径含中文，vosk 加载不了），
  本机放在 `%LOCALAPPDATA%\vosk\`；详见
  [backend_B/README.md §4.8](backend_B/README.md)。
- 全部 456 个测试通过（真机平台 463 个，0 跳过）。
  这里跳过的 7 个都是 frontend_C 的字体测试：offscreen 平台渲染不出任何字形，
  它们会自己跳过（`QT_QPA_PLATFORM=windows` 下正常运行）。

**没有验证（需要你在真机上看一眼）：**

- **摄像头 + MediaPipe 真实采集**：本机跑的是合成源与离线 CSV，
  真实摄像头这条路没有验过。
- **真人对着麦克风说话**：语音识别只用 SAPI 合成的语音验证过，
  真人说话的口音、语速、环境噪声都还没试。想看这一条行不行：
  `cd backend_B && python main.py --voice`，对麦克风说一句，
  看日志里有没有正确的 `语音识别：'...'`。
- **音色好不好听、音量合不合适**：播放本身已用耳朵验收过（确实出声），
  但"这次改上去的豆包音色对不对味"只有人站在机器前能判。
  要验收：配好豆包凭证后 `python main.py --tts doubao`，听一句。
  听不见时的排查顺序见 [常见问题 3](backend_B/README.md)
  —— 先查**端点音量**，再怀疑选错设备（"默认输出是虚拟声卡"那条旧结论已更正）。

---

## Django 后端（原型脚手架）

仓库里另有一个 Django 脚手架，与上面三个 TCP 模块相互独立（当前未被它们引用），
保留作为后续 B/S 架构演进的起点。实际安装版本为 **Django 6.0.6**（Python 3.14.3）。

```bash
cd <仓库根目录>
python manage.py migrate      # 初始化数据库（默认 SQLite）
python manage.py runserver    # 开发服务器 http://127.0.0.1:8000
```

### 结构

```
intelligent_home_robot/
├── manage.py
├── intelligent_home_robot/   # 项目配置包
│   ├── settings.py           # 配置（数据库、应用注册等）
│   ├── urls.py               # 根路由
│   ├── asgi.py / wsgi.py     # 部署入口
│   └── __init__.py
```

### 常用命令

```bash
python manage.py startapp <app名>   # 新建应用
python manage.py makemigrations     # 生成迁移
python manage.py createsuperuser    # 建管理员（后台 /admin）
```

> ⚠️ 注意 `runserver` 默认也用 **8000** 端口，而模块 A 也在 8000。
> 两者不要同时启动；需要并存时给 Django 换端口：`python manage.py runserver 8888`。
>
> 仓库根的 `pytest.ini` 用显式 `testpaths` 限定了测试目录，就是为了避免
> rootdir 递归收集到这里（`settings.py` 会让收集变慢甚至报错）。
