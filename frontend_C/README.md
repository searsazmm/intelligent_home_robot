# 模块 C —— 前端界面

一个**纯黑背景的窗口，上面只有一张白色的颜文字脸**。

按需求刻意**没有**输入框、没有聊天记录、没有按钮、没有任何文字。
用户全程用嘴说话（语音在模块 B），屏幕上只回馈「机器人现在什么表情」。
连接状态只出现在**标题栏**和日志里。

---

## 1. 快速开始

```bash
cd frontend_C

pip install -r requirements.txt      # 只有 PyQt5 一行

python main.py                       # 连接 127.0.0.1:8002
python main.py --fullscreen          # 全屏无边框（答辩用）
python main.py --demo                # 不连模块 B，四态循环（兜底演示）
python main.py --dump-glyphs         # 字形自检：看不到方块即为正常
```

**模块 B 没启动也能跑** —— 界面照常显示正常表情，后台指数退避重连，
B 一起来就自动接上。所以启动顺序反了不会坏，只是看不到表情变化。

跑测试（**不需要 pytest**，标准库 unittest 即可）：

```bash
python tests/test_expressions.py     # 23 个：颜文字表与状态规整
python tests/test_state_client.py    # 31 个：真 TCP 连接、分帧、重连、通道优先级
python tests/test_window.py          # 26 个（离屏）/ 33 个（真机平台）：纯黑底、渲染、字形、按键、关闭清理

# ⚠️ 字形断言（含"无豆腐块"）只在字体库非空的平台执行。
#    想在真机上跑（断言更强，0 skipped）：
QT_QPA_PLATFORM=windows python tests/test_window.py
```

---

## 2. 端口与连接方向

| 链路 | 端口 | C 的角色 | 数据格式 | 默认 |
| ---- | ---- | -------- | -------- | ---- |
| B ↔ C 对话 | 127.0.0.1:8002 | **客户端**（connect） | 一行一条 JSON，`\n` 分隔 | ✅ 连 |
| B → C 状态 | 127.0.0.1:8001 | **客户端**（connect） | 一行一个纯文本状态词 | ❌ 不连（`--also-status` 开） |

⚠️ 在两条链路上 **C 都是 connect 的一方**，B 才是 listen 的一方。
写反了会得到一个「连不上但也不知道为什么」的界面。

### 为什么默认只连 8002

两条通道都会推同一个状态，于是有两个互相独立的视图可能不一致：
B 在收到对话时会立刻往 8001 推，而 8002 要等下一个 0.2 秒的发布节拍 ——
**表情会来回抖**。

一个状态源、一个去重点，是更稳的选择。`--also-status` 是给「演示 api_doc §4
的专用状态通道」用的，此时 8001 是权威源，8002 的状态推送会被忽略（有测试钉住）。

---

## 3. 目录结构

```
frontend_C/
├── main.py              入口：命令行、日志、自检、装配、截图导出
├── requirements.txt     只有 PyQt5 一行（为什么，见文件里的注释）
├── c_core/              纯逻辑层，**不 import Qt**，可脱离图形环境单测
│   ├── expressions.py   颜文字表 + normalize_state（api_doc §4.3 的落点）
│   ├── protocol.py      \n 分帧（backend_B/core/protocol.py 的刻意副本，见 §7）
│   └── state_client.py  连 B、断线重连、状态去重与优先级
├── ui/                  Qt 层，只有这一层依赖 PyQt5
│   └── window.py        纯黑窗口、颜文字绘制、眨眼与轮换动画、字形自检窗口
└── tests/
    ├── test_expressions.py   23 个
    ├── test_state_client.py  31 个（起真 TCP 假服务端，不打桩 socket）
    └── test_window.py        26 个（默认离屏；真机平台下断言更强）
```

---

## 4. 界面实现里的两个决定

### 4.1 不用 `QLabel`，直接在 `paintEvent` 里画

需求是「纯黑底 + 白色颜文字」。用 `QLabel` 需要额外写
`background:transparent` 去压掉样式表的背景继承，而且它是按**字形包围盒**
居中的 —— 颜文字的包围盒上下留白很不对称（`(￣▽￣)` 和 `(╥﹏╥)` 高度差很多），
按包围盒居中的结果是**每换一个表情整张脸就上下跳一下**。

直接绘制则按「整行高度 + 基线」定位，换表情时脸是稳的。

### 4.2 从不透明度和去重两个地方防「抖」

两个都是「功能都对、但演示时肉眼看得出来」的缺陷：

- **状态先做去重再动动画。** B 每 15 秒心跳重发同一个状态；
  不去重的话表情每 15 秒会说不出所以然地闪一下。见
  [state_client.py](c_core/state_client.py) 的 `_apply_state`。
- **状态切换时 `set_state()` 会提前返回**，同一状态不重启淡入动画。

---

## 5. 字体：实测结果，不是推测

颜文字大量使用全角/CJK 字形：`＾` (U+FF3E)、`￣` (U+FFE3)、`﹏` (U+FE4F)、
`－` (U+FF0D)、`╥` (U+2565)、`・` (U+30FB)、`◇` (U+25C7)。

**本机实测（Windows 11 / Python 3.14.3）：**

| 字体 | 是否存在 |
| ---- | -------- |
| `Microsoft YaHei` | ❌ **不存在** —— 而这是网上最常见的写法 |
| `SimSun` / `SimHei` / `KaiTi` | ❌ 都不存在 |
| **`Microsoft YaHei UI`** | ✅ 存在，**首选** |
| `MS Gothic` · `Yu Gothic UI` · `Segoe UI Symbol` | ✅ 存在，候选 |

所以 [window.py](ui/window.py) 的 `pick_font_family()` 在**运行时探测**
`QFontDatabase()`，按优先级取第一个可用的，并用 `QFont.setFamilies()`
给出**回退链**（某个字体缺个别字形时 Qt 会继续往下找，而不是直接画豆腐块）。

> 写死 `Microsoft YaHei` 不会报错，只会静默回退到无衬线体，
> 然后字符悄悄变成方块。所以这里不能硬编码。

### 答辩前请在演示机上跑一次 `--dump-glyphs`

Qt 的字体回退会让「某个字形缺失」这件事**不报错**。所以务必**肉眼**确认：

```bash
python main.py --dump-glyphs
```

会把全部 16 个表情 + 16 个眨眼帧连同每个字符的码位平铺出来，
没有方块即为正常。**这件事必须在演示机上做**，换机器就可能换字体。

---

## 6. 命令行参数

```
python main.py [选项]

  --host HOST            模块 B 的地址（默认 127.0.0.1）
  --chat-port N          对话通道端口（默认 8002，api_doc §5）
  --status-port N        状态通道端口（默认 8001，api_doc §4）
  --also-status          额外连 8001（默认只连 8002，见 §2）

  --font-size N          颜文字字号（磅）。默认按窗口高度自适应
  --fullscreen           无边框全屏。ESC 退出全屏，Q 退出程序
  --debug-hud            左上角叠一行调试灰字（默认关）
  --log-file PATH        同时写日志到文件（全屏时看不到控制台，建议加）
  --log-level LEVEL      DEBUG / INFO / WARNING / ERROR

  --demo                 不连 B，四态循环切换（答辩兜底）

  --dump-glyphs          字形自检窗口
  --screenshot DIR       四态各导出一张 PNG 到 DIR，然后退出
```

`--screenshot` 会导出 `face_{状态}.png`、`face_{状态}_blink.png`
和一个 `glyphs.png`，**可以直接放进答辩 PPT**。

---

## 7. 为什么 `c_core/protocol.py` 是 `backend_B` 的副本

不是疏忽，是三个理由：

1. 三个模块之间**只许走 Socket**（api_doc §1）。C 直接 import B 的包，
   等于在架构上把两个模块粘死 —— B 换语言或 C 换机器都立刻崩。
2. C 必须能**单独运行**：可能只把 `frontend_C/` 拷到演示机上。
3. C 的单元测试不该要求 `backend_B` 出现在 `sys.path` 上。

同步风险很低：分帧规则由 api_doc §2.1 冻结（`\n` 分隔）。
**如果哪天真的改了分帧规则，两个文件必须一起改。**

### 7.1 那为什么这个包叫 `c_core` 而不是 `core`

因为 `backend_B` 也有一个顶层包叫 `core`，而**两个同名的顶层包不可能共存** ——
`sys.modules` 按顶层包名索引，先导入的那个会一直占着这个名字。

后果是 `pytest backend_B/tests frontend_C/tests` 直接收集失败：

```
ModuleNotFoundError: No module named 'core.expressions'
```

这**不是** pytest 配置问题，任何 `--import-mode` 都解决不了；用 `conftest.py`
在收集时动态换掉 `sys.modules["core"]` 也试过，同样不行 ——
pytest 会把两个目录的测试模块**全部导入完才开始执行**，
所以运行期 `core` 只能指向其中一方，另一方的 `mock.patch("core.…")`
这类字符串目标就会解析到错误的模块上去。

改名是唯一不留暗坑的解法。`c_core` 里的 `c` 就是模块 C，
和 `backend_A` / `backend_B` / `frontend_C` 的字母命名一致。

---

## 8. 联调步骤

按 api_doc §6.2 的顺序：**先 A → 再 B → 最后 C**。
（顺序其实不敏感，B 和 C 都会自动重连，但按文档来最省事。）

```bash
# 终端1：模块 A（无摄像头也能跑）
cd backend_A && python main.py --scenario sad

# 终端2：模块 B
cd backend_B && python main.py

# 终端3：模块 C
cd frontend_C && python main.py --log-file logs/frontend_C.log
```

| 剧本 | 预期 C 的颜文字 |
| ---- | --------------- |
| （默认） | `(＾▽＾)` 等正常脸轮换 |
| `--scenario sad` | 约 30 秒后切成 `(T＿T)` 等低落脸 |
| `--scenario drowsy` | 切成 `(=_=)` 等疲惫脸 |
| `--scenario no_face` | 切成 `(・_・?)` 等走神脸 |

**不需要摄像头、不需要模块 B** 的最小验证：

```bash
cd frontend_C
python main.py --demo              # 四态自己轮换，用来确认界面本身没问题
python main.py --dump-glyphs       # 确认字形没有方块
```

---

## 9. 验证状态（请如实阅读）

### 已验证

| 内容 | 怎么验的 |
| ---- | -------- |
| **背景纯黑** | 单元测试逐状态、含眨眼帧，断言四角与顶边中点像素为 `(0,0,0)` |
| **白色颜文字能渲染** | 单元测试：四态各自断言非黑像素计数 > 0 |
| **无豆腐块（字形缺失）** | ① 单元测试：拿一个保证缺字的码位渲染出"豆腐块样板"，逐个比对表里每个字符的**渲染结果**是否与之相同（比问字体库可靠 —— Qt 有字体回退，`inFontUcs4` 说"没有"的字常常其实画得出来）；② 在**真实 windows 平台**导出 `glyphs.png`，肉眼确认全部 32 帧 |
| 四种状态在屏幕上**互不相同** | 单元测试：四态的非黑像素计数两两不等 |
| 眨眼帧与睁眼帧在屏幕上不同 | 单元测试：像素计数不等 |
| **`snapshot()` 绕开淡入动画** | 单元测试：先断言直接 `grab()` 是黑的，再断言 `snapshot()` 不是 |
| 收起 `set_state()` 的去重 | 单元测试：重复设同一状态不重启淡入动画 |
| 未知状态回落 normal | 单元测试：`"angry"` / `""` / `None` / `123` / `["sad"]` 全部回 normal 且不抛 |
| **粘包 / 半包 / `\r\n`** | 单元测试：真 TCP 假服务端，把一条 JSON 切成两段发、三条挤在一包里发 |
| **断线重连、B 后启动也能接上** | 单元测试：掐断连接后断言出现**第 2 次**连接并恢复收报 |
| 两条通道的优先级 | 单元测试：8001 活着时 8002 的状态被忽略；8001 端口无人监听时 8002 说了算 |
| 脏报文不杀死连接线程 | 单元测试：非 JSON、空行、顶层是数组、`state` 是列表，之后仍能正常收报 |
| `stop()` 幂等、不泄漏线程 | 单元测试：连调两次不抛；`stop()` 后线程数回落 |
| 33 个窗口用例在**真机平台**全过 | `QT_QPA_PLATFORM=windows python tests/test_window.py` → `OK`（0 skipped） |

### 未验证 —— 请勿当作已完成

| 内容 | 为什么没验证 | 怎么补 |
| ---- | ------------ | ------ |
| **长时间挂在屏幕上的表现** | 没跑过数小时量级的浸泡测试 | 挂一晚上，看内存与是否卡顿 |
| 全屏**无边框**下的实际观感 | 单测跑在离屏/窗口模式，没进真全屏 | 答辩前用 `--fullscreen` 亲自看一次 |
| 高 DPI（150%/200% 缩放）下的观感 | 本机是 100% | 在缩放非 100% 的机器上跑 `--dump-glyphs` |
| 投影仪上的实际可读性 | 没有投影仪 | 字号可用 `--font-size` 调大 |

---

## 10. 已知问题

### `PyQt5` 在本机曾处于**残缺安装**状态

现象：`import PyQt5` **成功**，但 `from PyQt5.QtWidgets import ...` 报
`ModuleNotFoundError`。原因是 `site-packages\PyQt5\` 下只有 `Qt5/` 和一个
空的 `bindings/`，没有任何 `QtCore`/`QtWidgets` 的 `.pyd`，
而 `pip show PyQt5` 报「Package(s) not found」。

修法与**正确**的验证方式：

```bash
pip install --force-reinstall PyQt5

# 必须用这条验证。`import PyQt5` 在残缺安装上也会成功，是假阳性。
python -c "from PyQt5.QtWidgets import QApplication; print('OK')"
```

`main.py` 已经在导入失败时打印这段指引，而不是甩一个裸 traceback。

### 离屏平台画不出任何字（这是平台特性，不是缺陷）

`QT_QPA_PLATFORM=offscreen` 下 `QFontDatabase().families()` 返回 **0 个字体族**，
Qt 一个字都画不出来，截图是**全黑**的。所以 `tests/test_window.py` 分成两档：
「背景纯黑 / 尺寸 / 不崩溃」任何平台都测，「真切出白色的脸」只在字体库非空的
平台测（否则跳过并说明原因）。

**不要**为了让离屏测试也绿而去放宽字形断言 —— 那等于把唯一能发现
「字体选错导致满屏豆腐块」的测试删掉。要在真机平台验证就加
`QT_QPA_PLATFORM=windows`。

---

## 11. 降级矩阵

| 缺什么 / 出什么事 | 会怎样 | 怎么看出来 |
| --- | --- | --- |
| **模块 B 没启动** | 界面照常显示正常脸，后台退避重连 | 标题栏显示「断开」；日志提示重连 |
| **B 中途挂了** | 自动重连；状态回落 `normal`（api_doc §4.3） | 表情变回正常脸，随后自动恢复 |
| **8001 连不上** | 不影响主流程，8002 成为唯一状态源 | 日志重试 8001；表情照常变化 |
| **收到的状态是未知值** | 显示 `normal`，不崩不空白 | 日志 `状态 → normal` |
| **B 发来脏报文** | 跳过该条，连接保持 | 日志「收到非法报文」 |
| **没有候选字体** | 回退到 Qt 默认字体并**警告** | 日志 warning；用 `--dump-glyphs` 复核 |
| **全屏后想退出** | ESC 退出全屏，Q 退出程序 | — |
| **控制台被全屏挡住** | `--log-file` 写文件；`--debug-hud` 叠在角上 | — |
