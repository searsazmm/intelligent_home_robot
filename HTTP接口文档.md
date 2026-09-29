# 居家陪伴机器人 HTTP 接口文档（V1.0 — B↔C 补充接口）

> 本文档定义**模块 B 向前端 C 提供的 HTTP 接口**，由 Django（仓库 `intelligent_home_robot/` 项目）实现。
>
> **与 `api_doc.md` 的关系**：`api_doc.md` V1.0 的两条 TCP 链路（A→B 8000 实时指标、B→C 8001 实时状态串）**保持不变**，继续负责低延迟实时推送；本文档的 HTTP 接口负责 TCP 传不了的富数据：历史查询、对话记录、统计日报、配置管理、事件中心。
>
> 目标：给前端、后端、测试提供一份统一的接口契约。

---

## 1. 架构定位

```
[A 视觉] --TCP 8000 JSON--> [B 业务中枢] --TCP 8001 状态串--> [C 前端]
                                │
                                └--HTTP 8020 REST---------> [C 前端（页面数据）]
                                └--> SQLite（指标/状态/事件/对话落库）
```

- **实时状态**仍走 TCP 8001（毫秒级、单向、纯文本）；C 的主窗口表情切换**不依赖** HTTP。
- **HTTP（8020）**：C 的各页面按需拉取历史/统计/配置，及发送文字消息。
- B 每次状态切换、每分钟指标汇总、每个事件发生时写入数据库，供 HTTP 查询。

## 2. 前端页面与字段设计

### 2.1 页面总览

| 页面 | 主要职责 | 依赖接口 |
|---|---|---|
| P1 主陪伴窗口 | 常驻小窗：表情、提示语、实时体征 | §4.1、§4.2（TCP 8001 继续并行推送） |
| P2 对话界面 | 聊天记录、文字输入兜底 | §5.1、§5.2 |
| P3 健康监测 | 实时体征 + 历史曲线 + 今日统计 | §6.1、§6.2、§6.3 |
| P4 状态时间线 | 一天中各状态时段分布 | §7.1 |
| P5 预警中心 | 预警事件列表、确认处理 | §7.2、§7.3 |
| P6 设置 | 问候时段、灵敏度、语音、监控开关 | §8.1、§8.2 |

### 2.2 P1 主陪伴窗口

| 字段 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| current_state | string | §4.1 / TCP 8001 | 当前状态，驱动表情图 |
| state_desc | string | §4.1 | 状态中文描述（如"状态正常"） |
| greeting_text | string | §4.1 | 机器人当前正在说的话（与 TTS 同步） |
| hr / rr | number | §6.1 | 实时心率/呼吸率（二期字段，可为 null） |
| moduleAOnline | boolean | §4.2 | 视觉后端连接状态 |
| cameraOn | boolean | §4.2 | 摄像头是否启用 |

### 2.3 P2 对话界面

| 字段 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| messages[].role | string | §5.1 | `USER` / `ROBOT` |
| messages[].content | string | §5.1 | 消息文本 |
| messages[].trigger | string | §5.1 | 触发类型（用户发起/主动问候/情绪关怀…） |
| messages[].createdAt | string | §5.1 | 时间 |
| inputText → 发送 | string | §5.2 | 文字输入（语音是主路径，文字兜底） |

### 2.4 P3 健康监测

| 字段 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| latest | object | §6.1 | 最新一批指标（ear、眨眼频率、头姿、HR/RR、专注、压力） |
| history.points[] | array | §6.2 | 指标历史序列（按 metric + 时间范围查询，画曲线） |
| summary | object | §6.3 | 今日/指定日各指标统计 + 异常次数 |

### 2.5 P4 状态时间线

| 字段 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| segments[].state | string | §7.1 | 状态段 |
| segments[].startAt / endAt / durationSec | string/number | §7.1 | 起止与时长 |
| stateRatio | object | §7.1 | 各状态当日时长占比（画饼图/条形图） |

### 2.6 P5 预警中心

| 字段 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| events[].type | string | §7.2 | 事件类型（疑似跌倒/久闭眼/体征异常…） |
| events[].level | string | §7.2 | `INFO/WARN/CRITICAL` |
| events[].occurredAt / detail | string | §7.2 | 发生时间与详情 |
| events[].status | string | §7.2 | `PENDING/ACKNOWLEDGED/RESOLVED` |
| 确认操作 | — | §7.3 | 标记已处理 |

### 2.7 P6 设置

| 字段 | 类型 | 来源 | 说明 |
| --- | --- | --- | --- |
| greetingPeriods[] | array | §8.1/§8.2 | 早/下午/晚间问候：开关 + 时段 |
| proactiveLimitPerHour | number | §8.1/§8.2 | 主动交互每小时上限 |
| sensitivity | string | §8.1/§8.2 | 灵敏度 `LOW/MID/HIGH`（映射内部阈值组） |
| voiceEnabled / volume | bool/number | §8.1/§8.2 | 语音开关与音量 |
| monitoringEnabled | bool | §8.1/§8.2 | 摄像头感知总开关 |

---

## 3. 通用约定

### 3.1 Base URL

- **本地开发**：`http://127.0.0.1:8020`
- 端口 8020 避开 TCP 协议占用的 8000/8001。

### 3.2 鉴权方式

- **本机 C 客户端免鉴权**（同机进程间通信）。
- 预留请求头 `X-API-Token: <token>`：二期家属远程查看端启用，本版不做校验。

### 3.3 统一响应格式

```json
{
  "status": 200,
  "message": "success",
  "data": {},
  "timestamp": "2026-09-22T12:00:00"
}
```

- `status`：业务状态码，成功为 `200`
- `data`：返回数据，可能为对象、数组或 `null`

### 3.4 分页响应格式

分页接口的 `data` 统一为：

```json
{
  "list": [],
  "page": 1,
  "size": 10,
  "total": 100,
  "totalPages": 10
}
```

### 3.5 时间与排序约定

- 时间字段统一 ISO 8601：`2026-09-22T12:00:00`（本地时区）
- 分页参数：`page` 从 1 开始，`size` 默认 `10`

### 3.6 请求参数示例写法

- `path`：路径参数；`query`：查询参数；`body`：JSON 请求体
- 无对应参数时写 `null` 或空对象 `{}`

### 3.7 通用枚举约定

#### 3.7.1 用户状态（与 B→C TCP 状态集一致）

- `NORMAL`（正常）、`SAD`（情绪低落）、`TIRED`（疲惫）、`ABSENT`（发呆失神）
- `WARNING`（疑似跌倒/预警）、`GREETING`（问候中）
- ⚠️ `WARNING/GREETING` 属于协议扩展提案，见需求文档 §6，需全员确认
- 与 A 模块单帧标签的映射：`normal→NORMAL`、`tired→TIRED`、`sad→SAD`、`blank→ABSENT`（A 报 `blank`，B 滑动窗口融合后对外叫 `ABSENT`，两套命名并存是约定，勿混用）

#### 3.7.2 指标类型

- `EAR`（眼睑开合度）、`BLINK_RATE`（次/分钟）、`PITCH`、`YAW`、`ROLL`
- `HR`（心率）、`RR`（呼吸率）、`FOCUS`（专注度 0-100）、`STRESS`（压力 0-100）

#### 3.7.3 事件类型

- `FALL_SUSPECT`（疑似跌倒）、`EYES_CLOSED_LONG`（长时间闭眼）
- `SIGNS_ABNORMAL`（体征异常）、`LOW_MOOD`（情绪持续低落）、`DEVICE_OFFLINE`（设备离线）

#### 3.7.4 事件等级

- `INFO` / `WARN` / `CRITICAL`

#### 3.7.5 事件处理状态

- `PENDING`（未处理）/ `ACKNOWLEDGED`（已知晓）/ `RESOLVED`（已处理完毕）

#### 3.7.6 消息角色

- `USER` / `ROBOT`

#### 3.7.7 消息触发类型

- `USER_INITIATED`（用户发起）、`PROACTIVE_GREETING`（定时问候）、`EMOTION_CARE`（情绪关怀）、`REST_REMINDER`（休息提醒）、`SAFETY_ALERT`（安全预警）

#### 3.7.8 灵敏度

- `LOW` / `MID` / `HIGH`（映射 B 内部阈值组：判定时长与阈值放宽/收紧）

---

## 4. 系统状态接口

### 4.1 接口列表

| 方法 | 路径 | 权限 | 说明 |
| --- | --- | --- | --- |
| GET | `/api/v1/status/current` | 免鉴权 | 当前状态快照（供主窗口轮询） |
| GET | `/api/v1/system/health` | 免鉴权 | 模块连接健康检查 |

### 4.2 `GET /api/v1/status/current`

- **接口中文名**：获取当前状态快照
- **请求参数示例**

```json
{
  "path": {},
  "query": {},
  "body": null
}
```

- **响应 `data`**：`StatusVO`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| state | string | 当前状态，见 §3.7.1 |
| stateDesc | string | 状态中文描述 |
| greetingText | string | 当前播报文本（无则 `null`） |
| updatedAt | string | 状态最后变更时间 |

- **错误码**

| status | message | 条件 |
| --- | --- | --- |
| 200 | success | B 运行中恒返回；B 停止时 C 直接按断连兜底 |

### 4.3 `GET /api/v1/system/health`

- **接口中文名**：获取模块健康状态
- **请求参数示例**

```json
{
  "path": {},
  "query": {},
  "body": null
}
```

- **响应 `data`**：`HealthVO`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| moduleAOnline | boolean | A→B TCP 连接是否存活 |
| cameraOn | boolean | 摄像头是否启用 |
| dbOk | boolean | 数据库读写是否正常 |
| uptimeSec | number | B 进程运行秒数 |
| serverTime | string | 服务器当前时间 |

---

## 5. 对话接口

### 5.1 接口列表

| 方法 | 路径 | 权限 | 说明 |
| --- | --- | --- | --- |
| GET | `/api/v1/chat/messages` | 免鉴权 | 分页查询会话记录 |
| POST | `/api/v1/chat/messages` | 免鉴权 | 发送文字消息并获取回复 |

### 5.2 `GET /api/v1/chat/messages`

- **接口中文名**：分页查询会话记录
- **查询参数**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| page | number | 否 | 页码，默认 `1` |
| size | number | 否 | 每页数量，默认 `20` |
| date | string | 否 | 按日期过滤，如 `2026-09-22` |
| trigger | string | 否 | 按 §3.7.7 触发类型过滤 |

- **请求参数示例**

```json
{
  "query": {
    "page": 1,
    "size": 20,
    "date": "2026-09-22"
  }
}
```

- **响应 `data`**：分页 `ChatMessageVO[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | number | 消息 ID |
| role | string | `USER` / `ROBOT` |
| content | string | 消息文本 |
| trigger | string | 触发类型（`USER` 消息恒为 `USER_INITIATED`） |
| createdAt | string | 时间 |

### 5.3 `POST /api/v1/chat/messages`

- **接口中文名**：发送文字消息（文字兜底通道）
- **请求体**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| content | string | 是 | 用户消息文本，1-500 字符 |

- **请求参数示例**

```json
{
  "body": {
    "content": "今天有点闷，不想动。"
  }
}
```

- **响应 `data`**：机器人回复

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| userMessageId | number | 落库的用户消息 ID |
| reply | object | 机器人回复：`{ id, role: "ROBOT", content, trigger, createdAt }` |
| repliedBy | string | `LLM` / `FALLBACK`（本地话术库降级） |

- **错误码**

| status | message | 条件 |
| --- | --- | --- |
| 400 | 消息内容不能为空 | content 为空或超 500 字 |
| 504 | 对话引擎超时 | LLM 超时且本地话术也未返回（正常情况降级不报错） |

**业务规则**：同步返回（后端内部等待对话引擎，上限 10 秒后走降级话术）；该消息同时由 B 走 TTS 播报。

---

## 6. 健康指标接口

### 6.1 接口列表

| 方法 | 路径 | 权限 | 说明 |
| --- | --- | --- | --- |
| GET | `/api/v1/metrics/latest` | 免鉴权 | 最新一批指标 |
| GET | `/api/v1/metrics/history` | 免鉴权 | 指标历史序列 |
| GET | `/api/v1/metrics/summary` | 免鉴权 | 指标按日统计 |

### 6.2 `GET /api/v1/metrics/latest`

- **接口中文名**：获取最新指标
- **请求参数示例**

```json
{
  "path": {},
  "query": {},
  "body": null
}
```

- **响应 `data`**：`MetricsLatestVO`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| hasFace | boolean | 当前是否检测到人脸 |
| ear | number | 眼睑开合度（无脸为 `null`，下同） |
| blinkRate | number | 每分钟眨眼次数 |
| pitch / yaw / roll | number | 头部姿态角 |
| hr / rr | number | 心率/呼吸率（二期，未启用恒为 `null`） |
| focusScore | number | 专注度 0-100 |
| stressLevel | number | 压力 0-100 |
| sampledAt | string | 采样时间 |

### 6.3 `GET /api/v1/metrics/history`

- **接口中文名**：查询指标历史序列
- **查询参数**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| metric | string | 是 | 指标类型，见 §3.7.2（单次一个） |
| from | string | 是 | 起始时间 |
| to | string | 是 | 结束时间 |
| interval | number | 否 | 采样间隔秒数，默认 `60`（后端降采样） |

- **请求参数示例**

```json
{
  "query": {
    "metric": "HR",
    "from": "2026-09-22T08:00:00",
    "to": "2026-09-22T12:00:00",
    "interval": 300
  }
}
```

- **响应 `data`**：`MetricsSeriesVO`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| metric | string | 指标类型 |
| points[] | array | `{ "t": "2026-09-22T08:05:00", "v": 72.5 }` 序列 |
| count | number | 点数 |

- **错误码**

| status | message | 条件 |
| --- | --- | --- |
| 400 | 不支持的指标类型 | metric 不在枚举内 |
| 400 | 时间范围无效 | from ≥ to 或跨度超过 7 天 |

### 6.4 `GET /api/v1/metrics/summary`

- **接口中文名**：按日统计指标
- **查询参数**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| date | string | 否 | 统计日期，默认今天 |

- **响应 `data`**：`MetricsSummaryVO`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| date | string | 统计日期 |
| metrics | object | 每个指标的 `{ avg, min, max }` |
| abnormalCount | number | 体征/状态异常计数 |
| monitorMinutes | number | 有效监测时长（分钟） |

---

## 7. 状态历史与事件接口

### 7.1 接口列表

| 方法 | 路径 | 权限 | 说明 |
| --- | --- | --- | --- |
| GET | `/api/v1/states/timeline` | 免鉴权 | 状态时间线（按日） |
| GET | `/api/v1/events` | 免鉴权 | 分页查询事件 |
| PATCH | `/api/v1/events/{id}/ack` | 免鉴权 | 更新事件处理状态 |

### 7.2 `GET /api/v1/states/timeline`

- **接口中文名**：查询状态时间线
- **查询参数**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| date | string | 否 | 日期，默认今天 |

- **请求参数示例**

```json
{
  "query": {
    "date": "2026-09-22"
  }
}
```

- **响应 `data`**：`TimelineVO`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| date | string | 日期 |
| segments[] | array | `{ "state": "TIRED", "startAt": "...", "endAt": "...", "durationSec": 1200 }` |
| stateRatio | object | 各状态秒数占比，如 `{ "NORMAL": 0.72, "TIRED": 0.18, ... }` |

### 7.3 `GET /api/v1/events`

- **接口中文名**：分页查询事件
- **查询参数**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| page / size | number | 否 | 分页，默认 `1` / `10` |
| type | string | 否 | 事件类型，见 §3.7.3 |
| level | string | 否 | `INFO/WARN/CRITICAL` |
| status | string | 否 | 见 §3.7.5 |
| dateFrom / dateTo | string | 否 | 发生时间范围 |

- **请求参数示例**

```json
{
  "query": {
    "page": 1,
    "size": 10,
    "type": "FALL_SUSPECT",
    "status": "PENDING"
  }
}
```

- **响应 `data`**：分页 `EventVO[]`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | number | 事件 ID |
| type | string | 事件类型 |
| level | string | 事件等级 |
| detail | string | 事件详情文本（含触发时刻的指标快照摘要） |
| occurredAt | string | 发生时间 |
| status | string | 处理状态 |
| acknowledgedAt | string | 处理时间（未处理为 `null`） |

### 7.4 `PATCH /api/v1/events/{id}/ack`

- **接口中文名**：更新事件处理状态
- **Path 参数**

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| id | number | 是 | 事件 ID |

- **请求体**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| status | string | 是 | `ACKNOWLEDGED` / `RESOLVED` |

- **请求参数示例**

```json
{
  "path": {
    "id": 12
  },
  "body": {
    "status": "RESOLVED"
  }
}
```

- **响应 `data`**：更新后的 `EventVO`

- **错误码**

| status | message | 条件 |
| --- | --- | --- |
| 404 | 事件不存在 | id 无效 |

---

## 8. 设置接口

### 8.1 接口列表

| 方法 | 路径 | 权限 | 说明 |
| --- | --- | --- | --- |
| GET | `/api/v1/settings` | 免鉴权 | 读取当前设置 |
| PUT | `/api/v1/settings` | 免鉴权 | 更新设置（全量提交） |

### 8.2 `GET /api/v1/settings`

- **接口中文名**：读取设置
- **响应 `data`**：`SettingsVO`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| greetingPeriods[] | array | `{ "period": "MORNING", "enabled": true, "start": "07:30", "end": "09:30" }`，period 枚举 `MORNING/AFTERNOON/EVENING` |
| proactiveLimitPerHour | number | 主动交互每小时上限，1-10，默认 3 |
| sensitivity | string | `LOW/MID/HIGH`，默认 `MID` |
| voiceEnabled | boolean | TTS 开关，默认 true |
| volume | number | 音量 0-100，默认 60 |
| monitoringEnabled | boolean | 摄像头感知总开关，默认 true |

### 8.3 `PUT /api/v1/settings`

- **接口中文名**：更新设置
- **请求体**：与 `SettingsVO` 相同结构，全量提交

- **请求参数示例**

```json
{
  "body": {
    "greetingPeriods": [
      { "period": "MORNING", "enabled": true, "start": "07:30", "end": "09:30" },
      { "period": "AFTERNOON", "enabled": false, "start": "14:00", "end": "16:00" },
      { "period": "EVENING", "enabled": true, "start": "18:00", "end": "20:00" }
    ],
    "proactiveLimitPerHour": 3,
    "sensitivity": "MID",
    "voiceEnabled": true,
    "volume": 60,
    "monitoringEnabled": true
  }
}
```

- **响应 `data`**：更新后的 `SettingsVO`。B 的规则引擎在 1 秒内热加载新配置，无需重启。

- **错误码**

| status | message | 条件 |
| --- | --- | --- |
| 400 | 设置项不合法 | 时段格式错误、上限越界等 |

---

## 9. 错误码总表

| 业务码 | HTTP 状态 | 说明 |
| --- | --- | --- |
| 200 | 200 | 成功 |
| 400 | 400 | 请求参数错误 |
| 404 | 404 | 资源不存在 |
| 422 | 422 | 业务校验失败 |
| 500 | 500 | 服务器内部错误 |
| 504 | 504 | 上游（对话引擎等）超时 |

**C 端兜底原则**（与 api_doc §4.3 一致）：任何 HTTP 异常、超时、断连，页面显示最近一次缓存数据并标注"数据可能过期"，不得崩溃。

---

## 10. 开发说明

- HTTP 服务由 Django 实现（现有 `intelligent_home_robot/` 项目），与 B 的业务进程**同机部署**；Django 视图层直接读 B 落库的 SQLite，不重复实现业务逻辑。
- B 的落库时机：状态切换事件、事件触发、指标按分钟聚合、对话消息收发，各自写库，写库失败只记日志不影响实时链路。
- 建议二期将 `GET /status/current` 升级为 WebSocket `/ws/status` 推送，减少轮询；本版轮询间隔建议 2 秒。
- 后续若落 Swagger，可将本文档转换为 OpenAPI 3.0 YAML，便于联调与 Mock。
