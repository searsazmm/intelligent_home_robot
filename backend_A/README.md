# 后端 A：视觉感知模块（原型 V2）

摄像头 + MediaPipe FaceMesh，每帧计算指标，**双出口**：CSV 落盘 + TCP Socket 服务（api_doc §3）。
附加 rPPG 非接触脉搏波（心率/呼吸率/心跳间期，输出走独立波形 CSV，尚未进协议，见需求文档 §12.4）。

## 运行

```bash
cd backend_A
python vision_a.py                # 开预览窗口，q 退出，c 重新标定
python vision_a.py --seconds 10   # 采 10 秒自动退出（自测）
python vision_a.py --no-window    # 后台采数，不开窗口
python vision_a.py --no-socket    # 只写 CSV，不起 Socket
python vision_a.py --camera 1     # 指定摄像头编号（默认 0，失败自动回退 1）
python sim_b.py                   # 另开终端：模拟 B 连 8000 收流并按 api_doc §3 逐条校验
```

依赖：`opencv-contrib-python==4.13.0.90`、`mediapipe==0.10.14`（见根目录 requirements.txt）。

## 输出

| 文件 | 内容 |
|---|---|
| `data/vis_data_时间戳.csv` | 协议数据：表头 `timestamp,has_face,ear,blink_cnt,pitch,yaw,roll,emo_feature`（api_doc §3.4，UTF-8 无 BOM，`\n` 换行，数值 2 位小数） |
| `data/pulse_wave_时间戳.csv` | 实验数据：`timestamp,green,hr,rr,ibi_ms`（rPPG 波形与派生指标，未进协议） |
| TCP 127.0.0.1:8000 | 每帧一行 JSON + `\n`（字段同 CSV）；**无人脸帧照发心跳** `has_face=false`（api_doc §3.3）——B 靠心跳区分"没人"与"掉线"，B 断开自动等待重连 |

**rPPG 成熟度（诚实标注）**：HR 为**参考级**（安静场景可用，动作多时误差大）；**RR 需 20 秒窗口才输出；IBI/RR 为实验性**——
简单 FFT + 绿通道的质量极限，联调期可换 POS/CHROM 算法改进。全部输出禁止作为医疗结论（需求文档 §12.1 T3 红线）。

- 标签小写英文：`normal / tired / sad / blank`（api_doc §3.2 V1.1 枚举）。
- **无人脸帧跳过 CSV 写入**，Socket 照常发心跳。
- ⚠️ 8000 端口与 Django runserver 默认端口冲突：起 Socket 时别同时 `python manage.py runserver`（脚本检测到占用会打印警告并只出 CSV）。

## 基线标定（重要）

首次运行自动采集前 25 个有效帧（约 2 秒）的**个人基线**并保存到 `baseline.json`；
之后启动**直接加载，无需重新标定**。pitch/yaw/roll/嘴角弧度输出的是相对基线的偏移量
（解决 pitch 数值整体偏大的问题）。预览窗口按 `c` 键随时重标定，结果覆盖保存。
若加载的旧基线与当前姿态持续失配（如换人/换机位，连续 90 帧极端偏移），程序自动重采——
**自动重标仅当次生效**，落盘仍需按 `c` 人工确认，防止跌倒等异常姿态被固化成基线。
`baseline.json` 属个人派生数据，仅存本地，已加入 .gitignore 不进版本库。
**首次标定（或按 c 重标）时保持正常坐姿，不要歪头做表情。**

## 调参

所有阈值集中在 `config.json`。个体差异大时优先调：`ear_tired`（疲劳）、`sad_curvature`（难过）、
`gaze_still_th`（发呆判定灵敏度）、`pitch_scale/yaw_scale`（头姿灵敏度）。
`low_light_th` 为照度自检阈值（画面均值 0–255，默认 45）：低于阈值时控制台与预览窗口告警
"has_face=false 可能是光线问题而非无人"，供 B 侧区分"黑屋/离开/掉线"参考。
`csv_retention_days` 为采集 CSV 保留天数（默认 7）：启动时自动清理过期文件，设 0 关闭清理。

## 给 B 交付样例前的检查项

1. 程序退出时打印自测摘要：**四种标签都出现过**（对着镜头分别做正常/疲惫/难过/发呆各十几秒）；
2. CSV 行数 = 有人脸帧数，表头与 api_doc §3.4 逐字一致；
3. Socket 用 `telnet 127.0.0.1 8000` 或 B 侧脚本连上能收到 JSON 流（含无人脸心跳）。
