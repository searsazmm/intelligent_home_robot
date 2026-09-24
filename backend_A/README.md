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

依赖：`opencv-contrib-python==4.13.0.90`、`mediapipe==0.10.14`、`onnxruntime==1.23.2`（性别/年龄用，可选）（见根目录 requirements.txt）。

## 输出

| 文件 | 内容 |
|---|---|
| `data/vis_data_时间戳.csv` | 协议数据：表头 `timestamp,has_face,ear,blink_cnt,pitch,yaw,roll,emo_feature`（api_doc §3.4，UTF-8 无 BOM，`\n` 换行，数值 2 位小数） |
| `data/pulse_wave_时间戳.csv` | 实验数据：`timestamp,r,g,b,a_lab,b_lab,mouth_open,hr,rr,ibi_ms,sqi`（rPPG 原始三通道/双颊 Lab 色值/口部开口度 + 派生指标 + 质量分，未进协议；`mouth_open` 为唇 13/14 开口度时序，供 B 对话状态机，协议字段见 api_doc §3.5 V1.2 草案） |
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

## 性别/年龄估计（P2 演示项，可选）

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
`max_faces` 为同时跟踪的人脸数上限（默认 3）：多人入镜时按包围盒面积锁定主脸（通常离镜头最近者），
访客短暂入镜不会抢走主人指标；单脸场景 CPU 开销不变。

## 给 B 交付样例前的检查项

1. 程序退出时打印自测摘要：**四种标签都出现过**（对着镜头分别做正常/疲惫/难过/发呆各十几秒）；
2. CSV 行数 = 有人脸帧数，表头与 api_doc §3.4 逐字一致；
3. Socket 用 `telnet 127.0.0.1 8000` 或 B 侧脚本连上能收到 JSON 流（含无人脸心跳）。
