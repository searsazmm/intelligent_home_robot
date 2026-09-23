"""
模块 A：视觉感知（原型 V2）
============================
打开摄像头 → MediaPipe FaceMesh 提取 468 关键点 → 计算 EAR / 头部姿态（基线标定后）/
眨眼计数 / rPPG 心率呼吸 → 单帧 if-else 打标签 → 双出口输出：
  1. CSV（每帧一行，字段严格按 api_doc.md §3.4；无人脸帧跳过）
  2. Socket TCP 服务端（api_doc §3：127.0.0.1:8000，每帧 JSON+\n；无人脸照发心跳）

标签枚举（小写英文，禁止改动）：normal / tired / sad / blank
CSV 表头（列序固定，禁止改动）：timestamp,has_face,ear,blink_cnt,pitch,yaw,roll,emo_feature
rPPG 波形单独写 data/pulse_wave_*.csv（hr/rr/ibi 尚未进协议，见需求文档 §12.4）

运行：
    python vision_a.py                # 开预览窗口，q 退出，c 重新基线标定
    python vision_a.py --seconds 10   # 运行 10 秒自动退出（自测用）
    python vision_a.py --no-window    # 不开预览窗口
    python vision_a.py --no-socket    # 只写 CSV，不起 Socket 服务
"""

import argparse
import csv
import json
import math
import socket
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from rppg import Rppg

BASE_DIR = Path(__file__).resolve().parent
CSV_HEADER = ["timestamp", "has_face", "ear", "blink_cnt", "pitch", "yaw", "roll", "emo_feature"]
WAVE_HEADER = ["timestamp", "green", "hr", "rr", "ibi_ms"]

# FaceMesh 关键点索引（468 点 + refine 后的虹膜 468~477）
L_EYE = dict(h1=33, h2=133, t1=159, b1=145, t2=158, b2=153)   # 左眼
R_EYE = dict(h1=362, h2=263, t1=386, b1=374, t2=385, b2=380)  # 右眼
NOSE_TIP, FACE_TOP, FACE_BOTTOM = 1, 10, 152
CHEEK_L, CHEEK_R = 234, 454
EYE_OUTER_L, EYE_OUTER_R = 33, 263
IRIS_L = 468
BROW_L, BROW_R = 105, 334
MOUTH_CORNER_L, MOUTH_CORNER_R = 61, 291
LIP_UP, LIP_DOWN = 13, 14

_t0 = time.monotonic()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _d(a, b) -> float:
    """两个归一化关键点的欧氏距离。"""
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_ratio(lm, eye: dict) -> float:
    """单眼开合度 EAR：上下睑距 / 睑裂宽。"""
    vert = _d(lm[eye["t1"]], lm[eye["b1"]]) + _d(lm[eye["t2"]], lm[eye["b2"]])
    horiz = _d(lm[eye["h1"]], lm[eye["h2"]])
    return vert / (2.0 * horiz) if horiz > 1e-6 else 0.0


def head_angles(lm, cfg: dict):
    """头部姿态几何近似（单位：度。输出为相对个人基线的偏移量，见 Calibrator）。

    pitch/yaw：鼻尖相对面部几何中心的偏移比例 × 缩放系数；
    roll：左右眼外角连线倾角。粗估即可，判决阈值都在 B 侧。
    """
    face_w = _d(lm[CHEEK_L], lm[CHEEK_R])
    face_h = _d(lm[FACE_TOP], lm[FACE_BOTTOM])
    if face_w < 1e-6 or face_h < 1e-6:
        return 0.0, 0.0, 0.0
    cx = (lm[CHEEK_L].x + lm[CHEEK_R].x) / 2.0
    cy = (lm[FACE_TOP].y + lm[FACE_BOTTOM].y) / 2.0
    pitch = (lm[NOSE_TIP].y - cy) / face_h * cfg["pitch_scale"]
    yaw = (lm[NOSE_TIP].x - cx) / face_w * cfg["yaw_scale"]
    roll = math.degrees(math.atan2(lm[EYE_OUTER_R].y - lm[EYE_OUTER_L].y,
                                   lm[EYE_OUTER_R].x - lm[EYE_OUTER_L].x))
    return pitch, yaw, roll


class Calibrator:
    """个人基线标定：启动后采前 N 个有效帧的 pitch/yaw/roll/嘴角弧度取均值，
    之后所有输出都是相对基线的偏移量（解决 pitch 数值整体偏大的标定问题）。
    按 'c' 键可随时重新标定。"""

    def __init__(self, n: int = 25):
        self.n = n
        self._buf = []
        self.baseline = None   # (pitch, yaw, roll, curvature)

    def feeding(self) -> bool:
        return self.baseline is None

    def progress(self) -> int:
        return min(len(self._buf), self.n)

    def recalibrate(self):
        self._buf = []
        self.baseline = None

    def update(self, pitch, yaw, roll, curvature):
        """返回 (pitch, yaw, roll, curvature_delta)；标定中返回 None。"""
        if self.baseline is not None:
            return (round(pitch - self.baseline[0], 2), round(yaw - self.baseline[1], 2),
                    round(roll - self.baseline[2], 2), curvature - self.baseline[3])
        self._buf.append((pitch, yaw, roll, curvature))
        if len(self._buf) >= self.n:
            self.baseline = tuple(sum(col) / len(col) for col in zip(*self._buf))
            print("[A] 基线标定完成：pitch={:.2f} yaw={:.2f} roll={:.2f}".format(*self.baseline[:3]))
        return None


def mouth_curvature(lm, face_h: float) -> float:
    """嘴角相对唇中垂线偏移（正=嘴角下垂，疑似难过；标定后为相对偏移）。"""
    if face_h < 1e-6:
        return 0.0
    corner_y = (lm[MOUTH_CORNER_L].y + lm[MOUTH_CORNER_R].y) / 2.0
    center_y = (lm[LIP_UP].y + lm[LIP_DOWN].y) / 2.0
    return (corner_y - center_y) / face_h


class Labeler:
    """单帧标签判定：tired > sad > blank > normal（含眨眼状态机与视线静止检测）。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.blink_cnt = 0
        self._eye_closed = False
        self._blink_times = deque()          # 近 N 秒眨眼时刻，用于频率判断
        self._gaze = deque()                 # (t, iris_x) 视线静止检测

    def _update_blink(self, ear: float) -> None:
        c = self.cfg
        if not self._eye_closed and ear < c["ear_closed"]:
            self._eye_closed = True
            self.blink_cnt += 1
            self._blink_times.append(time.monotonic())
        elif self._eye_closed and ear > c["ear_open"]:
            self._eye_closed = False
        while self._blink_times and time.monotonic() - self._blink_times[0] > c["blink_rate_window_sec"]:
            self._blink_times.popleft()

    def _gaze_still(self, lm, face_w: float) -> bool:
        c = self.cfg
        now = time.monotonic()
        if face_w > 1e-6:
            self._gaze.append((now, lm[IRIS_L].x / face_w))
        while self._gaze and now - self._gaze[0][0] > c["gaze_still_sec"]:
            self._gaze.popleft()
        if len(self._gaze) < 10:             # 样本不足不做判断
            return False
        xs = [x for _, x in self._gaze]
        return (max(xs) - min(xs)) < c["gaze_still_th"]

    def make_label(self, ear: float, curvature_delta: float, lm, face_w: float) -> str:
        c = self.cfg
        self._update_blink(ear)
        # 1) tired：眼睑低于阈值，或近窗口内眨眼过于频繁
        if ear < c["ear_tired"] or len(self._blink_times) >= c["blink_rate_tired"]:
            return "tired"
        # 2) sad：嘴角相对个人基线下垂超过阈值
        if curvature_delta > c["sad_curvature"]:
            return "sad"
        # 3) blank：双眼睁开但视线长时间无位移
        if ear > c["ear_open"] and self._gaze_still(lm, face_w):
            return "blank"
        return "normal"


class SocketServer:
    """api_doc §3：A 为 TCP 服务端（127.0.0.1:8000），B 连入后每帧一行 JSON+\n。
    无人脸帧照发心跳（has_face=false，其余字段默认值）——B 靠心跳区分"没人"与"掉线"。
    发送失败即断开，等待 B 重连；本线程不阻塞摄像头主循环。"""

    def __init__(self, host: str, port: int):
        self._lock = threading.Lock()
        self._conn = None
        self._thread = threading.Thread(target=self._serve, args=(host, port), daemon=True)
        self._thread.start()

    def _serve(self, host, port):
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            print(f"[A] Socket 服务已启动：{host}:{port}（等待 B 连入）")
        except OSError as e:
            print(f"[A] ⚠️ Socket 启动失败：{e}（端口 {port} 可能被占用，"
                  f"比如开着 Django runserver；本次只出 CSV）")
            return
        while True:
            conn, addr = srv.accept()
            with self._lock:
                self._conn = conn
            print(f"[A] B 已连入：{addr}")

    def send(self, metrics: dict) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.sendall((json.dumps(metrics) + "\n").encode("utf-8"))
            except OSError:
                try:
                    self._conn.close()
                except OSError:
                    pass
                self._conn = None
                print("[A] B 断开，等待重连")


def open_camera(cfg: dict, index_override=None):
    """按 0 → 1 顺序尝试打开摄像头（Windows 用 DSHOW 后端，避免启动卡顿）。"""
    indices = [index_override] if index_override is not None else [cfg["camera_index"], cfg["camera_fallback_index"]]
    for idx in indices:
        flag = cv2.CAP_DSHOW if cfg.get("use_dshow") else 0
        cap = cv2.VideoCapture(idx, flag)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[A] 摄像头已打开：index={idx}")
                return cap
        cap.release()
    raise RuntimeError("摄像头打开失败：已尝试 index {}，请检查设备占用或改用 --camera 指定编号".format(indices))


def main():
    ap = argparse.ArgumentParser(description="模块 A 视觉感知（CSV + Socket）")
    ap.add_argument("--config", default=str(BASE_DIR / "config.json"))
    ap.add_argument("--camera", type=int, default=None, help="覆盖配置中的摄像头编号")
    ap.add_argument("--seconds", type=float, default=None, help="运行 N 秒后自动退出（自测用）")
    ap.add_argument("--no-window", action="store_true", help="不显示预览窗口")
    ap.add_argument("--no-socket", action="store_true", help="不起 Socket 服务，只写 CSV")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    labeler = Labeler(cfg)
    calibrator = Calibrator(cfg.get("calib_frames", 25))
    rppg = Rppg(cfg.get("rppg_hr_window", 8.0), cfg.get("rppg_rr_window", 20.0)) if cfg.get("rppg_enabled", True) else None

    csv_dir = BASE_DIR / cfg["csv_dir"]
    csv_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    csv_path = csv_dir / f"vis_data_{stamp}.csv"
    wave_path = csv_dir / f"pulse_wave_{stamp}.csv"
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")     # UTF-8 无 BOM
    writer = csv.writer(csv_file, lineterminator="\n")               # \n 换行
    writer.writerow(CSV_HEADER)
    wave_file = open(wave_path, "w", encoding="utf-8", newline="")
    wave_writer = csv.writer(wave_file, lineterminator="\n")
    wave_writer.writerow(WAVE_HEADER)
    print(f"[A] CSV 输出：{csv_path}")
    print(f"[A] 波形 CSV：{wave_path}")

    sock = None
    if cfg.get("socket_enabled", True) and not args.no_socket:
        sock = SocketServer(cfg.get("socket_host", "127.0.0.1"), cfg.get("socket_port", 8000))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = open_camera(cfg, args.camera)
    delay = max(1, int(1000 / cfg["target_fps"]))
    low_light_th = float(cfg.get("low_light_th", 45.0))   # 画面均值低于此值视为照度不足
    low_light = False
    low_light_cnt = 0
    label_stat = {"normal": 0, "tired": 0, "sad": 0, "blank": 0}
    total, with_face, no_face = 0, 0, 0

    def heartbeat(now: float) -> None:
        """无人脸/标定中的心跳帧（api_doc §3.3：不中断发包）。"""
        if sock:
            sock.send({"timestamp": round(time.monotonic() - _t0, 2), "has_face": False,
                       "ear": 0.0, "blink_cnt": labeler.blink_cnt,
                       "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "emo_feature": "normal"})

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[A] 读取画面失败，退出（摄像头可能被占用或被拔出）")
                break
            total += 1
            frame = cv2.flip(frame, 1)  # 镜像，自拍视角

            # 照度自检：黑屋/拉窗帘时 FaceMesh 检不出人脸，B 会把"太暗"误判成"无人"
            lum = float(frame.mean())
            is_low = lum < low_light_th
            if is_low:
                low_light_cnt += 1
            if is_low and not low_light:
                print(f"[A] ⚠️ 照度不足：画面均值 {lum:.0f} < {low_light_th:.0f}，"
                      f"人脸检测可能失效（has_face=false 可能是光线问题而非无人）")
            elif not is_low and low_light:
                print(f"[A] 照度恢复：均值 {lum:.0f}")
            low_light = is_low
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            now = time.monotonic() - _t0
            overlay = ["NO FACE (skipped)"]

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                face_w = _d(lm[CHEEK_L], lm[CHEEK_R])
                face_h = _d(lm[FACE_TOP], lm[FACE_BOTTOM])
                ear = round((eye_ratio(lm, L_EYE) + eye_ratio(lm, R_EYE)) / 2.0, 2)

                # rPPG：前额 ROI 绿通道（与人脸检测同帧喂入，人脸丢失即清空）
                if rppg is not None:
                    x1 = int(max(0.0, lm[CHEEK_L].x + 0.15 * face_w) * frame.shape[1])
                    x2 = int(min(1.0, lm[CHEEK_R].x - 0.15 * face_w) * frame.shape[1])
                    y1 = int(max(0.0, lm[FACE_TOP].y + 0.05 * face_h) * frame.shape[0])
                    y2 = int(max(y1 + 1, (lm[BROW_L].y + lm[BROW_R].y) / 2.0 * frame.shape[0]))
                    roi = frame[y1:y2, x1:x2, 1]  # 绿通道对血流最敏感
                    if roi.size > 0:
                        rppg.update(now, float(roi.mean()))
                        hr, rr, ibi = rppg.compute(now)
                        wave_writer.writerow([round(now, 2), round(float(roi.mean()), 2),
                                              hr if hr is not None else "",
                                              rr if rr is not None else "",
                                              "|".join(map(str, ibi))])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

                calib = calibrator.update(*head_angles(lm, cfg), mouth_curvature(lm, face_h))
                if calib is None:
                    # 标定中：不写 CSV，Socket 发心跳（2 秒左右即完成）
                    heartbeat(now)
                    overlay = [f"CALIBRATING {calibrator.progress()}/{calibrator.n} 保持正常坐姿"]
                else:
                    pitch, yaw, roll, curv_delta = calib
                    label = labeler.make_label(ear, curv_delta, lm, face_w)
                    with_face += 1
                    label_stat[label] += 1
                    m = {"timestamp": round(now, 2), "has_face": True, "ear": ear,
                         "blink_cnt": labeler.blink_cnt, "pitch": pitch, "yaw": yaw,
                         "roll": roll, "emo_feature": label}
                    writer.writerow([m[k] for k in CSV_HEADER])
                    if sock:
                        sock.send(m)
                    overlay = [f"{label.upper()}  EAR={ear:.2f} BLINK={labeler.blink_cnt}",
                               f"P={pitch:.1f} Y={yaw:.1f} R={roll:.1f}"]
                    if rppg is not None:
                        overlay.append(f"HR={rppg.hr if rppg.hr is not None else '--'}  "
                                       f"RR={rppg.rr if rppg.rr is not None else '--'}")
                    if cfg.get("draw_mesh"):
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, res.multi_face_landmarks[0],
                            mp.solutions.face_mesh.FACEMESH_CONTOURS)
            else:
                no_face += 1
                heartbeat(now)
                if rppg is not None:
                    rppg.reset()   # 人脸丢失清空 rPPG 缓冲，避免假波形

            if low_light:
                overlay.append("LOW LIGHT (add lamp)")

            for i, text in enumerate(overlay):
                color = (0, 0, 255) if text.startswith(("NO FACE", "LOW LIGHT")) else (0, 255, 0)
                cv2.putText(frame, text, (10, 30 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if not args.no_window:
                cv2.imshow("backend_A vision", frame)
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    calibrator.recalibrate()
                    print("[A] 重新标定：请保持正常坐姿约 2 秒")
            else:
                time.sleep(1.0 / cfg["target_fps"])

            if args.seconds is not None and time.monotonic() - _t0 >= args.seconds:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        wave_file.close()
        face_mesh.close()
        if rppg is not None:
            print(f"[A] rPPG 末次结果：HR={rppg.hr} bpm | RR={rppg.rr} 次/分 | "
                  f"最近 IBI(ms)={rppg.ibi_ms}")

    print("\n[A] ===== 自测摘要 =====")
    print(f"总帧数 {total} | 有人脸 {with_face}（已写 CSV）| 无人脸 {no_face}（已跳过）")
    print(f"标签分布：{label_stat}")
    print(f"累计眨眼 {labeler.blink_cnt} 次")
    print(f"低照度帧 {low_light_cnt}（画面均值 < {low_light_th:.0f}，期间人脸检测不可信）")
    print(f"CSV 文件：{csv_path}")
    print(f"波形 CSV：{wave_path}")


if __name__ == "__main__":
    main()
