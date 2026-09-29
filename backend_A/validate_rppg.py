"""
模块 A：rPPG 对照验证工具（Bland-Altman，课程级证据链）
========================================================
没有 ECG 设备，用智能手环心率读数做参照（手环自身也有 ±3-5 bpm 光电误差，
结论口径只写"摄像头 rPPG 与手环读数的一致性"，不写"精度绝对值"）。

两步使用：

  1) 采数据（开两个终端，手环戴好）：
       终端1: python vision_a.py                 # 正常运行写波形 CSV
       终端2: python validate_rppg.py --record   # 跟着提示走
     安坐，每隔 ≥30 秒看一眼手环心率，在提示符输入数字回车（如 72），
     采 8-10 个点后按 q 结束。全程保持安静坐姿、少说话少晃。

  2) 算一致性（录完即出，或事后单独跑）：
       python validate_rppg.py --compare --sync data/rppg_sync_xxx.csv \
                              --wave data/pulse_wave_xxx.csv

输出：偏差 bias（A−手环）、95% 一致性界限 LoA、MAE、Pearson r，
逐点对照表存 data/rppg_validation_*.csv（课程报告画 Bland-Altman 散点图直接用它）。
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MIN_PAIRS = 3        # 低于此数拒绝出统计结论
MIN_HR_SAMPLES = 4   # 每个参照点窗口内至少要这么多帧 HR 才可信


def _setup_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def newest_wave_csv() -> Path | None:
    files = sorted(DATA_DIR.glob("pulse_wave_*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def read_wave(path: Path):
    """[(t, hr), ...]：波形 CSV 中 hr 非空的行（被 SQI 门控置灰的行自动跳过）。"""
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            hr = (row.get("hr") or "").strip()
            if not hr:
                continue
            try:
                rows.append((t, float(hr)))
            except ValueError:
                continue
    return rows


def last_t(path: Path) -> float | None:
    """最后一行的 timestamp（含置灰行——对表只要时标）。"""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            lines = [ln for ln in f if ln.strip()]
        if len(lines) < 2:
            return None
        return float(lines[-1].split(",")[0])
    except (OSError, ValueError, IndexError):
        return None


def record(sync_path: Path, wave_path: Path, poll: float) -> None:
    """采参照点：对表（波形单调时标 ↔ 墙钟）→ 交互式记录手环读数 → 存 sync CSV。"""
    t0 = None
    while t0 is None:
        t0 = last_t(wave_path)
        if t0 is None:
            print("[验证] 等待波形文件出现数据行……")
            time.sleep(poll)
    offset = time.time() - t0   # 墙钟(t) ≈ offset + 波形时标；逐帧 flush 后误差 ≤1 行
    print(f"[验证] 对表完成：波形 t={t0:.1f}s ↔ 墙钟 {datetime.fromtimestamp(time.time()):%H:%M:%S}")
    print("[验证] 每隔 ≥30 秒看一眼手环，输入读数回车（如 72）；空行跳过，q 结束")

    samples = []
    while True:
        try:
            line = input(f"[第 {len(samples) + 1} 点] 手环心率> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("q", "quit", "exit"):
            break
        if not line:
            continue
        try:
            hr = int(float(line))
        except ValueError:
            print("  不是数字，重新输入")
            continue
        t = round(time.time() - offset, 2)
        samples.append((t, hr))
        print(f"  已记录：t={t:.1f}s 手环={hr}bpm")

    with open(sync_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["timestamp", "watch_hr"])
        for t, hr in samples:
            w.writerow([t, hr])
    print(f"[验证] 已保存 {len(samples)} 个参照点 → {sync_path}")
    if len(samples) < 5:
        print("[验证] ⚠️ 参照点少于 5 个，后续统计意义有限")
    if samples:
        print(f"[验证] 接着跑：python validate_rppg.py --compare --sync {sync_path} --wave {wave_path}")


def compare(sync_path: Path, wave_path: Path, window: float) -> None:
    """每个手环参照点取窗口 ±window/2 内 A 的有效 HR 均值配对，算 Bland-Altman 统计量。"""
    rows = read_wave(wave_path)
    if not rows:
        print("[验证] 波形 CSV 里没有有效 HR 行（可能全程被 SQI 门控置灰）")
        return
    ts = np.array([t for t, _ in rows])
    hrs = np.array([h for _, h in rows])

    pairs = []
    with open(sync_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                tw, whr = float(row["timestamp"]), float(row["watch_hr"])
            except (KeyError, TypeError, ValueError):
                continue
            m = (ts >= tw - window / 2) & (ts <= tw + window / 2)
            n = int(m.sum())
            if n < MIN_HR_SAMPLES:
                print(f"  t={tw:.0f}s 手环={whr:.0f} → 窗口内有效 HR 不足 {MIN_HR_SAMPLES} 帧（{n}），跳过")
                continue
            a_hr = float(hrs[m].mean())
            pairs.append((tw, whr, a_hr, a_hr - whr, n))
    if len(pairs) < MIN_PAIRS:
        print(f"[验证] 有效配对不足 {MIN_PAIRS} 组，无法出统计结论")
        return

    diff = np.array([p[3] for p in pairs])
    a_all = np.array([p[2] for p in pairs])
    w_all = np.array([p[1] for p in pairs])
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    mae = float(np.abs(diff).mean())
    r = (float(np.corrcoef(a_all, w_all)[0, 1])
         if float(a_all.std()) > 0 and float(w_all.std()) > 0 else float("nan"))

    print(f"\n[验证] ===== Bland-Altman 一致性（配对 {len(pairs)} 组，窗口 ±{window / 2:.0f}s）=====")
    print(f"{'t(s)':>7} {'手环':>6} {'A相机':>7} {'差值':>7} {'帧数':>5}")
    for tw, whr, a_hr, d, n in pairs:
        print(f"{tw:>7.0f} {whr:>6.0f} {a_hr:>7.1f} {d:>+7.1f} {n:>5}")
    print(f"偏差 bias（A−手环）= {bias:+.1f} bpm")
    print(f"95% 一致性界限 LoA = {loa_lo:+.1f} ~ {loa_hi:+.1f} bpm")
    print(f"MAE = {mae:.1f} bpm | Pearson r = {r:.2f}")

    out = DATA_DIR / f"rppg_validation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["timestamp", "watch_hr", "camera_hr", "diff", "hr_samples"])
        for tw, whr, a_hr, d, n in pairs:
            w.writerow([round(tw, 2), whr, round(a_hr, 2), round(d, 2), n])
    print(f"[验证] 逐点表 → {out}（课程报告画 Bland-Altman 散点图直接用它）")
    if abs(bias) <= 5 and (loa_hi - loa_lo) <= 15:
        print("[验证] 结论：与手环读数一致性良好（课程级证据，样本量小，手环自身也有误差）")
    else:
        print("[验证] 结论：偏差较大——检查采集时是否晃动/说话/侧脸，或按 README 调 rppg 参数后重采")


def main() -> None:
    _setup_stdout()
    ap = argparse.ArgumentParser(description="rPPG 手环对照验证（Bland-Altman）")
    ap.add_argument("--record", action="store_true", help="采参照点（跟随 vision_a 运行）")
    ap.add_argument("--compare", action="store_true", help="算一致性统计")
    ap.add_argument("--sync", default=None, help="sync CSV（--compare 用）")
    ap.add_argument("--wave", default=None, help="波形 CSV（默认取 data 下最新）")
    ap.add_argument("--window", type=float, default=30.0, help="配对窗口宽（秒，默认 30）")
    ap.add_argument("--poll", type=float, default=1.0, help="record 对表轮询间隔（秒）")
    args = ap.parse_args()

    wave_path = Path(args.wave) if args.wave else newest_wave_csv()
    if args.compare:
        if not args.sync or not Path(args.sync).exists():
            print("[验证] --compare 需要 --sync 指向 record 生成的参照点 CSV")
            sys.exit(1)
        if wave_path is None or not wave_path.exists():
            print("[验证] 找不到波形 CSV")
            sys.exit(1)
        compare(Path(args.sync), wave_path, args.window)
    else:
        # 默认（含 --record）：交互采集
        if wave_path is None or not wave_path.exists():
            print("[验证] data/ 下还没有波形 CSV，请先启动 vision_a.py")
            sys.exit(1)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        record(DATA_DIR / f"rppg_sync_{stamp}.csv", wave_path, args.poll)


if __name__ == "__main__":
    main()
