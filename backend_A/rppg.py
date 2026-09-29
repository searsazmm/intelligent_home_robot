"""
模块 A：rPPG 非接触脉搏波（需求文档 §12.2 P0/P1）
==================================================
输入：面额 ROI 三通道均值时序 (t, R, G, B)（vision_a 主循环每帧喂入，OpenCV BGR 由调用侧转好）
输出：脉搏波（CHROM 色度法）、心率 HR、呼吸率 RR、心跳间期序列 IBI、质量分 SQI

算法：
- CHROM（de Haan & Jeanne, 2014）：归一化三通道构造色度信号 X=3R-2G、Y=1.5R+G-1.5B，
  带通后 S = X - α·Y（α=std(X)/std(Y)），对运动伪影与光照漂移的鲁棒性显著优于裸绿通道
- SQI 质量门控（需求文档 §12.5 约定）：SNR = 带内功率(0.7-4Hz)/带外功率，
  SNR < SNR_MIN 时 HR/IBI 置灰（None/[]）；IBI 还要求过滤后间隔变异系数 CV<=0.3

只用 numpy（FFT 带通 + 重采样），不引入 scipy 依赖。
所有输出为"日常参考"级，禁止作为医疗结论（需求文档 §12.1 T3 红线）。
"""

import numpy as np
from collections import deque


class Rppg:
    SNR_MIN = 1.5          # 带内/带外功率比低于此值 → HR/IBI 置灰
    CV_MAX = 0.3           # IBI 过滤后变异系数上限

    def __init__(self, hr_window: float = 8.0, rr_window: float = 20.0):
        self.hr_window = hr_window        # 心率需要至少这么长的信号
        self.rr_window = rr_window        # 呼吸率需要更长的信号
        self._t = deque()                 # 时间戳（秒，程序运行时）
        self._r = deque()                 # R 通道均值
        self._g = deque()                 # G 通道均值
        self._b = deque()                 # B 通道均值
        self.hr = None                    # 心率 bpm（None=置灰/预热中）
        self.rr = None                    # 呼吸率 次/分钟
        self.ibi_ms = []                  # 最近的心跳间期（毫秒）
        self.sqi = None                   # 信号质量分 0-1（min(SNR/4, 1)）
        self._last_compute = -1.0

    def reset(self):
        """人脸丢失时清空缓冲，防止跨断点插值出假波形。"""
        self._t.clear()
        self._r.clear()
        self._g.clear()
        self._b.clear()
        self.hr = self.rr = None
        self.ibi_ms = []
        self.sqi = None

    def update(self, t: float, r: float, g: float, b: float) -> None:
        self._t.append(t)
        self._r.append(r)
        self._g.append(g)
        self._b.append(b)
        max_span = self.rr_window + 2.0
        while self._t and t - self._t[0] > max_span:
            self._t.popleft()
            self._r.popleft()
            self._g.popleft()
            self._b.popleft()

    # ---------- 信号处理 ----------

    @staticmethod
    def _detrend(sig: np.ndarray, fs: float) -> np.ndarray:
        w = max(3, int(fs))  # 1 秒滑动均值作为趋势项
        kernel = np.ones(w) / w
        return sig - np.convolve(sig, kernel, mode="same")

    @staticmethod
    def _bandpass(sig: np.ndarray, fs: float, lo: float, hi: float):
        """FFT 带通：返回 (freqs, 完整频谱幅值, 滤波后时域波形)。
        完整频谱保留带外成分，供 SQI 计算带内/带外功率比。"""
        n = len(sig)
        win = np.hanning(n)
        S = np.fft.rfft(sig * win)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        keep = (freqs >= lo) & (freqs <= hi)
        wave = np.fft.irfft(np.where(keep, S, 0), n)
        return freqs, np.abs(S), wave

    @staticmethod
    def _chrom(r, g, b):
        """CHROM 色度信号（输入为归一化通道）。"""
        return 3.0 * r - 2.0 * g, 1.5 * r + g - 1.5 * b

    @staticmethod
    def _peaks(wave: np.ndarray, fs: float, min_dist_sec: float = 0.45):
        """局部极大值 + 不应期（相邻心跳至少间隔 0.45s ≈ 133bpm 上限，
        静息场景足够）+ 幅度门槛（滤掉带通残留的小毛刺假峰）。"""
        dist = max(1, int(min_dist_sec * fs))
        thresh = 0.2 * float(np.max(np.abs(wave)))
        peaks = []
        for i in range(1, len(wave) - 1):
            if wave[i] > wave[i - 1] and wave[i] >= wave[i + 1] and wave[i] > thresh:
                if peaks and i - peaks[-1] < dist:
                    if wave[i] > wave[peaks[-1]]:
                        peaks[-1] = i
                else:
                    peaks.append(i)
        return np.array(peaks, dtype=int)

    def compute(self, now: float):
        """节流 0.5s 计算一次，返回 (hr, rr, ibi_ms, sqi)。
        信号长度不足或 SQI 不达标 → 对应输出置灰（None/[]），宁缺毋滥。

        HR/IBI 只用最近 hr_window 秒（滑动窗口，动作伪影很快滑出）；
        RR 用最近 rr_window 秒（呼吸频率低，需要长窗口）。"""
        if now - self._last_compute < 0.5:
            return self.hr, self.rr, self.ibi_ms, self.sqi
        self._last_compute = now
        self.hr = self.rr = None
        self.ibi_ms = []
        self.sqi = None

        t_all = np.array(self._t)
        if t_all.size < 2 or t_all[-1] - t_all[0] < self.hr_window or len(t_all) < 30:
            return self.hr, self.rr, self.ibi_ms, self.sqi
        fs = 1.0 / float(np.median(np.diff(t_all)))
        if fs < 5.0:  # 帧率过低，信号不可信
            return self.hr, self.rr, self.ibi_ms, self.sqi

        # --- CHROM 脉搏波 + HR + SQI：最近 hr_window 秒 ---
        mask = t_all >= t_all[-1] - self.hr_window
        t_hr = t_all[mask]
        grid = np.arange(t_hr[0], t_hr[-1], 1.0 / fs)

        def interp(ch):
            return np.interp(grid, t_hr, np.asarray(ch)[mask])

        R, G, B = interp(self._r), interp(self._g), interp(self._b)
        scale = max(float(R.mean()), 1e-6), max(float(G.mean()), 1e-6), max(float(B.mean()), 1e-6)
        X, Y = self._chrom(R / scale[0], G / scale[1], B / scale[2])
        _, _, Xf = self._bandpass(self._detrend(X, fs), fs, 0.7, 4.0)
        _, _, Yf = self._bandpass(self._detrend(Y, fs), fs, 0.7, 4.0)
        sy = float(np.std(Yf))
        if sy < 1e-9:
            return self.hr, self.rr, self.ibi_ms, self.sqi
        pulse = Xf - (float(np.std(Xf)) / sy) * Yf

        freqs, spec, wave = self._bandpass(pulse, fs, 0.7, 4.0)
        inb = (freqs >= 0.7) & (freqs <= 4.0)
        p_in = float((spec[inb] ** 2).sum())
        p_out = float((spec[~inb] ** 2).sum())
        snr = p_in / max(p_out, 1e-9)
        self.sqi = round(min(snr / 4.0, 1.0), 2)
        if p_in <= 0 or snr < self.SNR_MIN:
            return self.hr, self.rr, self.ibi_ms, self.sqi   # 质量门控：置灰
        self.hr = round(float(freqs[inb][int(np.argmax(spec[inb]))]) * 60.0)

        peaks = self._peaks(wave, fs)
        if len(peaks) >= 3:
            ibi = np.diff(peaks) / fs * 1000.0
            # 一致性过滤：真心跳的间隔聚集在中位数附近，偏离 40% 的视为假峰产物
            med = float(np.median(ibi))
            good = [x for x in ibi if abs(x - med) <= 0.4 * med]
            if len(good) >= 3:
                arr = np.array(good)
                if float(arr.std() / max(arr.mean(), 1e-6)) <= self.CV_MAX:
                    self.ibi_ms = [int(round(x)) for x in good[-10:]]
                # CV 超限 → IBI 置灰（HR 仍可信，心跳间期不输出垃圾）

        # --- 呼吸率：最近 rr_window 秒（CHROM 同法，呼吸频段） ---
        if t_all[-1] - t_all[0] >= self.rr_window:
            grid_r = np.arange(t_all[0], t_all[-1], 1.0 / fs)

            def interp_r(ch):
                return np.interp(grid_r, t_all, np.asarray(ch))

            R2, G2, B2 = interp_r(self._r), interp_r(self._g), interp_r(self._b)
            s2 = max(float(R2.mean()), 1e-6), max(float(G2.mean()), 1e-6), max(float(B2.mean()), 1e-6)
            X2, Y2 = self._chrom(R2 / s2[0], G2 / s2[1], B2 / s2[2])
            _, _, X2f = self._bandpass(self._detrend(X2, fs), fs, 0.1, 0.5)
            _, _, Y2f = self._bandpass(self._detrend(Y2, fs), fs, 0.1, 0.5)
            s2y = float(np.std(Y2f))
            sig_r = X2f - ((float(np.std(X2f)) / s2y) if s2y > 1e-9 else 0.0) * Y2f
            freqs2, spec2, _ = self._bandpass(sig_r, fs, 0.1, 0.5)
            if spec2.max() > 0:
                self.rr = round(float(freqs2[int(np.argmax(spec2))]) * 60.0, 1)
        return self.hr, self.rr, self.ibi_ms, self.sqi
