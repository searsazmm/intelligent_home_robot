"""
模块 A：rPPG 非接触脉搏波（需求文档 §12.2 P0/P1）
==================================================
输入：前额 ROI 绿通道均值时序（vision_a 主循环每帧喂入）
输出：脉搏波（FFT 带通后波形）、心率 HR、呼吸率 RR、心跳间期序列 IBI

只用 numpy（FFT 带通 + 重采样），不引入 scipy 依赖。
所有输出为"日常参考"级，禁止作为医疗结论（需求文档 §12.1 T3 红线）。
"""

import numpy as np
from collections import deque


class Rppg:
    def __init__(self, hr_window: float = 8.0, rr_window: float = 20.0):
        self.hr_window = hr_window        # 心率需要至少这么长的信号
        self.rr_window = rr_window        # 呼吸率需要更长的信号
        self._t = deque()                 # 时间戳（秒，程序运行时）
        self._g = deque()                 # 绿通道均值
        self.hr = None                    # 心率 bpm
        self.rr = None                    # 呼吸率 次/分钟
        self.ibi_ms = []                  # 最近的心跳间期（毫秒）
        self._last_compute = -1.0

    def reset(self):
        """人脸丢失时清空缓冲，防止跨断点插值出假波形。"""
        self._t.clear()
        self._g.clear()
        self.hr = self.rr = None
        self.ibi_ms = []

    def update(self, t: float, green: float) -> None:
        self._t.append(t)
        self._g.append(green)
        max_span = self.rr_window + 2.0
        while self._t and t - self._t[0] > max_span:
            self._t.popleft()
            self._g.popleft()

    # ---------- 信号处理 ----------

    @staticmethod
    def _detrend(sig: np.ndarray, fs: float) -> np.ndarray:
        w = max(3, int(fs))  # 1 秒滑动均值作为趋势项
        kernel = np.ones(w) / w
        return sig - np.convolve(sig, kernel, mode="same")

    @staticmethod
    def _bandpass(sig: np.ndarray, fs: float, lo: float, hi: float):
        """FFT 带通：返回 (freqs, 频谱幅值, 滤波后时域波形)。"""
        n = len(sig)
        win = np.hanning(n)
        spec = np.fft.rfft(sig * win)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        keep = (freqs >= lo) & (freqs <= hi)
        wave = np.fft.irfft(np.where(keep, spec, 0), n)
        return freqs, np.abs(spec) * keep, wave

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
        """节流 0.5s 计算一次；信号长度不足返回 None（前端显示预热中）。

        HR/IBI 只用最近 hr_window 秒（滑动窗口，动作伪影很快滑出）；
        RR 用最近 rr_window 秒（呼吸频率低，需要长窗口）。"""
        if now - self._last_compute < 0.5:
            return self.hr, self.rr, self.ibi_ms
        self._last_compute = now

        t_all = np.array(self._t)
        g_all = np.array(self._g)
        if t_all[-1] - t_all[0] < self.hr_window or len(t_all) < 30:
            return None, None, []

        fs = 1.0 / float(np.median(np.diff(t_all)))
        if fs < 5.0:  # 帧率过低，信号不可信
            return None, None, []

        # --- 心率 + IBI：最近 hr_window 秒 ---
        mask_hr = t_all >= t_all[-1] - self.hr_window
        t_hr, g_hr = t_all[mask_hr], g_all[mask_hr]
        grid = np.arange(t_hr[0], t_hr[-1], 1.0 / fs)
        sig = self._detrend(np.interp(grid, t_hr, g_hr), fs)
        freqs, mags, wave = self._bandpass(sig, fs, 0.7, 4.0)
        if mags.max() <= 0:
            return None, None, []
        self.hr = round(float(freqs[mags.argmax()]) * 60.0)

        peaks = self._peaks(wave, fs)
        if len(peaks) >= 3:
            ibi = np.diff(peaks) / fs * 1000.0
            # 一致性过滤：真心跳的间隔聚集在中位数附近，
            # 偏离 40% 以上的视为假峰/漏峰产物；过滤后不足 3 个则不输出
            med = float(np.median(ibi))
            good = [x for x in ibi if abs(x - med) <= 0.4 * med]
            self.ibi_ms = [int(round(x)) for x in good[-10:]] if len(good) >= 3 else []
        else:
            self.ibi_ms = []

        # --- 呼吸率：最近 rr_window 秒 ---
        if t_all[-1] - t_all[0] >= self.rr_window:
            grid_r = np.arange(t_all[0], t_all[-1], 1.0 / fs)
            sig_r = self._detrend(np.interp(grid_r, t_all, g_all), fs)
            freqs2, mags2, _ = self._bandpass(sig_r, fs, 0.1, 0.5)
            if mags2.max() > 0:
                self.rr = round(float(freqs2[mags2.argmax()]) * 60.0, 1)
        return self.hr, self.rr, self.ibi_ms
