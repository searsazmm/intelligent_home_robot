"""体征通道的单元测试：合成源的环路自检 + 监视器的时序。

标准库 unittest，不依赖 socket、不依赖摄像头。

运行::

    cd backend_A && python -m unittest discover -s tests -v
    cd backend_A && python tests/test_vitals.py

这个文件里最重要的不是"hr 等于多少"，而是三条**防静默失效**的断言：

1. 合成信号真的能把 HR 恢复出来（而不是恒为 ``None`` 或带边值）——
   否则这条通道在演示路径上就是死的，而没有任何东西会报错；
2. 让两个色度投影同号的配方**启动即报错**，不许静默跑出 44；
3. 时间轴回退（**摄像头重连**，不是 ``--loop``）必须触发重置 ——
   不重置的话 :class:`~rppg.Rppg` 的 ``fs`` 会算错，此后**一直**
   走置灰分支，表现得像"拔插一次摄像头之后体征就再也没有了"。

关于第 3 条里"44"这个数：``Rppg`` 的心率带通是 0.7–4.0Hz，
``0.7Hz × 60 = 42``，而 ``hr = round(freq×60)`` 取到的最低非零格点是
44。它不是"心率 44"，它是**没有信号**。
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _console  # noqa: E402,F401  （导入即生效：让中文输出不乱码）

from module_a_vision.vitals import (  # noqa: E402
    DEFAULT_VITALS,
    RppgMonitor,
    SyntheticVitalsSource,
    VitalsSpec,
)
from module_a_vision.wire import assert_rppg_contract  # noqa: E402

#: 合成信号的采样率。``Rppg.compute`` 在 ``fs < 5.0`` 时一律置灰，
#: 所以这个数必须 ≥5 —— 它也是默认 ``--fps``（10）。
FPS = 10.0

#: 跑多少秒。``Rppg`` 的心率窗口是 8 秒，留一倍余量。
RUN_SEC = 20.0

#: 心率报文的合法区间下限是 30，而"没有信号"会被挡在 42/44 上 ——
#: 用它来断言"确实有信号"。
NO_SIGNAL_MAX = 45


class _FakeVitalsSource:
    """永远取不到色的源，用来验证监视器不会把空帧当 0 填进去。"""

    def read_rgb(self, ts: float):
        return None

    def describe(self) -> str:
        return "假源（永远取不到色）"


def _recover(spec: VitalsSpec, *, seconds: float = RUN_SEC) -> tuple:
    """把合成信号喂给真的 :class:`~rppg.Rppg`，返回 ``compute`` 的四元组。"""
    from rppg import Rppg

    source = SyntheticVitalsSource(spec)
    rppg = Rppg()
    frames = int(seconds * FPS)
    for i in range(frames):
        t = i / FPS
        rppg.update(t, *source.read_rgb(t))
    return rppg.compute(frames / FPS)


class TestSyntheticLoopback(unittest.TestCase):
    """环路自检：声明一个 HR，看算法能不能把它捞回来。"""

    def test_declared_heart_rate_is_recovered(self) -> None:
        """**这一条就是"体征通道是不是死代码"的单元级守卫。**

        它同时排除了三种静默失效：

        * 合成源根本没接上 ``Rppg``（返回 ``None``）；
        * 信号被 CHROM 的 ``α`` 完整抵消（返回带边值 44）；
        * 峰值检测挑错了谐波（返回两倍）。

        容差 8bpm 不是随手取的：8 秒窗、10fps 下 FFT 的频率分辨率是
        ``1/8s = 0.125Hz``，换算成心率就是 **7.5bpm**。要求精确相等
        在物理上就不可能 —— 算法只能报出格点上的值。
        """
        hr = _recover(DEFAULT_VITALS)[0]
        self.assertIsNotNone(hr, "hr 是 None —— 信号根本没进算法")
        self.assertGreater(hr, NO_SIGNAL_MAX, f"hr={hr} 落在带边，说明信号被抵消了")
        self.assertAlmostEqual(hr, DEFAULT_VITALS.hr_bpm, delta=8.0)

    def test_different_declared_rates_give_different_recoveries(self) -> None:
        """**真正证明"是恢复出来的、不是原样抄回去"的那一条。**

        上一条只比对了一个值，单个值对不上也可能只是碰巧。这条让三个
        相差很大的声明值各跑一遍：它们必须给出三个**互相不同**、且各自
        贴近自己声明值的结果。把算法换成一个返回常数的桩，这里立刻红。

        （声明的 ``hr_bpm`` 都取了小数，而恢复值是整数格点 —— 抄是抄不出
        这种"接近但不相等"的形态的。剩下的那点误差就是 7.5bpm 分辨率。）
        """
        cases = (58.0, 76.4, 110.0)
        got = [_recover(VitalsSpec(hr_bpm=hr))[0] for hr in cases]
        for declared, recovered in zip(cases, got):
            with self.subTest(declared=declared):
                self.assertIsNotNone(recovered)
                self.assertAlmostEqual(recovered, declared, delta=8.0)
        self.assertEqual(len(set(got)), len(cases), f"三个声明值给出了重复的结果：{got}")

    def test_ibi_is_produced(self) -> None:
        """心跳间期也要出来 —— 它是"峰值检测真的在跑"的旁证。"""
        ibi = _recover(DEFAULT_VITALS)[2]
        self.assertTrue(ibi, "一条 IBI 都没有")
        self.assertTrue(all(isinstance(x, int) for x in ibi))
        self.assertEqual(ibi, sorted(ibi) or ibi)  # 只要求是整数列表，不要求有序

    def test_signal_quality_is_good(self) -> None:
        """SQI 要过门 —— 过不了的话 hr 会置灰，上一条就变成在测运气。"""
        self.assertGreaterEqual(_recover(DEFAULT_VITALS)[3], 0.5)

    def test_rr_is_not_asserted_because_it_is_not_modelled(self) -> None:
        """``rr`` **故意不做数值断言**，理由写下来免得后来人补一条假的。

        这份合成源只建模心搏引起的通道变化，**不建模呼吸** —— 而且
        ``Rppg`` 的心率支路里有一步 1 秒滑动均值的去趋势，会把呼吸频段
        的信号压掉九成。所以 ``rr`` 出不来不是 bug。

        真要测呼吸率，得先有一个建模了呼吸的源；在那之前，任何
        "``rr`` 应该约等于 16" 的断言都是在测试一个不存在的实现。
        这里只钉住"它要么是 None、要么合法"。
        """
        rr = _recover(DEFAULT_VITALS)[1]
        self.assertTrue(rr is None or 4.0 <= rr <= 40.0, f"rr={rr!r} 越界")


class TestVitalsSpecGuard(unittest.TestCase):
    """配方校验：同号即抵消，必须启动就报错。"""

    def test_same_sign_chrominance_is_rejected(self) -> None:
        """**CHROM 的 ``α = std(X)/std(Y)`` 取幅度、没有符号。**

        这组比例（``p_X`` 与 ``p_Y`` 同号）下残差恰好为 0，基频被完整
        抵消 —— 而这组比例**恰恰是物理上正确的那个**（真实皮肤的脉搏
        在 X 与 Y 上同号相加）。所以直觉去"修正"参数会正好把信号改没。

        实测：这组配方下 hr 恒为 44（带边），一个算得出来、却毫无意义
        的数字。必须在这里拦下，否则故障表现是"体征数值不好看"。
        """
        with self.assertRaises(ValueError) as ctx:
            VitalsSpec(pulsatility_r=0.030, pulsatility_g=0.020)
        self.assertIn("抵消", str(ctx.exception))

    def test_default_spec_has_opposite_signs(self) -> None:
        p_x, p_y = DEFAULT_VITALS.chrominance_direction()
        self.assertLess(p_x * p_y, 0.0, "默认配方的色度投影必须反号")

    def test_the_physically_correct_ratio_is_also_rejected(self) -> None:
        """**这一条比上一条更要紧：连"物理上正确"的比例也在拦下的范围里。**

        ``pulsatility_r=0.010`` / ``pulsatility_g=0.005`` 给出
        ``p_X = +0.02``、``p_Y = +0.02`` —— 两个投影同号，也就是真实皮肤
        的脉搏方向。它是**对的**物理，却是**看不见的**信号。

        所以要确认闸门拦的是"算不出心率"这件事本身，而不是只拦住
        某个明显离谱的极端值。否则后来人会把默认值往这个方向"修正"，
        而契约闸一声不吭。
        """
        # 手算一遍色度投影方向（公式与 VitalsSpec 类文档一致），
        # 而不是先构造一个 spec 再取 —— 构造函数会先把我们拦下来。
        p_x = 3.0 * 0.010 - 2.0 * 0.005
        p_y = 1.5 * 0.010 + 0.005
        self.assertGreater(p_x * p_y, 0.0, "这组本该是同号的")

        with self.assertRaises(ValueError):
            VitalsSpec(pulsatility_r=0.010, pulsatility_g=0.005)


class TestRppgMonitor(unittest.TestCase):
    """发送节流、时间轴回退、以及"取不到色"的处理。"""

    def test_emits_at_one_hertz(self) -> None:
        """20 秒的流、1Hz → 20 条（t=0,1,…,19 各一条）。

        速率不是随便定的：帧报文 10Hz、体征 1Hz。体征发得太密没有意义
        （``Rppg.compute`` 自己就按 0.5 秒节流），发得太稀则 B 侧判不出
        "体征断了"。
        """
        monitor = RppgMonitor(SyntheticVitalsSource(), emit_interval=1.0)
        emitted = [p for p in (monitor.feed(i / FPS) for i in range(200)) if p]
        self.assertEqual(len(emitted), 20)

    def test_emitted_packets_pass_their_own_contract(self) -> None:
        """监视器产出的东西必须过自己的契约闸 —— 不然是在给 B 造垃圾。"""
        monitor = RppgMonitor(SyntheticVitalsSource())
        for i in range(200):
            payload = monitor.feed(i / FPS)
            if payload is not None:
                with self.subTest(t=payload["timestamp"]):
                    self.assertEqual(assert_rppg_contract(payload), [])

    def test_timestamp_rollback_resets_the_buffer(self) -> None:
        """**时间轴往回走时必须重置，否则体征会无声地永久消失。**

        触发这条分支的是**摄像头重连**，不是 ``--loop``。
        ``capture/camera.py:107`` 在 ``read()`` 里重连成功时把
        ``_mono_start`` 重置成当时的 ``monotonic()``，而
        ``_last_ts = monotonic() - _mono_start`` —— 于是拔插一次摄像头
        或一次 USB 抖动，``ts`` 就从几百秒跳回 0 附近，**而服务并不重启**。

        （``--loop`` 不走这条：合成源与视频源的时间轴是 ``_index / fps``，
        ``_index`` 只在 ``open()`` 里归零，而服务全程只 ``open()`` 一次。
        实测 --loop 跑到 415s 依然是单调的。早先的注释把它写成 ``--loop``，
        是错的。）

        不重置的话，``Rppg`` 内部 ``fs = 1/median(diff(t))`` 会由一段跨越
        回退点的差值算出来（负数或极小值），此后**一直**走 ``fs < 5.0``
        的置灰分支 —— 表现是"体征再也出不来了"，而每一步都不报错。
        """
        monitor = RppgMonitor(SyntheticVitalsSource())
        for i in range(200):
            monitor.feed(i / FPS)
        self.assertEqual(monitor.resets, 0, "单调时间轴上不该有任何重置")

        monitor.feed(0.0)  # 时间轴回退（摄像头重连后 _mono_start 归零）
        self.assertEqual(monitor.resets, 1)

        # 回退之后仍要能重新产出 hr —— 只清空缓冲还不够，
        # 节流用的 _last_emit 也得跟着清，否则要再等一秒才开始发。
        emitted = [p for p in (monitor.feed(i / FPS) for i in range(200)) if p]
        self.assertTrue(emitted, "回退之后一条都没再发出来")
        hrs = [p["hr"] for p in emitted if p["hr"] is not None]
        self.assertTrue(hrs, "回退之后 hr 再也没恢复出来")

    def test_no_rgb_skips_the_frame_instead_of_filling_zeros(self) -> None:
        """取不到色时**跳过**，不要拿 0 去填。

        填 0 会在缓冲里留下一段人造波形 —— 比留个空缺危险得多：
        它看起来像一段信号，会一路走到 HR 上去。
        """
        monitor = RppgMonitor(_FakeVitalsSource())
        for i in range(200):
            self.assertIsNone(monitor.feed(i / FPS))
        self.assertEqual(monitor.packets, 0)

    def test_describe_says_it_is_not_a_measurement(self) -> None:
        """``describe()`` 里必须有"非生理测量"这句话。

        它会出现在 A 的启动横幅里。演示截图是最容易被当成实验结论的
        东西，所以这句话的位置是被刻意选在横幅上的。
        """
        text = RppgMonitor(SyntheticVitalsSource()).describe()
        self.assertIn("非生理测量", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
