"""A → B 的真实 socket 对端测试。

这个文件存在的理由，是**另外两个测试都抓不到的一类 bug**：
A 吐的报文形状对、字段类型对、单元测试全绿，但和 ``backend_B`` 的实际
读取逻辑对不上 —— 而 B 对缺失字段**静默回退默认值**，所以这种错的表现
不是崩溃，是"状态永远停在 absent"。没有真实对端，它就查不出来。

这里跑的是真 socket、真的 A 服务线程、真的 B 客户端：

* :class:`TestWireShape` —— 起 A，用裸 socket 收，逐条校验契约。
* :class:`TestAgainstRealEvaluator` —— 起 A，让**真的** ``VisionClient``
  连上去，喂**真的** ``VisionStateEvaluator``，然后断言两件事：
  状态确实被判成了 ``normal``（说明字段名对上了），
  以及**从失联抖动里恢复不过来**的那个失效没发生。

``backend_B`` 不在旁边时第二个类会自动跳过 —— A 的测试不该因为
B 缺席而变红，但 B 在的时候必须真的去测。

运行::

    cd backend_A && python -m unittest discover -s tests -v
    cd backend_A && python tests/test_interop_with_b.py
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _console  # noqa: E402,F401  （导入即生效：让中文输出不乱码）

from module_a_vision.server import build_server  # noqa: E402
from module_a_vision.wire import (  # noqa: E402
    KIND_FRAME,
    TYPED_MESSAGE_TYPES,
    V1_FIELDS,
    check_contract,
)

# ---------------------------------------------------------------- B 的可用性

_B_DIR = Path(_ROOT).parent / "backend_B"
_B_AVAILABLE = (_B_DIR / "core" / "vision_state.py").is_file()

if _B_AVAILABLE and str(_B_DIR) not in sys.path:
    # backend_B 的内部导入是以自己为根的绝对导入（`from core.protocol import …`），
    # 所以必须把 backend_B 本身放进 sys.path，而不是它的父目录。
    sys.path.insert(0, str(_B_DIR))

#: 采集窗口。够跨过 B 的 ``STATE_MIN_HOLD``（1.5s）并留出余量。
OBSERVE_SEC = 6.0

#: B 判定"看不见老人"的失联阈值（``backend_B/config.py:VISION_STALE_SECONDS``）。
#: 这里刻意用**字面量**而不是去 import B 的 config：如果万一 import 失败，
#: 这个数会静默变成别的东西，而"报文间隔必须远小于失联阈值"这条断言
#: 正是本文件要守的东西，不能让它跟着被测方一起漂。
B_STALE_SECONDS = 5.0


class _ServerHarness:
    """在临时端口上起一个真的 A 服务，后台线程跑。"""

    def __init__(self, scenario: str = "sad", **kwargs: object) -> None:
        # port=0 → 让操作系统挑一个空闲端口，测试之间不会互相抢 8000，
        # 也不会和开发时真开着的 A 撞车。
        self.server = build_server(
            source_kind="synthetic",
            scenario=scenario,
            host="127.0.0.1",
            port=0,
            real_time=True,
            emit_mode="v1",
            **kwargs,
        )
        sock = self.server.serve()
        self.port = sock.getsockname()[1]
        self._thread = threading.Thread(
            target=self.server.run, name="test-a-produce", daemon=True
        )
        self._thread.start()

    def connect(self) -> socket.socket:
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=3.0)
        sock.settimeout(2.0)
        return sock

    def stop(self) -> None:
        self.server.stop()
        self._thread.join(timeout=3.0)


def _read_lines(sock: socket.socket, seconds: float) -> list[dict]:
    """收若干秒的报文。按 ``\\n`` 分帧，返回解析后的字典列表。"""
    deadline = time.monotonic() + seconds
    buf = b""
    out: list[dict] = []
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            continue
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                out.append(json.loads(line.decode("utf-8")))
    return out


class TestWireShape(unittest.TestCase):
    """裸 socket 收 A 的报文，逐条校验契约。

    v1 流上现在有**两种**东西（见 :mod:`module_a_vision.wire` 的模块文档）：

    * **帧**报文 —— §3.2 的平铺 8 字段，**不带 ``type``**；
    * **类型化**报文 —— §3.5 的 ``rppg`` / ``focus``，**带 ``type``**。

    所以下面每一条都得先想清楚"这条断言到底是对谁说的"。混着断言会得到
    一条会在两种报文之间来回翻转的测试，而它失败时给不出任何线索。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.h = _ServerHarness()
        cls.sock = cls.h.connect()
        cls.samples = _read_lines(cls.sock, OBSERVE_SEC)
        cls.frames = [p for p in cls.samples if "type" not in p]
        cls.typed = [p for p in cls.samples if "type" in p]

    @classmethod
    def tearDownClass(cls) -> None:
        with contextlib.suppress(OSError):
            cls.sock.close()
        cls.h.stop()

    # ------------------------------------------------------------ 帧报文

    def test_received_enough_frames(self) -> None:
        """收到足够多的**帧** —— 这一条同时排除了"心脏跳"混进来。

        10fps 跑 6 秒应该接近 60 条。下限放到 30 是留给 Windows 上
        线程调度的抖动，但**远高于**"每 10 秒一条"会得到的 1 条。
        所以如果有人把逐帧流式改回按窗口上报，这里会立刻变红。

        数的是 ``cls.frames`` 而不是 ``cls.samples``：后者现在混着体征
        报文，拿它当帧率会把 1Hz 的体征也算成视觉帧。
        """
        self.assertGreater(
            len(self.frames),
            30,
            f"6 秒只收到 {len(self.frames)} 帧 —— 逐帧流式是不是被改掉了？",
        )

    def test_every_frame_has_exactly_the_eight_fields(self) -> None:
        self.assertTrue(self.frames, "一帧都没收到")
        for i, payload in enumerate(self.frames):
            with self.subTest(index=i):
                self.assertEqual(set(payload), set(V1_FIELDS))

    def test_every_message_passes_the_contract_gate(self) -> None:
        """整条流上**每一条**都要过闸 —— 帧和类型化报文各自的那一道。"""
        self.assertTrue(self.samples, "一条报文都没收到")
        for i, payload in enumerate(self.samples):
            with self.subTest(index=i):
                kind, problems = check_contract(payload)
                self.assertEqual(problems, [], f"{kind} 报文违规")

    def test_blink_counter_never_goes_backwards(self) -> None:
        """整条**帧**流上 ``blink_cnt`` 单调不减。

        B 的 ``_blink_rate_per_minute`` 检测到倒退就返回 None，
        于是**静默关掉一条疲劳判据** —— 不报错、不告警。
        也就是说这个 bug 在集成后是不可见的，只有在这里才拦得住。
        """
        counters = [p["blink_cnt"] for p in self.frames]
        self.assertEqual(counters, sorted(counters), "blink_cnt 出现倒退")

    def test_frame_rate_beats_the_stale_threshold(self) -> None:
        """**本文件最重要的一条断言。**

        帧报文的 ``timestamp`` 间隔必须**远小于** B 判定失联的 5 秒阈值。
        这条测试针对的是一个真实发生过的设计错误：曾经打算在 A 关窗时
        （每 10 秒）发一条 §3.2 报文。那样 B 每 10 秒里会有 5 秒判 absent，
        状态周期性抖动、C 端的卡片跟着闪 —— 而两个模块各自的单元测试
        **全都是绿的**，因为谁的逻辑都没错，错的是两者的时间尺度关系。
        """
        stamps = [p["timestamp"] for p in self.frames]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        self.assertTrue(gaps, "只收到一帧，算不出间隔")
        worst = max(gaps)
        self.assertLess(
            worst,
            B_STALE_SECONDS / 2,
            f"最大帧间隔 {worst:.2f}s，接近 B 的失联阈值 {B_STALE_SECONDS}s，"
            f"会造成状态抖动",
        )

    def test_stream_starts_with_a_face(self) -> None:
        """``sad`` 剧本开场是 25 秒安静段，所以帧流里必须有人脸。"""
        self.assertTrue(any(p["has_face"] for p in self.frames))
        self.assertTrue(all(isinstance(p["has_face"], bool) for p in self.frames))

    def test_emo_feature_reaches_low_after_calibration(self) -> None:
        """表情通道真的在工作，不是一个恒为 normal 的常量。

        6 秒的观察窗不够跨过表情分类器的 25 秒标定段，所以这条只断言
        **取值合法**；"能不能变成 low"由 ``--dry-run`` 与手工联调验证
        （见 README 的验证表）。这里刻意不断言 ``low`` 出现过 ——
        写一条自己都知道跑不到的断言，比不写更糟。
        """
        values = {p["emo_feature"] for p in self.frames}
        self.assertTrue(values <= {"normal", "low", "tired"}, f"非法取值 {values}")

    # ------------------------------------------------------------ 类型化报文

    def test_only_allowlisted_types_appear(self) -> None:
        """v1 流上只能出现 §3.5 白名单里的 ``type``。

        这条替换了原来那条 ``assertNotIn("type", payload)`` ——
        ``type`` 现在是**合法的鉴别符**了，继续笼统地禁止它会拦下正确
        的实现。但原来的意图必须保留下来，所以下面把三种**具体**的
        V2 报文名单独点出来断言：

        ``heartbeat`` / ``vision_window`` / ``vision_unusable`` 一个都
        不能出现。B 的 ``VisionSample.from_payload`` 会把任何一条缺 8
        字段的报文兜成一个 ``has_face=false`` 的样本推进判定器 ——
        也就是周期性伪造"看不见老人"。这正是 A 在 v1 模式下**刻意
        关掉心跳**的原因（``server.serve``），新开类型化通道时不能
        把这套失效再放进来一次。
        """
        seen = {p["type"] for p in self.typed}
        self.assertTrue(
            seen <= set(TYPED_MESSAGE_TYPES),
            f"出现了白名单外的报文类型：{sorted(seen - set(TYPED_MESSAGE_TYPES))}",
        )
        for forbidden in ("heartbeat", "vision_window", "vision_unusable"):
            self.assertNotIn(
                forbidden, seen, f"{forbidden} 混进了 v1 流，B 会当成幽灵帧"
            )

    def test_every_message_is_a_frame_or_a_known_typed_message(self) -> None:
        """兜掉"第三种东西"：每条报文要么是帧，要么是登记过的类型。

        这条与上面那条互补。上面查的是"``type`` 的取值合不合法"，
        这条查的是"**没有 ``type`` 的那些到底是不是帧**" —— 一条既
        不带 ``type``、字段又不对的报文会同时绕过两条单独的检查。
        """
        for i, payload in enumerate(self.samples):
            with self.subTest(index=i):
                kind, _ = check_contract(payload)
                self.assertIn(kind, {KIND_FRAME, *TYPED_MESSAGE_TYPES})

    def test_vitals_packets_actually_flow(self) -> None:
        """**观察窗内必须收到体征报文，否则体征通道就是死代码。**

        这一条是专门为下面那个失效写的，它真实发生过（在计划阶段被
        设计审查抓住）：默认演示源 ``--source synthetic`` 是**特征源**，
        全程不碰像素。如果 rPPG 只挂在"有像素"的分支上，演示路径上
        它一行都不会执行 —— 而**整套测试照绿**，因为没有一条断言在问
        "体征到底发出来没有"。所以这条断言不是凑数的，它是这一类
        静默失效的唯一守卫。

        下限取 3 而不是 1：1Hz 跑 6 秒应该收到 5-6 条，只收到 1 条说明
        "发了一条就停了"，那和一条都没有差不多。
        """
        vitals = [p for p in self.typed if p["type"] == "rppg"]
        self.assertGreaterEqual(
            len(vitals),
            3,
            f"{OBSERVE_SEC:.0f} 秒只收到 {len(vitals)} 条体征报文 —— "
            f"体征通道是死的？检查 build_server 有没有给合成源挂 vitals",
        )

    def test_focus_packets_actually_flow(self) -> None:
        """**观察窗内必须收到视线报文，而且要与帧数同量级。**

        与上面那条体征断言同一个理由，但**更硬**：视线报文与帧**同频**，
        所以它的条数应当和帧数相当（10fps 跑 6 秒 ≈ 60 条）。B 的
        ``PassiveCalibrator`` 要求 ``≥30 个样本``且相邻样本间隔 ≤1 秒，
        掉到 1Hz 以下它就每次 ``update()`` 都 ``reset()`` ——
        **"校准永远做不完"而不报任何错**。

        下限取帧数的**一半**而不是相等：裸 socket 的收发与拆包会丢一点
        边角（观察窗的头尾各切一刀），而"是不是同频"这个判断在 30 与 60
        之间已经足够清楚。真正的"一个不差"由 ``--dry-run`` 的收尾断言看着
        （那里没有 socket，数得清）。
        """
        focus = [p for p in self.typed if p["type"] == "focus"]
        self.assertGreaterEqual(
            len(focus),
            len(self.frames) // 2,
            f"{OBSERVE_SEC:.0f} 秒收到 {len(focus)} 条视线报文，"
            f"而帧有 {len(self.frames)} 条 —— 视线通道被抽稀了？"
            f"B 的 VAI 校准会因此永远做不完，且不报错",
        )

    def test_focus_gaze_is_null_exactly_when_quality_is_zero(self) -> None:
        """钉住 §3.5.3 的不变式：``gaze is None ⟺ gaze_quality == 0``。

        这条契约已经在 ``wire.assert_focus_contract`` 里查过一遍（上面
        ``test_every_message_passes_the_contract_gate`` 会跑），这里再在
        真实流上单独钉一次，是因为**它是对端唯一能拿到的"这一帧有没有量到"
        的信号**：B 只据 ``gaze is None`` 决定要不要把这一帧算进校准窗口，
        而**刻意不去看那个质量分**（那是 A 给自己打的分）。两者不一致时，
        B 采信哪一半都会算错。
        """
        focus = [p for p in self.typed if p["type"] == "focus"]
        self.assertTrue(focus, "一条视线报文都没有")
        for i, payload in enumerate(focus):
            with self.subTest(index=i):
                self.assertEqual(
                    payload["gaze"] is None,
                    payload["gaze_quality"] == 0.0,
                    f"gaze={payload['gaze']!r} 与 "
                    f"gaze_quality={payload['gaze_quality']!r} 不一致",
                )

    def test_vitals_hr_may_be_null(self) -> None:
        """``hr`` 允许是 ``null`` —— 那是**正常态**，不是错误。

        ``Rppg`` 在 8 秒窗口未满、SNR<1.5、或帧率<5Hz 时都会置灰。
        6 秒的观察窗**必然**落在"窗口未满"那一段里，所以这条断言
        实际上是在钉住"契约允许置灰"。任何"收到 rppg ⇒ hr 是数字"
        的假设都会让观察窗一缩短就红，而那不是被测代码的错。
        """
        vitals = [p for p in self.typed if p["type"] == "rppg"]
        self.assertTrue(vitals, "一条体征报文都没有")
        for payload in vitals:
            self.assertTrue(
                payload["hr"] is None or isinstance(payload["hr"], int),
                f"hr 既不是 null 也不是 int：{payload['hr']!r}",
            )


@unittest.skipUnless(_B_AVAILABLE, "backend_B 不在旁边，跳过对端测试")
class TestAgainstRealEvaluator(unittest.TestCase):
    """让**真的** B 客户端与判定器消费 A 的报文。"""

    @classmethod
    def setUpClass(cls) -> None:
        from core.vision_state import VisionStateEvaluator
        from core.vision_client import VisionClient

        cls.h = _ServerHarness()
        cls.evaluator = VisionStateEvaluator()
        cls.client = VisionClient(
            cls.evaluator, host="127.0.0.1", port=cls.h.port
        )
        cls._thread = threading.Thread(
            target=cls.client.run, name="test-b-client", daemon=True
        )
        cls._thread.start()

        # 等到真的有样本进了判定器再开始观察：连接建立之前
        # get_state() 返回的是"尚未收到视觉数据"，那是初始态不是抖动。
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and cls.client.frames_received < 5:
            time.sleep(0.05)

        # 再多等一会儿，让防抖走完（B 的 STATE_MIN_HOLD = 1.5 秒）。
        # 不等的话，下面观察到的头几秒是**启动过程**而不是稳定运行 ——
        # 而启动期本来就是 absent，把它算成抖动是测试自己的错。
        time.sleep(2.5)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.stop()
        cls.h.stop()

    def test_client_actually_connected_and_read_frames(self) -> None:
        self.assertTrue(self.client.connected, "B 客户端没连上 A")
        self.assertGreaterEqual(
            self.client.frames_received, 5, "连上了但一帧都没读到"
        )

    def test_no_bad_lines(self) -> None:
        """没有一行被 B 判为非法。

        行不通的 JSON、超长行、半包 —— 都会计进 ``bad_lines``。
        """
        self.assertEqual(self.client.bad_lines, 0)

    def test_typed_packets_reached_the_real_client(self) -> None:
        """**真实 B 客户端真的认出了 A 的 §3.5 报文。**

        上面 :class:`TestWireShape` 那几条查的是"裸 socket 收到的字节对不对"。
        这一条查的是另一半，也是更容易漏的那一半：**B 的代码读不读得懂**。
        两侧的契约可以各自自洽而互不认识 —— 一边把字段写成 ``gaze_quality``、
        另一边找 ``gaze_q``，两边单测全绿，合起来一条都用不上。

        所以这里断言的是 **B 自己的计数器**：收到过 rppg、没有未知类型、
        没有解析失败。计数器是 B 认了账的直接证据，不是我们替它推断的。

        ``unknown == 0`` 这条顺带钉住"别的类型一个都别冒出来" ——
        它在 B 侧才看得见（A 侧的白名单测试只查 A 自己发的那些）。
        """
        self.assertGreaterEqual(
            self.client.typed.counts.get("rppg", 0),
            3,
            f"真实 B 客户端一条 rppg 都没认出来（计数={self.client.typed.counts}）——"
            f" 两侧的字段名是不是对不上？",
        )
        self.assertEqual(self.client.typed.unknown, 0, "出现了白名单外的报文类型")
        self.assertEqual(self.client.typed.malformed, 0, "有报文没通过 B 侧的 §3.5 校验")

    def test_focus_packets_reached_the_real_client(self) -> None:
        """真实 B 客户端也认得出 ``focus`` 报文。

        与体征那条分开写，是因为两者在 B 侧走的是**两条独立的解析分支**
        （``_parse_rppg`` / ``_parse_focus``）。体征通了完全不能推出视线也通 ——
        而视线的失效方式恰恰是最安静的那种：B 把不认识的类型**丢弃并计数**，
        于是 A 这边照发不误，B 那边一条都没收到，谁的日志也不难看。
        """
        self.assertGreaterEqual(
            self.client.typed.counts.get("focus", 0),
            30,
            f"真实 B 客户端几乎没有收到 focus（计数={self.client.typed.counts}）——"
            f" 两项里有一侧没接上，或者视线通道根本没在跑",
        )

    def test_state_becomes_normal(self) -> None:
        """**字段名对上了没有，这一条说了算。**

        A 的 ``has_face`` / ``ear`` / ``emo_feature`` 只要有一个名字不对，
        B 就会静默走默认值，状态永远停在 ``absent``。所以"能判成 normal"
        不是一个平凡的断言，它是这条链路上唯一能证明字段对齐的观测。
        """
        deadline = time.monotonic() + 5.0
        state = self.evaluator.get_state()
        while time.monotonic() < deadline and state.state == "absent":
            time.sleep(0.1)
            state = self.evaluator.get_state()
        self.assertEqual(
            state.state,
            "normal",
            f"B 一直停在 {state.state}（{state.reason}）—— 检查 A 的字段名",
        )

    def test_no_staleness_jitter(self) -> None:
        """**观察期内一次都不能判失联。**

        抖动是间歇性的：持续十几秒地盯很容易恰好错过那 5 秒空窗，
        所以这条必须做成自动化断言。

        判据用 ``state.stale``，**不要用 reason 里有没有"未收到视觉数据"**：
        初始态那句是"**尚**未收到视觉数据"，正好包含那个子串 ——
        拿子串去匹配会把启动期误报成抖动（这是本文件第一版真实踩过的坑）。
        ``stale`` 是精确的：初始态为 False，失联态与断连态为 True。
        """
        problems: list[str] = []
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            state = self.evaluator.get_state()
            if state.stale:
                problems.append(state.reason)
            time.sleep(0.2)

        self.assertEqual(
            problems,
            [],
            f"出现失联抖动 {len(problems)} 次，例如：{problems[:3]}",
        )

    def test_estimator_never_saw_a_disconnect(self) -> None:
        """整段观察期内没有断连。

        ``sad`` 剧本开场有人脸，连接也该一直活着 —— 一旦断连，
        B 会立刻降级 ``absent``，C 那边就是卡片一闪。
        """
        self.assertFalse(self.evaluator.get_state().stale)


if __name__ == "__main__":
    unittest.main(verbosity=2)
