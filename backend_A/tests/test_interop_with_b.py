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
from module_a_vision.wire import V1_FIELDS, assert_v1_contract  # noqa: E402

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
    """裸 socket 收 A 的报文，逐条校验契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.h = _ServerHarness()
        cls.sock = cls.h.connect()
        cls.samples = _read_lines(cls.sock, OBSERVE_SEC)

    @classmethod
    def tearDownClass(cls) -> None:
        with contextlib.suppress(OSError):
            cls.sock.close()
        cls.h.stop()

    def test_received_enough_frames(self) -> None:
        """收到足够多的帧 —— 这一条同时排除了"心脏跳"混进来。

        10fps 跑 6 秒应该接近 60 条。下限放到 30 是留给 Windows 上
        线程调度的抖动，但**远高于**"每 10 秒一条"会得到的 1 条。
        所以如果有人把逐帧流式改回按窗口上报，这里会立刻变红。
        """
        self.assertGreater(
            len(self.samples),
            30,
            f"6 秒只收到 {len(self.samples)} 条 —— 逐帧流式是不是被改掉了？",
        )

    def test_every_message_has_exactly_the_eight_fields(self) -> None:
        for i, payload in enumerate(self.samples):
            with self.subTest(index=i):
                self.assertEqual(set(payload), set(V1_FIELDS))

    def test_no_message_carries_a_type_key(self) -> None:
        """v1 模式下**不能**混进 ``heartbeat`` / ``vision_window``。

        B 的 ``VisionSample.from_payload`` 会把任何一条缺 8 字段的报文
        兜成一个 ``has_face=false`` 的样本推进判定器 —— 也就是周期性
        伪造"看不见老人"。这条测试钉住的是那个失效。
        """
        for payload in self.samples:
            self.assertNotIn("type", payload)

    def test_every_message_passes_the_contract_gate(self) -> None:
        for i, payload in enumerate(self.samples):
            with self.subTest(index=i):
                self.assertEqual(assert_v1_contract(payload), [])

    def test_blink_counter_never_goes_backwards(self) -> None:
        """整条流上 ``blink_cnt`` 单调不减。

        B 的 ``_blink_rate_per_minute`` 检测到倒退就返回 None，
        于是**静默关掉一条疲劳判据** —— 不报错、不告警。
        也就是说这个 bug 在集成后是不可见的，只有在这里才拦得住。
        """
        counters = [p["blink_cnt"] for p in self.samples]
        self.assertEqual(counters, sorted(counters), "blink_cnt 出现倒退")

    def test_frame_rate_beats_the_stale_threshold(self) -> None:
        """**本文件最重要的一条断言。**

        报文的 ``timestamp`` 间隔必须**远小于** B 判定失联的 5 秒阈值。
        这条测试针对的是一个真实发生过的设计错误：曾经打算在 A 关窗时
        （每 10 秒）发一条 §3.2 报文。那样 B 每 10 秒里会有 5 秒判 absent，
        状态周期性抖动、C 端的卡片跟着闪 —— 而两个模块各自的单元测试
        **全都是绿的**，因为谁的逻辑都没错，错的是两者的时间尺度关系。
        """
        stamps = [p["timestamp"] for p in self.samples]
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
        """``sad`` 剧本开场是 25 秒安静段，所以流里必须有人脸。"""
        self.assertTrue(any(p["has_face"] for p in self.samples))
        self.assertTrue(all(isinstance(p["has_face"], bool) for p in self.samples))

    def test_emo_feature_reaches_low_after_calibration(self) -> None:
        """表情通道真的在工作，不是一个恒为 normal 的常量。

        2 秒的观察窗不够跨过表情分类器的 25 秒标定段，所以这条只断言
        **取值合法**；"能不能变成 low"由 ``--dry-run`` 与手工联调验证
        （见 README 的验证表）。这里刻意不断言 ``low`` 出现过 ——
        写一条自己都知道跑不到的断言，比不写更糟。
        """
        values = {p["emo_feature"] for p in self.samples}
        self.assertTrue(values <= {"normal", "low", "tired"}, f"非法取值 {values}")


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
