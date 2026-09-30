# -*- coding: utf-8 -*-
"""到模块 B 的连接：状态接收、去重、重连、通道优先级的单元测试。

跑法（在 frontend_C 目录下）：
    python -m pytest tests/ -v
    python tests/test_state_client.py

这里起的是**真的** TCP 连接（临时端口上的假服务端），不对 socket 打桩。
理由：这段代码要防的恰恰是分帧、半包、重连时序这些真实网络行为，
把 socket 换成假对象之后，测试和实现就变成了自说自话。
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_core.expressions import STATE_NORMAL, STATE_SAD, STATE_TIRED
from c_core.state_client import SOURCE_CHAT, SOURCE_STATUS, BackendLink

#: 所有测试的超时上限。够慢机器用，又不会让挂掉的测试拖住整条流水线。
TIMEOUT = 5.0


class FakeBackend:
    """最小的假模块 B：listen、接受连接、按脚本逐行发数据。

    刻意不做任何「聪明」的事：不发心跳、不校验客户端、不自动回包。
    真实服务的花样越多，测试失败时就越难判断是实现错了还是假服务端错了。
    """

    def __init__(self, port: int = 0) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", port))
        self._server.listen(5)
        self._server.settimeout(0.2)
        self.port = self._server.getsockname()[1]
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False
        self._accepted = threading.Event()
        #: **只增不减**的连接计数。用它而不是事件来等「有新的连接进来」——
        #: Event 是一次性的，第一次连上之后永远为真，于是「重连成功」的断言
        #: 会在根本没有重连的情况下直接通过。
        self._accept_count = 0
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    # -- 生命周期 --------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._server.accept()
            except (socket.timeout, OSError):
                continue
            conn.settimeout(0.2)
            with self._lock:
                self._clients.append(conn)
                self._accept_count += 1
            self._accepted.set()
            threading.Thread(target=self._drain, args=(conn,), daemon=True).start()

    @staticmethod
    def _drain(conn: socket.socket) -> None:
        """收走客户端发来的 ping / chat。

        必须收：客户端连上会先发 ping，不读的话发送缓冲区迟早会满，
        客户端再发就阻塞 —— 症状是「连接建立了但状态收不到」，很难归因。
        """
        try:
            while conn.recv(4096):
                pass
        except OSError:
            pass

    def accept_count(self) -> int:
        """历史累计接受过的连接数，只增不减。"""
        with self._lock:
            return self._accept_count

    def wait_for_accept(self, count: int, timeout: float = TIMEOUT) -> bool:
        """等累计连接数达到 ``count``。

        「等第 N 次连接」是重连测试唯一正确的等待方式：断连之后
        ``client_count()`` 会掉到 0，用事件又会因为第一次连接而永远为真。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.accept_count() >= count:
                return True
            self._accepted.wait(0.05)
            self._accepted.clear()
        return False

    def wait_for_client(self, timeout: float = TIMEOUT) -> bool:
        """等**首次**连接。"""
        return self.wait_for_accept(1, timeout)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    # -- 发数据 ----------------------------------------------------------

    def send_raw(self, data: bytes) -> None:
        """发原始字节。半包测试要自己控制切分点，所以留这个口子。"""
        with self._lock:
            clients = list(self._clients)
        for conn in clients:
            try:
                conn.sendall(data)
            except OSError:
                pass

    def send_line(self, text: str) -> None:
        self.send_raw((text + "\n").encode("utf-8"))

    def send_json(self, obj: dict) -> None:
        self.send_line(json.dumps(obj, ensure_ascii=False))

    def send_state(self, state: str, reason: str = "") -> None:
        self.send_json({"type": "state", "state": state, "reason": reason})

    # -- 断连 ------------------------------------------------------------

    def drop_clients(self) -> None:
        """掐断当前所有连接，但继续 listen（模拟模块 B 重启）。"""
        with self._lock:
            clients, self._clients = list(self._clients), []
        for conn in clients:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._closed = True
        self.drop_clients()
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)


class Recorder:
    """收集回调，提供可等待的事件。"""

    def __init__(self) -> None:
        self.states: list[tuple[str, str, str]] = []
        self.vais: list[dict] = []
        self.replies: list[dict] = []
        self.proactives: list[dict] = []
        self.links: list[tuple[str, bool]] = []
        self._event = threading.Event()
        self._lock = threading.Lock()

    def on_state(self, state: str, reason: str, source: str) -> None:
        with self._lock:
            self.states.append((state, reason, source))
        self._event.set()

    def on_vai(self, msg: dict) -> None:
        with self._lock:
            self.vais.append(msg)
        self._event.set()

    def on_reply(self, msg: dict) -> None:
        with self._lock:
            self.replies.append(msg)
        self._event.set()

    def on_proactive(self, msg: dict) -> None:
        with self._lock:
            self.proactives.append(msg)
        self._event.set()

    def on_link(self, source: str, connected: bool) -> None:
        with self._lock:
            self.links.append((source, connected))
        self._event.set()

    def wait_for_states(self, count: int, timeout: float = TIMEOUT) -> bool:
        """等状态回调攒够 ``count`` 次。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.states) >= count:
                    return True
            self._event.wait(0.05)
            self._event.clear()
        return False

    def wait_for_vai(self, count: int = 1, timeout: float = TIMEOUT) -> bool:
        """等专注度展示报文攒够 ``count`` 条。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.vais) >= count:
                    return True
            self._event.wait(0.05)
            self._event.clear()
        return False

    def wait_for_link(self, connected: bool, timeout: float = TIMEOUT) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if any(c is connected for _, c in self.links):
                    return True
            self._event.wait(0.05)
            self._event.clear()
        return False

    def last_state(self) -> str | None:
        with self._lock:
            return self.states[-1][0] if self.states else None

    def reset(self) -> None:
        with self._lock:
            self.states.clear()
        self._event.clear()


def build_link(recorder: Recorder, port: int, status_port: int | None = None) -> BackendLink:
    """构造一个退避很短的连接器 —— 测试不该为了等 1 秒退避而变慢。"""
    return BackendLink(
        host="127.0.0.1",
        chat_port=port,
        status_port=status_port,
        on_state=recorder.on_state,
        on_vai=recorder.on_vai,
        on_reply=recorder.on_reply,
        on_proactive=recorder.on_proactive,
        on_link=recorder.on_link,
        reconnect_initial=0.1,
        reconnect_max=0.3,
        socket_timeout=0.2,
    )


class LinkTestCase(unittest.TestCase):
    """公共脚手架：起假服务端、建连接、结束清理。"""

    def setUp(self) -> None:
        self.server = FakeBackend()
        self.recorder = Recorder()
        self.link = None

    def tearDown(self) -> None:
        if self.link is not None:
            self.link.stop()
        self.server.close()

    def start_link(self, status_port: int | None = None) -> BackendLink:
        self.link = build_link(self.recorder, self.server.port, status_port)
        self.link.start()
        self.assertTrue(self.server.wait_for_client(), "客户端没有连上来")
        return self.link


class TestConnection(LinkTestCase):

    def test_connects_to_chat_port(self):
        self.start_link()
        self.assertEqual(self.server.accept_count(), 1)

    def test_sends_ping_on_connect(self):
        """连上先探活，日志才能区分「连上了」和「连上了但没数据」。

        ping 是发给服务端的，这里只能从「服务端没有因为收到脏数据而挂掉」
        间接确认 —— 真正的格式断言在 test_protocol.py 里。
        """
        self.start_link()
        time.sleep(0.2)
        self.assertEqual(self.server.client_count(), 1)


class TestStateReception(LinkTestCase):

    def test_receives_state_from_chat_channel(self):
        self.start_link()
        self.server.send_state(STATE_SAD, "检测到情绪低落")
        self.assertTrue(self.recorder.wait_for_states(1))
        state, reason, source = self.recorder.states[0]
        self.assertEqual(state, STATE_SAD)
        self.assertEqual(reason, "检测到情绪低落")
        self.assertEqual(source, SOURCE_CHAT)

    def test_unknown_state_is_normalized_to_normal(self):
        """api_doc §4.3：收到未知内容时默认展示 normal，而不是崩或者显示空白。"""
        self.start_link()
        self.server.send_state("angry")
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.last_state(), STATE_NORMAL)

    def test_state_field_of_wrong_type_does_not_kill_the_thread(self):
        """state 是数字/列表时不许把连接线程带走。"""
        self.start_link()
        self.server.send_json({"type": "state", "state": [1, 2, 3]})
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.last_state(), STATE_NORMAL)

        # 线程还活着 —— 后续状态仍能收到
        self.recorder.reset()
        self.server.send_state(STATE_TIRED)
        self.assertTrue(self.recorder.wait_for_states(1), "线程在脏数据之后死了")
        self.assertEqual(self.recorder.last_state(), STATE_TIRED)

    def test_malformed_json_line_does_not_kill_the_thread(self):
        self.start_link()
        self.server.send_line("{这不是 JSON")
        self.server.send_line("")                      # 空行
        self.server.send_line("[1, 2, 3]")             # 顶层不是对象
        self.server.send_state(STATE_SAD)              # 之后仍然要能正常收
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.last_state(), STATE_SAD)

    def test_unknown_message_type_is_ignored(self):
        self.start_link()
        self.server.send_json({"type": "未来才有的报文", "state": STATE_SAD})
        self.server.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1))
        # 状态只因为那条正规的 state 报文而改变，未知类型不生效
        self.assertEqual(self.recorder.states, [(STATE_SAD, "", SOURCE_CHAT)])


class TestDeduplication(LinkTestCase):

    def test_repeated_state_is_dispatched_once(self):
        """B 每 15 秒心跳重发同一状态。不去重的话表情每 15 秒闪一下。

        这是本节最重要的一条 —— 它防的是一个「功能都对但演示时看得出抖」的缺陷。
        """
        self.start_link()
        for _ in range(5):
            self.server.send_state(STATE_SAD)
            time.sleep(0.05)
        self.assertTrue(self.recorder.wait_for_states(1))
        time.sleep(0.3)
        self.assertEqual(len(self.recorder.states), 1)
        self.assertEqual(self.recorder.last_state(), STATE_SAD)

    def test_state_change_is_dispatched(self):
        self.start_link()
        self.server.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1))
        self.server.send_state(STATE_TIRED)
        self.assertTrue(self.recorder.wait_for_states(2))
        self.assertEqual([s for s, _, _ in self.recorder.states], [STATE_SAD, STATE_TIRED])


class TestFraming(LinkTestCase):

    def test_line_split_across_packets(self):
        """半包：一条报文被 TCP 切成两段，必须拼回来。"""
        self.start_link()
        payload = json.dumps({"type": "state", "state": STATE_SAD}).encode("utf-8")
        self.server.send_raw(payload[:10])
        time.sleep(0.15)
        self.server.send_raw(payload[10:] + b"\n")
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.last_state(), STATE_SAD)

    def test_multiple_lines_in_one_packet(self):
        """粘包：一个 recv 里回来三条报文，必须逐条派发。"""
        self.start_link()
        blob = b"".join(
            json.dumps({"type": "state", "state": s}).encode("utf-8") + b"\n"
            for s in (STATE_SAD, STATE_TIRED, STATE_NORMAL)
        )
        self.server.send_raw(blob)
        self.assertTrue(self.recorder.wait_for_states(3))
        self.assertEqual(
            [s for s, _, _ in self.recorder.states],
            [STATE_SAD, STATE_TIRED, STATE_NORMAL],
        )

    def test_crlf_line_endings_are_tolerated(self):
        """对端误发 \\r\\n 时不能把 \\r 留在状态字符串里。"""
        self.start_link()
        self.server.send_raw(b'{"type":"state","state":"sad"}\r\n')
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.last_state(), STATE_SAD)


class TestDeathAndRecovery(LinkTestCase):

    def test_disconnect_falls_back_to_normal(self):
        """api_doc §4.3：断连异常时默认展示 normal。"""
        self.start_link()
        self.server.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1))

        self.server.drop_clients()
        self.assertTrue(self.recorder.wait_for_states(2), "断连后没有回落状态")
        state, reason, source = self.recorder.states[-1]
        self.assertEqual(state, STATE_NORMAL)
        self.assertEqual(source, SOURCE_CHAT)
        self.assertIn("断开", reason)

    def test_link_callback_reports_disconnect(self):
        self.start_link()
        self.assertTrue(self.recorder.wait_for_link(True))
        self.server.drop_clients()
        self.assertTrue(self.recorder.wait_for_link(False))

    def test_reconnects_and_resumes_receiving(self):
        """模块 B 重启后 C 必须自己接回来，不需要人工重启界面。"""
        self.start_link()
        first = self.server.accept_count()

        self.server.drop_clients()
        self.assertTrue(self.recorder.wait_for_link(False))

        # 等**第 2 次**连接，而不是「有没有连接」—— 后者第一次就为真了
        self.assertTrue(
            self.server.wait_for_accept(first + 1), "没有重连上来"
        )
        self.recorder.reset()
        self.server.send_state(STATE_TIRED)
        self.assertTrue(self.recorder.wait_for_states(1), "重连后收不到状态")
        self.assertEqual(self.recorder.last_state(), STATE_TIRED)

    def test_reconnect_after_backend_is_not_running_yet(self):
        """启动顺序反了（C 先起、B 后起）也必须能自动接上。

        这是最容易被忽略的一条：演示时经常先开界面再开后端。
        如果重连只试一次，用户看到的就是「界面一直是正常脸，坏了」。
        """
        port = self.server.port
        self.server.close()                 # 端口彻底消失，模拟 B 还没启动

        self.link = build_link(self.recorder, port)
        self.link.start()
        time.sleep(0.4)                     # 让它先失败几次
        self.assertEqual(self.recorder.states, [], "端口不通时不该派发任何状态")

        # 现在让 B 「启动」到同一个端口上
        try:
            self.server = FakeBackend(port=port)
        except OSError as exc:
            self.skipTest(f"临时端口 {port} 没能立即复用（{exc}），跳过；不影响被测逻辑")

        self.assertTrue(self.recorder.wait_for_link(True), "模块 B 起来后没有自动接上")
        # 客户端的 connect() 返回会**早于**服务端的 accept() 完成 ——
        # 此时连接还没进假服务端的 _clients，直接发等于发给空列表。
        # 所以必须等服务端真的接受到了，再发数据。
        self.assertTrue(self.server.wait_for_accept(1), "服务端没有接受连接")
        self.server.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1), "接上了但收不到状态")
        self.assertEqual(self.recorder.last_state(), STATE_SAD)


class TestMessageKinds(LinkTestCase):

    def test_reply_is_dispatched_and_does_not_change_state(self):
        """reply 是对话内容，不携带状态 —— 表情不该因为它变。"""
        self.start_link()
        self.server.send_json({"type": "reply", "text": "我在呢", "state": STATE_NORMAL})
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline and not self.recorder.replies:
            time.sleep(0.05)
        self.assertEqual(len(self.recorder.replies), 1)
        self.assertEqual(self.recorder.replies[0]["text"], "我在呢")
        self.assertEqual(self.recorder.states, [])

    def test_proactive_carries_state_and_updates_expression(self):
        """主动关怀带状态一起发（api_doc §5 V1.2）。

        否则会出现「B 在说关心的话、表情却还是正常脸」的割裂。
        """
        self.start_link()
        self.server.send_json({
            "type": "proactive", "text": "您还好吗？", "state": STATE_SAD,
            "reason": "情绪持续低落",
        })
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline and not self.recorder.proactives:
            time.sleep(0.05)
        self.assertEqual(len(self.recorder.proactives), 1)
        self.assertEqual(self.recorder.proactives[0]["text"], "您还好吗？")
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.last_state(), STATE_SAD)

    def test_proactive_without_state_leaves_expression_alone(self):
        self.start_link()
        self.server.send_json({"type": "proactive", "text": "早上好"})
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline and not self.recorder.proactives:
            time.sleep(0.05)
        self.assertEqual(len(self.recorder.proactives), 1)
        time.sleep(0.2)
        self.assertEqual(self.recorder.states, [])

    def test_pong_is_ignored(self):
        self.start_link()
        self.server.send_json({"type": "pong"})
        self.server.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.states, [(STATE_SAD, "", SOURCE_CHAT)])


class TestVaiReception(LinkTestCase):
    """专注度展示报文（api_doc §5.6）：**只派发，绝不碰状态**。

    这条约束在 C 侧最容易被"顺手"破坏 —— 报文里既然有 ``status``、
    看着又很像一个状态，接着写一句 ``self._apply_state(msg["status"], ...)``
    是极自然的动作。所以这里用**同一个连接**同时喂 vai 与 state，
    断言状态序列只由后者决定。
    """

    VAI = {
        "type": "vai",
        "index": 63.5,
        "index_status": "研究趋势（非认知专注）",
        "status": "有效",
        "reason": "校准已锁定",
        "modalities": ["gaze", "pose", "eye_open"],
        "recent_modalities": ["gaze", "pose", "eye_open"],
        "modality_config_id": "凝视+头姿+睁眼",
        "valid_seconds": 42.0,
        "note": "研究趋势（非认知专注）；日常参考，非医疗结论",
        "timestamp": 1.0,
    }

    def test_vai_is_dispatched_with_every_field_intact(self):
        """整条报文原样交给上层 —— 挑字段是 display_text 的事，不是传输层的事。"""
        self.start_link()
        self.server.send_json(self.VAI)
        self.assertTrue(self.recorder.wait_for_vai())
        self.assertEqual(self.recorder.vais[0], self.VAI)

    def test_vai_does_not_change_the_expression(self):
        """收到专注度**不能**让表情变 —— 那会变成第二个状态源。"""
        self.start_link()
        self.server.send_json(self.VAI)
        self.assertTrue(self.recorder.wait_for_vai())
        time.sleep(0.2)
        self.assertEqual(self.recorder.states, [])

    def test_a_stray_state_field_in_vai_is_still_not_a_state(self):
        """就算报文里真的混进了 ``state`` 字段，也不许它改状态。

        B 侧目前的 ``build_vai_message`` 不带这个字段，但"以后有人加上"是
        很可能发生的事 —— 那时这条测试会站出来说话，而不是悄悄多出一个状态源。
        """
        self.start_link()
        self.server.send_json({**self.VAI, "state": STATE_SAD})
        self.assertTrue(self.recorder.wait_for_vai())
        self.server.send_state(STATE_TIRED)
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.states, [(STATE_TIRED, "", SOURCE_CHAT)])

    def test_index_null_is_passed_through_as_none(self):
        """``index: null`` = 没有分数（不是 0 分）。传输层**不许**把它变成 0。"""
        self.start_link()
        self.server.send_json({**self.VAI, "index": None,
                               "index_status": "有效观察时长不足", "status": "观察不足"})
        self.assertTrue(self.recorder.wait_for_vai())
        self.assertIsNone(self.recorder.vais[0]["index"])
        self.assertNotEqual(self.recorder.vais[0]["index"], 0)

    def test_malformed_vai_does_not_kill_the_thread(self):
        """``vai`` 字段类型乱来（index 是字符串、modalities 是数字）也不许带走线程。"""
        self.start_link()
        self.server.send_json({"type": "vai", "index": "高", "modalities": 42})
        self.assertTrue(self.recorder.wait_for_vai())
        self.server.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1), "线程在脏报文之后死了")

    def test_error_message_does_not_kill_the_thread(self):
        self.start_link()
        self.server.send_json({"type": "error", "message": "模块 B 内部错误"})
        self.server.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1))


class TestStatusChannel(unittest.TestCase):
    """8001 状态通道（``--also-status``）。8001 是纯文本，不是 JSON。"""

    def setUp(self) -> None:
        self.chat = FakeBackend()
        self.status = FakeBackend()
        self.recorder = Recorder()
        self.link = build_link(self.recorder, self.chat.port, self.status.port)
        self.link.start()

    def tearDown(self) -> None:
        self.link.stop()
        self.chat.close()
        self.status.close()

    def test_plain_text_state_line_is_accepted(self):
        """api_doc §4.2：8001 上就是四种纯文本状态之一，没有 JSON 外壳。"""
        self.assertTrue(self.status.wait_for_client())
        self.status.send_line(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1))
        state, _, source = self.recorder.states[0]
        self.assertEqual(state, STATE_SAD)
        self.assertEqual(source, SOURCE_STATUS)

    def test_unknown_plain_text_falls_back_to_normal(self):
        self.assertTrue(self.status.wait_for_client())
        self.status.send_line("something_else")
        self.assertTrue(self.recorder.wait_for_states(1))
        self.assertEqual(self.recorder.last_state(), STATE_NORMAL)

    def test_status_channel_wins_over_chat_channel(self):
        """两条通道都在推状态时必须只有一个权威源，否则表情会来回抖。

        api_doc §4 把 8001 定为专用状态通道，所以它以 8001 为准。
        """
        self.assertTrue(self.status.wait_for_client())
        self.assertTrue(self.chat.wait_for_client())
        time.sleep(0.2)

        self.status.send_line(STATE_TIRED)
        self.assertTrue(self.recorder.wait_for_states(1))

        self.chat.send_state(STATE_SAD)      # 8002 说的应该被忽略
        time.sleep(0.3)
        self.assertEqual(len(self.recorder.states), 1)
        self.assertEqual(self.recorder.last_state(), STATE_TIRED)



class TestStatusChannelDown(unittest.TestCase):
    """8001 连不上（端口没人监听）时的降级行为。

    单独一个类，因为 TestStatusChannel 的 setUp 起的是**活的**状态服务端 ——
    在那边断言「8001 没连上」会和脚手架自相矛盾。
    """

    def setUp(self) -> None:
        self.chat = FakeBackend()
        self.recorder = Recorder()
        # 占一个端口再立刻放掉 —— 这个端口上没有任何人监听
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.dead_port = probe.getsockname()[1]
        probe.close()

        self.link = build_link(self.recorder, self.chat.port, self.dead_port)
        self.link.start()

    def tearDown(self) -> None:
        self.link.stop()
        self.chat.close()

    def test_chat_channel_is_authoritative_when_status_channel_is_down(self):
        """8001 连不上时，8002 是唯一的状态源，它说的话必须算数。

        否则会出现最糟的组合：用户只连了 8002（默认配置！），
        而实现又无条件地把 8002 的状态当成「次优先」丢掉 —— 表情永远不动。
        """
        self.assertTrue(self.chat.wait_for_client())
        self.chat.send_state(STATE_SAD)
        self.assertTrue(self.recorder.wait_for_states(1), "8002 的状态被错误地丢掉了")
        state, _, source = self.recorder.states[0]
        self.assertEqual(state, STATE_SAD)
        self.assertEqual(source, SOURCE_CHAT)

    def test_does_not_keep_retrying_status_channel_forever_in_this_test(self):
        """8001 连不上不该影响 8002 的主流程 —— 两条线程各自独立退避。"""
        self.assertTrue(self.chat.wait_for_client())
        self.chat.send_state(STATE_TIRED)
        self.assertTrue(self.recorder.wait_for_states(1))
        self.chat.send_state(STATE_NORMAL)
        self.assertTrue(self.recorder.wait_for_states(2))
        self.assertEqual(
            [s for s, _, _ in self.recorder.states], [STATE_TIRED, STATE_NORMAL]
        )


class TestLifecycle(unittest.TestCase):
    """启动与关闭。"""

    def test_stop_is_idempotent_and_prompt(self):
        server = FakeBackend()
        recorder = Recorder()
        link = build_link(recorder, server.port)
        try:
            link.start()
            self.assertTrue(server.wait_for_client())
            started = time.monotonic()
            link.stop()
            link.stop()          # 第二次调用不该抛
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 3.0, f"stop() 用了 {elapsed:.1f}s，太慢")
        finally:
            server.close()

    def test_stop_without_start_does_not_raise(self):
        server = FakeBackend()
        try:
            link = build_link(Recorder(), server.port)
            link.stop()
        finally:
            server.close()

    def test_stop_when_backend_never_existed(self):
        """端口上什么都没有时也要能干净退出 —— 否则「B 没启动」就成了死锁。"""
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        link = build_link(Recorder(), port)
        link.start()
        time.sleep(0.3)
        started = time.monotonic()
        link.stop()
        self.assertLess(time.monotonic() - started, 3.0)

    def test_send_chat_returns_false_when_not_connected(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        link = build_link(Recorder(), port)
        try:
            self.assertFalse(link.send_chat("测试"))
        finally:
            link.stop()

    def test_no_threads_leak_after_stop(self):
        server = FakeBackend()
        before = threading.active_count()
        link = build_link(Recorder(), server.port)
        try:
            link.start()
            self.assertTrue(server.wait_for_client())
            link.stop()
            time.sleep(0.4)
            self.assertLessEqual(
                threading.active_count(), before + 1,
                "stop() 之后还有线程没退出",
            )
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
