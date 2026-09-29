# -*- coding: utf-8 -*-
"""状态双通道扇出：8001（纯文本）+ 8002（JSON）必须同进同退。

为什么单独一个文件守这件事：这两条通道的**耦合是隐形的**。
8002 的那份推送寄生在 8001 的去重判断上（``status_server.publish`` 返回 True
才推），所以"哪里会推 8002"完全取决于"哪里调了 publish"，而不是由某段
独立的广播代码决定。这个结构本身没问题，但它坏起来非常安静：

  · 少调一处 → C 的表情卡在旧状态，等到 15 秒心跳才跟上，
    而日志照样写着"推送成功"（handle_chat 里那个状态覆盖就踩过一次）；
  · 日志只数 8001 的客户端 → C 只连 8002 时永远写「0 个客户端」，
    看着像广播断了，其实一切正常（这是这条日志被改掉的原因）。

所以下面既有**行为**断言（真的推没推），也有**日志文本**断言（数字对不对）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest import mock

_BACKEND_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_B)

#: backend_B/main.py 在这个测试进程里的**专用**模块名。
#: 绝不能叫 "main" —— 那个名字可能已经被 backend_A/main.py 占了（见下面的说明）。
_MAIN_MODULE_NAME = "backend_b_main"

import config
from core.ui_channel import StatusBroadcaster


def load_backend_b_main():
    """按**文件路径**加载 backend_B/main.py，**不要**写 ``import main``。

    ⚠️ 仓库里有**两个** main.py（backend_A 和 backend_B）。
    整仓一起跑 pytest 时，backend_A 的测试会先把 backend_A/ 放上 sys.path
    并注册 ``sys.modules["main"]``，于是 ``import main`` 拿到的是模块 A 那个，
    报 ``cannot import name 'BackendB'``。

    这个坑的恶劣之处在于**单跑这个文件时完全正常** —— 只有把整套测试一起跑
    才炸，很容易被当成偶发。按路径加载就与 sys.path 顺序无关了。

    （``config`` / ``core`` 没有这个问题：backend_A 顶层没有这两个名字。）
    """
    cached = sys.modules.get(_MAIN_MODULE_NAME)
    if cached is not None:
        return cached      # 模块体只是定义与 import，但没必要反复执行

    path = os.path.join(_BACKEND_B, "main.py")
    spec = importlib.util.spec_from_file_location(_MAIN_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    # 先登记再执行：main.py 内部若有自引用或 dataclass 解析会用到 sys.modules
    sys.modules[_MAIN_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def make_broadcaster() -> StatusBroadcaster:
    """造一个没有 start() 过的 StatusBroadcaster。

    不监听端口是**故意的**：``_broadcast`` 在客户端列表为空时返回 0，
    正是我们要的"没人连"场景，而且这条测试不需要碰网络、不用等超时。
    """
    return StatusBroadcaster()


class TestPublishLogCounts(unittest.TestCase):
    """修掉的那条误导日志：C 明明连着，却写「已推送给 0 个客户端」。"""

    def setUp(self):
        self.server = make_broadcaster()
        self.addCleanup(self.server.stop)

    def test_two_channel_counts_are_logged(self):
        """默认演示配置：C 只连 8002，日志要能看出「对话通道有 1 个」。"""
        with self.assertLogs("core.ui_channel", level="INFO") as captured:
            self.server.publish(config.STATE_SAD, mirror_clients=1)

        line = "".join(captured.output)
        self.assertIn("状态通道 0 个客户端", line)
        self.assertIn("对话通道 1 个", line)

    def test_mirroring_to_nobody_is_distinguishable_from_not_mirroring(self):
        """``mirror_clients=0`` 与 ``None`` 必须是两句不同的话。

        这正是参数用 None 而不是 0 当默认值的理由：前者是
        「镜像了，但 8002 上没人」，后者是「这次调用根本不做镜像」。
        两者混成一句会让这条日志失去它存在的唯一意义。
        """
        with self.assertLogs("core.ui_channel", level="INFO") as mirrored:
            self.server.publish(config.STATE_SAD, mirror_clients=0)
        with self.assertLogs("core.ui_channel", level="INFO") as unmirrored:
            self.server.publish(config.STATE_TIRED)

        self.assertIn("对话通道 0 个", "".join(mirrored.output))
        self.assertNotIn("对话通道", "".join(unmirrored.output))
        # 不镜像的那次也要说清数字，不能含糊
        self.assertIn("状态通道 0 个客户端", "".join(unmirrored.output))

    def test_heartbeat_does_not_log_a_change(self):
        """心跳推了但不该写「状态变更」—— 否则 15 秒一条，日志里分不清真假变化。"""
        self.server.publish(config.STATE_SAD, mirror_clients=1)
        with self.assertNoLogs("core.ui_channel", level="INFO") as _:
            self.server.publish(config.STATE_SAD, mirror_clients=1)


class TestPublishStateFanout(unittest.TestCase):
    """``BackendB._publish_state``：推 8001 与镜像 8002 必须成对。

    用 duck-typed 替身直接调用未绑定方法，而不是构造整个 BackendB ——
    那要拉起视觉源、历史文件、语音层，为了验一条"两行代码有没有成对执行"
    不值得。这里要钉的正是这两行的配对关系，替身刚好够。
    """

    def make_stub(self, published: bool):
        stub = mock.Mock()
        stub.status_server.publish.return_value = published
        stub.chat_server.client_count.return_value = 7
        return stub

    def _call(self, stub, state=config.STATE_SAD, reason="测试"):
        # 未绑定调用：跳过 __init__，只借这个方法本身的逻辑
        load_backend_b_main().BackendB._publish_state(stub, state, reason)

    def test_chat_channel_is_mirrored_when_status_channel_sent(self):
        stub = self.make_stub(published=True)
        self._call(stub)
        stub.chat_server.broadcast_state.assert_called_once_with(
            config.STATE_SAD, "测试")

    def test_chat_channel_is_not_mirrored_when_deduped(self):
        """8001 那边判为「没变化且没到心跳」时不能镜像 ——

        否则 C 会被同一状态反复刷屏，表情每 0.2 秒闪一次。
        """
        stub = self.make_stub(published=False)
        self._call(stub)
        stub.chat_server.broadcast_state.assert_not_called()

    def test_real_client_count_is_reported_to_the_log(self):
        """传进 publish 的必须是**真实**的 8002 客户端数，不能图省事写死。"""
        stub = self.make_stub(published=True)
        self._call(stub)
        _, kwargs = stub.status_server.publish.call_args
        self.assertEqual(kwargs["mirror_clients"], 7)

    def test_reason_reaches_the_chat_channel(self):
        """对话导致的状态覆盖要带上原因 —— C 的日志会把它显示出来，
        否则 C 端只会看到状态变了却不知道为什么。"""
        stub = self.make_stub(published=True)
        self._call(stub, reason="对话判定：用户说很累")
        args, _ = stub.chat_server.broadcast_state.call_args
        self.assertEqual(args[1], "对话判定：用户说很累")


class TestHandleChatPublishesToBothChannels(unittest.TestCase):
    """钉住**调用点**：对话导致的状态覆盖必须走 ``_publish_state``。

    上面那一类测的是 ``_publish_state`` 自己有没有成对；这一类测的是
    ``handle_chat`` 有没有用它。两者缺一不可 —— 修的那个缺口正是
    「方法写对了，但 handle_chat 绕过它直接调了 status_server.publish」，
    只测方法本身是发现不了的。
    """

    def test_state_override_goes_through_the_fanout_helper(self):
        BackendB = load_backend_b_main().BackendB

        stub = mock.Mock()
        # 视觉判定是 normal、对话结论是 sad —— 两者不同才会走到覆盖分支
        stub.evaluator.get_state.return_value = mock.Mock(state=config.STATE_NORMAL)
        stub.dialogue.respond.return_value = mock.Mock(
            state=config.STATE_SAD, reply="我在呢", reason="用户说很累",
            emotion_label="negative", emotion_score=-0.6, intent="chat")

        BackendB.handle_chat(stub, "我有点累")

        # 关键断言：状态覆盖一定经过那条"推 8001 + 镜像 8002"的路径
        stub._publish_state.assert_called_once()
        state, reason = stub._publish_state.call_args[0]
        self.assertEqual(state, config.STATE_SAD)
        self.assertIn("对话判定", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
