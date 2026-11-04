# -*- coding: utf-8 -*-
"""紧急求助通知器的测试：sender 注入，全程不联网。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.emergency import (                                      # noqa: E402
    FamilyAlerter,
    KIND_CRISIS,
    KIND_EMERGENCY,
    build_alert_payload,
    is_emergency_text,
)


def fake_sender(results):
    """造一个按序返回结果/异常的假发送函数，并记录调用。"""
    calls = []
    queue = list(results)

    def send(url, payload, timeout):
        calls.append((url, payload, timeout))
        outcome = queue.pop(0) if queue else None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    send.calls = calls
    return send


class TestPatterns(unittest.TestCase):

    def test_full_phrases_hit(self):
        for text in ("救命啊", "快叫救护车", "我喘不上气了", "打120",
                     "摔了一跤起不来了", "胸口疼得厉害"):
            self.assertTrue(is_emergency_text(text), text)

    def test_short_words_do_not_hit(self):
        """词表刻意不收裸短词：误报一次，这个功能就没人信了。"""
        self.assertFalse(is_emergency_text("电视里在报警"))
        self.assertFalse(is_emergency_text("外面在打雷"))
        self.assertFalse(is_emergency_text(""))
        self.assertFalse(is_emergency_text(None))


class TestPayload(unittest.TestCase):

    def test_shape_is_flat_and_self_describing(self):
        payload = build_alert_payload(KIND_EMERGENCY, "救命", timestamp=123.0)
        self.assertEqual(payload["type"], "home_robot_emergency")
        self.assertEqual(payload["kind"], "emergency")
        self.assertEqual(payload["text"], "救命")
        self.assertEqual(payload["timestamp"], 123.0)
        self.assertIn("求助", payload["message"])

    def test_crisis_message_differs(self):
        crisis = build_alert_payload(KIND_CRISIS, "我不想活了")
        body = build_alert_payload(KIND_EMERGENCY, "救命")
        self.assertNotEqual(crisis["message"], body["message"])


class TestFamilyAlerter(unittest.TestCase):

    def test_no_urls_is_a_quiet_no_op(self):
        """没配 webhook：不抛异常、不发送，只留日志。"""
        alerter = FamilyAlerter(urls=(), clock=lambda: 100.0)
        self.assertFalse(alerter.configured)
        self.assertFalse(alerter.alert(KIND_EMERGENCY, "救命"))

    def test_alert_returns_true_and_sends(self):
        sender = fake_sender([None])
        alerter = FamilyAlerter(urls=["http://hook/a"], sender=sender,
                                clock=lambda: 100.0)
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "救命"))
        alerter.join(1.0)
        self.assertEqual(len(sender.calls), 1)
        url, payload, timeout = sender.calls[0]
        self.assertEqual(url, "http://hook/a")
        self.assertEqual(payload["kind"], "emergency")

    def test_cooldown_dedupes_within_window(self):
        """同一句被切成两段 / 老人慌乱中重复喊，只真通知一次。"""
        now = [100.0]
        sender = fake_sender([None])
        alerter = FamilyAlerter(urls=["http://hook/a"], sender=sender,
                                cooldown=300.0, clock=lambda: now[0])
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "救命"))
        now[0] += 10.0
        self.assertFalse(alerter.alert(KIND_EMERGENCY, "救命啊快来人"))
        now[0] += 400.0                              # 冷却已过
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "又喊了一次"))
        alerter.join(1.0)
        self.assertEqual(len(sender.calls), 2)

    def test_kinds_cool_down_independently(self):
        """身体求助和危机语句分开记账：一条通知不吞掉另一条。"""
        sender = fake_sender([None, None])
        alerter = FamilyAlerter(urls=["http://hook/a"], sender=sender,
                                cooldown=300.0, clock=lambda: 100.0)
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "救命"))
        self.assertTrue(alerter.alert(KIND_CRISIS, "我不想活了"))
        alerter.join(1.0)
        self.assertEqual(len(sender.calls), 2)

    def test_one_url_failing_does_not_block_the_others(self):
        sender = fake_sender([OSError("第一个挂了"), None])
        alerter = FamilyAlerter(
            urls=["http://hook/a", "http://hook/b"],
            sender=sender, clock=lambda: 100.0)
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "救命"))
        alerter.join(1.0)
        self.assertEqual([c[0] for c in sender.calls],
                         ["http://hook/a", "http://hook/b"])

    def test_alert_never_raises(self):
        """网络炸穿也不能把异常抛进对话线程。"""
        def exploding_sender(url, payload, timeout):
            raise RuntimeError("线程里什么都可能坏")

        alerter = FamilyAlerter(urls=["http://hook/a"], sender=exploding_sender,
                                clock=lambda: 100.0)
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "救命"))
        alerter.join(1.0)                            # 抛异常即失败

    def test_force_bypasses_cooldown(self):
        sender = fake_sender([None, None])
        alerter = FamilyAlerter(urls=["http://hook/a"], sender=sender,
                                cooldown=300.0, clock=lambda: 100.0)
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "第一次"))
        self.assertTrue(alerter.alert(KIND_EMERGENCY, "第二次", force=True))
        alerter.join(1.0)
        self.assertEqual(len(sender.calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
