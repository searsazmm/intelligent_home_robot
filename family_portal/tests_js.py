# -*- coding: utf-8 -*-
"""waveform.js（实时体征波形）的 Node 单测 —— pytest 包装。

为什么这么绕：波形引擎是 JS，Python 侧没有 DOM/canvas；但全量回归
只有 pytest 一个入口，JS 逻辑若只靠"手动浏览器验收"就会悄悄烂掉。
Node 自带 test runner 零依赖，家庭演示机装了 node 即可跑（刚才的
语法检查也用它）；没有 node 的环境则**跳过**而不是失败 —— JS 单测
是增强项，不该卡死纯 Python 环境。

真正的断言都在 js_tests/waveform.test.cjs（形状函数性质、速率驱动
相位、滚动窗口、尺寸自检、离线平线），这里只负责"它真的跑绿了"。
"""

import shutil
import subprocess
from pathlib import Path
from unittest import TestCase

_JS_DIR = Path(__file__).parent / "js_tests"


class WaveformJsTests(TestCase):
    """node 执行 waveform.test.cjs，非零退出 = JS 断言失败。"""

    def test_waveform_js_logic(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node 不可用：JS 波形单测仅在装有 node 的环境运行")
        proc = subprocess.run(
            [node, str(_JS_DIR / "waveform.test.cjs")],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(
            proc.returncode, 0,
            "waveform.js 单测失败：\n" + (proc.stdout + proc.stderr)[-2000:],
        )
        # 输出摘要里确认真的跑了用例（防止"文件不存在静默 0 用例"）
        self.assertIn("# pass 10", proc.stdout)
