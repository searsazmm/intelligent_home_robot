# -*- coding: utf-8 -*-
"""家属门户前端 JS（waveform.js / live.js）的 Node 单测 —— pytest 包装。

为什么这么绕：波形/轮询是 JS，Python 侧没有 DOM/canvas；但全量回归
只有 pytest 一个入口，JS 逻辑若只靠"手动浏览器验收"就会悄悄烂掉。
Node 自带 test runner 零依赖，装有 node 即可跑；没有 node 的环境则
**跳过**而不是失败 —— JS 单测是增强项，不该卡死纯 Python 环境。

真正的断言都在 js_tests/*.test.cjs：
- waveform.test.cjs —— 形状函数性质、速率驱动相位、滚动窗口、
  尺寸自检、离线平线（10 个用例）
- live.test.cjs —— 离线判定（服务端时间优先/阈值边界/回退）、
  信号徽标分级、"更新于/最后数据/今日概览"文案（14 个用例）
这里只负责"它们真的跑绿了"。
"""

import shutil
import subprocess
from pathlib import Path
from unittest import TestCase

_JS_DIR = Path(__file__).parent / "js_tests"

#: (测试文件, 期望通过用例数) —— 用例数核对防"文件不存在静默 0 用例"
_JS_CASES = [
    ("waveform.test.cjs", 10),
    ("live.test.cjs", 14),
]


class FamilyJsTests(TestCase):
    """node 执行各 *.test.cjs，非零退出 = JS 断言失败。"""

    def test_family_js_logic(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node 不可用：前端 JS 单测仅在装有 node 的环境运行")
        for fname, expected_pass in _JS_CASES:
            with self.subTest(file=fname):
                proc = subprocess.run(
                    [node, str(_JS_DIR / fname)],
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(
                    proc.returncode, 0,
                    fname + " 失败：\n" + (proc.stdout + proc.stderr)[-2000:],
                )
                self.assertIn("# pass %d" % expected_pass, proc.stdout)
