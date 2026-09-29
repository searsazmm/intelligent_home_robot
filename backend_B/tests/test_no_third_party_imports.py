# -*- coding: utf-8 -*-
"""守住"模块 B 运行期零第三方依赖"这条硬约束。

为什么需要这条守卫
------------------
"零依赖"是后端 B 的**刻意设计**，不是巧合：答辩现场少一个装包失败的可能，
就少一个坑。但这条约束非常容易被无意间破坏 —— 顺手加一行 ``import requests``
就能让整个模块在别人的机器上起不来，而在自己机器上（装过 requests）
**完全看不出问题**。这种"只在我这儿能跑"的故障最难查。

和 test_voice_engines.py 里那条守卫的分工
-----------------------------------------
那边 ``TestSingleEngineImportGuard`` 是**运行期**验证：在干净子进程里 import
语音层，检查 sys.modules 里有没有混进 sounddevice / vosk 等。它更强，但只管
``core/voice/`` 一个包。

这里是**静态**扫描，覆盖 backend_B 下所有非测试代码 —— 拦住 main.py、
core/dialogue.py 这些地方新加的第三方顶层 import。两条一起才没有死角。

为什么是"顶层"import
---------------------
语音层的第三方 import 全都写在**函数体内部**惰性执行，所以"顶层有没有出现
第三方名字"正好等价于"是不是无条件依赖了它"。函数体内的 import 是本项目
刻意的写法（见 core/voice/__init__.py 的说明），不算违规。

⚠️ 扫的是真源码，不 import 被测模块 —— 这条测试自身必须是零依赖的，
   否则它就没资格检查别人。
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

#: backend_B 自己的顶层名字。它们不是第三方，但也不在标准库里。
#: 加新包时记得同步这里（漏了会失败得很直白，是安全的默认方向）。
LOCAL_MODULES = frozenset({"config", "core"})

#: 不扫描的目录。tests/ 排除是因为测试文件可以合理地 import pytest 之类的
#: 开发期工具 —— 运行期依赖和开发期依赖是两回事，这条守卫只管前者。
SKIP_DIRS = frozenset({"__pycache__", "tests", "data", ".pytest_cache"})

BACKEND_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def top_level_imports(path: str) -> set:
    """取一个 .py 文件**顶层**（不在函数/类/if 里）import 的顶层模块名。

    ``if TYPE_CHECKING:`` 块里的 import 不会被算进来 —— 它在 ``ast.If``
    节点内部，本来就不会执行。这正是想要的行为。
    """
    with open(path, "r", encoding="utf-8") as fp:
        tree = ast.parse(fp.read(), path)

    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 是相对 import（from . import x），只在本包内，天然安全
            if node.level:
                continue
            names.add((node.module or "").split(".")[0])
    names.discard("")
    return names


def iter_source_files():
    """backend_B 下所有非测试的 .py 文件。"""
    for root, dirs, files in os.walk(BACKEND_B):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)


class TestNoThirdPartyImports(unittest.TestCase):

    def test_stdlib_module_names_is_usable(self):
        """先确认判据本身是好的，免得守卫悄悄退化成"永远通过"。

        ``sys.stdlib_module_names`` 是 3.10+ 才有的。本机是 3.14，有。
        但如果哪天在更老的解释器上跑，它会 AttributeError —— 那时应该
        明确失败，而不是让下面那条测试静默全绿。
        """
        self.assertTrue(hasattr(sys, "stdlib_module_names"),
                        "需要 Python 3.10+ 才有 sys.stdlib_module_names")
        self.assertGreater(len(sys.stdlib_module_names), 100)
        self.assertIn("socket", sys.stdlib_module_names)

    def test_scanner_finds_files(self):
        """扫描器自己别是空转 —— 找不到文件的话下面那条是假绿。"""
        files = list(iter_source_files())
        self.assertGreater(len(files), 5, f"只扫到 {len(files)} 个文件，路径大概是错的")
        names = {os.path.basename(p) for p in files}
        self.assertIn("main.py", names)
        self.assertIn("config.py", names)

    def test_scanner_detects_a_planted_import(self):
        """反向验证：喂一个含第三方 import 的临时文件，扫描器必须报出来。

        **这条是整份文件里最重要的一个。** 没有它，扫描器一旦因为 bug
        而永远返回空集合，上面几条测试全都照样绿 —— 守卫就成了摆设。
        """
        import tempfile

        source = "import requests\nimport os\n\ndef f():\n    import vosk\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as fp:
            fp.write(source)
            path = fp.name
        try:
            found = top_level_imports(path)
        finally:
            os.unlink(path)

        self.assertIn("requests", found, "顶层的第三方 import 必须被扫出来")
        self.assertIn("os", found)                 # 标准库也照样被扫出来（由调用方过滤）
        self.assertNotIn("vosk", found, "函数体内的 import 不该算顶层，它是本项目允许的写法")

    def test_no_third_party_top_level_imports(self):
        """正式断言：backend_B 运行期代码的顶层 import 全在标准库内。"""
        allowed = set(sys.stdlib_module_names) | LOCAL_MODULES
        offenders = []

        for path in iter_source_files():
            for name in sorted(top_level_imports(path) - allowed):
                rel = os.path.relpath(path, BACKEND_B)
                offenders.append(f"{rel}: import {name}")

        self.assertEqual(
            offenders, [],
            "backend_B 的运行期代码出现了第三方顶层 import：\n  "
            + "\n  ".join(offenders)
            + "\n\n零第三方依赖是刻意设计（见 requirements.txt 的说明）。"
            "\n如果只是可选功能，请把 import 移到**函数体内部**惰性执行，"
            "\n并配一句缺库时的中文降级提示；如果确实必须新增运行期依赖，"
            "\n请同步改 requirements.txt 与本文件的 LOCAL_MODULES。",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
