"""测试输出用的控制台准备。

Windows 控制台默认不是 UTF-8，而这个代码库的日志、用例名、断言消息
**全是中文**。不处理的话，一次全绿的测试跑出来是一屏乱码 —— 没人会
因此去查代码，但每个人都会多看两眼，而这正是"测试输出可信"的一部分。

两个入口都要覆盖，所以放在模块级而不是 ``if __name__ == "__main__"`` 里：

* ``python tests/test_v1_contract.py``
* ``python -m unittest discover -s tests``   ← 这个不会执行测试文件的 ``__main__``
"""

from __future__ import annotations

import contextlib
import sys


def prepare() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


prepare()
