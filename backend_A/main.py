"""模块 A 的入口垫片。

存在的唯一理由是对齐团队仓库的运行约定：每个后端是自己的运行根，
``cd backend_A && python main.py``（``backend_B`` 就是这么用的）。

真正的入口是 :func:`module_a_vision.main.main`。这里不做任何加工 ——
参数解析、日志、退出码全在那边，避免出现两套行为。

    python main.py                       # 合成源 + v1 报文，起 8000 服务
    python main.py --scenario sad        # 换剧本
    python main.py --dry-run             # 不起服务，只打印报文并跑契约检查
    python main.py --emit v2             # 换成原生的 10 秒嵌套窗口

⚠️ **必须从本目录运行。** ``module_a_vision`` 内部用的是
``from shared.xxx import …`` 这种以本目录为根的绝对导入；
从仓库根跑 ``python -m backend_A.main`` 会因为找不到 ``shared`` 而失败。
"""

from __future__ import annotations

import sys

from module_a_vision.main import main

if __name__ == "__main__":
    sys.exit(main())
