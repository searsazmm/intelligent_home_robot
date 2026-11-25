# -*- coding: utf-8 -*-
"""BoopFace 表情组件（ui/face_web.py）的单元测试。

**全部测试不创建任何 Qt 实例**（不建 QApplication、不建 QWebEngineView）：
- 真正起一个 Chromium 渲染进程的「端到端探活」由 ``main.py --face=auto``
  的 ``wait_ready`` 探针在演示机上做（见仓库验收流程），pytest 里模拟
  它只会测到 mock，没有意义；
- 这里钉住的是**纯逻辑**：四态映射表、状态确认窗口、离线资源的存在性与
  「绝不触网」、exe 打包（``sys._MEIPASS``）路径解析、路径越界防护。

跑法（在 frontend_C 目录下）：
    python -m pytest tests/test_face_web.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_core.expressions import (  # noqa: E402
    STATE_ABSENT, STATE_NORMAL, STATE_SAD, STATE_TIRED, VALID_STATES,
)

try:
    from ui import face_web
    QT_IMPORT_ERROR = None
except Exception as exc:                                    # noqa: BLE001
    face_web = None                                         # type: ignore[assignment]
    QT_IMPORT_ERROR = exc


@unittest.skipIf(QT_IMPORT_ERROR is not None,
                 f"ui.face_web 导入失败（缺 PyQtWebEngine？）：{QT_IMPORT_ERROR}")
class TestStateActionsMapping(unittest.TestCase):
    """四态 → boopface 动作：Phase 1 的全部业务规则就在这张表里。"""

    def test_all_four_states_are_mapped(self):
        # api_doc §4.2 的四种状态一个不缺（state → action 的键集合必须全覆盖）
        self.assertEqual(set(face_web.STATE_ACTIONS),
                         {STATE_NORMAL, STATE_SAD, STATE_TIRED, STATE_ABSENT})
        self.assertEqual(set(face_web.STATE_ACTIONS), set(VALID_STATES))

    def test_mapping_values_match_the_spec(self):
        # 用户规格原话：normal→idle、tired→sleep、failure（sad）→垂眼垂头、
        # absent→bored
        self.assertEqual(face_web.STATE_ACTIONS[STATE_NORMAL], "idle")
        self.assertEqual(face_web.STATE_ACTIONS[STATE_TIRED], "sleep")
        self.assertEqual(face_web.STATE_ACTIONS[STATE_SAD], "failure")
        self.assertEqual(face_web.STATE_ACTIONS[STATE_ABSENT], "bored")

    def test_all_actions_are_in_the_js_whitelist(self):
        # Python 侧推的每个动作都必须被 face_web.html 的 ALLOWED 白名单放行，
        # 否则会被页面兜回 idle —— tired/absent 的表情会静默失效
        html = face_web.html_path().read_text(encoding="utf-8")
        for action in face_web.STATE_ACTIONS.values():
            self.assertIn(f"'{action}'", html)


class TestStateConfirmation(unittest.TestCase):
    """状态确认窗口（与 FaceWindow 同语义：2 票相同才切换）。"""

    def test_insufficient_votes_returns_none(self):
        self.assertIsNone(face_web.state_confirmed("normal", ["tired"], need=2))
        self.assertIsNone(face_web.state_confirmed("normal", [], need=2))

    def test_mixed_votes_return_none(self):
        # 中间掺了别的状态 = 毛刺序列，清票重来
        self.assertIsNone(face_web.state_confirmed("normal", ["tired", "sad"], need=2))

    def test_unanimous_votes_are_confirmed(self):
        self.assertEqual(
            face_web.state_confirmed("normal", ["tired", "tired"], need=2),
            "tired",
        )

    def test_need_of_one_confirms_immediately(self):
        self.assertEqual(face_web.state_confirmed("normal", ["sad"], need=1), "sad")


class TestOfflineResources(unittest.TestCase):
    """离线资源：html 与 boopface 库在位，且宿主页绝不引用网络。"""

    def test_host_html_exists_next_to_the_module(self):
        html = face_web.html_path()
        self.assertTrue(html.is_file(), f"宿主页缺失：{html}")
        self.assertEqual(html.name, "face_web.html")

    def test_boopface_lite_exists_in_static(self):
        root = face_web.static_root()
        lite = root / "family" / "js" / "boopface" / "boopface-lite.js"
        self.assertTrue(lite.is_file(), f"boopface-lite.js 缺失：{lite}")

    def test_host_html_imports_lite_relatively(self):
        # 相对引用（./static/...）才落在 boopface:// 同源内；写绝对 file://
        # 或 http(s):// 都会破坏「完全离线」
        html = face_web.html_path().read_text(encoding="utf-8")
        self.assertIn("'./static/family/js/boopface/boopface-lite.js'", html)

    def test_host_html_never_touches_the_network(self):
        html = face_web.html_path().read_text(encoding="utf-8")
        for marker in ("http://", "https://", "//cdn"):
            self.assertNotIn(marker, html.lower(), f"宿主页引用了网络资源：{marker}")

    def test_page_contract_flags_present(self):
        # 用户点名的页面配置：黑底白脸、关自动睡觉；以及 Python 侧探针/切换接口
        html = face_web.html_path().read_text(encoding="utf-8")
        self.assertIn('color="#ffffff"', html)
        self.assertIn('auto-sleep="0"', html)
        self.assertIn("setPointerFollow(false)", html)
        self.assertIn("window.__face", html)
        self.assertIn("ready: true", html)
        self.assertIn("setState", html)


class TestResourcePathResolution(unittest.TestCase):
    """boopface:// 的回源路径：dev / PyInstaller 双环境 + 越界防护。"""

    def setUp(self):
        self.root = face_web.static_root().resolve()
        self.html = face_web.html_path().resolve()

    def test_face_html_resolves(self):
        self.assertEqual(face_web.resolve_resource_path("/face.html", self.root, self.html),
                         self.html)

    def test_static_asset_resolves(self):
        got = face_web.resolve_resource_path(
            "/static/family/js/boopface/boopface-lite.js", self.root, self.html)
        self.assertIsNotNone(got)
        self.assertTrue(got.is_file())

    def test_traversal_is_rejected(self):
        # ../ 穿越出资源根 → None（handler 回 404）
        self.assertIsNone(face_web.resolve_resource_path(
            "/static/../../secret.txt", self.root, self.html))

    def test_unknown_paths_are_rejected(self):
        self.assertIsNone(face_web.resolve_resource_path("/etc/passwd", self.root, self.html))
        self.assertIsNone(face_web.resolve_resource_path("/static/none.js", self.root, self.html))
        self.assertIsNone(face_web.resolve_resource_path("/static/", self.root, self.html))

    def test_candidate_roots_cover_meipass_when_frozen(self):
        # PyInstaller 冻结环境下 _MEIPASS 必须排第一（数据在临时解包目录里）
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass is None:
            self.skipTest("非冻结环境（无 sys._MEIPASS），打包路径在 exe 构建时验证")
        self.assertEqual(face_web._candidate_roots()[0], Path(meipass))


if __name__ == "__main__":
    unittest.main()
