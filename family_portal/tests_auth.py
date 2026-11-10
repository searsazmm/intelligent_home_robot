# -*- coding: utf-8 -*-
"""二期鉴权（X-API-Token）：装饰器行为 + 登录/登出全流程。

设计要点（family_portal/views.py 的 require_auth）：
- ``FAMILY_API_TOKEN`` 为空 = 不启用，现有页面/接口行为**一字不变** ——
  这也是 tests.py 里那 22 个用例不需要任何 auth 夹具的原因；
- 启用后双通道：请求头 X-API-Token（脚本）或 session（浏览器登录）；
- API 未授权 = HTTP 401 + status=2（区别于 status=1 的"暂无数据"）；
- 页面未授权 = 302 到 /login/（带 ?next=，且 next 只收站内路径）。

令牌比较一律走 :func:`hmac.compare_digest`（防时序侧信道）。
"""

from django.test import TestCase

from family_portal.tests import TempDbMixin

TOKEN = "family-demo-token"


class DisabledAuthTests(TempDbMixin, TestCase):
    """默认配置（无令牌）：一切照旧，这是向后兼容的底线。"""

    def test_pages_and_api_open_without_token(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/stats/").status_code, 200)
        resp = self.client.get("/api/live/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], 0)

    def test_logout_when_disabled_goes_home(self):
        resp = self.client.get("/logout/")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp["Location"], "/")

    def test_nav_hides_logout_when_disabled(self):
        self.assertNotIn("退出", self.client.get("/").content.decode("utf-8"))


class EnabledAuthTests(TempDbMixin, TestCase):
    """启用令牌后的全部路径。override_settings 进出即恢复，互不影响。"""

    def setUp(self):
        super().setUp()
        # override_settings 对模块级 settings 生效；views 用
        # django.conf.settings 动态读取，因此每个请求都能感知。
        from django.test import override_settings
        ctx = override_settings(FAMILY_API_TOKEN=TOKEN)
        ctx.enable()
        self.addCleanup(ctx.disable)

    # ---- 未授权 ----

    def test_page_redirects_to_login_with_next(self):
        resp = self.client.get("/stats/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login/", resp["Location"])
        self.assertIn("next=", resp["Location"])

    def test_api_returns_401_contract(self):
        resp = self.client.get("/api/live/")
        self.assertEqual(resp.status_code, 401)
        payload = resp.json()
        self.assertEqual(payload["status"], 2)
        self.assertIn("X-API-Token", payload["message"])

    def test_wrong_token_is_rejected(self):
        resp = self.client.get("/api/alerts/", HTTP_X_API_TOKEN="wrong")
        self.assertEqual(resp.status_code, 401)

    # ---- 请求头通道（脚本场景）----

    def test_header_token_grants_api_access(self):
        resp = self.client.get("/api/live/", HTTP_X_API_TOKEN=TOKEN)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], 0)

    def test_header_token_also_opens_pages(self):
        # 页面同样接受 header —— 便于 curl 直接抓页面调试
        resp = self.client.get("/stats/", HTTP_X_API_TOKEN=TOKEN)
        self.assertEqual(resp.status_code, 200)

    # ---- 浏览器通道（登录/登出）----

    def test_login_flow(self):
        # 登录页本身不需要授权
        self.assertContains(self.client.get("/login/"), "家属登录")
        # 错误令牌：留在登录页并提示
        resp = self.client.post("/login/", {"token": "nope"})
        self.assertContains(resp, "令牌不正确")
        # 正确令牌：302 回首页，之后页面直接可访问
        resp = self.client.post("/login/", {"token": TOKEN})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp["Location"], "/")
        self.assertEqual(self.client.get("/report/").status_code, 200)
        # 退出后再访问又被打回
        self.client.get("/logout/")
        resp = self.client.get("/api/live/")
        self.assertEqual(resp.status_code, 401)

    def test_login_success_with_next_only_for_internal_paths(self):
        resp = self.client.post("/login/?next=/stats/", {"token": TOKEN})
        self.assertEqual(resp["Location"], "/stats/")

    def test_login_rejects_external_next(self):
        resp = self.client.post("/login/?next=https://evil.example.com", {"token": TOKEN})
        # 不回跳外站，落回首页
        self.assertEqual(resp["Location"], "/")

    def test_login_rejects_protocol_relative_next(self):
        resp = self.client.post("/login/?next=//evil.example.com", {"token": TOKEN})
        self.assertEqual(resp["Location"], "/")

    def test_empty_token_post_shows_error(self):
        resp = self.client.post("/login/", {"token": ""})
        self.assertContains(resp, "令牌不正确")

    def test_nav_shows_logout_after_login(self):
        self.client.post("/login/", {"token": TOKEN})
        self.assertContains(self.client.get("/"), "退出")


class TokenMatchingTests(TempDbMixin, TestCase):
    """令牌比较的边缘输入：判不匹配可以，绝不能 500。"""

    def setUp(self):
        super().setUp()
        from django.test import override_settings
        ctx = override_settings(FAMILY_API_TOKEN=TOKEN)
        ctx.enable()
        self.addCleanup(ctx.disable)

    def test_token_is_case_sensitive(self):
        resp = self.client.get("/api/live/", HTTP_X_API_TOKEN=TOKEN.upper())
        self.assertEqual(resp.status_code, 401)

    def test_token_with_surrounding_whitespace_rejected(self):
        # 环境变量侧 settings 已 strip；请求头侧**不做** strip ——
        # 复制粘贴带空格的令牌就是无效凭证，宽进等于没有边界
        resp = self.client.get("/api/live/", HTTP_X_API_TOKEN=" " + TOKEN + " ")
        self.assertEqual(resp.status_code, 401)

    def test_non_ascii_header_rejected_not_500(self):
        """畸形请求头（非 ASCII）走 TypeError 兜底 → 401，而非 500。

        hmac.compare_digest 的 str 版本只收 ASCII-only，恶意输入
        不该把鉴权路径炸成服务器错误。
        """
        resp = self.client.get("/api/live/", HTTP_X_API_TOKEN="令牌测试")
        self.assertEqual(resp.status_code, 401)

    def test_non_ascii_form_input_rejected_not_500(self):
        resp = self.client.post("/login/", {"token": "令牌测试"})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "令牌不正确")


class ApiCoverageTests(TempDbMixin, TestCase):
    """四个 REST 端点**全部**受保护 —— 漏掉任何一个都是裸奔的家属数据。"""

    API_PATHS = ("/api/live/", "/api/stats/", "/api/alerts/", "/api/report/")

    def setUp(self):
        super().setUp()
        from django.test import override_settings
        ctx = override_settings(FAMILY_API_TOKEN=TOKEN)
        ctx.enable()
        self.addCleanup(ctx.disable)

    def test_every_api_endpoint_rejects_anonymous(self):
        for path in self.API_PATHS:
            with self.subTest(path=path):
                resp = self.client.get(path)
                self.assertEqual(resp.status_code, 401, path)

    def test_every_api_endpoint_accepts_header_token(self):
        for path in self.API_PATHS:
            with self.subTest(path=path):
                resp = self.client.get(path, HTTP_X_API_TOKEN=TOKEN)
                self.assertEqual(resp.status_code, 200, path)

    def test_401_payload_fulfills_contract(self):
        """401 也是契约响应：四个键一个不少，data 为空 dict。"""
        payload = self.client.get("/api/live/").json()
        self.assertEqual(set(payload),
                         {"status", "message", "data", "timestamp"})
        self.assertEqual(payload["status"], 2)
        self.assertEqual(payload["data"], {})


class LoginEdgeCasesTests(TempDbMixin, TestCase):
    """登录页与回跳的边缘路径。"""

    def setUp(self):
        super().setUp()
        from django.test import override_settings
        ctx = override_settings(FAMILY_API_TOKEN=TOKEN)
        ctx.enable()
        self.addCleanup(ctx.disable)

    def test_next_relative_path_rejected(self):
        """/stats 没有 / 前缀 —— 不是站内绝对路径，落回首页。"""
        resp = self.client.post("/login/?next=stats", {"token": TOKEN})
        self.assertEqual(resp["Location"], "/")

    def test_next_to_login_page_does_not_loop(self):
        """next=/login/ 会被放行（站内路径）；登录后到登录页再手动进，
        行为无害 —— 这里钉住它**不产生重定向死循环**（直接 200/302 可终止）。"""
        resp = self.client.post("/login/?next=/login/", {"token": TOKEN})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp["Location"], "/login/")

    def test_session_persists_across_requests(self):
        self.client.post("/login/", {"token": TOKEN})
        for _ in range(3):
            self.assertEqual(self.client.get("/stats/").status_code, 200)

    def test_login_success_is_logged(self):
        with self.assertLogs("family_portal.views", level="INFO") as captured:
            self.client.post("/login/", {"token": TOKEN})
        self.assertIn("登录成功", "".join(captured.output))

    def test_csrf_protection_is_enforced(self):
        """真实浏览器场景：CSRF 中间件必须拦下没有令牌的跨站 POST。"""
        from django.test import Client
        strict = Client(enforce_csrf_checks=True)
        resp = strict.post("/login/", {"token": TOKEN})
        self.assertEqual(resp.status_code, 403)
