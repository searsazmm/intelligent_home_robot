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
