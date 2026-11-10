# -*- coding: utf-8 -*-
"""family_portal 的路由（挂在项目 urls.py 的 '' 前缀下）。"""
from django.urls import path

from . import views

app_name = "family_portal"

urlpatterns = [
    path("", views.home, name="home"),
    path("stats/", views.stats_page, name="stats"),
    path("report/", views.report_page, name="report"),
    path("api/live/", views.api_live, name="api_live"),
    path("api/stats/", views.api_stats, name="api_stats"),
    path("api/alerts/", views.api_alerts, name="api_alerts"),
    path("api/report/", views.api_report, name="api_report"),
]
