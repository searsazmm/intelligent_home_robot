# -*- coding: utf-8 -*-
"""family_portal —— 家属远程 Web 门户（8020 端口）。

家属（异地也行）在浏览器里了解家人情况：实时心率/呼吸、一周统计
折线图与状态分布、健康报告（含饮食建议）、预警记录。

数据来源：backend_B 落盘的 SQLite（config §17，core/vitals_log.py），
本 app **只读**（每请求临时 mode=ro 连接，用完即关）。B 未启动 /
库里没数据时页面诚实显示，不编造。

详细设计见 .trae/documents/family_portal_plan.md。
"""
