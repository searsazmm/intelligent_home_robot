# -*- coding: utf-8 -*-
"""后端 B 核心逻辑包。

模块划分：
    protocol        TCP 分帧与 JSON 编解码
    vision_client   连模块 A，收视觉 JSON，带断线重连 + 离线 CSV 回放
    vision_state    视觉原始特征 → normal/sad/tired/absent 四态判定
    text_emotion    用户对话文本 → 情绪
    dialogue        视觉状态 + 文本情绪 + 意图 → 回复
    history_store   CSV 历史对话记录读写
    ui_channel      对模块 C 的状态推送(8001)与双向对话(8002)
"""
