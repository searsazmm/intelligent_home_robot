"""api_doc §3.2 的线上报文投影层——A 的**第二种**出站格式。

两个概念不要混
--------------
本模块产出的是 **api_doc §3.2 的顶层平铺 8 字段**：

    {"timestamp":…, "has_face":…, "ear":…, "blink_cnt":…,
     "pitch":…, "yaw":…, "roll":…, "emo_feature":…}

它与 ``系统设计方案.md:281-288`` 里那个同名不同物的 ``compat_v1`` **不是一回事**：
那个是 V2 报文**内部的一个嵌套子对象**（6 个键，挂在 ``observations`` 旁边）。
照着文档的名字写一个有损的中间层，只会得到第三种互不兼容的形状。
所以本模块叫 ``wire``（线上格式），不叫 ``compat``。

逐帧，不是逐窗口
----------------
api_doc §3.2 是**逐帧流式**协议，一条报文一帧。这不是随手定的：

* ``blink_cnt`` 是"累计眨眼次数"——累计值只有被反复上报、由对端做差才有意义；
* ``backend_B`` 的 ``_sustained_pitch_down`` 要连续取多个样本，
  ``_blink_rate_per_minute`` 要跨样本 diff ``blink_cnt``；
* ``backend_B/config.py`` 的 ``VISION_STALE_SECONDS`` 默认 **5 秒**——
  若按 A 自己的 10 秒窗口上报，B 每 10 秒里会有 5 秒判 ``absent``，
  状态周期性抖动。两个各自正确的设计撞在一起，只能让上报密于失联判据。

注意区分：B 的 ``WINDOW_SECONDS = 10.0`` 是 **B 在样本流上的滑动窗口**，
和 A 的 10 秒聚合窗口是两回事。

数据从哪来
----------
8 个字段**全部来自单帧** :class:`~shared.frame_features.FrameFeatures`，
不需要编造任何值：

============================ ==================================================
``timestamp``                ``frame.ts``（采集起点的单调秒，非 Unix 时间）
``has_face``                 ``frame.has_face and frame.quality.valid``
``ear``                      ``(frame.ear_left + frame.ear_right) / 2``
``blink_cnt``                ``EyeTracker.blink_total``（**进程累计**，见下）
``pitch`` / ``yaw`` / ``roll``  ``frame.pitch_deg`` / ``yaw_deg`` / ``roll_deg``
``emo_feature``              逐帧分类结果 → :data:`EMOTION_TO_FEATURE`
============================ ==================================================

``ear_left`` / ``ear_right`` 在全部四个采集源里都有值（mediapipe 走真实关键点，
合成源与 CSV 回放直接给值），所以**无摄像头的本机路径**下 ``ear`` 也是真数据。
"""

from __future__ import annotations

import math
from typing import Any

from shared.enums import Emotion
from shared.frame_features import FrameFeatures

#: api_doc §3.2 规定的字段集合。**精确相等**，多一个少一个都算违规。
V1_FIELDS: tuple[str, ...] = (
    "timestamp",
    "has_face",
    "ear",
    "blink_cnt",
    "pitch",
    "yaw",
    "roll",
    "emo_feature",
)

#: 表情特征 → 文档允许的三个取值（§3.2：``normal`` / ``low`` / ``tired``）。
#:
#: A 内部的表情标签有四个取值（``normal``/``tired``/``sad``/``upset``），
#: 比文档多一个 —— ``sad`` 与 ``upset`` 都得压到 ``low``。
#: 这个映射是有损的，方向单一（文档侧不需要区分难过和烦闷），可以接受。
#:
#: ⚠️ ``TIRED`` 这一支**代码可达、数据不可达**：表情分类器对疲惫有刻意的
#: 阻尼（``TIRED_DAMPING``），表情标签在结构上不会变成 ``tired``。
#: 所以 B 判"疲惫"只能靠 ``ear`` / ``blink_cnt`` / ``pitch`` 三条路，
#: 不要指望 ``emo_feature=tired`` 能被测出来。
EMOTION_TO_FEATURE: dict[Emotion, str] = {
    Emotion.NORMAL: "normal",
    Emotion.SAD: "low",
    Emotion.UPSET: "low",
    Emotion.TIRED: "tired",
}

#: 文档允许的 ``emo_feature`` 取值集合。
EMO_FEATURE_VALUES: frozenset[str] = frozenset({"normal", "low", "tired"})

#: 无人脸窗口的保底取值。
_NO_FACE: dict[str, Any] = {
    "has_face": False,
    "ear": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "emo_feature": "normal",
}


def to_v1_sample(
    frame: FrameFeatures,
    *,
    blink_total: int,
    emotion: Emotion,
) -> dict[str, Any]:
    """把一帧投影成 api_doc §3.2 的平铺报文。

    :param frame: 一帧特征（A 内部的统一结构）。
    :param blink_total: **进程累计**的眨眼次数，来自 ``EyeTracker.blink_total``。
        刻意做成显式入参而不是从 ``frame`` 上读：它在类型上就说明了
        "这个值不属于这一帧，属于进程"，挡住"每帧都发同一个旧值却忘了更新"。
    :param emotion: 这一帧的表情分类结果。分类需要 :class:`NeutralCalibrator`
        的标定状态，所以由持有分类器的 :class:`WindowAggregator` 算好传进来，
        本模块只做映射 —— 保持纯函数、可脱离 socket 与标定单独测试。

    **画面质量不可用的帧一律按 ``has_face=False`` 上报。** 这是一个刻意的判断，
    理由值得写下来：api_doc §3.2 没有 quality 字段，无法表达"看见设备但看不清人"。
    若照字面报 ``has_face=true`` 而把 ``ear`` 填 0.0，B 的 ``_low_ear_seconds``
    会把这串 0 当成"EAR 持续低于阈值"，几秒后报出 ``tired`` ——
    一个**纯粹由缺字段制造出来的假疲劳**。压成 ``has_face=false`` 后 B 判 ``absent``
    （"看不见老人"），语义正确且不伪造。代价是失去那个区分，
    而它在 §3.2 里本来就表达不了（那是 V2 用 ``vision_unusable`` 表达的能力）。
    """
    if not (frame.has_face and frame.quality.valid):
        return {
            "timestamp": round(float(frame.ts), 6),
            **_NO_FACE,
            # 无人脸时**不清零** blink_cnt。这是一处偏离 §3.3.1 字面的地方：
            # "其余字段填默认值"针对的是**测量量**（ear/pitch/yaw/roll 在没有脸时
            # 确实无意义），而 blink_cnt 是**事件计数器** —— "这一帧没人脸"
            # 不等于"这个人从来没眨过眼"，清零才是编数据。
            # B 侧不受影响：`_blink_rate_per_minute` 先按 `has_face` 过滤样本。
            "blink_cnt": int(blink_total),
        }

    return {
        "timestamp": round(float(frame.ts), 6),
        "has_face": True,
        "ear": round((float(frame.ear_left) + float(frame.ear_right)) / 2.0, 4),
        "blink_cnt": int(blink_total),
        "pitch": round(float(frame.pitch_deg), 3),
        "yaw": round(float(frame.yaw_deg), 3),
        "roll": round(float(frame.roll_deg), 3),
        "emo_feature": EMOTION_TO_FEATURE.get(emotion, "normal"),
    }


def assert_v1_contract(payload: dict[str, Any]) -> list[str]:
    """校验一份报文是否严格符合 api_doc §3.2。返回违规项列表，空列表＝干净。

    这个函数是**出口契约闸**，和 :func:`privacy.guard.assert_clean` 并排装在
    :meth:`VisionServer.broadcast` 里。理由和隐私闸一样：闸门装在出口，
    将来谁加了字段都必须先过这一关。

    对端（``backend_B``）对缺失字段一律**静默回退默认值**，不报错也不断连。
    这意味着一个字段名拼错或类型不对的报文，表现不是崩溃，而是
    "状态永远停在 absent" —— 所以这道闸存在的意义就是把这个静默失效
    变成一个响亮的、可在出口拦下的失败。
    """
    problems: list[str] = []

    keys = set(payload)
    expected = set(V1_FIELDS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing:
        problems.append(f"缺少字段：{missing}")
    if extra:
        problems.append(f"多余字段：{extra}")

    # `type(x) is bool` 必须在 `is int` 之前判：Python 里 `True` 也是 `int`，
    # 用 isinstance 会让 has_face=True 通过 blink_cnt 的整数校验。
    if "has_face" in payload and type(payload["has_face"]) is not bool:
        problems.append(
            f"has_face 必须是 bool，实际是 {type(payload['has_face']).__name__}"
        )

    if "blink_cnt" in payload:
        # 刻意用 type() 而不是 isinstance()：isinstance(True, int) 为真，
        # 会让 True 悄悄通过整数校验。
        if type(payload["blink_cnt"]) is not int:
            problems.append(
                f"blink_cnt 必须是 int，实际是 {type(payload['blink_cnt']).__name__}"
            )

    for name in ("timestamp", "ear", "pitch", "yaw", "roll"):
        if name not in payload:
            continue
        value = payload[name]
        if type(value) is bool or not isinstance(value, (int, float)):
            problems.append(
                f"{name} 必须是数字，实际是 {type(value).__name__}"
            )
        elif not math.isfinite(float(value)):
            problems.append(f"{name} 不是有限数：{value!r}")

    if "emo_feature" in payload:
        feature = payload["emo_feature"]
        if feature not in EMO_FEATURE_VALUES:
            problems.append(
                f"emo_feature 取值非法：{feature!r}"
                f"（只允许 {sorted(EMO_FEATURE_VALUES)}）"
            )

    return problems


def describe_v1(payload: dict[str, Any]) -> str:
    """一行摘要，给 ``--dry-run`` 与日志用。"""
    face = "有人脸" if payload.get("has_face") else "无人脸"
    return (
        f"t={payload.get('timestamp', 0.0):>7.2f}s  {face}  "
        f"ear={payload.get('ear', 0.0):.3f}  "
        f"blink={payload.get('blink_cnt', 0):>4}  "
        f"pose=({payload.get('pitch', 0.0):>6.1f},"
        f"{payload.get('yaw', 0.0):>6.1f},"
        f"{payload.get('roll', 0.0):>6.1f})  "
        f"emo={payload.get('emo_feature', '?')}"
    )
