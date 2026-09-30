"""api_doc §3.2 / §3.5 的线上报文投影层——A 的**第二种**出站格式。

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

类型化报文（api_doc §3.5）
--------------------------

同一条 TCP 流上还有两种**带 ``type``** 的报文：

============================ ==================================================
``{"type":"rppg", …}``       体征：心率 / 呼吸率 / 心跳间期
``{"type":"focus", …}``      视线：偏离度 + 获取质量
============================ ==================================================

**帧报文仍然不带 ``type``。** §3.5 草案里曾写"现有视觉帧标 ``type:"frame"``"，
本模块刻意**不**那么做，理由有两条，任一条都足够：

1. §3.2 的 8 字段是**键集合精确相等**的，多一个 ``type`` 就会被
   :func:`assert_v1_contract` 判成"多余字段"。照草案标了，契约闸和文档
   会自相矛盾，而契约闸是先写好的那一个。
2. 草案同一段已经写了"B 收到不带 ``type`` 的包按 frame 处理"，所以
   不标同样兼容。

拆开的收益是**契约闸一个字符都不用改**：:func:`assert_v1_contract` 只管
帧的形状，:func:`check_contract` 只管分流。两者职责不重叠。

**分流必须是显式三路，未知类型一律抛异常**（:func:`check_contract`）。
这里有一个差点写进去的真 bug 值得记下来：若写成
``if type == "rppg": … else: assert_v1_contract(…)``，
那么 ``focus`` 报文会落进 ``else`` → 被判"多余字段" → 在
:meth:`VisionServer.broadcast` 里丢弃 → **VAI 永远拿不到视线数据**，
而所有"报文格式正确"的测试**全都是绿的**。所以默认分支不兜底，直接抛。
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

# ================================================================ §3.5 类型化报文

#: §3.5 的两种类型化报文。**帧报文不在这个表里**——它不带 ``type``。
MSG_RPPG = "rppg"
MSG_FOCUS = "focus"
TYPED_MESSAGE_TYPES: tuple[str, ...] = (MSG_RPPG, MSG_FOCUS)

#: :func:`check_contract` 鉴别结果的"帧"分支。**不是线上字面量**：
#: 线上帧报文没有 ``type`` 键，这个字符串只在本模块内部流转
#: （用来选择计数器），所以不要拿它去和报文里的值比较。
KIND_FRAME = "frame"

#: 体征报文的字段集合。``hr`` / ``rr`` 可以是 ``None``（见 :func:`assert_rppg_contract`），
#: 所以这里**没有** ``sqi``：质量由"hr 是不是 None"表达，而 :class:`~rppg.Rppg`
#: 的质量门控（SNR / 窗口长度 / 帧率）已经体现为置灰。多传一个质量分会
#: 让对端有第二个真相来源，两者不一致时无从判断该信谁。
RPPG_FIELDS: tuple[str, ...] = ("type", "timestamp", "hr", "rr", "ibi_ms")

#: 视线报文的字段集合。
FOCUS_FIELDS: tuple[str, ...] = ("type", "timestamp", "gaze", "gaze_quality")

#: ``hr`` 的合法区间（bpm）。下界 30 不是"再低就不可能"，而是
#: :class:`~rppg.Rppg` 的带通下限 0.7Hz ≈ 42bpm 减去一个 FFT 分辨率
#: 的量级；上界 240 同理由 4.0Hz 得来。越界的值一定是算错了，不是人。
HR_RANGE: tuple[float, float] = (30.0, 240.0)
#: ``rr`` 的合法区间（次/分）。对应 :class:`~rppg.Rppg` 的 0.1–0.5Hz 呼吸带。
RR_RANGE: tuple[float, float] = (4.0, 40.0)
#: 单个心跳间期的合法区间（毫秒）。下界约 240bpm、上界约 20bpm。
IBI_RANGE: tuple[float, float] = (200.0, 3000.0)
#: ``ibi_ms`` 列表的长度上限，防止有人把整个缓冲倒出来。
MAX_IBI_COUNT = 64

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


# ================================================================ §3.5 构造

def build_rppg_payload(
    timestamp: float,
    *,
    hr: int | None,
    rr: float | None,
    ibi_ms: list[int] | tuple[int, ...] = (),
) -> dict[str, Any]:
    """把一次 :meth:`Rppg.compute` 的结果投影成 §3.5 体征报文。

    ``hr`` / ``rr`` 为 ``None`` 是**正常态**，不是错误：:class:`~rppg.Rppg`
    在窗口未满、SNR 不达标、或帧率过低时都会置灰。投影层不做任何
    "兜一个默认值"的事——把置灰编成一个看起来正常的数字，正是这套系统
    最不该做的事（它会变成一条基于假数据的建议）。
    """
    return {
        "type": MSG_RPPG,
        "timestamp": round(float(timestamp), 6),
        "hr": None if hr is None else int(hr),
        "rr": None if rr is None else round(float(rr), 1),
        "ibi_ms": [int(x) for x in ibi_ms],
    }


def build_focus_payload(
    timestamp: float,
    *,
    gaze: float | None,
    gaze_quality: float,
) -> dict[str, Any]:
    """把一帧的视线信息投影成 §3.5 视线报文。

    :param gaze: 视线偏离正前方的程度 ``[0,1]``；``None`` 表示这一帧
        **估不出来**（例如人脸模型没有虹膜点）。``None`` 与 ``0.0``
        语义完全不同：后者是"正对着镜头"，前者是"不知道"。
        A 侧把这两者区分开，B 才可能不把"不知道"当成"很专注"。
    :param gaze_quality: ``[0,1]``。**必须与 ``gaze`` 是否为 ``None`` 一致**
        （见 :func:`assert_focus_contract` 的不变式）。
    """
    return {
        "type": MSG_FOCUS,
        "timestamp": round(float(timestamp), 6),
        "gaze": None if gaze is None else round(float(gaze), 4),
        "gaze_quality": round(float(gaze_quality), 4),
    }


# ================================================================ §3.5 契约闸

class UnknownMessageTypeError(ValueError):
    """v1 流上出现了一个既不是帧、也不在 §3.5 白名单里的 ``type``。

    **刻意抛异常而不是"当帧处理"。** 当帧处理会让它撞上
    :func:`assert_v1_contract` 的"多余字段"检查、被丢弃，于是问题表现成
    "某个功能一直不工作"；抛异常则把"谁发了一个没人认识的东西"变成
    启动后几秒内就能看到的失败。

    继承 :class:`ValueError`，这样 ``main()`` 里既有的
    ``except (ValueError, RuntimeError, ImportError)`` 能接住它、
    打出一行说明而不是一整段栈回溯。
    """


def assert_rppg_contract(payload: dict[str, Any]) -> list[str]:
    """校验一份体征报文。返回违规项列表，空列表＝干净。

    **``hr`` 为 ``None`` 是合法的**，而且是运行初期的常态：8 秒窗口
    没满之前 :class:`~rppg.Rppg` 一个数都给不出。任何"收到 rppg 报文
    就说明 hr 是数字"的假设都是错的，这条契约就是用来挡住那种假设的。
    """
    problems = _check_keys(payload, RPPG_FIELDS)
    problems.extend(_check_type_tag(payload, MSG_RPPG))
    problems.extend(_check_timestamp(payload))

    for name, lo, hi in (("hr", *HR_RANGE), ("rr", *RR_RANGE)):
        if name not in payload:
            continue
        value = payload[name]
        if value is None:
            continue          # 置灰是正常态，见 docstring
        if not _is_number(value):
            problems.append(f"{name} 必须是数字或 null，实际是 {type(value).__name__}")
        elif not lo <= float(value) <= hi:
            problems.append(f"{name} 超出合理区间 [{lo:g}, {hi:g}]：{value!r}")

    if "ibi_ms" in payload:
        ibi = payload["ibi_ms"]
        if not isinstance(ibi, (list, tuple)):
            problems.append(f"ibi_ms 必须是数组，实际是 {type(ibi).__name__}")
        elif len(ibi) > MAX_IBI_COUNT:
            problems.append(f"ibi_ms 过长：{len(ibi)} 项，上限 {MAX_IBI_COUNT}")
        else:
            for i, item in enumerate(ibi):
                if type(item) is not int:
                    problems.append(
                        f"ibi_ms[{i}] 必须是 int，实际是 {type(item).__name__}"
                    )
                elif not IBI_RANGE[0] <= item <= IBI_RANGE[1]:
                    problems.append(
                        f"ibi_ms[{i}] 超出合理区间 "
                        f"[{IBI_RANGE[0]:g}, {IBI_RANGE[1]:g}]：{item!r}"
                    )
    return problems


def assert_focus_contract(payload: dict[str, Any]) -> list[str]:
    """校验一份视线报文。返回违规项列表，空列表＝干净。

    钉住一条不变式：**``gaze is None`` ⟺ ``gaze_quality == 0``**。
    两者不一致就拦下——不一致意味着 A 在说"我不知道，但我知道得挺清楚"，
    而对端无论采信哪一半都会算错。
    """
    problems = _check_keys(payload, FOCUS_FIELDS)
    problems.extend(_check_type_tag(payload, MSG_FOCUS))
    problems.extend(_check_timestamp(payload))

    gaze = payload.get("gaze")
    quality = payload.get("gaze_quality")

    if "gaze" in payload and gaze is not None and not _is_number(gaze):
        problems.append(f"gaze 必须是数字或 null，实际是 {type(gaze).__name__}")
    if "gaze" in payload and gaze is not None and _is_number(gaze):
        if not 0.0 <= float(gaze) <= 1.0:
            problems.append(f"gaze 超出 [0,1]：{gaze!r}")

    if "gaze_quality" in payload:
        if not _is_number(quality):
            problems.append(
                f"gaze_quality 必须是数字，实际是 {type(quality).__name__}"
            )
        else:
            if not 0.0 <= float(quality) <= 1.0:
                problems.append(f"gaze_quality 超出 [0,1]：{quality!r}")
            # 不变式：估不出来 ⟺ 质量为 0。两头都要查，只查一头会放过
            # "gaze 有值但质量为 0"（对端会当成不可信而丢掉一个有效观测）。
            elif (gaze is None) != (float(quality) == 0.0):
                problems.append(
                    f"gaze 与 gaze_quality 不一致：gaze={gaze!r} 但 "
                    f"gaze_quality={quality!r}（约定 gaze is None ⟺ quality == 0）"
                )

    return problems


#: ``type`` → 契约校验函数。**只有帧报文不在表里**，它有单独的分支。
TYPED_CONTRACTS = {
    MSG_RPPG: assert_rppg_contract,
    MSG_FOCUS: assert_focus_contract,
}


def check_contract(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """路由 + 校验，返回 ``(种类, 违规项)``。

    这是 v1 流的**唯一**分流点，:meth:`VisionServer.broadcast` 与
    ``--dry-run`` 都调它。合成一个函数而不是各写一遍，是因为两者一旦
    分叉，"最快的排查工具"就会开始输出与实跑不符的"一切正常"。

    :raises UnknownMessageTypeError: ``type`` 存在但不在 §3.5 白名单里。
        不兜底——见 :class:`UnknownMessageTypeError`。
    """
    kind = payload.get("type")
    if kind is None:
        return KIND_FRAME, assert_v1_contract(payload)

    checker = TYPED_CONTRACTS.get(kind)
    if checker is None:
        raise UnknownMessageTypeError(
            f"v1 流上出现了未知报文类型 {kind!r}。已知类型："
            f"{'、'.join(TYPED_MESSAGE_TYPES)}；帧报文一律**不带** type。"
            f"若确实要加新类型，请在 wire.TYPED_CONTRACTS 里登记它的契约函数。"
        )
    return kind, checker(payload)


def describe_typed(payload: dict[str, Any]) -> str:
    """类型化报文的一行摘要，给 ``--dry-run`` 与日志用。"""
    kind = payload.get("type")
    if kind == MSG_RPPG:
        hr = payload.get("hr")
        rr = payload.get("rr")
        ibi = payload.get("ibi_ms") or []
        return (
            f"t={payload.get('timestamp', 0.0):>7.2f}s  [体征]  "
            f"hr={hr if hr is not None else '置灰'}  "
            f"rr={rr if rr is not None else '置灰'}  "
            f"ibi={len(ibi)} 拍"
        )
    if kind == MSG_FOCUS:
        gaze = payload.get("gaze")
        shown = "未取到" if gaze is None else f"{float(gaze):.3f}"
        return (
            f"t={payload.get('timestamp', 0.0):>7.2f}s  [视线]  "
            f"gaze_off={shown}  q={payload.get('gaze_quality', 0.0):.2f}"
        )
    return f"t={payload.get('timestamp', 0.0):>7.2f}s  {kind}"


# ================================================================ 内部

def _is_number(value: Any) -> bool:
    """数字，且**不是**布尔。``isinstance(True, int)`` 为真，得显式排掉。"""
    return type(value) is not bool and isinstance(value, (int, float))


def _check_keys(payload: dict[str, Any], expected: tuple[str, ...]) -> list[str]:
    keys = set(payload)
    want = set(expected)
    problems: list[str] = []
    missing = sorted(want - keys)
    extra = sorted(keys - want)
    if missing:
        problems.append(f"缺少字段：{missing}")
    if extra:
        problems.append(f"多余字段：{extra}")
    return problems


def _check_type_tag(payload: dict[str, Any], expected: str) -> list[str]:
    if payload.get("type") != expected:
        return [f"type 必须是 {expected!r}，实际是 {payload.get('type')!r}"]
    return []


def _check_timestamp(payload: dict[str, Any]) -> list[str]:
    if "timestamp" not in payload:
        return []
    value = payload["timestamp"]
    if not _is_number(value):
        return [f"timestamp 必须是数字，实际是 {type(value).__name__}"]
    if not math.isfinite(float(value)):
        return [f"timestamp 不是有限数：{value!r}"]
    return []
