# -*- coding: utf-8 -*-
"""颜文字表情表 —— 四种状态到字符串的映射。

**本模块刻意不 import Qt**，是纯数据 + 纯函数。两个理由：

1. 可以脱离 Qt 单测（CI 无图形环境时也能跑）；
2. 表情文案是产品内容，和渲染技术是两件事，混在一起改文案要动窗口代码。

api_doc §4.2 只允许 normal / sad / tired / absent 四种状态，禁止自造。
api_doc §4.3 还要求「前端 C 接收未知内容、断连异常时，默认展示 normal 状态」——
这条是硬要求，由 :func:`normalize_state` 兜住。

关于字体（已实测，不是推测）：
    颜文字大量使用全角/CJK 字形（``＾`` U+FF3E、``￣`` U+FFE3、``﹏`` U+FE4F、
    ``－`` U+FF0D）。开发机上 ``Microsoft YaHei`` / ``SimSun`` / ``SimHei``
    **都不存在**，能正常渲染的是 ``Microsoft YaHei UI``（首选）。
    全部 16 个颜文字已逐个渲染验证过无豆腐块，见 frontend_C/README.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# 四种合法状态（api_doc §4.2，禁止增删改名）
# --------------------------------------------------------------------------

STATE_NORMAL = "normal"
STATE_SAD = "sad"
STATE_TIRED = "tired"
STATE_ABSENT = "absent"

#: 顺序有意义：界面上的「状态轮换」按这个顺序走，从正常到异常
VALID_STATES: Tuple[str, ...] = (STATE_NORMAL, STATE_SAD, STATE_TIRED, STATE_ABSENT)

#: 未知内容一律回落到它（api_doc §4.3）
DEFAULT_STATE = STATE_NORMAL


@dataclass(frozen=True)
class Expression:
    """一个颜文字表情的两帧。

    眨眼帧**显式写死**，不靠正则去替换眼睛字符 —— 颜文字的结构千奇百怪
    （``(￣o￣) zzZ`` 里哪个是眼睛？``(・◇・)?`` 里 ``◇`` 是嘴还是鼻子？），
    任何「聪明的」通用替换规则都会在某些表情上出洋相。
    """

    face: str    #: 睁眼帧，平时显示这个
    blink: str   #: 眨眼帧，瞬间替换一下再换回来

    def __str__(self) -> str:  # 方便日志里直接打印
        return self.face


# --------------------------------------------------------------------------
# 表情表
#
# 设计原则：
#   - 一眼能看出情绪，不要抽象的符号堆砌（适老化）
#   - 同状态给 4 个，轮换显示，避免长时间盯着同一张脸显得死板
#   - 眨眼帧只动「眼睛」那一对字符，其余保持，这样闪一下才像眨眼
# --------------------------------------------------------------------------

EXPRESSIONS: Dict[str, Tuple[Expression, ...]] = {
    # ---- 正常：舒展的笑脸 ----
    STATE_NORMAL: (
        Expression("(＾▽＾)", "(－▽－)"),
        Expression("(￣▽￣)", "(－▽－)"),
        Expression("(＾ω＾)", "(－ω－)"),
        Expression("(⌒‿⌒)", "(－‿－)"),
    ),

    # ---- 情绪低落：眉眼向下、含泪 ----
    STATE_SAD: (
        # ⚠️ 这里原本是 (｡•́︿•̀｡)。它里面的 U+0301 / U+0300 是**组合用**附加符号，
        #    得和前一个字符叠在一起才有意义，而 Microsoft YaHei UI 没有把它们
        #    跟 • 合成一个字形 —— 真机上渲染成「一个点 + 右上角飘着一个孤立小撇」，
        #    不是豆腐块、也读得出意思，但看着别扭（导出 PNG 逐张核对时发现的）。
        #    (T＿T) 只用 ASCII 的 T 和全角下划线 U+FF3F，两者字形覆盖都没问题。
        Expression("(T＿T)", "(；＿；)"),          # 眨眼帧只换眼睛：T → ；，嘴不动
        Expression("(╥﹏╥)", "(－﹏－)"),
        Expression("(；﹏；)", "(；－；)"),          # 往两边淌泪，眨眼时泪线变短
        Expression("(´；ω；`)", "(－；ω；-)"),
    ),

    # ---- 疲惫：眼睛本来就半闭，眨眼帧压得更扁 ----
    STATE_TIRED: (
        Expression("(=_=)", "(－_－)"),
        Expression("(－_－)…", "(＿_＿)…"),
        Expression("(￣o￣) zzZ", "(￣－￣) zzZ"),
        Expression("(∪.∪ )...zzz", "(∪_∪ )...zzz"),
    ),

    # ---- 走神 / 无人：眼睛睁着但眼神飘走、带问号 ----
    STATE_ABSENT: (
        Expression("(・_・?)", "(－_－?)"),
        Expression("(°ー°〃)", "(－ー－〃)"),
        Expression("(・o・)", "(－o－)"),
        Expression("( ・◇・)?", "( －◇－)?"),
    ),
}


# --------------------------------------------------------------------------
# 对外函数
# --------------------------------------------------------------------------

def normalize_state(raw: object) -> str:
    """把任意输入规整成四种合法状态之一。

    api_doc §4.3 的硬要求：收到未知内容或断连异常时，一律回 ``normal``，
    保证程序稳定不崩溃。所以这里**任何**输入都必须有返回，绝不抛异常：

        >>> normalize_state("  SAD  ")
        'sad'
        >>> normalize_state("angry")      # 未知状态
        'normal'
        >>> normalize_state(None)
        'normal'
        >>> normalize_state(123)
        'normal'
    """
    if isinstance(raw, str):
        state = raw.strip().lower()
        if state in VALID_STATES:
            return state
    return DEFAULT_STATE


def expressions_for(state: object) -> Tuple[Expression, ...]:
    """取某状态的表情组。状态非法时自动回落到 normal 的那一组。"""
    return EXPRESSIONS[normalize_state(state)]


def all_faces(state: object = None) -> List[str]:
    """列出颜文字字符串。

    不传 state 时返回全部四态的（供 ``--dump-glyphs`` 逐字检查字形用）；
    传了 state 就只返回那一组。眨眼帧也一并列出 —— 它们同样要被检查，
    漏掉一个眨眼帧就可能出现「不眨眼时好好的，一眨眼变豆腐块」。
    """
    states: Sequence[str] = (normalize_state(state),) if state is not None else VALID_STATES
    faces: List[str] = []
    for name in states:
        for expression in EXPRESSIONS[name]:
            faces.append(expression.face)
            faces.append(expression.blink)
    return faces


def state_label(state: object) -> str:
    """状态的中文说明，只用于日志和标题栏，**不显示在窗口里**。

    窗口内容按需求只有颜文字本身，不能出现文字框。
    """
    return {
        STATE_NORMAL: "状态正常",
        STATE_SAD: "情绪低落",
        STATE_TIRED: "疲惫",
        STATE_ABSENT: "走神/无人",
    }[normalize_state(state)]
