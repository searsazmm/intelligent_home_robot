"""隐私子包——"原始帧不出模块"这条红线的可执行实现。

* :func:`assert_clean`——出站报文校验，命中敏感字段即抛
  :class:`PrivacyViolationError`。
* :func:`scrub_frame`——帧用完就地清零。
* :func:`redact_repr`——安全的日志 repr。

判定规则（分词 + 豁免表）与它要解决的冲突写在
:mod:`module_a_vision.privacy.guard` 的模块文档里——**改动词表前先读它**。
"""

from .guard import (
    MAX_DEPTH,
    REDACTED,
    SAFE_KEYS,
    SENSITIVE_TOKENS,
    PrivacyViolationError,
    assert_clean,
    find_violations,
    is_blob,
    is_sensitive_key,
    key_tokens,
    normalize_key,
    redact_repr,
    scrub_frame,
)

__all__ = [
    "PrivacyViolationError",
    "assert_clean",
    "find_violations",
    "scrub_frame",
    "redact_repr",
    # 判定规则（对外暴露是为了让测试与工具有统一的入口）
    "is_sensitive_key",
    "is_blob",
    "key_tokens",
    "normalize_key",
    "MAX_DEPTH",
    "REDACTED",
    "SAFE_KEYS",
    "SENSITIVE_TOKENS",
]
