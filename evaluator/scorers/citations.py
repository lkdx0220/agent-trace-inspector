# -*- coding: utf-8 -*-
"""引用真实性核对：区分精确验证与宽松验证。"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_META_ABSENCE_MARKERS = [
    "未在", "未提及", "未包含", "未收录", "未明确", "未找到", "无法回答",
    "没有", "不包含", "不在", "并不存在", "未验证",
]


def _is_meta_absence_quote(answer: str, quote: str) -> bool:
    idx = answer.find(quote)
    if idx < 0:
        return False
    around = answer[max(0, idx - 15):idx + len(quote) + 30]
    return any(marker in around for marker in _META_ABSENCE_MARKERS)


def check_citations(answer: str, contexts: str) -> Dict[str, Any]:
    """检查答案中引用的文本是否真的出现在检索结果中。

    区分“精确验证”和“宽松验证”：宽松匹配只对长度 >= 8 的引用生效，
    且要求所有字符按顺序出现，避免短引用/乱序引用被误判为已验证。
    """
    quoted = re.findall(r'「([^」]{4,80})」', answer)
    quoted += re.findall(r'"([^"]{4,80})"', answer)
    quoted = list(set(quoted))
    quoted = [q for q in quoted if not _is_meta_absence_quote(answer, q)]

    if not quoted:
        return {
            "total": 0,
            "verified": 0,
            "exact_verified": 0,
            "loose_verified": [],
            "unverified": [],
            "passed": True,
        }

    exact_verified: List[str] = []
    loose_verified: List[str] = []
    unverified: List[str] = []
    ctx_normalized = contexts.replace("\n", "").replace(" ", "")

    for quote in quoted:
        q_normalized = quote.replace("\n", "").replace(" ", "")
        if q_normalized in ctx_normalized:
            exact_verified.append(quote)
            continue
        found = 0
        pos = 0
        for ch in q_normalized:
            nxt = ctx_normalized.find(ch, pos)
            if nxt >= 0:
                found += 1
                pos = nxt + 1
        if len(q_normalized) >= 8 and found == len(q_normalized):
            loose_verified.append(quote)
        else:
            unverified.append(quote)

    return {
        "total": len(quoted),
        "verified": len(exact_verified) + len(loose_verified),
        "exact_verified": len(exact_verified),
        "loose_verified": loose_verified,
        "unverified": unverified,
        "passed": len(unverified) == 0,
    }
