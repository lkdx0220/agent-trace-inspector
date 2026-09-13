# -*- coding: utf-8 -*-
"""关键词判分：literal / semantic / structural 三类，及禁止词语义裁判。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from evaluator.scorers import judge

# 只有这些“通用概念/描述词”允许语义兜底；专名/书名/任务名必须精确命中。
SEMANTIC_ALLOWED_KEYWORDS = {
    # 防幻觉/未收录类
    "未收录", "无法回答", "未明确", "作者",
    # 通用行为/概念类
    "喝酒", "自由", "守护", "现实", "见证", "从容", "命运",
    "接纳", "承认过去", "童话", "备份", "抛弃", "人类", "扮演",
    "实力", "胜利者", "对立", "环形", "塔楼", "开场动画", "愿景",
}

# 这些禁止词不裸词判违规：可能是复述原文设定（如渊下宫三界观）。
SEMANTIC_FORBIDDEN_KEYWORDS = {"黑暗"}

_NEGATIONS = ["不认为", "并非", "并不", "不是", "没有", "否认", "否定", "绝非", "不可能"]


def normalize_keyword(item: Any) -> Tuple[str, str, List[str]]:
    """返回 (text, match, terms)；旧字符串格式自动判断 semantic/literal。"""
    if isinstance(item, str):
        text = item.strip()
        match = "semantic" if text in SEMANTIC_ALLOWED_KEYWORDS else "literal"
        return text, match, [text]
    if isinstance(item, dict):
        text = str(item.get("text") or "").strip()
        match = str(item.get("match") or ("semantic" if text in SEMANTIC_ALLOWED_KEYWORDS else "literal"))
        if match not in {"literal", "semantic", "structural"}:
            match = "literal"
        aliases = [str(a).strip() for a in (item.get("aliases") or []) if str(a).strip()]
        return text, match, [text] + aliases
    return "", "literal", []


def _strip_negation(text: str, keyword: str) -> bool:
    """keyword 出现在 text 中，且不在否定语境里。"""
    if not keyword:
        return False
    idx = text.find(keyword)
    if idx == -1:
        return False
    prefix = text[max(0, idx - 6):idx]
    return not any(neg in prefix for neg in _NEGATIONS)


def _literal_hit(answer: str, terms: List[str]) -> bool:
    return any(_strip_negation(answer, term) for term in terms if term)


def semantic_keyword_check(answer: str, keyword: str) -> Optional[bool]:
    """判断答案是否表达该关键词含义；judge 不可用返回 None。"""
    prompt = f"""判断以下【答案】是否表达了「{keyword}」的含义。
允许同义改写、近义表述或等价说法，不要求出现原词。
例如关键词为「未收录」时，「并未包含」「不在列表中」「未提及」等表述均算表达了该含义。

【答案】
{judge.untrusted("ANSWER", answer[: judge.get_settings().answer_chars])}

只输出「是」或「否」："""
    try:
        result = judge.call_judge(prompt, max_tokens=8)
    except judge.JudgeUnavailableError:
        return None
    return result.strip().startswith("是")


def semantic_forbidden_check(question: str, answer: str, keyword: str) -> Optional[bool]:
    """判断禁止词是否是简化标签式违规；judge 不可用返回 None（调用方保守判违规）。"""
    prompt = f"""你是关键词违规裁判。
用户问题：
{judge.untrusted("QUESTION", question or "（未提供）")}

答案：
{judge.untrusted("ANSWER", answer[: judge.get_settings().answer_chars])}

判断答案中的「{keyword}」：
- 如果它是在把提问对象简单定性/等同为“黑暗、邪恶、外来敌人”这类简化标签，输出「是」；
- 如果它只是在复述知识库原文设定、剧情对话、专有设定（例如渊下宫三界观、能量体系、组织历史），输出「否」。

只输出「是」或「否」："""
    try:
        result = judge.call_judge(prompt, max_tokens=8)
    except judge.JudgeUnavailableError:
        return None
    return result.strip().startswith("是")


def check_must_contain(
    answer: str,
    keywords: List[Any],
    match_mode: str = "all",
    hit_rate_threshold: float = 0.8,
) -> Dict[str, Any]:
    if not keywords:
        return {
            "passed": True,
            "hit": [],
            "semantic_hit": [],
            "miss": [],
            "hit_rate": 1.0,
            "judge_unavailable": False,
            "semantic_unavailable": [],
        }

    hits: List[str] = []
    semantic_hits: List[str] = []
    misses: List[str] = []
    semantic_unavailable: List[str] = []

    for raw in keywords:
        text, match, terms = normalize_keyword(raw)
        if not text:
            continue
        if _literal_hit(answer, terms):
            hits.append(text)
            continue
        if match == "semantic":
            flag = semantic_keyword_check(answer, text)
            if flag is True:
                semantic_hits.append(text)
            elif flag is None:
                misses.append(text)
                semantic_unavailable.append(text)
            else:
                misses.append(text)
        else:
            misses.append(text)

    total = len(hits) + len(semantic_hits) + len(misses)
    hit_rate = (len(hits) + len(semantic_hits)) / total if total else 1.0
    if match_mode == "any":
        passed = len(hits) + len(semantic_hits) >= 1
    else:
        passed = hit_rate >= hit_rate_threshold

    return {
        "passed": passed,
        "hit": hits,
        "semantic_hit": semantic_hits,
        "miss": misses,
        "hit_rate": round(hit_rate, 2),
        "judge_unavailable": bool(semantic_unavailable),
        "semantic_unavailable": semantic_unavailable,
    }


def check_must_not_contain(answer: str, keywords: List[Any], question: str = "") -> Dict[str, Any]:
    if not keywords:
        return {
            "passed": True,
            "violations": [],
            "judge_unavailable": False,
            "semantic_unavailable": [],
        }

    violations: List[str] = []
    semantic_unavailable: List[str] = []
    for raw in keywords:
        text, match, terms = normalize_keyword(raw)
        if not text:
            continue
        if not _literal_hit(answer, terms):
            continue
        if match == "semantic" or text in SEMANTIC_FORBIDDEN_KEYWORDS:
            flag = semantic_forbidden_check(question, answer, text)
            if flag is False:
                continue
            if flag is None:
                semantic_unavailable.append(text)
            violations.append(text)
        else:
            violations.append(text)

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "judge_unavailable": bool(semantic_unavailable),
        "semantic_unavailable": semantic_unavailable,
    }
