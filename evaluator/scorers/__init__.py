# -*- coding: utf-8 -*-
"""Shared scorers：judge / RAGAS / 关键词 / 引用。"""
from __future__ import annotations

from typing import Any, Dict

from evaluator.scorers import citations, judge, keywords, ragas  # noqa: F401

__all__ = ["citations", "judge", "keywords", "ragas", "score_case"]


def score_case(
    case: Dict[str, Any],
    answer: str,
    contexts: str,
    reference: str = "",
    eval_mode: str = "",
) -> Dict[str, Any]:
    """对一题的 answer/contexts 打分，返回与旧 harness 兼容的结果结构。"""
    judge.reset_errors()
    question = str(case.get("question") or "")
    match_mode = str(case.get("match_mode") or "all")
    must_contain = case.get("must_contain") or []
    must_not_contain = case.get("must_not_contain") or []
    reference_answer = reference or str(case.get("reference_answer") or "")
    mode = eval_mode or str(case.get("eval_mode") or "auto")

    ragas_result = {
        "faithfulness": ragas.score_faithfulness(answer, contexts),
        "answer_relevancy": ragas.score_answer_relevancy(answer, question),
        "context_precision": ragas.score_context_precision(contexts, question),
        "context_recall": ragas.score_context_recall(contexts, reference_answer),
    }

    if mode == "manual":
        must_contain_result: Dict[str, Any] = {
            "passed": True,
            "hit": [],
            "semantic_hit": [],
            "miss": [],
            "hit_rate": 1.0,
            "judge_unavailable": False,
            "semantic_unavailable": [],
            "skipped": "manual",
        }
        must_not_contain_result: Dict[str, Any] = {
            "passed": True,
            "violations": [],
            "judge_unavailable": False,
            "semantic_unavailable": [],
            "skipped": "manual",
        }
    else:
        must_contain_result = keywords.check_must_contain(answer, must_contain, match_mode)
        must_not_contain_result = keywords.check_must_not_contain(answer, must_not_contain, question)

    citation_result = citations.check_citations(answer, contexts)
    judge_errors = judge.get_errors()
    judge_valid = (
        not judge_errors
        and not must_contain_result.get("judge_unavailable")
        and not must_not_contain_result.get("judge_unavailable")
    )

    return {
        "ragas": ragas_result,
        "must_contain_result": must_contain_result,
        "must_not_contain_result": must_not_contain_result,
        "citation_result": citation_result,
        "judge_valid": judge_valid,
        "judge_errors": judge_errors,
        "keyword_judge_unavailable": {
            "must_contain": must_contain_result.get("semantic_unavailable", []),
            "must_not_contain": must_not_contain_result.get("semantic_unavailable", []),
        },
    }
