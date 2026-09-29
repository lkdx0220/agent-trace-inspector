# -*- coding: utf-8 -*-
"""Shared scorers：judge / RAGAS / 关键词 / 引用 / 确定性检查。"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluator.scorers import citations, deterministic, judge, keywords, ragas  # noqa: F401

__all__ = ["citations", "deterministic", "judge", "keywords", "ragas", "score_case"]


def score_case(
    case: Dict[str, Any],
    answer: str,
    contexts: str,
    reference: str = "",
    eval_mode: str = "",
    hit_rate_threshold: float = 0.8,
) -> Dict[str, Any]:
    """对一题的 answer/contexts 打分，返回与旧 harness 兼容的结果结构。"""
    judge.reset_errors()
    judge.reset_samples()
    settings = judge.get_settings()
    truncation = {
        "answer": len(answer or "") > int(settings.answer_chars),
        "contexts": len(contexts or "") > int(settings.contexts_chars),
        "reference": len(reference or "") > int(settings.reference_chars),
        "limits": {
            "answer_chars": settings.answer_chars,
            "contexts_chars": settings.contexts_chars,
            "reference_chars": settings.reference_chars,
        },
    }
    question = str(case.get("question") or "")
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
        matched_variant = None
        variant_results: List[Dict[str, Any]] = []
        keyword_reasons: List[str] = []
    else:
        keyword_result = keywords.check_case_keywords(case, answer, hit_rate_threshold=hit_rate_threshold)
        must_contain_result = keyword_result["must_contain_result"]
        must_not_contain_result = keyword_result["must_not_contain_result"]
        matched_variant = keyword_result.get("matched_variant")
        variant_results = keyword_result.get("variant_results") or []
        keyword_reasons = keyword_result.get("reasons") or []

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
        "matched_variant": matched_variant,
        "variant_results": variant_results,
        "keyword_reasons": keyword_reasons,
        "keyword_passed": bool(must_contain_result.get("passed") and must_not_contain_result.get("passed")),
        "citation_result": citation_result,
        "judge_valid": judge_valid,
        "judge_errors": judge_errors,
        "judge_samples": judge.get_samples(),
        "judge_truncation": truncation,
        "keyword_judge_unavailable": {
            "must_contain": must_contain_result.get("semantic_unavailable", []),
            "must_not_contain": must_not_contain_result.get("semantic_unavailable", []),
        },
    }
