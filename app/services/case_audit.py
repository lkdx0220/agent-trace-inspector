# -*- coding: utf-8 -*-
"""passed 病例轻量审计。

目标：评估器只保证“关键词/路由/工具”通过，但“通过”不一定代表答案质量好。
本模块对 passed 病例做低价复核：
1. 确定性相似度：4-gram Jaccard / 参考答案包含度 / 关键词覆盖 / 长度比；
2. 可选轻量 LLM（deepseek-v4-flash）比对实际答案与参考答案，输出强弱判断；
3. 记录到 case_audits 表，供批量查看“可疑通过”。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from app.services.doctor_tools import _short_circuit_answer
from app.services.eval_store import (
    get_case_audit,
    get_run,
    get_test_case,
    list_case_audits,
    save_case_audit,
)
from app.services.evaluator import _deepseek_key

AUDIT_MODEL = "deepseek-v4-flash"
LLM_TIMEOUT = 60


def _ngrams(text: str, n: int = 4) -> set:
    text = (text or "").strip()
    if len(text) < n:
        return {text} if text else set()
    return set(text[i:i + n] for i in range(len(text) - n + 1))


def _deterministic_audit(answer: str, expected: str, must_contain: List[str], must_not_contain: List[str]) -> Dict[str, Any]:
    ans = (answer or "").strip()
    ref = (expected or "").strip()
    out: Dict[str, Any] = {
        "answer_chars": len(ans),
        "reference_chars": len(ref),
        "jaccard": 0.0,
        "containment": 0.0,
        "keyword_rate": 0.0,
        "length_ratio": 0.0,
        "score": 0.0,
        "short_circuit": _short_circuit_answer(ans),
        "verdict": "unknown",
    }
    if not ref:
        out["verdict"] = "no_reference"
        return out

    a_grams = _ngrams(ans)
    r_grams = _ngrams(ref)
    if a_grams or r_grams:
        union = a_grams | r_grams
        if union:
            out["jaccard"] = round(len(a_grams & r_grams) / len(union), 4)
        if r_grams:
            out["containment"] = round(len(a_grams & r_grams) / len(r_grams), 4)

    kws = [str(k) for k in (must_contain or []) if k]
    if kws:
        out["keyword_rate"] = round(sum(1 for k in kws if k in ans) / len(kws), 4)
    if ref:
        out["length_ratio"] = round(min(1.0, len(ans) / max(1, len(ref))), 4)

    # 综合分：参考答案覆盖占大头，关键词覆盖不能全信。
    score = (
        0.45 * out["jaccard"]
        + 0.25 * out["containment"]
        + 0.15 * out["keyword_rate"]
        + 0.15 * out["length_ratio"]
    )
    out["score"] = round(score, 4)

    if out["short_circuit"]:
        out["verdict"] = "suspicious"
    elif out["score"] >= 0.75:
        out["verdict"] = "strong"
    elif out["score"] >= 0.5:
        out["verdict"] = "medium"
    else:
        out["verdict"] = "weak"
    return out


def _llm_audit(question: str, answer: str, reference: str) -> Optional[Dict[str, Any]]:
    """轻量 LLM 比较实际答案与参考答案。失败/无 key 返回 None。"""
    key = _deepseek_key()
    if not key:
        return None
    prompt = f"""你是答案一致性轻量评审。请比较【实际答案】与【参考答案】，判断实际答案是否真的覆盖了参考答案的核心内容。

只输出 JSON，不要其他文字，格式：
{{"verdict": "strong|partial|weak|contradicts|unverifiable", "score": 0.0, "reason": "一句话原因"}}

说明：
- strong：实际答案完整/基本覆盖参考核心；
- partial：覆盖一部分，但缺关键点或表述偏差；
- weak：只沾边，核心内容缺失；
- contradicts：实际答案与参考结论矛盾；
- unverifiable：无法判断。

【题目】
{question}

【参考答案】
{reference[:3000]}

【实际答案】
{answer[:3000]}
"""
    base = {
        "model": AUDIT_MODEL,
        "messages": [
            {"role": "system", "content": "你是轻量答案一致性审计器，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }
    payloads = [{**base, "thinking": {"type": "disabled"}}, base]
    for payload in payloads:
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            content = resp.json()["choices"][0]["message"]["content"].strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                continue
            data = json.loads(content[start:end])
            return {
                "verdict": str(data.get("verdict") or "unverifiable"),
                "score": float(data.get("score") or 0.0),
                "reason": str(data.get("reason") or "")[:500],
            }
        except Exception:
            continue
    return None


def _merge(det: Dict[str, Any], llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not llm:
        return det
    merged = dict(det)
    merged["llm_verdict"] = llm["verdict"]
    merged["llm_score"] = round(llm["score"], 4)
    merged["llm_reason"] = llm["reason"]
    # LLM 结果优先作为最终判断；但确定性“短路”仍标记可疑。
    if det.get("short_circuit"):
        merged["verdict"] = "suspicious"
    elif llm["verdict"] in {"strong", "partial"}:
        merged["verdict"] = "strong" if llm["verdict"] == "strong" else "medium"
    elif llm["verdict"] in {"weak", "contradicts"}:
        merged["verdict"] = "weak"
    else:
        merged["verdict"] = det["verdict"]
    # 若 LLM 分数缺失/异常，保留确定性分。
    if not (0.0 <= merged.get("llm_score", 0.0) <= 1.0):
        merged["llm_score"] = det["score"]
    return merged


def audit_case(run_id: str, case_id: str, use_llm: bool = True) -> Dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise ValueError(f"Run 不存在: {run_id}")
    result = next((r for r in run.get("results", []) if r.get("case_id") == case_id), None)
    if not result:
        raise ValueError(f"Run {run_id} 中不存在 case {case_id}")
    case = get_test_case(case_id)
    if not case:
        raise ValueError(f"TestCase 不存在: {case_id}")

    answer = str(result.get("answer") or "")
    expected = str(case.get("expected_answer") or "")
    det = _deterministic_audit(
        answer,
        expected,
        case.get("must_contain") or [],
        case.get("must_not_contain") or [],
    )
    llm = _llm_audit(str(result.get("question") or case.get("question") or ""), answer, expected) if use_llm else None
    merged = _merge(det, llm)

    record = {
        "run_id": run_id,
        "case_id": case_id,
        "question": result.get("question") or case.get("question") or "",
        "passed": bool(result.get("passed")),
        "verdict": merged["verdict"],
        "score": merged["score"],
        "details": merged,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    save_case_audit(record)
    existing = get_case_audit(run_id, case_id)
    return existing or record


def audit_run(run_id: str, use_llm: bool = True, only_passed: bool = True) -> Dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise ValueError(f"Run 不存在: {run_id}")
    cases = run.get("results", [])
    if only_passed:
        cases = [r for r in cases if r.get("passed")]
    results = []
    for r in cases:
        try:
            results.append(audit_case(run_id, r["case_id"], use_llm=use_llm))
        except Exception:
            # 不把内部异常细节写入审计结果，避免泄露路径/Key/堆栈。
            results.append({
                "run_id": run_id,
                "case_id": r.get("case_id"),
                "question": r.get("question"),
                "passed": r.get("passed"),
                "verdict": "error",
                "score": 0.0,
                "details": {"error": "audit failed"},
                "created_at": datetime.now().astimezone().isoformat(),
            })
    weak = [x for x in results if x.get("verdict") in {"weak", "suspicious", "contradicts"}]
    return {
        "run_id": run_id,
        "audited": len(results),
        "weak_count": len(weak),
        "weak_cases": weak,
        "results": results,
    }
