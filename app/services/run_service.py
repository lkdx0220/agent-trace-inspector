# -*- coding: utf-8 -*-
"""运行与回归服务：基于已存储 Trace 做离线评测。"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List

from app.db import get_trace, list_traces
from app.services.eval_store import get_run, list_test_cases, save_run
from app.services.evaluator import compute_run_summary, evaluate_trace_for_case
from schemas.eval import RunCaseResult, RunRecord


def create_offline_run(name: str = "离线评测") -> RunRecord:
    cases = list_test_cases()
    traces = list_traces()
    trace_by_question = {}
    for t in traces:
        q = t.get("question")
        if q and q not in trace_by_question:
            trace_by_question[q] = t["trace_id"]

    results: List[RunCaseResult] = []
    for case in cases:
        trace_id = trace_by_question.get(case["question"])
        trace = get_trace(trace_id) if trace_id else None
        from schemas.eval import TestCase
        case_obj = TestCase(**case)
        results.append(evaluate_trace_for_case(case_obj, trace))

    summary = compute_run_summary(results)
    record = RunRecord(
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        name=name,
        created_at=datetime.now().astimezone(),
        total_cases=summary["total_cases"],
        passed_cases=summary["passed_cases"],
        failed_cases=summary["failed_cases"],
        pass_rate=summary["pass_rate"],
        avg_duration_ms=summary["avg_duration_ms"],
        results=results,
    )
    save_run(record)
    return record


def compare_runs(run_a_id: str, run_b_id: str) -> Dict[str, Any]:
    run_a = get_run(run_a_id)
    run_b = get_run(run_b_id)
    if not run_a or not run_b:
        raise ValueError("run not found")

    a_by_case = {r["case_id"]: r for r in run_a.get("results", [])}
    b_by_case = {r["case_id"]: r for r in run_b.get("results", [])}

    changed = []
    for case_id in sorted(set(a_by_case) | set(b_by_case)):
        ra = a_by_case.get(case_id)
        rb = b_by_case.get(case_id)
        if not ra or not rb:
            continue
        if ra.get("passed") != rb.get("passed"):
            changed.append({
                "case_id": case_id,
                "question": ra.get("question") or rb.get("question"),
                "run_a_passed": ra.get("passed"),
                "run_b_passed": rb.get("passed"),
            })

    return {
        "run_a": {"run_id": run_a["run_id"], "name": run_a["name"], "summary": run_a["summary"]},
        "run_b": {"run_id": run_b["run_id"], "name": run_b["name"], "summary": run_b["summary"]},
        "changed_count": len(changed),
        "changed": changed,
    }
