# -*- coding: utf-8 -*-
"""Phase 6 DB bridge：把新 evaluator 的 runs/<run_id>/ 结果转成旧 RunRecord 并写入 inspector.db。

目的：Web UI / 对比 / 报告 / 医生仍按旧表结构读取，但实际评测由新 evaluator 执行。
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# evaluator/db_bridge.py -> evaluator -> agent-trace-inspector
_INSPECTOR_DIR = str(Path(__file__).resolve().parents[1])
if _INSPECTOR_DIR not in sys.path:
    sys.path.insert(0, _INSPECTOR_DIR)

from evaluator.manifest import RunManifest  # noqa: E402
from evaluator.scorers import deterministic  # noqa: E402
from schemas.eval import RunCaseResult, RunRecord  # noqa: E402


def _load_cases(path: str) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(q.get("id")): q for q in (data.get("questions") or []) if isinstance(q, dict)}


def _load_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(results_dir / "q_*.json"))):
        try:
            rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def build_case_result(
    row: Dict[str, Any],
    case: Optional[Dict[str, Any]],
    project_path: str = "",
) -> RunCaseResult:
    case = case or {}
    answer = str(row.get("answer") or "")
    agent_status = str(row.get("agent_status") or "ok")
    trace = ((row.get("raw") or {}).get("trace")) or None

    must_contain_result = row.get("must_contain_result") or {}
    must_not_contain_result = row.get("must_not_contain_result") or {}
    keyword_pass = row.get("keyword_passed")
    if keyword_pass is None:
        keyword_pass = bool(must_contain_result.get("passed") and must_not_contain_result.get("passed"))
    keyword_pass = bool(keyword_pass)

    reasons: List[str] = list(row.get("keyword_reasons") or [])
    if not reasons and not keyword_pass:
        reasons.extend(f"缺少必须包含：{kw}" for kw in (must_contain_result.get("miss") or []))
        reasons.extend(f"出现禁止包含：{kw}" for kw in (must_not_contain_result.get("violations") or []))

    tools_check = {"passed": None, "missing": [], "actual": [], "expected": []}
    route_check = {"passed": None, "expected": None, "actual": None}
    prompt_check = {"passed": None, "violations": [], "evidence": "无 Trace"}
    if trace:
        tools_check = deterministic.check_expected_tools(trace, case.get("expected_tools") or [])
        route_check = deterministic.check_expected_route(trace, case.get("expected_route"))
        plan_prompt = ""
        if project_path:
            try:
                from app.services.system_prompts import get_plan_system_prompt
                plan_prompt = get_plan_system_prompt(project_path)
            except Exception:
                plan_prompt = ""
        prompt_check = deterministic.check_prompt_compliance(trace, plan_prompt)

    if tools_check.get("passed") is False:
        reasons.append("缺少预期工具：" + "、".join(tools_check.get("missing") or []))
    if route_check.get("passed") is False:
        reasons.append(f"路由不匹配：期望 {route_check.get('expected')}，实际 {route_check.get('actual')}")
    if prompt_check.get("passed") is False:
        reasons.extend(prompt_check.get("violations") or [])
    if agent_status != "ok":
        reasons.append(f"Agent 状态异常：{agent_status} {row.get('agent_error') or ''}".strip())

    passed = (
        keyword_pass
        and tools_check.get("passed") is not False
        and route_check.get("passed") is not False
        and prompt_check.get("passed") is not False
        and agent_status == "ok"
    )

    metrics: Dict[str, Any] = {}
    if trace:
        metrics.update(deterministic.collect_metrics_from_trace(trace))
    metrics["tool_count"] = row.get("tool_count") if row.get("tool_count") is not None else metrics.get("tool_count", 0)
    metrics["llm_count"] = metrics.get("llm_count", 0)
    elapsed = row.get("elapsed")
    if elapsed is not None:
        metrics["duration_ms"] = int(float(elapsed) * 1000)
    metrics.update({
        "init_seconds": row.get("init_seconds"),
        "agent_seconds": row.get("agent_seconds"),
        "ragas": row.get("ragas"),
        "judge_valid": row.get("judge_valid"),
        "judge_errors": row.get("judge_errors") or [],
        "citation_result": row.get("citation_result"),
        "matched_variant": row.get("matched_variant"),
        "adapter": row.get("adapter"),
        "prompt_compliance": prompt_check,
        "tool_check": tools_check,
        "route_check": route_check,
    })

    return RunCaseResult(
        case_id=str(row.get("id") or case.get("id") or ""),
        question=str(row.get("question") or case.get("question") or ""),
        passed=bool(passed),
        keyword_pass=keyword_pass,
        matched_variant=row.get("matched_variant"),
        tool_pass=_bool_or_none(tools_check.get("passed")),
        route_pass=_bool_or_none(route_check.get("passed")),
        prompt_pass=_bool_or_none(prompt_check.get("passed")),
        prompt_violations=list(prompt_check.get("violations") or []),
        answer=answer,
        trace_id=((row.get("raw") or {}).get("trace_id")) or (trace or {}).get("trace_id"),
        actual_tools=tools_check.get("actual") or deterministic.collect_tools_from_trace(trace),
        metrics=metrics,
        reasons=reasons,
    )


def build_run_record(
    run_dir: str | Path,
    name: str = "",
    project_path: str = "",
) -> RunRecord:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = RunManifest.read(manifest_path) if manifest_path.exists() else RunManifest(run_id=run_dir.name)
    cases = _load_cases(manifest.evalset_file)
    rows = _load_rows(run_dir / "results")
    if not rows:
        raise ValueError(f"没有找到评测结果: {run_dir / 'results'}")

    results = [
        build_case_result(row, cases.get(str(row.get("id"))), project_path=project_path)
        for row in rows
    ]
    total = len(results)
    passed_cases = sum(1 for r in results if r.passed)
    durations = [r.metrics.get("duration_ms") for r in results if r.metrics.get("duration_ms")]
    created_at: datetime
    try:
        created_at = datetime.fromisoformat(manifest.created_at)
    except Exception:
        created_at = datetime.now().astimezone()

    return RunRecord(
        run_id=manifest.run_id,
        name=name or manifest.notes or f"{manifest.adapter} {manifest.run_id}",
        created_at=created_at,
        agent_name="genshin_story_agent" if manifest.adapter == "genshin" else manifest.adapter,
        total_cases=total,
        passed_cases=passed_cases,
        failed_cases=total - passed_cases,
        pass_rate=round(passed_cases / total * 100, 1) if total else 0.0,
        avg_duration_ms=round(sum(durations) / len(durations), 1) if durations else 0.0,
        results=results,
    )


def save_run_record(record: RunRecord) -> None:
    from app.services.eval_store import save_run
    save_run(record)
