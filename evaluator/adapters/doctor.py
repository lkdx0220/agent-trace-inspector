# -*- coding: utf-8 -*-
"""项目医生 adapter：子进程 + JSON 协议。

父进程通过 stdin 传入：
  {
    "case_id": "D-F1-routing",
    "question": "...",
    "run_id": "run_86ebe199",
    "target_case_id": "F1",
    "project_path": "C:/.../CASE-原神剧情助手-修改用",
    "case": {...}
  }
子进程通过 stdout 返回 evaluator.contract.AgentResult JSON。

设计要点：
1. adapter 不直接 import 被测原神项目，只调用 inspector 自己的医生入口；
2. 医生输出是结构化 JSON，adapter 把它压成“可判分文本 answer”：
   - resolution.conclusion_kind / stage / primary_root_cause / target_file
   - report.diagnosis.summary / issue_classification
   - report.prescriptions（每条 issue/root_cause/target_file/evidence_ids）
3. 完整 resolution/fact_sheet/coverage 放进 contexts，供 RAGAS/引用类 scorer 使用。
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# evaluator/adapters/doctor.py -> adapters -> evaluator -> agent-trace-inspector
_INSPECTOR_DIR = str(Path(__file__).resolve().parents[2])
if _INSPECTOR_DIR not in sys.path:
    sys.path.insert(0, _INSPECTOR_DIR)

DEFAULT_PROJECT_PATH = r"C:\Users\24701\Desktop\原神剧情\CASE-原神剧情助手-修改用"


def _compose_answer(doctor: Dict[str, Any]) -> str:
    """把医生结构化输出压成判分文本；关键词 scorer 只读 answer 字段。"""
    resolution = doctor.get("resolution") or {}
    report = doctor.get("report") or {}
    diagnosis = report.get("diagnosis") or {}
    prescriptions = report.get("prescriptions") or []

    lines: List[str] = [
        f"conclusion_kind: {resolution.get('conclusion_kind', '')}",
        f"stage: {resolution.get('stage', '')}",
        f"primary_root_cause: {resolution.get('primary_root_cause', '')}",
        f"target_file: {resolution.get('target_file', '')}",
        f"diagnosis_summary: {diagnosis.get('summary', '')}",
        f"issue_classification: {diagnosis.get('issue_classification', '')}",
        f"prescriptions_count: {len(prescriptions)}",
    ]
    for index, item in enumerate(prescriptions, 1):
        lines.append(
            f"[处方{index}] issue={item.get('issue', '')} | "
            f"root_cause={item.get('root_cause', '')} | "
            f"target_file={item.get('target_file', '')} | "
            f"evidence_ids={','.join(str(e) for e in (item.get('evidence_ids') or []))}"
        )
    return "\n".join(lines)


def _compose_contexts(doctor: Dict[str, Any]) -> str:
    """医生完整诊断上下文，截断到判分窗口内。"""
    pipeline = doctor.get("pipeline")
    payload = {
        "resolution": doctor.get("resolution"),
        "coverage": doctor.get("coverage"),
        "report": doctor.get("report"),
        "fact_sheet": doctor.get("fact_sheet"),
        "pipeline": {
            "evidence_order_ids": list((doctor.get("evidence_by_order") or {}).keys()),
            "stage_count": len((pipeline or {}).get("evidence_by_order") or {}) if isinstance(pipeline, dict) else 0,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)[:20000]


def _compose_tool_trace(doctor: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence_by_order = doctor.get("evidence_by_order") or {}
    trace: List[Dict[str, Any]] = []
    for order in doctor.get("lab_orders") or []:
        order_id = str(order.get("id") or "")
        entries = evidence_by_order.get(order_id) or []
        first_ok = next((e for e in entries if e.get("ok")), None)
        summary = str((first_ok or {}).get("summary") or "")
        trace.append({
            "name": order_id,
            "status": "success" if first_ok else "error",
            "result_preview": summary[:300],
            "result_length": len(summary),
        })
    return trace


def run_agent(case: Dict[str, Any]) -> Dict[str, Any]:
    """运行项目医生，返回 AgentResult 兼容 dict。"""
    from evaluator.contract import (
        AgentResult,
        AgentTimings,
        STATUS_AGENT_ERROR,
        STATUS_EMPTY,
        STATUS_OK,
    )

    run_id = str(case.get("run_id") or "")
    target_case_id = str(case.get("target_case_id") or case.get("target_id") or "")
    if not run_id or not target_case_id:
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error="doctor adapter 需要 case.run_id 与 case.target_case_id",
            adapter="doctor",
        ).to_dict()

    project_path = str(
        case.get("project_path")
        or os.environ.get("DOCTOR_PROJECT_PATH")
        or DEFAULT_PROJECT_PATH
    )

    started = time.time()
    buffer = io.StringIO()
    try:
        from app.services.project_doctor import prescribe_run_case
    except Exception as exc:
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"import project_doctor failed: {type(exc).__name__}: {exc}",
            adapter="doctor",
            timings=AgentTimings(agent_seconds=round(time.time() - started, 1)),
        ).to_dict()

    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            doctor = prescribe_run_case(run_id, target_case_id, project_path=project_path)
    except Exception as exc:
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"prescribe_run_case failed: {type(exc).__name__}: {exc}",
            adapter="doctor",
            timings=AgentTimings(agent_seconds=round(time.time() - started, 1)),
        ).to_dict()

    if not doctor.get("ok"):
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"医生返回失败: {doctor.get('error') or 'unknown'}",
            adapter="doctor",
            timings=AgentTimings(agent_seconds=round(time.time() - started, 1)),
            raw={"doctor_error": doctor.get("error")},
        ).to_dict()

    answer = _compose_answer(doctor)
    resolution = doctor.get("resolution") or {}
    model_name = str(doctor.get("model") or "")
    result = AgentResult(
        answer=answer,
        contexts=_compose_contexts(doctor),
        tool_trace=_compose_tool_trace(doctor),
        timings=AgentTimings(
            init_seconds=0.0,
            agent_seconds=round(time.time() - started, 1),
        ),
        status=STATUS_OK if answer.strip() else STATUS_EMPTY,
        error="" if answer.strip() else "医生 answer 为空",
        adapter="doctor",
        adapter_version="1.0",
        agent_version=f"doctor-pipeline-1.0/{model_name}",
        raw={
            "run_id": run_id,
            "target_case_id": target_case_id,
            "coverage": doctor.get("coverage"),
            "conclusion_kind": resolution.get("conclusion_kind"),
            "stage": resolution.get("stage"),
            "report_fallback": bool((doctor.get("report") or {}).get("_note") == "fallback"),
        },
    )
    return result.to_dict()


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw or "{}")
    except Exception as exc:
        print(json.dumps({"status": "agent_error", "error": f"invalid stdin JSON: {exc}"}, ensure_ascii=False))
        return 2
    result = run_agent(payload if isinstance(payload, dict) else {})
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") != "agent_error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
