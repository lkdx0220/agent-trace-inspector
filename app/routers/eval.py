# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.services.auth import require_local_or_token
from app.services.eval_store import get_run, import_golden_set, list_runs, list_test_cases
from app.services.path_guard import ensure_project_path
from app.services.rate_limit import audit_rate_limit, diagnose_rate_limit
from app.services.run_service import compare_runs, create_offline_run

router = APIRouter(prefix="/api", tags=["eval"], dependencies=[Depends(require_local_or_token)])
INSPECTOR_ROOT = Path(__file__).resolve().parents[2]


@router.get("/testcases")
def get_testcases() -> List[Dict[str, Any]]:
    return list_test_cases()


@router.post("/testcases/import")
def import_testcases(data: Dict[str, Any]) -> Dict[str, Any]:
    count = import_golden_set(data)
    return {"success": True, "imported": count}


@router.post("/runs/offline")
def run_offline(name: str = "离线评测") -> Dict[str, Any]:
    record = create_offline_run(name=name)
    return record.model_dump()


@router.post("/runs/live")
def run_live_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 6：Web UI 实时评测改走新 evaluator 的薄 CLI，结束后桥接写回 inspector.db。"""
    project_path = payload.get("project_path", "")
    try:
        project_path = str(ensure_project_path(project_path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    case_ids = payload.get("case_ids", [])
    if not isinstance(case_ids, list) or not all(isinstance(x, str) for x in case_ids):
        raise HTTPException(status_code=400, detail="case_ids 必须是字符串数组")
    if not case_ids or len(case_ids) > 100:
        raise HTTPException(status_code=400, detail="case_ids 数量必须在 1..100")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    name = str(payload.get("name") or "实时评测")
    cmd = [
        sys.executable,
        str(INSPECTOR_ROOT / "tools" / "run_evalset.py"),
        "--adapter", "genshin",
        "--workspace", str(Path(project_path).parent),
        "--ids", ",".join(case_ids),
        "--run-id", run_id,
        "--name", name,
        "--timeout", "600",
        "--force",
        "--save-db",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(INSPECTOR_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="实时评测超时")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise HTTPException(status_code=500, detail=f"实时评测失败: {tail}")
    record = get_run(run_id)
    if not record:
        tail = (proc.stdout or "")[-800:]
        raise HTTPException(status_code=500, detail=f"评测已结束但未找到 run {run_id}: {tail}")
    # 兼容旧 /runs/live 返回结构：除 summary 外，顶层也带一份汇总字段。
    summary = record.get("summary") or {}
    for key in ("total_cases", "passed_cases", "failed_cases", "pass_rate", "avg_duration_ms"):
        if key in summary:
            record[key] = summary[key]
    return record


def _md_to_html(text: str) -> str:
    import markdown
    from app.services.html_sanitizer import sanitize_html
    rendered = markdown.markdown(text or "", extensions=["tables", "fenced_code", "nl2br"])
    return sanitize_html(rendered)


@router.get("/runs/{run_id}/report/{case_id}")
def get_report_result(run_id: str, case_id: str) -> Dict[str, Any]:
    from app.services.eval_store import get_report
    text = get_report(run_id, case_id)
    if not text:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"case_id": case_id, "report": text, "report_html": _md_to_html(text)}


@router.post("/runs/{run_id}/report/{case_id}")
def generate_report(run_id: str, case_id: str) -> Dict[str, Any]:
    from app.services.report_generator import generate_analysis_report
    try:
        text = generate_analysis_report(run_id, case_id)
        return {"case_id": case_id, "report": text, "report_html": _md_to_html(text)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/runs/{run_id}/diagnose/{case_id}")
def get_diagnosis_result(run_id: str, case_id: str) -> Dict[str, Any]:
    from app.services.eval_store import get_diagnosis
    d = get_diagnosis(run_id, case_id)
    if not d:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    return d


@router.post("/runs/{run_id}/diagnose/{case_id}")
def diagnose(run_id: str, case_id: str, _rate: None = Depends(diagnose_rate_limit)) -> Dict[str, Any]:
    from app.services.diagnoser import diagnose_run_case
    try:
        return diagnose_run_case(run_id, case_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/runs")
def get_runs() -> List[Dict[str, Any]]:
    return list_runs()


@router.get("/runs/{run_id}")
def get_run_detail(run_id: str) -> Dict[str, Any]:
    data = get_run(run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Run not found")
    return data


@router.post("/runs/{run_id}/audit/{case_id}")
def audit_case_endpoint(run_id: str, case_id: str, payload: Dict[str, Any] = None, _rate: None = Depends(audit_rate_limit)) -> Dict[str, Any]:
    """对单个病例做轻量答案一致性审计（passed/failed 均可，通常跑 passed）。"""
    from app.services.case_audit import audit_case
    payload = payload or {}
    try:
        return audit_case(run_id, case_id, use_llm=bool(payload.get("use_llm", True)))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/runs/{run_id}/audit")
def audit_run_endpoint(run_id: str, payload: Dict[str, Any] = None, _rate: None = Depends(audit_rate_limit)) -> Dict[str, Any]:
    """批量审计一个 Run 的所有 passed 病例。"""
    from app.services.case_audit import audit_run
    payload = payload or {}
    try:
        return audit_run(run_id, use_llm=bool(payload.get("use_llm", True)), only_passed=bool(payload.get("only_passed", True)))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/runs/{run_id}/audit")
def list_audits(run_id: str) -> Dict[str, Any]:
    from app.services.eval_store import list_case_audits
    items = list_case_audits(run_id)
    return {"run_id": run_id, "count": len(items), "items": items}


@router.get("/compare")
def compare(run_a: str, run_b: str) -> Dict[str, Any]:
    try:
        return compare_runs(run_a, run_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
