# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.services.eval_store import get_run, import_golden_set, list_runs, list_test_cases
from app.services.run_service import compare_runs, create_offline_run

router = APIRouter(prefix="/api", tags=["eval"])


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
    from app.services.live_runner import run_live
    record = run_live(
        project_path=payload.get("project_path", ""),
        case_ids=payload.get("case_ids", []),
        run_name=payload.get("name", "实时评测"),
    )
    return record.model_dump()


def _md_to_html(text: str) -> str:
    import markdown
    return markdown.markdown(text or "", extensions=["tables", "fenced_code", "nl2br"])


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
def diagnose(run_id: str, case_id: str) -> Dict[str, Any]:
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


@router.get("/compare")
def compare(run_a: str, run_b: str) -> Dict[str, Any]:
    try:
        return compare_runs(run_a, run_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
