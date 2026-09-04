# -*- coding: utf-8 -*-
"""项目医生 API。"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.services.doctor_tools import DEFAULT_PROJECT_PATH
from app.services.eval_store import get_prescription, save_prescription
from app.services.rate_limit import doctor_rate_limit
from app.services.project_doctor import prescribe_run_case

router = APIRouter(prefix="/api", tags=["doctor"])


@router.post("/runs/{run_id}/doctor/{case_id}")
def run_doctor(run_id: str, case_id: str, payload: Optional[Dict[str, Any]] = None, _rate: None = Depends(doctor_rate_limit)) -> Dict[str, Any]:
    payload = payload or {}
    project_path = payload.get("project_path") or DEFAULT_PROJECT_PATH
    model = payload.get("model")
    try:
        out = prescribe_run_case(run_id, case_id, project_path=project_path, model=model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if out.get("ok"):
        # 保存完整处方与证据（当前规模可接受，单题约 100~150KB）
        stored = dict(out)
        save_prescription(run_id, case_id, stored, model=out.get("model") or "")
    return out


@router.get("/runs/{run_id}/doctor/{case_id}")
def get_doctor_result(run_id: str, case_id: str) -> Dict[str, Any]:
    data = get_prescription(run_id, case_id)
    if not data:
        raise HTTPException(status_code=404, detail="Doctor prescription not found")
    return data
