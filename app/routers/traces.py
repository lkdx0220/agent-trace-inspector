# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List

from app.db import get_trace, get_timeline, list_traces, save_trace
from app.services.metrics import compute_trace_metrics
from schemas.trace import Trace

router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.post("/import")
def import_trace(trace: Trace) -> Dict[str, Any]:
    """导入一条 Trace JSON（直接请求体，也接受 Trace 模型）。"""
    save_trace(trace)
    return {
        "success": True,
        "trace_id": trace.trace_id,
        "message": "Trace 已导入",
    }


@router.get("")
def get_all_traces() -> List[Dict[str, Any]]:
    return list_traces()


@router.get("/{trace_id}")
def get_one_trace(trace_id: str) -> Dict[str, Any]:
    data = get_trace(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return data


@router.get("/{trace_id}/metrics")
def get_trace_metrics(trace_id: str) -> Dict[str, Any]:
    data = get_trace(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return compute_trace_metrics(data)


@router.get("/{trace_id}/timeline")
def get_trace_timeline(trace_id: str) -> List[Dict[str, Any]]:
    data = get_timeline(trace_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trace timeline not found")
    return data
