# -*- coding: utf-8 -*-
"""Trace 指标计算。基于完整 Trace JSON，不依赖前端。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def _ms(a: Any, b: Any) -> int:
    if not a or not b:
        return 0
    try:
        if isinstance(a, str):
            a = datetime.fromisoformat(a)
        if isinstance(b, str):
            b = datetime.fromisoformat(b)
        return int((b - a).total_seconds() * 1000)
    except Exception:
        return 0


def compute_trace_metrics(trace: Dict[str, Any]) -> Dict[str, Any]:
    root = trace.get("root_span") or {}
    spans = []

    def walk(span):
        spans.append(span)
        for c in span.get("children", []):
            walk(c)

    walk(root)

    tools = [s for s in spans if s.get("span_type") == "tool"]
    llms = [s for s in spans if s.get("span_type") == "llm"]
    total_tool_ms = 0
    tool_by_name: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        start = t.get("start_time")
        end = t.get("end_time")
        d = _ms(start, end)
        total_tool_ms += d
        name = t.get("name") or "?"
        entry = tool_by_name.setdefault(name, {
            "count": 0,
            "success": 0,
            "not_found": 0,
            "intercepted": 0,
            "error": 0,
            "total_ms": 0,
        })
        entry["count"] += 1
        entry[t.get("status") or "success"] += 1
        entry["total_ms"] += d

    tool_names = sorted(tool_by_name.keys())
    not_found = sum(1 for t in tools if t.get("status") == "not_found")
    intercepted = sum(1 for t in tools if t.get("status") == "intercepted")
    errors = sum(1 for t in tools if t.get("status") == "error")
    meltdown = sum(1 for t in tools if t.get("meltdown_trigger"))

    phase_latency = {
        "rewrite": _ms(root.get("children", [{}])[0].get("start_time") if root.get("children") else None,
                      next((c.get("end_time") for c in root.get("children", []) if c.get("span_type") == "rewrite"), None)),
        "assess": None,
        "router": None,
        "plan": None,
        "answer": None,
    }

    # 从 spans 中直接取阶段时间
    for p in ("rewrite", "assess", "router", "answer"):
        span = next((s for s in spans if s.get("span_type") == p), None)
        if span:
            phase_latency[p] = _ms(span.get("start_time"), span.get("end_time"))
    plan_spans = [s for s in spans if s.get("span_type") == "llm" and s.get("name") in ("plan_agent", "fast_agent")]
    if plan_spans:
        phase_latency["plan"] = sum(_ms(s.get("start_time"), s.get("end_time")) for s in plan_spans)

    total_phase = sum(v or 0 for v in phase_latency.values())
    total_measured = total_phase + total_tool_ms
    duration = trace.get("duration_ms") or 0
    unattributed_ms = max(0, duration - total_measured)

    return {
        "trace_id": trace.get("trace_id"),
        "question": trace.get("question"),
        "duration_ms": trace.get("duration_ms"),
        "unattributed_ms": unattributed_ms,
        "span_count": len(spans),
        "tool_count": len(tools),
        "llm_count": len(llms),
        "unique_tools": tool_names,
        "tool_metrics": tool_by_name,
        "not_found_count": not_found,
        "intercepted_count": intercepted,
        "error_count": errors,
        "meltdown_count": meltdown,
        "total_tool_latency_ms": total_tool_ms,
        "avg_tool_latency_ms": round(total_tool_ms / len(tools), 1) if tools else 0,
        "phase_latency_ms": phase_latency,
        "events_count": len(trace.get("trace_events") or []),
    }
