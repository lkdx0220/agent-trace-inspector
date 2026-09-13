# -*- coding: utf-8 -*-
"""确定性 Trace 检查：工具调用、路由模式、系统提示词工具规则合规。

这些检查只依赖 Trace JSON 和提示词原文，不调用 LLM，也不 import 被测项目。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def walk_spans(span: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(span, dict):
        return out
    out.append(span)
    for child in span.get("children") or []:
        out.extend(walk_spans(child))
    return out


def collect_tools_from_trace(trace: Optional[Dict[str, Any]]) -> List[str]:
    if not trace:
        return []
    tools: List[str] = []
    for span in walk_spans(trace.get("root_span") or {}):
        if span.get("span_type") == "tool" and span.get("name"):
            tools.append(str(span["name"]))
    return tools


def collect_metrics_from_trace(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not trace:
        return {"duration_ms": None, "tool_count": 0, "llm_count": 0}
    spans = walk_spans(trace.get("root_span") or {})
    tools = [s for s in spans if s.get("span_type") == "tool"]
    llms = [s for s in spans if s.get("span_type") in ("llm", "answer")]
    return {
        "duration_ms": trace.get("duration_ms"),
        "tool_count": len(tools),
        "llm_count": len(llms),
    }


def collect_plan_texts(trace: Optional[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for event in (trace or {}).get("trace_events") or []:
        if not isinstance(event, dict) or event.get("event") != "plan":
            continue
        data = event.get("data") or {}
        text = data.get("execution_plan") or data.get("plan_text") or ""
        if text:
            texts.append(str(text))
    return texts


def check_expected_tools(trace: Optional[Dict[str, Any]], expected_tools: Optional[List[str]]) -> Dict[str, Any]:
    """expected_tools 为空表示不检查（passed=None，与旧 evaluator 的 tool_pass=None 对齐）。"""
    actual = collect_tools_from_trace(trace)
    expected = [str(t) for t in (expected_tools or []) if str(t).strip()]
    if not expected:
        return {"passed": None, "expected": [], "actual": actual, "missing": []}
    missing = [name for name in expected if name not in actual]
    return {"passed": not missing, "expected": expected, "actual": actual, "missing": missing}


def check_expected_route(trace: Optional[Dict[str, Any]], expected_route: Optional[str]) -> Dict[str, Any]:
    actual = ((trace or {}).get("metadata") or {}).get("execution_mode")
    if not expected_route:
        return {"passed": None, "expected": None, "actual": actual}
    return {
        "passed": actual == expected_route,
        "expected": expected_route,
        "actual": actual,
    }


def check_prompt_compliance(trace: Optional[Dict[str, Any]], plan_prompt_text: str) -> Dict[str, Any]:
    """系统提示词工具规则合规：非豁免问题必须调用工具，或给出 tool_skip_reason。

    返回 passed:
      True  = 有工具调用，或未调工具但 plan 文本写了 tool_skip_reason
      False = 无工具调用且无豁免
      None  = 无法判定（没有提示词原文，或 Trace 没有 plan 事件）
    """
    if not plan_prompt_text:
        return {
            "passed": None,
            "violations": [],
            "evidence": "未找到规划系统提示词文件，跳过系统提示词合规检查",
            "tool_count": len(collect_tools_from_trace(trace)),
            "plan_count": 0,
        }

    actual_tools = collect_tools_from_trace(trace)
    plans = collect_plan_texts(trace)
    if actual_tools:
        return {
            "passed": True,
            "violations": [],
            "evidence": "存在工具调用，符合系统提示词工具要求",
            "tool_count": len(actual_tools),
            "plan_count": len(plans),
        }
    if not plans:
        return {
            "passed": None,
            "violations": [],
            "evidence": "trace 中缺少 plan 事件，无法判断是否违反工具调用规则",
            "tool_count": 0,
            "plan_count": 0,
        }
    if any("tool_skip_reason" in text for text in plans):
        return {
            "passed": True,
            "violations": [],
            "evidence": "未调用工具但执行报告包含 tool_skip_reason，属于系统提示词允许的豁免场景",
            "tool_count": 0,
            "plan_count": len(plans),
        }
    return {
        "passed": False,
        "violations": ["违反系统提示词：非豁免场景必须调用工具，但实际工具调用次数为 0"],
        "evidence": "无工具调用，且 plan 执行报告中未出现 tool_skip_reason",
        "tool_count": 0,
        "plan_count": len(plans),
    }
