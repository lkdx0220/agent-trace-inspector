# -*- coding: utf-8 -*-
"""evaluator.scorers 纯函数补盲：确定性 Trace 检查与引用核对。"""

from __future__ import annotations

from evaluator.scorers import citations, deterministic


def test_citations_meta_absence_and_empty():
    assert citations._is_meta_absence_quote("未在「引蝶之章」中找到", "引蝶之章") is True
    assert citations._is_meta_absence_quote("「引蝶之章」讲了什么", "引蝶之章") is False
    result = citations.check_citations("没有引号", "上下文")
    assert result == {
        "total": 0,
        "verified": 0,
        "exact_verified": 0,
        "loose_verified": [],
        "unverified": [],
        "passed": True,
    }


def test_citations_exact_loose_and_unverified():
    exact = citations.check_citations("他说「ABCDEFGH」", "前缀 ABCDEFGH 后缀")
    assert exact["exact_verified"] == 1
    assert exact["passed"] is True

    loose = citations.check_citations("「ABCDEFGH」", "A-B-C-D-E-F-G-H")
    assert loose["loose_verified"] == ["ABCDEFGH"]
    assert loose["passed"] is True

    bad = citations.check_citations("「ZZZZZZZZ」", "完全无关")
    assert bad["unverified"] == ["ZZZZZZZZ"]
    assert bad["passed"] is False


def test_deterministic_walk_and_collect():
    assert deterministic.walk_spans(None) == []
    assert deterministic.collect_tools_from_trace(None) == []
    assert deterministic.collect_metrics_from_trace(None) == {
        "duration_ms": None,
        "tool_count": 0,
        "llm_count": 0,
    }

    trace = {
        "duration_ms": 123,
        "root_span": {
            "span_type": "agent",
            "children": [
                {"span_type": "tool", "name": "hybrid_search"},
                {"span_type": "llm", "name": "plan"},
                {"span_type": "answer", "name": "answer"},
            ],
        },
        "trace_events": [
            {"event": "plan", "data": {"execution_plan": "调用 hybrid_search"}},
            {"event": "plan", "data": {"plan_text": "备用 plan 文本"}},
            {"event": "tool_start", "data": {}},
        ],
    }
    assert deterministic.collect_tools_from_trace(trace) == ["hybrid_search"]
    metrics = deterministic.collect_metrics_from_trace(trace)
    assert metrics["tool_count"] == 1
    assert metrics["llm_count"] == 2
    assert deterministic.collect_plan_texts(trace) == ["调用 hybrid_search", "备用 plan 文本"]


def test_deterministic_expected_tools_and_route():
    trace = {"root_span": {"span_type": "agent", "children": [{"span_type": "tool", "name": "hybrid_search"}]}}
    assert deterministic.check_expected_tools(trace, None)["passed"] is None
    assert deterministic.check_expected_tools(trace, ["hybrid_search"])["passed"] is True
    missing = deterministic.check_expected_tools(trace, ["query_quest"])
    assert missing["passed"] is False
    assert missing["missing"] == ["query_quest"]

    assert deterministic.check_expected_route(trace, None)["passed"] is None
    assert deterministic.check_expected_route({"metadata": {"execution_mode": "L2"}}, "L2")["passed"] is True


def test_deterministic_prompt_compliance_branches():
    no_prompt = deterministic.check_prompt_compliance({}, "")
    assert no_prompt["passed"] is None

    with_tool = deterministic.check_prompt_compliance(
        {"root_span": {"span_type": "tool", "name": "hybrid_search"}},
        "规则",
    )
    assert with_tool["passed"] is True

    no_plan = deterministic.check_prompt_compliance({"root_span": {"span_type": "agent"}}, "规则")
    assert no_plan["passed"] is None

    skipped = deterministic.check_prompt_compliance(
        {
            "root_span": {"span_type": "agent"},
            "trace_events": [{"event": "plan", "data": {"execution_plan": "tool_skip_reason: 不需要工具"}}],
        },
        "规则",
    )
    assert skipped["passed"] is True

    violation = deterministic.check_prompt_compliance(
        {
            "root_span": {"span_type": "agent"},
            "trace_events": [{"event": "plan", "data": {"execution_plan": "正常计划"}}],
        },
        "规则",
    )
    assert violation["passed"] is False
    assert violation["violations"]
