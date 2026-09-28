# -*- coding: utf-8 -*-
"""最小 smoke test：让 pytest 有真实可执行的确定性测试。

只测纯函数/纯数据结构，不启动 FastAPI、不访问网络、不读原项目。
"""
from evaluator.contract import AgentResult, AgentTimings
from evaluator.scorers.keywords import normalize_keyword
from app.services.coverage_gate import missing_orders
from app.services.path_guard import is_allowed_project_path


def test_agent_result_roundtrip():
    result = AgentResult(
        answer="测试答案",
        contexts="测试上下文",
        tool_trace=[{"name": "hybrid_search", "status": "success"}],
        timings=AgentTimings(init_seconds=1.5, agent_seconds=2.5),
        adapter="smoke",
    )
    data = result.to_dict()
    restored = AgentResult.from_dict(data)
    assert restored.answer == "测试答案"
    assert restored.tool_trace[0]["name"] == "hybrid_search"
    assert restored.timings.init_seconds == 1.5


def test_normalize_keyword_literal_and_semantic():
    text, match, terms = normalize_keyword("钟离")
    assert text == "钟离"
    assert match == "literal"
    assert terms == ["钟离"]

    text, match, terms = normalize_keyword("未收录")
    assert text == "未收录"
    assert match == "semantic"
    assert terms == ["未收录"]

    text, match, terms = normalize_keyword({"text": "愿景", "match": "semantic", "aliases": ["宏愿"]})
    assert text == "愿景"
    assert match == "semantic"
    assert terms == ["愿景", "宏愿"]


def test_coverage_gate_finds_missing_orders():
    orders = [{"id": "LO-001"}, {"id": "LO-002"}]
    evidence = {"LO-001": [{"ok": True, "summary": "done"}]}
    missing = missing_orders(orders, evidence)
    assert [o["id"] for o in missing] == ["LO-002"]


def test_path_guard_allow_and_deny():
    assert is_allowed_project_path(
        "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用"
    )
    assert not is_allowed_project_path("C:/Windows")
    assert not is_allowed_project_path("")


def test_diagnostic_pipeline_stage_registry():
    from app.services.diagnostic_pipeline import _STAGE_HANDLERS
    assert set(_STAGE_HANDLERS) == {
        "stage_input",
        "stage_routing",
        "stage_planning",
        "stage_tool_execution",
        "stage_knowledge_truth",
        "stage_answer",
        "stage_version",
        "stage_evaluator",
    }


def test_diagnostic_pipeline_unknown_stage_is_evidence_error():
    from app.services.diagnostic_pipeline import _run_stage
    out = _run_stage({"id": "LO-X", "category": "no_such_stage"}, {})
    assert out["ok"] is True
    assert out["status"] == "completed"
    assert "error" in out["data"]


def test_lab_check_handler_registry():
    from app.services.doctor_tools import _LAB_CHECK_HANDLERS
    assert set(_LAB_CHECK_HANDLERS) == {
        "trace_replay",
        "plan_intent",
        "trace_truth_audit",
        "prompt_rule",
        "missing_keyword",
        "forbidden_keyword",
        "not_found_tool",
        "prompt_violation",
        "zero_tool",
        "answer_integrity",
        "generic_failure",
    }


def test_lab_check_unknown_order_category():
    from app.services.doctor_tools import run_lab_check
    out = run_lab_check("LO-NOPE", {"lab_orders": [{"id": "LO-NOPE", "category": "no_such_category"}]})
    assert out["ok"] is False
    assert "未支持的检查类别" in out["summary"]


def test_resolve_cause_rule_registry():
    from app.services.diagnostic_pipeline import _RESOLVE_RULES, _rule_other
    assert len(_RESOLVE_RULES) == 9
    state = {"chain": [], "missing_kws": [], "forbidden_kws": []}
    result = _rule_other({}, {}, state)
    assert result["conclusion_kind"] == "other"


def test_fallback_default_prescription():
    from app.services.project_doctor import _fallback_default_prescription
    prescriptions = _fallback_default_prescription({})
    assert len(prescriptions) == 1
    assert prescriptions[0]["evidence_ids"] == []
    assert prescriptions[0]["evidence_level"] == "L1"
