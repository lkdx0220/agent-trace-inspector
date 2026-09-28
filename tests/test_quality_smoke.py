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


def test_first_str_arg_and_knowledge_query_terms():
    from app.services.diagnostic_pipeline import _first_str_arg, _knowledge_query_terms
    assert _first_str_arg({"a": 1, "b": " 散兵 "}) == "散兵"
    missing, forbidden, queries, nf_terms, char_terms = _knowledge_query_terms(
        {"must_contain": ["童话", "备份"], "must_not_contain": ["黑暗"]},
        "散兵为什么没被删记忆？",
        [{"name": "query_character", "args": {"name": "散兵"}}],
    )
    assert missing == ["童话", "备份"]
    assert forbidden == ["黑暗"]
    assert "散兵为什么没被删记忆？ 童话" in queries
    assert "散兵" in nf_terms
    assert char_terms == ["散兵"]


def test_knowledge_summary_empty():
    from app.services.diagnostic_pipeline import _knowledge_summary
    summary = _knowledge_summary({}, {})
    assert "无缺失词" in summary
    assert "无禁词" in summary


def test_audit_summary_plan_mismatch_and_short_circuit():
    from app.services.doctor_tools import _audit_summary
    summary = _audit_summary(
        {"plan_intents": ["hybrid_search"]},
        ["评测器未报零工具违规"],
        [{"name": "query_character"}],
        "当前知识库未收录",
        [],
    )
    assert "plan 文本规划调用" in summary
    assert "评测器一致性差异" in summary
    assert "not_found 工具" in summary
    assert "短路串" in summary


def test_fact_sheet_routing_applier():
    from app.services.diagnostic_pipeline import _empty_fact_sheet, _apply_fact_routing
    sheet = _empty_fact_sheet("答案", None, {"actual_tools": ["hybrid_search"]})
    assert sheet["actual_tools"] == ["hybrid_search"]
    assert sheet["routing_event_seen"] is False
    _apply_fact_routing(sheet, {
        "route_event_seen": True,
        "current_code_hard_rule_hit": True,
        "current_code_required_tools": ["query_quest"],
        "injected_tools": ["hybrid_search"],
        "missing_required_tools": ["query_quest"],
    })
    assert sheet["routing_event_seen"] is True
    assert sheet["routing_missing_required_tools"] == ["query_quest"]


def test_doctor_tools_definition():
    from app.services.project_doctor import _doctor_tools
    names = {t["function"]["name"] for t in _doctor_tools()}
    assert "run_lab_check" not in names
    assert {"record_verified_claim", "pin_fact"} <= names


def test_doctor_autofill_feedback_no_missing():
    from app.services.project_doctor import _doctor_autofill_feedback
    assert _doctor_autofill_feedback({}, [], {}) is None


def test_lint_question_meta_rejects_bad_match_mode():
    from evaluator.lint import _lint_question_meta
    errors, warnings, info = [], [], []
    _lint_question_meta({"match_mode": "bad", "must_contain": [], "must_not_contain": []}, "Q1", "questions[0]", errors, warnings, info)
    assert any("match_mode 非法" in e for e in errors)


def test_lint_keyword_item_semantic_skips_zero_hit():
    from collections import Counter
    from evaluator.lint import _lint_keyword_item
    errors, warnings, zero_hit, semantic_skipped, unclassified = [], [], [], [], []
    stats = Counter()
    _lint_keyword_item(
        "Q1", "questions[0]", "must_contain", {"text": "未收录", "match": "semantic"}, "",
        [{"answer": "没有", "contexts": ""}], errors, warnings, stats,
        unclassified, zero_hit, semantic_skipped,
    )
    assert stats["semantic"] == 1
    assert semantic_skipped == ["Q1:未收录"]
    assert zero_hit == []


def test_report_escape_helpers():
    from evaluator.report import _esc_br, _dot, _color_bar
    assert "<br>" in _esc_br("a" + chr(10) + "b")
    assert "&lt;" in _esc_br("<")
    assert "dot-ok" in _dot(True)
    assert "dot-fail" in _dot(False)
    assert "#5cb878" in _color_bar(5)
    assert "?" in _color_bar(None)


def test_build_case_keyword_state():
    from evaluator.db_bridge import _case_keyword_state
    _, _, keyword_pass, reasons = _case_keyword_state({
        "must_contain_result": {"passed": False, "miss": ["童话"]},
        "must_not_contain_result": {"passed": True, "violations": []},
    })
    assert keyword_pass is False
    assert any("缺少必须包含：童话" in r for r in reasons)


def test_build_check_results_no_trace_defaults():
    from evaluator.db_bridge import _build_check_results
    tools_check, route_check, prompt_check = _build_check_results({}, {}, None, "")
    assert tools_check["passed"] is None
    assert route_check["passed"] is None
    assert prompt_check["evidence"] == "无 Trace"
