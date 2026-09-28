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


def test_run_evalset_summarize_rows():
    from tools.run_evalset import _summarize_rows
    rows = [
        {"id": "Q1", "keyword_passed": True},
        {"id": "Q2", "keyword_passed": False},
    ]
    passed, failures = _summarize_rows(rows)
    assert passed == 1
    assert failures == ["Q2"]


def test_collect_reason_keywords():
    from app.services.lab_orders import _collect_reason_keywords
    reasons = ["缺少必须包含：童话", "缺少必须包含：备份", "出现禁止包含：黑暗"]
    assert _collect_reason_keywords(reasons, "缺少必须包含：") == ["童话", "备份"]
    assert _collect_reason_keywords(reasons, "出现禁止包含：") == ["黑暗"]


def test_gate_validate_kind():
    from app.services.project_doctor import _gate_validate_kind, CONSISTENCY_KINDS
    issues = []
    assert _gate_validate_kind({"conclusion_kind": "recall_failure"}, 0, issues) == "recall_failure"
    assert issues == []
    assert _gate_validate_kind({}, 0, issues) is None
    assert issues
    assert "recall_failure" in CONSISTENCY_KINDS


def test_metrics_collect_and_summarize_tools():
    from app.services.metrics import _collect_spans, _summarize_tools
    root = {"span_type": "agent", "children": [{"span_type": "tool", "name": "hybrid_search", "status": "success"}]}
    spans = _collect_spans(root)
    assert len(spans) == 2
    summary = _summarize_tools([s for s in spans if s.get("span_type") == "tool"])
    assert summary["tool_names"] == ["hybrid_search"]
    assert summary["tool_by_name"]["hybrid_search"]["count"] == 1


def test_build_trace_header():
    from datetime import datetime
    from exporter.genshin_exporter import _build_trace_header
    trace_id, response_mode, root = _build_trace_header({"final_response": "当前知识库未收录。"}, "Q", datetime.now())
    assert trace_id.startswith("trace_")
    assert response_mode == "not_found"
    assert root.span_type.value == "agent"


def test_adapter_module_and_payload():
    from evaluator.harness import _adapter_module, _build_adapter_payload
    assert _adapter_module("genshin") == "evaluator.adapters.genshin"
    assert _adapter_module("nope") is None
    payload = _build_adapter_payload({"id": "Q1", "question": "q"})
    assert payload["case_id"] == "Q1"
    assert payload["question"] == "q"


def test_build_initial_state():
    from evaluator.adapters.genshin import _build_initial_state
    state = _build_initial_state("Q", "ctx")
    assert state["user_query"] == "Q"
    assert state["conversation_history"] == [{"user": "ctx", "assistant": "（上轮回答略）"}]


def test_evaluate_must_contain_literal():
    from app.services.evaluator import _evaluate_must_contain
    result = _evaluate_must_contain("答案包含钟离", ["钟离"], "all", None)
    assert result["contains_ok"] is True
    assert result["hit_count"] == 1


def test_render_detail_card_error():
    from evaluator.report import _render_detail_card
    html = _render_detail_card({"id": "Q1", "category": "x", "error": "boom"})
    assert "boom" in html


def test_deterministic_trace_checks():
    from evaluator.scorers.deterministic import (
        check_expected_route, check_expected_tools, check_prompt_compliance, collect_tools_from_trace,
    )
    trace = {
        "metadata": {"execution_mode": "L2"},
        "root_span": {"span_type": "agent", "children": [{"span_type": "tool", "name": "hybrid_search"}]},
        "trace_events": [{"event": "plan", "data": {"execution_plan": "调用 hybrid_search"}}],
    }
    assert collect_tools_from_trace(trace) == ["hybrid_search"]
    assert check_expected_tools(trace, ["hybrid_search"])["passed"] is True
    assert check_expected_route(trace, "L2")["passed"] is True
    assert check_prompt_compliance(trace, "规则原文")["passed"] is True
    trace_zero = {
        "metadata": {},
        "root_span": {"span_type": "agent"},
        "trace_events": [{"event": "plan", "data": {"execution_plan": "不调用工具"}}],
    }
    assert check_prompt_compliance(trace_zero, "规则原文")["passed"] is False


def test_coverage_gate_validate_prescriptions():
    from app.services.coverage_gate import validate_prescriptions
    orders = [{"id": "LO-001"}]
    evidence = {"LO-001": [{"ok": True, "summary": "ok"}]}
    report = {
        "diagnosis": {"primary_root_cause": "x"},
        "prescriptions": [{"root_cause": "x", "evidence_ids": ["LO-001"]}],
    }
    assert validate_prescriptions(report, orders, evidence, [])["valid"] is True
    bad = {
        "diagnosis": {"primary_root_cause": "x"},
        "prescriptions": [{"root_cause": "x", "evidence_ids": ["EXT-999"]}],
    }
    assert validate_prescriptions(bad, orders, evidence, [])["valid"] is False


def test_compute_trace_metrics_synthetic():
    from datetime import datetime
    from app.services.metrics import compute_trace_metrics
    t0 = datetime.now().isoformat()
    t1 = datetime.now().isoformat()
    trace = {
        "trace_id": "t1",
        "question": "q",
        "duration_ms": 1000,
        "root_span": {
            "span_type": "agent",
            "start_time": t0,
            "end_time": t1,
            "children": [{"span_type": "tool", "name": "hybrid_search", "status": "success", "start_time": t0, "end_time": t1}],
        },
        "trace_events": [],
    }
    metrics = compute_trace_metrics(trace)
    assert metrics["tool_count"] == 1
    assert metrics["unique_tools"] == ["hybrid_search"]
    assert metrics["span_count"] == 2


def test_generate_lab_orders_basic():
    from app.services.lab_orders import generate_lab_orders
    result = {"reasons": ["缺少必须包含：童话"], "answer": "x", "prompt_pass": None}
    case = {"case_id": "R3", "must_contain": ["童话"]}
    trace = {"root_span": {"span_type": "agent", "children": [{"span_type": "tool", "name": "hybrid_search", "status": "success"}]}}
    orders = generate_lab_orders(result, case, trace, None)
    ids = [o["id"] for o in orders]
    assert "LO-001" in ids
    assert "LO-KW-01" in ids
    assert "LO-AI-01" in ids


def test_lint_cases_temp_file():
    import json
    from pathlib import Path
    from evaluator.lint import lint_cases
    path = Path("quality/reports/test_lint_cases_temp.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "metadata": {"project": "genshin", "total_questions": 1},
        "questions": [{
            "id": "Q1", "question": "q", "category": "c", "difficulty": "easy",
            "match_mode": "all", "must_contain": ["钟离"], "must_not_contain": [], "reference_answer": "钟离",
        }],
    }, ensure_ascii=False), encoding="utf-8")
    try:
        report = lint_cases(str(path))
        assert report["ok"] is True
        assert report["keyword_stats"]["unclassified"] == 1
    finally:
        path.unlink(missing_ok=True)


def test_render_summary_row():
    from evaluator.report import _render_summary_row
    row = {
        "id": "Q1", "category": "c", "question": "问题",
        "ragas": {"faithfulness": 5, "answer_relevancy": 4, "context_precision": 3, "context_recall": 2},
        "must_contain_result": {"passed": True, "hit_rate": 1.0, "miss": [], "semantic_hit": [], "hit": ["钟离"], "violations": []},
        "must_not_contain_result": {"passed": True, "violations": []},
        "citation_result": {"passed": True, "total": 0, "verified": 0, "unverified": []},
        "elapsed": 1.0,
    }
    html = _render_summary_row(row)
    assert "Q1" in html
    assert "问题" in html


def test_html_sanitizer_blocks_scripts():
    from app.services.html_sanitizer import sanitize_html
    raw = '<p onclick="x()">hi<script>alert(1)</script><a href="javascript:bad()">x</a><a href="https://example.com">ok</a></p>'
    out = sanitize_html(raw)
    assert "<script" not in out
    assert "onclick" not in out
    assert "javascript:" not in out
    assert "https://example.com" in out


def test_harness_median_helpers():
    from evaluator.harness import _median, _median_ragas
    assert _median([1, 3, 2]) == 2
    assert _median([]) is None
    med = _median_ragas([
        {"ragas": {"faithfulness": 4, "answer_relevancy": 2}},
        {"ragas": {"faithfulness": 2, "answer_relevancy": 4}},
    ])
    assert med["faithfulness"] == 3
    assert med["answer_relevancy"] == 3
