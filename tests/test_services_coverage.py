# -*- coding: utf-8 -*-
"""服务层确定性测试：case_audit / report_generator / project_doctor / diagnostic_pipeline / eval_store。

原则：不访问真实 LLM、不访问真实知识库、不写真实 DB；需要项目路径的地方用 monkeypatch。
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db
from app import main as app_main
from app.services import (
    case_audit,
    diagnostic_pipeline,
    eval_store,
    project_doctor,
    report_generator,
    subprocess_env,
    system_prompts,
)
from evaluator import contract
from schemas.eval import AnswerVariant, RunCaseResult, RunRecord
from schemas.eval import TestCase as EvalTestCase


@pytest.fixture
def workdir():
    base = (Path("quality/reports/test_work") / uuid.uuid4().hex).resolve()
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


# ============================================================
# case_audit
# ============================================================


def test_case_audit_helpers():
    assert case_audit._ngrams("abcdef", 4) == {"abcd", "bcde", "cdef"}
    det = case_audit._deterministic_audit("答案提到钟离", "钟离 岩王帝君", ["钟离"], [])
    assert det["keyword_rate"] == 1.0
    assert det["verdict"] in {"strong", "medium", "weak", "suspicious"}

    no_ref = case_audit._deterministic_audit("x", "", [], [])
    assert no_ref["verdict"] == "no_reference"
    short = case_audit._deterministic_audit("当前知识库未收录。", "参考答案", [], [])
    assert short["verdict"] == "suspicious"

    merged = case_audit._merge(det, {"verdict": "weak", "score": 0.2, "reason": "r"})
    assert merged["verdict"] == "weak"
    merged2 = case_audit._merge(
        {"verdict": "weak", "score": 0.1, "short_circuit": "当前知识库未收录"},
        {"verdict": "strong", "score": 0.9, "reason": "r"},
    )
    assert merged2["verdict"] == "suspicious"
    assert case_audit._merge(det, None) == det


# ============================================================
# report_generator
# ============================================================


def test_report_generator_tool_texts_and_rows():
    trace = {
        "root_span": {
            "span_type": "agent",
            "children": [
                {"span_type": "tool", "result_preview": "A"},
                {"span_type": "tool", "result_preview": "B"},
            ],
        }
    }
    assert report_generator._all_tool_texts(trace) == "A\nB"

    missing_rows, bad_rows = report_generator._build_report_rows(
        ["x"],
        ["y"],
        "x in tools",
        {"must_contain": {"x": "why"}, "must_not_contain": {"y": "no"}},
    )
    assert missing_rows[0]["in_tool_output"] == "是"
    assert missing_rows[0]["why_required"] == "why"
    assert bad_rows[0]["in_tool_output"] == "否"
    assert bad_rows[0]["why_forbidden"] == "no"


def test_report_generator_keyword_evidence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        report_generator,
        "evaluate_keywords",
        lambda *_args, **_kwargs: {
            "miss": ["x"],
            "violations": ["y"],
            "variant_results": [{"name": "主", "passed": False, "hit_rate": 0.5, "miss": ["x"], "violations": ["y"]}],
        },
    )
    ctx = {
        "answer": "a",
        "case_id": "C1",
        "result": {"question": "q"},
        "ref_answer": "r",
        "must_contain": ["x"],
        "must_not_contain": ["y"],
        "case": {"match_mode": "all"},
        "alternatives": [],
        "project_path": "p",
    }
    kw_result, miss, violations, variants = report_generator._build_keyword_evidence(ctx)
    assert kw_result["miss"] == ["x"]
    assert miss == ["x"]
    assert violations == ["y"]
    assert variants[0]["name"] == "主"


# ============================================================
# project_doctor 纯 helper
# ============================================================


def test_project_doctor_extract_json():
    assert project_doctor._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert project_doctor._extract_json('text {"a": 2} tail') == {"a": 2}
    assert project_doctor._extract_json("not json") is None


def test_project_doctor_memory_and_tool_content():
    assert project_doctor._format_memory_block({}) == ""
    block = project_doctor._format_memory_block(
        {
            "verified_claims": [{"id": "C1", "claim": "x", "evidence_ids": ["LO-1"]}],
            "pinned_facts": [{"id": "P1", "text": "y", "evidence_id": "LO-1"}],
        }
    )
    assert "C1" in block
    assert "P1" in block

    small = project_doctor._tool_content({"a": 1}, max_chars=100)
    assert '"a": 1' in small
    big = project_doctor._tool_content({"summary": "s", "data": "x" * 500}, max_chars=100)
    assert "high-signal preview" in big


def test_project_doctor_meaningful_terms_and_corpus():
    terms = project_doctor._meaningful_terms("钟离岩王帝君")
    assert "钟离" in terms
    corpus = project_doctor._evidence_corpus_for_ids(
        ["LO-1"],
        {"LO-1": [{"ok": True, "summary": "证据"}]},
        [{"id": "EXT-1", "ok": True, "result": {"x": 1}}],
    )
    assert "证据" in corpus


def test_project_doctor_negative_and_probe_helpers():
    assert project_doctor._has_negative_conclusion({"diagnosis": {"summary": "知识库真缺"}}) is True
    assert project_doctor._has_negative_conclusion({"diagnosis": {"summary": "回答阶段未整合"}}) is False
    assert project_doctor._has_search_probe({"LO-1": [{"ok": True, "category": "missing_keyword"}]}, []) is True
    assert project_doctor._has_search_probe({}, []) is False
    assert (
        project_doctor._has_kb_gap_probe(
            {
                "LO-1": [
                    {
                        "ok": True,
                        "category": "missing_keyword",
                        "data": {"keyword": "x", "kb_probe": {"ok": True, "data": {"queries": {"q": "nothing"}}}},
                    }
                ]
            },
            [],
        )
        is True
    )
    assert (
        project_doctor._evidence_says_keyword_exists(
            {"LO-1": [{"ok": True, "category": "missing_keyword", "data": {"where": {"tool_results": True}}}]}
        )
        is True
    )


# ============================================================
# diagnostic_pipeline stages
# ============================================================


def test_diagnostic_generate_orders_and_input_stage():
    orders = diagnostic_pipeline.generate_pipeline_orders({}, {}, None)
    assert len(orders) == 8
    assert orders[0]["id"].startswith("LO-STG-")

    summary, data = diagnostic_pipeline._stage_input(
        {
            "result": {"answer": "abc", "question": "q", "passed": False, "reasons": ["r"]},
            "case": {"category": "c", "difficulty": "d", "must_contain": ["x"]},
        }
    )
    assert "答案长度 3" in summary
    assert data["answer_chars"] == 3
    assert data["must_contain"] == ["x"]


def test_diagnostic_routing_stage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        diagnostic_pipeline,
        "routing_probe",
        lambda _project, _question: {
            "ok": True,
            "data": {"hard_rule_hit": True, "required_tools": ["query_quest"]},
        },
    )
    trace = {
        "trace_events": [{"event": "route", "data": {"intent_labels": ["B"], "injected_tools": []}}],
    }
    summary, data = diagnostic_pipeline._stage_routing(
        {"trace": trace, "result": {"question": "q"}, "project_path": "p"}
    )
    assert data["route_event_seen"] is True
    assert data["missing_required_tools"] == ["query_quest"]
    assert "查询" not in summary


def test_diagnostic_planning_and_tool_stage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(diagnostic_pipeline, "get_plan_system_prompt", lambda _p: "plan-prompt")
    monkeypatch.setattr(diagnostic_pipeline, "get_answer_system_prompt", lambda _p: "answer-prompt")
    trace = {
        "trace_events": [
            {"event": "plan", "data": {"execution_plan": "调用 hybrid_search", "tool_call_names": []}},
        ],
        "root_span": {
            "span_type": "agent",
            "children": [
                {"span_type": "tool", "name": "hybrid_search", "status": "success", "result_preview": "hit"},
            ],
        },
    }
    summary, data = diagnostic_pipeline._stage_planning({"trace": trace, "project_path": "p"})
    assert "hybrid_search" in data["plan_intents"]
    assert data["plan_prompt_head"] == "plan-prompt"

    summary2, data2 = diagnostic_pipeline._stage_tool_execution({"trace": trace})
    assert data2["tool_count"] == 1
    assert data2["tool_names"] == ["hybrid_search"]


def test_diagnostic_version_and_evaluator_stage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        diagnostic_pipeline,
        "trace_snapshot_status",
        lambda _trace, _project: {"trace_snapshot_known": True, "prompt_clean": True, "code_clean": True},
    )
    summary, data = diagnostic_pipeline._stage_version({"trace": {}, "project_path": "p"})
    assert data["trace_snapshot_known"] is True
    assert "known=True" in summary

    trace = {
        "root_span": {"span_type": "agent", "children": []},
        "trace_events": [],
    }
    summary2, data2 = diagnostic_pipeline._stage_evaluator(
        {"trace": trace, "result": {"passed": False, "reasons": ["缺少必须包含：x"]}, "case": {"must_contain": ["x"]}}
    )
    assert "audit" in data2
    assert data2["reasons"] == ["缺少必须包含：x"]


# ============================================================
# eval_store（临时 DB）
# ============================================================


def test_eval_store_roundtrip(monkeypatch: pytest.MonkeyPatch, workdir: Path):
    monkeypatch.setattr(db, "DEFAULT_DB", workdir / "eval.db")
    monkeypatch.setattr(eval_store, "GOLDEN_SET_PATH", workdir / "golden.json")

    case = EvalTestCase(
        case_id="C1",
        question="q",
        expected_answer="a",
        must_contain=["x"],
        must_not_contain=["y"],
        match_mode="all",
        alternatives=[AnswerVariant(name="v1", must_contain=["z"])],
    )
    eval_store.save_test_case(case)
    stored_case = eval_store.get_test_case("C1")
    assert stored_case is not None
    assert stored_case["question"] == "q"
    assert eval_store.list_test_cases()[0]["case_id"] == "C1"

    run = RunRecord(
        run_id="R1",
        name="test",
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        results=[RunCaseResult(case_id="C1", question="q", passed=True)],
    )
    eval_store.save_run(run)
    stored_run = eval_store.get_run("R1")
    assert stored_run is not None
    assert stored_run["name"] == "test"
    assert eval_store.list_runs()[0]["run_id"] == "R1"

    eval_store.save_diagnosis(
        "R1",
        "C1",
        "t1",
        {"root_cause": "rc", "evidence": [{"x": 1}], "suggestion": "s", "confidence": 0.5},
        "prompt",
    )
    diag = eval_store.get_diagnosis("R1", "C1")
    assert diag is not None
    assert diag["root_cause"] == "rc"
    assert diag["evidence"] == [{"x": 1}]

    eval_store.save_report("R1", "C1", "report")
    assert eval_store.get_report("R1", "C1") == "report"

    eval_store.save_prescription("R1", "C1", {"a": 1}, model="m")
    prescription = eval_store.get_prescription("R1", "C1")
    assert prescription is not None
    assert prescription["payload"] == {"a": 1}
    assert prescription["model"] == "m"

    eval_store.save_case_audit(
        {
            "run_id": "R1",
            "case_id": "C1",
            "question": "q",
            "passed": True,
            "verdict": "strong",
            "score": 0.9,
            "details": {"k": "v"},
        }
    )
    audit = eval_store.get_case_audit("R1", "C1")
    assert audit is not None
    assert audit["passed"] is True
    assert audit["details"] == {"k": "v"}
    assert eval_store.list_case_audits("R1")[0]["verdict"] == "strong"


# ============================================================
# system_prompts
# ============================================================


def test_system_prompts_read_and_missing(workdir: Path, monkeypatch):
    from app.services import path_guard

    project = workdir / "proj"
    prompt_dir = project / "prompts" / "system"
    prompt_dir.mkdir(parents=True)
    monkeypatch.setattr(path_guard, "ALLOWED_PROJECT_PATHS", [project.resolve()])
    (prompt_dir / "agent_system_v4_plan.txt").write_text(
        "head\n===== 不调工具的前置检查 =====\n必须调用工具\n===== 工具选择策略 =====\ntail",
        encoding="utf-8",
    )
    (prompt_dir / "agent_system_v4_answer.txt").write_text("answer", encoding="utf-8")

    assert "必须调用工具" in system_prompts.get_plan_system_prompt(str(project))
    assert system_prompts.get_answer_system_prompt(str(project)) == "answer"
    excerpt = system_prompts.get_tool_requirement_excerpt(str(project))
    assert "必须调用工具" in excerpt
    assert system_prompts.get_plan_system_prompt(str(workdir / "missing")) == ""


# ============================================================
# subprocess_env
# ============================================================


def test_build_child_env_sanitizes_kb(monkeypatch: pytest.MonkeyPatch, workdir: Path):
    project = workdir / "proj"
    (project / "kb_vectors_m3").mkdir(parents=True)
    (project / "kb_vectors").mkdir()
    monkeypatch.setattr(subprocess_env, "ensure_project_path", lambda _p: project)

    monkeypatch.setenv("KB_VECTOR_DIR", str(project / "kb_vectors_backup"))
    monkeypatch.setenv("KB_EMBEDDING_BACKEND", "bge-m3")
    env = subprocess_env.build_child_env(str(project))
    assert "KB_VECTOR_DIR" not in env
    assert "KB_EMBEDDING_BACKEND" not in env

    monkeypatch.setenv("KB_VECTOR_DIR", str(project / "kb_vectors"))
    monkeypatch.setenv("KB_EMBEDDING_BACKEND", "bge-m3")
    env = subprocess_env.build_child_env(str(project))
    assert env["KB_VECTOR_DIR"].endswith("kb_vectors")
    assert env["KB_EMBEDDING_BACKEND"] == "text-embedding-v4"


# ============================================================
# app.main / contract / exporter 纯 helper
# ============================================================


def test_app_status_and_root(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_main, "init_db", lambda: None)
    client = TestClient(app_main.app)
    assert client.get("/api/status").json()["status"] == "ok"
    assert client.get("/").status_code == 200


def test_contract_read_write_roundtrip(workdir: Path):
    result = contract.AgentResult(answer="ok", contexts="ctx")
    path = workdir / "result.json"
    contract.write_result(result, path)
    loaded = contract.read_result(path)
    assert loaded.answer == "ok"
    assert loaded.contexts == "ctx"


def test_exporter_trace_header_and_tool_calls():
    from exporter import genshin_exporter

    started = datetime.now()
    trace_id, response_mode, root = genshin_exporter._build_trace_header(
        {"final_response": "当前知识库未收录。"}, "q", started
    )
    assert trace_id.startswith("trace_")
    assert response_mode == "not_found"
    assert root.span_type.value == "agent"

    msg = SimpleNamespace(tool_calls=[{"id": "call_1", "name": "hybrid_search", "args": {"query": "x"}}])
    calls = genshin_exporter._tool_calls_from_ai(msg)  # type: ignore[arg-type]
    assert calls[0].name == "hybrid_search"
    assert calls[0].tool_call_id == "call_1"
