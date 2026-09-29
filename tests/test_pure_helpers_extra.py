# -*- coding: utf-8 -*-
"""纯函数/确定性工具测试：路径守卫、项目地图、manifest、config、DB、doctor 证据工具。

原则：不启动 FastAPI、不访问网络、不调用真实 LLM、不读原项目真实知识库。
所有需要外部路径的测试都用 workdir / monkeypatch。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest
from fastapi import HTTPException

from app.db import get_conn, get_timeline, get_trace, init_db, list_traces, save_trace
from app.services import auth, doctor_tools, project_map
from evaluator import config as evaluator_config
from evaluator import harness, manifest
from evaluator.contract import AgentResult, AgentTimings
from exporter import genshin_exporter, safe_paths
from schemas.trace import Span, SpanStatus, SpanType, Trace, TraceMetadata


@pytest.fixture
def workdir():
    """不用 pytest 的 tmp_path：本机/沙箱下系统临时目录清理会触发 WinError 5。

    统一在 quality/reports/test_work 下建独立目录，测试结束尽力清理。
    """
    base = (Path("quality/reports/test_work") / uuid.uuid4().hex).resolve()
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


# ============================================================
# auth
# ============================================================


class _FakeClient:
    def __init__(self, host: str):
        self.host = host


class _FakeRequest:
    def __init__(self, host: str):
        self.client = _FakeClient(host)


def _request(host: str) -> Any:
    return cast(Any, _FakeRequest(host))


def test_require_local_or_token_allows_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INSPECTOR_API_TOKEN", raising=False)
    auth.require_local_or_token(_request("127.0.0.1"), authorization=None, x_api_token=None)


def test_require_local_or_token_rejects_remote(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INSPECTOR_API_TOKEN", raising=False)
    with pytest.raises(HTTPException) as exc:
        auth.require_local_or_token(_request("10.0.0.9"), authorization=None, x_api_token=None)
    assert exc.value.status_code == 403


def test_require_local_or_token_accepts_bearer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSPECTOR_API_TOKEN", "s3cret")
    auth.require_local_or_token(
        _request("10.0.0.9"),
        authorization="Bearer s3cret",
        x_api_token=None,
    )
    with pytest.raises(HTTPException) as exc:
        auth.require_local_or_token(
            _request("10.0.0.9"),
            authorization="Bearer wrong",
            x_api_token=None,
        )
    assert exc.value.status_code == 401


# ============================================================
# exporter.safe_paths
# ============================================================


def test_safe_paths_project_root_guard(workdir):
    assert safe_paths.ensure_project_path_local(safe_paths.ALLOWED_PROJECT_ROOT) == Path(
        safe_paths.ALLOWED_PROJECT_ROOT
    )
    with pytest.raises(ValueError):
        safe_paths.ensure_project_path_local(workdir)


def test_safe_paths_trace_and_cases_guards(workdir):
    inside_trace = safe_paths.ALLOWED_TRACE_ROOT / "demo.json"
    assert safe_paths.ensure_trace_out_path(inside_trace) == inside_trace.resolve()
    with pytest.raises(ValueError):
        safe_paths.ensure_trace_out_path(workdir / "evil.json")

    inside_cases = safe_paths.ALLOWED_CASES_ROOT / "cases.json"
    assert safe_paths.ensure_cases_file_path(inside_cases) == inside_cases.resolve()
    with pytest.raises(ValueError):
        safe_paths.ensure_cases_file_path(workdir / "cases.json")


def test_safe_paths_redact_sensitive():
    text = 'key=sk-abcdefghijklmnop token="Bearer abcdefghijklmnop"'
    redacted = safe_paths.redact_sensitive(text)
    assert "sk-abcdefghijklmnop" not in redacted
    assert "Bearer abcdefghijklmnop" not in redacted
    assert "[REDACTED]" in redacted


# ============================================================
# project_map
# ============================================================


def test_project_map_iter_symbols_handles_syntax_error():
    symbols = project_map._iter_code_symbols("def broken(:\n")
    assert symbols == []


def test_project_map_iter_symbols_extracts_function_and_class():
    source = (
        "def foo():\n"
        '    """foo doc"""\n'
        "    return 1\n\n"
        "class Bar:\n"
        '    """bar doc"""\n'
        "    def method(self):\n"
        '        """method doc"""\n'
        "        return 2\n"
    )
    symbols = project_map._iter_code_symbols(source)
    assert [s["type"] for s in symbols] == ["function", "class"]
    assert symbols[0]["name"] == "foo"
    assert symbols[0]["doc"] == "foo doc"
    assert symbols[1]["methods"][0]["name"] == "method"


def test_project_map_generate_and_format(workdir, monkeypatch: pytest.MonkeyPatch):
    (workdir / "a.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(project_map, "ensure_project_path", lambda _: workdir)
    monkeypatch.setattr(project_map, "MAP_FILES", {"a.py": "demo role"})
    data = project_map.generate_project_map(str(workdir))
    assert data["files"][0]["exists"] is True
    assert data["files"][0]["symbols"][0]["name"] == "hello"

    payload = {
        "project_path": "X",
        "files": [
            {
                "path": "a.py",
                "role": "r",
                "exists": True,
                "symbols": [{"type": "function", "name": "f", "line": 1, "doc": ""}],
            },
            {"path": "p.txt", "role": "prompt", "exists": True, "symbols": []},
        ],
    }
    text = project_map.format_project_map(payload, max_chars=500)
    assert "a.py" in text
    assert "p.txt" in text
    assert "截断" in project_map.format_project_map(payload, max_chars=5)


# ============================================================
# evaluator.config
# ============================================================


def test_evaluator_config_value_and_deep_merge():
    assert evaluator_config.config_value({"a": {"b": 1}}, "a.b") == 1
    assert evaluator_config.config_value({"a": {}}, "a.b", "fallback") == "fallback"
    merged = evaluator_config._deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}


def test_evaluator_load_config_merges_defaults(workdir):
    path = workdir / "config.yaml"
    path.write_text("harness:\n  timeout_seconds: 123\njudge:\n  model: test-model\n", encoding="utf-8")
    config = evaluator_config.load_config(str(path))
    assert config["harness"]["timeout_seconds"] == 123
    assert config["harness"]["concurrency"] == 4
    assert config["judge"]["model"] == "test-model"
    assert config["judge"]["params"]["max_tokens"] == 512


def test_evaluator_load_config_rejects_non_mapping(workdir):
    path = workdir / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        evaluator_config.load_config(str(path))


# ============================================================
# evaluator.manifest
# ============================================================


def test_manifest_hash_helpers(workdir):
    assert manifest.sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert manifest.sha256_file(workdir / "missing.bin") is None
    file_path = workdir / "data.bin"
    file_path.write_bytes(b"abc")
    assert manifest.sha256_file(file_path) == manifest.sha256_text("abc")


def test_manifest_git_and_kb_metadata(workdir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manifest, "_git_commit", lambda _cwd: "")
    monkeypatch.setattr(manifest, "_git_status_porcelain", lambda _cwd: "")
    git = manifest.git_metadata(workdir)
    assert git["commit"] is None
    assert git["dirty"] is False
    assert manifest.kb_manifest_hash(workdir) is None

    kb_dir = workdir / "kb_vectors_m3"
    kb_dir.mkdir()
    (kb_dir / "manifest.json").write_text('{"snapshot_hash": "abc"}', encoding="utf-8")
    assert manifest.kb_manifest_hash(workdir) == "abc"


def test_manifest_roundtrip_and_case_status(workdir):
    item = manifest.RunManifest(run_id="r1", adapter="genshin", case_count=2)
    item.set_case_status("A", "cached")
    path = workdir / "manifest.json"
    item.write(path)
    loaded = manifest.RunManifest.read(path)
    assert loaded.run_id == "r1"
    assert loaded.status_per_case == {"A": "cached"}


def test_build_manifest_captures_files(workdir, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manifest, "_git_commit", lambda _cwd: "")
    monkeypatch.setattr(manifest, "_git_status_porcelain", lambda _cwd: "")
    evalset = workdir / "cases.json"
    evalset.write_text('{"questions": []}', encoding="utf-8")
    item = manifest.build_manifest(
        run_id="r2",
        adapter="genshin",
        project_path=workdir,
        evalset_path=evalset,
        case_count=1,
        judge_model="deepseek-chat",
        notes="demo",
    )
    assert item.run_id == "r2"
    assert item.evalset_sha256 == manifest.sha256_file(evalset)
    assert item.config_sha256 is None


# ============================================================
# evaluator.harness
# ============================================================


def test_harness_safe_qid_and_digest():
    assert harness._safe_qid("a/b c", 1) == "a_b_c"
    assert harness._safe_qid("", 7) == "case7"
    result = {"id": "x", "answer": "a", "ragas": {}}
    digest = harness.content_digest(result)
    assert digest == harness.content_digest(dict(result))
    changed = dict(result, answer="b")
    assert digest != harness.content_digest(changed)


def test_harness_validate_cached_result(workdir):
    case = {"id": "x", "question": "q"}
    cached: Dict[str, Any] = {
        "schema_version": harness.RESULT_SCHEMA_VERSION,
        "id": "x",
        "question": "q",
        "category": "c",
        "answer": "a",
        "ragas": {},
        "must_contain_result": {},
        "must_not_contain_result": {},
        "citation_result": {},
        "elapsed": 1.0,
        "provenance": {"question_sha256": harness.case_sha256(case)},
    }
    cached["content_digest"] = harness.content_digest(cached)
    assert harness._validate_cached_result(cached, case) == (True, "")
    assert harness._validate_cached_result(cached, {"id": "x", "question": "changed"})[0] is False
    assert harness._validate_cached_result({"id": "x"}, case)[0] is False


def test_harness_aggregate_repeat_rows_majority():
    rows = [
        {
            "keyword_passed": True,
            "ragas": {"faithfulness": 5},
            "agent_seconds": 1.0,
            "init_seconds": 0.5,
            "elapsed": 2.0,
        },
        {
            "keyword_passed": True,
            "ragas": {"faithfulness": 3},
            "agent_seconds": 2.0,
            "init_seconds": 0.5,
            "elapsed": 3.0,
        },
        {
            "keyword_passed": False,
            "ragas": {"faithfulness": 1},
            "agent_seconds": 3.0,
            "init_seconds": 0.5,
            "elapsed": 4.0,
        },
    ]
    out = harness.aggregate_repeat_rows({"stability": "noisy"}, rows)
    assert out["keyword_votes"] == "2/3"
    assert out["keyword_passed"] is True
    assert out["stability"] == "noisy"
    assert out["ragas"]["faithfulness"] == 3
    assert harness.aggregate_repeat_rows({}, []) == {}


# ============================================================
# evaluator.contract
# ============================================================


def test_contract_validate_and_roundtrip(workdir):
    result = AgentResult(answer="", status="bad-status", adapter="x")
    issues = result.validate()
    assert any("status 非法" in item for item in issues)
    assert any("warning" in item for item in issues)

    healthy = AgentResult(answer="ok", contexts="ctx", timings=AgentTimings(init_seconds=1.0))
    target = workdir / "result.json"
    from evaluator import contract

    contract.write_result(healthy, target)
    loaded = contract.read_result(target)
    assert loaded.answer == "ok"
    assert loaded.timings.init_seconds == 1.0


# ============================================================
# app.db
# ============================================================


def _sample_trace(trace_id: str = "trace_test_1") -> Trace:
    root = Span(
        span_id="span_agent",
        span_type=SpanType.AGENT,
        name="agent",
        status=SpanStatus.SUCCESS,
        children=[
            Span(
                span_id="span_tool",
                span_type=SpanType.TOOL,
                name="hybrid_search",
                status=SpanStatus.SUCCESS,
                step_index=1,
                result_preview="hello",
            )
        ],
    )
    return Trace(
        trace_id=trace_id,
        question="q",
        metadata=TraceMetadata(execution_mode="L2", intent_labels=["A"]),
        root_span=root,
        trace_events=[{"event": "tool_end", "timestamp": 1.0, "data": {"name": "hybrid_search"}}],
    )


def test_db_save_list_get_timeline(workdir):
    db_path = workdir / "inspector.db"
    init_db(db_path)
    conn = get_conn(db_path)
    conn.close()

    trace = _sample_trace()
    save_trace(trace, db_path)
    rows = list_traces(db_path)
    assert rows[0]["trace_id"] == trace.trace_id
    loaded = get_trace(trace.trace_id, db_path)
    assert loaded is not None
    assert loaded["trace_id"] == trace.trace_id
    assert get_trace("missing", db_path) is None
    timeline = get_timeline(trace.trace_id, db_path)
    assert [item["span_id"] for item in timeline] == ["span_agent", "span_tool"]


# ============================================================
# doctor_tools
# ============================================================


def test_doctor_short_circuit_and_kb_probe():
    assert doctor_tools._short_circuit_answer("当前知识库未收录。") == "当前知识库未收录"
    assert doctor_tools._short_circuit_answer("正常回答") is None
    assert doctor_tools.kb_probe_contains({"ok": True, "data": {"queries": {"q": "未找到角色"}}}, "钟离") is False
    probe = {"ok": True, "data": {"queries": {"q": "header\n钟离的资料在这里"}}}
    assert doctor_tools.kb_probe_contains(probe, "钟离") is True


def test_doctor_keyword_conclusion_branches():
    assert "回答阶段未整合" in doctor_tools.keyword_retrieval_conclusion(
        "x", {"tool_results": True, "final_answer": False}, {}
    )
    assert "工具返回和最终答案都包含" in doctor_tools.keyword_retrieval_conclusion(
        "x", {"tool_results": True, "final_answer": True}, {}
    )
    assert "不是知识库缺失" in doctor_tools.keyword_retrieval_conclusion(
        "钟离",
        {"tool_results": False, "final_answer": False},
        {"ok": True, "data": {"queries": {"q": "header\n钟离"}}},
    )
    assert "知识库确实缺" in doctor_tools.keyword_retrieval_conclusion(
        "x", {"tool_results": False, "final_answer": False}, {"ok": True, "data": {"queries": {"q": "nothing"}}}
    )


def test_doctor_trace_helpers_and_plan_signal():
    trace = {
        "root_span": {
            "span_type": "agent",
            "children": [
                {"span_type": "tool", "name": "hybrid_search", "result_full": "钟离的资料"},
            ],
        },
        "trace_events": [
            {
                "event": "plan",
                "data": {"execution_plan": "调用 hybrid_search 查询钟离", "tool_call_names": ["hybrid_search"]},
            },
            {"event": "llm_start", "data": {"role": "plan_retry"}},
        ],
    }
    spans = doctor_tools.collect_tool_spans(trace)
    assert len(spans) == 1
    where = doctor_tools._find_where("钟离", trace, "答案提到钟离")
    assert where["tool_results"] is True
    assert where["final_answer"] is True
    assert where["plan_text"] is True
    signal = doctor_tools._plan_signal(trace)
    assert signal["plan_events"] == 1
    assert signal["plan_retry_events"] == 1
    assert signal["tool_call_names"] == ["hybrid_search"]


def test_doctor_plan_intent_and_path_guards(workdir):
    intents = doctor_tools._extract_plan_tool_intents(
        ["应调用 hybrid_search", "查询 query_quest", 'query_character("散兵")']
    )
    assert "hybrid_search" in intents
    assert "query_quest" in intents
    assert "query_character" in intents

    root = workdir
    (root / "pkg").mkdir()
    assert doctor_tools._safe_relative(root, "pkg/a.py") == root / "pkg" / "a.py"
    assert doctor_tools._safe_relative(root, "../outside.py") is None
    assert doctor_tools._safe_relative(root, str(root / "abs.py")) is None
    assert doctor_tools._is_unsafe_rel_path(".env") is True
    assert doctor_tools._is_unsafe_rel_path("pkg/.env") is True
    assert doctor_tools._is_unsafe_rel_path("pkg/main.py") is False


# ============================================================
# genshin_exporter 纯 helper
# ============================================================


def test_exporter_status_and_tool_call_lookup():
    assert genshin_exporter._status_of_tool_result("[系统拦截] 禁止") == SpanStatus.INTERCEPTED
    assert genshin_exporter._status_of_tool_result("工具执行出错") == SpanStatus.ERROR
    assert genshin_exporter._status_of_tool_result("未找到角色") == SpanStatus.NOT_FOUND
    assert genshin_exporter._status_of_tool_result("正常返回") == SpanStatus.SUCCESS

    msg = SimpleNamespace(tool_calls=[{"id": "call_1", "name": "hybrid_search", "args": {"query": "x"}}])
    found = genshin_exporter._find_tool_call([msg], "call_1")  # type: ignore[list-item]
    assert found is not None
    assert found["name"] == "hybrid_search"
    assert genshin_exporter._find_tool_call([msg], "missing") is None  # type: ignore[list-item]
