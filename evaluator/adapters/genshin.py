# -*- coding: utf-8 -*-
"""原神项目 adapter：子进程 + JSON 协议。

父进程通过 stdin 传入：
  {"case_id": "R7", "question": "...", "context": "...", "case": {...}}
子进程通过 stdout 返回 evaluator.contract.AgentResult JSON。

Phase 6 改动：
- 不再只返回 answer/tool_trace；
- 在进程内复用 exporter.build_trace_from_result 生成真实 Trace（含 trace_events），
  放进 raw.trace，供确定性检查（工具/路由/提示词合规）和 DB bridge 使用。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# evaluator/adapters/genshin.py -> adapters -> evaluator -> agent-trace-inspector -> 工作区
_DEFAULT_WORKSPACE = str(Path(__file__).resolve().parents[3])


def workspace_dir() -> str:
    return os.environ.get("GOLDEN_TEST_WORKSPACE") or _DEFAULT_WORKSPACE


def _case_dir() -> str:
    return os.path.join(workspace_dir(), "CASE-原神剧情助手-修改用")


def _trace_from_result(result: Dict[str, Any], question: str, started_at: datetime, case_dir: str, events: List[Dict[str, Any]]):
    """复用导出器的 Trace 构建逻辑，返回 Trace 或 None。"""
    try:
        from exporter.genshin_exporter import (
            _add_missing_llm_spans_from_events,
            _enrich_trace_with_events,
            _fill_assess_router_times,
            build_trace_from_result,
        )
    except Exception:
        return None
    try:
        trace = build_trace_from_result(result, question, started_at, project_path=Path(case_dir))
        if events:
            _enrich_trace_with_events(trace, events)
            _add_missing_llm_spans_from_events(trace, events)
            _fill_assess_router_times(trace, events)
        trace.trace_events = events
        return trace
    except Exception:
        return None


def _tool_payload_from_trace(trace: Any) -> List[Dict[str, Any]]:
    from evaluator.scorers.deterministic import walk_spans

    payload: List[Dict[str, Any]] = []
    for span in walk_spans(trace.root_span.model_dump(mode="json") if hasattr(trace.root_span, "model_dump") else trace.root_span):
        if span.get("span_type") != "tool":
            continue
        payload.append({
            "name": span.get("name") or "?",
            "status": span.get("status") or "success",
            "args": span.get("tool_args") or {},
            "result_preview": str(span.get("result_preview") or "")[:300],
            "result_length": span.get("result_length") or len(str(span.get("result_full") or "")),
            "result_full": str(span.get("result_full") or span.get("result_preview") or ""),
        })
    return payload


def _tool_payload_from_messages(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    from langchain_core.messages import ToolMessage

    payload: List[Dict[str, Any]] = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            content = msg.content if hasattr(msg, "content") else str(msg)
            name = msg.name if hasattr(msg, "name") else "unknown"
            if content and len(content) > 10:
                payload.append({
                    "name": name,
                    "status": "success",
                    "args": {},
                    "result_preview": str(content)[:300],
                    "result_length": len(str(content)),
                    "result_full": str(content),
                })
    return payload


def run_agent(question: str, context: str = "") -> Dict[str, Any]:
    """运行原项目 Agent，返回 AgentResult 兼容 dict。"""
    from evaluator.contract import AgentResult, AgentTimings, STATUS_AGENT_ERROR, STATUS_EMPTY, STATUS_OK

    case_dir = _case_dir()
    if not os.path.isdir(case_dir):
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"被测项目目录不存在: {case_dir}",
            adapter="genshin",
        ).to_dict()
    if case_dir not in sys.path:
        sys.path.insert(0, case_dir)

    t0 = time.time()
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from genshin_story_agent import create_agent_workflow
        try:
            import app.trace_recorder as tracer
        except Exception:
            tracer = None
    except Exception as exc:
        sys.stdout = old_stdout
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"import genshin_story_agent failed: {type(exc).__name__}: {exc}",
            adapter="genshin",
        ).to_dict()
    finally:
        sys.stdout = old_stdout

    try:
        agent = create_agent_workflow()
    except Exception as exc:
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"create_agent_workflow failed: {type(exc).__name__}: {exc}",
            adapter="genshin",
        ).to_dict()
    t_ready = time.time()

    conversation_history = []
    if context:
        conversation_history = [{"user": context, "assistant": "（上轮回答略）"}]

    state = {
        "user_query": question,
        "rewritten_query": None,
        "alias_notes": None,
        "conversation_history": conversation_history,
        "conversation_summary": "",
        "messages": [],
        "final_response": None,
        "iteration": 0,
        "plan_retry": 0,
        "execution_plan": None,
        "intent_labels": None,
        "run_id": None,
        "execution_mode": None,
        "fast_iteration": 0,
    }

    events: List[Dict[str, Any]] = []
    if tracer is not None:
        try:
            tracer.enable_trace(events.append)
        except Exception:
            tracer = None

    sys.stdout = io.StringIO()
    started_at = datetime.now().astimezone()
    t_invoke_start = time.time()
    try:
        result = agent.invoke(state)
    except Exception as exc:
        if tracer is not None:
            try:
                tracer.disable_trace()
            except Exception:
                pass
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"agent.invoke failed: {type(exc).__name__}: {exc}",
            adapter="genshin",
            timings=AgentTimings(init_seconds=round(t_ready - t0, 1), agent_seconds=round(time.time() - t_invoke_start, 1)),
        ).to_dict()
    finally:
        if tracer is not None:
            try:
                tracer.disable_trace()
            except Exception:
                pass
        sys.stdout = old_stdout
    t_invoke_end = time.time()

    answer = result.get("final_response", "") or ""
    trace = _trace_from_result(result, question, started_at, case_dir, events)

    if trace is not None:
        trace_tools = _tool_payload_from_trace(trace)
    else:
        trace_tools = _tool_payload_from_messages(result)

    tool_contents = []
    tool_trace: List[Dict[str, Any]] = []
    for item in trace_tools:
        full = item.pop("result_full", "")
        if full and len(full) > 10:
            tool_contents.append(f"[{item['name']}]\n{full}")
        tool_trace.append({
            "name": item.get("name"),
            "status": item.get("status"),
            "args": item.get("args") or {},
            "result_preview": item.get("result_preview") or "",
            "result_length": item.get("result_length") or 0,
        })

    raw: Dict[str, Any] = {
        "trace_id": getattr(trace, "trace_id", None) if trace is not None else None,
        "trace": trace.model_dump(mode="json") if trace is not None else None,
        "trace_build_ok": trace is not None,
    }

    return AgentResult(
        answer=answer,
        contexts="\n\n---\n\n".join(tool_contents),
        tool_trace=tool_trace,
        timings=AgentTimings(
            init_seconds=round(t_ready - t0, 1),
            agent_seconds=round(t_invoke_end - t_invoke_start, 1),
        ),
        status=STATUS_OK if answer.strip() else STATUS_EMPTY,
        error="" if answer.strip() else "final_response 为空",
        adapter="genshin",
        adapter_version="1.0",
        agent_version=os.environ.get("GOLDEN_TEST_AGENT_VERSION", ""),
        raw=raw,
    ).to_dict()


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw or "{}")
    except Exception as exc:
        print(json.dumps({"status": "agent_error", "error": f"invalid stdin JSON: {exc}"}, ensure_ascii=False))
        return 2
    result = run_agent(str(payload.get("question") or ""), str(payload.get("context") or ""))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") != "agent_error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
