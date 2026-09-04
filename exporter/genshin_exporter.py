# -*- coding: utf-8 -*-
"""原神剧情助手 Trace 导出器（P0）。

不修改原项目代码，通过 sys.path 动态加载原项目入口，
调用其 create_agent_workflow() 跑一轮，并把 State 转成
Agent Trace Inspector 的 Trace JSON。

用法：
    cd agent-trace-inspector
    python -m exporter.genshin_exporter \
        --project-path "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用" \
        --question "胡桃传说任务讲了什么？" \
        --out "data/traces/sample_run.json"
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from schemas.trace import (
    AgentInfo,
    SourceSnapshot,
    Span,
    SpanStatus,
    SpanType,
    ToolCallData,
    Trace,
    TraceMetadata,
)

MELTDOWN_TRIGGER_TOOLS = {"load_book_content", "load_quest_content", "find_first_mention"}


def _load_agent_module(project_path: Path):
    """动态加载原神剧情助手入口，避免修改原项目。"""
    entry = Path(project_path) / "genshin_story_agent.py"
    if not entry.exists():
        raise FileNotFoundError(f"未找到原项目入口：{entry}")

    # 让原项目的 app / intent_router / character_aliases 等顶层模块可以被 import
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))

    spec = importlib.util.spec_from_file_location("genshin_story_agent_inspector", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _status_of_tool_result(content: str) -> SpanStatus:
    """按返回内容开头判断失败/拦截/错误。

    不能用全文包含判断，因为长任务的正常正文里也可能出现
    “未找到”“不存在”等词。
    """
    stripped = content.lstrip()
    if stripped.startswith("[系统拦截]") or "系统拦截" in content[:50]:
        return SpanStatus.INTERCEPTED
    if stripped.startswith("工具执行出错"):
        return SpanStatus.ERROR
    for keyword in ("未找到", "未收录", "不存在", "无匹配", "No match", "not found"):
        if stripped.startswith(keyword):
            return SpanStatus.NOT_FOUND
    return SpanStatus.SUCCESS


def _find_tool_call(messages: List[BaseMessage], tool_call_id: Optional[str]):
    """从最近的 AIMessage 中查找匹配 tool_call_id 的参数。"""
    if not tool_call_id:
        return None
    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if tc.get("id") == tool_call_id or tc.get("tool_call_id") == tool_call_id:
                    return tc
    return None


def _tool_calls_from_ai(msg: BaseMessage) -> List[ToolCallData]:
    return [
        ToolCallData(
            name=tc.get("name", "?"),
            args=tc.get("args", {}),
            tool_call_id=tc.get("id"),
        )
        for tc in getattr(msg, "tool_calls", None) or []
    ]


def build_trace_from_result(
    result: Dict[str, Any],
    question: str,
    started_at: datetime,
    agent_info: Optional[AgentInfo] = None,
    project_path: Optional[Path] = None,
) -> Trace:
    """把 create_agent_workflow().invoke() 的结果转成 Trace。"""
    messages: List[BaseMessage] = result.get("messages") or []
    execution_mode = result.get("execution_mode")
    final_response = result.get("final_response") or ""

    trace_id = f"trace_{started_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    response_mode = "not_found" if final_response.strip() == "当前知识库未收录。" else "found"

    root = Span(
        span_id="span_agent",
        span_type=SpanType.AGENT,
        name="genshin_story_agent",
        step_index=0,
        start_time=started_at,
        input={"question": question},
        output={"final_response": final_response, "response_mode": response_mode},
    )
    step = 0

    def add_span(parent: Span, span: Span) -> Span:
        nonlocal step
        step += 1
        span.step_index = step
        parent.children.append(span)
        return span

    # 1) rewrite
    rewrite = add_span(root, Span(
        span_id="span_rewrite",
        span_type=SpanType.REWRITE,
        name="rewrite_query",
        input={"user_query": question},
        output={
            "rewritten_query": result.get("rewritten_query"),
            "alias_notes": result.get("alias_notes") or "",
        },
    ))

    # 2) assess
    add_span(rewrite, Span(
        span_id="span_assess",
        span_type=SpanType.ASSESS,
        name="assess_query",
        input={"user_query": question},
        output={"execution_mode": execution_mode},
    ))

    # 3) 分支：L1 快速路径 / L2 完整路径
    current_llm_parent: Optional[Span] = None

    if execution_mode == "L1":
        current_llm_parent = add_span(rewrite, Span(
            span_id="span_fast_agent",
            span_type=SpanType.LLM,
            name="fast_agent",
            output={},
        ))
    else:
        add_span(rewrite, Span(
            span_id="span_router",
            span_type=SpanType.ROUTER,
            name="intent_router",
            input={"user_query": question},
            output={
                "intent_labels": result.get("intent_labels") or [],
                "injected_tools": [],
            },
        ))

    # 4) 扫描消息，生成 plan/answer/tool span
    last_ai_with_tool_calls: Optional[BaseMessage] = None
    plan_count = 0

    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            content = msg.content or ""

            if tool_calls:
                # Plan 阶段（L2）或 fast_agent 阶段（L1）
                plan_count += 1
                if execution_mode == "L2":
                    parent = current_llm_parent or rewrite
                    current_llm_parent = add_span(parent, Span(
                        span_id=f"span_plan_{plan_count}",
                        span_type=SpanType.LLM,
                        name="plan_agent",
                        output={
                            "execution_plan": content,
                            "tool_call_names": [tc.get("name") for tc in tool_calls],
                        },
                        tool_calls=_tool_calls_from_ai(msg),
                    ))
                last_ai_with_tool_calls = msg
            elif content and "final_response" in result:
                # 最终回答
                if execution_mode == "L1" and current_llm_parent is not None:
                    add_span(current_llm_parent, Span(
                        span_id="span_answer_l1",
                        span_type=SpanType.ANSWER,
                        name="answer_agent",
                        output={"final_response": content, "short_circuit": False},
                    ))
                else:
                    add_span(rewrite, Span(
                        span_id="span_answer",
                        span_type=SpanType.ANSWER,
                        name="answer_agent",
                        output={
                            "final_response": content,
                            "short_circuit": content.strip() == "当前知识库未收录。",
                        },
                    ))
        elif isinstance(msg, ToolMessage):
            content = msg.content or ""
            status = _status_of_tool_result(content)
            tc = _find_tool_call(
                messages[: messages.index(msg)] if msg in messages else messages,
                getattr(msg, "tool_call_id", None),
            )
            args = tc.get("args", {}) if tc else {}
            tool_name = (
                tc.get("name")
                if tc
                else (getattr(msg, "name", None) or "?")
            )
            tool_span = Span(
                span_id=f"span_tool_{step + 1}",
                span_type=SpanType.TOOL,
                name=tool_name or "?",
                status=status,
                tool_args=args,
                tool_call_id=getattr(msg, "tool_call_id", None),
                result_preview=content[:500],
                result_full=content,
                result_length=len(content),
                meltdown_trigger=(
                    (getattr(msg, "name", "") in MELTDOWN_TRIGGER_TOOLS)
                    and status == SpanStatus.SUCCESS
                ),
            )
            if current_llm_parent is not None:
                add_span(current_llm_parent, tool_span)
            else:
                add_span(rewrite, tool_span)

    root.end_time = datetime.now().astimezone()
    duration_ms = int((root.end_time - started_at).total_seconds() * 1000)

    source_snapshot = None
    if project_path is not None:
        try:
            from app.services.source_snapshot import capture_snapshot
            source_snapshot = SourceSnapshot(**capture_snapshot(str(project_path)))
        except Exception:
            source_snapshot = None

    return Trace(
        trace_id=trace_id,
        agent=agent_info or AgentInfo(),
        question=question,
        created_at=started_at,
        duration_ms=duration_ms,
        metadata=TraceMetadata(
            execution_mode=execution_mode,
            intent_labels=result.get("intent_labels") or [],
            alias_notes=result.get("alias_notes"),
            iteration=result.get("iteration"),
            plan_retry=result.get("plan_retry"),
            response_mode=response_mode,
            run_id=result.get("run_id"),
        ),
        source_snapshot=source_snapshot,
        root_span=root,
    )


def _enrich_trace_with_events(trace: Trace, events: list) -> None:
    """使用原项目 Trace Hook 采集的事件，补充 Span 时间戳/状态/预览。

    覆盖：
    - tool_start / tool_end → ToolSpan
    - llm_start / llm_end → plan / answer / fast 的 LLMSpan
    - rewrite / assess / route / answer_start / answer_end → 阶段 Span
    """
    from datetime import datetime as _dt

    starts = {}
    ends = {}
    by_event = {}
    llm_starts = {}
    llm_ends = {}
    plan_events = []

    for ev in events:
        data = ev.get("data") or {}
        etype = ev.get("event")
        tid = data.get("tool_call_id")
        role = data.get("role")
        if etype == "tool_start" and tid:
            starts[tid] = ev
        elif etype == "tool_end" and tid:
            ends[tid] = ev
        elif etype == "llm_start" and role:
            llm_starts.setdefault(role, []).append(ev)
        elif etype == "llm_end" and role:
            llm_ends.setdefault(role, []).append(ev)
        elif etype == "plan":
            plan_events.append(ev)
        else:
            by_event.setdefault(etype, []).append(ev)

    llm_idx = {role: 0 for role in llm_starts}

    def _take(role):
        i = llm_idx.get(role, 0)
        if i < len(llm_starts.get(role, [])):
            s = llm_starts[role][i]
            e = llm_ends.get(role, [None])[i] if i < len(llm_ends.get(role, [])) else None
            llm_idx[role] = i + 1
            return s, e
        return None, None

    def walk(span):
        if span.span_type == SpanType.TOOL and span.tool_call_id:
            start = starts.get(span.tool_call_id)
            end = ends.get(span.tool_call_id)
            if start:
                span.start_time = _dt.fromtimestamp(start["timestamp"]).astimezone()
            if end:
                span.end_time = _dt.fromtimestamp(end["timestamp"]).astimezone()
                data = end.get("data") or {}
                if data.get("status"):
                    span.status = SpanStatus(data["status"])
                if data.get("result_preview") is not None:
                    span.result_preview = data["result_preview"]
                if data.get("result_length") is not None:
                    span.result_length = data["result_length"]
                if data.get("meltdown_trigger") is not None:
                    span.meltdown_trigger = data["meltdown_trigger"]

        elif span.span_type == SpanType.LLM:
            role = None
            if span.name == "plan_agent":
                role = "plan"
            elif span.name == "answer_agent":
                role = "answer"
            elif span.name == "fast_agent":
                role = "fast"
            if role:
                s, e = _take(role)
                if s:
                    span.start_time = _dt.fromtimestamp(s["timestamp"]).astimezone()
                    if (s.get("data") or {}).get("model"):
                        span.model = s["data"]["model"]
                if e:
                    span.end_time = _dt.fromtimestamp(e["timestamp"]).astimezone()
                    data = e.get("data") or {}
                    if data.get("model"):
                        span.model = data["model"]
            # 兼容旧 fallback：没有 llm_start/end 时，用 plan 事件补 end_time
            if span.name == "plan_agent" and not span.end_time and plan_events:
                span.end_time = _dt.fromtimestamp(plan_events[0]["timestamp"]).astimezone()
                plan_events.pop(0)

        elif span.span_type == SpanType.REWRITE and by_event.get("rewrite"):
            span.end_time = _dt.fromtimestamp(by_event["rewrite"][0]["timestamp"]).astimezone()
            span.start_time = trace.created_at
        elif span.span_type == SpanType.ASSESS and by_event.get("assess"):
            span.end_time = _dt.fromtimestamp(by_event["assess"][0]["timestamp"]).astimezone()
        elif span.span_type == SpanType.ROUTER and by_event.get("route"):
            span.end_time = _dt.fromtimestamp(by_event["route"][0]["timestamp"]).astimezone()
        elif span.span_type == SpanType.ANSWER:
            starts_ans = by_event.get("answer_start")
            ends_ans = by_event.get("answer_end")
            if starts_ans:
                span.start_time = _dt.fromtimestamp(starts_ans[-1]["timestamp"]).astimezone()
            if (starts_ans[-1].get("data") or {}).get("model"):
                span.model = starts_ans[-1]["data"]["model"]
            if ends_ans:
                span.end_time = _dt.fromtimestamp(ends_ans[-1]["timestamp"]).astimezone()

        for child in span.children:
            walk(child)

    walk(trace.root_span)


def _fill_assess_router_times(trace: Trace, events: list) -> None:
    """给 assess/router 阶段补 start/end，减少未归属时间。"""
    from datetime import datetime as _dt

    by_event = {}
    for ev in events:
        etype = ev.get("event")
        if etype in ("assess", "route", "rewrite"):
            by_event.setdefault(etype, ev)

    assess = router = rewrite = None
    def walk(span):
        nonlocal assess, router, rewrite
        if span.span_type == SpanType.ASSESS:
            assess = span
        elif span.span_type == SpanType.ROUTER:
            router = span
        elif span.span_type == SpanType.REWRITE:
            rewrite = span
        for c in span.children:
            walk(c)
    walk(trace.root_span)

    if rewrite and not assess:
        pass
    if assess:
        if rewrite and assess.start_time is None and rewrite.end_time:
            assess.start_time = rewrite.end_time
        if assess.end_time is None and by_event.get("assess"):
            assess.end_time = _dt.fromtimestamp(by_event["assess"]["timestamp"]).astimezone()
    if router:
        if assess and router.start_time is None and assess.end_time:
            router.start_time = assess.end_time
        if router.end_time is None and by_event.get("route"):
            router.end_time = _dt.fromtimestamp(by_event["route"]["timestamp"]).astimezone()


def _add_missing_llm_spans_from_events(trace: Trace, events: list) -> None:
    """根据 llm_start/llm_end 事件，补上没有落在 State 消息里的 LLM Span。

    覆盖：
    - plan_retry（强制重试）
    - 最后一段“无工具调用、直接进入回答”的 plan LLM 调用
    - 其他没有被消息重建覆盖的 LLM 调用
    """
    from datetime import datetime as _dt

    # 收集当前已有的 LLM Span 时间段，避免重复添加
    existing = []
    def collect(span):
        if span.span_type == SpanType.LLM and span.start_time and span.end_time:
            existing.append((span.start_time, span.end_time))
        for child in span.children:
            collect(child)
    collect(trace.root_span)

    def covered(st, en):
        for es, ee in existing:
            if abs((es - st).total_seconds()) < 1 and abs((ee - en).total_seconds()) < 1:
                return True
        return False

    # 找 rewrite 作为父节点（plan/answer 都挂在它下面）
    parent = None
    def find_parent(span):
        nonlocal parent
        if span.span_type == SpanType.REWRITE and span.name == "rewrite_query":
            parent = span
            return
        for child in span.children:
            find_parent(child)
    find_parent(trace.root_span)
    if parent is None:
        return

    starts = {}
    ends = {}
    for ev in events:
        data = ev.get("data") or {}
        role = data.get("role")
        if ev.get("event") == "llm_start" and role:
            starts.setdefault(role, []).append(ev)
        elif ev.get("event") == "llm_end" and role:
            ends.setdefault(role, []).append(ev)

    new_spans = []
    for role, start_events in starts.items():
        end_events = ends.get(role, [])
        for i, sev in enumerate(start_events):
            eev = end_events[i] if i < len(end_events) else None
            st = _dt.fromtimestamp(sev["timestamp"]).astimezone()
            en = _dt.fromtimestamp(eev["timestamp"]).astimezone() if eev else None
            if not en:
                continue
            if covered(st, en):
                continue
            if role == "answer":
                # answer 已有 ANSWER Span，不重复添加 LLM span
                continue
            if role == "fast" and any(s.span_type == SpanType.LLM and s.name == "fast_agent" for s in []):
                continue
            name = "plan_agent_retry" if role == "plan_retry" else ("plan_agent" if role == "plan" else "fast_agent")
            span = Span(
                span_id=f"span_llm_event_{role}_{i}",
                span_type=SpanType.LLM,
                name=name,
                start_time=st,
                end_time=en,
                model=(sev.get("data") or {}).get("model"),
                output={},
                status=SpanStatus.SUCCESS,
            )
            new_spans.append(span)
            existing.append((st, en))

    parent.children.extend(new_spans)


def run_and_export(project_path: Path, question: str, out_path: Path, context: str = "") -> Trace:
    started_at = datetime.now().astimezone()
    module = _load_agent_module(project_path)

    # 启用原项目可选 Trace Hook（默认关闭，这里由导出器开启）
    import app.trace_recorder as tracer
    events = []
    def _sink(ev):
        events.append(ev)
    tracer.enable_trace(_sink)

    agent = module.create_agent_workflow()
    conv_history = [{"user": context, "assistant": ""}] if context.strip() else []
    result = agent.invoke({
        "user_query": question,
        "rewritten_query": None,
        "alias_notes": None,
        "conversation_history": conv_history,
        "conversation_summary": "",
        "messages": [],
        "final_response": None,
        "iteration": 0,
        "plan_retry": 0,
        "execution_plan": None,
        "intent_labels": None,
        "run_id": f"inspector_{uuid.uuid4().hex[:8]}",
        "execution_mode": None,
        "fast_iteration": 0,
    })

    tracer.disable_trace()

    trace = build_trace_from_result(result, question, started_at, project_path=project_path)
    _enrich_trace_with_events(trace, events)
    _add_missing_llm_spans_from_events(trace, events)
    _fill_assess_router_times(trace, events)
    trace.trace_events = events
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(trace.to_json(), encoding="utf-8")
    print(f"[Trace] 已导出：{out_path}")
    print(f"[Trace] 最终回答：{result.get('final_response', '')[:80]}")
    return trace


def main():
    parser = argparse.ArgumentParser(description="原神剧情助手 Trace 导出器")
    parser.add_argument("--project-path", required=True, help="CASE-原神剧情助手-修改用 目录")
    parser.add_argument("--question", required=True, help="要运行的问题")
    parser.add_argument("--out", default="data/traces/trace_latest.json", help="输出 JSON 路径")
    parser.add_argument("--context", default="", help="上一轮对话上下文（可选）")
    args = parser.parse_args()

    run_and_export(Path(args.project_path), args.question, Path(args.out), args.context)


if __name__ == "__main__":
    main()
