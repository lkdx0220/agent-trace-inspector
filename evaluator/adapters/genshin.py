# -*- coding: utf-8 -*-
"""原神项目 adapter：子进程 + JSON 协议。

父进程通过 stdin 传入：
  {"case_id": "R7", "question": "...", "context": "..."}
子进程通过 stdout 返回 evaluator.contract.AgentResult JSON。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# evaluator/adapters/genshin.py -> adapters -> evaluator -> agent-trace-inspector -> 工作区
_DEFAULT_WORKSPACE = str(Path(__file__).resolve().parents[3])


def workspace_dir() -> str:
    return os.environ.get("GOLDEN_TEST_WORKSPACE") or _DEFAULT_WORKSPACE


def _case_dir() -> str:
    return os.path.join(workspace_dir(), "CASE-原神剧情助手-修改用")


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
    }

    sys.stdout = io.StringIO()
    t_invoke_start = time.time()
    try:
        result = agent.invoke(state)
    except Exception as exc:
        return AgentResult(
            status=STATUS_AGENT_ERROR,
            error=f"agent.invoke failed: {type(exc).__name__}: {exc}",
            adapter="genshin",
            timings=AgentTimings(init_seconds=round(t_ready - t0, 1), agent_seconds=round(time.time() - t_invoke_start, 1)),
        ).to_dict()
    finally:
        sys.stdout = old_stdout
    t_invoke_end = time.time()

    answer = result.get("final_response", "") or ""

    from langchain_core.messages import ToolMessage

    tool_contents = []
    tool_trace = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            content = msg.content if hasattr(msg, "content") else str(msg)
            name = msg.name if hasattr(msg, "name") else "unknown"
            if content and len(content) > 10:
                tool_contents.append(f"[{name}]\n{content}")
                tool_trace.append({
                    "name": name,
                    "status": "success",
                    "result_preview": str(content)[:300],
                    "result_length": len(str(content)),
                })

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
