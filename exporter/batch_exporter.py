# -*- coding: utf-8 -*-
"""同进程批量导出器：一次加载 Agent，连续跑多个用例，每题重置状态。

与单题子进程模式的差别：
- 知识库/向量库/LLM 客户端只在进程启动时加载一次；
- 每题使用全新的 user_query / messages / conversation_history，不共享上下文；
- Trace 事件每题独立收集，不会混入上一题。

注意：本模块只用于评测/观测，不改变原项目 Agent 的线上运行方式。
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas.trace import Trace

from exporter.safe_paths import ensure_trace_out_path, redact_sensitive
from exporter.genshin_exporter import (
    _load_agent_module,
    _enrich_trace_with_events,
    _add_missing_llm_spans_from_events,
    _fill_assess_router_times,
    build_trace_from_result,
)


class AgentBatchRunner:
    """同一个 Python 进程内顺序跑多个 Agent 用例，避免每题重复加载知识库。"""

    def __init__(self, project_path: Path | str):
        self.project_path = Path(project_path)
        self.module = _load_agent_module(self.project_path)
        self.agent = self.module.create_agent_workflow()
        self.events: List[Dict[str, Any]] = []

        # 在项目模块加载成功后再导入其 trace_recorder，避免与观测端同名 app 包冲突。
        import app.trace_recorder as tracer
        self._tracer = tracer

        # 单例 Trace sink：只注册一次，之后靠 self.events 换新列表隔离每题。
        self._enabled = False

    def _enable_trace(self) -> None:
        if self._enabled:
            return

        def _sink(ev: Dict[str, Any]) -> None:
            self.events.append(ev)

        self._tracer.enable_trace(_sink)
        self._enabled = True

    def run_case(
        self,
        question: str,
        context: str = "",
        out_path: Optional[Path] = None,
    ) -> Trace:
        """跑一题。每题开始前清空事件列表，并用全新的 Agent 状态。"""
        self._enable_trace()
        self.events = []

        started_at = datetime.now().astimezone()
        conv_history = [{"user": context, "assistant": ""}] if context.strip() else []
        result = self.agent.invoke({
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

        trace = build_trace_from_result(
            result,
            question,
            started_at,
            project_path=self.project_path,
        )
        _enrich_trace_with_events(trace, self.events)
        _add_missing_llm_spans_from_events(trace, self.events)
        _fill_assess_router_times(trace, self.events)
        trace.trace_events = self.events

        if out_path is not None:
            out_path = ensure_trace_out_path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(redact_sensitive(trace.to_json()), encoding="utf-8")
            print(f"[BatchTrace] 已导出：{out_path}")
        return trace

    def close(self) -> None:
        if self._enabled:
            self._tracer.disable_trace()
            self._enabled = False
