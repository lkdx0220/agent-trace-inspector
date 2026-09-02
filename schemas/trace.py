# -*- coding: utf-8 -*-
"""Agent Trace Inspector · Trace JSON Schema v0.1

参考 DeepEval 的 Trace/Span 树模型设计。
目标是让 Agent 运行轨迹可以：
1. 被本工具确定性分析；
2. 被 SQLite 索引；
3. 被人类 / 外部 AI 直接阅读。

设计文档：
    C:/Users/24701/Desktop/原神剧情/AgentTraceInspector设计文档.md
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1"


class SpanType(str, Enum):
    AGENT = "agent"
    REWRITE = "rewrite"
    ASSESS = "assess"
    ROUTER = "router"
    LLM = "llm"
    TOOL = "tool"
    ANSWER = "answer"


class SpanStatus(str, Enum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    INTERCEPTED = "intercepted"
    ERROR = "error"


class ToolCallData(BaseModel):
    """LLM 输出的一个工具调用。"""
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class Span(BaseModel):
    """通用 Span。各类型通过 span_type 区分，特有字段可选。"""
    span_id: str
    span_type: SpanType
    name: str
    status: SpanStatus = SpanStatus.SUCCESS
    step_index: int = 0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    children: List["Span"] = Field(default_factory=list)

    # ---- tool 专用 ----
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    result_preview: Optional[str] = None
    result_full: Optional[str] = None
    result_length: Optional[int] = None
    meltdown_trigger: Optional[bool] = None

    # ---- llm 专用 ----
    model: Optional[str] = None
    tool_calls: Optional[List[ToolCallData]] = None
    token_usage: Optional[Dict[str, Any]] = None


class AgentInfo(BaseModel):
    name: str = "genshin_story_agent"
    version: Optional[str] = None
    git_commit: Optional[str] = None


class TraceMetadata(BaseModel):
    """原神剧情助手特有信号。"""
    execution_mode: Optional[Literal["L1", "L2"]] = None
    intent_labels: Optional[List[str]] = None
    alias_notes: Optional[str] = None
    iteration: Optional[int] = None
    plan_retry: Optional[int] = None
    response_mode: Optional[Literal["found", "not_found"]] = None
    run_id: Optional[str] = None


class Trace(BaseModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    trace_id: str
    agent: AgentInfo = Field(default_factory=AgentInfo)
    question: str
    created_at: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[int] = None
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)
    root_span: Span
    trace_events: Optional[List[Dict[str, Any]]] = None

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=False)
