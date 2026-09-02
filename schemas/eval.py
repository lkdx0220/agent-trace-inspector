# -*- coding: utf-8 -*-
"""评测核心模型：测试用例、运行、单题结果。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnswerVariant(BaseModel):
    """同一道题的另一种可接受答案标准（双答案/多答案机制）。

    - name：变体名称，用于报告与诊断展示
    - must_contain / must_not_contain / match_mode：该变体独立的关键词判定标准
    - expected_answer：该变体对应的参考答案（可选）
    """

    name: str = ""
    expected_answer: Optional[str] = None
    must_contain: List[str] = Field(default_factory=list)
    must_not_contain: List[str] = Field(default_factory=list)
    match_mode: str = "all"


class TestCase(BaseModel):
    case_id: str
    question: str
    category: str = ""
    difficulty: str = ""
    expected_answer: Optional[str] = None
    must_contain: List[str] = Field(default_factory=list)
    must_not_contain: List[str] = Field(default_factory=list)
    expected_tools: List[str] = Field(default_factory=list)
    expected_route: Optional[str] = None  # "L1" / "L2"
    match_mode: str = "all"  # "all"（默认，命中率阈值）+ "any"（命中任意一个即通过）
    alternatives: List[AnswerVariant] = Field(default_factory=list)


class RunCaseResult(BaseModel):
    case_id: str
    question: str
    passed: bool = False
    keyword_pass: bool = False
    matched_variant: Optional[str] = None
    tool_pass: Optional[bool] = None
    route_pass: Optional[bool] = None
    prompt_pass: Optional[bool] = None
    prompt_violations: List[str] = Field(default_factory=list)
    answer: Optional[str] = None
    trace_id: Optional[str] = None
    actual_tools: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    run_id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    agent_name: str = "genshin_story_agent"
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0
    avg_duration_ms: float = 0.0
    results: List[RunCaseResult] = Field(default_factory=list)
