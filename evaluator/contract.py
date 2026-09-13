# -*- coding: utf-8 -*-
"""evaluator 与被测项目之间的最小契约（Phase 1）。

设计原则：
1. 契约只约定 JSON 字段，不约定实现语言/传输方式；
2. 推荐父进程起子进程 + JSON 交换，evaluator 不 import 被测项目代码；
3. 每题必须返回 status，超时/异常/空答案分别可识别，不允许静默降级。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONTRACT_VERSION = "1.0"

STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_AGENT_ERROR = "agent_error"
STATUS_EMPTY = "empty"
ALLOWED_STATUSES = (STATUS_OK, STATUS_TIMEOUT, STATUS_AGENT_ERROR, STATUS_EMPTY)


@dataclass
class AgentTimings:
    """耗时必须拆开，冷启动不能混进 Agent 耗时。"""

    init_seconds: float = 0.0
    agent_seconds: float = 0.0
    scoring_seconds: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value or 0.0) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentTimings":
        data = data or {}
        return cls(
            init_seconds=float(data.get("init_seconds") or 0.0),
            agent_seconds=float(data.get("agent_seconds") or 0.0),
            scoring_seconds=float(data.get("scoring_seconds") or 0.0),
        )


@dataclass
class AgentResult:
    """适配器唯一需要返回的结构。"""

    answer: str = ""
    contexts: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    timings: AgentTimings = field(default_factory=AgentTimings)
    status: str = STATUS_OK
    error: str = ""
    adapter: str = ""
    adapter_version: str = CONTRACT_VERSION
    agent_version: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": CONTRACT_VERSION,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version or CONTRACT_VERSION,
            "agent_version": self.agent_version,
            "status": self.status,
            "error": self.error,
            "answer": self.answer,
            "contexts": self.contexts,
            "tool_trace": self.tool_trace,
            "timings": self.timings.to_dict(),
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        return cls(
            answer=str(data.get("answer") or ""),
            contexts=str(data.get("contexts") or ""),
            tool_trace=list(data.get("tool_trace") or []),
            timings=AgentTimings.from_dict(data.get("timings")),
            status=str(data.get("status") or STATUS_OK),
            error=str(data.get("error") or ""),
            adapter=str(data.get("adapter") or ""),
            adapter_version=str(data.get("adapter_version") or CONTRACT_VERSION),
            agent_version=str(data.get("agent_version") or ""),
            raw=dict(data.get("raw") or {}),
        )

    def validate(self) -> List[str]:
        """返回问题列表；带 warning: 前缀的是提醒，不是硬错误。"""
        issues: List[str] = []
        if self.status not in ALLOWED_STATUSES:
            issues.append(f"status 非法: {self.status!r}，允许 {ALLOWED_STATUSES}")
        if not isinstance(self.answer, str):
            issues.append("answer 必须是字符串")
        if not isinstance(self.contexts, str):
            issues.append("contexts 必须是字符串")
        if not isinstance(self.tool_trace, list):
            issues.append("tool_trace 必须是数组")
        for key, value in self.timings.to_dict().items():
            if value < 0:
                issues.append(f"timings.{key} 不能为负数: {value}")
        if self.status == STATUS_OK and not self.answer.strip():
            issues.append("warning: status=ok 但 answer 为空，建议改为 status=empty")
        if self.status != STATUS_OK and not self.error.strip():
            issues.append("warning: 非 ok 状态建议填写 error 原因")
        return issues


def read_result(path: str | Path) -> AgentResult:
    with open(path, "r", encoding="utf-8") as f:
        return AgentResult.from_dict(json.load(f))


def write_result(result: AgentResult, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
