# -*- coding: utf-8 -*-
"""导出器侧的路径校验与敏感信息脱敏。

注意：exporter 不能 import 观测端的 app.services.*，否则会与原项目同名 app 包冲突。
因此这里只使用标准库，并自带原项目路径白名单。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union

INSPECTOR_ROOT = Path(__file__).resolve().parent.parent

# 当前只允许导出原项目本体这一固定路径。
ALLOWED_PROJECT_ROOT = Path(
    "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用"
).resolve()

# Trace JSON 只允许写入观测端 data/traces。
ALLOWED_TRACE_ROOT = (INSPECTOR_ROOT / "data" / "traces").resolve()
# cases 输入文件只允许放在观测端 data 目录下。
ALLOWED_CASES_ROOT = (INSPECTOR_ROOT / "data").resolve()

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(
        r"((?:api[_-]?key|access[_-]?token|secret|token)\s*[:=]\s*[\"']?)([A-Za-z0-9._\-]{12,})",
        re.IGNORECASE,
    ),
]


def ensure_project_path_local(project_path: Union[str, Path]) -> Path:
    """校验 project_path 必须是固定的原项目根目录。"""
    p = Path(project_path).expanduser().resolve()
    if p != ALLOWED_PROJECT_ROOT:
        raise ValueError(f"project_path 不在允许范围内: {project_path}")
    return p


def _resolve(path: Union[str, Path]) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = INSPECTOR_ROOT / p
    return p.resolve()


def ensure_trace_out_path(out_path: Union[str, Path]) -> Path:
    """Trace 输出路径必须位于 inspector/data/traces 下。"""
    p = _resolve(out_path)
    try:
        p.relative_to(ALLOWED_TRACE_ROOT)
    except ValueError:
        raise ValueError(f"Trace 输出路径越界: {out_path}")
    return p


def ensure_cases_file_path(cases_path: Union[str, Path]) -> Path:
    """批量 cases 文件必须位于 inspector/data 下。"""
    p = _resolve(cases_path)
    try:
        p.relative_to(ALLOWED_CASES_ROOT)
    except ValueError:
        raise ValueError(f"cases 文件路径越界: {cases_path}")
    return p


def redact_sensitive(text: str) -> str:
    """对落盘 Trace 做最小脱敏，避免把 API Key/Bearer Token 写进 JSON。"""
    result = str(text or "")
    for pattern in _SECRET_PATTERNS:
        def _repl(match: re.Match) -> str:
            if match.lastindex and match.lastindex > 1:
                return match.group(1) + "[REDACTED]"
            return "[REDACTED]"

        result = pattern.sub(_repl, result)
    return result
