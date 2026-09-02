# -*- coding: utf-8 -*-
"""原项目系统提示词只读挂载。

评估器/报告生成器需要知道 Agent 的系统提示词中与工具调用相关的硬规则，
才能把“该调用工具却没调用”识别为 Agent 违反系统提示词，而不是题目设置问题。

本模块只做读取，绝不写回原项目；读取失败时返回空字符串，调用方应自行降级。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_PROJECT_PATH = "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用"
PLAN_PROMPT_REL = Path("prompts/system/agent_system_v4_plan.txt")
ANSWER_PROMPT_REL = Path("prompts/system/agent_system_v4_answer.txt")


def _prompt_path(project_path: Optional[str], rel: Path) -> Path:
    base = Path(project_path or DEFAULT_PROJECT_PATH)
    return base / rel


def get_plan_system_prompt(project_path: Optional[str] = None) -> str:
    """读取规划 Agent 的系统提示词（只读）。"""
    path = _prompt_path(project_path, PLAN_PROMPT_REL)
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def get_answer_system_prompt(project_path: Optional[str] = None) -> str:
    """读取回答 Agent 的系统提示词（只读）。"""
    path = _prompt_path(project_path, ANSWER_PROMPT_REL)
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def get_tool_requirement_excerpt(project_path: Optional[str] = None, limit: int = 2000) -> str:
    """抽取规划提示词中与「必须调用工具/豁免条件」直接相关的规则段落。

    返回内容仅供评估器/报告模型理解规则，不替代完整系统提示词。
    """
    text = get_plan_system_prompt(project_path)
    if not text:
        return ""

    # 优先截取“不调工具的前置检查”到“工具选择策略”这一段，它完整包含硬性工具要求。
    start_marker = "===== 不调工具的前置检查 ====="
    end_marker = "===== 工具选择策略 ====="
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start >= 0 and end > start:
        excerpt = text[start:end]
    else:
        excerpt = text

    # 再补上开头的“必须调用工具”总括，确保即使段落解析失败也有核心规则。
    head = text[:600]
    combined = head + "\n\n" + excerpt
    return combined[:limit]
