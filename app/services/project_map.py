# -*- coding: utf-8 -*-
"""项目地图生成器（只读）。

借鉴 Aider 的 RepoMap 思路：不把整个项目塞进 LLM 上下文，
而是生成一份轻量“架构地图”（文件路径 + 关键函数/类 + 行号 + 职责说明），
医生需要用哪一块时，再通过只读工具打开对应文件的具体行。

本模块绝不修改被扫描的原项目。
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PROJECT_PATH = "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用"

# 医生常备地图：只索引这些文件，不索引知识库数据文件
MAP_FILES: Dict[str, str] = {
    "app/agent/nodes.py": "Agent 核心节点：assess/rewrite/plan/fast/answer/route、重试、熔断、短路逻辑",
    "app/agent/executor.py": "LangGraph 图编排与 Trace Hook 事件发射",
    "app/schema.py": "状态定义与系统提示词加载",
    "app/retrieval.py": "hybrid_search、查询改写、角色/术语别名展开、向量+关键词融合",
    "app/tools/query.py": "具体实体查询工具：角色/任务/剧情/书籍等精确查询",
    "character_aliases.py": "角色别名映射表、resolve_aliases、ALIAS_MAP",
    "prompts/system/agent_system_v4_plan.txt": "规划阶段系统提示词（工具调用硬规则、拼写纠错）",
    "prompts/system/agent_system_v4_answer.txt": "回答阶段系统提示词（诚实原则、近似对象、未收录规则）",
    "prompts/system/agent_fast_answer.txt": "L1 快速路径系统提示词",
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "data", "logs"}


def _safe_read(path: Path, limit: int = 200000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def _iter_code_symbols(text: str) -> List[Dict[str, Any]]:
    """用 AST 提取顶层函数/类名、行号、docstring 首行。"""
    symbols = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return symbols
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            symbols.append({
                "type": "function",
                "name": node.name,
                "line": node.lineno,
                "doc": (doc or "").strip().splitlines()[0][:120] if doc else "",
            })
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            methods = [
                {"type": "method", "name": n.name, "line": n.lineno,
                 "doc": (ast.get_docstring(n) or "").strip().splitlines()[0][:120] if ast.get_docstring(n) else ""}
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            symbols.append({
                "type": "class",
                "name": node.name,
                "line": node.lineno,
                "doc": (doc or "").strip().splitlines()[0][:120] if doc else "",
                "methods": methods[:12],
            })
    return symbols


def generate_project_map(project_path: str = DEFAULT_PROJECT_PATH) -> Dict[str, Any]:
    root = Path(project_path)
    files = []
    for rel, role in MAP_FILES.items():
        path = root / rel
        if not path.exists():
            files.append({
                "path": rel,
                "role": role,
                "exists": False,
                "symbols": [],
            })
            continue
        text = _safe_read(path)
        files.append({
            "path": rel,
            "role": role,
            "exists": True,
            "size_chars": len(text),
            "symbols": _iter_code_symbols(text),
        })
    return {
        "project_path": str(root),
        "project_name": root.name,
        "files": files,
    }


def format_project_map(project_map: Dict[str, Any], max_chars: int = 6000) -> str:
    lines = [f"项目：{project_map.get('project_path')}"]
    for f in project_map.get("files", []):
        lines.append(f"\n## {f['path']}  （{f.get('role','')}）")
        if not f.get("exists"):
            lines.append("  [文件不存在]")
            continue
        if f["path"].endswith(".txt"):
            lines.append("  [系统提示词文件，完整内容通过 read_system_prompt 读取]")
            continue
        for sym in f.get("symbols", []):
            kind = sym["type"]
            lines.append(f"  {kind} {sym['name']}  (line {sym['line']})")
            if sym.get("doc"):
                lines.append(f"      {sym['doc']}")
            for m in sym.get("methods", []):
                lines.append(f"      method {m['name']} (line {m['line']})")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[项目地图被截断，可用 read_project_file 查看细节]"
    return text
