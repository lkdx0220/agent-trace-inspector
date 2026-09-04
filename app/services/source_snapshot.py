# -*- coding: utf-8 -*-
"""原项目源码/提示词版本快照（只读）。

用于解决“历史 Trace 被当前代码污染”：
- 导出 Trace 时记录当时关键文件哈希 + git 状态；
- 医生分析旧 Trace 时对比当前工作区，识别版本是否一致；
- 没有快照的历史 Trace 只能以“当前视角”分析，不能断言当时行为。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.path_guard import ensure_project_path

PROMPT_FILES = [
    "prompts/system/agent_system_v4_plan.txt",
    "prompts/system/agent_system_v4_answer.txt",
]

CODE_FILES = [
    "app/agent/nodes.py",
    "app/agent/executor.py",
    "app/retrieval.py",
    "app/tools/query.py",
    "character_aliases.py",
]


def _rel_to_path(project_path: Path, rel: str) -> Path:
    return project_path / Path(rel.replace("/", os.sep))


def _file_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _hash_files(project_path: Path, rels: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in rels:
        digest = _file_sha256(_rel_to_path(project_path, rel))
        if digest:
            out[rel] = digest
    return out


def _git_info(project_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"git_commit": None, "git_dirty": None, "dirty_files": []}
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_path), capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            info["git_commit"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_path), capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
            info["git_dirty"] = bool(lines)
            info["dirty_files"] = [ln[2:].strip() for ln in lines[:50]]
    except Exception:
        pass
    return info


def capture_snapshot(project_path: str) -> Dict[str, Any]:
    """捕获当前工作区的关键文件哈希与 git 状态，作为 Trace 的版本快照。"""
    try:
        base = ensure_project_path(project_path)
    except ValueError as e:
        raise ValueError(str(e))
    base = Path(base)
    git = _git_info(base)
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "dirty_files": git["dirty_files"],
        "file_hashes": _hash_files(base, PROMPT_FILES + CODE_FILES),
    }


def compare_snapshot(snapshot: Optional[Dict[str, Any]], project_path: str) -> Dict[str, Any]:
    """比较 Trace 快照与当前工作区，返回是否一致。"""
    current = capture_snapshot(project_path)
    if not snapshot or not isinstance(snapshot, dict):
        return {
            "trace_snapshot_known": False,
            "prompt_clean": False,
            "code_clean": False,
            "all_clean": False,
            "changed_files": [],
            "current": current,
        }
    old_hashes = snapshot.get("file_hashes") or {}
    cur_hashes = current.get("file_hashes") or {}
    changed = []
    prompt_changed = False
    code_changed = False
    all_paths = set(old_hashes) | set(cur_hashes) | set(PROMPT_FILES) | set(CODE_FILES)
    for rel in sorted(all_paths):
        if rel not in old_hashes or rel not in cur_hashes or old_hashes[rel] != cur_hashes[rel]:
            changed.append(rel)
            if rel in PROMPT_FILES:
                prompt_changed = True
            if rel in CODE_FILES:
                code_changed = True
    # git 也是辅助信号：Trace 快照 dirty 且当前 dirty 无法精确知道是否同一批改动
    return {
        "trace_snapshot_known": True,
        "prompt_clean": not prompt_changed,
        "code_clean": not code_changed,
        "all_clean": not changed,
        "changed_files": changed,
        "current": current,
    }


def trace_snapshot_status(trace: Optional[Dict[str, Any]], project_path: str) -> Dict[str, Any]:
    """供医生 FactSheet 使用。"""
    if not trace:
        return {
            "trace_snapshot_known": False,
            "prompt_clean": False,
            "code_clean": False,
            "all_clean": False,
            "changed_files": [],
        }
    snapshot = trace.get("source_snapshot") or {}
    return compare_snapshot(snapshot, project_path)
