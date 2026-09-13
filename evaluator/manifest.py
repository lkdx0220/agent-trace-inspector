# -*- coding: utf-8 -*-
"""Run manifest：一次跑测的版本戳与参数，落盘为 runs/<run_id>/manifest.json。

没有 manifest，跨轮/跨项目的结果不可比，只能靠手工快照目录对账。
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_VERSION = "1.0"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(cwd: str | Path) -> str:
    """固定命令：git rev-parse HEAD。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def _git_status_porcelain(cwd: str | Path) -> str:
    """固定命令：git status --porcelain。"""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def git_metadata(project_path: str | Path) -> Dict[str, Any]:
    """读取被测项目 git commit / dirty，失败时返回已知为空。"""
    cwd = Path(project_path)
    if not cwd.exists():
        return {"commit": None, "dirty": None, "changed_files": []}
    commit = _git_commit(cwd) or None
    porcelain = _git_status_porcelain(cwd)
    changed = [line[3:].strip() for line in porcelain.splitlines() if line.strip()]
    return {"commit": commit, "dirty": bool(changed), "changed_files": changed}


def kb_manifest_hash(project_path: str | Path, kb_dir: str = "kb_vectors_m3") -> Optional[str]:
    """读取 KB manifest 里的 snapshot hash；没有则返回 None。"""
    manifest_path = Path(project_path) / kb_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("kb_snapshot_hash", "snapshot_hash", "content_hash", "hash"):
        value = data.get(key)
        if value:
            return str(value)
    return sha256_file(manifest_path)


@dataclass
class RunManifest:
    run_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    manifest_version: str = MANIFEST_VERSION
    adapter: str = ""
    adapter_version: str = ""
    project_commit: Optional[str] = None
    project_dirty: Optional[bool] = None
    project_changed_files: List[str] = field(default_factory=list)
    kb_manifest_hash: Optional[str] = None
    evalset_file: str = ""
    evalset_sha256: Optional[str] = None
    case_count: int = 0
    judge_model: str = ""
    judge_params: Dict[str, Any] = field(default_factory=dict)
    judge_window: Dict[str, Any] = field(default_factory=dict)
    scorer_version: str = ""
    config_file: str = ""
    config_sha256: Optional[str] = None
    host: str = field(default_factory=platform.node)
    python_version: str = field(default_factory=platform.python_version)
    status_per_case: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        known = {f: data.get(f) for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def set_case_status(self, case_id: str, status: str) -> None:
        self.status_per_case[str(case_id)] = str(status)


def build_manifest(
    run_id: str,
    adapter: str,
    project_path: str | Path,
    evalset_path: str | Path,
    case_count: int,
    judge_model: str,
    judge_params: Optional[Dict[str, Any]] = None,
    judge_window: Optional[Dict[str, Any]] = None,
    scorer_version: str = "",
    config_path: str | Path = "",
    kb_dir: str = "kb_vectors_m3",
    adapter_version: str = "",
    notes: str = "",
) -> RunManifest:
    git = git_metadata(project_path)
    manifest = RunManifest(
        run_id=run_id,
        adapter=adapter,
        adapter_version=adapter_version,
        project_commit=git["commit"],
        project_dirty=git["dirty"],
        project_changed_files=git["changed_files"],
        kb_manifest_hash=kb_manifest_hash(project_path, kb_dir),
        evalset_file=str(evalset_path),
        evalset_sha256=sha256_file(evalset_path),
        case_count=int(case_count),
        judge_model=judge_model,
        judge_params=judge_params or {},
        judge_window=judge_window or {},
        scorer_version=scorer_version,
        config_file=str(config_path or ""),
        config_sha256=sha256_file(config_path) if config_path else None,
        notes=notes,
    )
    return manifest
