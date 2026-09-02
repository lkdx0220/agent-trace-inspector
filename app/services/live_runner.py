# -*- coding: utf-8 -*-
"""实时 Runner：通过子进程调用原项目导出器，逐题生成真实 Trace 并评测。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.db import save_trace
from app.services.eval_store import list_test_cases, save_run
from app.services.evaluator import compute_run_summary, evaluate_trace_for_case
from schemas.eval import RunRecord, TestCase
from schemas.trace import Trace

INSPECTOR_ROOT = Path(__file__).resolve().parent.parent.parent


def _env_with_keys(project_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env_file = Path(project_path) / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _export_one(project_path: Path, question: str, out_path: Path, context: str = "", timeout: int = 360) -> Optional[Trace]:
    cmd = [
        sys.executable, "-m", "exporter.genshin_exporter",
        "--project-path", str(project_path),
        "--question", question,
        "--out", str(out_path),
    ]
    if context.strip():
        cmd += ["--context", context]
    env = _env_with_keys(project_path)
    result = subprocess.run(
        cmd,
        cwd=str(INSPECTOR_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        print("[LiveRunner] 导出失败:", result.stderr[-2000:])
        return None
    if not out_path.exists():
        return None
    try:
        return Trace.model_validate_json(out_path.read_text(encoding="utf-8"))
    except Exception as e:
        print("[LiveRunner] Trace 解析失败:", e)
        return None


def run_live(project_path: str, case_ids: List[str], run_name: str = "实时评测") -> RunRecord:
    project = Path(project_path)
    all_cases = list_test_cases()
    selected = [c for c in all_cases if c["case_id"] in case_ids]
    if not selected:
        raise ValueError("没有匹配的测试用例")

    # 读取 golden_test_set 中的 context 字段，用于上下文处理题
    ctx_map = {}
    try:
        gs = json.loads(Path("C:/Users/24701/Desktop/原神剧情/golden_test_set.json").read_text(encoding="utf-8"))
        ctx_map = {q["id"]: (q.get("context") or "") for q in gs.get("questions", [])}
    except Exception:
        pass

    traces_dir = INSPECTOR_ROOT / "data" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in selected:
        case_obj = TestCase(**case)
        out_path = traces_dir / f"trace_live_{case_obj.case_id}.json"
        print(f"[LiveRunner] 开始: {case_obj.case_id} {case_obj.question[:40]}")
        trace = _export_one(project, case_obj.question, out_path, context=ctx_map.get(case_obj.case_id, ""))
        if trace is None:
            from schemas.eval import RunCaseResult
            results.append(RunCaseResult(case_id=case_obj.case_id, question=case_obj.question, passed=False, reasons=["导出 Trace 失败"]))
            continue
        save_trace(trace)
        results.append(evaluate_trace_for_case(case_obj, trace.model_dump()))

    summary = compute_run_summary(results)
    record = RunRecord(
        run_id=f"run_{uuid.uuid4().hex[:8]}",
        name=run_name,
        created_at=datetime.now().astimezone(),
        total_cases=summary["total_cases"],
        passed_cases=summary["passed_cases"],
        failed_cases=summary["failed_cases"],
        pass_rate=summary["pass_rate"],
        avg_duration_ms=summary["avg_duration_ms"],
        results=results,
    )
    save_run(record)
    return record
