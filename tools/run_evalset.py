#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2 CLI：共享 harness + scorers + adapter 的薄入口。

用法：
  python tools/run_evalset.py --ids=F2 --force
  python tools/run_evalset.py --force
  python tools/run_evalset.py --cases "C:/.../golden_test_set.json" --runs-dir runs
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

INSPECTOR_DIR = Path(__file__).resolve().parents[1]
if str(INSPECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(INSPECTOR_DIR))

from evaluator import harness  # noqa: E402
from evaluator.manifest import build_manifest  # noqa: E402
from evaluator.report import generate_html  # noqa: E402
from evaluator.scorers import judge  # noqa: E402


def _default_workspace() -> str:
    import os
    return os.environ.get("GOLDEN_TEST_WORKSPACE") or str(INSPECTOR_DIR.parent)


def _load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("questions") or [])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run evalset with shared harness + scorers")
    parser.add_argument("--cases", default="", help="题集 JSON；默认 <workspace>/golden_test_set.json")
    parser.add_argument("--workspace", default="", help="被测项目工作区；默认 inspector 上一级")
    parser.add_argument("--runs-dir", default="", help="运行目录；默认 inspector/runs")
    parser.add_argument("--run-id", default="", help="自定义 run_id")
    parser.add_argument("--ids", default="", help="只跑指定题，逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="最多跑几题（调试用）")
    parser.add_argument("--timeout", type=int, default=300, help="单题超时秒数")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--force", action="store_true", help="忽略缓存，全部重跑")
    parser.add_argument("--no-report", action="store_true", help="不生成 HTML 报告")
    args = parser.parse_args(argv)

    workspace = args.workspace or _default_workspace()
    cases_path = args.cases or str(Path(workspace) / "golden_test_set.json")
    if not Path(cases_path).exists():
        print(f"[错误] 题集不存在: {cases_path}")
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(1000, 9999)}"
    runs_dir = Path(args.runs_dir) if args.runs_dir else INSPECTOR_DIR / "runs"
    run_dir = runs_dir / run_id
    results_dir = run_dir / "results"
    manifest_path = run_dir / "manifest.json"
    cache_path = run_dir / "judge_cache.json"

    cases = _load_cases(cases_path)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if str(c.get("id")) in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("[错误] 过滤后没有题目")
        return 2

    judge_settings = judge.JudgeSettings(
        model=args.judge_model,
        workspace=workspace,
        cache_path=str(cache_path),
        contexts_chars=60000,
        answer_chars=20000,
        reference_chars=20000,
    )
    judge.configure(judge_settings)

    project_path = str(Path(workspace) / "CASE-原神剧情助手-修改用")
    manifest = build_manifest(
        run_id=run_id,
        adapter="genshin",
        project_path=project_path,
        evalset_path=cases_path,
        case_count=len(cases),
        judge_model=judge_settings.model,
        judge_params={
            "temperature": judge_settings.temperature,
            "max_tokens": judge_settings.max_tokens,
            "thinking": judge_settings.thinking,
        },
        judge_window={
            "contexts_chars": judge_settings.contexts_chars,
            "answer_chars": judge_settings.answer_chars,
            "reference_chars": judge_settings.reference_chars,
        },
        scorer_version="phase2-1.0",
        adapter_version="1.0",
        notes="phase2 shared harness",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write(manifest_path)
    print(f"[run] {run_id} cases={len(cases)} results={results_dir} manifest={manifest_path}")

    rows = harness.run_evalset(
        cases=cases,
        adapter_name="genshin",
        workspace=workspace,
        results_dir=str(results_dir),
        manifest=manifest,
        manifest_path=str(manifest_path),
        timeout_seconds=args.timeout,
        force=args.force,
    )
    judge.save_cache()

    passed = sum(
        1
        for r in rows
        if r.get("error") is None
        and (r.get("must_contain_result") or {}).get("passed")
        and (r.get("must_not_contain_result") or {}).get("passed")
    )
    failures = [r.get("id") for r in rows if not (r.get("error") is None and (r.get("must_contain_result") or {}).get("passed"))]
    print(f"[结果] {passed}/{len(rows)} 硬指标通过；失败: {failures}")

    if not args.no_report:
        report_path = run_dir / "report.html"
        generate_html(rows, str(report_path), judge_model=judge_settings.model)
        print(f"[报告] {report_path}")

    print(f"[manifest] {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
