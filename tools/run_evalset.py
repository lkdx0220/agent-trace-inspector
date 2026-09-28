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

from evaluator import config as evaluator_config  # noqa: E402
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
    parser.add_argument("--cases", default="", help="题集 JSON；默认 evalset/<adapter>/cases.json，不存在时 genshin 回退 <workspace>/golden_test_set.json")
    parser.add_argument("--adapter", default="genshin", choices=["genshin", "doctor"], help="被测 adapter：genshin 或 doctor")
    parser.add_argument("--workspace", default="", help="被测项目工作区；默认 inspector 上一级")
    parser.add_argument("--runs-dir", default="", help="运行目录；默认 inspector/runs")
    parser.add_argument("--run-id", default="", help="自定义 run_id")
    parser.add_argument("--ids", default="", help="只跑指定题，逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="最多跑几题（调试用）")
    parser.add_argument("--timeout", type=int, default=0, help="单题超时秒数；默认 genshin=300，doctor=900")
    parser.add_argument("--config", default="", help="evaluator 配置文件路径；默认 evaluator/config.yaml，不存在则 config.example.yaml")
    parser.add_argument("--judge-model", default="", help="覆盖配置里的 judge.model")
    parser.add_argument("--force", action="store_true", help="忽略缓存，全部重跑")
    parser.add_argument("--no-report", action="store_true", help="不生成 HTML 报告")
    parser.add_argument("--save-db", action="store_true", help="跑完后把结果桥接写入 inspector.db（Web UI 用）")
    parser.add_argument("--name", default="", help="Run 显示名称，仅在 --save-db 时使用")
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="覆盖每题的重复次数（0=按题集 agent_repeat：stable=1/occasional=2/noisy=3，多数通过）",
    )
    args = parser.parse_args(argv)

    config_path = evaluator_config.config_path_for_manifest(args.config)
    try:
        config = evaluator_config.load_config(args.config)
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        return 2
    if not config_path:
        print("[错误] 没有找到 evaluator 配置文件（config.yaml / config.example.yaml）")
        return 2
    print(f"[config] {config_path}")

    workspace = args.workspace or _default_workspace()
    if args.cases:
        cases_path = args.cases
    else:
        local_cases = INSPECTOR_DIR / "evalset" / args.adapter / "cases.json"
        if local_cases.exists():
            cases_path = str(local_cases)
        elif args.adapter == "genshin":
            cases_path = str(Path(workspace) / "golden_test_set.json")
        else:
            cases_path = str(local_cases)
    if not Path(cases_path).exists():
        print(f"[错误] 题集不存在: {cases_path}")
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(1000, 9999)}"
    runs_dir_value = str(args.runs_dir or evaluator_config.config_value(config, "harness.runs_dir", "runs") or "runs")
    runs_dir = Path(runs_dir_value)
    if not runs_dir.is_absolute():
        runs_dir = INSPECTOR_DIR / runs_dir
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

    default_timeout = 900 if args.adapter == "doctor" else 300
    configured_timeout = evaluator_config.config_value(config, "harness.timeout_seconds", None)
    timeout_seconds = args.timeout or max(int(configured_timeout or 0), default_timeout)
    judge_model = args.judge_model or str(evaluator_config.config_value(config, "judge.model", "deepseek-chat"))
    scorer_version = "evaluator-phase2-1.0"
    hit_rate_threshold = float(
        evaluator_config.config_value(config, "scoring.must_contain_default_hit_rate", 0.8) or 0.8
    )

    judge_settings = judge.JudgeSettings(
        model=judge_model,
        temperature=float(evaluator_config.config_value(config, "judge.params.temperature", 0.0) or 0.0),
        max_tokens=int(evaluator_config.config_value(config, "judge.params.max_tokens", 512) or 512),
        thinking=str(evaluator_config.config_value(config, "judge.params.thinking", "disabled") or "disabled"),
        contexts_chars=int(evaluator_config.config_value(config, "judge.window.contexts_chars", 60000) or 60000),
        answer_chars=int(evaluator_config.config_value(config, "judge.window.answer_chars", 20000) or 20000),
        reference_chars=int(evaluator_config.config_value(config, "judge.window.reference_chars", 20000) or 20000),
        repeat=max(1, int(evaluator_config.config_value(config, "judge.repeat", 2) or 2)),
        scorer_version=scorer_version,
        workspace=workspace,
        cache_path=str(cache_path),
    )
    judge.configure(judge_settings)

    original_project_path = str(Path(workspace) / "CASE-原神剧情助手-修改用")
    manifest_project_path = str(INSPECTOR_DIR) if args.adapter == "doctor" else original_project_path
    manifest = build_manifest(
        run_id=run_id,
        adapter=args.adapter,
        project_path=manifest_project_path,
        evalset_path=cases_path,
        case_count=len(cases),
        judge_model=judge_settings.model,
        judge_params={
            "temperature": judge_settings.temperature,
            "max_tokens": judge_settings.max_tokens,
            "thinking": judge_settings.thinking,
            "repeat": judge_settings.repeat,
        },
        judge_window={
            "contexts_chars": judge_settings.contexts_chars,
            "answer_chars": judge_settings.answer_chars,
            "reference_chars": judge_settings.reference_chars,
        },
        scorer_version=scorer_version,
        config_path=config_path,
        adapter_version="1.0",
        notes=f"{args.adapter} adapter",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write(manifest_path)
    print(f"[run] {run_id} cases={len(cases)} results={results_dir} manifest={manifest_path}")

    rows = harness.run_evalset(
        cases=cases,
        adapter_name=args.adapter,
        workspace=workspace,
        results_dir=str(results_dir),
        manifest=manifest,
        manifest_path=str(manifest_path),
        timeout_seconds=timeout_seconds,
        force=args.force,
        hit_rate_threshold=hit_rate_threshold,
        repeat_override=args.repeat,
    )
    judge.save_cache()

    passed = sum(
        1
        for r in rows
        if not r.get("error")
        and bool(r.get("keyword_passed") if r.get("keyword_passed") is not None else (
            (r.get("must_contain_result") or {}).get("passed")
            and (r.get("must_not_contain_result") or {}).get("passed")
        ))
    )
    failures = [
        r.get("id")
        for r in rows
        if r.get("error")
        or not bool(r.get("keyword_passed") if r.get("keyword_passed") is not None else (
            (r.get("must_contain_result") or {}).get("passed")
            and (r.get("must_not_contain_result") or {}).get("passed")
        ))
    ]
    print(f"[结果] {passed}/{len(rows)} 关键词硬指标通过；失败: {failures}")

    if not args.no_report:
        report_path = run_dir / "report.html"
        generate_html(rows, str(report_path), judge_model=judge_settings.model)
        print(f"[报告] {report_path}")

    if args.save_db:
        from evaluator import db_bridge
        record = db_bridge.build_run_record(
            run_dir=run_dir,
            name=args.name,
            project_path=original_project_path,
        )
        db_bridge.save_run_record(record)
        print(f"[DB] 已写入 inspector.db: {record.run_id}  {record.passed_cases}/{record.total_cases} 通过")

    print(f"[manifest] {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    raise SystemExit(main())
