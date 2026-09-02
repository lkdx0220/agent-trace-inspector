# -*- coding: utf-8 -*-
"""批量项目医生巡诊：对 run 中所有失败 case 生成/复用处方，并输出项目健康概览。

用法:
    python run_doctor_all.py --run run_3054e696
    python run_doctor_all.py --run run_3054e696 --force
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.services.doctor_tools import DEFAULT_PROJECT_PATH
from app.services.eval_store import get_prescription, get_run, save_prescription
from app.services.project_doctor import prescribe_run_case

REPORT_DIR = Path(__file__).resolve().parent / "data" / "reports"


def _is_fresh(payload: Dict[str, Any]) -> bool:
    """判断处方是否已带有 VulnClaw 反幻觉闸门字段。"""
    report = payload.get("report") or {}
    return bool(
        report.get("_grounding")
        and report.get("_near_miss")
        and report.get("diagnosis", {}).get("evidence_level")
    )


def _run_one(run_id: str, case_id: str, project_path: str, model: str | None, force: bool) -> Dict[str, Any]:
    existing = get_prescription(run_id, case_id)
    if existing and not force and _is_fresh(existing["payload"]):
        print(f"[复用] {case_id}: 已有新版本处方 ({existing['created_at']})")
        return existing["payload"]

    print(f"[诊断] {case_id}: 运行项目医生 ...")
    out = prescribe_run_case(run_id, case_id, project_path=project_path, model=model)
    if not out.get("ok"):
        print(f"[失败] {case_id}: {out.get('error')}")
        return out
    save_prescription(run_id, case_id, out, model=out.get("model") or "")
    print(f"[完成] {case_id}: coverage={out.get('coverage')}")
    return out


def _write_overview(run_id: str, run: Dict[str, Any], results_by_case: Dict[str, Dict[str, Any]]) -> Path:
    lines: List[str] = []
    failed = [r for r in run["results"] if not r.get("passed")]
    passed = [r for r in run["results"] if r.get("passed")]
    lines.append(f"# 项目医生巡诊总览：{run_id}")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Run 名称：{run.get('name') or run_id}")
    lines.append(f"- 题目数：{len(run['results'])}，通过：{len(passed)}，失败：{len(failed)}")
    lines.append("")

    if not failed:
        lines.append("**全部通过，无待诊失败题。**")
        lines.append("")

    for r in failed:
        case_id = r.get("case_id") or r.get("id") or "?"
        payload = results_by_case.get(case_id, {})
        report = payload.get("report") or {}
        diag = report.get("diagnosis") or {}
        lines.append(f"## {case_id}：{r.get('question') or ''}")
        lines.append("")
        lines.append(f"- 最终答案：{(r.get('answer') or '')[:200]}")
        lines.append(f"- 失败原因：{json.dumps(r.get('reasons'), ensure_ascii=False)}")
        lines.append(f"- 诊断分类：{diag.get('issue_classification')} / {diag.get('data_vs_recall')}")
        lines.append(f"- 诊断摘要：{diag.get('summary')}")
        lines.append(f"- 证据等级：{diag.get('evidence_level')}")
        lines.append("")
        rxs = report.get("prescriptions") or []
        for i, rx in enumerate(rxs, 1):
            lines.append(f"### 处方 {i}（{rx.get('severity')} / {rx.get('evidence_level')}）")
            lines.append(f"- 问题：{rx.get('issue')}")
            lines.append(f"- 根因：{rx.get('root_cause')}")
            lines.append(f"- 目标文件：{rx.get('target_file')}")
            lines.append(f"- 建议：{rx.get('suggestion')}")
            lines.append(f"- 证据：{', '.join(rx.get('evidence_ids') or [])}")
            lines.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"doctor_overview_{run_id}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="批量项目医生巡诊")
    parser.add_argument("--run", required=True, help="run_id")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true", help="即使已有新处方也重新诊断")
    args = parser.parse_args()

    run = get_run(args.run)
    if not run:
        print(f"未找到 run: {args.run}")
        return 1

    failed = [r for r in run["results"] if not r.get("passed")]
    print(f"run={args.run} 共 {len(run['results'])} 题，失败 {len(failed)} 题: {[r.get('case_id') for r in failed]}")

    results_by_case: Dict[str, Dict[str, Any]] = {}
    for r in failed:
        case_id = r.get("case_id") or r.get("id") or "?"
        results_by_case[case_id] = _run_one(args.run, case_id, args.project, args.model, args.force)

    out_path = _write_overview(args.run, run, results_by_case)
    print(f"已生成项目医生总览: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
