# -*- coding: utf-8 -*-
"""把一次真实评测的指定题结果导出为本地回归夹具。

夹具目录放在 data/regression/fixtures/（已被 .gitignore 排除，不进入公开仓库）。
tests/test_known_case_regressions.py 会在夹具存在时，用当前 scorers 复算这些
历史答案，验证判分结果没有漂移。

用法：
    python tools/export_regression_fixtures.py \
        --run-dir runs/c4gate_full26_20260919 --ids F5,R5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "regression" / "fixtures"
DEFAULT_CASES = ROOT / "evalset" / "genshin" / "cases.json"
ALLOWED_RUN_ROOTS = (ROOT / "runs",)
ALLOWED_CASES_ROOTS = (ROOT, ROOT.parent)
ALLOWED_OUT_ROOTS = (ROOT / "data",)


def _resolve_under(path_value: str | Path, roots: tuple[Path, ...], *, must_exist: bool = True) -> Path:
    """把 CLI 传入的路径限定在白名单根目录内，避免任意路径读写。"""
    raw = Path(path_value).expanduser()
    if not raw.is_absolute():
        raw = ROOT / raw
    try:
        resolved = raw.resolve(strict=must_exist)
    except OSError as exc:
        raise ValueError(f"路径不存在或不可访问: {path_value}") from exc
    for root in roots:
        try:
            resolved.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        return resolved
    raise ValueError(f"路径不在允许目录内: {path_value}")


def _load_json(path: Path) -> Dict[str, Any]:
    # path 由 _resolve_under 限定在允许目录内，不接受任意用户路径。
    return json.loads(path.read_text(encoding="utf-8"))  # NOSONAR


def _load_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    data = _load_json(path)
    return {str(q.get("id")): q for q in data.get("questions") or [] if isinstance(q, dict)}


def _is_literal_only(case: Dict[str, Any]) -> bool:
    for kind in ("must_contain", "must_not_contain"):
        for item in case.get(kind) or []:
            if isinstance(item, dict) and item.get("match") != "literal":
                return False
    for alt in case.get("alternatives") or []:
        for kind in ("must_contain", "must_not_contain"):
            for item in alt.get(kind) or []:
                if isinstance(item, dict) and item.get("match") != "literal":
                    return False
    return True


def _export_one(
    case_id: str,
    row: Dict[str, Any],
    case: Dict[str, Any],
    source_run_id: str,
    out_dir: Path,
) -> Path:
    fixture = {
        "case_id": case_id,
        "source_run_id": source_run_id,
        "answer": str(row.get("answer") or ""),
        "contexts": str(row.get("contexts") or ""),
        "agent_status": str(row.get("agent_status") or "ok"),
        "keyword_passed": row.get("keyword_passed"),
        "must_contain_result": row.get("must_contain_result"),
        "must_not_contain_result": row.get("must_not_contain_result"),
        "actual_tools": [t.get("name") for t in (row.get("tool_trace") or []) if t.get("name")],
        "case": {
            "id": case.get("id"),
            "question": case.get("question"),
            "must_contain": case.get("must_contain") or [],
            "must_not_contain": case.get("must_not_contain") or [],
            "match_mode": case.get("match_mode") or "all",
            "alternatives": case.get("alternatives") or [],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def export_fixtures(run_dir: str, ids: List[str], out_dir: str) -> List[Path]:
    run_path = _resolve_under(run_dir, ALLOWED_RUN_ROOTS)
    results_dir = run_path / "results"
    if not results_dir.exists():
        raise ValueError(f"results 目录不存在: {results_dir}")
    manifest_path = run_path / "manifest.json"
    cases_path = _resolve_under(DEFAULT_CASES, ALLOWED_CASES_ROOTS)
    if manifest_path.exists():
        try:
            manifest = _load_json(manifest_path)
            raw_cases = str(manifest.get("evalset_file") or "")
            if raw_cases:
                cases_path = _resolve_under(raw_cases, ALLOWED_CASES_ROOTS)
        except (OSError, json.JSONDecodeError, ValueError):
            cases_path = _resolve_under(DEFAULT_CASES, ALLOWED_CASES_ROOTS)
    if not cases_path.exists():
        raise ValueError(f"题集文件不存在: {cases_path}")
    cases = _load_cases(cases_path)
    output_root = _resolve_under(out_dir or str(DEFAULT_OUT), ALLOWED_OUT_ROOTS, must_exist=False)

    selected = ids or [path.stem.lstrip("q_") for path in sorted(results_dir.glob("q_*.json"))]
    exported: List[Path] = []
    for case_id in selected:
        row_path = results_dir / f"q_{case_id}.json"
        if not row_path.exists():
            print(f"跳过（无结果文件）: {case_id}")
            continue
        case = cases.get(case_id) or {}
        if not _is_literal_only(case):
            print(f"跳过（含语义关键词，避免测试依赖网络 judge）: {case_id}")
            continue
        row = _load_json(row_path)
        exported.append(_export_one(case_id, row, case, run_path.name, output_root))
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="导出真实评测结果作为本地回归夹具")
    parser.add_argument("--run-dir", required=True, help="runs/<run_id> 目录")
    parser.add_argument("--ids", default="", help="逗号分隔的题 ID；缺省导出全部纯字面题")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    args = parser.parse_args()

    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    exported = export_fixtures(args.run_dir, ids, args.out)
    for path in exported:
        print(f"已导出: {path}")


if __name__ == "__main__":
    main()
