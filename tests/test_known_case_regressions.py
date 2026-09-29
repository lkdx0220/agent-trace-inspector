# -*- coding: utf-8 -*-
"""已知失败/回归病例：用当前 scorers 复算历史答案，验证判分没有漂移。

夹具由 tools/export_regression_fixtures.py 从真实 run 结果导出到
data/regression/fixtures/（本地目录，已 gitignore）。没有夹具时自动跳过，
公开仓库 CI 不会因此失败。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from evaluator.scorers.keywords import check_case_keywords

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "regression" / "fixtures"


def _load_fixtures() -> List[Dict[str, Any]]:
    fixtures: List[Dict[str, Any]] = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fixtures.append(json.loads(path.read_text(encoding="utf-8")))
    return fixtures


def test_known_case_keyword_results_stable():
    fixtures = _load_fixtures()
    if not fixtures:
        pytest.skip("没有本地回归夹具（先运行 tools/export_regression_fixtures.py）")

    for fixture in fixtures:
        case_id = str(fixture.get("case_id"))
        case = fixture.get("case") or {}
        answer = str(fixture.get("answer") or "")
        expected_pass = bool(fixture.get("keyword_passed"))

        result = check_case_keywords(case, answer, hit_rate_threshold=0.8)
        got_miss = result["must_contain_result"].get("miss") or []
        expected_miss = (fixture.get("must_contain_result") or {}).get("miss") or []
        got_violations = result["must_not_contain_result"].get("violations") or []
        expected_violations = (fixture.get("must_not_contain_result") or {}).get("violations") or []

        assert result["passed"] == expected_pass, (
            f"{case_id} keyword_pass 漂移：期望 {expected_pass}，实际 {result['passed']}"
            f"；miss={got_miss} violations={got_violations}"
        )
        assert got_miss == expected_miss, (
            f"{case_id} miss 漂移：期望 {expected_miss}，实际 {got_miss}"
        )
        assert got_violations == expected_violations, (
            f"{case_id} violations 漂移：期望 {expected_violations}，实际 {got_violations}"
        )
