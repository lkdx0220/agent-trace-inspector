# -*- coding: utf-8 -*-
"""题集体检（Phase 1）。

用法：
  python -m evaluator.lint --cases "C:/.../golden_test_set.json"
  python -m evaluator.lint --cases cases.json --results "C:/.../golden_test_results_full26_windowfix_20260913"
  python -m evaluator.lint --cases cases.json --json

检查项：
1. metadata.total_questions 与实际条数一致；
2. 题面/字段/类型/match_mode/alternatives 基本结构；
3. 关键词分类：literal / semantic / structural；旧字符串格式给迁移警告；
4. 已废弃题面词（如「万国诸卷拾遗」）提示；
5. 给 --results 时，统计每个 must_contain 在历史 answer/contexts 里的命中数，0 命中的给提示。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_MATCH = {"literal", "semantic", "structural"}
ALLOWED_MATCH_MODE = {"all", "any", None}
ALLOWED_STABILITY = {"stable", "occasional", "noisy", "unmeasured"}
REQUIRED_FIELDS = ("id", "question", "category", "reference_answer", "must_contain", "must_not_contain")

# 已知的“题面里不该再出现的旧词/错字”，按项目维护。
# allowed_cases：该题是故意保留旧词的（例如 X7 专门考形近字纠错），不告警。
OBSOLETE_QUESTION_TERMS = {
    "万国诸卷拾遗": {
        "hint": "该词条已并入「地图文本」，题面应改为“地图文本中……”",
        "allowed_cases": [],
    },
    "影蝶之章": {
        "hint": "应为「引蝶之章」；X7 是故意保留形近字的纠错题",
        "allowed_cases": ["X7"],
    },
}


def _normalize_keyword(item: Any) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """返回 (text, match, raw)；旧字符串格式 match=None。"""
    if isinstance(item, str):
        return item.strip(), None, None
    if isinstance(item, dict):
        return str(item.get("text") or "").strip(), item.get("match"), item
    return None, None, None


def _keyword_hit(keyword: str, text: str) -> bool:
    if not keyword:
        return False
    return keyword in (text or "")


def _lint_question_identity(
    q: Dict[str, Any],
    index: int,
    question_ids: List[str],
    errors: List[str],
) -> Tuple[str, str, str]:
    label = f"questions[{index}]"
    for field in REQUIRED_FIELDS:
        if field not in q:
            errors.append(f"{label} 缺少字段 {field}")
    qid = str(q.get("id") or "")
    if not qid:
        errors.append(f"{label} id 为空")
    else:
        if qid in question_ids:
            errors.append(f"id 重复: {qid}")
        question_ids.append(qid)
    question_text = str(q.get("question") or "")
    if not question_text:
        errors.append(f"{label} question 为空")
    return label, qid, question_text


def _lint_obsolete_terms(
    question_text: str,
    qid: str,
    label: str,
    warnings: List[str],
    info: List[str],
) -> None:
    for old_term, rule in OBSOLETE_QUESTION_TERMS.items():
        if old_term not in question_text:
            continue
        hint = rule.get("hint", "")
        if qid in (rule.get("allowed_cases") or []):
            info.append(f"{qid} 题面包含「{old_term}」：{hint}")
        else:
            warnings.append(f"{qid or label} 题面仍包含旧词「{old_term}」：{hint}")


def _lint_match_mode(q: Dict[str, Any], qid: str, label: str, errors: List[str]) -> None:
    match_mode = q.get("match_mode")
    if match_mode not in ALLOWED_MATCH_MODE:
        errors.append(f"{qid or label} match_mode 非法: {match_mode!r}")


def _lint_stability(q: Dict[str, Any], qid: str, label: str, errors: List[str]) -> Any:
    stability = q.get("stability")
    if stability is not None and stability not in ALLOWED_STABILITY:
        errors.append(f"{qid or label} stability 非法: {stability!r}（允许 {sorted(ALLOWED_STABILITY)}）")
    return stability


def _lint_agent_repeat(
    q: Dict[str, Any],
    qid: str,
    label: str,
    stability: Any,
    errors: List[str],
    warnings: List[str],
    info: List[str],
) -> None:
    agent_repeat = q.get("agent_repeat")
    if agent_repeat is None:
        return
    if not isinstance(agent_repeat, int) or isinstance(agent_repeat, bool) or agent_repeat < 1:
        errors.append(f"{qid or label} agent_repeat 必须是 >=1 的整数: {agent_repeat!r}")
    elif stability == "noisy" and agent_repeat < 3:
        warnings.append(f"{qid or label} 标为 noisy 但 agent_repeat={agent_repeat} < 3，判分分辨率不足")
    elif stability == "stable" and agent_repeat > 1:
        info.append(f"{qid or label} 标为 stable 但 agent_repeat={agent_repeat}，多跑不额外增加信息")
    elif agent_repeat % 2 == 0:
        warnings.append(
            f"{qid or label} agent_repeat={agent_repeat} 为偶数，会出现平票（当前规则平票判不通过），建议用 1 或 3"
        )


def _lint_criteria_and_lists(q: Dict[str, Any], qid: str, label: str, errors: List[str]) -> None:
    if q.get("criteria_version") not in (None, "original", "relaxed"):
        errors.append(f"{qid or label} criteria_version 非法: {q.get('criteria_version')!r}")
    if not isinstance(q.get("must_contain"), list):
        errors.append(f"{qid or label} must_contain 必须是数组")
    if not isinstance(q.get("must_not_contain"), list):
        errors.append(f"{qid or label} must_not_contain 必须是数组")


def _lint_question_meta(
    q: Dict[str, Any],
    qid: str,
    label: str,
    errors: List[str],
    warnings: List[str],
    info: List[str],
) -> None:
    _lint_match_mode(q, qid, label, errors)
    # 稳定性标签与取票次数：噪声地板 4%-16%，noisy 题必须 >=3 次多数通过才判得住。
    stability = _lint_stability(q, qid, label, errors)
    _lint_agent_repeat(q, qid, label, stability, errors, warnings, info)
    _lint_criteria_and_lists(q, qid, label, errors)


def _lint_keyword_raw(
    qid: str,
    label: str,
    kind: str,
    raw: Any,
    errors: List[str],
    keyword_stats: Counter,
    keyword_unclassified: List[str],
) -> Optional[Tuple[str, str]]:
    text, match, _ = _normalize_keyword(raw)
    if not text:
        errors.append(f"{qid or label} {kind} 有空关键词")
        return None
    if match is None:
        keyword_stats["unclassified"] += 1
        keyword_unclassified.append(f"{qid}:{text}")
        return None
    if match not in ALLOWED_MATCH:
        errors.append(f"{qid or label} 关键词「{text}」match 非法: {match!r}")
        return None
    keyword_stats[match] += 1
    return text, match


def _lint_keyword_hits(
    text: str,
    match: str,
    kind: str,
    qid: str,
    label: str,
    answer: str,
    results_cache: List[Dict[str, Any]],
    warnings: List[str],
    zero_hit: List[Dict[str, str]],
    semantic_zero_skipped: List[str],
) -> None:
    if kind != "must_contain":
        return
    if results_cache and match != "semantic":
        hits = sum(1 for r in results_cache if _keyword_hit(text, str(r.get("answer") or "")))
        ctx_hits = sum(1 for r in results_cache if _keyword_hit(text, str(r.get("contexts") or "")))
        if hits == 0 and ctx_hits == 0:
            zero_hit.append({"case_id": qid, "keyword": text, "answer_hits": "0", "ctx_hits": "0"})
    elif results_cache and match == "semantic":
        # 语义词本来就允许同义改写，不能因为字面 0 命中就告警；
        # 它的有效性由 judge 在跑测时判定。
        semantic_zero_skipped.append(f"{qid}:{text}")
    elif not answer:
        warnings.append(f"{qid or label} 缺少 reference_answer，无法做词表体检")


def _lint_keyword_item(
    qid: str,
    label: str,
    kind: str,
    raw: Any,
    answer: str,
    results_cache: List[Dict[str, Any]],
    errors: List[str],
    warnings: List[str],
    keyword_stats: Counter,
    keyword_unclassified: List[str],
    zero_hit: List[Dict[str, str]],
    semantic_zero_skipped: List[str],
) -> None:
    normalized = _lint_keyword_raw(qid, label, kind, raw, errors, keyword_stats, keyword_unclassified)
    if normalized is None:
        return
    text, match = normalized
    _lint_keyword_hits(
        text,
        match,
        kind,
        qid,
        label,
        answer,
        results_cache,
        warnings,
        zero_hit,
        semantic_zero_skipped,
    )


def _load_lint_cases(cases_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not os.path.exists(cases_path):
        return None, {"ok": False, "errors": [f"题集文件不存在: {cases_path}"], "warnings": [], "info": []}
    try:
        with open(cases_path, "r", encoding="utf-8") as f:  # NOSONAR: 本地体检 CLI 显式题集路径
            data = json.load(f)
    except Exception as exc:
        return None, {
            "ok": False,
            "errors": [f"题集 JSON 解析失败: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "info": [],
        }
    if not isinstance(data, dict):
        return None, {"ok": False, "errors": ["题集根节点必须是 JSON 对象"], "warnings": [], "info": []}
    return data, None


def _validate_lint_metadata(
    data: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
) -> List[Dict[str, Any]]:
    if not data.get("schema_version"):
        warnings.append('缺少 schema_version，建议补 "1.0"')
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("缺少 metadata 对象")
        metadata = {}
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("缺少非空的 questions 数组")
        questions = []
    total = metadata.get("total_questions")
    if not isinstance(total, int):
        errors.append("metadata.total_questions 必须是整数")
    elif total != len(questions):
        errors.append(f"metadata.total_questions={total} 与实际题数 {len(questions)} 不一致")
    if not metadata.get("project"):
        warnings.append("metadata.project 为空，建议填项目标识（如 genshin / doctor）")
    return questions


def _lint_questions(
    questions: List[Dict[str, Any]],
    results_cache: List[Dict[str, Any]],
    errors: List[str],
    warnings: List[str],
    info: List[str],
    keyword_stats: Counter,
    keyword_unclassified: List[str],
    zero_hit: List[Dict[str, str]],
    semantic_zero_skipped: List[str],
) -> None:
    question_ids: List[str] = []
    for index, q in enumerate(questions):
        if not isinstance(q, dict):
            errors.append(f"questions[{index}] 不是对象")
            continue
        label, qid, question_text = _lint_question_identity(q, index, question_ids, errors)
        _lint_obsolete_terms(question_text, qid, label, warnings, info)
        _lint_question_meta(q, qid, label, errors, warnings, info)
        answer = str(q.get("reference_answer") or "")
        for kind in ("must_contain", "must_not_contain"):
            for raw in q.get(kind) or []:
                _lint_keyword_item(
                    qid,
                    label,
                    kind,
                    raw,
                    answer,
                    results_cache,
                    errors,
                    warnings,
                    keyword_stats,
                    keyword_unclassified,
                    zero_hit,
                    semantic_zero_skipped,
                )


def _build_lint_info(
    results_dir: str,
    questions: List[Dict[str, Any]],
    results_cache: List[Dict[str, Any]],
    warnings: List[str],
    info: List[str],
    keyword_stats: Counter,
    keyword_unclassified: List[str],
    zero_hit: List[Dict[str, str]],
    semantic_zero_skipped: List[str],
) -> None:
    if keyword_unclassified:
        warnings.append(
            f"还有 {len(keyword_unclassified)} 个关键词是旧字符串格式，建议迁移为 "
            '{"text":"...","match":"literal|semantic|structural"}'
        )
    if zero_hit:
        warnings.append(f"{len(zero_hit)} 个 must_contain 在历史结果里 answer/contexts 命中均为 0，建议复核")
    if semantic_zero_skipped:
        info.append(f"{len(semantic_zero_skipped)} 个语义关键词跳过字面零命中体检（由 judge 判同义表达）")
    if results_dir and not results_cache:
        warnings.append(f"results 目录没有找到 q_*.json: {results_dir}")
    info.append(f"题数: {len(questions)}")
    info.append(
        "关键词分类: "
        f"literal={keyword_stats.get('literal', 0)}, "
        f"semantic={keyword_stats.get('semantic', 0)}, "
        f"structural={keyword_stats.get('structural', 0)}, "
        f"unclassified={keyword_stats.get('unclassified', 0)}"
    )


def lint_cases(cases_path: str, results_dir: str = "") -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    data, error_report = _load_lint_cases(cases_path)
    if error_report is not None:
        return error_report
    questions = _validate_lint_metadata(data or {}, errors, warnings)

    keyword_stats: Counter = Counter()
    keyword_unclassified: List[str] = []
    zero_hit: List[Dict[str, str]] = []
    semantic_zero_skipped: List[str] = []
    results_cache: List[Dict[str, Any]] = []
    if results_dir:
        results_cache = _load_results(results_dir)

    _lint_questions(
        questions,
        results_cache,
        errors,
        warnings,
        info,
        keyword_stats,
        keyword_unclassified,
        zero_hit,
        semantic_zero_skipped,
    )
    _build_lint_info(
        results_dir,
        questions,
        results_cache,
        warnings,
        info,
        keyword_stats,
        keyword_unclassified,
        zero_hit,
        semantic_zero_skipped,
    )

    return {
        "ok": not errors,
        "cases": cases_path,
        "results": results_dir or None,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "keyword_stats": dict(keyword_stats),
        "keyword_unclassified": keyword_unclassified[:50],
        "zero_hit_keywords": zero_hit,
        "semantic_zero_skipped": semantic_zero_skipped[:50],
    }


def _load_results(results_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "q_*.json"))):  # NOSONAR: 本地体检 CLI 显式结果目录
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluator 题集体检")
    parser.add_argument("--cases", required=True, help="题集 JSON 路径")
    parser.add_argument("--results", default="", help="可选：历史结果目录（含 q_*.json）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    report = lint_cases(args.cases, args.results)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"[题集] {report.get('cases')}")
        if report.get("results"):
            print(f"[结果] {report.get('results')}")
        for line in report.get("info", []):
            print(f"  {line}")
        for line in report.get("warnings", []):
            print(f"  [warn] {line}")
        for line in report.get("errors", []):
            print(f"  [error] {line}")
        if report.get("zero_hit_keywords"):
            print("  [零命中明细]")
            for item in report["zero_hit_keywords"][:30]:
                print(f"    {item['case_id']} / {item['keyword']}")
        print("  [结论] " + ("通过（有警告）" if report.get("warnings") else "通过"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())
