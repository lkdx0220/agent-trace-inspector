# -*- coding: utf-8 -*-
"""出题可判性校验：lint + 必含词的 KB 可达性（零 LLM 调用、零 token）。

给"看不到 Agent 代码与既有题集"的出题人自查用。三类规则：

  1. 必含词 · literal：只报"知识库里一次都检索不到"的词，标为【待确认】——
     不是必然判 NG：模型可能自行写出 KB 里没有的措辞（如 KB 写「五百年」而判据写「500岁」）。
     若该词是事实性专名（人名/地名/书名/任务名/道具名），0 命中 = 必然判 NG，必须换词或改判据。
  2. 必含词 · semantic：同义改写，不要求字面命中，跳过可达性检查。
  3. 锚点 · must_not_contain：
     - 出现在参考要点里 = ERROR（判据自相矛盾）；
     - 在 KB 里有命中是允许的（真实存在但答错的对象）；
     - semantic 锚点无意义 = WARN（改成 literal）。

用法：
    python tools/check_cases_kb.py --cases evalset/genshin/cases_heldout.json
    python tools/check_cases_kb.py --cases <题集> --kb <被测项目路径> --no-graph
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
DEFAULT_KB = REPO.parent / "CASE-原神剧情助手-修改用"
sys.path.insert(0, str(REPO))

MIN_HITS_OK = 3  # literal 必含词建议的最低 KB 命中数
MIN_HITS_WARN = 1  # 1-2 命中算边缘


def _keyword_rows(case: dict):
    """产出 (来源, 类别, 文本, match)；来源含题目与 alternatives。"""
    rows = []
    for kind in ("must_contain", "must_not_contain"):
        for raw in case.get(kind) or []:
            if isinstance(raw, dict):
                rows.append(("题目", kind, str(raw.get("text") or "").strip(), str(raw.get("match") or "literal")))
            else:
                rows.append(("题目", kind, str(raw).strip(), "literal"))
    for index, alt in enumerate(case.get("alternatives") or [], 1):
        if not isinstance(alt, dict):
            continue
        for kind in ("must_contain", "must_not_contain"):
            for raw in alt.get(kind) or []:
                if isinstance(raw, dict):
                    rows.append(
                        (f"备选{index}", kind, str(raw.get("text") or "").strip(), str(raw.get("match") or "literal"))
                    )
                else:
                    rows.append((f"备选{index}", kind, str(raw).strip(), "literal"))
    return [r for r in rows if r[2]]


def _iter_dump_texts(kb: Path):
    """kb_vectors_m3/chunk_dump.jsonl（六个集合的向量语料）逐条正文。"""
    for rel in ("kb_vectors_m3/chunk_dump.jsonl", "kb_vectors/chunk_dump.jsonl"):
        path = kb / rel
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = row.get("document")
                if text:
                    yield str(text)


def _iter_graph_texts(kb: Path):
    """kb_vectors/wiki_entry_graph.json 的标题 + 正文。"""
    path = kb / "kb_vectors" / "wiki_entry_graph.json"
    if not path.exists():
        return
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    entries = graph.get("entries") or {}
    rows = entries.values() if isinstance(entries, dict) else entries
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        yield f"{entry.get('title') or ''}\n{entry.get('story_text') or entry.get('full_text') or ''}"


def _collect_literal_words(cases: List[dict]) -> List[str]:
    return sorted({text for case in cases for _, _, text, match in _keyword_rows(case) if match != "semantic"})


def _count_kb_hits(kb: Path, words: List[str], use_graph: bool) -> Tuple[Dict[str, int], Dict[str, int]]:
    hits = {word: 0 for word in words}
    for text in _iter_dump_texts(kb):
        for word in words:
            if word in text:
                hits[word] += 1
    graph_hits = {word: 0 for word in words}
    if use_graph:
        for text in _iter_graph_texts(kb):
            for word in words:
                if word in text:
                    graph_hits[word] += 1
    return hits, graph_hits


def _check_one_case(
    case: dict,
    hits: Dict[str, int],
    graph_hits: Dict[str, int],
    verbose: bool,
) -> Tuple[List[str], List[str], List[str]]:
    errors: List[str] = []
    confirm: List[str] = []
    warnings: List[str] = []
    cid = case.get("id")
    reference = str(case.get("reference_answer") or "")
    if verbose:
        print(f"\n--- {cid}　{case.get('category', '')}　{case.get('difficulty', '')}")
    semantic_skipped = 0
    for _source, kind, text, match in _keyword_rows(case):
        if match == "semantic":
            semantic_skipped += 1
            continue
        total = hits.get(text, 0) + graph_hits.get(text, 0)
        detail = f"向量语料 {hits.get(text, 0)} 段 / 词条图 {graph_hits.get(text, 0)} 条"
        if kind == "must_contain":
            if total == 0:
                confirm.append(
                    f"{cid} 必含词「{text}」KB 0 命中：若是专名则必然判 NG（换词或改判据）；若是措辞则需人工确认模型能否自行写出"
                )
                tag = "待确认"
            elif total < MIN_HITS_OK:
                warnings.append(f"{cid} 必含词「{text}」KB 命中仅 {total} 次，边缘（建议 >= {MIN_HITS_OK}）")
                tag = "WARN"
            else:
                tag = "ok"
            if verbose:
                print(f"    [{tag}] 必含「{text}」　{detail}")
        else:
            if text in reference:
                errors.append(f"{cid} 锚点「{text}」出现在参考要点里：判据自相矛盾")
                tag = "ERROR"
            else:
                tag = "ok"
            if verbose:
                print(f"    [{tag}] 锚点「{text}」　{detail}（有命中是允许的：真实存在但答错的对象）")
    if verbose and semantic_skipped:
        print(f"    （跳过 {semantic_skipped} 个 semantic 词：同义改写不要求字面命中）")
    return errors, confirm, warnings


def _print_check_summary(
    name: str,
    cases: int,
    errors: List[str],
    confirm: List[str],
    warnings: List[str],
) -> None:
    print(f"\n=== 汇总 {name} ===")
    print(f"题数 {cases}　ERROR {len(errors)}　待确认 {len(confirm)}　WARN {len(warnings)}")
    for line in errors:
        print(f"  [ERROR] {line}")
    for line in confirm:
        print(f"  [待确认] {line}")
    for line in warnings:
        print(f"  [WARN] {line}")


def check(cases_path: Path, kb: Path, use_graph: bool = True, verbose: bool = True) -> dict:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("questions") or []
    literal_words = _collect_literal_words(cases)
    if verbose:
        print(f"[扫描] {cases_path.name}：{len(cases)} 题 / 需查 KB 的字面词 {len(literal_words)} 个 …", flush=True)
    hits, graph_hits = _count_kb_hits(kb, literal_words, use_graph)

    errors, confirm, warnings = [], [], []
    for case in cases:
        e, c, w = _check_one_case(case, hits, graph_hits, verbose)
        errors.extend(e)
        confirm.extend(c)
        warnings.extend(w)

    if verbose:
        _print_check_summary(cases_path.name, len(cases), errors, confirm, warnings)
    return {"cases": len(cases), "errors": errors, "confirm": confirm, "warnings": warnings}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="题集可判性校验（lint + KB 可达性）")
    parser.add_argument("--cases", required=True, help="题集 JSON 路径")
    parser.add_argument("--kb", default=str(DEFAULT_KB), help="被测项目路径（含 kb_vectors_m3 与 kb_vectors）")
    parser.add_argument("--no-graph", dest="graph", action="store_false", default=True, help="跳过词条图（更快）")
    parser.add_argument("--quiet", action="store_true", help="只打汇总")
    args = parser.parse_args(argv)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"[错误] 题集不存在: {cases_path}")
        return 2

    try:
        from evaluator.lint import lint_cases

        lint = lint_cases(str(cases_path))
        for line in lint.get("errors") or []:
            print(f"[lint 错误] {line}")
        for line in lint.get("warnings") or []:
            print(f"[lint 警告] {line}")
        print(f"[lint] {'通过' if not (lint.get('errors') or []) else '未通过'}")
    except Exception as exc:  # lint 失败不阻断可达性检查
        print(f"[lint] 跳过（{type(exc).__name__}: {exc}）")

    result = check(cases_path, Path(args.kb), use_graph=args.graph, verbose=not args.quiet)
    if not result["errors"]:
        print("结论: 无判据自相矛盾（0 ERROR）；待确认项需出题人确认措辞可否自行写出")
        return 0
    print("结论: 未通过——锚点与参考要点矛盾，必须修正")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
