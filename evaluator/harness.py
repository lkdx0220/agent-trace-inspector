# -*- coding: utf-8 -*-
"""共享 harness：子进程跑 adapter、超时 kill、打分、落盘、manifest 更新。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluator.contract import AgentResult, AgentTimings
from evaluator.manifest import RunManifest, sha256_text
from evaluator.scorers import judge, score_case

RESULT_SCHEMA_VERSION = "3"
MAX_AGENT_REPEAT = 9  # 重复次数上限：题集里的 agent_repeat 是数据，不能让数据决定资源消耗


def _resolve_repeat(case: Dict[str, Any], repeat_override: int = 0) -> int:
    """取每题重复次数：显式覆盖 > case.agent_repeat > 1；非法值回退 1，并夹到上限。"""
    raw = repeat_override or case.get("agent_repeat") or 1
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, MAX_AGENT_REPEAT))


def _safe_qid(qid: Any, index: int) -> str:
    """落盘文件名用的安全 id：只留字母数字与 _.-，其余替换为 _，并限长。"""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(qid or "")).strip("._")
    return (safe or f"case{index}")[:64]


def content_digest(result: Dict[str, Any]) -> str:
    core = {
        "id": result.get("id"),
        "adapter": result.get("adapter"),
        "question": result.get("question"),
        "category": result.get("category"),
        "answer": result.get("answer"),
        "tool_trace": result.get("tool_trace"),
        "trace_id": (result.get("raw") or {}).get("trace_id"),
        "ragas": result.get("ragas"),
        "must_contain_result": result.get("must_contain_result"),
        "must_not_contain_result": result.get("must_not_contain_result"),
        "citation_result": result.get("citation_result"),
        "elapsed": result.get("elapsed"),
        "init_seconds": result.get("init_seconds"),
        "agent_seconds": result.get("agent_seconds"),
        "judge_valid": result.get("judge_valid"),
        "judge_errors": result.get("judge_errors"),
        "judge_samples": result.get("judge_samples"),
        "judge_truncation": result.get("judge_truncation"),
    }
    payload = json.dumps(core, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def case_sha256(case: Dict[str, Any]) -> str:
    return sha256_text(json.dumps(case, ensure_ascii=False, sort_keys=True))


def _validate_cached_result(cached: Any, case: Dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(cached, dict):
        return False, "缓存不是 JSON 对象"
    if cached.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False, f"schema_version 不匹配: {cached.get('schema_version')!r}"
    if cached.get("id") != case.get("id"):
        return False, "缓存 id 与题目不一致"
    prov = cached.get("provenance") or {}
    if prov.get("question_sha256") != case_sha256(case):
        return False, "题目内容已变化，缓存失效"
    if "error" not in cached:
        required = [
            "question", "category", "answer", "ragas",
            "must_contain_result", "must_not_contain_result",
            "citation_result", "elapsed",
        ]
        missing = [k for k in required if k not in cached]
        if missing:
            return False, f"缓存缺少字段: {missing}"
        if not isinstance(cached.get("ragas"), dict):
            return False, "ragas 字段不是对象"
    expected_digest = cached.get("content_digest")
    if not expected_digest or expected_digest != content_digest(cached):
        return False, "content_digest 不匹配，缓存可能被篡改或损坏"
    return True, ""


def _child_env(workspace: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["GOLDEN_TEST_WORKSPACE"] = workspace
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_case(
    adapter_name: str,
    case: Dict[str, Any],
    workspace: str,
    timeout_seconds: int = 300,
    cwd: Optional[str] = None,
) -> AgentResult:
    """起一个内置 adapter 子进程跑单题；超时直接 kill。"""
    if adapter_name == "genshin":
        adapter_module = "evaluator.adapters.genshin"
    elif adapter_name == "doctor":
        adapter_module = "evaluator.adapters.doctor"
    else:
        return AgentResult(status="agent_error", error=f"未知 adapter: {adapter_name}", adapter=adapter_name)
    payload = {
        "case_id": str(case.get("id") or ""),
        "question": str(case.get("question") or ""),
        "context": str(case.get("context") or ""),
        "run_id": str(case.get("run_id") or ""),
        "target_case_id": str(case.get("target_case_id") or ""),
        "project_path": str(case.get("project_path") or ""),
        "case": case,
    }
    run_cwd = cwd or str(Path(__file__).resolve().parents[1])
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", adapter_module],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(timeout_seconds),
            cwd=run_cwd,
            env=_child_env(workspace),
        )
    except subprocess.TimeoutExpired:
        return AgentResult(
            status="timeout",
            error=f"adapter 超时（{timeout_seconds}s，子进程已终止）",
            adapter=adapter_name,
            timings=AgentTimings(init_seconds=0.0, agent_seconds=round(time.time() - started, 1)),
        )

    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return AgentResult(
            status="agent_error",
            error=f"adapter 退出码 {proc.returncode}: {(proc.stderr or '')[-500:]}",
            adapter=adapter_name,
        )

    raw = (proc.stdout or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        lines = [line for line in raw.splitlines() if line.strip()]
        try:
            data = json.loads(lines[-1]) if lines else {}
        except Exception as exc:
            return AgentResult(
                status="agent_error",
                error=f"adapter stdout 不是合法 JSON: {type(exc).__name__}: {exc}",
                adapter=adapter_name,
            )

    result = AgentResult.from_dict(data if isinstance(data, dict) else {})
    issues = [item for item in result.validate() if not item.startswith("warning:")]
    if issues:
        return AgentResult(
            status="agent_error",
            error="adapter 结果校验失败: " + "；".join(issues),
            adapter=result.adapter,
            raw=result.raw,
        )
    return result


def _median(values) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else round((vals[mid - 1] + vals[mid]) / 2, 1)


def _median_ragas(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多次跑的四维分逐维取中位数；某维全缺时退回第一份。"""
    dims = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    out: Dict[str, Any] = {}
    for dim in dims:
        vals = [
            (r.get("ragas") or {}).get(dim)
            for r in rows
            if isinstance((r.get("ragas") or {}).get(dim), (int, float))
        ]
        out[dim] = _median(vals) if vals else (rows[0].get("ragas") or {}).get(dim)
    return out


def _compact_run(row: Dict[str, Any]) -> Dict[str, Any]:
    """单次跑的摘要，写进聚合行的 runs 列表，便于事后看方差。"""
    mc = row.get("must_contain_result") or {}
    return {
        "agent_status": row.get("agent_status"),
        "keyword_passed": row.get("keyword_passed"),
        "must_contain_passed": mc.get("passed"),
        "hit_rate": mc.get("hit_rate"),
        "miss": mc.get("miss"),
        "tool_count": row.get("tool_count"),
        "answer_chars": len(str(row.get("answer") or "")),
        "context_chars": len(str(row.get("contexts") or "")),
        "agent_seconds": row.get("agent_seconds"),
        "judge_valid": row.get("judge_valid"),
        "ragas": row.get("ragas"),
    }


def aggregate_repeat_rows(case: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把同一题的 N 次跑合成一行：必含按多数通过，四维分取中位数，保留每次跑的摘要。

    单题只跑一次时保持原行不变，仅补 repeat / keyword_votes / stability / criteria_version。
    重复跑是给噪声题用的：噪声地板约 4%-16%（同代码重复跑会有 1-4 题自己翻转），
    单次结果不足以判定，故 noise 题取 3 次多数通过。
    """
    if not rows:
        return {}
    repeat = len(rows)
    passed = [bool(r.get("keyword_passed")) for r in rows]
    votes = sum(passed)
    majority = votes * 2 > repeat
    if repeat == 1:
        row = dict(rows[0])
    else:
        pick = next((r for r, p in zip(rows, passed) if p == majority), rows[0])
        row = dict(pick)
        row["ragas"] = _median_ragas(rows)
        row["agent_seconds"] = _median([r.get("agent_seconds") for r in rows])
        row["init_seconds"] = _median([r.get("init_seconds") for r in rows])
        row["elapsed"] = _median([r.get("elapsed") for r in rows])
        row["runs"] = [_compact_run(r) for r in rows]
    row["repeat"] = repeat
    row["keyword_votes"] = f"{votes}/{repeat}"
    row["keyword_passed"] = majority
    row["stability"] = case.get("stability", "unmeasured")
    row["criteria_version"] = case.get("criteria_version", "original")
    row["content_digest"] = content_digest(row)
    return row


def _build_result_row(case: Dict[str, Any], agent_result: AgentResult, score: Dict[str, Any]) -> Dict[str, Any]:
    timings = agent_result.timings
    row = {
        "id": case.get("id"),
        "adapter": agent_result.adapter,
        "question": case.get("question"),
        "category": case.get("category"),
        "difficulty": case.get("difficulty", ""),
        "test_type": case.get("test_type", ""),
        "answer": agent_result.answer,
        "contexts": agent_result.contexts,
        "reference_answer": case.get("reference_answer", ""),
        "must_contain_keywords": case.get("must_contain", []),
        "must_not_contain_keywords": case.get("must_not_contain", []),
        "match_mode": case.get("match_mode", "all"),
        "eval_mode": case.get("eval_mode", "auto"),
        "ragas": score.get("ragas"),
        "must_contain_result": score.get("must_contain_result"),
        "must_not_contain_result": score.get("must_not_contain_result"),
        "matched_variant": score.get("matched_variant"),
        "variant_results": score.get("variant_results"),
        "keyword_passed": score.get("keyword_passed"),
        "keyword_reasons": score.get("keyword_reasons"),
        "citation_result": score.get("citation_result"),
        "judge_valid": score.get("judge_valid"),
        "judge_errors": score.get("judge_errors", []),
        "judge_samples": score.get("judge_samples", []),
        "judge_truncation": score.get("judge_truncation", {}),
        "keyword_judge_unavailable": score.get("keyword_judge_unavailable", {}),
        "elapsed": round(float(timings.init_seconds or 0) + float(timings.agent_seconds or 0), 1),
        "init_seconds": timings.init_seconds,
        "agent_seconds": timings.agent_seconds,
        "tool_count": len(agent_result.tool_trace or []),
        "tool_trace": agent_result.tool_trace,
        "agent_status": agent_result.status,
        "agent_error": agent_result.error,
    }
    if agent_result.status != "ok":
        row["error"] = agent_result.error or agent_result.status
    row["raw"] = agent_result.raw
    row["schema_version"] = RESULT_SCHEMA_VERSION
    row["provenance"] = {"question_sha256": case_sha256(case)}
    row["content_digest"] = content_digest(row)
    return row


def run_evalset(
    cases: List[Dict[str, Any]],
    adapter_name: str = "genshin",
    workspace: str = "",
    results_dir: str = "",
    manifest: Optional[RunManifest] = None,
    manifest_path: str = "",
    timeout_seconds: int = 300,
    force: bool = False,
    hit_rate_threshold: float = 0.8,
    repeat_override: int = 0,
) -> List[Dict[str, Any]]:
    """跑完整题集：子进程 adapter + 打分 + 每题落盘 + manifest 更新。"""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, 1):
        qid = str(case.get("id") or f"case{index}")
        result_path = out_dir / f"q_{_safe_qid(qid, index)}.json"
        print(f"[{index}/{len(cases)}] {qid} ...", flush=True)

        if not force and result_path.exists():
            try:
                cached = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if cached is not None:
                valid, reason = _validate_cached_result(cached, case)
                if valid:
                    rows.append(cached)
                    if manifest is not None:
                        manifest.set_case_status(qid, "cached")
                        if manifest_path:
                            manifest.write(manifest_path)
                    print(f"[{index}/{len(cases)}] {qid} 缓存命中", flush=True)
                    continue

        repeat = _resolve_repeat(case, repeat_override)
        run_rows: List[Dict[str, Any]] = []
        for attempt in range(1, repeat + 1):
            agent_result = run_case(adapter_name, case, workspace, timeout_seconds)
            score = score_case(
                case,
                agent_result.answer,
                agent_result.contexts,
                reference=str(case.get("reference_answer") or ""),
                eval_mode=str(case.get("eval_mode") or ""),
                hit_rate_threshold=hit_rate_threshold,
            )
            run_rows.append(_build_result_row(case, agent_result, score))
            if repeat > 1:
                last = run_rows[-1]
                print(
                    f"    第 {attempt}/{repeat} 次: status={last.get('agent_status')} "
                    f"mc={(last.get('must_contain_result') or {}).get('passed')} "
                    f"agent={last.get('agent_seconds')}s",
                    flush=True,
                )
        row = aggregate_repeat_rows(case, run_rows)
        rows.append(row)
        result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        if manifest is not None:
            manifest.set_case_status(qid, str(row.get("agent_status") or "unknown"))
            if manifest_path:
                manifest.write(manifest_path)

        print(
            f"[{index}/{len(cases)}] {qid} status={row.get('agent_status')} "
            f"init={row.get('init_seconds')}s agent={row.get('agent_seconds')}s "
            f"mc={row.get('keyword_passed')}({row.get('keyword_votes')}) "
            f"judge_valid={row.get('judge_valid')}",
            flush=True,
        )

    judge.save_cache()
    return rows
