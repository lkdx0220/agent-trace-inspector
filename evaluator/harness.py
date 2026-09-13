# -*- coding: utf-8 -*-
"""共享 harness：子进程跑 adapter、超时 kill、打分、落盘、manifest 更新。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluator.contract import AgentResult, AgentTimings
from evaluator.manifest import RunManifest, sha256_text
from evaluator.scorers import judge, score_case

RESULT_SCHEMA_VERSION = "2"


def content_digest(result: Dict[str, Any]) -> str:
    core = {
        "id": result.get("id"),
        "question": result.get("question"),
        "category": result.get("category"),
        "answer": result.get("answer"),
        "ragas": result.get("ragas"),
        "must_contain_result": result.get("must_contain_result"),
        "must_not_contain_result": result.get("must_not_contain_result"),
        "citation_result": result.get("citation_result"),
        "elapsed": result.get("elapsed"),
        "init_seconds": result.get("init_seconds"),
        "agent_seconds": result.get("agent_seconds"),
        "judge_valid": result.get("judge_valid"),
        "judge_errors": result.get("judge_errors"),
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
    if adapter_name != "genshin":
        return AgentResult(status="agent_error", error=f"未知 adapter: {adapter_name}", adapter=adapter_name)
    payload = {
        "case_id": str(case.get("id") or ""),
        "question": str(case.get("question") or ""),
        "context": str(case.get("context") or ""),
    }
    run_cwd = cwd or str(Path(__file__).resolve().parents[1])
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "evaluator.adapters.genshin"],
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
            adapter="",
            timings=AgentTimings(init_seconds=0.0, agent_seconds=round(time.time() - started, 1)),
        )

    if proc.returncode != 0 and not (proc.stdout or "").strip():
        return AgentResult(
            status="agent_error",
            error=f"adapter 退出码 {proc.returncode}: {(proc.stderr or '')[-500:]}",
            adapter="",
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
                adapter="",
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


def _build_result_row(case: Dict[str, Any], agent_result: AgentResult, score: Dict[str, Any]) -> Dict[str, Any]:
    timings = agent_result.timings
    row = {
        "id": case.get("id"),
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
        "citation_result": score.get("citation_result"),
        "judge_valid": score.get("judge_valid"),
        "judge_errors": score.get("judge_errors", []),
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
) -> List[Dict[str, Any]]:
    """跑完整题集：子进程 adapter + 打分 + 每题落盘 + manifest 更新。"""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, 1):
        qid = str(case.get("id") or f"case{index}")
        result_path = out_dir / f"q_{qid}.json"
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

        agent_result = run_case(adapter_name, case, workspace, timeout_seconds)
        score = score_case(
            case,
            agent_result.answer,
            agent_result.contexts,
            reference=str(case.get("reference_answer") or ""),
            eval_mode=str(case.get("eval_mode") or ""),
        )
        row = _build_result_row(case, agent_result, score)
        rows.append(row)
        result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        if manifest is not None:
            manifest.set_case_status(qid, str(row.get("agent_status") or "unknown"))
            if manifest_path:
                manifest.write(manifest_path)

        print(
            f"[{index}/{len(cases)}] {qid} status={row.get('agent_status')} "
            f"init={row.get('init_seconds')}s agent={row.get('agent_seconds')}s "
            f"mc={row.get('must_contain_result', {}).get('passed')} "
            f"judge_valid={row.get('judge_valid')}",
            flush=True,
        )

    judge.save_cache()
    return rows
