#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""6+1 质量门：ruff + mypy + radon + vulture + pytest-cov + import-linter。

设计原则：
1. 全部走确定性工具，不调用 LLM；
2. 每条检查独立运行、独立超时、独立记录；
3. 硬失败（ruff/mypy/import-linter/测试失败）返回非 0；
4. 覆盖率、复杂度、死代码先作为 warning 暴露，不阻塞开发；
5. 结果写 quality/reports/，同时打印摘要。

用法：
    python tools/quality_gate.py
    python tools/quality_gate.py --strict
    python tools/quality_gate.py --skip pytest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows 控制台/管道默认可能是 GBK；统一成 UTF-8，避免中文和符号导致
# UnicodeEncodeError 把质量门自身搞崩。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
QUALITY_DIR = ROOT / "quality"
REPORTS_DIR = QUALITY_DIR / "reports"
TARGETS = ["app", "evaluator", "schemas", "exporter", "tools"]
RADON_TARGETS = ["app", "evaluator", "schemas", "exporter", "tools"]

TOOL_TIMEOUT = 600


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MYPYPATH"] = str(ROOT)
    return env


def _run(cmd: List[str], timeout: int = TOOL_TIMEOUT) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "duration_s": round(time.monotonic() - start, 2),
            "error": None,
        }
    except FileNotFoundError as e:
        return {"cmd": cmd, "returncode": 127, "stdout": "", "stderr": str(e), "duration_s": 0.0, "error": str(e)}
    except subprocess.TimeoutExpired:
        return {
            "cmd": cmd,
            "returncode": 124,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "duration_s": round(time.monotonic() - start, 2),
            "error": f"timeout after {timeout}s",
        }


def _py_module(name: str) -> List[str]:
    return [sys.executable, "-m", name]


def _first_lines(text: str, limit: int = 8) -> List[str]:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    return lines[:limit]


# ---------------------------------------------------------------
# 1) ruff
# ---------------------------------------------------------------

def check_ruff() -> Dict[str, Any]:
    cmd = _py_module("ruff") + ["check", "--output-format", "json"] + TARGETS
    r = _run(cmd)
    issues: List[Dict[str, Any]] = []
    if r["returncode"] not in (0, 1):
        return {"name": "ruff", "status": "error", "duration_s": r["duration_s"], "issue_count": 0,
                "summary": f"ruff 执行失败 rc={r['returncode']}", "top": _first_lines(r["stderr"])}
    raw = (r["stdout"] or "").strip()
    if raw:
        try:
            issues = json.loads(raw)
        except Exception:
            issues = []
    count = len(issues) if isinstance(issues, list) else 0
    top = []
    if isinstance(issues, list):
        for item in issues[:10]:
            top.append(
                f"{item.get('filename')}:{item.get('location', {}).get('row', '?')} "
                f"{item.get('code', '')} {item.get('message', '')}"
            )
    return {
        "name": "ruff",
        "status": "pass" if count == 0 else "fail",
        "duration_s": r["duration_s"],
        "issue_count": count,
        "summary": f"{count} 个 lint 问题",
        "top": top,
    }


# ---------------------------------------------------------------
# 2) mypy
# ---------------------------------------------------------------

MYPY_ERR_RE = re.compile(r": error: ")


def check_mypy() -> Dict[str, Any]:
    cmd = _py_module("mypy") + [
        "--show-error-codes",
        "--no-error-summary",
        "--explicit-package-bases",
    ] + TARGETS
    r = _run(cmd)
    text = (r["stdout"] or "") + "\n" + (r["stderr"] or "")
    errors = [line for line in text.splitlines() if MYPY_ERR_RE.search(line)]
    if r["returncode"] not in (0, 1):
        return {"name": "mypy", "status": "error", "duration_s": r["duration_s"], "issue_count": len(errors),
                "summary": f"mypy 执行失败 rc={r['returncode']}", "top": _first_lines(text)}
    return {
        "name": "mypy",
        "status": "pass" if not errors else "fail",
        "duration_s": r["duration_s"],
        "issue_count": len(errors),
        "summary": f"{len(errors)} 个类型检查错误",
        "top": errors[:10],
    }


# ---------------------------------------------------------------
# 3) radon
# ---------------------------------------------------------------

def check_radon() -> Dict[str, Any]:
    cmd = _py_module("radon") + ["cc", "-s", "-j"] + RADON_TARGETS
    r = _run(cmd)
    if r["returncode"] not in (0, 1):
        return {"name": "radon", "status": "error", "duration_s": r["duration_s"], "issue_count": 0,
                "summary": f"radon 执行失败 rc={r['returncode']}", "top": _first_lines(r["stderr"])}
    try:
        data = json.loads(r["stdout"] or "{}")
    except Exception:
        return {"name": "radon", "status": "error", "duration_s": r["duration_s"], "issue_count": 0,
                "summary": "radon JSON 解析失败", "top": _first_lines(r["stdout"])}
    buckets = {"C": 0, "D": 0, "E": 0, "F": 0}
    hotspots: List[Tuple[int, str]] = []
    current_keys: set[str] = set()
    for path, blocks in (data or {}).items():
        norm_path = str(path).replace("\\", "/")
        for b in blocks or []:
            rank = str(b.get("rank") or "A")
            if rank in buckets:
                buckets[rank] += 1
                if rank in {"D", "E", "F"}:
                    key = f"{norm_path}::{b.get('name', '')}"
                    current_keys.add(key)
                hotspots.append((int(b.get("complexity") or 0), f"{path}:{b.get('lineno', '?')} {b.get('name', '')} rank={rank} cc={b.get('complexity')}"))
    hotspots.sort(reverse=True)
    # C 级（11-20）属于可接受范围，不作为 warning；只监控 D/E/F。
    total_bad = buckets["D"] + buckets["E"] + buckets["F"]

    # 复杂度基线：允许已有 D/E/F 债务继续以 warning 暴露，但禁止新增。
    baseline_path = QUALITY_DIR / "radon_baseline.json"
    baseline_keys: set[str] = set()
    if baseline_path.exists():
        try:
            baseline_keys = set(json.loads(baseline_path.read_text(encoding="utf-8")).get("functions") or [])
        except Exception:
            baseline_keys = set()
    new_keys = sorted(current_keys - baseline_keys)
    if new_keys:
        status = "fail"
        summary = f"新增 {len(new_keys)} 个高复杂度函数（D/E/F）；当前 C={buckets['C']} D={buckets['D']} E={buckets['E']} F={buckets['F']}"
    else:
        status = "warn" if total_bad else "pass"
        summary = f"C={buckets['C']} D={buckets['D']} E={buckets['E']} F={buckets['F']}（只把 D/E/F 视为高复杂度债务；C 级不阻塞）"
    return {
        "name": "radon",
        "status": status,
        "duration_s": r["duration_s"],
        "issue_count": total_bad,
        "summary": summary,
        "buckets": buckets,
        "new_high_complexity": new_keys,
        "current_high_complexity": sorted(current_keys),
        "top": [h[1] for h in hotspots[:10]],
    }


# ---------------------------------------------------------------
# 4) vulture
# ---------------------------------------------------------------

def check_vulture() -> Dict[str, Any]:
    cmd = _py_module("vulture") + TARGETS + [
        "--min-confidence", "80",
        "--exclude", "quality,.venv,__pycache__,data,runs",
    ]
    r = _run(cmd)
    text = (r["stdout"] or "") + "\n" + (r["stderr"] or "")
    # vulture 正常退出码是 0；有发现时也可能为 1/3。
    findings = [line for line in text.splitlines() if "confidence" in line]
    return {
        "name": "vulture",
        "status": "warn" if findings else "pass",
        "duration_s": r["duration_s"],
        "issue_count": len(findings),
        "summary": f"{len(findings)} 个疑似死代码/未使用符号（warning）",
        "top": findings[:10],
    }


COVERAGE_WARN = 30.0
COVERAGE_FAIL = 25.0


# ---------------------------------------------------------------
# 5) pytest + coverage
# ---------------------------------------------------------------

def check_pytest() -> Dict[str, Any]:
    cov_json = REPORTS_DIR / "coverage.json"
    cmd = _py_module("pytest") + [
        "-q",
        "-p", "no:cacheprovider",
        "--ignore-glob=pytest-cache-files-*",
        "--cov=app",
        "--cov=evaluator",
        "--cov=schemas",
        "--cov=exporter",
        "--cov-report=term-missing",
        f"--cov-report=json:{cov_json}",
    ]
    r = _run(cmd)
    text = (r["stdout"] or "") + "\n" + (r["stderr"] or "")
    percent: Optional[float] = None
    try:
        (REPORTS_DIR / "pytest_full.txt").write_text(text, encoding="utf-8")
    except Exception:
        pass
    if cov_json.exists():
        try:
            cov = json.loads(cov_json.read_text(encoding="utf-8"))
            percent = float((cov.get("totals") or {}).get("percent_covered") or 0.0)
        except Exception:
            percent = None
    if r["returncode"] == 0:
        if percent is None:
            status = "warn"
            summary = "测试通过，但无法读取覆盖率"
        elif percent < COVERAGE_FAIL:
            status = "fail"
            summary = f"测试通过，但覆盖率 {percent:.2f}% < fail 线 {COVERAGE_FAIL}%"
        elif percent < COVERAGE_WARN:
            status = "warn"
            summary = f"测试通过，覆盖率 {percent:.2f}%（低于 warn 线 {COVERAGE_WARN}%）"
        else:
            status = "pass"
            summary = f"测试通过，覆盖率={percent:.2f}%"
    elif r["returncode"] == 5:
        status = "warn"
        summary = "没有收集到测试（pytest exit 5）；先用 warning 暴露"
    else:
        status = "fail"
        summary = f"测试失败 rc={r['returncode']}"
    return {
        "name": "pytest",
        "status": status,
        "duration_s": r["duration_s"],
        "issue_count": 0 if status == "pass" else 1,
        "summary": summary,
        "coverage_percent": percent,
        "top": _first_lines(text, 12),
    }


# ---------------------------------------------------------------
# 6) import-linter
# ---------------------------------------------------------------

def check_import_linter() -> Dict[str, Any]:
    config = QUALITY_DIR / "importlinter.ini"
    exe = shutil.which("lint-imports")
    if exe:
        cmd = [exe, "--config", str(config)]
    else:
        cmd = _py_module("importlinter.cli") + ["--config", str(config)]
    r = _run(cmd)
    text = (r["stdout"] or "") + "\n" + (r["stderr"] or "")
    match = re.search(r"Contracts:\s*\d+\s+kept,\s*(\d+)\s+broken", text)
    broken_count = int(match.group(1)) if match else 0
    if r["returncode"] == 0 and broken_count == 0:
        status = "pass"
        summary = "架构契约通过"
    elif broken_count > 0:
        status = "fail"
        summary = f"架构契约被破坏：{broken_count} 条"
    else:
        status = "error"
        summary = f"import-linter 执行失败 rc={r['returncode']}"
    return {
        "name": "import-linter",
        "status": status,
        "duration_s": r["duration_s"],
        "issue_count": 0 if status == "pass" else 1,
        "summary": summary,
        "top": _first_lines(text, 12),
    }


CHECKS = {
    "ruff": check_ruff,
    "mypy": check_mypy,
    "radon": check_radon,
    "vulture": check_vulture,
    "pytest": check_pytest,
    "import-linter": check_import_linter,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="6+1 确定性质量门")
    parser.add_argument("--strict", action="store_true", help="warn 也视为失败")
    parser.add_argument("--skip", action="append", default=[], help="跳过某条检查，可重复")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for name, fn in CHECKS.items():
        if name in args.skip:
            results[name] = {"name": name, "status": "skipped", "duration_s": 0.0, "issue_count": 0,
                             "summary": "已跳过", "top": []}
            continue
        print(f"[quality-gate] running {name} ...", flush=True)
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"name": name, "status": "error", "duration_s": 0.0, "issue_count": 0,
                             "summary": f"{type(e).__name__}: {e}", "top": []}

    statuses = [r.get("status") for r in results.values()]
    if any(s in {"fail", "error"} for s in statuses):
        overall = "fail"
    elif any(s == "warn" for s in statuses):
        overall = "warn"
    else:
        overall = "pass"
    if args.strict and overall == "warn":
        overall = "fail"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "repo": str(ROOT),
        "strict": bool(args.strict),
        "overall": overall,
        "checks": results,
    }
    latest = REPORTS_DIR / "quality_gate_latest.json"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = REPORTS_DIR / f"quality_gate_{stamp}.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stamped.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"QUALITY GATE: {overall.upper()}  ({stamp})")
    print("=" * 72)
    for name, r in results.items():
        print(f"[{r.get('status', '?'):>5}] {name:<14} {r.get('summary', '')}")
        for line in (r.get("top") or [])[:5]:
            print(f"        - {line}")
    print("-" * 72)
    print(f"report: {latest}")

    if overall == "pass":
        return 0
    if overall == "warn":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
