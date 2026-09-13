# -*- coding: utf-8 -*-
"""全量评测入口（Phase 6 后统一走共享 evaluator）。

用法：
    python run_full_25.py

默认读取 evalset/genshin/cases.json 的全部题目，逐题起独立 adapter 子进程，
跑完后由 evaluator/db_bridge.py 写回 inspector.db，供 Web UI 查看。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "evalset" / "genshin" / "cases.json"


def main() -> int:
    if not CASES.exists():
        print(f"[错误] 题集不存在: {CASES}")
        return 2
    data = json.loads(CASES.read_text(encoding="utf-8"))
    ids = [str(q.get("id")) for q in data.get("questions") or [] if q.get("id")]
    if not ids:
        print("[错误] 题集为空")
        return 2

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_evalset.py"),
        "--adapter", "genshin",
        "--ids", ",".join(ids),
        "--force",
        "--save-db",
        "--name", "全量评测",
        "--timeout", "600",
    ]
    print(f"[入口] 全量评测 {len(ids)} 题 -> tools/run_evalset.py")
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
