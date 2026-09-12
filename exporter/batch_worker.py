# -*- coding: utf-8 -*-
"""批量导出 Worker：在独立子进程中一次加载 Agent，顺序跑多题。

这里不 import 观测端的 app.* / services.*（避免和项目的 app 包同名冲突），
只负责调用 Agent、写出 Trace JSON，然后退出。

用法（在 agent-trace-inspector 目录）：
    python -m exporter.batch_worker --project-path ... --cases /tmp/cases.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

from exporter.batch_exporter import AgentBatchRunner
from exporter.safe_paths import ensure_cases_file_path, ensure_project_path_local, ensure_trace_out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--cases", required=True, help="JSON 文件：list of {case_id, question, context, out_path}")
    args = parser.parse_args()

    project_path = ensure_project_path_local(args.project_path)
    cases_path = ensure_cases_file_path(args.cases)
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    runner = AgentBatchRunner(project_path)
    try:
        for idx, case in enumerate(cases, 1):
            case_id = case["case_id"]
            question = case["question"]
            context = case.get("context", "")
            out_path = ensure_trace_out_path(case["out_path"])
            print(f"[worker] START {case_id} | {question[:50]}", flush=True)
            t0 = time.time()
            runner.run_case(question, context=context, out_path=out_path)
            print(f"[worker] DONE {case_id} | {time.time() - t0:.1f}s", flush=True)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
