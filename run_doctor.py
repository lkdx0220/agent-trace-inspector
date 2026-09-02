# -*- coding: utf-8 -*-
"""项目医生 CLI：python run_doctor.py --run run_3054e696 --case R3"""
import argparse
import json
import sys

from app.services.doctor_tools import DEFAULT_PROJECT_PATH
from app.services.eval_store import save_prescription
from app.services.project_doctor import prescribe_run_case


def main() -> int:
    parser = argparse.ArgumentParser(description="项目医生：评测失败病例归因与处方")
    parser.add_argument("--run", required=True, help="run_id")
    parser.add_argument("--case", required=True, help="case_id，如 R3")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH, help="原项目路径")
    parser.add_argument("--model", default=None, help="医生 LLM 模型（默认 qwen3.7-max）")
    parser.add_argument("--save", action="store_true", help="保存处方到 inspector.db")
    args = parser.parse_args()

    out = prescribe_run_case(args.run, args.case, project_path=args.project, model=args.model)
    if not out.get("ok"):
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    if args.save:
        save_prescription(args.run, args.case, out, model=out.get("model") or "")

    report = out.get("report") or {}
    print(json.dumps({
        "coverage": out.get("coverage"),
        "diagnosis": report.get("diagnosis"),
        "prescriptions": report.get("prescriptions"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
