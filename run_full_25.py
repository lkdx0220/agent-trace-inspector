# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')
from app.services.live_runner import run_live
from app.services.eval_store import list_test_cases

project_path = r'C:\Users\24701\Desktop\原神剧情\CASE-原神剧情助手-修改用'
ids = [c['case_id'] for c in list_test_cases()]

print('开始跑 25 题全量评测...')
record = run_live(project_path=project_path, case_ids=ids, run_name='25题全量评测')
print('=' * 50)
print('RUN_ID:', record.run_id)
print('通过:', record.passed_cases, '/', record.total_cases)
print('通过率:', record.pass_rate, '%')
print('失败题:', [r.case_id for r in record.results if not r.passed])
