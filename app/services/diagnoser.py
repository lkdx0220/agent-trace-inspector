# -*- coding: utf-8 -*-
"""AI 归因诊断：根据失败结果 + Trace 日志，找出为什么关键词没通过。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from app.db import get_trace
from app.services.eval_store import get_diagnosis, get_run, list_test_cases, save_diagnosis
from app.services.evaluator import check_prompt_compliance
from app.services.system_prompts import get_tool_requirement_excerpt


def _api_key(project_path: Optional[str] = None) -> str:
    env = os.environ.get("DASHSCOPE_API_KEY", "")
    if env:
        return env
    if project_path:
        env_file = Path(project_path) / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("DASHSCOPE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _trace_summary(trace: Dict[str, Any]) -> str:
    if not trace:
        return "（无 Trace）"
    lines = []
    def walk(span, depth=0):
        name = span.get("name") or span.get("span_type")
        stype = span.get("span_type")
        extra = ""
        if stype == "tool":
            args = span.get("tool_args") or {}
            extra = f" args={json.dumps(args, ensure_ascii=False)[:120]} status={span.get('status')} len={span.get('result_length')}"
            preview = (span.get("result_preview") or "")[:160].replace("\n", " ")
            extra += f" preview={preview}"
        lines.append(f"{name}{extra}")
        for c in span.get("children", []):
            walk(c, depth + 1)
    walk(trace.get("root_span") or {})
    return "\n".join(lines[:30])


def diagnose_run_case(
    run_id: str,
    case_id: str,
    project_path: str = "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用",
) -> Dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise ValueError("Run not found")
    result = next((r for r in run.get("results", []) if r.get("case_id") == case_id), None)
    if not result:
        raise ValueError("Case result not found")

    trace = get_trace(result.get("trace_id") or "") if result.get("trace_id") else None
    question = result.get("question", "")
    answer = result.get("answer", "")
    reasons = result.get("reasons", [])

    missing_keywords = [r for r in reasons if "缺少必须包含" in r]
    bad_keywords = [r for r in reasons if "出现禁止包含" in r]
    prompt_compliance = check_prompt_compliance(trace, project_path) if trace else {"passed": None, "violations": [], "evidence": "无 Trace"}
    prompt_rule_excerpt = get_tool_requirement_excerpt(project_path)
    cases = list_test_cases()
    case_meta = next((c for c in cases if c["case_id"] == case_id), None)
    alternatives = (case_meta or {}).get("alternatives", []) or []

    prompt = f"""你是一个 Agent 运行诊断专家。下面是一次知识库问答 Agent 的评测失败记录，请你阅读日志，找出根因。

【用户问题】
{question}

【AI 最终答案】
{answer}

【评测失败原因】
{json.dumps(reasons, ensure_ascii=False)}

【需要关注的缺失/违规关键词】
{json.dumps({"缺少必须包含": missing_keywords, "出现禁止包含": bad_keywords}, ensure_ascii=False)}

【Trace 工具调用摘要】
{_trace_summary(trace)}

【本题目双答案/备选答案标准】
{json.dumps(alternatives, ensure_ascii=False, indent=2)}

【系统提示词合规检查（只读挂载）】
{json.dumps(prompt_compliance, ensure_ascii=False, indent=2)}

【Agent 系统提示词工具规则（只读挂载，节选）】
{prompt_rule_excerpt}

请输出 JSON（不要其他内容）：
{{
  "root_cause": "最可能的原因，中文，1-2句",
  "evidence": ["具体工具名/事件，说明为什么支持这个判断"],
  "suggestion": "建议如何修改，中文，1-2句",
  "confidence": 0.0
}}
"""

    api_key = _api_key(project_path)
    if not api_key:
        return {"error": "缺少 DASHSCOPE_API_KEY"}

    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen3.7-max",
                "messages": [
                    {"role": "system", "content": "你是严格的 Agent 运行诊断助手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return {"error": f"LLM 调用失败: {resp.status_code} {resp.text[:200]}"}
        content = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            diag = json.loads(content[start:end])
            save_diagnosis(run_id, case_id, result.get("trace_id") or "", diag, prompt)
            diag["prompt"] = prompt
            return diag
        diag = {"root_cause": content, "evidence": [], "suggestion": "", "confidence": 0}
        save_diagnosis(run_id, case_id, result.get("trace_id") or "", diag, prompt)
        diag["prompt"] = prompt
        return diag
    except Exception as e:
        return {"error": f"诊断失败: {e}"}
