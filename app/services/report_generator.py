# -*- coding: utf-8 -*-
"""运行分析报告生成器：基于题目语义元数据 + Trace 工具证据，输出自然语言分析报告。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests

from app.db import get_trace
from app.services.diagnoser import _api_key, _trace_summary
from app.services.eval_store import get_run, save_report
from app.services.evaluator import check_prompt_compliance, evaluate_keywords
from app.services.system_prompts import get_tool_requirement_excerpt
from schemas.eval import TestCase

METADATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "case_metadata_full.json"


def _load_metadata() -> Dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def _all_tool_texts(trace: Dict[str, Any]) -> str:
    parts = []
    def walk(span):
        if span.get("span_type") == "tool":
            preview = span.get("result_preview") or ""
            if preview:
                parts.append(preview)
        for c in span.get("children", []):
            walk(c)
    walk(trace.get("root_span") or {})
    return "\n".join(parts)


def generate_analysis_report(
    run_id: str,
    case_id: str,
    project_path: str = "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用",
) -> str:
    run = get_run(run_id)
    if not run:
        raise ValueError("Run not found")
    result = next((r for r in run.get("results", []) if r.get("case_id") == case_id), None)
    if not result:
        raise ValueError("Case result not found")

    trace = get_trace(result.get("trace_id") or "") if result.get("trace_id") else None
    meta = _load_metadata().get(case_id, {})
    rationale = meta.get("keyword_rationale", {})
    assessment_point = meta.get("assessment_point", "")

    answer = result.get("answer") or ""
    ref_answer = ""
    # find from test cases? not in run result. We can load from golden yaml? Use saved test case from DB.
    from app.services.eval_store import list_test_cases
    cases = list_test_cases()
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if case:
        ref_answer = case.get("expected_answer") or ""

    must_contain = (case or {}).get("must_contain", []) or []
    must_not_contain = (case or {}).get("must_not_contain", []) or []
    alternatives = (case or {}).get("alternatives", []) or []
    kw_result = evaluate_keywords(
        answer,
        TestCase(
            case_id=case_id,
            question=result.get("question", ""),
            expected_answer=ref_answer,
            must_contain=must_contain,
            must_not_contain=must_not_contain,
            match_mode=(case or {}).get("match_mode", "all"),
            alternatives=alternatives,
        ),
    )
    missing = kw_result["miss"]
    forbidden_in_answer = kw_result["violations"]
    variant_summary = []
    for vr in kw_result.get("variant_results", []):
        variant_summary.append({
            "name": vr.get("name", ""),
            "passed": vr.get("passed", False),
            "hit_rate": vr.get("hit_rate", 0),
            "miss": vr.get("miss", []),
            "violations": vr.get("violations", []),
        })

    tool_text = _all_tool_texts(trace) if trace else ""
    tool_summary = _trace_summary(trace) if trace else "（无 Trace）"
    prompt_compliance = check_prompt_compliance(trace, project_path) if trace else {"passed": None, "violations": [], "evidence": "无 Trace"}
    prompt_rule_excerpt = get_tool_requirement_excerpt(project_path)

    missing_rows = []
    for kw in missing:
        in_tools = kw in tool_text
        missing_rows.append({
            "keyword": kw,
            "in_tool_output": "是" if in_tools else "否",
            "reference_context": rationale.get("must_contain", {}).get(kw, ""),
            "why_required": rationale.get("must_contain", {}).get(kw, ""),
        })

    bad_rows = []
    for kw in forbidden_in_answer:
        in_tools = kw in tool_text
        bad_rows.append({
            "keyword": kw,
            "in_tool_output": "是" if in_tools else "否",
            "reference_note": rationale.get("must_not_contain", {}).get(kw, ""),
            "why_forbidden": rationale.get("must_not_contain", {}).get(kw, ""),
        })

    prompt = f"""你是 Agent 运行分析报告撰写者。你的任务是**基于证据写一份分析报告**，不做修改建议、不提出优化方案。

【本题考察点】
{assessment_point}

【用户问题】
{result.get("question", "")}

【参考答案】
{ref_answer}

【AI 最终答案】
{answer}

【答案标准/备选答案（双答案机制）】
{json.dumps(variant_summary, ensure_ascii=False, indent=2)}

【确定性失败信息】
缺失必须包含词：{json.dumps([r['keyword'] for r in missing_rows], ensure_ascii=False)}
违规出现禁止词：{json.dumps([r['keyword'] for r in bad_rows], ensure_ascii=False)}

【逐关键词证据】
缺失关键词证据：
{json.dumps(missing_rows, ensure_ascii=False, indent=2)}
违规关键词证据：
{json.dumps(bad_rows, ensure_ascii=False, indent=2)}

【Trace 工具调用摘要】
{tool_summary}

【工具返回文本片段（仅预览，可能不完整）】
{tool_text[:4000]}

【关键词语义说明】
{json.dumps(rationale, ensure_ascii=False, indent=2)}

【Agent 系统提示词合规检查（只读挂载）】
{json.dumps(prompt_compliance, ensure_ascii=False, indent=2)}

【Agent 系统提示词工具规则（只读挂载，节选）】
{prompt_rule_excerpt}

请输出一份详细的自然语言运行分析报告，结构如下：
一、总体结论（是否通过，主要问题类型）
二、缺失关键词逐项分析
  - 每个关键词：是否在工具返回中出现？如果出现但答案没有，说明“检索成功但生成遗漏”；如果没出现，说明“知识库/检索不足”
  - 结合参考答案上下文说明这个关键词对应什么要点
三、违规关键词逐项分析
  - 每个关键词：是否在工具返回中出现？如果出现，说明“检索内容污染”；如果没出现，说明“疑似训练知识/幻觉”
  - 结合题目语义说明为什么这个关键词不应该出现
四、训练知识/幻觉线索
  - 指出答案中哪些内容在工具返回中没有依据
  - 哪些内容疑似来自模型训练知识
五、系统提示词合规性
  - 如果确定性检查标记了违反系统提示词（如非豁免场景未调用工具），必须明确归为“Agent 违反系统提示词”，不要归为“题目设置问题”
六、问题归属
  - Agent 问题 / 题目设置问题 / 知识库数据问题
注意：不要给修改建议，只做分析。"""

    api_key = _api_key(project_path)
    if not api_key:
        return "缺少 DASHSCOPE_API_KEY"

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
                    {"role": "system", "content": "你是严格的 Agent 运行分析报告撰写者，只输出报告正文，不输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            },
            timeout=180,
        )
        if resp.status_code != 200:
            return f"LLM 调用失败: {resp.status_code} {resp.text[:200]}"
        report_text = resp.json()["choices"][0]["message"]["content"].strip()
        save_report(run_id, case_id, report_text)
        return report_text
    except Exception as e:
        return f"报告生成失败: {e}"
