# -*- coding: utf-8 -*-
"""确定性评测引擎（核心）。

优先级：
1. 关键词（must_contain / must_not_contain）
2. 工具匹配（expected_tools）
3. 路由匹配（expected_route）
4. 从 Trace 提取耗时、工具次数等指标
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from app.services.system_prompts import get_plan_system_prompt
from schemas.eval import RunCaseResult, TestCase


def _collect_tools_from_trace(trace: Dict[str, Any]) -> List[str]:
    tools = []
    def walk(span):
        if span.get("span_type") == "tool" and span.get("name"):
            tools.append(span["name"])
        for child in span.get("children", []):
            walk(child)
    walk(trace.get("root_span") or {})
    return tools


def _collect_metrics_from_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    spans = []
    def walk(s):
        spans.append(s)
        for c in s.get("children", []):
            walk(c)
    walk(trace.get("root_span") or {})
    tools = [s for s in spans if s.get("span_type") == "tool"]
    llms = [s for s in spans if s.get("span_type") in ("llm", "answer")]
    return {
        "duration_ms": trace.get("duration_ms"),
        "tool_count": len(tools),
        "llm_count": len(llms),
    }


def _strip_negation(text: str, keyword: str) -> bool:
    idx = text.find(keyword)
    if idx == -1:
        return False
    prefix = text[max(0, idx - 6):idx]
    negations = ["不认为", "并非", "并不", "不是", "没有", "否认", "否定", "绝非", "不可能"]
    for neg in negations:
        if neg in prefix:
            return False
    return True


# 与 _run_golden_test.py 保持一致：只有这些“通用概念/描述词”允许语义兜底；
# 其余关键词（专名、任务名、书名、地名、角色名、核心设定术语）必须字符串精确命中，
# 避免 LLM 语义裁判把近似但不等同的内容误判为命中。
SEMANTIC_ALLOWED_KEYWORDS = {
    # 防幻觉/未收录类
    "未收录", "无法回答", "未明确", "作者",
    # 通用行为/概念类
    "喝酒", "自由", "守护", "现实", "见证", "从容", "命运",
    "接纳", "承认过去", "童话", "备份", "抛弃", "人类", "扮演",
    "实力", "胜利者", "对立", "环形", "塔楼", "开场动画",
}


SEMANTIC_JUDGE_MODEL = "deepseek-v4-flash"


def _deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    env_file = Path("C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用/.env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=",1)[1].strip().strip('"').strip("'")
    return ""


def semantic_keyword_batch_check(answer: str, keywords: List[str]) -> List[str]:
    """用轻量 LLM（关闭思考的 deepseek-v4-flash）批量判断答案是否在语义上覆盖关键词。

    不做白名单/正则匹配：由模型判断「并未被收录」「未找到」「不在列表中」等
    是否等价于要求的关键词，例如「未收录」。
    注意：调用方必须先按 SEMANTIC_ALLOWED_KEYWORDS 过滤，只有通用概念/描述词
    才允许进入此语义兜底；专名、书名、角色名、核心设定术语必须严格字符串命中。
    """
    if not keywords:
        return []
    ans_short = (answer or "")[:3000].strip()
    if not ans_short:
        return []

    key = _deepseek_key()
    if not key:
        return []

    numbered = "\n".join(f"{i + 1}. {kw}" for i, kw in enumerate(keywords))
    prompt = f"""判断以下【答案】是否在语义上表达了列表中每个【关键词】的含义。
允许同义改写、近义表述或等价说法，不要求出现原词。
例如：关键词「未收录」时，「并未包含」「不在列表中」「未提及」「未找到相关内容」「没有检索到」「并未被收录」等均算表达了该含义。
在知识库问答语境下，「未找到」「未检索到」与「未收录」可以视为等价表达。

【关键词列表】
{numbered}

【答案】
{ans_short}

请只输出 JSON，格式为：{{"hits": [命中的关键词编号（数字）]}}
"""

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    base_payload = {
        "model": SEMANTIC_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严谨的语义等价裁判。只输出简短 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }

    # 通过 OpenAI 格式的 thinking.type=disabled 关闭 DeepSeek 思考模式；
    # 若代理/网关不识别该参数，则退化为普通请求。
    payloads = [
        {**base_payload, "thinking": {"type": "disabled"}},
        base_payload,
    ]
    for payload in payloads:
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code != 200:
                continue
            content = resp.json()["choices"][0]["message"]["content"].strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                continue
            data = json.loads(content[start:end])
            hits = data.get("hits", []) or []
            result = []
            for item in hits:
                if isinstance(item, int) and 1 <= item <= len(keywords):
                    result.append(keywords[item - 1])
                elif isinstance(item, str) and item in keywords:
                    result.append(item)
            return result
        except Exception:
            continue
    return []


def _evaluate_single_keyword_variant(
    answer: str,
    must_contain: List[str],
    must_not_contain: List[str],
    match_mode: str,
) -> Dict[str, Any]:
    """按单个答案标准（变体）执行关键词判定。"""
    reasons = []
    hit_count = 0
    semantic_hit = []
    miss = []

    semantic_candidates = []
    for kw in must_contain:
        if not kw:
            continue
        if _strip_negation(answer or "", kw):
            hit_count += 1
        elif kw in SEMANTIC_ALLOWED_KEYWORDS:
            # 只有白名单中的通用概念/描述词允许语义兜底
            semantic_candidates.append(kw)
        else:
            # 专名/任务名/书名/角色名/核心设定术语等必须严格字符串命中
            miss.append(kw)
            reasons.append(f"缺少必须包含：{kw}")

    if semantic_candidates:
        semantic_hits = set(semantic_keyword_batch_check(answer or "", semantic_candidates))
        for kw in semantic_candidates:
            if kw in semantic_hits:
                hit_count += 1
                semantic_hit.append(kw)
            else:
                miss.append(kw)
                reasons.append(f"缺少必须包含：{kw}")

    hit_rate = hit_count / len(must_contain) if must_contain else 1.0
    if match_mode == "any":
        contains_ok = hit_count >= 1
    else:
        contains_ok = hit_rate >= 0.75

    violations = []
    for kw in must_not_contain:
        if kw and _strip_negation(answer or "", kw):
            violations.append(kw)
            reasons.append(f"出现禁止包含：{kw}")

    not_contains_ok = len(violations) == 0
    passed = contains_ok and not_contains_ok
    return {
        "passed": passed,
        "contains_ok": contains_ok,
        "not_contains_ok": not_contains_ok,
        "hit_rate": round(hit_rate, 2),
        "hit_count": hit_count,
        "semantic_hit": semantic_hit,
        "miss": miss,
        "violations": violations,
        "reasons": reasons,
    }


def _variant_score(r: Dict[str, Any]) -> tuple:
    return (
        1 if r["passed"] else 0,
        r["hit_rate"],
        -len(r["violations"]),
        -len(r["miss"]),
    )


def evaluate_keywords(answer: str, case: TestCase) -> Dict[str, Any]:
    """支持“双答案/多答案标准”：主标准 + alternatives，任一标准通过即视为关键词通过。

    返回结构：
    - passed / contains_ok / not_contains_ok / hit_rate / hit_count / semantic_hit
    - miss / violations / reasons：代表“命中的最佳变体”或“当前主标准”的判定结果
    - variant_results：每个答案变体的完整判定，供报告/诊断展示
    - matched_variant：通过时命中的变体名称；未通过时为 None
    """
    variants = [
        {
            "name": "标准答案",
            "must_contain": case.must_contain,
            "must_not_contain": case.must_not_contain,
            "match_mode": case.match_mode,
        }
    ]
    for alt in case.alternatives:
        variants.append({
            "name": alt.name or f"备选答案{len(variants)}",
            "must_contain": alt.must_contain or [],
            "must_not_contain": alt.must_not_contain or [],
            "match_mode": alt.match_mode or "all",
        })

    variant_results = []
    for v in variants:
        r = _evaluate_single_keyword_variant(
            answer,
            v["must_contain"],
            v["must_not_contain"],
            v["match_mode"],
        )
        r["name"] = v["name"]
        variant_results.append(r)

    any_passed = any(r["passed"] for r in variant_results)
    # 通过时取第一个通过变体；未通过时取综合得分最高的变体作为代表
    if any_passed:
        rep = next(r for r in variant_results if r["passed"])
    else:
        rep = max(variant_results, key=_variant_score)

    reasons = list(rep["reasons"])
    if not any_passed and len(variant_results) > 1:
        reasons.append(f"不符合任一答案标准（共 {len(variant_results)} 个变体）")

    return {
        "passed": any_passed,
        "matched_variant": rep["name"] if any_passed else None,
        "variant_results": variant_results,
        "contains_ok": rep["contains_ok"],
        "not_contains_ok": rep["not_contains_ok"],
        "hit_rate": rep["hit_rate"],
        "hit_count": rep["hit_count"],
        "semantic_hit": rep["semantic_hit"],
        "miss": rep["miss"],
        "violations": rep["violations"],
        "reasons": reasons,
    }


def _plan_texts_from_trace(trace: Dict[str, Any]) -> List[str]:
    """提取 trace 中 plan 事件的 execution_plan 文本，用于判断是否有 tool_skip_reason。"""
    texts = []
    for ev in trace.get("trace_events") or []:
        if ev.get("event") != "plan":
            continue
        data = ev.get("data") or {}
        text = data.get("execution_plan") or ""
        if text:
            texts.append(text)
    return texts


def check_prompt_compliance(trace: Dict[str, Any], project_path: Optional[str] = None) -> Dict[str, Any]:
    """基于原项目系统提示词，做只读的「工具调用硬规则」合规检查。

    规则来源：规划模块系统提示词要求非豁免问题必须调用工具；
    若未调用工具，执行报告中必须给出 tool_skip_reason。
    返回:
      passed: True/False/None（None 表示无法判定，例如未读到系统提示词或 trace 没有 plan 事件）
      violations: 违规描述列表
      evidence: 判定依据
    """
    plan_prompt = get_plan_system_prompt(project_path)
    if not plan_prompt:
        return {"passed": None, "violations": [], "evidence": "未找到规划系统提示词文件，跳过系统提示词合规检查"}

    if _collect_tools_from_trace(trace):
        return {"passed": True, "violations": [], "evidence": "存在工具调用，符合系统提示词工具要求"}

    plans = _plan_texts_from_trace(trace)
    if not plans:
        return {"passed": None, "violations": [], "evidence": "trace 中缺少 plan 事件，无法判断是否违反工具调用规则"}

    if any("tool_skip_reason" in text for text in plans):
        return {"passed": True, "violations": [], "evidence": "未调用工具但执行报告包含 tool_skip_reason，属于系统提示词允许的豁免场景"}

    return {
        "passed": False,
        "violations": ["违反系统提示词：非豁免场景必须调用工具，但实际工具调用次数为 0"],
        "evidence": "无工具调用，且 plan 执行报告中未出现 tool_skip_reason",
    }


def evaluate_trace_for_case(case: TestCase, trace: Optional[Dict[str, Any]]) -> RunCaseResult:
    result = RunCaseResult(case_id=case.case_id, question=case.question)

    if trace is None:
        result.reasons.append("没有找到对应的 Trace")
        return result

    answer = (trace.get("root_span", {}).get("output", {}) or {}).get("final_response") or ""
    metadata = trace.get("metadata") or {}
    actual_tools = _collect_tools_from_trace(trace)

    result.answer = answer
    result.trace_id = trace.get("trace_id")
    result.actual_tools = actual_tools
    result.metrics = _collect_metrics_from_trace(trace)

    # 1) 关键词
    kw = evaluate_keywords(answer, case)
    result.keyword_pass = kw["passed"]
    result.matched_variant = kw.get("matched_variant")
    result.reasons.extend(kw["reasons"])

    # 2) 工具匹配
    if case.expected_tools:
        missing = [t for t in case.expected_tools if t not in actual_tools]
        result.tool_pass = not missing
        if missing:
            result.reasons.append(f"缺少预期工具：{', '.join(missing)}")
    else:
        result.tool_pass = None

    # 3) 路由匹配
    if case.expected_route and metadata.get("execution_mode") != case.expected_route:
        result.route_pass = False
        result.reasons.append(f"路由不匹配：期望 {case.expected_route}，实际 {metadata.get('execution_mode')}")
    else:
        result.route_pass = None if not case.expected_route else True

    # 4) 系统提示词合规检查（只读挂载原项目系统提示词）
    pc = check_prompt_compliance(trace)
    result.prompt_pass = pc["passed"]
    result.prompt_violations = pc["violations"]
    result.reasons.extend(pc["violations"])

    # 5) 综合通过
    result.passed = (
        result.keyword_pass
        and (result.tool_pass is not False)
        and (result.route_pass is not False)
        and (result.prompt_pass is not False)
    )
    return result


def compute_run_summary(results: List[RunCaseResult]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    durations = [r.metrics.get("duration_ms") for r in results if r.metrics.get("duration_ms")]
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "avg_duration_ms": round(sum(durations) / len(durations), 1) if durations else 0.0,
    }
