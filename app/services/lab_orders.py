# -*- coding: utf-8 -*-
"""确定性检查单生成器（LabOrders）。

医生的“查什么”不交给 LLM 自由发挥，而是由评测器的确定性输出映射生成：
- 每个失败原因（缺少必须包含/出现禁止包含/违反系统提示词等）都生成对应的检查单；
- 永远包含 3 条基础检查：Trace 工具重放、规划意图、系统提示词工具规则；
- 检查单本身是数据，LLM 只能选择执行顺序，不能删改检查项。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _span_list(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trace:
        return []
    out: List[Dict[str, Any]] = []
    def walk(span: Dict[str, Any]) -> None:
        out.append(span)
        for child in span.get("children", []) or []:
            walk(child)
    walk(trace.get("root_span") or {})
    return out


def generate_lab_orders(
    result: Dict[str, Any],
    case: Dict[str, Any],
    trace: Optional[Dict[str, Any]],
    prompt_compliance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """根据评测结果生成强制检查单。返回顺序即建议执行顺序。"""
    orders: List[Dict[str, Any]] = []
    seen: set = set()

    def add(
        oid: str,
        category: str,
        title: str,
        question: str,
        tool: str,
        params: Dict[str, Any],
        why: str,
        evidence_type: str,
    ) -> None:
        if oid in seen:
            return
        seen.add(oid)
        orders.append({
            "id": oid,
            "category": category,
            "title": title,
            "question": question,
            "tool": tool,
            "params": params,
            "why": why,
            "evidence_type": evidence_type,
            "required": True,
        })

    reasons = [str(r) for r in (result.get("reasons") or [])]
    spans = _span_list(trace)
    tool_spans = [s for s in spans if s.get("span_type") == "tool"]
    answer = str(result.get("answer") or "")
    actual_tools = [str(t) for t in (result.get("actual_tools") or [])]

    # ---- 基础检查：任何病例都做 ----
    add("LO-001", "trace_replay", "重放 Trace 工具调用",
        "这次运行实际调用了哪些工具？每个工具的入参、状态、返回内容是什么？最终答案是否真正使用了工具返回？",
        "run_lab_check", {"lab_order_id": "LO-001"},
        "工具调用是判定“知识库真缺 vs 召回失败”的一手证据", "trace_tool_replay")
    add("LO-002", "plan_intent", "核对规划阶段意图",
        "plan 阶段是否输出了执行计划？计划里是否有 tool_calls 或 tool_skip_reason？重试了几次？",
        "run_lab_check", {"lab_order_id": "LO-002"},
        "无工具调用类失败（如 R3/X4）根因常在 plan 阶段", "plan_intent")
    add("LO-003", "prompt_rule", "读取系统提示词工具规则",
        "规划/回答系统提示词对工具调用、未收录、近似对象的硬规则原文是什么？",
        "run_lab_check", {"lab_order_id": "LO-003"},
        "判断 Agent 是否违反系统提示词必须以原文为准", "prompt_rule_excerpt")
    add("LO-EV-01", "trace_truth_audit", "Trace 真相重算与评测一致性审计",
        "从 raw Trace 独立重算：plan 文本是否规划了工具但实际没调用？是否存在 not_found/短路？答案缺失的关键词/禁词是否与评测器 reasons 一致？",
        "run_lab_check", {"lab_order_id": "LO-EV-01"},
        "评测器可能漏报或误报，必须有一次不依赖 reasons 的独立重算作为对照", "trace_truth_audit")

    # ---- 缺少必须包含 ----
    missing = []
    for r in reasons:
        if "缺少必须包含" in r:
            k = r.split("缺少必须包含：", 1)[1].strip()
            if k and k not in missing:
                missing.append(k)
    for i, k in enumerate(missing, 1):
        add(f"LO-KW-{i:02d}", "missing_keyword", f"核查关键词「{k}」的出处",
            f"关键词「{k}」在本次 Trace 的工具返回里出现过吗？在最终答案里出现过吗？知识库本身是否包含该信息？",
            "run_lab_check", {"lab_order_id": f"LO-KW-{i:02d}", "keyword": k},
            f"评测器判定缺少必须包含：{k}。医生必须先分清：知识库没有、检索没召回、还是回答阶段漏用",
            "keyword_evidence")

    # ---- 出现禁止包含 ----
    forbidden = []
    for r in reasons:
        if "出现禁止包含" in r:
            k = r.split("出现禁止包含：", 1)[1].strip()
            if k and k not in forbidden:
                forbidden.append(k)
    for i, k in enumerate(forbidden, 1):
        add(f"LO-FB-{i:02d}", "forbidden_keyword", f"追踪禁止词「{k}」的来源",
            f"禁止词「{k}」第一次出现在哪个阶段？是工具返回带入、plan/提示词误导、还是回答阶段自己生成？",
            "run_lab_check", {"lab_order_id": f"LO-FB-{i:02d}", "keyword": k},
            f"评测器判定出现禁止包含：{k}。医生需要定位污染源再开药",
            "forbidden_source_evidence")

    # ---- 工具未找到（status=not_found）----
    nf_seen = set()
    for span in tool_spans:
        if span.get("status") != "not_found":
            continue
        name = span.get("name") or ""
        args = span.get("tool_args") or {}
        key = f"{name}:{str(args)[:120]}"
        if key in nf_seen:
            continue
        nf_seen.add(key)
        i = len(nf_seen)
        add(f"LO-NF-{i:02d}", "not_found_tool", f"复检未命中工具 {name}",
            f"工具 {name} 返回 not_found。它的入参是什么？换成别名/规范名/纠错名再查，知识库能否查到？",
            "run_lab_check", {"lab_order_id": f"LO-NF-{i:02d}", "tool_name": name, "tool_args": args},
            f"Trace 显示 {name} 未命中；这是别名问题、拼写问题还是数据缺失，需要复检",
            "not_found_recheck_evidence")

    # ---- 违反系统提示词 ----
    prompt_pass = result.get("prompt_pass")
    if prompt_compliance:
        prompt_pass = prompt_compliance.get("passed")
    violations = list(result.get("prompt_violations") or [])
    if prompt_compliance and prompt_compliance.get("violations"):
        violations = list(prompt_compliance.get("violations"))
    if prompt_pass is False or violations:
        add("LO-PR-01", "prompt_violation", "核对系统提示词合规违规",
            "plan 文本里有没有 tool_skip_reason？plan_retry 执行了几次？为什么最终 tool_call_count=0？这是模型失误还是提示词/代码缺陷？",
            "run_lab_check", {"lab_order_id": "LO-PR-01"},
            "确定性合规检查标记了违反系统提示词：" + ("；".join(violations) or "非豁免场景必须调用工具"),
            "prompt_compliance_evidence")

    # ---- 没有调用任何工具 ----
    if not tool_spans:
        add("LO-ZT-01", "zero_tool", "确认零工具调用链路",
            "从 assess→rewrite→plan→answer 全链路确认：是否真的没有 tool_calls？plan_retry 是否触发？answer 是否走了 short_circuit 直接输出未收录？",
            "run_lab_check", {"lab_order_id": "LO-ZT-01"},
            "本次 Trace 没有任何 tool span，需要确认是模型没发 tool_calls 还是导出器漏记",
            "zero_tool_evidence")

    # ---- 回答完整性：最终答案 vs 工具返回 ----
    if tool_spans and answer:
        add("LO-AI-01", "answer_integrity", "比对最终答案与工具返回",
            "最终答案的内容与工具返回的重合度如何？是否存在工具查到了正确内容、但 answer 阶段仍输出“未收录/未找到”的情况？",
            "run_lab_check", {"lab_order_id": "LO-AI-01"},
            "有些失败（如 X7）是工具已命中但回答阶段未整合，必须单独检查",
            "answer_integrity_evidence")

    # ---- 无失败原因但 case 未通过：兜底检查 ----
    if not reasons:
        add("LO-GE-01", "generic_failure", "失败但评测器无具体原因",
            "评测器没有给出具体失败原因，但该题 passed=False。请重放全部事件，定位是评测器空转还是导出/执行异常。",
            "run_lab_check", {"lab_order_id": "LO-GE-01"},
            "失败兜底检查，防止空手开药", "generic_evidence")

    return orders


def format_lab_orders_for_prompt(orders: List[Dict[str, Any]]) -> str:
    lines = ["===== 强制检查单（覆盖闸门：全部完成才允许输出最终医嘱）====="]
    for o in orders:
        lines.append(
            f"- {o['id']} [{o['category']}] {o['title']}\n"
            f"    待回答问题：{o['question']}\n"
            f"    为什么查：{o['why']}"
        )
    return "\n".join(lines)
