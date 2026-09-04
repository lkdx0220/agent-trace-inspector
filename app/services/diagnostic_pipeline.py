# -*- coding: utf-8 -*-
"""项目医生 · 确定性诊断流水线（迭代后架构）。

设计目标：
- “查什么”不再由 LLM 决定，而是由 8 个固定阶段探针一次性完成；
- 每个阶段只读探查，产出结构化证据，写入 evidence_by_order；
- FactSheet 从阶段证据中独立重算确定事实；
- CausalResolver 用 first-failure-wins 的因果梯子在 FactSheet 上做归因；
- LLM 只在判定层内做解释与开处方，不再负责采证/归因。

因果梯子：
input → routing → planning → knowledge_gap → query_alias → recall_snippet
      → answer_composition → answer_contamination → evaluator → version
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.services.doctor_tools import (
    DEFAULT_PROJECT_PATH,
    _events,
    _extract_plan_tool_intents,
    _find_where,
    _plan_signal,
    _short_circuit_answer,
    _trace_truth_audit,
    _tool_text,
    collect_tool_spans,
    inspect_aliases_multi,
    kb_probe_contains,
    knowledge_probe_batch,
    now_iso,
    raw_kb_contains_multi,
    routing_probe,
)
from app.services.source_snapshot import trace_snapshot_status
from app.services.system_prompts import get_answer_system_prompt, get_plan_system_prompt

STAGE_ORDER: List[Tuple[str, str, str, str]] = [
    ("LO-STG-01", "stage_input", "输入与评测标准重放", "题目、评测标准、最终答案与评测器判定的完整重放"),
    ("LO-STG-02", "stage_routing", "路由阶段审计", "route 事件的 intent_labels / injected_tools 与当前路由硬规则对照"),
    ("LO-STG-03", "stage_planning", "规划阶段审计", "plan 文本、工具意图、tool_call_names、重试与豁免原因"),
    ("LO-STG-04", "stage_tool_execution", "工具执行审计", "实际 tool span 的入参、状态、返回长度、not_found/error"),
    ("LO-STG-05", "stage_knowledge_truth", "知识库真值核对", "逐关键词核对工具返回/最终答案/知识库检索/原始数据/别名映射"),
    ("LO-STG-06", "stage_answer", "回答阶段审计", "最终答案与工具返回的重合度、短路串、禁词来源"),
    ("LO-STG-07", "stage_version", "版本快照核对", "Trace 源码快照与当前工作区的文件哈希/git 状态对比"),
    ("LO-STG-08", "stage_evaluator", "评测器一致性审计", "从 raw Trace 独立重算，发现评测器漏报/误报"),
]

CONSISTENCY_KINDS = {
    "routing_failure", "plan_output_failure", "knowledge_gap", "query_alias_failure",
    "recall_snippet_failure", "answer_composition", "answer_contamination",
    "prompt_violation", "alias_mapping", "evaluator_error", "version_unknown", "other",
}


def generate_pipeline_orders(
    result: Dict[str, Any],
    case: Dict[str, Any],
    trace: Optional[Dict[str, Any]],
    prompt_compliance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """固定 8 阶段检查单。所有病例永远执行同一套阶段，不再按症状临时生成。"""
    orders: List[Dict[str, Any]] = []
    for oid, category, title, why in STAGE_ORDER:
        orders.append({
            "id": oid,
            "category": category,
            "title": title,
            "question": why,
            "tool": "run_stage",
            "params": {},
            "why": why,
            "evidence_type": category,
            "required": True,
        })
    return orders


def _evidence(oid: str, category: str, summary: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "order_id": oid,
        "status": "completed",
        "category": category,
        "summary": summary,
        "data": data,
        "created_at": now_iso(),
    }


def _route_events(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in _events(trace) if e.get("event") == "route"]


def _tool_calls_signal(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans = collect_tool_spans(trace)
    calls = []
    for s in spans:
        full = s.get("result_full") or s.get("result_preview") or ""
        calls.append({
            "span_id": s.get("span_id"),
            "name": s.get("name"),
            "status": s.get("status"),
            "args": s.get("tool_args") or {},
            "result_length": s.get("result_length"),
            "preview": str(full)[:300],
        })
    return calls


def _run_stage(order: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    oid = order["id"]
    category = order["category"]
    trace = ctx.get("trace")
    result = ctx.get("result") or {}
    case = ctx.get("case") or {}
    answer = str(result.get("answer") or "")
    question = str(result.get("question") or case.get("question") or "")
    project_path = str(ctx.get("project_path") or DEFAULT_PROJECT_PATH)

    try:
        if category == "stage_input":
            data = {
                "question": question,
                "case_category": case.get("category"),
                "difficulty": case.get("difficulty"),
                "answer": answer[:1000],
                "answer_chars": len(answer),
                "must_contain": case.get("must_contain") or [],
                "must_not_contain": case.get("must_not_contain") or [],
                "match_mode": case.get("match_mode"),
                "alternatives": case.get("alternatives") or [],
                "reasons": result.get("reasons") or [],
                "passed": result.get("passed"),
                "trace_id": result.get("trace_id"),
            }
            return _evidence(oid, category, f"题目与评测标准已重放；答案长度 {len(answer)}", data)

        if category == "stage_routing":
            route_events = _route_events(trace)
            route = (route_events[-1].get("data") or {}) if route_events else {}
            intent_labels = [str(x) for x in (route.get("intent_labels") or [])]
            injected = [str(x) for x in (route.get("injected_tools") or [])]
            probe = routing_probe(project_path, question)
            pdata = probe.get("data") or {}
            hard_rule_hit = bool(pdata.get("hard_rule_hit"))
            required = [str(x) for x in (pdata.get("required_tools") or [])]
            missing_required = [t for t in required if t not in injected]
            data = {
                "route_event_seen": bool(route_events),
                "route_event_count": len(route_events),
                "intent_labels": intent_labels,
                "injected_tools": injected,
                "current_code_hard_rule_hit": hard_rule_hit,
                "current_code_required_tools": required,
                "missing_required_tools": missing_required,
                "routing_probe": probe,
                "version_note": "required_tools 来自当前 intent_router.py；是否适用于 Trace 时刻由 LO-STG-07 判定",
            }
            summary = (
                f"route 事件 {len(route_events)} 个；intent={intent_labels or '无'}；"
                f"injected {len(injected)} 个工具；当前代码硬规则命中={hard_rule_hit}；"
                f"缺失必要工具={missing_required or '无'}"
            )
            return _evidence(oid, category, summary, data)

        if category == "stage_planning":
            signal = _plan_signal(trace)
            plan_texts = signal.get("execution_plans") or []
            plan_intents = _extract_plan_tool_intents(plan_texts)
            plan_prompt_head = ""
            answer_prompt_head = ""
            try:
                plan_prompt_head = (get_plan_system_prompt(project_path) or "")[:2200]
            except Exception:
                pass
            try:
                answer_prompt_head = (get_answer_system_prompt(project_path) or "")[:2200]
            except Exception:
                pass
            data = {
                "plan_events": signal.get("plan_events"),
                "plan_retry_events": signal.get("plan_retry_events"),
                "execution_plans": plan_texts,
                "tool_call_names": signal.get("tool_call_names"),
                "tool_skip_reason": signal.get("tool_skip_reason"),
                "plan_intents": plan_intents,
                "plan_intent_mismatch_hint": bool(plan_intents) and not signal.get("tool_call_names"),
                "plan_prompt_head": plan_prompt_head,
                "answer_prompt_head": answer_prompt_head,
                "version_note": "plan/answer 提示词为当前工作区版本；是否等于 Trace 时刻版本由 LO-STG-07 判定",
            }
            summary = (
                f"plan 事件 {signal.get('plan_events')} 次，plan_retry {signal.get('plan_retry_events')} 次；"
                f"文本工具意图={plan_intents or '无'}；结构化 tool_call_names={signal.get('tool_call_names') or '无'}；"
                f"skip_reason={signal.get('tool_skip_reason') or '无'}"
            )
            return _evidence(oid, category, summary, data)

        if category == "stage_tool_execution":
            calls = _tool_calls_signal(trace)
            not_found = [c for c in calls if c.get("status") == "not_found"]
            errors = [c for c in calls if c.get("status") in {"error", "failed"}]
            names = [str(c.get("name")) for c in calls if c.get("name")]
            data = {
                "tool_count": len(calls),
                "tool_calls": calls,
                "tool_names": names,
                "not_found_tools": not_found,
                "error_tools": errors,
                "zero_tool_no_skip": len(calls) == 0,
            }
            summary = (
                f"实际工具 span {len(calls)} 个；" +
                ("；".join(f"{c['name']}={c['status']}" for c in calls) or "无工具调用") +
                f"；not_found {len(not_found)} 个，error {len(errors)} 个"
            )
            return _evidence(oid, category, summary, data)

        if category == "stage_knowledge_truth":
            missing = [str(k) for k in (case.get("must_contain") or []) if str(k).strip()]
            forbidden = [str(k) for k in (case.get("must_not_contain") or []) if str(k).strip()]
            not_found_tools = [c for c in _tool_calls_signal(trace) if c.get("status") == "not_found"]

            # 批量检索：每个缺失词构造一条“题目 + 关键词”查询，一条子进程完成。
            queries: List[str] = []
            for kw in missing:
                q = f"{question} {kw}".strip()
                if q not in queries:
                    queries.append(q)
            nf_terms: List[str] = []
            for c in not_found_tools:
                args = c.get("args") or {}
                for v in args.values():
                    if isinstance(v, str) and v.strip():
                        nf_terms.append(v.strip())
                        if v.strip() not in queries:
                            queries.append(v.strip())
                        break

            raw_keywords = missing + forbidden + nf_terms
            kb_probe = knowledge_probe_batch(project_path, queries, top_k=5) if queries else {"ok": True, "data": {"queries": {}}}
            raw_probe = raw_kb_contains_multi(project_path, raw_keywords) if raw_keywords else {"ok": True, "data": {}}
            alias_probe = inspect_aliases_multi(
                project_path,
                [c.get("args") and next(iter(c.get("args").values()), "") for c in not_found_tools if c.get("name") == "query_character"],
            ) if any(c.get("name") == "query_character" for c in not_found_tools) else {"ok": True, "data": {}}

            kb_queries = (kb_probe.get("data") or {}).get("queries") or {}
            raw_data = (raw_probe.get("data") or {})
            alias_data = (alias_probe.get("data") or {})

            # 当前提示词/规则视角：用于禁词来源判断，版本适用性由 STG-07 单独说明。
            plan_prompt_text = ""
            answer_prompt_text = ""
            try:
                plan_prompt_text = get_plan_system_prompt(project_path)
                answer_prompt_text = get_answer_system_prompt(project_path)
            except Exception:
                pass

            keyword_results: Dict[str, Any] = {}
            for kw in missing:
                where = _find_where(kw, trace, answer)
                q = f"{question} {kw}".strip()
                raw_out = kb_queries.get(q)
                probe_like = {"ok": kb_probe.get("ok"), "data": {"queries": {q: raw_out}}}
                kb_hit = kb_probe_contains(probe_like, kw)
                raw_entry = raw_data.get(kw) or {}
                keyword_results[kw] = {
                    "where": where,
                    "kb_hit": kb_hit,
                    "raw_contains": bool(raw_entry.get("contains")),
                    "raw_hits": raw_entry.get("hits") or [],
                    "kb_query": q,
                    "kb_output_is_error": isinstance(raw_out, str) and raw_out.startswith("ERROR"),
                }

            forbidden_results: Dict[str, Any] = {}
            for kw in forbidden:
                where = _find_where(kw, trace, answer)
                raw_entry = raw_data.get(kw) or {}
                forbidden_results[kw] = {
                    "where": where,
                    "raw_contains": bool(raw_entry.get("contains")),
                    "raw_hits": raw_entry.get("hits") or [],
                    "prompt_current_contains": (kw in plan_prompt_text) or (kw in answer_prompt_text),
                }

            nf_results: List[Dict[str, Any]] = []
            for c in not_found_tools:
                args = c.get("args") or {}
                term = ""
                for v in args.values():
                    if isinstance(v, str) and v:
                        term = str(v)
                        break
                raw_entry = raw_data.get(term) or {}
                alias_entry = alias_data.get(term) or {}
                nf_results.append({
                    "tool_name": c.get("name"),
                    "term": term,
                    "raw_contains": bool(raw_entry.get("contains")),
                    "raw_hits": raw_entry.get("hits") or [],
                    "alias_canonical": alias_entry.get("canonical"),
                    "alias_variants": alias_entry.get("variants") or [],
                })

            data = {
                "keyword_results": keyword_results,
                "forbidden_results": forbidden_results,
                "not_found_results": nf_results,
                "kb_probe": kb_probe,
                "raw_probe": raw_probe,
                "alias_probe": alias_probe,
            }
            missing_summary = "；".join(
                f"{kw}(tool={r['where'].get('tool_results')},answer={r['where'].get('final_answer')},kb={r.get('kb_hit')},raw={r.get('raw_contains')})"
                for kw, r in keyword_results.items()
            ) or "无缺失词"
            forbidden_summary = "；".join(
                f"{kw}(answer={r['where'].get('final_answer')},tool={r['where'].get('tool_results')},prompt={r.get('prompt_current_contains')})"
                for kw, r in forbidden_results.items()
            ) or "无禁词"
            return _evidence(oid, category, f"知识库真值核对完成。缺失词: {missing_summary}。禁词: {forbidden_summary}", data)

        if category == "stage_answer":
            corpus = _tool_text(collect_tool_spans(trace))
            if corpus and answer:
                def grams(s: str) -> set:
                    return set(s[i:i + 4] for i in range(max(0, len(s) - 3)))
                overlap = grams(answer) & grams(corpus)
                ratio = len(overlap) / max(1, len(grams(answer)))
            else:
                ratio = 0.0
            answer_ev = next((e for e in _events(trace) if e.get("event") == "answer_end"), None)
            short = _short_circuit_answer(answer)
            data = {
                "answer_chars": len(answer),
                "tool_corpus_chars": len(corpus),
                "4gram_overlap_ratio": round(float(ratio), 4),
                "short_circuit_pattern": short,
                "answer_end_event": {k: (str(v)[:200]) for k, v in ((answer_ev or {}).get("data") or {}).items()},
            }
            summary = f"答案 {len(answer)} 字，与工具返回 4-gram 重合率={ratio:.2%}，短路串={short or '无'}"
            return _evidence(oid, category, summary, data)

        if category == "stage_version":
            status = trace_snapshot_status(trace, project_path)
            data = status
            known = status.get("trace_snapshot_known")
            changed = status.get("changed_files") or []
            summary = (
                f"Trace 源码快照 known={known}；"
                f"prompt_clean={status.get('prompt_clean')}；code_clean={status.get('code_clean')}；"
                f"不一致文件 {len(changed)} 个：{changed[:8]}"
            )
            return _evidence(oid, category, summary, data)

        if category == "stage_evaluator":
            audit = _trace_truth_audit(ctx)
            data = {
                "audit": audit,
                "reasons": result.get("reasons") or [],
                "passed": result.get("passed"),
                "keyword_pass": result.get("keyword_pass"),
                "tool_pass": result.get("tool_pass"),
                "route_pass": result.get("route_pass"),
                "prompt_pass": result.get("prompt_pass"),
                "prompt_violations": result.get("prompt_violations") or [],
                "actual_tools": result.get("actual_tools") or [],
            }
            return _evidence(oid, category, audit.get("summary") or "评测器一致性审计完成", data)

        return _evidence(oid, category, f"未知阶段 {category}", {"error": f"未知阶段 {category}"})

    except Exception as e:
        return {
            "ok": True,
            "order_id": oid,
            "status": "completed_with_error",
            "category": category,
            "summary": f"阶段执行异常: {e}",
            "data": {"error": repr(e)},
            "created_at": now_iso(),
        }


def run_diagnostic_pipeline(
    ctx: Dict[str, Any],
    evidence_by_order: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """执行全部 8 阶段探针并写入证据库。返回 orders/evidence/coverage。"""
    evidence_by_order = evidence_by_order if evidence_by_order is not None else {}
    orders = ctx.get("lab_orders") or generate_pipeline_orders(
        ctx.get("result") or {}, ctx.get("case") or {}, ctx.get("trace")
    )
    for order in orders:
        ev = _run_stage(order, ctx)
        evidence_by_order.setdefault(order["id"], []).append(ev)
    done = sum(1 for o in orders if any(e.get("ok") for e in evidence_by_order.get(o["id"], [])))
    return {
        "orders": orders,
        "evidence_by_order": evidence_by_order,
        "coverage": {
            "total_orders": len(orders),
            "completed_orders": done,
            "missing_orders": [],
            "coverage": round(done / len(orders), 4) if orders else 1.0,
            "complete": done == len(orders),
        },
    }


def build_fact_sheet(
    ctx: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """从 8 阶段证据独立重算 FactSheet。

    该表是结论一致性闸门的唯一权威依据；LLM 不能通过文字改写它。
    """
    trace = ctx.get("trace")
    result = ctx.get("result") or {}
    case = ctx.get("case") or {}
    answer = str(result.get("answer") or "")
    audit = _trace_truth_audit(ctx)

    sheet: Dict[str, Any] = {
        "answer": answer,
        "actual_tools": [str(x) for x in (audit.get("actual_tools") or [])],
        "tool_count": len(collect_tool_spans(trace)),
        "plan_intents": [str(x) for x in (audit.get("plan_intents") or [])],
        "plan_tool_call_names_empty": not bool(audit.get("plan_tool_call_names")),
        "answer_short_circuit": bool(audit.get("answer_short_circuit")),
        "not_found_tools": audit.get("not_found_tools") or [],
        "evaluator_error_detected": bool(audit.get("evaluator_discrepancies")),
        "zero_tool_no_skip": len(collect_tool_spans(trace)) == 0,
        "prompt_trace_version_known": False,
        "trace_source_snapshot_known": False,
        "prompt_snapshot_clean": False,
        "code_snapshot_clean": False,
        "trace_version_clean": False,
        "tool_output_contains": {},
        "final_answer_contains": {},
        "kb_probe_contains": {},
        "raw_data_contains": {},
        "alias_map_contains": {},
        "prompt_current_contains": {},
        "routing_event_seen": False,
        "routing_hard_rule_hit": False,
        "routing_expected_tools": [],
        "routing_actual_tools": [],
        "routing_missing_required_tools": [],
        "contamination_source_found": False,
        "stage_summaries": {},
    }

    for oid, evs in evidence_by_order.items():
        for ev in evs:
            if not ev.get("ok"):
                continue
            cat = ev.get("category") or ""
            data = ev.get("data") or {}
            sheet["stage_summaries"][oid] = ev.get("summary") or ""

            if cat == "stage_routing":
                sheet["routing_event_seen"] = bool(data.get("route_event_seen"))
                sheet["routing_hard_rule_hit"] = bool(data.get("current_code_hard_rule_hit"))
                sheet["routing_expected_tools"] = [str(x) for x in (data.get("current_code_required_tools") or [])]
                sheet["routing_actual_tools"] = [str(x) for x in (data.get("injected_tools") or [])]
                sheet["routing_missing_required_tools"] = [str(x) for x in (data.get("missing_required_tools") or [])]

            elif cat == "stage_planning":
                sheet["plan_intents"] = [str(x) for x in (data.get("plan_intents") or [])]
                sheet["plan_tool_call_names_empty"] = not bool(data.get("tool_call_names"))
                skip = data.get("tool_skip_reason")
                sheet["zero_tool_no_skip"] = sheet["zero_tool_no_skip"] and not bool(skip)

            elif cat == "stage_tool_execution":
                sheet["not_found_tools"] = data.get("not_found_tools") or []

            elif cat == "stage_knowledge_truth":
                for kw, r in (data.get("keyword_results") or {}).items():
                    where = r.get("where") or {}
                    sheet["tool_output_contains"][kw] = bool(where.get("tool_results"))
                    sheet["final_answer_contains"][kw] = bool(where.get("final_answer"))
                    sheet["kb_probe_contains"][kw] = bool(r.get("kb_hit"))
                    sheet["raw_data_contains"][kw] = bool(r.get("raw_contains"))
                for kw, r in (data.get("forbidden_results") or {}).items():
                    where = r.get("where") or {}
                    sheet["tool_output_contains"][kw] = bool(where.get("tool_results"))
                    sheet["final_answer_contains"][kw] = bool(where.get("final_answer"))
                    sheet["raw_data_contains"][kw] = bool(r.get("raw_contains"))
                    if r.get("prompt_current_contains"):
                        sheet["prompt_current_contains"][kw] = True
                    if where.get("final_answer") and (where.get("tool_results") or r.get("prompt_current_contains")):
                        sheet["contamination_source_found"] = True
                for item in (data.get("not_found_results") or []):
                    term = str(item.get("term") or "")
                    if not term:
                        continue
                    sheet["alias_map_contains"][term] = bool(item.get("alias_canonical")) or len(item.get("alias_variants") or []) > 1
                    sheet["raw_data_contains"][term] = bool(item.get("raw_contains"))

            elif cat == "stage_answer":
                sheet["answer_short_circuit"] = bool(data.get("short_circuit_pattern"))

            elif cat == "stage_version":
                sheet["trace_source_snapshot_known"] = bool(data.get("trace_snapshot_known"))
                sheet["prompt_snapshot_clean"] = bool(data.get("prompt_clean"))
                sheet["code_snapshot_clean"] = bool(data.get("code_clean"))
                sheet["trace_version_clean"] = bool(data.get("all_clean"))
                sheet["prompt_trace_version_known"] = bool(data.get("trace_snapshot_known") and data.get("prompt_clean"))

            elif cat == "stage_evaluator":
                sheet["evaluator_error_detected"] = bool((data.get("audit") or {}).get("evaluator_discrepancies"))

    # 兜底：用实际工具与评测结果补齐事实。
    if not sheet["actual_tools"]:
        sheet["actual_tools"] = [str(x) for x in (result.get("actual_tools") or [])]
    return sheet


def _missing_keywords(sheet: Dict[str, Any], must_contain: List[str]) -> List[str]:
    missing: List[str] = []
    for kw in must_contain:
        kw = str(kw)
        if not kw:
            continue
        if sheet["final_answer_contains"].get(kw) is False:
            missing.append(kw)
    return missing


def _forbidden_hits(sheet: Dict[str, Any], must_not: List[str]) -> List[str]:
    return [str(kw) for kw in must_not if str(kw) and sheet["final_answer_contains"].get(str(kw)) is True]


def resolve_cause(fact_sheet: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """CausalResolver：first-failure-wins 因果梯子。

    只在确定性 FactSheet 上归因；某一步命中后立即返回，不再往更后面的阶段归因。
    """
    result = ctx.get("result") or {}
    case = ctx.get("case") or {}
    must_contain = [str(k) for k in (case.get("must_contain") or []) if str(k).strip()]
    must_not = [str(k) for k in (case.get("must_not_contain") or []) if str(k).strip()]
    actual = fact_sheet.get("actual_tools") or []
    missing_kws = _missing_keywords(fact_sheet, must_contain)
    forbidden_kws = _forbidden_hits(fact_sheet, must_not)
    chain: List[str] = []

    def _res(stage: str, kind: str, label: str, root: str, evidence_ids: List[str], target_file: str, suggestion: str, confidence: str = "high") -> Dict[str, Any]:
        return {
            "stage": stage,
            "conclusion_kind": kind,
            "label": label,
            "primary_root_cause": root,
            "evidence_ids": evidence_ids,
            "target_file": target_file,
            "suggestion_template": suggestion,
            "confidence": confidence,
            "causal_chain": chain,
            "missing_keywords": missing_kws,
            "forbidden_keywords": forbidden_kws,
        }

    chain.append("STG-01 input: 题目/评测标准已重放")

    # 1. 路由失败：当前路由硬规则要求 D 组工具，但 Trace 注入列表缺失。
    if fact_sheet.get("routing_event_seen") and fact_sheet.get("routing_missing_required_tools"):
        missing_tools = fact_sheet["routing_missing_required_tools"]
        root = (
            f"路由阶段暴露的工具集不完整：当前路由硬规则对本题要求注入 D 组任务/剧情工具，"
            f"但 Trace 实际 injected_tools={fact_sheet.get('routing_actual_tools')}，"
            f"缺失 {missing_tools}。Planner 后续只能用被暴露的工具，因此根因在路由阶段，不在 Planner。"
        )
        chain.append("STG-02 routing: 失败（必需工具未暴露）")
        return _res(
            "routing", "routing_failure", "路由阶段未暴露任务/剧情查询工具",
            root,
            ["LO-STG-02", "LO-STG-07"],
            "intent_router.py",
            "确保任务/剧情元数据类问题的确定性硬规则先于 LLM 路由结果执行，并强制并入 D 组工具；为 F1 类“角色的传说任务叫什么名字”问题做路由级回归用例。",
        )

    # 2. 规划输出失败：plan 文本明确要调工具，但结构化 tool_calls 为空/实际零调用。
    plan_intents = [str(x) for x in (fact_sheet.get("plan_intents") or [])]
    if not actual and plan_intents:
        chain.append("STG-02 routing: 通过（无缺失工具证据）")
        chain.append("STG-03 planning: 失败（文本意图存在但结构化输出/实际调用为空）")
        root = (
            f"plan 文本已规划调用 {plan_intents}，但实际工具调用为 0；"
            f"plan_tool_call_names_empty={fact_sheet.get('plan_tool_call_names_empty')}。"
            "这是 Planner 的文本推理与结构化 tool_calls 输出脱节，属于规划阶段失败。"
        )
        return _res(
            "planning", "plan_output_failure", "Planner 文本意图未转化为结构化工具调用",
            root,
            ["LO-STG-03", "LO-STG-04"],
            "app/agent/nodes.py",
            "加强 plan 阶段 tool_calls 的结构化输出校验：文本层判定需要工具时强制重试直到产出 tool_calls；重试后仍为空则走明确熔断路径，并在执行报告中显式记录 tool_skip_reason。",
        )

    # 3. 知识库真缺：所有缺失词在原始数据与检索中都不存在。
    if missing_kws:
        all_gap = True
        for kw in missing_kws:
            if fact_sheet["raw_data_contains"].get(kw) is not False or fact_sheet["kb_probe_contains"].get(kw) is not False:
                all_gap = False
                break
        if all_gap:
            chain.extend(["STG-02 routing: 通过", "STG-03 planning: 通过", "STG-05 knowledge: 失败（原始数据与检索均未命中）"])
            root = (
                f"缺少必须包含词 {missing_kws}：这些词在本次 Trace 工具返回、当前知识库检索、核心原始数据文件中均未出现。"
                "证据支持“知识库原始数据确实缺该信息”，但只能覆盖当前检索口径与核心数据文件，不能证明全部历史版本。"
            )
            return _res(
                "knowledge", "knowledge_gap", "知识库原始数据缺少评测要求的信息",
                root,
                ["LO-STG-05", "LO-STG-07"],
                "content_data/",
                "先扩充知识库原始数据（或确认规范名称/别名口径），再验证检索与回答链路；若数据在非核心文件中存在，应把该文件纳入数据目录。",
                "medium",
            )

    # 4. 别名映射失败：not_found 工具的词条在当前别名表已能解析到规范名，且原始数据存在。
    nf_tools = fact_sheet.get("not_found_tools") or []
    alias_fail: Optional[Dict[str, Any]] = None
    for nf in nf_tools:
        args = nf.get("args") or {}
        term = ""
        for v in args.values():
            if isinstance(v, str) and v:
                term = str(v)
                break
        if term and fact_sheet.get("alias_map_contains", {}).get(term) and fact_sheet.get("raw_data_contains", {}).get(term):
            alias_fail = {"term": term, "tool": nf.get("name")}
            break
    if alias_fail:
        chain.extend(["STG-02 routing: 通过", "STG-03 planning: 通过", "STG-04 tool: 出现 not_found", "STG-05 knowledge: 别名可解析且原始数据存在"])
        root = (
            f"工具 {alias_fail['tool']} 查询「{alias_fail['term']}」返回 not_found；"
            f"当前别名映射能解析到规范名，且知识库原始数据中存在该词条。"
            "属于查询词未做别名规范化的失败，而不是数据缺失。"
        )
        return _res(
            "alias", "query_alias_failure", "查询词未做别名/规范名解析",
            root,
            ["LO-STG-04", "LO-STG-05"],
            "character_aliases.py",
            "在工具入口统一做别名→规范名解析（query_character 已实现，需覆盖其他实体工具）；同时让 plan 阶段对实体查询使用规范名构造查询词。",
        )

    # 5. 召回/切片失败：工具返回没暴露关键词，但知识库原始数据存在（检索命中更坐实）。
    if missing_kws:
        recall_kws = [kw for kw in missing_kws
                      if fact_sheet["raw_data_contains"].get(kw) is True
                      and fact_sheet["tool_output_contains"].get(kw) is False]
        if recall_kws:
            kb_hit_kws = [kw for kw in recall_kws if fact_sheet["kb_probe_contains"].get(kw) is True]
            chain.extend(["STG-02 routing: 通过", "STG-03 planning: 通过", "STG-05 knowledge: 失败（原始数据在但工具未返回）"])
            root = (
                f"缺少必须包含词 {recall_kws}：知识库原始数据中存在，"
                f"其中当前检索已命中 {kb_hit_kws or '无'}，但本次 Trace 的工具返回均未包含这些关键词。"
                "根因在查询词构造或召回切片（snippet 未覆盖关键段落），不是知识库缺失。"
            )
            return _res(
                "recall", "recall_snippet_failure", "查询/召回未把原始数据中的关键段落暴露给回答阶段",
                root,
                ["LO-STG-04", "LO-STG-05"],
                "app/retrieval.py",
                "改善长任务/多段内容的召回完整性：提高结果数、按任务拆分多 snippet、BM25 与向量结果做段落级补齐；plan 阶段对概念类问题增加机制性补充查询。",
            )

    # 6. 回答阶段未整合：工具已返回关键词，但最终答案没用。
    if missing_kws:
        compose_kws = [kw for kw in missing_kws
                       if fact_sheet["tool_output_contains"].get(kw) is True
                       and fact_sheet["final_answer_contains"].get(kw) is False]
        if compose_kws:
            chain.extend(["STG-02 routing: 通过", "STG-03 planning: 通过", "STG-06 answer: 失败（工具命中但答案未使用）"])
            root = (
                f"工具返回中已包含 {compose_kws}，但最终答案没有使用这些内容。"
                "回答阶段没有整合已召回的正确信息，属于 answer 阶段失败。"
            )
            return _res(
                "answer", "answer_composition", "回答阶段未整合工具已返回的正确信息",
                root,
                ["LO-STG-05", "LO-STG-06"],
                "prompts/system/agent_system_v4_answer.txt",
                "回答提示词要求答案必须覆盖工具返回中的核心事实；对近似对象/拼写纠错后的结果允许按规则整合，而不是直接输出未收录。",
            )

    # 7. 答案污染：禁止词出现在最终答案，且来源可定位到工具返回或当前提示词。
    if forbidden_kws:
        contaminated = [kw for kw in forbidden_kws
                        if fact_sheet.get("tool_output_contains", {}).get(kw) is True
                        or fact_sheet.get("prompt_current_contains", {}).get(kw) is True]
        if contaminated:
            chain.extend(["STG-02 routing: 通过", "STG-03 planning: 通过", "STG-06 answer: 失败（禁止词进入最终答案）"])
            root = (
                f"禁止词 {contaminated} 出现在最终答案中；"
                f"工具返回命中 {[k for k in contaminated if fact_sheet.get('tool_output_contains', {}).get(k) is True] or '无'}，"
                f"当前提示词命中 {[k for k in contaminated if fact_sheet.get('prompt_current_contains', {}).get(k) is True] or '无'}。"
                "答案阶段没有过滤或误用了带污染源的原文内容。"
            )
            return _res(
                "answer", "answer_contamination", "禁止词经工具/提示词污染进入最终答案",
                root,
                ["LO-STG-05", "LO-STG-06"],
                "prompts/system/agent_system_v4_answer.txt",
                "回答提示词增加范围纪律：只使用题干限定范围内的工具内容；来自其他任务/背景段的专名必须被过滤或替换为中性表达。",
            )

    # 8. 评测器一致性问题：Trace 独立重算与 reasons 有可见差异。
    if fact_sheet.get("evaluator_error_detected"):
        chain.extend(["STG-02..06: 未发现更上游失败", "STG-08 evaluator: 发现评测差异"])
        root = (
            "Trace 真相重算发现了评测器未体现的差异（如 plan 规划工具但实际未调、零工具无豁免、"
            "not_found 工具或关键词判定不一致），本次失败可能不是 Agent 行为而是评测器漏报/误报。"
        )
        return _res(
            "evaluator", "evaluator_error", "评测器判定与 Trace 事实存在差异",
            root,
            ["LO-STG-08"],
            "app/services/evaluator.py",
            "根据 LO-STG-08 的 evaluator_discrepancies 逐条核对评测器规则与 Trace 导出字段，补齐漏报或修正误报。",
        )

    # 9. 版本未知/不一致：没有找到上游确定失败时，不能对历史行为下强断言。
    version_note = ""
    if not fact_sheet.get("trace_source_snapshot_known"):
        version_note = "Trace 没有源码/提示词版本快照，无法证明当前代码在 Trace 运行时刻生效。"
    elif not fact_sheet.get("trace_version_clean"):
        version_note = f"Trace 快照与当前工作区不一致：{fact_sheet.get('changed_files') or '未知'}。"
    if version_note:
        chain.append("STG-02..06: 未发现明确的上游确定性失败")
        chain.append("STG-07 version: 无法确定历史版本")
        return _res(
            "version", "version_unknown", "历史 Trace 版本无法确定",
            "无法定位到输入/路由/规划/工具/知识/回答阶段的确定性失败：" + version_note,
            ["LO-STG-07", "LO-STG-08"],
            "",
            "升级导出器为所有新 Trace 写入 source_snapshot；对旧 Trace 只能按当前版本给通用改进建议，禁止断言旧代码违规。",
            "low",
        )

    chain.append("STG-02..08: 未发现上述任一确定性失败")
    return _res(
        "other", "other", "需要人工复核的未分类失败",
        "8 个阶段探针均未命中确定性失败模式；需要结合证据人工复核。",
        [oid for oid in fact_sheet.get("stage_summaries", {})],
        "",
        "人工复核 LO-STG-01..08 证据后重跑医生。",
        "low",
    )


def format_pipeline_for_prompt(
    ctx: Dict[str, Any],
    fact_sheet: Dict[str, Any],
    resolution: Dict[str, Any],
    max_chars: int = 9000,
) -> str:
    """把 FactSheet 与 CausalResolver 结果压缩成 LLM 可读的权威上下文。"""
    result = ctx.get("result") or {}
    case = ctx.get("case") or {}
    stage_lines = ["确定性流水线 8/8 已执行："]
    for oid, summary in (fact_sheet.get("stage_summaries") or {}).items():
        stage_lines.append(f"- {oid}: {summary}")
    sheet_view = {
        k: fact_sheet.get(k)
        for k in (
            "actual_tools", "plan_intents", "plan_tool_call_names_empty", "answer_short_circuit",
            "not_found_tools", "evaluator_error_detected", "zero_tool_no_skip",
            "prompt_trace_version_known", "trace_source_snapshot_known",
            "prompt_snapshot_clean", "code_snapshot_clean", "trace_version_clean",
            "routing_event_seen", "routing_hard_rule_hit", "routing_expected_tools",
            "routing_actual_tools", "routing_missing_required_tools",
            "tool_output_contains", "final_answer_contains", "kb_probe_contains",
            "raw_data_contains", "alias_map_contains", "contamination_source_found",
        )
    }
    text = "\n".join(stage_lines)
    text += "\n\n【FactSheet（确定性权威事实，LLM 不得改写）】\n" + json.dumps(sheet_view, ensure_ascii=False, indent=2)
    text += "\n\n【CausalResolver 因果梯子判定（权威归因）】\n" + json.dumps(resolution, ensure_ascii=False, indent=2)
    text += "\n\n【评测结果】\n" + json.dumps({
        "passed": result.get("passed"),
        "reasons": result.get("reasons") or [],
        "prompt_pass": result.get("prompt_pass"),
        "prompt_violations": result.get("prompt_violations") or [],
    }, ensure_ascii=False, indent=2)
    text += "\n\n【LLM 职责】\n- 归因必须与 CausalResolver 的 conclusion_kind 一致；若确实发现新证据可用 read 工具补充并说明，但不得推翻确定事实。\n- 处方必须给通用机制建议，禁止写死具体题目词。\n- 引用证据 ID 时只能使用 stage 证据 ID 或 EXT ID。"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[FactSheet 过长已截断，可用 evidence_view 回看各 LO-STG 证据]"
    return text
