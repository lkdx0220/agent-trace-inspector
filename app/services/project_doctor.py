# -*- coding: utf-8 -*-
"""项目医生主循环。

流程：
1. 从评测 Run 取单题失败结果 + TestCase + Trace；
2. 确定性生成 LabOrders（检查单）；
3. qwen3.7-max + function calling 执行检查（LLM 可自由调只读工具）；
4. 覆盖闸门：LLM 想提前结束也不行，编排器自动补齐缺失检查；
5. 覆盖 100% 后，LLM 输出最终医嘱 JSON；
6. 校验每一条 prescription 的 evidence_ids 都能在证据库中找到，否则拒绝保存。

证据只能由 doctor_tools 工具写入，LLM 无法伪造。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.db import get_trace
from app.services.coverage_gate import coverage_status, evidence_ids, validate_prescriptions
from app.services.doctor_tools import (
    DEFAULT_PROJECT_PATH,
    _events,
    collect_tool_spans,
    dispatch_llm_tool,
    kb_probe_contains,
    llm_tool_definitions,
    run_lab_check,
)
from app.services.eval_store import get_run, list_test_cases
from app.services.evaluator import check_prompt_compliance
from app.services.diagnostic_pipeline import (
    CONSISTENCY_KINDS as PIPELINE_CONSISTENCY_KINDS,
    build_fact_sheet as build_fact_sheet_pipeline,
    format_pipeline_for_prompt,
    generate_pipeline_orders,
    resolve_cause,
    run_diagnostic_pipeline,
)
from app.services.path_guard import ensure_project_path
from app.services.project_map import format_project_map, generate_project_map
from app.services.qwen_client import get_qwen_endpoints
from app.services.source_snapshot import trace_snapshot_status

DOCTOR_MODEL = os.environ.get("DOCTOR_MODEL", "qwen3.7-max")
MAX_LLM_TURNS = 20
MAX_TOOL_CONTENT_CHARS = 6000


def _api_keys(project_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """返回 Qwen 接口配置（含 base_url）列表；兼容旧调用，内部改用 qwen_client。"""
    return get_qwen_endpoints(project_path)


def _trace_summary(trace: Optional[Dict[str, Any]]) -> str:
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
        lines.append(("  " * depth) + f"{name}{extra}")
        for c in span.get("children", []) or []:
            walk(c, depth + 1)
    walk(trace.get("root_span") or {})
    return "\n".join(lines[:40])


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    text = content.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兜底：取第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _format_memory_block(ctx: Dict[str, Any]) -> str:
    """把医生长期记忆（Verified Claims / Pinned Facts）注入系统提示词。"""
    claims = ctx.get("verified_claims") or []
    facts = ctx.get("pinned_facts") or []
    if not claims and not facts:
        return ""
    lines = ["", "【医生长期记忆（来自 record_verified_claim / pin_fact）】"]
    if claims:
        lines.append("Verified Claims:")
        for c in claims:
            lines.append(f"- {c.get('id')}: {c.get('claim')}  (证据: {', '.join(c.get('evidence_ids', []))})")
    if facts:
        lines.append("Pinned Facts:")
        for f in facts:
            lines.append(f"- {f.get('id')}: {f.get('text')}  (证据: {f.get('evidence_id')})")
    lines.append("以上事实已通过证据接地校验，请在后续推理中持续使用；不得与证据冲突。")
    return "\n".join(lines)


def _build_system_prompt(ctx: Dict[str, Any]) -> str:
    result = ctx["result"]
    case = ctx["case"]
    trace = ctx["trace"]
    project_map = ctx["project_map"]
    resolution = ctx.get("resolution") or {}
    fact_sheet = ctx.get("fact_sheet") or {}
    pipeline_text = ctx.get("pipeline_text") or format_pipeline_for_prompt(ctx, fact_sheet, resolution)

    schema = {
        "diagnosis": {
            "summary": "一句话结论",
            "primary_root_cause": "最可能的根因（必须与 CausalResolver 的结论一致，允许补充更具体的证据解释）",
            "issue_classification": "routing_failure|plan_output_failure|knowledge_gap|query_alias_failure|recall_snippet_failure|answer_composition|answer_contamination|prompt_violation|evaluator_error|version_unknown|other",
            "data_vs_recall": "knowledge_base_missing|recall_failure|answer_stage_failure|routing_failure|version_unknown|other",
            "key_evidence": ["与结论直接相关的证据 ID"],
        },
        "prescriptions": [
            {
                "issue": "诊断出的问题",
                "root_cause": "根因，必须能被 evidence_ids 里的证据支持，且不得与 FactSheet 矛盾",
                "conclusion_kind": "routing_failure|plan_output_failure|knowledge_gap|query_alias_failure|recall_snippet_failure|answer_composition|answer_contamination|prompt_violation|evaluator_error|version_unknown|other",
                "fact_assertions": [
                    {
                        "fact_type": "tool_output_contains|final_answer_contains|kb_probe_contains|raw_data_contains|plan_text_mentions_tool|actual_tool_called|plan_tool_call_names_empty|answer_short_circuit|prompt_current_contains|prompt_trace_version_known|alias_map_contains|not_found_tool|zero_tool_no_skip|evaluator_error_detected|routing_missing_required_tools|contamination_source_found|trace_source_snapshot_known",
                        "params": {"keyword": "相关关键词（或 term/tool_name）"},
                        "expected": True
                    }
                ],
                "evidence_ids": ["LO-STG-01", "LO-STG-05"],
                "suggestion": "具体、可执行的修改建议（指到文件/提示词/别名表）",
                "target_file": "建议修改的相对路径（只读分析，不实际修改）",
                "severity": "high|medium|low",
                "expected_effect": "修改后预期解决哪个评测项",
            }
        ],
        "confidence": 0.0,
    }

    try:
        version_status = trace_snapshot_status(trace, str(ctx.get("project_path") or DEFAULT_PROJECT_PATH))
    except Exception:
        version_status = {"trace_snapshot_known": False, "prompt_clean": False, "code_clean": False, "all_clean": False, "changed_files": []}
    if version_status.get("trace_snapshot_known"):
        version_text = "\n".join([
            "- 该 Trace 带有源码/提示词版本快照：known=true",
            f"- prompt_snapshot_clean={version_status.get('prompt_clean')}  code_snapshot_clean={version_status.get('code_clean')}  all_clean={version_status.get('all_clean')}",
            f"- 当前与 Trace 时刻不一致的文件: {version_status.get('changed_files') or []}",
            "- 判定 Agent 历史行为时，只允许在当前文件与 Trace 快照一致的范围内引用当前代码；不一致时只能描述当前版本，不得断言旧行为。",
        ])
    else:
        version_text = "\n".join([
            "- 该 Trace 没有源码/提示词版本快照：known=false",
            "- 只能以“当前工作区视角”做分析，不能断言 Trace 运行时刻的提示词/代码/知识库状态。",
            "- 涉及“违反当时提示词/当时代码”的结论应标记为 version_unknown，或只给出当前版本的改进建议。",
        ])

    return f"""你是“项目医生”：一位了解原神剧情助手项目架构、系统提示词与知识库实现的 AI 诊断专家。
你面对一份评测失败病例。确定性诊断流水线已经完成全部 8 阶段采证与 CausalResolver 归因，你的任务：
（1）阅读下面 FactSheet 与因果梯子判定，理解证据链；
（2）把结论用自然语言解释清楚，并开出处方（prescriptions）；
（3）每条处方必须绑定真实证据 ID，且 conclusion_kind 必须与 CausalResolver 判定一致。

【项目地图（只读）】
{project_map}

【评测失败病例】
- 题目: {result.get('question')}
- 分类: {case.get('category')} / {case.get('difficulty')}
- 最终答案: {result.get('answer')}
- 失败原因: {json.dumps(result.get('reasons'), ensure_ascii=False)}
- 实际工具: {json.dumps(result.get('actual_tools'), ensure_ascii=False)}
- prompt_pass: {result.get('prompt_pass')}  违规: {json.dumps(result.get('prompt_violations'), ensure_ascii=False)}
- 评测标准: must_contain={json.dumps(case.get('must_contain'), ensure_ascii=False)} must_not_contain={json.dumps(case.get('must_not_contain'), ensure_ascii=False)} match_mode={case.get('match_mode')} alternatives={json.dumps(case.get('alternatives'), ensure_ascii=False)}

【Trace 摘要】
- trace_id: {trace.get('trace_id') if trace else '无'}
- trace_created_at: {trace.get('created_at') if trace else '无'}
{_trace_summary(trace)}

【版本快照状态】
{version_text}

{pipeline_text}

【工具使用铁律】
1. 8 个 LO-STG 阶段证据已由编排器确定执行完毕，你不能再改查什么；这些证据是权威事实。
2. 额外自由调查（read_project_file / grep_project / read_system_prompt / search_knowledge_base）会产生 EXT-001、EXT-002… 等 ID，仅用于补充处方细节，不得推翻 FactSheet。
3. evidence_search / evidence_view 只读回看已记录证据，不会产生新的 EXT 证据；需要证明某句话出自证据时，应引用它所在的 LO/EXT ID，而不是重复描述。
4. record_verified_claim / pin_fact 是医生记忆工具：它们会做证据接地校验，只允许记录有真实证据支撑的事实；这些记录会随最终结果一起保存。
5. 禁止编造证据；prescription 的 evidence_ids 引用了不存在的 ID 会被覆盖闸门拒绝。
6. 最终处方会做“证据闸门”检查：prescription.root_cause 必须能在其 evidence_ids 对应证据的原文中找到至少一个 2 字以上中文术语或 4 字符以上代码词；涉及“违反/规则”必须同时引用提示词证据和 Trace 行为证据；涉及知识库/召回必须引用检索/缺失词证据；涉及具体代码文件必须引用读文件/grep 证据。
6.1 还会做“结论一致性闸门”：每条 prescription 必须写 conclusion_kind 和 fact_assertions；conclusion_kind 与 CausalResolver 不一致、fact_assertions 与 FactSheet 矛盾、或缺少必要断言组合都会被拒绝。
7. 你只读原项目，绝不建议或执行写操作。处方是建议文本，不落盘到原项目。
8. 版本纪律：read_system_prompt 返回的是当前工作区版本。禁止把“当前版本新增/删除的规则”当作 Trace 运行时刻已生效的规则来判定历史违规；只能引用同时包含提示词原文和 Trace 行为证据的结论。
9. 防止过拟合：处方必须指向“通用机制/通用规则/通用检索策略”，禁止建议把任何具体题目的关键词、题目名、数字写死进代码或提示词（例如禁止建议“在 nodes.py 里写死 XXX 自动补搜”）。

【最终输出格式】
现在直接输出一个 JSON 对象（不要 markdown 围栏外的其他文字）：
{json.dumps(schema, ensure_ascii=False, indent=2)}

{_format_memory_block(ctx)}"""

def _call_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], endpoints: Any) -> Dict[str, Any]:
    """调用千问兼容接口；支持单个 endpoint 或 endpoint 列表，自动按顺序重试。

    endpoint 形如 {"base_url": "...", "api_key": "..."}。
    """
    if isinstance(endpoints, dict):
        eps = [endpoints]
    else:
        eps = list(endpoints or [])
    if not eps:
        raise RuntimeError("缺少 Qwen API Key")
    payload = {
        "model": DOCTOR_MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
        "stream": False,
    }
    errors = []
    for ep in eps:
        base_url = str(ep.get("base_url") or "").rstrip("/")
        api_key = str(ep.get("api_key") or "")
        url = base_url + "/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=300,
            )
            if resp.status_code == 200:
                return resp.json()
            errors.append(f"[{ep.get('source','?')}] {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            errors.append(f"[{ep.get('source','?')}] {repr(e)}")
    raise RuntimeError(f"LLM 调用失败（尝试了 {len(eps)} 个接口）: {' | '.join(errors)}")


def _tool_content(result: Dict[str, Any], max_chars: int = MAX_TOOL_CONTENT_CHARS) -> str:
    """VulnClaw 式高信号 preview：上下文只放摘要/关键行，完整 raw 保留在证据库。"""
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text

    signal_markers = (
        "summary", "path", "line", "keyword", "tool", "error", "status",
        "hit", "probe", "root", "evidence", "truncat", "query", "file",
        "reason", "violation", "not_found", "未收录", "未找到",
    )
    lines = []
    for line in text.splitlines():
        low = line.lower()
        if any(m in low for m in signal_markers):
            stripped = line.strip()[:300]
            if stripped and stripped not in lines[-8:]:
                lines.append(stripped)
        if len(lines) >= 36:
            break

    header = (
        f"[high-signal preview] raw_size={len(text)} chars; "
        "完整 evidence 已保存，可用 evidence_view / evidence_search 查看。"
    )
    body = "\n".join(lines) if lines else ""
    head_tail = text[:max_chars // 2].rstrip() + "\n...[中间省略]...\n" + text[-max_chars // 2:].lstrip()
    if body:
        return "\n".join([header, "", "[signal lines]", body, "", "[head/tail]", head_tail])
    return "\n".join([header, "", head_tail])


def _record_verified_claim(
    args: Dict[str, Any],
    ctx: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """确定性地记录一条“已验证事实”。只允许引用真实存在的证据 ID，
    且 claim 必须能在所引证据原文中找到字面术语（L4 级）。"""
    claim = str(args.get("claim") or "").strip()
    eids = [str(e) for e in (args.get("evidence_ids") or [])]
    if not claim:
        return {"ok": False, "error": "claim 不能为空"}
    valid = evidence_ids(evidence_by_order, extra_evidence)
    unknown = [e for e in eids if e not in valid]
    if unknown:
        return {"ok": False, "error": f"引用了不存在的证据 ID: {unknown}"}
    corpus = _evidence_corpus_for_ids(eids, evidence_by_order, extra_evidence)
    terms = _meaningful_terms(claim)
    if not any(t.lower() in corpus.lower() for t in terms):
        return {"ok": False, "error": f"claim 没有字面证据支撑（候选术语 {terms[:6]}）"}
    records = ctx.setdefault("verified_claims", [])
    cid = f"C{len(records) + 1:03d}"
    rec = {
        "id": cid,
        "claim": claim,
        "evidence_ids": eids,
        "evidence_level": "L4",
        "created_at": datetime.now().isoformat(),
    }
    records.append(rec)
    return {
        "ok": True,
        "claim_id": cid,
        "claim": claim,
        "evidence_ids": eids,
        "message": f"已记录 VerifiedClaim {cid}，证据：{', '.join(eids)}",
    }


def _pin_fact(
    args: Dict[str, Any],
    ctx: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """记录一条必须持续出现在上下文中的高信号事实，必须绑定真实证据。"""
    text = str(args.get("text") or "").strip()
    eid = str(args.get("evidence_id") or "").strip()
    if not text:
        return {"ok": False, "error": "text 不能为空"}
    valid = evidence_ids(evidence_by_order, extra_evidence)
    if eid not in valid:
        return {"ok": False, "error": f"evidence_id 不存在: {eid}"}
    corpus = _evidence_corpus_for_ids([eid], evidence_by_order, extra_evidence)
    terms = _meaningful_terms(text)
    if not any(t.lower() in corpus.lower() for t in terms):
        return {"ok": False, "error": f"pinned fact 没有在 {eid} 中找到字面支撑"}
    facts = ctx.setdefault("pinned_facts", [])
    if any(f.get("text") == text for f in facts):
        return {"ok": True, "cached": True, "message": "该事实已固定，未重复添加"}
    fid = f"P{len(facts) + 1:03d}"
    rec = {
        "id": fid,
        "text": text,
        "evidence_id": eid,
        "created_at": datetime.now().isoformat(),
    }
    facts.append(rec)
    return {"ok": True, "fact_id": fid, "text": text, "evidence_id": eid}


def _execute_tool(
    name: str,
    args: Dict[str, Any],
    ctx: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if name == "run_lab_check":
        oid = args.get("lab_order_id", "")
        prior = evidence_by_order.get(oid, [])
        if any(e.get("ok") for e in prior):
            return {"ok": True, "order_id": oid, "cached": True, "summary": prior[0].get("summary")}
        result = run_lab_check(oid, ctx)
        evidence_by_order.setdefault(oid, []).append(result)
        return result

    # 验证事实/固定事实：只写医生记忆，不产生 EXT 证据。
    if name == "record_verified_claim":
        return _record_verified_claim(args, ctx, evidence_by_order, extra_evidence)
    if name == "pin_fact":
        return _pin_fact(args, ctx, evidence_by_order, extra_evidence)

    # 证据回看类工具只读已记录证据，不产生新 EXT 证据。
    if name in {"evidence_search", "evidence_view"}:
        return dispatch_llm_tool(name, args, ctx, evidence_by_order, extra_evidence)

    result = dispatch_llm_tool(name, args, ctx, evidence_by_order, extra_evidence)
    ev = {
        "id": f"EXT-{len(extra_evidence) + 1:03d}",
        "tool": name,
        "args": args,
        "ok": result.get("ok", True),
        "result": result,
        "created_at": datetime.now().isoformat(),
    }
    extra_evidence.append(ev)
    return result


def _autofill_missing(
    ctx: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """编排器兜底：LLM 没做完的检查单，由确定性代码直接补齐。"""
    from app.services.coverage_gate import missing_orders
    filled = []
    for order in missing_orders(ctx["lab_orders"], evidence_by_order):
        result = run_lab_check(order["id"], ctx)
        evidence_by_order.setdefault(order["id"], []).append(result)
        filled.append({"order_id": order["id"], "ok": result.get("ok"), "summary": result.get("summary")})
    return filled


def _assertions_from_patterns(
    kind: str,
    fact_sheet: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """按结论类型生成与 FactSheet 一致的事实断言（兜底报告用）。"""
    assertions: List[Dict[str, Any]] = []
    for fact_type, expected, param_field in _required_fact_patterns(kind):
        a: Dict[str, Any] = {"fact_type": fact_type, "expected": expected}
        if param_field in {"keyword", "term"}:
            vals = fact_sheet.get(fact_type, {})
            key = next((k for k, v in vals.items() if bool(v) == expected), "")
            a["params"] = {param_field: key or "?"}
        assertions.append(a)
    return assertions


def _fallback_report(
    result: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: Optional[List[Dict[str, Any]]] = None,
    resolution: Optional[Dict[str, Any]] = None,
    fact_sheet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """医生 LLM 失败时的确定性兜底：不允许 0 处方。

    优先使用 CausalResolver 的确定性归因直接生成处方；没有归因时回退到按检查单类别归纳。
    """
    if resolution and str(resolution.get("conclusion_kind") or "") not in {"", "other"}:
        kind = str(resolution["conclusion_kind"])
        sheet = fact_sheet or {}
        eids = [str(e) for e in (resolution.get("evidence_ids") or [])]
        valid = evidence_ids(evidence_by_order, extra_evidence or [])
        eids = [e for e in eids if e in valid]
        if not eids:
            eids = [oid for oid in evidence_by_order.keys() if any(e.get("ok") for e in evidence_by_order[oid])]
        prescription = {
            "issue": str(resolution.get("label") or "确定性归因"),
            "root_cause": str(resolution.get("primary_root_cause") or ""),
            "conclusion_kind": kind,
            "fact_assertions": _assertions_from_patterns(kind, sheet),
            "evidence_ids": eids,
            "suggestion": str(resolution.get("suggestion_template") or "人工复核 8 阶段证据后确定修改方案。"),
            "target_file": str(resolution.get("target_file") or ""),
            "severity": "high" if kind not in {"version_unknown", "other"} else "medium",
            "expected_effect": f"解决 {result.get('reasons') or result.get('case_id')} 对应的失败项",
            "evidence_level": "L4" if eids else "L1",
        }
        return {
            "diagnosis": {
                "summary": str(resolution.get("label") or "确定性归因"),
                "primary_root_cause": str(resolution.get("primary_root_cause") or ""),
                "issue_classification": kind,
                "key_evidence": eids,
            },
            "prescriptions": [prescription],
            "confidence": 0.0,
            "_note": "fallback_pipeline",
        }

    prescriptions: List[Dict[str, Any]] = []
    evidence_ids_used: List[str] = []
    for oid, evs in evidence_by_order.items():
        ev = next((e for e in evs if e.get("ok")), None)
        if not ev:
            continue
        category = ev.get("category") or ""
        data = ev.get("data") or {}
        keyword = str(data.get("keyword") or "")
        if category == "missing_keyword" and keyword:
            root = str(data.get("conclusion") or f"关键词「{keyword}」的出处需要人工复核")
            prescriptions.append({
                "issue": f"缺少必须包含关键词「{keyword}」",
                "root_cause": root,
                "evidence_ids": [oid],
                "suggestion": "请人工核对上述证据：若知识库能命中但工具未返回，优先检查查询词/召回切片；若工具已返回但答案未用，优先检查回答阶段整合。",
                "target_file": "",
                "severity": "high",
                "expected_effect": f"解决缺少必须包含：{keyword}",
                "evidence_level": "L2",
            })
            evidence_ids_used.append(oid)
        elif category == "forbidden_keyword" and keyword:
            root = str(data.get("conclusion") or f"禁止词「{keyword}」的来源需要人工定位")
            prescriptions.append({
                "issue": f"答案出现禁止词「{keyword}」",
                "root_cause": root,
                "evidence_ids": [oid],
                "suggestion": "请人工核对词出现的阶段；若是工具/提示词带入，考虑在回答阶段过滤或调整提示词；若是答案自行生成，考虑收紧回答规则。",
                "target_file": "",
                "severity": "high",
                "expected_effect": f"解决出现禁止包含：{keyword}",
                "evidence_level": "L2",
            })
            evidence_ids_used.append(oid)
        elif category in {"prompt_violation", "zero_tool"}:
            root = (
                "非豁免场景未调用工具，或存在系统提示词合规违规；"
                f"确定性证据：{str(ev.get('summary') or '')}"
            )
            prescriptions.append({
                "issue": "工具调用/系统提示词合规问题",
                "root_cause": root,
                "evidence_ids": [oid],
                "suggestion": "请人工确认 plan/answer 链路：是否有 tool_skip_reason、plan_retry 是否发生、answer 是否走了 not_found 短路。",
                "target_file": "",
                "severity": "high",
                "expected_effect": "解决零工具调用或违反系统提示词的失败",
                "evidence_level": "L2",
            })
            evidence_ids_used.append(oid)
        elif category == "trace_truth_audit":
            mism = data.get("plan_intent_mismatch") or {}
            dis = data.get("evaluator_discrepancies") or []
            if mism:
                planned = "、".join(mism.get("plan_intents") or [])
                actual = "、".join(mism.get("actual_tools") or []) if mism.get("actual_tools") else "无"
                root = f"plan 文本规划调用 {planned}，但实际工具调用为 {actual}；属于 plan 结构化工具输出缺失"
                prescriptions.append({
                    "issue": "plan 意图与实际工具调用不一致",
                    "root_cause": root,
                    "evidence_ids": [oid],
                    "suggestion": "重点排查 plan LLM 的结构化 tool_calls 输出：文本层已判断需要调用工具，但结构化输出层未生成调用；同时检查导出器是否遗漏 tool span。",
                    "target_file": "",
                    "severity": "high",
                    "expected_effect": "定位 plan 文本有意图但未实际调工具的问题",
                    "evidence_level": "L2",
                })
                evidence_ids_used.append(oid)
            elif dis:
                root = "Trace 真相重算发现评测器可能存在遗漏：" + "；".join(dis[:4])
                prescriptions.append({
                    "issue": "评测器一致性差异",
                    "root_cause": root,
                    "evidence_ids": [oid],
                    "suggestion": "请人工核对 Trace 真相重算结果与评测器 reasons；若确认评测器漏报，应补强评测器或 LabOrders 生成规则。",
                    "target_file": "",
                    "severity": "medium",
                    "expected_effect": "纠正评测器漏报/误报带来的诊断盲区",
                    "evidence_level": "L2",
                })
                evidence_ids_used.append(oid)
            else:
                root = "Trace 真相重算完成，未发现 plan 意图与工具调用的明确不一致"
                prescriptions.append({
                    "issue": "Trace 真相重算",
                    "root_cause": root,
                    "evidence_ids": [oid],
                    "suggestion": "该病例的 Trace 层面未见明显异常，继续结合其他证据判断。",
                    "target_file": "",
                    "severity": "low",
                    "expected_effect": "完成 Trace 独立对照",
                    "evidence_level": "L2",
                })
                evidence_ids_used.append(oid)

        elif category == "answer_integrity":
            root = str(data.get("summary") or "最终答案与工具返回之间可能存在整合问题")
            prescriptions.append({
                "issue": "回答阶段整合待确认",
                "root_cause": root,
                "evidence_ids": [oid],
                "suggestion": "请人工比对工具返回完整内容与最终答案，确认是否存在工具已命中但回答短路/漏用。",
                "target_file": "",
                "severity": "medium",
                "expected_effect": "排查回答阶段的漏整合问题",
                "evidence_level": "L2",
            })
            evidence_ids_used.append(oid)

    if not prescriptions and evidence_by_order:
        first_oid = next(iter(evidence_by_order.keys()))
        first_ev = next((e for e in evidence_by_order[first_oid] if e.get("ok")), None)
        if first_ev:
            root = f"医生 LLM 未产出有效 JSON，已执行检查单 {first_oid}；确定性证据：{str(first_ev.get('summary') or '待人工复核')}"
            prescriptions.append({
                "issue": "医生 LLM 产出失败，需人工复核",
                "root_cause": root,
                "evidence_ids": [first_oid],
                "suggestion": "基于检查单证据人工复核，或重跑医生诊断。",
                "target_file": "",
                "severity": "medium",
                "expected_effect": "完成人工复核并形成处方",
                "evidence_level": "L2",
            })
            evidence_ids_used.append(first_oid)
    elif not prescriptions:
        prescriptions.append({
            "issue": "医生 LLM 产出失败且无可执行检查单",
            "root_cause": "医生 LLM 未在限定轮次内产出有效最终 JSON，且检查单证据为空，需人工重跑诊断。",
            "evidence_ids": [],
            "suggestion": "检查 API Key/模型调用或重跑医生。",
            "target_file": "",
            "severity": "medium",
            "expected_effect": "恢复医生诊断",
            "evidence_level": "L1",
        })

    ev_summary = {
        oid: [e.get("summary") for e in evs]
        for oid, evs in evidence_by_order.items()
    }
    return {
        "diagnosis": {
            "summary": "医生 LLM 未在限定轮次内产出有效最终 JSON，已自动补齐全部检查单，以下为确定性证据与候选处方。",
            "primary_root_cause": "医生 LLM 产出失败；具体根因需人工基于下方证据确认",
            "issue_classification": "other",
            "key_evidence": evidence_ids_used or list(evidence_by_order.keys()),
        },
        "prescriptions": prescriptions,
        "confidence": 0.0,
        "_evidence_digest": ev_summary,
        "_note": "fallback",
    }


def _evidence_corpus_for_ids(
    eids: List[str],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> str:
    """拼接指定证据 ID 对应的原始证据文本，用于逐字接地校验。"""
    parts: List[str] = []
    eid_set = set(eids)
    for oid in eid_set:
        for ev in evidence_by_order.get(oid, []):
            if ev.get("ok"):
                parts.append(json.dumps(ev, ensure_ascii=False))
    for ev in extra_evidence:
        if ev.get("id") in eid_set and ev.get("ok") is not False:
            parts.append(json.dumps(ev.get("result") or ev, ensure_ascii=False))
    return "\n".join(parts)


def _meaningful_terms(text: str) -> List[str]:
    """从文本中取出可做字面验证的术语：2 字以上中文连续串，4 字符以上英文/代码记号。"""
    if not text:
        return []
    cjk = re.findall(r"[一-鿿]{2,}", text)
    words = re.findall(r"[A-Za-z0-9_./:-]{4,}", text)
    # 中文整段往往不会逐字出现在证据里，因此同时用 2 字滑动窗做最小接地。
    bigrams: List[str] = []
    for run in cjk:
        if len(run) >= 2:
            bigrams.extend(run[i:i + 2] for i in range(len(run) - 1))
    return list(dict.fromkeys(cjk + bigrams + words))



def _evidence_type_profile(
    eids: List[str],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """汇总一组证据 ID 的类型信息，供针对性接地检查使用。"""
    categories: set = set()
    tools: set = set()
    loid: set = set()
    for eid in eids:
        for ev in evidence_by_order.get(eid, []):
            if ev.get("ok"):
                categories.add(ev.get("category") or "")
                loid.add(eid)
        for ev in extra_evidence:
            if ev.get("id") == eid and ev.get("ok") is not False:
                tools.add(ev.get("tool") or "")
                loid.add(eid)
    return {
        "categories": categories,
        "tools": tools,
        "order_ids": loid,
    }


def _has_trace_level_evidence(profile: Dict[str, Any]) -> bool:
    cats = profile["categories"]
    return bool(cats & {
        "trace_replay", "plan_intent", "prompt_violation", "zero_tool",
        "stage_input", "stage_routing", "stage_planning", "stage_tool_execution",
        "stage_answer", "stage_evaluator",
    })


def _has_prompt_rule_evidence(profile: Dict[str, Any]) -> bool:
    cats = profile["categories"]
    return bool(cats & {"prompt_rule", "stage_planning", "stage_version", "stage_knowledge_truth"}) or "read_system_prompt" in profile["tools"]


def _has_kb_probe_evidence(profile: Dict[str, Any]) -> bool:
    cats = profile["categories"]
    return bool(cats & {"missing_keyword", "not_found_tool", "stage_knowledge_truth"}) or "search_knowledge_base" in profile["tools"] or "inspect_aliases" in profile["tools"]


def _has_file_evidence(profile: Dict[str, Any], target: str = "") -> bool:
    tools = profile["tools"]
    cats = profile["categories"]
    if bool(tools & {"read_project_file", "grep_project", "read_system_prompt"}):
        return True
    t = str(target).lower()
    # 迭代后架构：文件级结论可以由对应阶段探针直接支撑（探针执行了当前代码/提示词）。
    if "intent_router.py" in t:
        return "stage_routing" in cats
    if any(x in t for x in ("prompts/", "agent_system")):
        return bool(cats & {"stage_planning", "stage_version", "stage_knowledge_truth"})
    if any(x in t for x in ("app/retrieval.py", "character_aliases.py", "app/tools/query.py", "content_data/")):
        return "stage_knowledge_truth" in cats
    if any(x in t for x in ("app/agent/nodes.py", "app/agent/executor.py")):
        return bool(cats & {"stage_planning", "stage_tool_execution"})
    return bool(cats & {"stage_routing", "stage_planning", "stage_tool_execution", "stage_knowledge_truth"})


def _final_grounding_check(
    report: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """VulnClaw 式证据闸门：医嘱中的根因必须能在所引用证据里找到字面术语。

    除字面接地外，按结论类型校验证据性质：
    - 涉及“违反/违规/系统提示词/规则”的处方，必须同时引用提示词证据和 Trace 级证据；
    - 涉及“知识库/召回/切片/查询词”的处方，必须引用知识库探查证据；
    - 涉及具体代码/文件行的处方，必须引用 read_project_file / grep_project 证据。
    """
    issues: List[str] = []
    prescriptions = report.get("prescriptions") or []
    for i, p in enumerate(prescriptions):
        eids = p.get("evidence_ids") or []
        corpus = _evidence_corpus_for_ids(eids, evidence_by_order, extra_evidence)
        if not corpus:
            issues.append(f"prescriptions[{i}] 引用的 evidence_ids 没有对应证据内容")
            continue

        root = str(p.get("root_cause") or "")
        terms = _meaningful_terms(root)
        grounded = [t for t in terms if t.lower() in corpus.lower()]
        if not grounded:
            issues.append(
                f"prescriptions[{i}] root_cause 没有任何字面证据支撑"
                f"（引用 {eids}，候选术语 {terms[:8]}）"
            )
            continue

        profile = _evidence_type_profile(eids, evidence_by_order, extra_evidence)

        # 规则/违规类结论：必须有当前提示词原文 + Trace 行为证据。
        if any(m in root for m in ("违反", "违规", "系统提示词", "提示词规则", "规则未落地", "P3")):
            if not _has_prompt_rule_evidence(profile):
                issues.append(
                    f"prescriptions[{i}] 包含“违反/提示词/规则”类根因，但 evidence_ids {eids} 没有 read_system_prompt 或 LO-003(prompt_rule) 证据"
                )
            if not _has_trace_level_evidence(profile):
                issues.append(
                    f"prescriptions[{i}] 包含“违反/违规”类根因，但 evidence_ids {eids} 没有 Trace 级行为证据（LO-001/LO-002/LO-PR-01/LO-ZT-01）"
                )

        # 知识库/召回类结论：必须有检索或缺失词探针证据。
        if any(m in root for m in ("知识库", "召回", "未召回", "切片", "查询词", "没搜到", "检索")):
            if not _has_kb_probe_evidence(profile):
                issues.append(
                    f"prescriptions[{i}] 提到知识库/召回/检索，但 evidence_ids {eids} 没有 missing_keyword/not_found_tool 或 search_knowledge_base/inspect_aliases 证据"
                )

        # 代码/文件行级结论：必须有读代码或 grep 证据。
        target = str(p.get("target_file") or "")
        if target or any(m in root for m in ("app/", "prompts/", "nodes.py", "query.py", "retrieval.py", "character_aliases.py", "行 ")):
            if not _has_file_evidence(profile, target):
                issues.append(
                    f"prescriptions[{i}] 指向具体代码/文件，但 evidence_ids {eids} 没有 read_project_file / grep_project / read_system_prompt 证据"
                )

    return {"valid": not issues, "issues": issues}


NEGATIVE_MARKERS = (
    "题目设置", "评测标准", "知识库真缺", "知识库缺失", "数据缺失",
    "数据不存在", "知识库中没有", "知识库未收录", "无需修改",
    "无修改建议", "无法归因",
)


def _has_negative_conclusion(report: Dict[str, Any]) -> bool:
    diag = report.get("diagnosis") or {}
    texts = [
        diag.get("summary", ""), diag.get("primary_root_cause", ""),
        diag.get("issue_classification", ""),
    ]
    texts.extend(str(p.get("root_cause") or "") for p in (report.get("prescriptions") or []))
    blob = "\n".join(str(t) for t in texts)
    return any(m in blob for m in NEGATIVE_MARKERS)


def _has_search_probe(
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> bool:
    """是否存在至少一条真正做过知识库/工具/代码探查的证据。"""
    probe_categories = {"missing_keyword", "forbidden_keyword", "not_found_tool", "prompt_rule", "zero_tool"}
    for evs in evidence_by_order.values():
        for ev in evs:
            if ev.get("ok") and ev.get("category") in probe_categories:
                return True
    for ev in extra_evidence:
        if ev.get("ok") is False:
            continue
        if ev.get("tool") in {
            "search_knowledge_base", "inspect_aliases", "read_project_file",
            "grep_project", "read_system_prompt",
        }:
            return True
    return False


def _has_kb_gap_probe(
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> bool:
    """针对“知识库/数据真缺”结论，要求确实执行过能证明缺失的探查。"""
    for evs in evidence_by_order.values():
        for ev in evs:
            if not ev.get("ok"):
                continue
            cat = ev.get("category")
            if cat == "not_found_tool":
                return True
            if cat == "missing_keyword":
                data = ev.get("data") or {}
                kb = data.get("kb_probe") or {}
                kw = str(data.get("keyword") or "")
                # 有检索但正文未命中，才是“缺失”方向的证据
                if kb.get("ok") and kw and not kb_probe_contains(kb, kw):
                    return True
    for ev in extra_evidence:
        if ev.get("ok") is False:
            continue
        if ev.get("tool") in {"search_knowledge_base", "inspect_aliases"}:
            return True
    return False


def _evidence_says_keyword_exists(
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
) -> bool:
    """如果 missing_keyword 证据显示关键词已经在工具返回或知识库检索中出现，
    就不能再下“数据/题目缺失”的结论。"""
    for evs in evidence_by_order.values():
        for ev in evs:
            if not ev.get("ok"):
                continue
            if ev.get("category") != "missing_keyword":
                continue
            data = ev.get("data") or {}
            where = data.get("where") or {}
            if where.get("tool_results") is True:
                return True
            kb = data.get("kb_probe") or {}
            kw = str(data.get("keyword") or "")
            if kb.get("ok") and kw and kb_probe_contains(kb, kw):
                return True
    return False


def _near_miss_gate(
    report: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """VulnClaw 式近成功闸门：医生不能在没有足够探查证据时，
    轻易把失败归为“题目设置问题 / 知识库真缺 / 无需修改”。"""
    issues: List[str] = []
    prescriptions = report.get("prescriptions") or []
    kinds = {str(p.get("conclusion_kind") or "") for p in prescriptions}
    # 医生已经显式把根因分成非“知识库真缺”类（召回/回答/plan/版本/别名/评测器），
    # 不再用近成功闸门拦截；该闸门只针对“知识库真缺/题目设置/无需修改”类结论。
    if kinds and "knowledge_gap" not in kinds:
        return {"valid": True, "issues": []}
    if not _has_negative_conclusion(report):
        return {"valid": True, "issues": []}
    blob = "\n".join([
        str((report.get("diagnosis") or {}).get("summary", "")),
        str((report.get("diagnosis") or {}).get("primary_root_cause", "")),
        str((report.get("diagnosis") or {}).get("issue_classification", "")),
    ] + [str(p.get("root_cause") or "") for p in (report.get("prescriptions") or [])])
    kb_gap_markers = ("知识库真缺", "知识库缺失", "数据缺失", "数据不存在", "知识库中没有", "知识库未收录", "不存在")
    if any(m in blob for m in kb_gap_markers) and not _has_kb_gap_probe(evidence_by_order, extra_evidence):
        issues.append(
            "结论把失败归为“知识库真缺/数据不存在”，但没有一条 missing_keyword/not_found_tool 或知识库检索证据能证明该数据确实缺失"
        )
    if not _has_search_probe(evidence_by_order, extra_evidence):
        issues.append(
            "结论包含“题目/数据/知识库不存在/无需修改”类判断，"
            "但没有执行知识库检索、工具复检或代码/提示词读取证据"
        )
    if _evidence_says_keyword_exists(evidence_by_order):
        issues.append(
            "已有 missing_keyword 证据显示关键词在工具返回或知识库检索中出现；"
            "不能下“数据/题目缺失”结论，应改为回答阶段/集成失败类根因"
        )
    return {"valid": not issues, "issues": issues}



# ============================================================
# 结论一致性闸门（FactSheet + fact_assertions）
# ============================================================

CONSISTENCY_KINDS = set(PIPELINE_CONSISTENCY_KINDS) | {
    "knowledge_gap", "recall_failure", "answer_composition",
    "plan_output_failure", "prompt_violation", "alias_mapping",
    "evaluator_error",
}


def _build_fact_sheet(
    ctx: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """结论一致性闸门使用的 FactSheet（由确定性诊断流水线构建）。"""
    return build_fact_sheet_pipeline(ctx, evidence_by_order, extra_evidence)



def _assertion_actual(sheet: Dict[str, Any], assertion: Dict[str, Any]) -> Optional[bool]:
    """读取一条 fact_assertion 在 FactSheet 中的实际布尔值；无法确定返回 None。"""
    fact_type = str(assertion.get("fact_type") or "")
    params = assertion.get("params") or {}
    simple = {
        "plan_text_mentions_tool": bool(sheet.get("plan_intents")),
        "plan_tool_call_names_empty": bool(sheet.get("plan_tool_call_names_empty")),
        "answer_short_circuit": bool(sheet.get("answer_short_circuit")),
        "prompt_trace_version_known": bool(sheet.get("prompt_trace_version_known")),
        "zero_tool_no_skip": bool(sheet.get("zero_tool_no_skip")),
        "evaluator_error_detected": bool(sheet.get("evaluator_error_detected")),
        "trace_source_snapshot_known": bool(sheet.get("trace_source_snapshot_known")),
        "prompt_snapshot_clean": bool(sheet.get("prompt_snapshot_clean")),
        "code_snapshot_clean": bool(sheet.get("code_snapshot_clean")),
        "trace_version_clean": bool(sheet.get("trace_version_clean")),
        "routing_event_seen": bool(sheet.get("routing_event_seen")),
        "routing_hard_rule_hit": bool(sheet.get("routing_hard_rule_hit")),
        "routing_missing_required_tools": bool(sheet.get("routing_missing_required_tools")),
        "contamination_source_found": bool(sheet.get("contamination_source_found")),
        "version_unknown": bool(not sheet.get("trace_source_snapshot_known") or not sheet.get("trace_version_clean")),
    }
    if fact_type in simple:
        return simple[fact_type]
    if fact_type == "actual_tool_called":
        actual = sheet.get("actual_tools") or []
        tool = str(params.get("tool") or "")
        if tool:
            return tool in actual
        return bool(actual)
    if fact_type == "not_found_tool":
        nf = sheet.get("not_found_tools") or []
        tool = str(params.get("tool_name") or "")
        if tool:
            return any(str(x.get("name")) == tool for x in nf)
        return bool(nf)
    if fact_type in {"tool_output_contains", "final_answer_contains", "kb_probe_contains", "raw_data_contains", "alias_map_contains", "prompt_current_contains"}:
        key = ""
        if fact_type == "alias_map_contains":
            key = str(params.get("term") or "")
        else:
            key = str(params.get("keyword") or "")
        if not key:
            return None
        val = sheet.get(fact_type, {}).get(key)
        if val is None:
            return None
        return bool(val)
    return None


def _required_fact_patterns(kind: str) -> List[Tuple[str, bool, str]]:
    """返回该结论类型必须出现的 (fact_type, expected, param_field) 三元组。"""
    if kind == "routing_failure":
        return [
            ("routing_event_seen", True, ""),
            ("routing_missing_required_tools", True, ""),
        ]
    if kind == "knowledge_gap":
        return [
            ("tool_output_contains", False, "keyword"),
            ("kb_probe_contains", False, "keyword"),
            ("raw_data_contains", False, "keyword"),
        ]
    if kind == "recall_failure":
        return [
            ("tool_output_contains", False, "keyword"),
            ("raw_data_contains", True, "keyword"),
        ]
    if kind == "recall_snippet_failure":
        return [
            ("tool_output_contains", False, "keyword"),
            ("raw_data_contains", True, "keyword"),
        ]
    if kind == "answer_composition":
        return [
            ("tool_output_contains", True, "keyword"),
            ("final_answer_contains", False, "keyword"),
        ]
    if kind == "answer_contamination":
        return [
            ("final_answer_contains", True, "keyword"),
            ("contamination_source_found", True, ""),
        ]
    if kind == "plan_output_failure":
        return [
            ("plan_text_mentions_tool", True, ""),
            ("actual_tool_called", False, "tool"),
        ]
    if kind == "prompt_violation":
        return [
            ("prompt_trace_version_known", True, ""),
            ("zero_tool_no_skip", True, ""),
        ]
    if kind == "query_alias_failure":
        return [
            ("not_found_tool", True, "tool_name"),
            ("alias_map_contains", True, "term"),
            ("raw_data_contains", True, "keyword"),
        ]
    if kind == "alias_mapping":
        return [
            ("alias_map_contains", False, "term"),
            ("raw_data_contains", True, "keyword"),
        ]
    if kind == "evaluator_error":
        return [
            ("evaluator_error_detected", True, ""),
        ]
    if kind == "version_unknown":
        return [
            ("trace_source_snapshot_known", False, ""),
        ]
    return []



def _sheet_satisfies_required(
    sheet: Dict[str, Any],
    fact_type: str,
    expected: bool,
    param_field: str,
) -> bool:
    """判断 FactSheet 是否已经有事实能证明某个必需断言，不依赖 LLM 是否显式写出。"""
    if param_field in {"keyword", "term"}:
        vals = sheet.get(fact_type, {})
        return any(bool(v) == expected for v in vals.values())
    if fact_type == "plan_text_mentions_tool":
        return bool(sheet.get("plan_intents")) == expected
    if fact_type == "actual_tool_called":
        return bool(sheet.get("actual_tools")) == expected
    if fact_type in {
        "plan_tool_call_names_empty", "answer_short_circuit",
        "prompt_trace_version_known", "zero_tool_no_skip",
        "evaluator_error_detected", "trace_source_snapshot_known",
        "prompt_snapshot_clean", "code_snapshot_clean", "trace_version_clean",
        "routing_event_seen", "routing_hard_rule_hit",
        "routing_missing_required_tools", "contamination_source_found",
        "version_unknown",
    }:
        return bool(sheet.get(fact_type)) == expected
    if fact_type == "not_found_tool":
        return bool(sheet.get("not_found_tools")) == expected
    return False


def _conclusion_consistency_gate(
    report: Dict[str, Any],
    ctx: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """一致性闸门：医生的每条处方必须可被 FactSheet 中的确定事实支持。"""
    issues: List[str] = []
    sheet = _build_fact_sheet(ctx, evidence_by_order, extra_evidence)
    prescriptions = report.get("prescriptions") or []
    for i, p in enumerate(prescriptions):
        kind = str(p.get("conclusion_kind") or "")
        if not kind:
            issues.append(f"prescriptions[{i}] 缺少 conclusion_kind（可选值: {sorted(CONSISTENCY_KINDS)}）")
            continue
        if kind not in CONSISTENCY_KINDS:
            issues.append(f"prescriptions[{i}] conclusion_kind 非法: {kind}")
            continue

        assertions = p.get("fact_assertions") or []
        if not isinstance(assertions, list) or not assertions:
            issues.append(f"prescriptions[{i}] 缺少 fact_assertions 数组")
            continue

        required_patterns = _required_fact_patterns(kind)
        required_types = {ft for ft, _, _ in required_patterns}

        # 逐条断言：非必要且无法判定的事实不硬拒；必要事实无法判定或与事实矛盾则拒绝。
        for j, a in enumerate(assertions):
            fact_type = str(a.get("fact_type") or "")
            expected = a.get("expected")
            if isinstance(expected, str):
                expected = expected.strip().lower() in {"true", "1", "yes", "是"}
            actual = _assertion_actual(sheet, a)
            if actual is None:
                if fact_type in required_types:
                    issues.append(
                        f"prescriptions[{i}].fact_assertions[{j}] 无法从确定证据中判定: {a}"
                    )
                # 非必要断言无法判定时不拒绝，留下但不作为硬证据。
            elif bool(actual) != bool(expected):
                issues.append(
                    f"prescriptions[{i}].fact_assertions[{j}] 与事实矛盾: "
                    f"expected={expected} actual={actual}, assertion={a}"
                )

        # 规则要求的断言组合必须齐全；若 LLM 没显式写，
        # 但只要 FactSheet 已能确定该事实且符合 expected，也视为满足。
        for fact_type, expected, param_field in required_patterns:
            matched = False
            for a in assertions:
                if str(a.get("fact_type") or "") != fact_type:
                    continue
                if bool(a.get("expected")) != expected:
                    continue
                if param_field:
                    key = str((a.get("params") or {}).get(param_field) or "")
                    if not key:
                        continue
                matched = True
                break
            if not matched and _sheet_satisfies_required(sheet, fact_type, expected, param_field):
                matched = True
            if not matched:
                issues.append(
                    f"prescriptions[{i}] 结论类型 {kind} 缺少必要断言: "
                    f"fact_type={fact_type}, expected={expected}"
                )

    # 结论与 CausalResolver 对齐：流水线已经给出权威归因，医生不能另立结论。
    resolution = ctx.get("resolution") or {}
    expected_kind = str(resolution.get("conclusion_kind") or "")
    if expected_kind and expected_kind != "other":
        kinds = [str(p.get("conclusion_kind") or "") for p in prescriptions]
        if expected_kind not in kinds:
            issues.append(
                f"CausalResolver 判定为 {expected_kind}，但所有 prescription 的 conclusion_kind={kinds}；"
                "必须至少一条处方与确定性归因一致"
            )

    return {"valid": not issues, "issues": issues, "fact_sheet": sheet}


def _assign_evidence_levels(
    report: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> None:
    """三层证据等级：L1=纯推断，L2=结构化重放/间接证据，L4=直接源码/KB/工具验证。"""
    direct_categories = {"missing_keyword", "forbidden_keyword", "not_found_tool", "stage_knowledge_truth", "stage_routing", "stage_tool_execution", "stage_version"}
    direct_tools = {
        "search_knowledge_base", "inspect_aliases", "read_project_file",
        "grep_project", "read_system_prompt",
    }
    levels: List[str] = []
    for p in report.get("prescriptions") or []:
        eids = set(p.get("evidence_ids") or [])
        direct = False
        indirect = False
        for oid in eids:
            for ev in evidence_by_order.get(oid, []):
                if not ev.get("ok"):
                    continue
                cat = ev.get("category")
                if cat in direct_categories:
                    direct = True
                elif cat:
                    indirect = True
        for ev in extra_evidence:
            if ev.get("id") in eids and ev.get("ok") is not False:
                if ev.get("tool") in direct_tools:
                    direct = True
        level = "L4" if direct else ("L2" if indirect else "L1")
        p["evidence_level"] = level
        levels.append(level)
    diag = report.get("diagnosis") or {}
    diag["evidence_level"] = "L4" if "L4" in levels else ("L2" if levels else "L1")


def prescribe_run_case(
    run_id: str,
    case_id: str,
    project_path: str = DEFAULT_PROJECT_PATH,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """项目医生入口。返回完整诊断过程与处方。"""
    global DOCTOR_MODEL
    if model:
        DOCTOR_MODEL = model
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        raise ValueError(str(e))

    run = get_run(run_id)
    if not run:
        raise ValueError(f"Run 不存在: {run_id}")
    result = next((r for r in run.get("results", []) if r.get("case_id") == case_id), None)
    if not result:
        raise ValueError(f"Run {run_id} 中不存在 case {case_id}")

    cases = list_test_cases()
    case = next((c for c in cases if c.get("case_id") == case_id), None)
    if not case:
        raise ValueError(f"测试用例不存在: {case_id}")

    trace = get_trace(result.get("trace_id") or "") if result.get("trace_id") else None
    prompt_compliance = check_prompt_compliance(trace, project_path) if trace else {"passed": None, "violations": [], "evidence": "无 Trace"}
    orders = generate_pipeline_orders(result, case, trace, prompt_compliance)

    ctx: Dict[str, Any] = {
        "run_id": run_id,
        "case_id": case_id,
        "project_path": project_path,
        "result": result,
        "case": case,
        "trace": trace,
        "prompt_compliance": prompt_compliance,
        "lab_orders": orders,
        "project_map": format_project_map(generate_project_map(project_path)),
        "verified_claims": [],
        "pinned_facts": [],
    }

    # 迭代后架构：编排器先确定性地跑完 8 阶段探针并归因，
    # LLM 只负责在判定层内解释与开处方，不再决定“查什么”。
    pipeline = run_diagnostic_pipeline(ctx)
    evidence_by_order = pipeline["evidence_by_order"]
    extra_evidence: List[Dict[str, Any]] = []
    fact_sheet = build_fact_sheet_pipeline(ctx, evidence_by_order, extra_evidence)
    resolution = resolve_cause(fact_sheet, ctx)
    ctx["fact_sheet"] = fact_sheet
    ctx["resolution"] = resolution
    ctx["pipeline_text"] = format_pipeline_for_prompt(ctx, fact_sheet, resolution)
    ctx["pipeline_result"] = pipeline

    api_keys = _api_keys(project_path)
    if not api_keys:
        return {"ok": False, "error": "缺少 DASHSCOPE_API_KEY", "lab_orders": orders}

    tools = [t for t in llm_tool_definitions() if (t.get("function") or {}).get("name") != "run_lab_check"] + [
        {
            "type": "function",
            "function": {
                "name": "record_verified_claim",
                "description": "记录一条已通过证据验证的事实。claim 必须能在 evidence_ids 对应证据原文中找到字面术语，否则会被拒绝。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "已验证事实的一句话描述"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "支撑该事实的证据 ID，如 LO-001、EXT-001",
                        },
                    },
                    "required": ["claim", "evidence_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pin_fact",
                "description": "固定一条高信号事实到医生长期记忆；必须绑定一个真实证据 ID，且文本需在该证据原文中有字面支撑。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "要固定的事实"},
                        "evidence_id": {"type": "string", "description": "支撑证据 ID"},
                    },
                    "required": ["text", "evidence_id"],
                },
            },
        },
    ]
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(ctx)},
        {"role": "user", "content": f"诊断对象 run={run_id} case={case_id}。8 阶段证据与 CausalResolver 归因已就绪，请输出最终医嘱 JSON。"},
    ]
    final_report: Optional[Dict[str, Any]] = None
    last_content = ""

    for turn in range(1, MAX_LLM_TURNS + 1):
        # 每轮把最新 Verified Claims / Pinned Facts 刷入系统提示词，保持长期记忆。
        messages[0]["content"] = _build_system_prompt(ctx)
        try:
            resp = _call_llm(messages, tools, api_keys)
            msg = resp["choices"][0].get("message") or {}
        except Exception as e:
            return {
                "ok": False,
                "error": f"医生 LLM 调用失败: {e}",
                "lab_orders": orders,
                "coverage": coverage_status(orders, evidence_by_order),
                "evidence_by_order": evidence_by_order,
                "extra_evidence": extra_evidence,
            }

        content = str(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") or []

        if tool_calls:
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                result = _execute_tool(name, args, ctx, evidence_by_order, extra_evidence)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or f"tc-{turn}",
                    "content": _tool_content(result),
                })
            continue

        # LLM 想结束：先过覆盖闸门
        last_content = content
        missing = [o for o in orders if not any(e.get("ok") for e in evidence_by_order.get(o["id"], []))]
        if missing:
            filled = _autofill_missing(ctx, evidence_by_order)
            digest = "\n".join(
                f"{f['order_id']}: {'OK' if f['ok'] else 'ERROR'} {f.get('summary') or ''}"
                for f in filled
            )
            messages.append({
                "role": "user",
                "content": (
                    f"你试图在检查单未完成时结束，编排器已自动补齐以下检查：\n{digest}\n"
                    "现在证据库已完整。请基于全部证据重新输出最终医嘱 JSON，不要再提前结束。"
                ),
            })
            continue

        # 覆盖已 100%：尝试解析最终 JSON
        report = _extract_json(content)
        validation = validate_prescriptions(report or {}, orders, evidence_by_order, extra_evidence)
        grounding = _final_grounding_check(report or {}, evidence_by_order, extra_evidence)
        near_miss = _near_miss_gate(report or {}, evidence_by_order, extra_evidence)
        consistency = _conclusion_consistency_gate(report or {}, ctx, evidence_by_order, extra_evidence)
        if report:
            _assign_evidence_levels(report, evidence_by_order, extra_evidence)
        combined_issues = validation["issues"] + grounding["issues"] + near_miss["issues"] + consistency["issues"]
        if validation["valid"] and grounding["valid"] and near_miss["valid"] and consistency["valid"] and report:
            final_report = report
            final_report["_coverage"] = coverage_status(orders, evidence_by_order)
            final_report["_grounding"] = {"valid": True, "issues": []}
            final_report["_near_miss"] = {"valid": True, "issues": []}
            final_report["_consistency"] = {"valid": True, "issues": []}
            break

        feedback = (
            "你输出的内容不是符合 schema、证据链闭合、逐字接地（grounded）、通过近成功闸门且结论一致性的 JSON。校验结果："
            + json.dumps({"validation": validation, "grounding": grounding, "near_miss": near_miss, "consistency": consistency, "issues": combined_issues}, ensure_ascii=False)
        )
        if not content:
            feedback = "你没有输出任何内容。请输出最终医嘱 JSON。"
        messages.append({"role": "user", "content": feedback})

    if final_report is None:
        final_report = _fallback_report(result, evidence_by_order, extra_evidence=extra_evidence, resolution=ctx.get("resolution"), fact_sheet=ctx.get("fact_sheet"))

    return {
        "ok": True,
        "run_id": run_id,
        "case_id": case_id,
        "model": DOCTOR_MODEL,
        "created_at": datetime.now().isoformat(),
        "report": final_report,
        "lab_orders": orders,
        "coverage": coverage_status(orders, evidence_by_order),
        "fact_sheet": fact_sheet,
        "resolution": resolution,
        "pipeline": pipeline,
        "evidence_by_order": evidence_by_order,
        "extra_evidence": extra_evidence,
        "verified_claims": ctx.get("verified_claims", []),
        "pinned_facts": ctx.get("pinned_facts", []),
    }
