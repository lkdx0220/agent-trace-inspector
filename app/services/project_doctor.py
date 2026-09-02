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
from app.services.lab_orders import format_lab_orders_for_prompt, generate_lab_orders
from app.services.project_map import format_project_map, generate_project_map

DASHSCOPE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DOCTOR_MODEL = os.environ.get("DOCTOR_MODEL", "qwen3.7-max")
MAX_LLM_TURNS = 12
MAX_TOOL_CONTENT_CHARS = 6000


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
    orders = ctx["lab_orders"]
    project_map = ctx["project_map"]

    schema = {
        "diagnosis": {
            "summary": "一句话结论",
            "primary_root_cause": "最可能的根因（必须区分：知识库真缺 / 检索召回失败 / 别名映射缺失 / 回答阶段未整合 / 违反系统提示词 / planner 未调工具 / 评测标准问题）",
            "issue_classification": "knowledge_base_gap|retrieval_recall|alias_mapping|answer_composition|prompt_violation|planner_tool_calling|evaluator_standard|other",
            "data_vs_recall": "knowledge_base_missing|recall_failure|answer_stage_failure|prompt_violation|other",
            "key_evidence": ["与结论直接相关的证据 ID"],
        },
        "prescriptions": [
            {
                "issue": "诊断出的问题",
                "root_cause": "根因，必须能被 evidence_ids 里的证据支持",
                "evidence_ids": ["LO-001", "EXT-001"],
                "suggestion": "具体、可执行的修改建议（指到文件/提示词/别名表）",
                "target_file": "建议修改的相对路径（只读分析，不实际修改）",
                "severity": "high|medium|low",
                "expected_effect": "修改后预期解决哪个评测项",
            }
        ],
        "confidence": 0.0,
    }

    return f"""你是“项目医生”：一位了解原神剧情助手项目架构、系统提示词与知识库实现的 AI 诊断专家。
你面对一份评测失败病例。你的任务不是再评一次分，而是：
（1）执行全部强制检查单，拿到证据；
（2）区分“知识库真的没有数据”与“数据存在但召回/别名/回答阶段失败”；
（3）开出处方（prescriptions），每条处方必须绑定证据 ID。

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
{_trace_summary(trace)}

{format_lab_orders_for_prompt(orders)}

【工具使用铁律】
1. run_lab_check 是唯一能把证据写入证据库的工具。每一条检查单都必须执行（可由你调用，缺的编排器会自动补齐）。
2. 证据 ID = lab_order_id；额外自由调查（read_project_file / grep_project / read_system_prompt / search_knowledge_base）会产生 EXT-001、EXT-002… 等 ID。
3. evidence_search / evidence_view 只读回看已记录证据，不会产生新的 EXT 证据；需要证明某句话出自证据时，应引用它所在的 LO/EXT ID，而不是重复描述。
4. record_verified_claim / pin_fact 是医生记忆工具：它们会做证据接地校验，只允许记录有真实证据支撑的事实；这些记录会随最终结果一起保存。
5. 禁止编造证据；prescription 的 evidence_ids 引用了不存在的 ID 会被覆盖闸门拒绝。
6. 最终处方会做“证据闸门”检查：prescription.root_cause 必须能在其 evidence_ids 对应证据的原文中找到至少一个 2 字以上中文术语或 4 字符以上代码词。禁止写没有任何字面证据支撑的根因。
7. 知识库检索是子进程调用原项目，结果可能较长；优先相信检查单返回的结构化证据。
8. 你只读原项目，绝不建议或执行写操作。处方是建议文本，不落盘到原项目。

【最终输出格式】
在所有检查单完成前不要输出最终答案。覆盖 100% 后，输出一个 JSON 对象（不要 markdown 围栏外的其他文字）：
{json.dumps(schema, ensure_ascii=False, indent=2)}

先开始执行 run_lab_check。{_format_memory_block(ctx)}"""


def _call_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], api_key: str) -> Dict[str, Any]:
    payload = {
        "model": DOCTOR_MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
        "stream": False,
    }
    resp = requests.post(
        DASHSCOPE_COMPAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM 调用失败 {resp.status_code}: {resp.text[:500]}")
    return resp.json()


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


def _fallback_report(result: Dict[str, Any], evidence_by_order: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    ev_summary = {
        oid: [e.get("summary") for e in evs]
        for oid, evs in evidence_by_order.items()
    }
    return {
        "diagnosis": {
            "summary": "医生 LLM 未在限定轮次内产出有效最终 JSON，已自动补齐全部检查单，以下为确定性证据。",
            "primary_root_cause": "（待人工复核）",
            "issue_classification": "other",
            "key_evidence": list(evidence_by_order.keys()),
        },
        "prescriptions": [],
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


def _final_grounding_check(
    report: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """VulnClaw 式证据闸门：医嘱中的根因必须能在所引用证据里找到字面术语。

    这不是语义匹配，而是最低成本的防幻觉：如果医生声称的根因（如
    “hybrid_search 未召回幕切”）在它引用的 LO/EXT 证据全文里连一个
    2 字词都找不到，说明该结论没有落到真实工具输出上。
    """
    issues: List[str] = []
    prescriptions = report.get("prescriptions") or []
    for i, p in enumerate(prescriptions):
        eids = p.get("evidence_ids") or []
        corpus = _evidence_corpus_for_ids(eids, evidence_by_order, extra_evidence)
        if not corpus:
            issues.append(f"prescriptions[{i}] 引用的 evidence_ids 没有对应证据内容")
            continue
        terms = _meaningful_terms(str(p.get("root_cause") or ""))
        grounded = [t for t in terms if t.lower() in corpus.lower()]
        if not grounded:
            issues.append(
                f"prescriptions[{i}] root_cause 没有任何字面证据支撑"
                f"（引用 {eids}，候选术语 {terms[:8]}）"
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


def _assign_evidence_levels(
    report: Dict[str, Any],
    evidence_by_order: Dict[str, List[Dict[str, Any]]],
    extra_evidence: List[Dict[str, Any]],
) -> None:
    """三层证据等级：L1=纯推断，L2=结构化重放/间接证据，L4=直接源码/KB/工具验证。"""
    direct_categories = {"missing_keyword", "forbidden_keyword", "not_found_tool"}
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
    orders = generate_lab_orders(result, case, trace, prompt_compliance)

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

    api_key = _api_key(project_path)
    if not api_key:
        return {"ok": False, "error": "缺少 DASHSCOPE_API_KEY", "lab_orders": orders}

    tools = llm_tool_definitions() + [
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
    evidence_by_order: Dict[str, List[Dict[str, Any]]] = {}
    extra_evidence: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(ctx)},
        {"role": "user", "content": f"请开始诊断 run={run_id} case={case_id}。先执行强制检查单。"},
    ]
    final_report: Optional[Dict[str, Any]] = None
    last_content = ""

    for turn in range(1, MAX_LLM_TURNS + 1):
        # 每轮把最新 Verified Claims / Pinned Facts 刷入系统提示词，保持长期记忆。
        messages[0]["content"] = _build_system_prompt(ctx)
        try:
            resp = _call_llm(messages, tools, api_key)
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
        if report:
            _assign_evidence_levels(report, evidence_by_order, extra_evidence)
        combined_issues = validation["issues"] + grounding["issues"] + near_miss["issues"]
        if validation["valid"] and grounding["valid"] and near_miss["valid"] and report:
            final_report = report
            final_report["_coverage"] = coverage_status(orders, evidence_by_order)
            final_report["_grounding"] = {"valid": True, "issues": []}
            final_report["_near_miss"] = {"valid": True, "issues": []}
            break

        feedback = (
            "你输出的内容不是符合 schema、证据链闭合、逐字接地（grounded）且通过近成功闸门的 JSON。校验结果："
            + json.dumps({"validation": validation, "grounding": grounding, "near_miss": near_miss, "issues": combined_issues}, ensure_ascii=False)
        )
        if not content:
            feedback = "你没有输出任何内容。请输出最终医嘱 JSON。"
        messages.append({"role": "user", "content": feedback})

    if final_report is None:
        final_report = _fallback_report(result, evidence_by_order)

    return {
        "ok": True,
        "run_id": run_id,
        "case_id": case_id,
        "model": DOCTOR_MODEL,
        "created_at": datetime.now().isoformat(),
        "report": final_report,
        "lab_orders": orders,
        "coverage": coverage_status(orders, evidence_by_order),
        "evidence_by_order": evidence_by_order,
        "extra_evidence": extra_evidence,
        "verified_claims": ctx.get("verified_claims", []),
        "pinned_facts": ctx.get("pinned_facts", []),
    }
