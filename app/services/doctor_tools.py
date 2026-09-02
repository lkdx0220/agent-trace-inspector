# -*- coding: utf-8 -*-
"""项目医生只读工具集 + 确定性 run_lab_check。

铁律：
1. 所有证据只能由工具函数返回，LLM 不能编造；
2. 对原项目的一切探查都是只读（read_text / grep / 子进程导入原项目查询函数）；
3. 知识库检索通过子进程跑原项目代码，不把原项目 import 进 inspector 进程。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.system_prompts import (
    get_answer_system_prompt,
    get_plan_system_prompt,
    get_tool_requirement_excerpt,
)

INSPECTOR_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROJECT_PATH = "C:/Users/24701/Desktop/原神剧情/CASE-原神剧情助手-修改用"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "data", "logs"}
MAX_FILE_CHARS = 30000
MAX_GREP_HITS = 50


def now_iso() -> str:
    return datetime.now().isoformat()


def _walk_spans(span: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(span, dict):
        return out
    out.append(span)
    for child in span.get("children", []) or []:
        out.extend(_walk_spans(child))
    return out


def collect_tool_spans(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trace:
        return []
    return [s for s in _walk_spans(trace.get("root_span") or {}) if s.get("span_type") == "tool"]


def _tool_text(spans: List[Dict[str, Any]]) -> str:
    parts = []
    for s in spans:
        full = s.get("result_full") or s.get("result_preview") or ""
        if full:
            parts.append(str(full))
    return "\n".join(parts)


def _events(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in (trace or {}).get("trace_events") or [] if isinstance(e, dict)]


def _find_where(keyword: str, trace: Optional[Dict[str, Any]], answer: str) -> Dict[str, Any]:
    """确定性定位一个关键词在 trace 各阶段的位置。"""
    hits = {"tool_results": False, "final_answer": False, "plan_text": False, "events": []}
    kw = str(keyword)
    if kw and kw in str(answer or ""):
        hits["final_answer"] = True
    if kw:
        if kw in _tool_text(collect_tool_spans(trace)):
            hits["tool_results"] = True
        for ev in _events(trace):
            text = json.dumps(ev, ensure_ascii=False)
            if kw in text:
                hits["events"].append(ev.get("event"))
                data = ev.get("data") or {}
                if "execution_plan" in data and kw in str(data["execution_plan"]):
                    hits["plan_text"] = True
    return hits


def _short_circuit_answer(answer: str) -> Optional[str]:
    """识别常见的回答阶段短路串。"""
    patterns = [
        "当前知识库未收录", "未找到「", "未找到\"", "知识库未收录", "无法回答",
    ]
    for p in patterns:
        if p in answer:
            return p
    return None


# ============================================================
# 原项目只读探查（子进程）
# ============================================================

def _load_env(project_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env_file = project_path / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _run_probe(project_path: Path, code: str, timeout: int = 240) -> Dict[str, Any]:
    tmp_dir = INSPECTOR_ROOT / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    probe_file = tmp_dir / f"doctor_probe_{uuid.uuid4().hex}.py"
    marker = f"@@DOCTOR_PROBE_{uuid.uuid4().hex}@@"
    print_line = (
        "print("
        + json.dumps(marker)
        + " + json.dumps(out, ensure_ascii=False) + "
        + json.dumps(marker)
        + ")"
    )
    probe_file.write_text(code + "\n" + print_line + "\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(probe_file)],
            cwd=str(project_path),
            env=_load_env(project_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        stdout = result.stdout or ""
        if marker in stdout:
            left = stdout.split(marker, 1)[1]
            payload = left.rsplit(marker, 1)[0]
            try:
                return {"ok": True, "data": json.loads(payload)}
            except Exception as e:
                return {"ok": False, "error": f"probe JSON 解析失败: {e}", "stdout_tail": stdout[-1500:]}
        return {
            "ok": False,
            "error": f"probe 未输出结果 marker（rc={result.returncode}）",
            "stderr_tail": (result.stderr or "")[-1500:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"probe 超时（>{timeout}s）"}
    finally:
        try:
            probe_file.unlink()
        except Exception:
            pass


def kb_probe_contains(probe: Dict[str, Any], keyword: str) -> bool:
    """判断知识库检索 probe 的返回正文是否真的命中关键词。

    不能直接在整个 probe JSON 上做子串匹配：hybrid_search 的输出头部/查询键
    总是包含 query 本身，会造成“检索到了”的假阳性。这里跳过 header line，
    只检查结果正文；单行的“未找到/没有找到”消息不算命中。
    """
    if not keyword or not probe.get("ok"):
        return False
    data = probe.get("data") or {}
    queries = data.get("queries") or {}
    for q, raw in queries.items():
        text = str(raw)
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        # 单行未找到消息：不算命中
        if len(lines) <= 1 and ("未找到" in text or "没有找到" in text):
            continue
        # 跳过第一行 header，检查结果正文
        body = "\n".join(lines[1:])
        if keyword in body:
            return True
    return False


def keyword_retrieval_conclusion(keyword: str, where: Dict[str, Any], probe: Dict[str, Any]) -> str:
    """把缺失关键词的证据整理成一句确定性结论，防止 LLM 把“工具返回没有”
    误写成“知识库没有/任务未被召回”。"""
    tool_hit = bool(where.get("tool_results"))
    final_hit = bool(where.get("final_answer"))
    kb_hit = kb_probe_contains(probe, keyword)
    if tool_hit and final_hit:
        return "工具返回和最终答案都包含该关键词，说明评测/证据链无问题"  # 理论不会出现在 missing 检查
    if tool_hit and not final_hit:
        return "工具返回已包含该关键词，但最终答案未使用 → 回答阶段未整合/漏用"
    if kb_hit and not tool_hit:
        return "知识库检索能命中该关键词，但本次 Trace 的工具返回未包含 → 查询词/召回/切片问题，不是知识库缺失"
    if not kb_hit:
        return "知识库检索也未命中该关键词 → 可能知识库确实缺此信息，或需要更规范的专名/同义查询词；不能仅凭本次搜索断言“数据库不存在”"
    return "关键词证据待人工复核"


def search_knowledge_base(project_path: str, query: str, top_k: int = 6) -> Dict[str, Any]:
    """子进程调用原项目 hybrid_search（只读）。"""
    code = (
        "import sys, json, os\n"
        + "PROJECT = " + json.dumps(str(Path(project_path))) + "\n"
        + "sys.path.insert(0, PROJECT)\n"
        + "os.chdir(PROJECT)\n"
        + "from app.retrieval import hybrid_search\n"
        + "out = {'queries': {}}\n"
        + "for q in " + json.dumps([query], ensure_ascii=False) + ":\n"
        + "    try:\n"
        + "        out['queries'][q] = hybrid_search.invoke({'query': q, 'top_k': " + str(int(top_k)) + "})\n"
        + "    except Exception as e:\n"
        + "        out['queries'][q] = 'ERROR: ' + repr(e)\n"
    )
    return _run_probe(Path(project_path), code)


def inspect_aliases(project_path: str, term: str) -> Dict[str, Any]:
    """子进程读取原项目 character_aliases.py 并执行 resolve_aliases（只读）。"""
    code = (
        "import sys, os, json\n"
        + "PROJECT = " + json.dumps(str(Path(project_path))) + "\n"
        + "sys.path.insert(0, PROJECT)\n"
        + "os.chdir(PROJECT)\n"
        + "from character_aliases import ALIAS_MAP, resolve_aliases\n"
        + "term = " + json.dumps(str(term), ensure_ascii=False) + "\n"
        + "out = {\n"
        + "    'term': term,\n"
        + "    'canonical': ALIAS_MAP.get(term),\n"
        + "    'variants': resolve_aliases(term),\n"
        + "}\n"
    )
    return _run_probe(Path(project_path), code)


# ============================================================
# LLM 可直接调用的只读工具
# ============================================================

def _safe_relative(project_root: Path, rel: str) -> Optional[Path]:
    p = (project_root / rel).resolve()
    try:
        p.relative_to(project_root.resolve())
    except ValueError:
        return None
    return p


def read_project_file(project_path: str, rel_path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
    root = Path(project_path).resolve()
    path = _safe_relative(root, rel_path)
    if path is None or not path.exists() or not path.is_file():
        return {"ok": False, "error": f"文件不存在或越界: {rel_path}"}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return {"ok": False, "error": f"读取失败: {e}"}
    start = max(1, int(start_line))
    end = min(len(lines), int(end_line)) if end_line and int(end_line) > 0 else len(lines)
    if start > len(lines):
        return {"ok": False, "error": f"起始行超过文件长度 {len(lines)}"}
    body = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start - 1, end))
    return {
        "ok": True,
        "path": rel_path,
        "line_start": start,
        "line_end": end,
        "total_lines": len(lines),
        "content": body[:MAX_FILE_CHARS],
        "truncated": len(body) > MAX_FILE_CHARS,
    }


def grep_project(project_path: str, pattern: str, rel_path: str = "") -> Dict[str, Any]:
    root = Path(project_path).resolve()
    hits: List[Dict[str, Any]] = []
    if not pattern:
        return {"ok": False, "error": "pattern 不能为空"}
    regex = re.compile(re.escape(pattern), re.IGNORECASE)
    if rel_path:
        cand = _safe_relative(root, rel_path)
        candidates = [cand] if cand else []
    else:
        candidates = [root]
    for cand in candidates:
        targets = [cand] if cand.is_file() else cand.rglob("*")
        for p in targets:
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix not in {".py", ".txt", ".json", ".md"}:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    try:
                        rel = p.relative_to(root).as_posix()
                    except ValueError:
                        rel = str(p)
                    hits.append({"path": rel, "line": i, "content": line.strip()[:200]})
                    if len(hits) >= MAX_GREP_HITS:
                        return {"ok": True, "pattern": pattern, "hits": hits, "truncated": True}
    return {"ok": True, "pattern": pattern, "hits": hits, "truncated": False}


def read_system_prompt(project_path: str, name: str) -> Dict[str, Any]:
    name = (name or "").lower()
    if name in {"plan", "agent_system_v4_plan", "规划"}:
        return {"ok": True, "name": "agent_system_v4_plan", "content": get_plan_system_prompt(project_path)}
    if name in {"answer", "agent_system_v4_answer", "回答"}:
        return {"ok": True, "name": "agent_system_v4_answer", "content": get_answer_system_prompt(project_path)}
    if name in {"tool_rule", "tool_requirement", "工具规则"}:
        return {"ok": True, "name": "tool_requirement_excerpt", "content": get_tool_requirement_excerpt(project_path)}
    return {"ok": False, "error": f"未知系统提示词: {name}（可选 plan/answer/tool_rule）"}


# ============================================================
# 证据回看工具（只读，不产生新证据）
# ============================================================

def _evidence_items(
    evidence_by_order: Optional[Dict[str, List[Dict[str, Any]]]],
    extra_evidence: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """把检查单证据与额外调查证据统一成可搜索/可回看的记录列表。"""
    items: List[Dict[str, Any]] = []
    for oid, evs in (evidence_by_order or {}).items():
        for ev in evs:
            if not ev.get("ok"):
                continue
            items.append({
                "id": oid,
                "tool": "run_lab_check",
                "args": {"lab_order_id": oid},
                "status": 0,
                "summary": str(ev.get("summary") or ""),
                "content": json.dumps(ev, ensure_ascii=False),
            })
    for ev in extra_evidence or []:
        if ev.get("ok") is False:
            continue
        items.append({
            "id": str(ev.get("id") or ""),
            "tool": str(ev.get("tool") or ""),
            "args": ev.get("args") or {},
            "status": 0,
            "summary": (ev.get("result") or {}).get("summary", "") if isinstance(ev.get("result"), dict) else "",
            "content": json.dumps(ev.get("result") or ev, ensure_ascii=False),
        })
    return items


def _evidence_text(ev_items: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{item['id']}: {item['content']}" for item in ev_items)


def evidence_search(
    query: str,
    evidence_by_order: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    extra_evidence: Optional[List[Dict[str, Any]]] = None,
    evidence_id: str = "",
    context_chars: int = 180,
    limit: int = 12,
) -> Dict[str, Any]:
    """在已记录的 evidence（检查单 + EXT 证据）中做子串搜索，返回带上下文的片段。"""
    pattern = str(query or "").strip()
    if not pattern:
        return {"ok": False, "error": "evidence_search 需要非空 query"}
    items = _evidence_items(evidence_by_order, extra_evidence)
    if evidence_id:
        items = [it for it in items if it["id"] == evidence_id]
    hits: List[Dict[str, Any]] = []
    max_hits = max(1, min(50, int(limit or 12)))
    ctx = max(0, min(2000, int(context_chars or 180)))
    for item in items:
        raw = item["content"]
        lower = raw.lower()
        needle = pattern.lower()
        start = 0
        while True:
            idx = lower.find(needle, start)
            if idx < 0:
                break
            s = max(0, idx - ctx)
            e = min(len(raw), idx + len(pattern) + ctx)
            hits.append({
                "evidence_id": item["id"],
                "tool": item["tool"],
                "offset": idx,
                "snippet": raw[s:e],
            })
            start = idx + max(1, len(needle))
            if len(hits) >= max_hits:
                break
        if len(hits) >= max_hits:
            break
    if not hits:
        return {"ok": True, "query": pattern, "hits": [], "summary": f"证据中未找到「{pattern}」"}
    return {
        "ok": True,
        "query": pattern,
        "hits": hits,
        "summary": f"证据搜索命中 {len(hits)} 处",
        "hits_text": "\n\n---\n\n".join(
            f"{h['evidence_id']}@{h['offset']} tool={h['tool']}\n{h['snippet']}" for h in hits[:12]
        ),
    }


def evidence_view(
    evidence_id: str,
    evidence_by_order: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    extra_evidence: Optional[List[Dict[str, Any]]] = None,
    offset: int = 0,
    limit: int = 6000,
) -> Dict[str, Any]:
    """分页查看一条完整 evidence 的原始内容（只读）。"""
    items = _evidence_items(evidence_by_order, extra_evidence)
    target = next((it for it in items if it["id"] == evidence_id), None)
    if target is None:
        return {"ok": False, "error": f"未找到证据 {evidence_id}"}
    raw = target["content"]
    start = max(0, int(offset or 0))
    try:
        max_chars = int(limit)
    except (TypeError, ValueError):
        max_chars = 6000
    if max_chars <= 0:
        max_chars = max(0, len(raw) - start)
    chunk = raw[start:start + max_chars]
    end = start + len(chunk)
    return {
        "ok": True,
        "evidence_id": evidence_id,
        "tool": target["tool"],
        "raw_size": len(raw),
        "offset": start,
        "end": end,
        "next_offset": end if end < len(raw) else None,
        "content": chunk,
    }


# ============================================================
# 确定性 run_lab_check
# ============================================================

def _plan_signal(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从 trace_events 提取 plan 相关信号。"""
    signal: Dict[str, Any] = {
        "plan_events": 0,
        "plan_retry_events": 0,
        "execution_plans": [],
        "tool_call_names": [],
        "tool_skip_reason": None,
    }
    for ev in _events(trace):
        event = ev.get("event")
        data = ev.get("data") or {}
        if event == "plan":
            signal["plan_events"] += 1
            plan_text = data.get("execution_plan") or data.get("plan_text") or ""
            if plan_text:
                signal["execution_plans"].append(str(plan_text)[:1200])
            names = data.get("tool_call_names") or []
            if names:
                signal["tool_call_names"].extend(str(n) for n in names)
            if data.get("tool_skip_reason"):
                signal["tool_skip_reason"] = data.get("tool_skip_reason")
        if event and "retry" in str(event):
            signal["plan_retry_events"] += 1
    signal["metadata"] = (trace or {}).get("metadata") or {}
    return signal


def run_lab_check(order_id: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """执行一条确定性检查单，返回结构化证据。此函数只能由编排器/LLM 调用，
    其返回内容才会被写入证据库。"""
    orders = {o["id"]: o for o in (ctx.get("lab_orders") or [])}
    order = orders.get(order_id)
    if not order:
        return {"ok": False, "order_id": order_id, "status": "error", "summary": f"未知检查单 {order_id}", "data": {}}

    trace = ctx.get("trace")
    result = ctx.get("result") or {}
    case = ctx.get("case") or {}
    answer = str(result.get("answer") or "")
    question = str(result.get("question") or case.get("question") or "")
    project_path = str(ctx.get("project_path") or DEFAULT_PROJECT_PATH)
    category = order.get("category")
    params = order.get("params") or {}
    tool_spans = collect_tool_spans(trace)
    spans = _walk_spans((trace or {}).get("root_span") or {}) if trace else []
    data: Dict[str, Any] = {}
    summary = ""

    try:
        if category == "trace_replay":
            calls = []
            for s in tool_spans:
                full = s.get("result_full") or s.get("result_preview") or ""
                calls.append({
                    "span_id": s.get("span_id"),
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "args": s.get("tool_args") or {},
                    "result_length": s.get("result_length"),
                    "preview": str(full)[:300],
                })
            answer_ev = next((e for e in _events(trace) if e.get("event") == "answer_end"), None)
            data = {
                "tool_count": len(calls),
                "tool_calls": calls,
                "answer_end_event": {k: (str(v)[:200]) for k, v in ((answer_ev or {}).get("data") or {}).items()},
                "final_answer": answer[:500],
            }
            summary = f"共 {len(calls)} 个工具 span；" + ("；".join(f"{c['name']}={c['status']}" for c in calls) or "无工具调用")

        elif category == "plan_intent":
            signal = _plan_signal(trace)
            data = signal
            summary = (
                f"plan 事件 {signal['plan_events']} 次，plan_retry {signal['plan_retry_events']} 次，"
                f"tool_call_names={signal['tool_call_names'] or '[]'}，"
                f"tool_skip_reason={signal['tool_skip_reason']}"
            )

        elif category == "prompt_rule":
            data = {
                "plan_tool_rule": get_tool_requirement_excerpt(project_path),
                "answer_prompt_head": get_answer_system_prompt(project_path)[:2500],
            }
            summary = "已读取规划工具规则节选与回答提示词开头"

        elif category == "missing_keyword":
            kw = params.get("keyword", "")
            where = _find_where(kw, trace, answer)
            probe = search_knowledge_base(project_path, f"{question} {kw}", top_k=5)
            kb_hit = kb_probe_contains(probe, kw)
            conclusion = keyword_retrieval_conclusion(kw, where, probe)
            data = {
                "keyword": kw,
                "where": where,
                "kb_probe": probe,
                "kb_hit": kb_hit,
                "retrieval_conclusion": conclusion,
            }
            summary = f"「{kw}」工具返回={where['tool_results']}，最终答案={where['final_answer']}，知识库检索命中={kb_hit}；{conclusion}"

        elif category == "forbidden_keyword":
            kw = params.get("keyword", "")
            where = _find_where(kw, trace, answer)
            data = {"keyword": kw, "where": where}
            summary = f"「{kw}」工具返回={where['tool_results']}，最终答案={where['final_answer']}"

        elif category == "not_found_tool":
            name = params.get("tool_name") or ""
            args = params.get("tool_args") or {}
            first_arg = ""
            for v in args.values():
                if isinstance(v, str) and v:
                    first_arg = v
                    break
            probe = search_knowledge_base(project_path, first_arg or question, top_k=6)
            aliases = None
            if name == "query_character" and first_arg:
                aliases = inspect_aliases(project_path, first_arg)
            data = {
                "tool_name": name,
                "tool_args": args,
                "recheck_query": first_arg or question,
                "kb_probe": probe,
                "alias_probe": aliases,
            }
            summary = f"复检 {name}({first_arg or '?'})：知识库检索与别名解析已完成"

        elif category == "prompt_violation":
            signal = _plan_signal(trace)
            data = {
                "plan_signal": signal,
                "prompt_pass": result.get("prompt_pass"),
                "prompt_violations": result.get("prompt_violations") or [],
                "tool_count": len(tool_spans),
            }
            summary = f"tool_count={len(tool_spans)}，plan={signal['plan_events']}次，retry={signal['plan_retry_events']}次，skip_reason={signal['tool_skip_reason']}"

        elif category == "zero_tool":
            events = [{"event": e.get("event"), "data_keys": list((e.get("data") or {}).keys())} for e in _events(trace)]
            data = {"events": events, "span_types": [s.get("span_type") for s in spans], "metadata": (trace or {}).get("metadata") or {}}
            summary = f"确认 0 个工具 span；事件序列={[e['event'] for e in events]}"

        elif category == "answer_integrity":
            corpus = _tool_text(tool_spans)
            if corpus and answer:
                def grams(s: str) -> set:
                    return set(s[i:i + 4] for i in range(max(0, len(s) - 3)))
                overlap = grams(answer) & grams(corpus)
                ratio = len(overlap) / max(1, len(grams(answer)))
            else:
                ratio = 0.0
            short = _short_circuit_answer(answer)
            data = {
                "tool_corpus_chars": len(corpus),
                "answer_chars": len(answer),
                "4gram_overlap_ratio": round(float(ratio), 4),
                "short_circuit_pattern": short,
                "tool_success_with_corpus": bool(corpus) and len(corpus) > 100,
            }
            summary = f"答案与工具返回 4-gram 重合率={ratio:.2%}，短路串={short}"

        elif category == "generic_failure":
            data = {
                "events": [{"event": e.get("event"), "data": (e.get("data") or {})} for e in _events(trace)][:20],
                "tool_spans": [s.get("name") for s in tool_spans],
                "result": {k: v for k, v in result.items() if k != "answer"},
            }
            summary = "已完成失败兜底重放"

        else:
            return {"ok": False, "order_id": order_id, "status": "error", "summary": f"未支持的检查类别: {category}", "data": {}}

        return {
            "ok": True,
            "order_id": order_id,
            "status": "completed",
            "category": category,
            "summary": summary,
            "data": data,
            "created_at": now_iso(),
        }
    except Exception as e:
        return {"ok": False, "order_id": order_id, "status": "error", "summary": f"执行失败: {e}", "data": {}}


# ============================================================
# OpenAI function-calling 工具声明
# ============================================================

def llm_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_lab_check",
                "description": "执行一条强制检查单（LabOrder），返回结构化证据。所有 lab_order_id 都执行完并通过覆盖闸门后才能给出最终医嘱。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lab_order_id": {
                            "type": "string",
                            "description": "检查单编号，如 LO-001",
                        }
                    },
                    "required": ["lab_order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_project_file",
                "description": "只读查看原项目代码文件的具体行范围（不修改任何文件）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rel_path": {"type": "string", "description": "相对项目根目录的路径，如 app/agent/nodes.py"},
                        "start_line": {"type": "integer", "description": "起始行，默认 1"},
                        "end_line": {"type": "integer", "description": "结束行，默认 200"},
                    },
                    "required": ["rel_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_project",
                "description": "在原项目代码/提示词中搜索关键字（只读）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "要搜索的关键字"},
                        "rel_path": {"type": "string", "description": "可选，限定某个相对路径"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_system_prompt",
                "description": "只读读取原项目系统提示词（plan/answer/tool_rule）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "plan、answer 或 tool_rule"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "通过原项目 hybrid_search 检索知识库（子进程只读，结果较长时会被截断）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索词"},
                        "top_k": {"type": "integer", "description": "结果数，默认 6"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evidence_search",
                "description": "在已记录的检查单/额外证据全文里做子串搜索，返回带上下文片段。可验证某个关键词/文件名/调用是否真的出现在证据中。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要搜索的关键词"},
                        "evidence_id": {"type": "string", "description": "可选，限定某条证据 ID，如 LO-001 或 EXT-001"},
                        "context_chars": {"type": "integer", "description": "上下文长度，默认 180"},
                        "limit": {"type": "integer", "description": "最多返回命中数，默认 12"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evidence_view",
                "description": "分页查看一条已记录证据的完整原始内容（只读，不产生新证据）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "evidence_id": {"type": "string", "description": "证据 ID，如 LO-001 或 EXT-001"},
                        "offset": {"type": "integer", "description": "起始偏移，默认 0"},
                        "limit": {"type": "integer", "description": "读取字符数，默认 6000"},
                    },
                    "required": ["evidence_id"],
                },
            },
        },
    ]


def dispatch_llm_tool(
    name: str,
    args: Dict[str, Any],
    ctx: Dict[str, Any],
    evidence_by_order: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    extra_evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    project_path = str(ctx.get("project_path") or DEFAULT_PROJECT_PATH)
    if name == "run_lab_check":
        return run_lab_check(args.get("lab_order_id", ""), ctx)
    if name == "read_project_file":
        return read_project_file(
            project_path,
            args.get("rel_path", ""),
            int(args.get("start_line") or 1),
            int(args.get("end_line") or 200),
        )
    if name == "grep_project":
        return grep_project(project_path, args.get("pattern", ""), args.get("rel_path") or "")
    if name == "read_system_prompt":
        return read_system_prompt(project_path, args.get("name", ""))
    if name == "search_knowledge_base":
        return search_knowledge_base(project_path, args.get("query", ""), int(args.get("top_k") or 6))
    if name == "evidence_search":
        return evidence_search(
            args.get("query", ""),
            evidence_by_order=evidence_by_order,
            extra_evidence=extra_evidence,
            evidence_id=args.get("evidence_id", ""),
            context_chars=int(args.get("context_chars") or 180),
            limit=int(args.get("limit") or 12),
        )
    if name == "evidence_view":
        return evidence_view(
            args.get("evidence_id", ""),
            evidence_by_order=evidence_by_order,
            extra_evidence=extra_evidence,
            offset=int(args.get("offset") or 0),
            limit=int(args.get("limit") or 6000),
        )
    return {"ok": False, "error": f"未知工具: {name}"}
