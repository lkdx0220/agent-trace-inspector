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

from app.services.path_guard import ensure_project_path
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


def _run_probe(
    project_path: Path,
    code: str,
    timeout: int = 240,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """运行一段只读 Python 探针。

    安全设计：用户/题目相关内容通过独立的 JSON payload 文件传给子进程，
    不再拼接到 Python 源码中，避免“代码注入/命令注入”。
    """
    tmp_dir = INSPECTOR_ROOT / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    probe_file = tmp_dir / f"doctor_probe_{uuid.uuid4().hex}.py"
    payload_file = None
    marker = f"@@DOCTOR_PROBE_{uuid.uuid4().hex}@@"
    print_line = (
        "print("
        + json.dumps(marker)
        + " + json.dumps(out, ensure_ascii=False) + "
        + json.dumps(marker)
        + ")"
    )
    # 子进程 stdout 在 Windows 默认用 GBK 编码，强制 UTF-8，避免中文探针结果乱码。
    probe_code = ["import sys\n", "sys.stdout.reconfigure(encoding='utf-8')\n"]
    if payload is not None:
        payload_file = tmp_dir / f"doctor_probe_{uuid.uuid4().hex}.json"
        payload_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        probe_code.append(
            "import json\n"
            "PAYLOAD = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        )
    probe_code.append(code)
    probe_code.append("\n" + print_line + "\n")
    probe_file.write_text("".join(probe_code), encoding="utf-8")
    cmd = [sys.executable, str(probe_file)]
    if payload_file is not None:
        cmd.append(str(payload_file))
    try:
        result = subprocess.run(
            cmd,
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
            out_payload = left.rsplit(marker, 1)[0]
            try:
                return {"ok": True, "data": json.loads(out_payload)}
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
        if payload_file is not None:
            try:
                payload_file.unlink()
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
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    payload = {"project_path": str(project_path), "queries": [str(query)], "top_k": int(top_k)}
    code = (
        "import os\n"
        "PROJECT = PAYLOAD['project_path']\n"
        "sys.path.insert(0, PROJECT)\n"
        "os.chdir(PROJECT)\n"
        "from app.retrieval import hybrid_search\n"
        "out = {'queries': {}}\n"
        "for q in PAYLOAD['queries']:\n"
        "    try:\n"
        "        out['queries'][q] = hybrid_search.invoke({'query': q, 'top_k': PAYLOAD['top_k']})\n"
        "    except Exception as e:\n"
        "        out['queries'][q] = 'ERROR: ' + repr(e)\n"
    )
    return _run_probe(Path(project_path), code, payload=payload)


def raw_kb_contains(project_path: str, keyword: str) -> Dict[str, Any]:
    """子进程只读扫描知识库原始数据文件，返回是否包含关键词及命中文件。

    用于区分“知识库真缺”与“检索/切片没暴露”：检索 probe 可能因为片段截断
    没返回关键词，但原始 JSON/Python 知识库文件中可能确实存在。
    """
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    code = (
        "import os\n"
        "import sys, os, json\n"
        + "PROJECT = " + json.dumps(str(Path(project_path))) + "\n"
        + "sys.path.insert(0, PROJECT)\n"
        + "os.chdir(PROJECT)\n"
        + "keyword = " + json.dumps(str(keyword), ensure_ascii=False) + "\n"
        + "targets = [\n"
        "    'content_data/quests_processed.json',\n"
        "    'content_data/quests_主线.json',\n"
        "    'content_data/quests_传说任务.json',\n"
        "    'content_data/quests_世界任务.json',\n"
        "    'content_data/quests_魔神任务.json',\n"
        "    'content_data/quests_活动活动.json',\n"
        "    'content_data/quests_其他任务.json',\n"
        "    'content_data/quests_彩蛋剧情.json',\n"
        "    'content_data/quests_地图事件.json',\n"
        "    'content_data/quests_委托任务.json',\n"
        "    'content_data/quests_部族纪闻.json',\n"
        "    'content_data/quests_游逸旅闻.json',\n"
        "    'content_data/lore.json',\n"
        "    'content_data/concepts.json',\n"
        "    'content_data/npcs_processed.json',\n"
        "    'content_data/npcs_wiki_details.json',\n"
        "    'genshin_knowledge_base/roles.py',\n"
        "    'genshin_knowledge_base/quests.py',\n"
        "    'genshin_knowledge_base/main_story.py',\n"
        "    'genshin_knowledge_base/regions.py',\n"
        "]\n"
        + "hits = []\n"
        + "for rel in targets:\n"
        + "    p = os.path.join(PROJECT, rel)\n"
        + "    if not os.path.exists(p):\n"
        + "        continue\n"
        + "    try:\n"
        + "        with open(p, 'r', encoding='utf-8', errors='replace') as f:\n"
        + "            text = f.read()\n"
        + "        if keyword in text:\n"
        + "            hits.append(rel)\n"
        + "    except Exception:\n"
        + "        continue\n"
        + "out = {'keyword': keyword, 'contains': bool(hits), 'hits': hits}\n"
    )
    return _run_probe(Path(project_path), code)


def inspect_aliases(project_path: str, term: str) -> Dict[str, Any]:
    """子进程读取原项目 character_aliases.py 并执行 resolve_aliases（只读）。"""
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    payload = {"project_path": str(project_path), "term": str(term)}
    code = (
        "import os\n"
        "PROJECT = PAYLOAD['project_path']\n"
        "sys.path.insert(0, PROJECT)\n"
        "os.chdir(PROJECT)\n"
        "from character_aliases import ALIAS_MAP, resolve_aliases\n"
        "term = PAYLOAD['term']\n"
        "out = {\n"
        "    'term': term,\n"
        "    'canonical': ALIAS_MAP.get(term),\n"
        "    'variants': resolve_aliases(term),\n"
        "}\n"
    )
    return _run_probe(Path(project_path), code, payload=payload)


def knowledge_probe_batch(project_path: str, queries: List[str], top_k: int = 5, timeout: int = 480) -> Dict[str, Any]:
    """一次子进程内批量执行 hybrid_search，避免每个关键词都重新加载原项目。

    返回 {"queries": {q: 原始返回文本或 "ERROR: ..."}}。
    """
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    queries = [str(q) for q in (queries or []) if str(q).strip()]
    if not queries:
        return {"ok": True, "data": {"queries": {}}}
    code = (
        "import sys, os, json\n"
        + "PROJECT = " + json.dumps(str(Path(project_path))) + "\n"
        + "sys.path.insert(0, PROJECT)\n"
        + "os.chdir(PROJECT)\n"
        + "from app.retrieval import hybrid_search\n"
        + "out = {'queries': {}}\n"
        + "for q in " + json.dumps(queries, ensure_ascii=False) + ":\n"
        + "    try:\n"
        + "        out['queries'][q] = hybrid_search.invoke({'query': q, 'top_k': " + str(int(top_k)) + "})\n"
        + "    except Exception as e:\n"
        + "        out['queries'][q] = 'ERROR: ' + repr(e)\n"
    )
    return _run_probe(Path(project_path), code, timeout=timeout)


RAW_KB_TARGETS = [
    "content_data/quests_processed.json",
    "content_data/quests_主线.json",
    "content_data/quests_传说任务.json",
    "content_data/quests_世界任务.json",
    "content_data/quests_魔神任务.json",
    "content_data/quests_活动活动.json",
    "content_data/quests_其他任务.json",
    "content_data/quests_彩蛋剧情.json",
    "content_data/quests_地图事件.json",
    "content_data/quests_委托任务.json",
    "content_data/quests_部族纪闻.json",
    "content_data/quests_游逸旅闻.json",
    "content_data/lore.json",
    "content_data/concepts.json",
    "content_data/npcs_processed.json",
    "content_data/npcs_wiki_details.json",
    "genshin_knowledge_base/roles.py",
    "genshin_knowledge_base/quests.py",
    "genshin_knowledge_base/main_story.py",
    "genshin_knowledge_base/regions.py",
]


def raw_kb_contains_multi(project_path: str, keywords: List[str], timeout: int = 300) -> Dict[str, Any]:
    """一次子进程扫描全部核心原始数据文件，返回每个关键词是否存在于原始数据。"""
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    keywords = [str(k) for k in (keywords or []) if str(k).strip()]
    if not keywords:
        return {"ok": True, "data": {}}
    code = (
        "import sys, os, json\n"
        + "PROJECT = " + json.dumps(str(Path(project_path))) + "\n"
        + "sys.path.insert(0, PROJECT)\n"
        + "os.chdir(PROJECT)\n"
        + "keywords = " + json.dumps(keywords, ensure_ascii=False) + "\n"
        + "targets = " + json.dumps(RAW_KB_TARGETS, ensure_ascii=False) + "\n"
        + "texts = {}\n"
        + "for rel in targets:\n"
        + "    p = os.path.join(PROJECT, rel)\n"
        + "    if not os.path.exists(p):\n"
        + "        continue\n"
        + "    try:\n"
        + "        with open(p, 'r', encoding='utf-8', errors='replace') as f:\n"
        + "            texts[rel] = f.read()\n"
        + "    except Exception:\n"
        + "        continue\n"
        + "out = {}\n"
        + "for kw in keywords:\n"
        + "    hits = [rel for rel, text in texts.items() if kw in text]\n"
        + "    out[kw] = {'keyword': kw, 'contains': bool(hits), 'hits': hits}\n"
    )
    return _run_probe(Path(project_path), code, timeout=timeout)


def inspect_aliases_multi(project_path: str, terms: List[str], timeout: int = 180) -> Dict[str, Any]:
    """一次子进程批量解析多个词条的别名映射。"""
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    terms = [str(t) for t in (terms or []) if str(t).strip()]
    if not terms:
        return {"ok": True, "data": {}}
    code = (
        "import sys, os, json\n"
        + "PROJECT = " + json.dumps(str(Path(project_path))) + "\n"
        + "sys.path.insert(0, PROJECT)\n"
        + "os.chdir(PROJECT)\n"
        + "from character_aliases import ALIAS_MAP, resolve_aliases\n"
        + "terms = " + json.dumps(terms, ensure_ascii=False) + "\n"
        + "out = {}\n"
        + "for term in terms:\n"
        + "    try:\n"
        + "        out[term] = {'term': term, 'canonical': ALIAS_MAP.get(term), 'variants': resolve_aliases(term)}\n"
        + "    except Exception as e:\n"
        + "        out[term] = {'term': term, 'canonical': None, 'variants': [], 'error': repr(e)}\n"
    )
    return _run_probe(Path(project_path), code, timeout=timeout)


def routing_probe(project_path: str, question: str, timeout: int = 180) -> Dict[str, Any]:
    """子进程只读读取当前 intent_router.py 的确定性路由规则。

    返回 tool_groups / hard_rule_hit / required_tools（当前版本对本题强制要求的工具集）。
    这是“当前代码”视角，是否适用于 Trace 运行时刻由版本快照单独判定。
    """
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    code = (
        "import sys, os, json\n"
        + "PROJECT = " + json.dumps(str(Path(project_path))) + "\n"
        + "sys.path.insert(0, PROJECT)\n"
        + "os.chdir(PROJECT)\n"
        + "from intent_router import TOOL_GROUPS, _TASK_METADATA_HARD_RULE\n"
        + "question = " + json.dumps(str(question), ensure_ascii=False) + "\n"
        + "out = {\n"
        + "    'tool_groups': TOOL_GROUPS,\n"
        + "    'hard_rule_hit': bool(_TASK_METADATA_HARD_RULE.search(question)),\n"
        + "    'required_tools': list(TOOL_GROUPS.get('D', [])) if _TASK_METADATA_HARD_RULE.search(question) else [],\n"
        + "}\n"
    )
    return _run_probe(Path(project_path), code, timeout=timeout)



# ============================================================
# LLM 可直接调用的只读工具
# ============================================================

def _safe_relative(project_root: Path, rel: str) -> Optional[Path]:
    """严格限制相对路径必须位于 project_root 内。

    在 resolve 前先拒绝绝对路径和 .. 段，避免路径穿越。
    """
    if not rel or not isinstance(rel, str):
        return None
    rel_clean = rel.replace("\\", "/").strip()
    if not rel_clean or Path(rel_clean).is_absolute():
        return None
    parts = [x for x in rel_clean.split("/") if x]
    if any(x == ".." for x in parts):
        return None
    # 允许形如 app/agent/nodes.py
    p = (project_root / rel_clean).resolve()
    try:
        p.relative_to(project_root.resolve())
    except ValueError:
        return None
    return p


UNSAFE_FILE_PARTS = (
    ".git", ".env", ".pem", ".key", "secret", "credential",
    "token", "auth", "node_modules", "__pycache__", ".venv", "venv",
)


def _is_unsafe_rel_path(rel_path: str) -> bool:
    low = (rel_path or "").replace("\\", "/").lower()
    return any(part in low for part in UNSAFE_FILE_PARTS)


def read_project_file(project_path: str, rel_path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
    try:
        root = ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    root = root.resolve()
    if _is_unsafe_rel_path(rel_path):
        return {"ok": False, "error": f"禁止读取敏感/非源码文件: {rel_path}"}
    path = _safe_relative(root, rel_path)
    if path is None or not path.exists() or not path.is_file():
        return {"ok": False, "error": f"文件不存在或越界: {rel_path}"}
    if _is_unsafe_rel_path(path.as_posix()):
        return {"ok": False, "error": f"禁止读取敏感/非源码文件: {rel_path}"}
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
    try:
        root = ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    root = root.resolve()
    hits: List[Dict[str, Any]] = []
    if not pattern:
        return {"ok": False, "error": "pattern 不能为空"}
    regex = re.compile(re.escape(pattern), re.IGNORECASE)
    if rel_path:
        if _is_unsafe_rel_path(rel_path):
            return {"ok": True, "pattern": pattern, "hits": [], "truncated": False}
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
            if _is_unsafe_rel_path(p.as_posix()):
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



def _project_vcs_info(project_path: str) -> Dict[str, Any]:
    """只读获取原项目当前 git 版本与工作区状态，供提示词证据标注“当前版本”。"""
    root = Path(project_path)
    head = None
    dirty = None
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            head = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            dirty = bool(r.stdout.strip())
    except Exception:
        pass
    return {
        "source": "current_working_tree",
        "git_head": head,
        "git_dirty": dirty,
        "read_at": now_iso(),
    }


# 系统提示词文件在项目根下的相对路径
def _prompt_rel_path(name: str) -> str:
    mapping = {
        "plan": "prompts/system/agent_system_v4_plan.txt",
        "answer": "prompts/system/agent_system_v4_answer.txt",
        "tool_rule": "prompts/system/agent_system_v4_plan.txt",
    }
    return mapping.get(name, "")


def read_system_prompt(project_path: str, name: str) -> Dict[str, Any]:
    """只读读取系统提示词，并标注“当前工作区版本”元数据。

    注意：返回的是诊断时刻的当前工作区文件，不一定等于 Trace 运行时的版本。
    医生不得把当前规则直接当作旧 Trace 已生效的规则来判违规。
    """
    try:
        ensure_project_path(project_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    name = (name or "").lower()
    vcs = _project_vcs_info(project_path)
    rel = _prompt_rel_path(name)
    file_meta: Dict[str, Any] = {}
    if rel:
        fp = Path(project_path) / rel
        if fp.exists():
            st = fp.stat()
            file_meta = {
                "rel_path": rel,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
            }
    if name in {"plan", "agent_system_v4_plan", "规划"}:
        return {
            "ok": True, "name": "agent_system_v4_plan",
            "content": get_plan_system_prompt(project_path),
            "source_snapshot": vcs, "file_meta": file_meta,
            "version_notice": "此文件为当前工作区版本，不一定等于 Trace 运行时的版本；判定违规前必须确认版本一致。",
        }
    if name in {"answer", "agent_system_v4_answer", "回答"}:
        return {
            "ok": True, "name": "agent_system_v4_answer",
            "content": get_answer_system_prompt(project_path),
            "source_snapshot": vcs, "file_meta": file_meta,
            "version_notice": "此文件为当前工作区版本，不一定等于 Trace 运行时的版本；判定违规前必须确认版本一致。",
        }
    if name in {"tool_rule", "tool_requirement", "工具规则"}:
        return {
            "ok": True, "name": "tool_requirement_excerpt",
            "content": get_tool_requirement_excerpt(project_path),
            "source_snapshot": vcs, "file_meta": file_meta,
            "version_notice": "此文件为当前工作区版本，不一定等于 Trace 运行时的版本；判定违规前必须确认版本一致。",
        }
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
        is_retry_event = bool(event and "retry" in str(event))
        is_retry_start = event == "llm_start" and str(data.get("role") or "") == "plan_retry"
        if is_retry_event or is_retry_start:
            signal["plan_retry_events"] += 1
    signal["metadata"] = (trace or {}).get("metadata") or {}
    return signal




# 常见工具名，用于从 plan 文本中识别“计划要调用但实际未调用”的工具意图。
KNOWN_TOOL_NAMES = {
    "hybrid_search", "query_character", "query_region", "query_story",
    "query_weapon", "query_quest", "query_monster", "query_artifact",
    "query_material", "query_food", "query_book", "query_npc",
    "load_quest_content", "search_knowledge_base",
}


def _extract_plan_tool_intents(plan_texts: List[str]) -> List[str]:
    """从 plan 文本中提取“明确写到要调用”的工具名，不做语义猜测。"""
    intents: List[str] = []
    for text in plan_texts:
        for m in re.finditer(r"(?:调用|使用|检索|查询)\s*([A-Za-z_][A-Za-z0-9_]*)", text):
            name = m.group(1)
            if name in KNOWN_TOOL_NAMES or name.startswith("query_"):
                intents.append(name)
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
            name = m.group(1)
            if name in KNOWN_TOOL_NAMES:
                intents.append(name)
    return list(dict.fromkeys(intents))


def _trace_truth_audit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """从 raw Trace 独立重算事实，不依赖评测器 reasons。

    重点解决两类盲区：
    1. plan 文本明确规划调用工具，但实际 tool_calls 为空；
    2. 评测器漏报 / 误报的可见差异（缺词、禁词、零工具违规、not_found）。
    """
    trace = ctx.get("trace")
    result = ctx.get("result") or {}
    case = ctx.get("case") or {}
    answer = str(result.get("answer") or "")
    tool_spans = collect_tool_spans(trace)
    actual = []
    for s in tool_spans:
        if s.get("name"):
            actual.append(str(s.get("name")))
    for t in (result.get("actual_tools") or []):
        if t not in actual:
            actual.append(str(t))

    signal = _plan_signal(trace)
    plan_texts = signal.get("execution_plans") or []
    intents = _extract_plan_tool_intents(plan_texts)
    plan_names = [str(n) for n in (signal.get("tool_call_names") or [])]

    plan_intent_mismatch = None
    missing = [i for i in intents if i not in actual]
    if intents and missing:
        plan_intent_mismatch = {
            "plan_intents": intents,
            "actual_tools": actual,
            "missing": missing,
            "plan_tool_call_names": plan_names,
            "note": "plan 文本明确规划调用工具，但实际工具调用缺失",
        }
    elif plan_names and not actual:
        plan_intent_mismatch = {
            "plan_intents": plan_names,
            "actual_tools": actual,
            "missing": plan_names,
            "plan_tool_call_names": plan_names,
            "note": "plan 已给出结构化 tool_call_names，但实际没有任何工具 span",
        }

    not_found_tools = []
    error_tools = []
    for s in tool_spans:
        status = s.get("status")
        item = {
            "name": s.get("name"),
            "args": s.get("tool_args") or {},
            "status": status,
        }
        if status == "not_found":
            not_found_tools.append(item)
        elif status in {"error", "failed"}:
            error_tools.append(item)

    short = _short_circuit_answer(answer)

    must_contain = case.get("must_contain") or []
    must_not = case.get("must_not_contain") or []
    must_contain_findings = []
    for kw in must_contain:
        if not kw:
            continue
        where = _find_where(kw, trace, answer)
        must_contain_findings.append({
            "keyword": kw,
            "in_answer": where["final_answer"],
            "in_tool_results": where["tool_results"],
        })
    forbidden_findings = []
    for kw in must_not:
        if not kw:
            continue
        forbidden_findings.append({
            "keyword": kw,
            "in_answer": kw in answer,
            "in_tool_results": _find_where(kw, trace, answer)["tool_results"],
        })

    reasons_text = "\n".join(result.get("reasons") or [])
    discrepancies: List[str] = []
    for kw in must_contain:
        if kw and kw not in answer and f"缺少必须包含：{kw}" not in reasons_text:
            discrepancies.append(f"评测器未报“缺少必须包含：{kw}”")
    for kw in must_not:
        if kw and kw in answer and f"出现禁止包含：{kw}" not in reasons_text:
            discrepancies.append(f"评测器未报“出现禁止包含：{kw}”")

    if plan_intent_mismatch and "违反系统提示词" not in reasons_text:
        discrepancies.append("plan 规划调用工具但实际未调用，评测器未标记系统提示词违规")
    if not tool_spans and not signal.get("tool_skip_reason") and "违反系统提示词" not in reasons_text:
        discrepancies.append("零工具调用且无 tool_skip_reason，评测器未标记系统提示词违规")
    if not_found_tools and not any("未找到" in r or "not_found" in r.lower() for r in result.get("reasons") or []):
        discrepancies.append(f"存在 {len(not_found_tools)} 个 not_found 工具，评测器 reasons 未体现")

    summary_parts = []
    if plan_intent_mismatch:
        summary_parts.append(
            "plan 文本规划调用 " + "、".join(plan_intent_mismatch["plan_intents"]) +
            "，实际工具调用=" + (str(actual) if actual else "无") +
            "；plan 结构化输出缺失"
        )
    if discrepancies:
        summary_parts.append("评测器一致性差异：" + "；".join(discrepancies[:6]))
    if not_found_tools:
        summary_parts.append("not_found 工具：" + "、".join(str(x["name"]) for x in not_found_tools))
    if short:
        summary_parts.append(f"答案存在短路串：{short}")
    if not summary_parts:
        summary_parts.append("Trace 真相重算完成，未发现 plan 意图与工具调用的明确不一致")

    return {
        "actual_tools": actual,
        "tool_count": len(tool_spans),
        "plan_intents": intents,
        "plan_tool_call_names": plan_names,
        "plan_intent_mismatch": plan_intent_mismatch,
        "not_found_tools": not_found_tools,
        "error_tools": error_tools,
        "answer_short_circuit": short,
        "must_contain_findings": must_contain_findings,
        "must_not_contain_findings": forbidden_findings,
        "evaluator_discrepancies": discrepancies,
        "trace_metadata": (trace or {}).get("metadata") or {},
        "summary": "；".join(summary_parts),
    }

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

        elif category == "trace_truth_audit":
            audit = _trace_truth_audit(ctx)
            data = audit
            summary = audit.get("summary", "")

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
            raw_probe = raw_kb_contains(project_path, kw)
            raw_hit = bool(raw_probe.get("ok") and (raw_probe.get("data") or {}).get("contains"))
            conclusion = keyword_retrieval_conclusion(kw, where, probe)
            data = {
                "keyword": kw,
                "where": where,
                "kb_probe": probe,
                "kb_hit": kb_hit,
                "raw_probe": raw_probe,
                "raw_hit": raw_hit,
                "retrieval_conclusion": conclusion,
            }
            summary = f"「{kw}」工具返回={where['tool_results']}，最终答案={where['final_answer']}，知识库检索命中={kb_hit}，原始数据命中={raw_hit}；{conclusion}"

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
            raw_probe = None
            if name == "query_character" and first_arg:
                aliases = inspect_aliases(project_path, first_arg)
            if first_arg:
                raw_probe = raw_kb_contains(project_path, first_arg)
            data = {
                "tool_name": name,
                "tool_args": args,
                "recheck_query": first_arg or question,
                "kb_probe": probe,
                "alias_probe": aliases,
                "raw_probe": raw_probe,
            }
            summary = f"复检 {name}({first_arg or '?'})：知识库检索、原始数据扫描与别名解析已完成"

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
                "description": "只读读取原项目系统提示词（plan/answer/tool_rule）。返回当前工作区版本及 git 元数据；注意该版本不一定等于 Trace 运行时的版本。",
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
                "name": "inspect_raw_kb",
                "description": "只读扫描原项目知识库原始数据文件（quests/roles/lore/npcs等），确认某个词是否真的在底层数据中。区别于 search_knowledge_base 的召回结果。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "要扫描的关键词"},
                    },
                    "required": ["keyword"],
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
    if name == "inspect_raw_kb":
        return raw_kb_contains(project_path, args.get("keyword", ""))
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
