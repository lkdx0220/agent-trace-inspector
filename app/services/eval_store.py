# -*- coding: utf-8 -*-
"""评测结果存储层：测试用例 / Run / RunCase。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.db import get_conn
from schemas.eval import RunRecord, TestCase


def _ensure_tables(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS test_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT UNIQUE NOT NULL,
            question TEXT,
            category TEXT,
            difficulty TEXT,
            expected_answer TEXT,
            must_contain TEXT,
            must_not_contain TEXT,
            expected_tools TEXT,
            expected_route TEXT,
            match_mode TEXT DEFAULT 'all',
            alternatives TEXT
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT,
            agent_name TEXT,
            summary_json TEXT,
            results_json TEXT
        );

        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            trace_id TEXT,
            root_cause TEXT,
            evidence_json TEXT,
            suggestion TEXT,
            confidence REAL,
            prompt_text TEXT,
            report_text TEXT,
            created_at TEXT,
            UNIQUE(run_id, case_id)
        );

        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            model TEXT,
            payload_json TEXT,
            created_at TEXT,
            UNIQUE(run_id, case_id)
        );
        """
    )
    try:
        conn.execute("ALTER TABLE diagnoses ADD COLUMN report_text TEXT")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE test_cases ADD COLUMN match_mode TEXT DEFAULT 'all'")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE test_cases ADD COLUMN alternatives TEXT")
    except Exception:
        pass
    conn.commit()


def save_test_case(case: TestCase) -> None:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO test_cases
            (case_id, question, category, difficulty, expected_answer,
             must_contain, must_not_contain, expected_tools, expected_route,
             match_mode, alternatives)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
              question=excluded.question,
              category=excluded.category,
              difficulty=excluded.difficulty,
              expected_answer=excluded.expected_answer,
              must_contain=excluded.must_contain,
              must_not_contain=excluded.must_not_contain,
              expected_tools=excluded.expected_tools,
              expected_route=excluded.expected_route,
              match_mode=excluded.match_mode,
              alternatives=excluded.alternatives
            """,
            (
                case.case_id, case.question, case.category, case.difficulty,
                case.expected_answer,
                json.dumps(case.must_contain, ensure_ascii=False),
                json.dumps(case.must_not_contain, ensure_ascii=False),
                json.dumps(case.expected_tools, ensure_ascii=False),
                case.expected_route,
                case.match_mode,
                json.dumps([a.model_dump() for a in case.alternatives], ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def import_golden_set(data: Dict[str, Any]) -> int:
    questions = data.get("questions", [])
    count = 0
    for q in questions:
        case = TestCase(
            case_id=q.get("id", ""),
            question=q.get("question", ""),
            category=q.get("category", ""),
            difficulty=q.get("difficulty", ""),
            expected_answer=q.get("reference_answer"),
            must_contain=q.get("must_contain", []) or [],
            must_not_contain=q.get("must_not_contain", []) or [],
            match_mode=q.get("match_mode", "all"),
            alternatives=q.get("alternatives", []) or [],
        )
        save_test_case(case)
        count += 1
    return count


def import_reference_yaml(path: str) -> int:
    """从 reference_answers_golden.yaml 导入完整参考答案。"""
    import yaml
    data = yaml.safe_load(open(path, encoding="utf-8"))
    conn = get_conn()
    count = 0
    try:
        _ensure_tables(conn)
        for case_id, item in data.items():
            conn.execute(
                """
                UPDATE test_cases SET
                  question = ?,
                  expected_answer = ?,
                  must_contain = ?,
                  must_not_contain = ?,
                  difficulty = ?
                WHERE case_id = ?
                """,
                (
                    item.get("question", ""),
                    item.get("answer", ""),
                    json.dumps(item.get("must_contain", []), ensure_ascii=False),
                    json.dumps(item.get("must_not_contain", []), ensure_ascii=False),
                    item.get("difficulty", ""),
                    case_id,
                ),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()
    return count


def save_diagnosis(run_id: str, case_id: str, trace_id: str, diagnosis: dict, prompt_text: str) -> None:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO diagnoses
            (run_id, case_id, trace_id, root_cause, evidence_json, suggestion, confidence, prompt_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, case_id) DO UPDATE SET
              trace_id=excluded.trace_id,
              root_cause=excluded.root_cause,
              evidence_json=excluded.evidence_json,
              suggestion=excluded.suggestion,
              confidence=excluded.confidence,
              prompt_text=excluded.prompt_text,
              created_at=excluded.created_at
            """,
            (
                run_id, case_id, trace_id,
                diagnosis.get("root_cause", ""),
                json.dumps(diagnosis.get("evidence", []), ensure_ascii=False),
                diagnosis.get("suggestion", ""),
                diagnosis.get("confidence", 0),
                prompt_text,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_diagnosis(run_id: str, case_id: str) -> Optional[dict]:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT run_id, case_id, trace_id, root_cause, evidence_json, suggestion, confidence, prompt_text, created_at FROM diagnoses WHERE run_id=? AND case_id=?",
            (run_id, case_id),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["evidence"] = json.loads(d.pop("evidence_json") or "[]")
        return d
    finally:
        conn.close()


def save_report(run_id: str, case_id: str, report_text: str) -> None:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO diagnoses (run_id, case_id, report_text, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, case_id) DO UPDATE SET
              report_text=excluded.report_text,
              created_at=excluded.created_at
            """,
            (run_id, case_id, report_text, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_report(run_id: str, case_id: str) -> Optional[str]:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT report_text FROM diagnoses WHERE run_id=? AND case_id=?",
            (run_id, case_id),
        ).fetchone()
        if not row or not row["report_text"]:
            return None
        return row["report_text"]
    finally:
        conn.close()


def list_test_cases() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        rows = conn.execute(
            "SELECT case_id, question, category, difficulty, expected_answer, must_contain, must_not_contain, expected_tools, expected_route, match_mode, alternatives FROM test_cases ORDER BY case_id"
        ).fetchall()
        return _rows_to_cases(rows)
    finally:
        conn.close()


def get_test_case(case_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute("SELECT case_id, question, category, difficulty, expected_answer, must_contain, must_not_contain, expected_tools, expected_route, match_mode FROM test_cases WHERE case_id=?", (case_id,)).fetchone()
        if not row:
            return None
        return _rows_to_cases([row])[0]
    finally:
        conn.close()


GOLDEN_SET_PATH = Path("C:/Users/24701/Desktop/原神剧情/golden_test_set.json")


def _golden_match_modes() -> Dict[str, str]:
    """从 golden_test_set.json 读取 match_mode（all / any），弥补 DB 未存储该字段。"""
    try:
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        return {
            q.get("id", ""): (q.get("match_mode") or "all")
            for q in data.get("questions", [])
        }
    except Exception:
        return {}


def _rows_to_cases(rows) -> List[Dict[str, Any]]:
    modes = _golden_match_modes()
    out = []
    for r in rows:
        d = dict(r)
        d["must_contain"] = json.loads(d.get("must_contain") or "[]")
        d["must_not_contain"] = json.loads(d.get("must_not_contain") or "[]")
        d["expected_tools"] = json.loads(d.get("expected_tools") or "[]")
        d["alternatives"] = json.loads(d.get("alternatives") or "[]")
        # golden_test_set 是 match_mode 的权威来源；不在 golden 里的自定义题回退到 DB 存储值
        d["match_mode"] = modes.get(d.get("case_id", ""), d.get("match_mode") or "all")
        out.append(d)
    return out


def save_run(record: RunRecord) -> None:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO runs (run_id, name, created_at, agent_name, summary_json, results_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              name=excluded.name,
              summary_json=excluded.summary_json,
              results_json=excluded.results_json
            """,
            (
                record.run_id,
                record.name,
                record.created_at.isoformat(),
                record.agent_name,
                json.dumps(record.model_dump(mode="json", exclude={"results"}), ensure_ascii=False),
                json.dumps([r.model_dump(mode="json") for r in record.results], ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_runs() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        rows = conn.execute("SELECT run_id, name, created_at, agent_name, summary_json, results_json FROM runs ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["summary"] = json.loads(d.pop("summary_json") or "{}")
            d["results"] = json.loads(d.pop("results_json") or "[]")
            out.append(d)
        return out
    finally:
        conn.close()


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute("SELECT run_id, name, created_at, agent_name, summary_json, results_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        d["results"] = json.loads(d.pop("results_json") or "[]")
        return d
    finally:
        conn.close()


def save_prescription(run_id: str, case_id: str, payload: Dict[str, Any], model: str = "") -> None:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        conn.execute(
            """
            INSERT INTO prescriptions (run_id, case_id, model, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, case_id) DO UPDATE SET
                model=excluded.model,
                payload_json=excluded.payload_json,
                created_at=excluded.created_at
            """,
            (run_id, case_id, model, json.dumps(payload, ensure_ascii=False), datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_prescription(run_id: str, case_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT model, payload_json, created_at FROM prescriptions WHERE run_id=? AND case_id=?",
            (run_id, case_id),
        ).fetchone()
        if not row:
            return None
        return {
            "run_id": run_id,
            "case_id": case_id,
            "model": row["model"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload_json"] or "{}"),
        }
    finally:
        conn.close()
