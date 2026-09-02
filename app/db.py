# -*- coding: utf-8 -*-
"""SQLite 存储层。P1 先用标准库 sqlite3，保持零额外依赖。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas.trace import Span, Trace

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "inspector.db"


def get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT UNIQUE NOT NULL,
                question TEXT,
                created_at TEXT,
                duration_ms INTEGER,
                execution_mode TEXT,
                intent_labels TEXT,
                response_mode TEXT,
                raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                parent_id TEXT,
                span_id TEXT NOT NULL,
                span_type TEXT,
                name TEXT,
                status TEXT,
                step_index INTEGER,
                start_time TEXT,
                end_time TEXT,
                data_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);

            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                event TEXT,
                timestamp REAL,
                data_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_trace ON trace_events(trace_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _span_to_row(trace_id: str, parent_id: Optional[str], span: Span) -> Dict[str, Any]:
    return {
        "trace_id": trace_id,
        "parent_id": parent_id,
        "span_id": span.span_id,
        "span_type": span.span_type.value,
        "name": span.name,
        "status": span.status.value,
        "step_index": span.step_index,
        "start_time": span.start_time.isoformat() if span.start_time else None,
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "data_json": span.model_dump_json(),
    }


def save_trace(trace: Trace, db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        conn.execute(
            """
            INSERT INTO traces
            (trace_id, question, created_at, duration_ms, execution_mode, intent_labels, response_mode, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                question=excluded.question,
                created_at=excluded.created_at,
                duration_ms=excluded.duration_ms,
                execution_mode=excluded.execution_mode,
                intent_labels=excluded.intent_labels,
                response_mode=excluded.response_mode,
                raw_json=excluded.raw_json
            """,
            (
                trace.trace_id,
                trace.question,
                trace.created_at.isoformat(),
                trace.duration_ms,
                trace.metadata.execution_mode,
                json.dumps(trace.metadata.intent_labels or [], ensure_ascii=False),
                trace.metadata.response_mode,
                trace.model_dump_json(indent=2),
            ),
        )

        # 清掉旧 spans，重新落库（简单 UPSERT 策略）
        conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
        conn.execute("DELETE FROM trace_events WHERE trace_id = ?", (trace.trace_id,))

        def insert_spans(span: Span, parent_id: Optional[str]) -> None:
            row = _span_to_row(trace.trace_id, parent_id, span)
            conn.execute(
                """
                INSERT INTO spans
                (trace_id, parent_id, span_id, span_type, name, status, step_index, start_time, end_time, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["trace_id"], row["parent_id"], row["span_id"],
                    row["span_type"], row["name"], row["status"],
                    row["step_index"], row["start_time"], row["end_time"],
                    row["data_json"],
                ),
            )
            for child in span.children:
                insert_spans(child, span.span_id)

        insert_spans(trace.root_span, None)

        for ev in trace.trace_events or []:
            conn.execute(
                """
                INSERT INTO trace_events (trace_id, event, timestamp, data_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    ev.get("event"),
                    ev.get("timestamp"),
                    json.dumps(ev.get("data") or {}, ensure_ascii=False),
                ),
            )

        conn.commit()
    finally:
        conn.close()


def list_traces(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            """
            SELECT trace_id, question, created_at, duration_ms, execution_mode,
                   intent_labels, response_mode
            FROM traces
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trace(trace_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        row = conn.execute("SELECT raw_json FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if not row:
            return None
        return json.loads(row["raw_json"])
    finally:
        conn.close()


def get_timeline(trace_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            """
            SELECT span_id, parent_id, span_type, name, status, step_index,
                   start_time, end_time
            FROM spans
            WHERE trace_id = ?
            ORDER BY step_index ASC
            """,
            (trace_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
