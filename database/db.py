"""SQLite database layer for ThreatLens AI."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@contextmanager
def get_connection(db_path: str = "database/threatlens.db"):
    """Context manager for SQLite connections with WAL mode."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = "database/threatlens.db"):
    """Initialize database from schema.sql."""
    schema_path = Path(__file__).parent / "schema.sql"
    with get_connection(db_path) as conn:
        conn.executescript(schema_path.read_text())


def create_investigation(inv_type: str, input_content: str) -> str:
    """Create a new investigation record. Returns investigation ID."""
    import uuid

    inv_id = f"INV-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO investigations (id, type, input_content, status)
               VALUES (?, ?, ?, 'pending')""",
            (inv_id, inv_type, input_content),
        )
    return inv_id


def update_investigation(inv_id: str, **kwargs):
    """Update investigation fields dynamically."""
    allowed = {"risk_level", "confidence", "summary", "status", "human_approved", "reviewer_note"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [inv_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE investigations SET {set_clause} WHERE id = ?", values)


def log_tool_call(inv_id: str, step_order: int, tool_name: str, arguments: dict, result: Any, duration_ms: int):
    """Record a tool execution."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO tool_calls (investigation_id, step_order, tool_name, arguments, result, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (inv_id, step_order, tool_name, json.dumps(arguments), json.dumps(result) if result else None, duration_ms),
        )


def add_evidence(inv_id: str, indicator_type: str, indicator_value: str, source_tool: str,
                 source_record: Any, veracity: str = "unknown", confidence: float = 0.0):
    """Add an evidence item."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO evidence_items (investigation_id, indicator_type, indicator_value,
                                         source_tool, source_record, veracity, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (inv_id, indicator_type, indicator_value, source_tool,
             json.dumps(source_record) if source_record else None, veracity, confidence),
        )


def add_correlation(inv_id: str, indicator_a: str, indicator_b: str, relationship: str, strength: float):
    """Add a correlation between two indicators."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO correlations (investigation_id, indicator_a, indicator_b, relationship, strength)
               VALUES (?, ?, ?, ?, ?)""",
            (inv_id, indicator_a, indicator_b, relationship, strength),
        )


def get_investigation(inv_id: str) -> Optional[dict]:
    """Get a single investigation by ID."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM investigations WHERE id = ?", (inv_id,)).fetchone()
        if not row:
            return None
        return dict(row)


def get_investigations(limit: int = 50, status: Optional[str] = None) -> list[dict]:
    """List recent investigations."""
    with get_connection() as conn:
        query = """SELECT * FROM investigations"""
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_tool_calls(inv_id: str) -> list[dict]:
    """Get all tool calls for an investigation, ordered by step."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM tool_calls WHERE investigation_id = ? ORDER BY step_order",
            (inv_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("arguments"):
                try:
                    d["arguments"] = json.loads(d["arguments"])
                except Exception:
                    pass
            if d.get("result"):
                try:
                    d["result"] = json.loads(d["result"])
                except Exception:
                    pass
            results.append(d)
        return results


def get_evidence(inv_id: str) -> list[dict]:
    """Get all evidence items for an investigation."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM evidence_items WHERE investigation_id = ? ORDER BY id",
            (inv_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("source_record"):
                try:
                    d["source_record"] = json.loads(d["source_record"])
                except Exception:
                    pass
            results.append(d)
        return results


def get_correlations(inv_id: str) -> list[dict]:
    """Get all correlations for an investigation."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM correlations WHERE investigation_id = ?",
            (inv_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_evaluation_run(run_data: dict):
    """Save an evaluation run record.

    Never crashes the evaluation if the database write fails — logs a
    warning instead so the agent can continue.
    """
    import logging

    logger = logging.getLogger("threatlens.db")
    try:
        with get_connection() as conn:
            # Ensure table exists (safety net if init_db wasn't called)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_version TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    tool_selection_correct BOOLEAN,
                    argument_accuracy REAL,
                    task_completed BOOLEAN,
                    classification_correct BOOLEAN,
                    evidence_grounding REAL,
                    tool_call_count INTEGER,
                    escalation_appropriate BOOLEAN,
                    actual_output TEXT,
                    reference_output TEXT,
                    completed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                """INSERT INTO evaluation_runs
                   (agent_version, scenario_id, tool_selection_correct, argument_accuracy,
                    task_completed, classification_correct, evidence_grounding,
                    tool_call_count, escalation_appropriate, actual_output, reference_output)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_data.get("agent_version", "unknown"),
                    run_data.get("scenario_id", "unknown"),
                    run_data.get("tool_selection_correct"),
                    run_data.get("argument_accuracy"),
                    run_data.get("task_completed"),
                    run_data.get("classification_correct"),
                    run_data.get("evidence_grounding"),
                    run_data.get("tool_call_count", 0),
                    run_data.get("escalation_appropriate"),
                    json.dumps(run_data.get("actual_output", {})),
                    json.dumps(run_data.get("reference_output", {})),
                ),
            )
    except Exception as exc:
        logger.warning(f"Failed to save evaluation run: {exc}")


def get_evaluation_stats(agent_version: Optional[str] = None) -> dict:
    """Get aggregate evaluation stats."""
    with get_connection() as conn:
        query = "SELECT * FROM evaluation_runs"
        params = []
        if agent_version:
            query += " WHERE agent_version = ?"
            params.append(agent_version)

        rows = conn.execute(query, params).fetchall()
        if not rows:
            return {"count": 0}

        total = len(rows)
        completed = sum(1 for r in rows if r["task_completed"])
        correct_class = sum(1 for r in rows if r["classification_correct"])
        correct_tool = sum(1 for r in rows if r["tool_selection_correct"])
        avg_arg_acc = sum(r["argument_accuracy"] or 0 for r in rows) / total
        avg_evidence = sum(r["evidence_grounding"] or 0 for r in rows) / total
        avg_calls = sum(r["tool_call_count"] or 0 for r in rows) / total
        correct_escalation = sum(1 for r in rows if r["escalation_appropriate"])

        return {
            "count": total,
            "completion_rate": completed / total if total else 0,
            "classification_accuracy": correct_class / total if total else 0,
            "tool_selection_accuracy": correct_tool / total if total else 0,
            "avg_argument_accuracy": avg_arg_acc,
            "avg_evidence_grounding": avg_evidence,
            "avg_tool_calls": avg_calls,
            "escalation_accuracy": correct_escalation / total if total else 0,
        }


def log_audit_event(inv_id: str, step_order: int, event_type: str, payload: Any):
    """Log an audit event (llm_decision, tool_call, tool_result, correlation, risk_assessment, escalation)."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO audit_log (investigation_id, step_order, event_type, payload)
               VALUES (?, ?, ?, ?)""",
            (inv_id, step_order, event_type, json.dumps(payload) if payload else None),
        )


def log_review_decision(inv_id: str, decision: str, note: str, reviewer: str):
    """Log a human review decision."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO review_records (investigation_id, decision, note, reviewer)
               VALUES (?, ?, ?, ?)""",
            (inv_id, decision, note, reviewer),
        )


def get_audit_trail(inv_id: str) -> list[dict]:
    """Get full audit trail for an investigation."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE investigation_id = ? ORDER BY step_order, id",
            (inv_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("payload"):
                try:
                    d["payload"] = json.loads(d["payload"])
                except Exception:
                    pass
            results.append(d)
        return results


def get_review_records(inv_id: str) -> list[dict]:
    """Get review records for an investigation."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM review_records WHERE investigation_id = ? ORDER BY timestamp",
            (inv_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_model_version(model_name: str, version: str, metrics: dict, is_active: bool = False):
    """Save model version info."""
    with get_connection() as conn:
        if is_active:
            conn.execute("UPDATE model_versions SET is_active = 0 WHERE model_name = ?", (model_name,))
        conn.execute(
            """INSERT INTO model_versions (model_name, version, metrics_json, is_active)
               VALUES (?, ?, ?, ?)""",
            (model_name, version, json.dumps(metrics), is_active),
        )


def get_model_versions(model_name: str) -> list[dict]:
    """Get model versions for a model."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM model_versions WHERE model_name = ? ORDER BY trained_at DESC",
            (model_name,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("metrics_json"):
                try:
                    d["metrics_json"] = json.loads(d["metrics_json"])
                except Exception:
                    pass
            results.append(d)
        return results
