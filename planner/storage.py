"""SQLite persistence for relocation plans and their later-observed
outcomes. Shares the same database file as the rest of the pipeline."""

import sqlite3
import time
from pathlib import Path

DEFAULT_PLANNER_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS relocation_plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    window_start REAL NOT NULL,
    record_id TEXT NOT NULL,
    source_shard TEXT NOT NULL,
    destination_shard TEXT NOT NULL,
    predicted_cost REAL NOT NULL,
    predicted_benefit REAL NOT NULL,
    expected_value REAL NOT NULL,
    confidence REAL NOT NULL,
    outcome_checked INTEGER NOT NULL DEFAULT 0,
    was_hot INTEGER
);
"""


def init_db(path=None) -> Path:
    path = Path(path) if path else DEFAULT_PLANNER_DB_PATH
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    return path


def write_plan(path, window_start: float, moves: list):
    if not moves:
        return
    conn = sqlite3.connect(path, timeout=5)
    conn.executemany(
        "INSERT INTO relocation_plans "
        "(created_at, window_start, record_id, source_shard, destination_shard, "
        "predicted_cost, predicted_benefit, expected_value, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                time.time(),
                window_start,
                m.record_id,
                m.source_shard,
                m.destination_shard,
                m.predicted_cost,
                m.predicted_benefit,
                m.expected_value,
                m.confidence,
            )
            for m in moves
        ],
    )
    conn.commit()
    conn.close()


def unchecked_plans(path) -> list:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM relocation_plans WHERE outcome_checked = 0").fetchall()
    conn.close()
    return rows


def mark_outcome(path, plan_id: int, was_hot: bool):
    conn = sqlite3.connect(path, timeout=5)
    conn.execute(
        "UPDATE relocation_plans SET outcome_checked = 1, was_hot = ? WHERE plan_id = ?",
        (int(was_hot), plan_id),
    )
    conn.commit()
    conn.close()
