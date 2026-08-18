"""SQLite persistence for computed heat scores and weight-refit history.
Shares the same database file as Component 1's metric_windows table."""

import json
import sqlite3
import time
from pathlib import Path

DEFAULT_HEAT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS heat_windows (
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    record_id TEXT NOT NULL,
    heat_score REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_heat_windows_record ON heat_windows (record_id, window_start);

CREATE TABLE IF NOT EXISTS weight_history (
    window_index INTEGER NOT NULL,
    ts REAL NOT NULL,
    weights_json TEXT NOT NULL
);
"""


def init_db(path=None) -> Path:
    path = Path(path) if path else DEFAULT_HEAT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    return path


def clear_heat_tables(path):
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM heat_windows")
    conn.execute("DELETE FROM weight_history")
    conn.commit()
    conn.close()


def write_heat_window(path, window_start: float, window_end: float, heat_scores: dict):
    if not heat_scores:
        return
    rows = [(window_start, window_end, record_id, score) for record_id, score in heat_scores.items()]
    conn = sqlite3.connect(path, timeout=5)
    conn.executemany(
        "INSERT INTO heat_windows (window_start, window_end, record_id, heat_score) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def write_weight_history(path, window_index: int, weights: dict):
    conn = sqlite3.connect(path, timeout=5)
    conn.execute(
        "INSERT INTO weight_history (window_index, ts, weights_json) VALUES (?, ?, ?)",
        (window_index, time.time(), json.dumps(weights)),
    )
    conn.commit()
    conn.close()
