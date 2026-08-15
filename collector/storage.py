"""SQLite persistence for windowed metrics, co-access counts, and the
1-in-100 diagnostic query log. A new connection is opened per call --
writes are infrequent (once per window, or ~1% of requests) so this stays
simple and avoids cross-thread sqlite connection sharing.
"""

import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS metric_windows (
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    record_id TEXT NOT NULL,
    access_count INTEGER NOT NULL,
    write_count INTEGER NOT NULL,
    cache_miss_count INTEGER NOT NULL,
    avg_latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metric_windows_record
    ON metric_windows (record_id, window_start);

CREATE TABLE IF NOT EXISTS co_access_windows (
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    record_a TEXT NOT NULL,
    record_b TEXT NOT NULL,
    count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    ts REAL NOT NULL,
    record_id TEXT NOT NULL,
    op TEXT NOT NULL,
    shard TEXT NOT NULL,
    latency_ms REAL NOT NULL
);
"""


def init_db(path=None) -> Path:
    path = Path(path) if path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    return path


def write_window(path, window_start: float, window_end: float, counters: dict):
    if not counters:
        return
    rows = [
        (
            window_start,
            window_end,
            record_id,
            c.access_count,
            c.write_count,
            c.cache_miss_count,
            c.avg_latency_ms,
        )
        for record_id, c in counters.items()
    ]
    conn = sqlite3.connect(path, timeout=5)
    conn.executemany(
        "INSERT INTO metric_windows "
        "(window_start, window_end, record_id, access_count, write_count, cache_miss_count, avg_latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def write_co_access(path, window_start: float, window_end: float, co_access: dict):
    if not co_access:
        return
    rows = [(window_start, window_end, a, b, count) for (a, b), count in co_access.items()]
    conn = sqlite3.connect(path, timeout=5)
    conn.executemany(
        "INSERT INTO co_access_windows (window_start, window_end, record_a, record_b, count) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def write_query_log(path, record_id: str, op: str, shard: str, latency_ms: float):
    conn = sqlite3.connect(path, timeout=5)
    conn.execute(
        "INSERT INTO query_log (ts, record_id, op, shard, latency_ms) VALUES (?, ?, ?, ?, ?)",
        (time.time(), record_id, op, shard, latency_ms),
    )
    conn.commit()
    conn.close()
