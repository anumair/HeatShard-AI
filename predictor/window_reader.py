"""Replays the metric_windows table written by Component 1 chronologically,
grouped by window -- decoupled from live traffic generation so the heat
index (and, later, the predictor) can be recomputed over any recorded
scenario without re-running the load generator.
"""

import sqlite3
from collections import OrderedDict, namedtuple

from collector.storage import DEFAULT_DB_PATH

WindowRow = namedtuple("WindowRow", ["access_count", "write_count", "cache_miss_count", "avg_latency_ms"])


def read_windows(db_path=None):
    """Yields (window_start, window_end, {record_id: WindowRow}) in chronological order."""
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM metric_windows ORDER BY window_start ASC").fetchall()
    conn.close()

    grouped = OrderedDict()
    for r in rows:
        key = (r["window_start"], r["window_end"])
        grouped.setdefault(key, {})[r["record_id"]] = WindowRow(
            access_count=r["access_count"],
            write_count=r["write_count"],
            cache_miss_count=r["cache_miss_count"],
            avg_latency_ms=r["avg_latency_ms"],
        )

    for (window_start, window_end), counters in grouped.items():
        yield window_start, window_end, counters
