"""Rolls up Component 1's per-record counters into the current per-shard
load estimate the Relocation Planner balances against.
"""

import sqlite3

from collector.storage import DEFAULT_DB_PATH
from common.shard_client import ShardCluster


def record_recent_load(db_path=None, window_lookback: int = 5) -> dict:
    """{record_id: total_access_count} summed over the most recent
    `window_lookback` windows recorded in metric_windows."""
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    windows = [
        r["window_start"]
        for r in conn.execute(
            "SELECT DISTINCT window_start FROM metric_windows ORDER BY window_start DESC LIMIT ?",
            (window_lookback,),
        ).fetchall()
    ]
    if not windows:
        conn.close()
        return {}

    placeholders = ",".join("?" * len(windows))
    rows = conn.execute(
        f"SELECT record_id, SUM(access_count) as total FROM metric_windows "
        f"WHERE window_start IN ({placeholders}) GROUP BY record_id",
        windows,
    ).fetchall()
    conn.close()

    return {r["record_id"]: float(r["total"]) for r in rows}


def shard_loads(record_load: dict, cluster: ShardCluster = None) -> dict:
    """Roll up per-record load to each record's current (hash-based) home shard."""
    cluster = cluster or ShardCluster()
    loads = {name: 0.0 for name in cluster.shard_names()}
    for record_id, load in record_load.items():
        loads[cluster.shard_for_key(record_id)] += load
    return loads
