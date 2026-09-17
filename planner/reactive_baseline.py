"""Stage 7 baseline: reactive threshold-based rebalancing.

What existing dynamic sharding systems typically do (METHODOLOGY.md
Section 1): monitor per-shard load (our proxy for CPU/latency) and
migrate an ENTIRE shard's records once its load crosses a fixed
multiple of the cluster average. No predictions, no per-record
targeting, no dependency graph -- purely reactive to load that has
already happened, and only ever able to act once a shard is already
overloaded, never before.
"""

from common.shard_client import ShardCluster

DEFAULT_THRESHOLD_MULTIPLIER = 1.5  # a shard is "overloaded" once its load exceeds this x the cluster average


def reactive_plan(
    shard_load: dict,
    record_load: dict,
    cluster: ShardCluster = None,
    threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
) -> dict:
    """Returns {"moves": [(record_id, source_shard, destination_shard), ...],
    "overloaded_shards": [...], "destination": shard_name}."""
    cluster = cluster or ShardCluster()
    avg_load = sum(shard_load.values()) / len(shard_load) if shard_load else 0.0
    threshold = avg_load * threshold_multiplier

    overloaded = [shard for shard, load in shard_load.items() if load > threshold]
    if not overloaded:
        return {"moves": [], "overloaded_shards": [], "destination": None}

    destination = min(shard_load, key=shard_load.get)  # naive target: whichever shard is least loaded right now

    moves = [
        (record_id, cluster.shard_for_key(record_id), destination)
        for record_id in record_load
        if cluster.shard_for_key(record_id) in overloaded and cluster.shard_for_key(record_id) != destination
    ]

    return {"moves": moves, "overloaded_shards": overloaded, "destination": destination}
