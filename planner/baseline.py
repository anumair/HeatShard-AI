"""Naive reactive baseline: whole-shard migration.

Whenever any record on a shard is predicted (or observed) hot, a coarse
shard-level reactive system migrates every record on that shard, not
just the hot ones. Used purely as the Stage 6 checkpoint's data-movement
comparison against HeatShard's targeted, record-level plan.
"""

from common.shard_client import ShardCluster


def whole_shard_migration_volume(record_load: dict, hot_record_ids: set, cluster: ShardCluster = None) -> dict:
    cluster = cluster or ShardCluster()
    affected_shards = {cluster.shard_for_key(rid) for rid in hot_record_ids}
    moved_records = [rid for rid in record_load if cluster.shard_for_key(rid) in affected_shards]
    return {
        "affected_shards": sorted(affected_shards),
        "records_moved": len(moved_records),
        "record_ids": moved_records,
    }
