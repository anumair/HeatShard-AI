"""Cluster topology shared by every component (collector, simulator, planner, ...)."""

SHARDS = [
    {"name": "shard-0", "host": "localhost", "port": 7001},
    {"name": "shard-1", "host": "localhost", "port": 7002},
    {"name": "shard-2", "host": "localhost", "port": 7003},
    {"name": "shard-3", "host": "localhost", "port": 7004},
    {"name": "shard-4", "host": "localhost", "port": 7005},
]

WINDOW_SECONDS = 5
