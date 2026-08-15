"""Thin wrapper around the Redis shards defined in config.py.

Stage 0 uses static hash-based placement (crc32(key) % num_shards). Later
stages layer a routing/override table on top of this so the Relocation
Planner can move individual records without changing the hash function.
"""

import zlib

import redis

from common.config import SHARDS


class ShardCluster:
    def __init__(self, shards=None, socket_timeout=2.0):
        self.shards = shards or SHARDS
        self._clients = {
            s["name"]: redis.Redis(
                host=s["host"], port=s["port"], socket_timeout=socket_timeout, decode_responses=True
            )
            for s in self.shards
        }

    def shard_names(self):
        return list(self._clients.keys())

    def client(self, shard_name):
        return self._clients[shard_name]

    def shard_for_key(self, key: str) -> str:
        names = self.shard_names()
        index = zlib.crc32(key.encode("utf-8")) % len(names)
        return names[index]

    def client_for_key(self, key: str):
        return self.client(self.shard_for_key(key))

    def all_clients(self):
        return self._clients.items()
