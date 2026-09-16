"""Abstraction over the record/key universe the simulator draws from.

Two sources, same interface, so the Zipf sampler / load generator /
flash-sale scenario don't care which is in use:

  synthetic  - product:i / reviews:i / inventory:i (the original
               Stage 0-3 scheme, still available via --key-source
               synthetic; writes use a plain placeholder payload since
               there's no "real" content to preserve)
  olist      - real product_ids from the Olist Brazilian E-commerce
               dataset (see build_olist_keyspace.py), the default. Its
               third co-access leg is "order" (order volume/freight)
               rather than "inventory", since Olist has no stock table.
               Writes re-serialize the record's real data instead of a
               random placeholder, so a GET always returns real content
               even after a write has touched the key.
"""

import json
from pathlib import Path

DEFAULT_OLIST_PATH = Path(__file__).resolve().parent.parent / "data" / "olist_records.json"


class KeySpace:
    def __init__(self, num_keys, record_id_fn, related_keys_fn, source="synthetic", payload_fn=None):
        self.num_keys = num_keys
        self.source = source
        self._record_id_fn = record_id_fn
        self._related_keys_fn = related_keys_fn
        self._payload_fn = payload_fn

    def record_id(self, rank: int) -> str:
        return self._record_id_fn(rank)

    def related_keys(self, rank: int):
        return self._related_keys_fn(rank)

    def payload_for(self, key: str, fallback: str) -> str:
        """Value to write on a SET to `key`. Falls back to `fallback`
        (a synthetic placeholder) when this key space has no real
        content for the key -- always true for the synthetic source."""
        if self._payload_fn is None:
            return fallback
        return self._payload_fn(key, fallback)

    @classmethod
    def synthetic(cls, num_keys: int) -> "KeySpace":
        return cls(
            num_keys=num_keys,
            record_id_fn=lambda i: f"product:{i}",
            related_keys_fn=lambda i: [f"product:{i}", f"reviews:{i}", f"inventory:{i}"],
            source="synthetic",
        )

    @classmethod
    def from_olist(cls, path=None, num_keys=None) -> "KeySpace":
        path = Path(path) if path else DEFAULT_OLIST_PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run simulator/build_olist_keyspace.py first")
        with open(path) as f:
            data = json.load(f)

        product_ids = data["product_ids"]
        if num_keys:
            product_ids = product_ids[:num_keys]

        payloads = {}
        for pid in product_ids:
            rec = data["records"][pid]
            payloads[f"product:{pid}"] = json.dumps(rec["product"])
            payloads[f"review:{pid}"] = json.dumps(rec["review"])
            payloads[f"order:{pid}"] = json.dumps(rec["order"])

        def record_id(i):
            return f"product:{product_ids[i % len(product_ids)]}"

        def related_keys(i):
            pid = product_ids[i % len(product_ids)]
            return [f"product:{pid}", f"review:{pid}", f"order:{pid}"]

        def payload_for(key, fallback):
            return payloads.get(key, fallback)

        return cls(
            num_keys=len(product_ids),
            record_id_fn=record_id,
            related_keys_fn=related_keys,
            source="olist",
            payload_fn=payload_for,
        )
