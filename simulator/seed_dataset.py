"""Seed Redis with real product/review/order payloads from data/olist_records.json
(built by build_olist_keyspace.py), so GETs against the olist key space return
actual dataset content instead of placeholder strings. Run once before a
scenario that uses --key-source olist.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.keyspace import DEFAULT_OLIST_PATH  # noqa: E402
from common.shard_client import ShardCluster  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Seed the cluster with real Olist product/review/order records")
    parser.add_argument("--path", default=str(DEFAULT_OLIST_PATH))
    args = parser.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    cluster = ShardCluster()
    written = 0
    for product_id, rec in data["records"].items():
        cluster.client_for_key(f"product:{product_id}").set(f"product:{product_id}", json.dumps(rec["product"]))
        cluster.client_for_key(f"review:{product_id}").set(f"review:{product_id}", json.dumps(rec["review"]))
        cluster.client_for_key(f"order:{product_id}").set(f"order:{product_id}", json.dumps(rec["order"]))
        written += 3

    print(f"seeded {written} keys ({len(data['records'])} products x 3) from {args.path}")


if __name__ == "__main__":
    main()
