# HeatShard AI

Predictive hotspot management for distributed e-commerce databases. See
`METHODOLOGY.md` for the full problem statement, architecture, and evaluation
design.

## Status: Stage 0 — Setup & Scaffolding

A running 5-node Redis "cluster" with a trivial cross-shard load generator.
No intelligence yet — this stage just proves the skeleton works end to end.

```
collector/    # Stage 1: per-record metrics collection
predictor/    # Stage 3-4: adaptive heat index + prediction engine
planner/      # Stage 5-6: dependency graph + relocation planner
dashboard/    # Stage 8: visual demo
simulator/    # load generator (Stage 0 trivial -> Stage 2 Zipfian + flash sales)
common/       # shared config and shard client used by every component
```

### Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d          # start the 5 Redis shards
python check_cluster.py       # confirm read/write + latency on every shard
python simulator/load_generator.py --duration 10 --rate 40 --num-keys 50
```

`check_cluster.py` should show `OK` for all 5 shards. The load generator
report shows requests landing across all 5 shards (basic hash placement on
key, `common/shard_client.py`).

### Stop it

```bash
docker compose down
```
