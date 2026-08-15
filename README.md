# HeatShard AI

Predictive hotspot management for distributed e-commerce databases. See
`METHODOLOGY.md` for the full problem statement, architecture, and evaluation
design.

## Status: Stage 1 — Metrics Collection Layer

Component 1 is implemented: every simulated request now passes through a
collector middleware that tracks per-record counters, a co-access graph,
and samples 1-in-100 queries for diagnostics, snapshotted into SQLite every
window.

```
collector/    # Stage 1: metrics middleware, windowing, event feed, diagnostics
predictor/    # Stage 3-4: adaptive heat index + prediction engine
planner/      # Stage 5-6: dependency graph + relocation planner
dashboard/    # Stage 8: visual demo
simulator/    # load generator (Stage 0 trivial -> Stage 2 Zipfian + flash sales)
common/       # shared config and shard client used by every component
data/         # events.json (checked in, illustrative) + generated *.db files (gitignored)
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
key, `common/shard_client.py`), plus how many distinct records and co-access
pairs it collected in the final partial window.

### Metrics collection (Stage 1)

```bash
python collector/inspect_metrics.py            # latest windowed rows + top co-access pairs
python collector/inspect_metrics.py --record product:7
python collector/checkpoint_test.py            # baseline -> spike -> cooldown demo
python collector/seed_events.py --lead-seconds 3600   # refresh data/events.json
```

`checkpoint_test.py` is the Stage 1 checkpoint test from the implementation
plan: it hammers one record for 9s and prints its per-window access_count,
which should visibly jump during the spike window and decay right after.

### Stop it

```bash
docker compose down
```
