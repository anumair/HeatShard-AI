# HeatShard AI

Predictive hotspot management for distributed e-commerce databases. See
`METHODOLOGY.md` for the full problem statement, architecture, and evaluation
design.

## Status: Stage 2 — Load Simulator with Flash-Sale Injection

Component 1 (metrics collection) is implemented, and the load simulator now
produces realistic, repeatable test traffic: Zipfian-distributed baseline
demand plus a scriptable flash-sale scenario with scheduled event metadata
and a ground-truth manifest for later prediction-model training.

```
collector/    # Stage 1: metrics middleware, windowing, event feed, diagnostics
predictor/    # Stage 3-4: adaptive heat index + prediction engine
planner/      # Stage 5-6: dependency graph + relocation planner
dashboard/    # Stage 8: visual demo
simulator/    # load generator (uniform/zipf) + flash_sale_scenario.py (Stage 2)
common/       # shared config and shard client used by every component
data/         # events.json (checked in, illustrative) + generated *.db/scenarios (gitignored)
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

### Flash-sale scenario (Stage 2)

```bash
python simulator/flash_sale_scenario.py --seed 7 --spike-num-records 3
```

Runs four phases against the live cluster: `baseline` (Zipfian traffic) ->
`pre_spike` (event metadata for the flash sale is registered in the event
channel right away, but traffic hasn't moved yet) -> `spike` (traffic
heavily biased toward the chosen records) -> `cooldown`. All parameters
(number of spike records, magnitude, bias, phase durations, skew, seed) are
CLI flags, so the same scenario can be replayed exactly via `--seed`.

Each run writes a manifest to `data/scenarios/scenario_<ts>.json` (mirrored
to `data/last_scenario.json`) recording exactly which records spiked and
the wall-clock boundary of each phase -- this is the ground truth Stage 4
will use to label training windows as positive/negative examples. It also
prints each spike record's per-window access_count series so the rise
during `spike` and drop-off in `cooldown` are directly visible, e.g.:

```
product:9   1  2  1  21  27  22   2   1
```

### Stop it

```bash
docker compose down
```
