# HeatShard AI

Predictive hotspot management for distributed e-commerce databases. See
`METHODOLOGY.md` for the full problem statement, architecture, and evaluation
design.

## Status: Stage 3 — Adaptive Heat Index

Component 2 is implemented: raw per-record counters are turned into a
single per-window "heat score" -- decayed, z-score normalized, weighted,
and periodically re-weighted against observed outcomes.

```
collector/    # Stage 1: metrics middleware, windowing, event feed, diagnostics
predictor/    # Stage 3: adaptive heat index (Stage 4 prediction engine still to come)
planner/      # Stage 5-6: dependency graph + relocation planner
dashboard/    # Stage 8: visual demo
simulator/    # load generator (uniform/zipf) + flash_sale_scenario.py (Stage 2)
common/       # shared config and shard client used by every component
data/         # events.json (checked in, illustrative) + generated *.db/scenarios/*.png (gitignored)
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

### Adaptive Heat Index (Stage 3)

```bash
python predictor/compute_heat.py --refit-every 10 --min-refit-samples 20
python predictor/inspect_heat.py --record product:9
python predictor/plot_heat.py --record product:9 --record product:20
```

Run this right after a scenario (while `data/events.json` still reflects
that same run's registered events). `compute_heat.py` replays the
recorded windows chronologically through `predictor/heat_index.py`:

- 7 raw signals per record per window (QPS, growth rate, avg latency,
  read/write ratio, cache-miss rate, popularity share, event signal)
- exponentially decayed between windows so heat cools gradually instead
  of resetting to zero the instant a record goes quiet
- z-score normalized across that window's active records, then combined
  via the weighted `Heat_i(t)` formula from the methodology
- weights start fixed (`predictor/features.py`) and are periodically
  refit via linear regression against observed outcomes -- did the
  record actually spike in the window that followed? `--refit-every`
  defaults to 10 here for a fast demo; the methodology's target cadence
  for real use is 20-50 windows, with more history the refit is far less
  noisy than what you'll see in a 15-20 window demo run.

`inspect_heat.py` prints an ASCII bar chart per window; `plot_heat.py`
renders an actual PNG (`data/heat_plot.png`) -- the Stage 3 checkpoint
test: heat should visibly rise leading into the flash sale and decay
afterward for the spike records identified in the scenario's manifest.

### Stop it

```bash
docker compose down
```
