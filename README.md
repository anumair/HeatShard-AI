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
data/         # events.json (checked in, illustrative) + generated *.db/scenarios/*.png/olist_records.json (gitignored)
```

### Record/key space: Olist Brazilian E-commerce (default)

The record/key space is real product ids from the [Olist Brazilian
E-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
-- this is the default everywhere (`--key-source olist`), not an add-on.
Traffic *shape* stays fully synthetic (Zipfian baseline + flash-sale
injection, unchanged); only the record identities and their payloads are
real. Requires a Kaggle account + API token (`~/.kaggle/access_token`,
see Kaggle's own "API Token" setup page) and `pip install -r
requirements.txt` (adds `pandas`, `kagglehub`) -- both are part of setup
below, not optional.

Olist has no inventory/stock table, so the third leg of the
product/reviews/inventory co-access grouping is repurposed as `order:<id>`
(order volume + freight) -- the closest real analogue available.
`data/olist_records.json` is gitignored and not redistributed (Olist's
data is CC BY-NC-SA licensed); regenerate it locally with your own Kaggle
credentials via the setup step below.

The original fully-synthetic key space (`product:i` / `reviews:i` /
`inventory:i`, no dataset/credentials needed) is still available as an
explicit fallback: pass `--key-source synthetic` to `load_generator.py`
or `flash_sale_scenario.py`.

### Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d          # start the 5 Redis shards
python check_cluster.py       # confirm read/write + latency on every shard

# one-time dataset setup (needs a Kaggle API token, see above)
python simulator/build_olist_keyspace.py --num-products 200   # ETL, writes data/olist_records.json
python simulator/seed_dataset.py                              # writes real product/review/order payloads into Redis

python simulator/load_generator.py --duration 10 --rate 40 --num-keys 50
```

`check_cluster.py` should show `OK` for all 5 shards. The load generator
report shows requests landing across all 5 shards (basic hash placement on
key, `common/shard_client.py`), plus how many distinct records and co-access
pairs it collected in the final partial window. Skip the two dataset-setup
lines (and add `--key-source synthetic`) if you don't have a Kaggle token
handy yet.

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
during `spike` and drop-off in `cooldown` are directly visible, e.g.
(record ids are real Olist product ids by default):

```
product:36f60d45225e60c7da4558b070ce4b60   1  1  1  23  26  22  1
```

### Adaptive Heat Index (Stage 3)

```bash
python predictor/compute_heat.py --refit-every 10 --min-refit-samples 20
python predictor/inspect_heat.py --record "product:<a spike record id from the scenario output>"
python predictor/plot_heat.py --record "product:<id 1>" --record "product:<id 2>"
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
