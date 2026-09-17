# HeatShard AI

Predictive hotspot management for distributed e-commerce databases. See
`METHODOLOGY.md` for the full problem statement, architecture, and evaluation
design.

## Status: Stage 6 — Intelligent Relocation Planner

Component 5 is implemented -- the project's core novel contribution:
predictions + the dependency graph + current shard load turn into a
concrete, cost-justified relocation plan, with a naive whole-shard
baseline for comparison and outcome tracking that feeds back into
Stage 4's adaptive confidence threshold.

```
collector/    # Stage 1: metrics middleware, windowing, event feed, diagnostics
predictor/    # Stage 3: adaptive heat index. Stage 4: trend + XGBoost prediction engine
planner/      # Stage 5: dependency graph. Stage 6: relocation planner + naive baseline
dashboard/    # Stage 8: visual demo
simulator/    # load generator (uniform/zipf) + flash_sale_scenario.py (Stage 2) + generate_training_runs.py (Stage 4)
common/       # shared config and shard client used by every component
data/         # events.json (checked in, illustrative) + generated *.db/scenarios/*.png/olist_records.json/training_runs/xgb_model.json (gitignored)
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

### Hybrid Prediction Engine (Stage 4)

```bash
# 1. generate several scenarios as labeled training data (~1min each)
python simulator/generate_training_runs.py --num-runs 16 --seed-start 200

# 2. train the XGBoost sub-model on all of them (repeat --db/--events per run)
python predictor/train_xgboost.py \
  --db data/training_runs/run_200.db --events data/training_runs/run_200_events.json \
  --db data/training_runs/run_201.db --events data/training_runs/run_201_events.json \
  ... # one pair per generated run

# 3. run a fresh (held-out) scenario, then the prediction engine over it
python simulator/flash_sale_scenario.py --seed 999
python predictor/run_prediction.py --manifest data/scenarios/scenario_<ts from step 3>.json
```

`predictor/heat_index.py` (Stage 3) is reused as-is for feature
extraction; Stage 4 adds:

- `predictor/trend_model.py` -- fits a slope over each record's last 5
  heat scores, z-scores it across the window's population, sigmoid ->
  `p_trend`
- `predictor/xgb_model.py` + `train_xgboost.py` -- an XGBoost classifier
  trained on the heat index's own 7 normalized features, using the exact
  same "did this record spike in the window that followed" labeling the
  weight refitting already relies on (via `HeatIndex`'s `sample_callback`
  hook), so training labels stay consistent with the rest of the system.
  One scenario alone rarely has enough positive (spike) windows to train
  on well -- `generate_training_runs.py` produces several independent
  scenarios into separate db/events pairs for this.
- `predictor/prediction_engine.py` -- averages `p_trend`/`p_xgb` into
  `p_ensemble`, with confidence = model agreement (`1 - |p_trend - p_xgb|`);
  degrades gracefully to trend-only if no XGBoost model is loaded yet
- `predictor/confidence.py` -- `AdaptiveThreshold`: tracks the precision
  of the last 10 flagged predictions and raises the acting threshold when
  flags have been low-value, lowers it when they've been reliably correct

`run_prediction.py` is the Stage 4 checkpoint test: it prints each spike
record's `p_ensemble`/confidence trajectory against the scenario's phase
boundaries. On a held-out scenario (seed unseen during training), spike
records get flagged with high confidence right at the spike's first
window -- computed from the *prior* (still-quiet) window's features, so
it's a genuinely proactive flag, not a reactive one. With only a handful
of pre-spike windows in a short demo scenario, don't expect a long,
gradually-rising lead-up -- the real signal is the flag landing on the
transition window itself rather than several windows into the spike.

### Dependency Graph (Stage 5)

```bash
python planner/inspect_graph.py                              # graph summary + most-connected records
python planner/inspect_graph.py --record "product:<some id>"  # given a record, return its neighbors
python planner/plot_graph.py                                  # PNG sanity check (data/dependency_graph.png)
```

`planner/dependency_graph.py` builds the graph directly from Component
1's `co_access_windows` table: aggregates every recorded pair's count
across all windows, drops pairs below `--min-co-access` (default 3) to
keep it sparse, and exposes `neighbors(record_id)`, `has_edge(a, b)`,
and `edge_weight(a, b)` for Stage 6's Relocation Planner to query when
computing its cross-shard penalty.

The sanity check: since the simulator only ever calls
`collector.transaction(...)` on a product's own product/review/order
triple, the resulting graph should be a set of fully disjoint triangles
-- one per product, no cross-links between unrelated products. Verified:
a 30-record run produced exactly 10 isolated triangles (30 edges, 10
connected components), and raising `--min-co-access` from 3 to 15 dropped
weakly-supported edges as expected (10 components -> 4).

### Relocation Planner (Stage 6)

```bash
python planner/run_relocation.py                              # plan from the earliest actionable window
python planner/run_relocation.py --window-start <ts> --min-probability 0.5   # target a specific window
python planner/check_outcomes.py                               # resolve outcomes, feed Stage 4's adaptive threshold
```

Run this after `predictor/run_prediction.py` has populated the
`predictions` table for the scenario. `planner/relocation_planner.py`
implements the methodology's formula for each candidate record *i* and
destination shard *j*:

```
ExpectedValue(i, j) = P(hotspot_i) * Benefit(i, j)
                     - RelocationCost(i, j)
                     - (1 - P(hotspot_i)) * WastedCost(i, j)
                     - CrossShardPenalty(i, j)
```

`Benefit(i, j)` is the reduction in cluster load variance a hypothetical
move would produce, so "destination chosen to minimize resulting load
variance" falls directly out of the formula. `CrossShardPenalty` queries
Stage 5's dependency graph -- moving a record away from shards holding
its co-accessed neighbors costs more. Only positive-EV moves make the
plan; exact optimal assignment is NP-hard, so a greedy heuristic picks
the single best (candidate, destination) pair by EV, applies it, and
recomputes every remaining candidate against the updated shard loads
and placement before picking the next move.

`run_relocation.py` also runs the **headline Stage 6 checkpoint**: a
naive whole-shard-migration baseline (move everything on any shard that
hosts a predicted hotspot) for comparison. On a real flash-sale run:

```
HeatShard (targeted):  5 record(s) relocated
Naive (whole-shard):   116 record(s) relocated (4 shards fully migrated)
reduction: 95.7% less data movement than the naive baseline
```

The plan correctly included the scenario's actual flash-sale record
(planned in the last pre-spike window, before the spike hit) and kept a
product's `review`/`order` counterparts co-located rather than splitting
them across shards -- the dependency graph's cross-shard penalty working
as intended.

`check_outcomes.py` closes the loop: for every planned move, it checks
whether the record actually went hot in the following window (reusing
`HeatIndex`'s own spike rule) and feeds that outcome into a fresh
`AdaptiveThreshold`. In one real run, 9 of 10 relocated records turned
out to be false positives (from noisy early-window candidates) and only
the genuine flash-sale record was correctly hot -- that low precision
pushed the threshold up from 0.60 to 0.65, exactly the self-correcting
behavior the methodology calls for.

### Stop it

```bash
docker compose down
```
