# HeatShard AI

Predictive hotspot management for distributed e-commerce databases. See
`METHODOLOGY.md` for the full problem statement, architecture, and evaluation
design.

## Status: All 9 stages complete

Stage 8 shipped a live, interactive web dashboard (FastAPI + a
hand-built HTML/CSS/JS frontend, no framework) that reuses every
stage's own modules directly to show the full pipeline running in real
time: shard load, per-record heat/predictions, the relocation plan,
the Stage 7 baseline comparison, and the system's own self-tuning --
with controls to launch a flash sale and watch it happen live. Stage 9
is the project report (methodology, per-stage implementation detail,
all evaluation metrics, limitations, and a viva walkthrough) -- ask the
maintainer for a copy.

```
collector/    # Stage 1: metrics middleware, windowing, event feed, diagnostics
predictor/    # Stage 3: adaptive heat index. Stage 4: trend + XGBoost prediction engine
planner/      # Stage 5: dependency graph. Stage 6: relocation planner. Stage 7: baselines + evaluate.py
dashboard/    # Stage 8: server.py (FastAPI) + static/ (frontend) -- the live visual demo
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

#### Honest accuracy numbers (scenario-level held-out evaluation)

"Training accuracy" is a misleading metric here -- positive (spike)
windows are only ~1% of samples, so a model that never predicts "hot"
already scores ~99%. `train_xgboost.py --test-db/--test-events` (held
out from training entirely, never seen) reports real precision/recall
instead. It's a **scenario-level** split, not a random row split, since
adjacent windows within one scenario are correlated (decayed features
carry over) -- a random split would leak information between train and
test.

`--test-db`/`--test-events` also exposed that plain accuracy AND the
"textbook" `scale_pos_weight` (the full class-imbalance ratio, ~100x)
were both misleading: at that weight, held-out precision collapsed to
3.5% (858 false positives for 31 true positives). Sweeping the weight
against held-out data found ~20 as the actual best precision/recall
trade-off (`predictor/train_xgboost.py`'s `DEFAULT_MAX_SCALE_POS_WEIGHT`);
override with `--scale-pos-weight` if you want a different point on that
curve.

We also tested whether *more* training data was the bottleneck:
starting from 22 scenarios of one fixed shape (104 positive samples,
best held-out F1 ~0.125), `simulator/generate_training_runs.py
--vary-params` generated 24 more scenarios with randomized spike
magnitude/bias/num-records/skew (not just a new seed) for 46 total (275
positive samples). Held-out F1 improved to ~0.147 -- a real but modest
gain. Conclusion: more diverse data helps somewhat, but isn't the main
lever left to pull; the harder constraints are the label design (only
1-2 windows per scenario ever qualify as a "spike" under the
transition-onset rule) and the small feature set. Worth knowing before
sinking more time into generating additional scenarios expecting a
large jump. In practice the deployed ensemble (trend + XGBoost +
adaptive threshold, not the XGBoost classifier's raw 0.5-cutoff
predictions) still flagged real spike records correctly in the Stage 4
and Stage 6 checkpoint runs -- isolated classifier metrics understate
how the full pipeline behaves.

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

### Baselines & Benchmarking (Stage 7)

```bash
python planner/evaluate.py --min-probability 0.5
```

Run after `predictor/run_prediction.py`. Implements the two comparison
systems from the methodology (Section 4.2) and evaluates all three
against the same recorded scenario:

- **static** -- no rebalancing, ever
- **reactive** (`planner/reactive_baseline.py`) -- migrates an entire
  shard once its *observed* rolling load crosses a fixed multiple of
  the cluster average (default 1.5x). No predictions, no per-record
  targeting -- purely reactive, and can only ever act once a shard is
  already overloaded.
- **heatshard** -- Stage 6's plan, evaluated at its own natural decision
  point (the earliest window with an actionable prediction)

Each system is judged at *its own* natural decision point rather than
one shared snapshot -- reactive scans the scenario chronologically for
the first window where load actually crosses the threshold, which is
later than HeatShard's proactive trigger by design. Ground truth ("did
a record actually become hot") reuses `HeatIndex`'s own spike rule,
scanned across the whole scenario. `evaluate.py` prints a results table
and renders `data/evaluation.png` with the methodology's five defined
metrics (Section 4.3): data movement, precision, recall, false-positive
rate, and load variance before/after.

A representative run:

```
system         moved  precision   recall   FP rate  var before  var after
static             0       0.00     0.00      0.00        1370       1370
reactive          18       0.06     0.11      0.94        1457       3089
heatshard          1       1.00     0.11      0.00        1370        580

HeatShard acted 12s earlier than the reactive baseline would have noticed anything
```

Two findings stood out and are worth keeping for the report's Results
section:

1. **The reactive baseline's blind "migrate everything to the
   least-loaded shard" strategy makes load variance *worse*, not
   better** (1457 -> 3089) -- dumping an entire overloaded shard onto a
   single destination just creates a new hotspot there. HeatShard's
   variance-aware placement (Benefit = actual variance reduction) cuts
   variance by more than half using 18x less data movement.
2. **Precision/recall vary meaningfully run to run**, and one honest
   edge case is worth documenting rather than hiding: a genuine
   ground-truth spike record can have too large an individual load to
   fit any single destination shard without overshooting and making
   variance worse, in which case the planner (correctly, by its own
   ExpectedValue logic) declines to move it -- it isn't a bug, but it
   does mean the current single-destination-per-record design
   sometimes passes over the "obvious" candidate in favor of a smaller,
   more placement-friendly one. Worth flagging in the report's
   limitations section (Stage 9) alongside decision-support scope and
   cross-shard join tradeoffs.

### Dashboard (Stage 8)

```bash
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8420
# open http://127.0.0.1:8420
```

A single-page dashboard (`dashboard/server.py` + `dashboard/static/`) that reuses
every stage's modules directly -- `ShardCluster`, `DependencyGraph`,
`compute_plan`, `reactive_plan`, `evaluate_system`, etc. -- rather than
reimplementing any of their logic. Panels:

- **Shard load** and a **live scenario log** (streams a running
  `flash_sale_scenario.py` subprocess in real time)
- **Records** -- a heat/prediction scatter plus table, flagged hotspots highlighted
- **Relocation plan** -- the current plan's moves with cost/benefit/EV
- **Dependency graph** -- the Stage 5 co-access triangles, rendered as an
  interactive SVG (hover a node for its id)
- **Evaluation** -- the Stage 7 static/reactive/HeatShard comparison,
  including the lead-time headline stat, recomputed live
- **System self-tuning** -- the heat index's weight-refit history and the
  adaptive confidence threshold over time, with a clear empty-state
  message when a scenario hasn't produced enough windows to trigger either
  yet (rather than a blank chart)

**Controls:** "Launch Flash Sale" starts a real scenario as a background
process (the DB resets to a clean slate first) and the dashboard polls
its live output and shard load while it runs -- the phase timeline at
the top tracks baseline/pre_spike/spike/cooldown with a live cursor,
using a manifest written with *planned* phase boundaries the moment the
scenario starts (the accurate, as-observed manifest overwrites it once
the run finishes). "Run Prediction & Planning" runs
`compute_heat.py` + `run_prediction.py` + `run_relocation.py` against
whatever was just recorded.

Chart.js is vendored locally (`dashboard/static/vendor/`) rather than
loaded from a CDN, so the dashboard works offline during a demo.

### Stop it

```bash
docker compose down
```
