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
brew install libomp           # macOS only: XGBoost needs the OpenMP runtime

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
  refit via ridge-regularised least squares (then smoothed and floored so
  no signal collapses to zero) against observed outcomes -- was the record
  hot in the window that followed (`predictor/labels.py`)? `--refit-every`
  defaults to 10 here for a fast demo; the methodology's target cadence
  for real use is 20-50 windows, with more history the refit is far less
  noisy than what you'll see in a 15-20 window demo run.

`inspect_heat.py` prints an ASCII bar chart per window; `plot_heat.py`
renders an actual PNG (`data/heat_plot.png`) -- the Stage 3 checkpoint
test: heat should visibly rise leading into the flash sale and decay
afterward for the spike records identified in the scenario's manifest.

### Hybrid Prediction Engine (Stage 4) -- v2

```bash
# 1. generate varied, deliberately imperfect training scenarios (~1 min each)
python simulator/generate_training_runs.py --num-runs 60 --seed-start 300 --vary-params --out-dir data/training_v2

# 2. a separate, never-trained-on set for held-out evaluation
python simulator/generate_training_runs.py --num-runs 16 --seed-start 9000 --vary-params --out-dir data/test_v2

# 3. train (writes data/xgb_model.json + data/xgb_model.meta.json)
python predictor/train_xgboost.py --db 'data/training_v2/run_*.db' --test-db 'data/test_v2/run_*.db'

# 4. held-out evaluation: ablation, early warning, per-record-kind breakdown
python predictor/evaluate_prediction.py --db 'data/test_v2/run_*.db'

# 5. run the engine over any recorded scenario
python predictor/run_prediction.py --manifest data/scenarios/scenario_<ts>.json
```

(Each run's event feed is expected next to its db as `run_<seed>_events.json`.)

**What it predicts.** `P(record is hot in the next window)`, where "hot" is
defined once in `predictor/labels.py` and shared by training labels, the
weight fitter, the adaptive threshold's feedback, relocation-outcome
checking and the Stage 7 ground truth: the record's load is at least
2.5 requests/s **and** at least 3x its own *lagged* baseline (the mean of
the 6 windows that ended 3 windows ago, so a surge can't absorb itself
into its own baseline). It is a *sustained-surge* label.

**Components**

- `predictor/trend_model.py` -- statistical sub-model: a damped-trend
  (Holt) forecast of the record's next-window count, compared against the
  exact "hot bar" the label uses, squashed to `p_trend`. Stateless.
- `predictor/xgb_model.py` + `train_xgboost.py` -- XGBoost on 21 features
  (`predictor/features.py`): the heat index's 7 z-scored signals plus
  per-second rate history (lags, ratio to the recent and lagged baselines,
  share of traffic), the trend forecast, the heat score, and **raw event
  timing** (announced flag, seconds-until-start, magnitude).
- `predictor/calibration.py` -- isotonic calibration of both sub-models on
  out-of-fold scores, so `p_ensemble` is a real probability for the
  Relocation Planner's ExpectedValue arithmetic.
- `predictor/prediction_engine.py` -- learned blend (weight chosen to
  maximise out-of-fold average precision; stored in the model meta),
  confidence = agreement of the two calibrated sub-models; degrades to
  trend-only without a trained model.
- `predictor/confidence.py` -- `AdaptiveThreshold`: starts at the
  F1-optimal threshold found at training time and holds the validated
  precision band (raises the bar when live precision drops below it,
  lowers it when comfortably above), instead of searching blind.
- `predictor/pipeline.py` -- one `FeaturePipeline` used identically by
  training, replay and evaluation, so a feature or label can never be
  computed two different ways.

**Event channel, leak-free.** Events carry `announced_at`; replay keeps an
event invisible until that instant (the earlier version loaded the final
event file, so replay "knew" about flash sales ~20 s before they were
announced). Records with an announced upcoming event are scored every
window even with zero traffic, so a cold record can be flagged before its
first request.

**A harder simulator.** `flash_sale_scenario.py` can add (a) *surprise*
spikes with no event metadata, (b) *decoy* events that are announced but
never surge, and (c) ramped spikes. Without these every hot record was
announced, so "announced => hot" looked like near-perfect prediction.

#### Held-out accuracy (16 scenarios never trained on; 12,502 record-windows, 169 hot)

| variant | avg. precision | F1 @ validated threshold |
|---|---|---|
| heat score alone | 0.19 | -- |
| current rate alone ("hot now => hot next") | 0.29 | -- |
| trend sub-model alone | 0.41 | 0.54 |
| XGBoost sub-model alone | 0.78 | 0.74 |
| **blend (the engine)** | **0.79** | **0.735** (P 0.68, R 0.80) |

Scenario-grouped 5-fold cross-validation on the 60 training scenarios
gives F1 0.80 (P 0.78, R 0.83). The threshold (0.43) is fixed at training
time and *not* tuned on the held-out set.

By record kind (held-out):

| record kind | hot rows | recall | note |
|---|---|---|---|
| announced flash sale | 130 | 0.91 | precision 0.72 |
| surprise spike (no event) | 28 | 0.64 | caught once traffic starts; 0% *before* it starts -- nothing can predict it |
| decoy event (announced, fizzled) | 0 | -- | **0 of 277 decoy rows flagged** |
| ordinary records | 11 | 0.00 | |

Early warning (hot next window, still quiet this window -- invisible to a
reactive system): the engine flags 65% of announced flash sales before
their first hot window (20 cases); unannounced onsets are not predictable.

#### What changed, and why the old 0.147 isn't comparable

The previous engine scored F1 ~0.147. Diagnosis on 46 recorded scenarios
(scenario-grouped CV) found the label, not the model, was the limit: the
old "next window > 3x the last-5-window mean" rule fired on only 1% of
samples, 80% of those on low-traffic Zipf-tail noise, and only 54 of 1,050
genuine flash-sale rows were positive (50 of them in the first two spike
windows). A model that knew *exactly* which records were flash-sale
records scored F1 0.08 against that label. So **part of the jump to
~0.74 is a redefinition of success** (sustained surge instead of
one-window rising edge), which matches what relocation actually needs.
The part that is genuine model improvement: on the *old* label, history
features alone lifted F1 from 0.27 to 0.41 in the same experiment.

Other findings from that diagnosis, all fixed here: event signal leaked
into replay before announcement; the z-scored event feature lost timing;
the trend model (slope of heat scores) was weaker than the raw current
rate; the adaptive threshold was fed the broken label; heat-index weight
refits could collapse onto one feature (now ridge-regularised, smoothed,
floored at 0.02).

Limits to be honest about: all data is simulated; the sub-model ablation
shows the trend model adds little on top of XGBoost (blend weight 0.9);
a 24-config hyper-parameter sweep moved held-out AP by only 0.78-0.80, so
the remaining ceiling is information (unannounced spikes), not tuning.

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
the shared `predictor/labels.py` hot rule) and feeds that outcome into a fresh
`AdaptiveThreshold`. In one real run, 9 of 10 relocated records turned
out to be false positives (from noisy early-window candidates) and only
the genuine flash-sale record was correctly hot -- that low precision
pushed the threshold up from 0.60 to 0.65, exactly the self-correcting
behavior the methodology calls for.

### Baselines & Benchmarking (Stage 7)

```bash
python planner/evaluate.py
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
one shared snapshot: reactive scans the scenario chronologically (once it
has a full lookback of history) for the first window where load actually
crosses the threshold; HeatShard acts at the earliest window with an
actionable prediction. Ground truth ("did a record actually become hot")
is the shared `predictor/labels.py` rule -- a sustained surge against the
record's own lagged baseline -- scanned across the whole scenario.
`evaluate.py` prints a results table for one scenario and renders
`data/evaluation.png` with the methodology's five defined metrics
(Section 4.3): data movement, precision, recall, false-positive rate, and
load variance before/after.

**One scenario is illustrative, not a result.** Precision, recall and
timing vary a lot by seed, so the headline numbers come from
`planner/evaluate_aggregate.py`, which runs N independent, freshly seeded
scenarios and reports mean +/- std:

```bash
python planner/evaluate_aggregate.py --num-runs 15 --seed-start 5000
# re-score already-generated scenarios (e.g. with a different --min-probability):
python planner/evaluate_aggregate.py --num-runs 15 --seed-start 5000 --reuse-existing
```

The 15 scenarios are drawn from the same varied, deliberately imperfect
distribution the model trains on (unannounced surprise spikes, decoy
events that never surge, ramped spikes, 3-5 s windows) but with seeds the
model never saw. Variance is reported two ways: at each system's *own
decision time*, and at the *load peak* -- the window where the cluster is
most imbalanced -- so a system that acts early is not penalised for the
load not having built up yet, and one that acts late is not credited for
it.

Result over 15 never-trained-on scenarios (mean +/- std):

| metric | static | reactive | **HeatShard** |
|---|---|---|---|
| records moved | 0 | 22.9 +/- 9.9 | **1.8 +/- 0.9** |
| precision | -- | 0.05 +/- 0.04 | **0.93 +/- 0.25** |
| recall | -- | 0.20 +/- 0.18 | **0.30 +/- 0.15** |
| false-positive rate | -- | 0.89 +/- 0.24 | **0.00** |
| load variance, at decision time (before -> after) | 3364 | 3786 -> 8578 (**worse**) | 3364 -> 2128 (**-37%**) |
| load variance, at the load peak (before -> after) | 6869 | 6869 -> 10938 (**+59%**) | 6869 -> 4362 (**-36%**) |

What holds up:

- **~13x less data movement** than reactive (1.8 vs 22.9 records), with near-perfect precision
  and zero false-positive relocations.
- **Reactive's blind "migrate the whole shard to the least-loaded shard"
  makes balance worse, not better**, on average -- it dumps an entire shard
  onto one destination and creates a new hotspot there.
- At its decision moment HeatShard lowers load variance in 14 of 15 runs
  (the 15th found no positive-expected-value move and did nothing).

What does not hold up, stated plainly:

- **The balance benefit at the load peak is real but uneven.** Variance
  at the peak drops in 10 of 15 runs (the other 5 are unchanged or
  slightly worse); the per-run median reduction is ~27% (mean 30%). The
  pooled -36% in the table is weighted toward the high-variance
  scenarios, so quote the per-run median alongside it.
- **Recall is low (0.30).** HeatShard moves ~2 records per scenario while
  ~6 become hot. Some are unannounced surprise spikes nothing can predict
  early; the planner also declines moves whose expected value is not
  positive, and relocates each record to a single destination only.
- **There is no real lead-time advantage in this protocol.** HeatShard
  acted before the first hot window in only 2 of 15 runs (mean -0.3 s
  +/- 2.5 s). It cannot commit on an announcement alone because decoy
  events (announced, never surge) are in the data: a calibrated model
  honestly reports ~0.1-0.2 probability at that moment. It commits the
  window traffic appears -- but then well before the surge is established
  (hot windows last ~4). Reactive "acts before the first hot window" in 7
  of 14 runs only because its untuned 1.5x threshold fires on ordinary
  Zipf skew (89% false-positive relocations), not because it anticipates
  anything.
- Precision is bimodal: 1.0 in 14 runs, 0.0 in the run with no moves.

**Planner fix found during the final regression check.** `compute_plan()`
used to size every candidate's load from the *last* windows of the whole
recording rather than the decision window. A record flagged before its spike
looked tiny (no benefit, no move), and in short scenarios post-decision spike
load leaked back into the plan. It now reads load as of the decision window
and sizes a predicted-hot record at no less than the hot-level load
(`HOT_MIN_QPS` over the lookback). The table above is after that fix.

Sensitivity of the planner's candidate-probability floor (picked on 16
*validation* scenarios, `planner/sweep_probability_floor.py`, not on the
15 above): lowering it makes HeatShard act earlier but worse -- at 0.15 it
acts before the first hot window in 12/16 runs but peak-variance
reduction falls from 42% to 36% and precision from 0.81 to 0.72, and at
0.05 it collapses (precision 0.12). 0.3 was best on every balance metric,
so it stays the default.

Superseded numbers: earlier aggregated results in this README (precision
0.27, recall 0.19, lead time 2.2 s, -56% variance) were produced with the
v1 prediction engine, the easier simulator and an evaluation directory
that could be reused across runs (the collector appends to an existing
db); they are not comparable and should not be quoted.

### Dashboard (Stage 8)

```bash
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8420
# open http://127.0.0.1:8420
```

A single-page dashboard (`dashboard/server.py` + `dashboard/static/`) that reuses
every stage's modules directly -- `ShardCluster`,
`compute_plan`, `reactive_plan`, `evaluate_system`, etc. -- rather than
reimplementing any of their logic. Panels:

- **Shard load** and a **live scenario log** (streams a running
  `flash_sale_scenario.py` subprocess in real time)
- **Records** -- a heat/prediction scatter plus table, flagged hotspots
  highlighted. A window slider picks which window to inspect; it opens on the
  *peak* window (highest P(hotspot)) because the latest window of a finished
  scenario is its quiet cooldown
- **P(hotspot) over time** -- the six hottest records' probability lines over
  the scenario's phase bands, the adaptive threshold (dashed), the moment the
  event was announced, and large dots where a record was flagged
- **Relocation plan** -- the current plan's moves with cost/benefit/EV
- **Evaluation (this scenario)** -- the Stage 7 static/reactive/HeatShard
  comparison recomputed live. Reported as counts ("1 of 1 moves truly hot"),
  not percentages: one scenario moves a handful of records, so a "100%
  precision" headline would be meaningless
- **Evaluation (many unseen scenarios)** -- reads `data/evaluation_aggregate.json`
  (written by `planner/evaluate_aggregate.py`): per-scenario dots with mean and
  +/- 1 std for records moved, precision, recall and peak-load variance, plus
  headline stats (e.g. precision perfect in 14 of 15 runs, one run made no move)
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
