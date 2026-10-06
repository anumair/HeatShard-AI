"""Stage 8: HeatShard AI live dashboard backend.

A thin FastAPI layer over the existing pipeline -- every endpoint reuses
Stage 1-7's own modules directly (ShardCluster,
compute_plan, reactive_plan, evaluate_system, ...) rather than
reimplementing any of their logic. The only new code here is data
shaping for the frontend and two controls for running the live demo:
kicking off a flash-sale scenario (a real-time background process) and
running the predict+plan pipeline against whatever the scenario just
recorded.
"""

import json
import random
import subprocess
import sys
import sqlite3
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import collector.storage as collector_storage  # noqa: E402
import planner.storage as planner_storage  # noqa: E402
import predictor.storage as heat_storage  # noqa: E402
from collector.storage import DEFAULT_DB_PATH  # noqa: E402
from common.shard_client import ShardCluster  # noqa: E402
from planner.evaluate import evaluate_system, ever_hot_records, find_reactive_trigger, first_hot_window_start  # noqa: E402
from planner.reactive_baseline import DEFAULT_THRESHOLD_MULTIPLIER, reactive_plan  # noqa: E402
from planner.run_relocation import MIN_CANDIDATE_PROBABILITY, compute_plan  # noqa: E402
from planner.shard_load import record_recent_load, shard_loads  # noqa: E402
from predictor.train_xgboost import DEFAULT_MODEL_PATH  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MANIFEST_PATH = PROJECT_ROOT / "data" / "last_scenario.json"

app = FastAPI(title="HeatShard AI Dashboard")


def ensure_full_schema():
    """Every stage's tables (metric_windows, heat_windows, predictions,
    relocation_plans, ...) live in one shared db file, but each stage's
    own init_db() only runs when its CLI script does. Worse, sqlite3
    silently creates an empty stub file the instant anything connects to
    a missing path -- so even a guard like "only run this if the file
    exists" is unsafe, since some other endpoint's plain connect() can
    have already created an empty, schema-less file first. Always call
    this unconditionally (CREATE TABLE IF NOT EXISTS makes it cheap and
    idempotent either way) so the full schema exists before any request
    is served, no matter what state the file was in."""
    collector_storage.init_db(DEFAULT_DB_PATH)
    heat_storage.init_db(DEFAULT_DB_PATH)
    planner_storage.init_db(DEFAULT_DB_PATH)


ensure_full_schema()


@app.middleware("http")
async def ensure_schema_before_every_request(request, call_next):
    """The one-time call above only covers the file's state at server
    startup. This project's whole workflow involves deleting
    data/metrics.db directly (a plain `rm`) to reset between demos --
    that bypasses both of the in-app reset paths (startup and
    ScenarioRun.start()) entirely, and the next request would recreate an
    empty, schema-less stub via sqlite3's auto-create-on-connect and 500.
    Reasserting the schema on every request (cheap: CREATE TABLE IF NOT
    EXISTS, sub-millisecond) is the only way to be robust against the db
    file being deleted by anything outside this process."""
    ensure_full_schema()
    return await call_next(request)


# ---------------------------------------------------------------------------
# Background scenario runner: flash_sale_scenario.py runs in real time
# (rate-limited traffic, ~30-60s), so it's launched as a subprocess and
# polled rather than run inline.
# ---------------------------------------------------------------------------
class ScenarioRun:
    def __init__(self):
        self.proc = None
        self.lines = []
        self.lock = threading.Lock()
        self.args = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, args: list):
        if self.is_running():
            raise RuntimeError("a scenario is already running")
        if DEFAULT_DB_PATH.exists():
            DEFAULT_DB_PATH.unlink()  # clean slate per scenario, matches CLI usage pattern
        ensure_full_schema()

        self.args = args
        self.lines = []
        cmd = [sys.executable, "-u", "simulator/flash_sale_scenario.py"] + args  # -u: unbuffered, so log lines stream live instead of arriving in one block at exit
        self.proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\n"))

    def status(self) -> dict:
        with self.lock:
            lines = list(self.lines[-300:])
        return {
            "running": self.is_running(),
            "returncode": self.proc.returncode if self.proc else None,
            "lines": lines,
        }


_scenario = ScenarioRun()


def load_manifest():
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def phase_for(window_start, window_end, phases):
    if not phases:
        return None
    mid = (window_start + window_end) / 2
    for phase in phases:
        if phase["start"] <= mid <= phase["end"]:
            return phase["name"]
    return None


# ---------------------------------------------------------------------------
# Read-only data endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
def get_status():
    exists = DEFAULT_DB_PATH.exists()
    cluster = ShardCluster()
    try:
        cluster_up = all(cluster.client(name).ping() for name in cluster.shard_names())
    except Exception:
        cluster_up = False
    result = {
        "db_exists": exists,
        "cluster_up": cluster_up,
        "shards": cluster.shard_names(),
        "model_exists": Path(DEFAULT_MODEL_PATH).exists(),
        "scenario_running": _scenario.is_running(),
        "num_windows": 0,
        "window_start_min": None,
        "window_start_max": None,
    }
    if exists:
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        row = conn.execute(
            "SELECT MIN(window_start), MAX(window_start), COUNT(DISTINCT window_start) FROM metric_windows"
        ).fetchone()
        has_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] > 0
        has_plan = conn.execute("SELECT COUNT(*) FROM relocation_plans").fetchone()[0] > 0
        conn.close()
        result["window_start_min"], result["window_start_max"], result["num_windows"] = row
        result["has_predictions"] = has_predictions
        result["has_plan"] = has_plan
    result["manifest"] = load_manifest()
    return result


@app.get("/api/shard-load")
def get_shard_load(window_lookback: int = 5):
    cluster = ShardCluster()
    record_load = record_recent_load(str(DEFAULT_DB_PATH), window_lookback=window_lookback)
    shard_load = shard_loads(record_load, cluster)
    values = list(shard_load.values())
    avg = sum(values) / len(values) if values else 0.0
    return {"shards": [{"name": k, "load": v} for k, v in shard_load.items()], "average": avg}


def _prediction_windows(conn):
    """One row per prediction window: start, end, max P(hot), how many flagged."""
    return [
        dict(r)
        for r in conn.execute(
            "SELECT window_start, window_end, MAX(p_ensemble) AS max_p, SUM(flagged) AS flagged "
            "FROM predictions GROUP BY window_start, window_end ORDER BY window_start ASC"
        ).fetchall()
    ]


@app.get("/api/records")
def get_records(limit: int = 300, window_start: float = None):
    """Per-record heat / P(hot) for one window. Defaults to the PEAK window
    (the one with the highest P(hot)), not the latest: the latest window of a
    finished scenario is its cooldown, where every record is quiet and the
    chart would look empty."""
    if not DEFAULT_DB_PATH.exists():
        return {"window_start": None, "records": [], "windows": []}

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    pred_windows = _prediction_windows(conn)
    manifest = load_manifest()
    phases = manifest.get("phases") if manifest else None
    windows = [
        {
            "window_start": w["window_start"],
            "phase": phase_for(w["window_start"], w["window_end"], phases),
            "max_p": w["max_p"],
            "flagged": int(w["flagged"] or 0),
        }
        for w in pred_windows
    ]
    peak = max(windows, key=lambda w: (w["max_p"] or 0)) if windows else None
    if window_start is None:
        if peak is not None:
            window_start = peak["window_start"]
        else:
            window_start = conn.execute("SELECT MAX(window_start) FROM metric_windows").fetchone()[0]
    if window_start is None:
        conn.close()
        return {"window_start": None, "records": [], "windows": windows}

    mw = {r["record_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM metric_windows WHERE window_start = ?", (window_start,)
    ).fetchall()}
    heat = {r["record_id"]: r["heat_score"] for r in conn.execute(
        "SELECT record_id, heat_score FROM heat_windows WHERE window_start = ?", (window_start,)
    ).fetchall()}
    pred = {r["record_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM predictions WHERE window_start = ?", (window_start,)
    ).fetchall()}
    conn.close()

    cluster = ShardCluster()
    records = []
    for rid in set(mw) | set(heat) | set(pred):
        p = pred.get(rid, {})
        records.append({
            "record_id": rid,
            "shard": cluster.shard_for_key(rid),
            "access_count": mw.get(rid, {}).get("access_count"),
            "heat_score": heat.get(rid),
            "p_ensemble": p.get("p_ensemble"),
            "confidence": p.get("confidence"),
            "flagged": bool(p.get("flagged")),
        })
    records.sort(key=lambda r: (r["p_ensemble"] if r["p_ensemble"] is not None else (r["heat_score"] or 0)), reverse=True)
    return {
        "window_start": window_start,
        "peak_window_start": peak["window_start"] if peak else None,
        "records": records[:limit],
        "windows": windows,
    }


@app.get("/api/prediction-timeline")
def get_prediction_timeline(top: int = 6):
    """P(hot) over time for the records that got hottest, with the scenario's
    phases and the adaptive threshold -- the Stage 4 checkpoint as a picture."""
    if not DEFAULT_DB_PATH.exists():
        return {"available": False}
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    windows = _prediction_windows(conn)
    if not windows:
        conn.close()
        return {"available": False}
    t0 = windows[0]["window_start"]

    top_ids = [
        r["record_id"]
        for r in conn.execute(
            "SELECT record_id, MAX(p_ensemble) AS m FROM predictions GROUP BY record_id ORDER BY m DESC LIMIT ?", (top,)
        ).fetchall()
    ]
    manifest = load_manifest() or {}
    kinds = {}
    for key, label in (("event_spike_records", "announced"), ("surprise_records", "surprise"), ("decoy_records", "decoy")):
        for rid in manifest.get(key, []):
            kinds[rid] = label
    if not manifest.get("event_spike_records"):  # older manifests: every spike record was announced
        for rid in manifest.get("spike_records", []):
            kinds.setdefault(rid, "announced")

    series = []
    for rid in top_ids:
        rows = conn.execute(
            "SELECT window_start, p_ensemble, flagged FROM predictions WHERE record_id = ? ORDER BY window_start", (rid,)
        ).fetchall()
        series.append({
            "record_id": rid,
            "kind": kinds.get(rid, "other"),
            "points": [{"t": r["window_start"] - t0, "p": r["p_ensemble"], "flagged": bool(r["flagged"])} for r in rows],
        })
    thresholds = [
        {"t": r["window_start"] - t0, "threshold": r["threshold"]}
        for r in conn.execute("SELECT window_start, MAX(threshold) AS threshold FROM predictions GROUP BY window_start ORDER BY window_start").fetchall()
    ]
    conn.close()

    phases = [
        {"name": ph["name"], "start": ph["start"] - t0, "end": ph["end"] - t0}
        for ph in (manifest.get("phases") or [])
    ]
    registered = manifest.get("event_registered_at")
    return {
        "available": True,
        "series": series,
        "thresholds": thresholds,
        "phases": phases,
        "event_announced_at": (registered - t0) if registered else None,
    }


@app.get("/api/aggregate-evaluation")
def get_aggregate_evaluation():
    """Stored result of planner/evaluate_aggregate.py (N independent scenarios)."""
    path = Path(__file__).resolve().parent.parent / "data" / "evaluation_aggregate.json"
    if not path.exists():
        return {"available": False, "reason": "run planner/evaluate_aggregate.py to generate the multi-scenario results"}
    with open(path) as f:
        data = json.load(f)
    data["available"] = True
    return data


@app.get("/api/relocation-plan")
def get_relocation_plan():
    if not DEFAULT_DB_PATH.exists():
        return {"moves": [], "window_start": None, "predictions_ready": False}
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    predictions_ready = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] > 0
    latest_created = conn.execute("SELECT MAX(created_at) FROM relocation_plans").fetchone()[0]
    if latest_created is None:
        conn.close()
        return {"moves": [], "window_start": None, "predictions_ready": predictions_ready}
    rows = conn.execute(
        "SELECT * FROM relocation_plans WHERE created_at >= ? ORDER BY expected_value DESC",
        (latest_created - 1.0,),
    ).fetchall()
    conn.close()
    moves = [dict(r) for r in rows]
    return {"moves": moves, "window_start": moves[0]["window_start"] if moves else None, "predictions_ready": predictions_ready}


@app.get("/api/evaluation")
def get_evaluation(min_probability: float = MIN_CANDIDATE_PROBABILITY, threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER):
    if not DEFAULT_DB_PATH.exists():
        return {"available": False, "reason": "no scenario data yet"}

    cluster = ShardCluster()
    hot_records = ever_hot_records(str(DEFAULT_DB_PATH))
    first_hot = first_hot_window_start(str(DEFAULT_DB_PATH))
    heatshard_result = compute_plan(str(DEFAULT_DB_PATH), min_probability=min_probability, cluster=cluster)
    if heatshard_result is None:
        return {"available": False, "reason": "no predictions found -- run the prediction pipeline first"}

    heatshard_window = heatshard_result["window_start"]
    heatshard_moves = [(m.record_id, m.source_shard, m.destination_shard) for m in heatshard_result["moves"]]

    reactive_window, reactive_shard_load, reactive_record_load = find_reactive_trigger(
        str(DEFAULT_DB_PATH), cluster, threshold_multiplier
    )
    reactive_result = (
        reactive_plan(reactive_shard_load, reactive_record_load, cluster, threshold_multiplier)
        if reactive_window is not None
        else {"moves": []}
    )

    systems = [
        ("static", [], heatshard_result["loads"], heatshard_result["record_load"]),
        (
            "reactive",
            reactive_result["moves"],
            reactive_shard_load or heatshard_result["loads"],
            reactive_record_load or heatshard_result["record_load"],
        ),
        ("heatshard", heatshard_moves, heatshard_result["loads"], heatshard_result["record_load"]),
    ]
    results = [evaluate_system(name, moves, sl, rl, hot_records) for name, moves, sl, rl in systems]

    return {
        "available": True,
        "ground_truth_hot_count": len(hot_records),
        "heatshard_window": heatshard_window,
        "reactive_window": reactive_window,
        "lead_time_seconds": (reactive_window - heatshard_window) if reactive_window is not None else None,
        # positive = HeatShard decided BEFORE the first hot window actually appeared
        "lead_vs_first_hot_seconds": (first_hot - heatshard_window) if first_hot is not None else None,
        "results": results,
    }


@app.get("/api/weight-history")
def get_weight_history():
    if not DEFAULT_DB_PATH.exists():
        return {"history": []}
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM weight_history ORDER BY window_index ASC").fetchall()
    conn.close()
    return {"history": [{"window_index": r["window_index"], "weights": json.loads(r["weights_json"])} for r in rows]}


@app.get("/api/threshold-history")
def get_threshold_history():
    if not DEFAULT_DB_PATH.exists():
        return {"history": []}
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM threshold_history ORDER BY window_index ASC").fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Actions: run a scenario, then run the predict+plan pipeline against it
# ---------------------------------------------------------------------------
@app.post("/api/scenario/run")
def run_scenario(payload: dict = Body(default={})):
    if _scenario.is_running():
        raise HTTPException(409, "a scenario is already running")

    seed = payload.get("seed") or random.randint(1, 100_000)
    args = [
        "--seed", str(seed),
        "--rate", str(payload.get("rate", 30)),
        "--window-seconds", str(payload.get("window_seconds", 3)),
        "--baseline-seconds", str(payload.get("baseline_seconds", 15)),
        "--lead-seconds", str(payload.get("lead_seconds", 9)),
        "--spike-seconds", str(payload.get("spike_seconds", 18)),
        "--cooldown-seconds", str(payload.get("cooldown_seconds", 15)),
        "--spike-num-records", str(payload.get("spike_num_records", 3)),
        "--spike-magnitude", str(payload.get("spike_magnitude", 8.0)),
        "--spike-bias", str(payload.get("spike_bias", 0.85)),
        # optional realism knobs (0 = the original, easy scenario): unannounced
        # surprise spikes, decoy events that never surge, and a ramped spike
        "--surprise-num-records", str(payload.get("surprise_num_records", 0)),
        "--decoy-num-records", str(payload.get("decoy_num_records", 0)),
        "--ramp-seconds", str(payload.get("ramp_seconds", 0)),
    ]
    try:
        _scenario.start(args)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"started": True, "seed": seed}


@app.get("/api/scenario/status")
def scenario_status():
    return _scenario.status()


@app.post("/api/pipeline/run")
def run_pipeline(payload: dict = Body(default={})):
    if _scenario.is_running():
        raise HTTPException(409, "wait for the running scenario to finish first")
    if not DEFAULT_DB_PATH.exists():
        raise HTTPException(400, "no scenario data yet -- run a scenario first")

    # compute_heat.py persists heat_windows/weight_history (Stage 3) --
    # run_prediction.py computes heat internally too but never writes those
    # tables, so both are needed for the dashboard's heat scores and
    # self-tuning charts to have anything to show.
    heat = subprocess.run(
        [sys.executable, "predictor/compute_heat.py"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120
    )

    pred_cmd = [sys.executable, "predictor/run_prediction.py"]
    if MANIFEST_PATH.exists():
        pred_cmd += ["--manifest", str(MANIFEST_PATH)]
    pred = subprocess.run(pred_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120)

    min_probability = payload.get("min_probability", MIN_CANDIDATE_PROBABILITY)
    plan = subprocess.run(
        [sys.executable, "planner/run_relocation.py", "--min-probability", str(min_probability)],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60,
    )

    return {
        "heat": {"returncode": heat.returncode, "stdout": heat.stdout[-4000:], "stderr": heat.stderr[-2000:]},
        "prediction": {"returncode": pred.returncode, "stdout": pred.stdout[-4000:], "stderr": pred.stderr[-2000:]},
        "planning": {"returncode": plan.returncode, "stdout": plan.stdout[-4000:], "stderr": plan.stderr[-2000:]},
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
