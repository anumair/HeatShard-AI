"""Stage 8: HeatShard AI live dashboard backend.

A thin FastAPI layer over the existing pipeline -- every endpoint reuses
Stage 1-7's own modules directly (ShardCluster, DependencyGraph,
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
from planner.dependency_graph import DEFAULT_MIN_CO_ACCESS, DependencyGraph  # noqa: E402
from planner.evaluate import evaluate_system, ever_hot_records, find_reactive_trigger  # noqa: E402
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
    own init_db() only runs when its CLI script does. A fresh scenario
    creates just the collector's tables, so any endpoint touching a later
    stage's table would 500 with "no such table" until the pipeline has
    run once. Call this right after the db file is (re)created so the
    full schema exists from the first moment, even empty."""
    collector_storage.init_db(DEFAULT_DB_PATH)
    heat_storage.init_db(DEFAULT_DB_PATH)
    planner_storage.init_db(DEFAULT_DB_PATH)


if DEFAULT_DB_PATH.exists():
    ensure_full_schema()


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


@app.get("/api/records")
def get_records(limit: int = 300):
    if not DEFAULT_DB_PATH.exists():
        return {"window_start": None, "records": []}

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    latest_pred = conn.execute("SELECT MAX(window_start) FROM predictions").fetchone()[0]
    latest_mw = conn.execute("SELECT MAX(window_start) FROM metric_windows").fetchone()[0]
    window_start = latest_pred if latest_pred is not None else latest_mw
    if window_start is None:
        conn.close()
        return {"window_start": None, "records": []}

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
    return {"window_start": window_start, "records": records[:limit]}


@app.get("/api/relocation-plan")
def get_relocation_plan():
    if not DEFAULT_DB_PATH.exists():
        return {"moves": [], "window_start": None}
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    latest_created = conn.execute("SELECT MAX(created_at) FROM relocation_plans").fetchone()[0]
    if latest_created is None:
        conn.close()
        return {"moves": [], "window_start": None}
    rows = conn.execute(
        "SELECT * FROM relocation_plans WHERE created_at >= ? ORDER BY expected_value DESC",
        (latest_created - 1.0,),
    ).fetchall()
    conn.close()
    moves = [dict(r) for r in rows]
    return {"moves": moves, "window_start": moves[0]["window_start"] if moves else None}


@app.get("/api/evaluation")
def get_evaluation(min_probability: float = MIN_CANDIDATE_PROBABILITY, threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER):
    if not DEFAULT_DB_PATH.exists():
        return {"available": False, "reason": "no scenario data yet"}

    cluster = ShardCluster()
    hot_records = ever_hot_records(str(DEFAULT_DB_PATH))
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
        "results": results,
    }


@app.get("/api/dependency-graph")
def get_dependency_graph(min_co_access: int = DEFAULT_MIN_CO_ACCESS):
    if not DEFAULT_DB_PATH.exists():
        return {"nodes": [], "edges": [], "num_components": 0}
    graph = DependencyGraph.build(str(DEFAULT_DB_PATH), min_co_access=min_co_access)
    nodes = [{"id": r, "role": r.split(":")[0]} for r in graph.records()]
    edges = [{"source": a, "target": b, "weight": w} for a, b, w in graph.all_edges()]
    return {"nodes": nodes, "edges": edges, "num_components": len(graph.connected_components())}


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
