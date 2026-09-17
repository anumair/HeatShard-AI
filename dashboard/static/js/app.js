/* HeatShard AI dashboard — polls the FastAPI backend and renders every panel.
   No build step, no framework: plain fetch + Chart.js + hand-rolled SVG for
   the dependency graph, matching the rest of this project's "no unnecessary
   dependency" style. */

const COLORS = {
  text: "#e5e9f0",
  dim: "#8b96a5",
  faint: "#5b6572",
  grid: "rgba(255,255,255,0.06)",
  accent: "#f97316",
  brand: "#22d3ee",
  ok: "#34d399",
  warn: "#fbbf24",
  danger: "#f87171",
  shards: ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f472b6"],
  systems: { static: "#5b6572", reactive: "#f87171", heatshard: "#22d3ee" },
  roles: { product: "#60a5fa", review: "#f472b6", order: "#34d399", reviews: "#f472b6", inventory: "#34d399" },
};

Chart.defaults.color = COLORS.dim;
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.borderColor = COLORS.grid;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return Number(n).toFixed(digits);
}

function shardColor(name) {
  const idx = parseInt(name.replace("shard-", ""), 10) || 0;
  return COLORS.shards[idx % COLORS.shards.length];
}

function shortId(recordId) {
  const [prefix, rest] = recordId.split(":");
  return rest && rest.length > 8 ? `${prefix}:${rest.slice(0, 8)}` : recordId;
}

/* ------------------------------------------------------------------ */
/* Chart instances (created once, updated in place)                    */
/* ------------------------------------------------------------------ */
const charts = {};

function ensureChart(id, config) {
  if (charts[id]) {
    charts[id].data = config.data;
    if (config.options) charts[id].options = config.options;
    charts[id].update("none");
    return charts[id];
  }
  charts[id] = new Chart($(id).getContext("2d"), config);
  return charts[id];
}

/* ------------------------------------------------------------------ */
/* Status / controls                                                   */
/* ------------------------------------------------------------------ */
let scenarioPolling = false;

function setPill(id, state, label) {
  const el = $(id);
  el.className = `status-pill ${state}`;
  el.querySelector("span").nextSibling ? (el.lastChild.textContent = label) : null;
  el.innerHTML = `<span class="dot"></span>${label}`;
}

async function refreshStatus() {
  let status;
  try {
    status = await api("/api/status");
  } catch (e) {
    setPill("pillCluster", "bad", "Cluster offline");
    return null;
  }

  setPill("pillCluster", status.cluster_up ? "ok" : "bad", status.cluster_up ? "Cluster live" : "Cluster unreachable");
  setPill(
    "pillScenario",
    status.scenario_running ? "warn" : status.db_exists ? "ok" : "",
    status.scenario_running ? "Scenario running" : status.db_exists ? "Scenario idle" : "No scenario yet"
  );
  setPill(
    "pillPipeline",
    status.has_predictions ? "ok" : "",
    status.has_predictions ? "Predictions ready" : "Pipeline not run"
  );

  $("btnLaunch").disabled = status.scenario_running;
  $("btnPipeline").disabled = status.scenario_running || !status.db_exists;

  if (status.num_windows) {
    $("windowInfo").textContent = `${status.num_windows} windows recorded`;
  } else {
    $("windowInfo").textContent = "No scenario data yet";
  }

  renderPhaseTrack(status.manifest);

  if (status.scenario_running && !scenarioPolling) {
    scenarioPolling = true;
    pollScenarioLog();
  }

  return status;
}

function renderPhaseTrack(manifest) {
  const track = $("phaseTrack");
  if (!manifest || !manifest.phases || !manifest.phases.length) {
    track.hidden = true;
    return;
  }
  track.hidden = false;
  const now = Date.now() / 1000;
  const phases = manifest.phases;
  const segs = track.querySelectorAll(".phase-seg");
  segs.forEach((seg) => {
    const name = seg.dataset.phase;
    const phase = phases.find((p) => p.name === name);
    seg.classList.remove("active", "past");
    if (!phase) return;
    if (now >= phase.start && now <= phase.end) seg.classList.add("active");
    else if (now > phase.end) seg.classList.add("past");
  });

  const cursor = $("phaseCursor");
  const start = phases[0].start;
  const end = phases[phases.length - 1].end;
  const span = end - start;
  if (span > 0 && now >= start && now <= end) {
    cursor.style.display = "block";
    cursor.style.left = `${((now - start) / span) * 100}%`;
  } else {
    cursor.style.display = "none";
  }
}

async function pollScenarioLog() {
  while (true) {
    let status;
    try {
      status = await api("/api/scenario/status");
    } catch (e) {
      break;
    }
    $("logBox").textContent = status.lines.length ? status.lines.join("\n") : "Waiting for output…";
    $("logBox").scrollTop = $("logBox").scrollHeight;
    $("logSub").textContent = status.running ? "running…" : `finished (code ${status.returncode})`;

    if (!status.running) {
      scenarioPolling = false;
      await refreshAll();
      break;
    }
    await refreshShardLoad();
    await sleep(1500);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* ------------------------------------------------------------------ */
/* Shard load                                                          */
/* ------------------------------------------------------------------ */
async function refreshShardLoad() {
  let data;
  try {
    data = await api("/api/shard-load");
  } catch (e) {
    return;
  }
  const labels = data.shards.map((s) => s.name);
  const values = data.shards.map((s) => s.load);
  const colors = labels.map(shardColor);

  ensureChart("chartShardLoad", {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6, maxBarThickness: 44 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `load ${fmt(c.raw, 0)}` } } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: COLORS.grid }, beginAtZero: true },
      },
    },
  });
}

/* ------------------------------------------------------------------ */
/* Records: heat / prediction scatter + table                          */
/* ------------------------------------------------------------------ */
async function refreshRecords() {
  let data;
  try {
    data = await api("/api/records");
  } catch (e) {
    return;
  }
  const records = data.records || [];

  const flagged = records.filter((r) => r.flagged);
  const normal = records.filter((r) => !r.flagged);

  const toPoint = (r) => ({
    x: r.heat_score ?? 0,
    y: r.p_ensemble ?? 0,
    r: Math.min(18, 4 + Math.sqrt(r.access_count || 1)),
    record: r,
  });

  ensureChart("chartRecords", {
    type: "bubble",
    data: {
      datasets: [
        { label: "flagged", data: flagged.map(toPoint), backgroundColor: COLORS.accent + "cc", borderColor: COLORS.accent },
        { label: "not flagged", data: normal.map(toPoint), backgroundColor: COLORS.brand + "55", borderColor: COLORS.brand + "88" },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: (c) => {
              const r = c.raw.record;
              return `${shortId(r.record_id)} · heat ${fmt(r.heat_score)} · P ${fmt(r.p_ensemble)}`;
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "heat score" }, grid: { color: COLORS.grid } },
        y: { title: { display: true, text: "P(hotspot)" }, grid: { color: COLORS.grid }, min: 0, max: 1 },
      },
    },
  });

  const tbody = document.querySelector("#tableRecords tbody");
  tbody.innerHTML = "";
  records.slice(0, 40).forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${shortId(r.record_id)}</td>
      <td><span class="chip chip-shard" style="background:${shardColor(r.shard)}">${r.shard}</span></td>
      <td class="num">${r.access_count ?? "–"}</td>
      <td class="num">${fmt(r.heat_score)}</td>
      <td class="num">${fmt(r.p_ensemble)}</td>
      <td class="num">${fmt(r.confidence)}</td>
      <td>${r.flagged ? '<span class="chip chip-flagged">FLAGGED</span>' : ""}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ------------------------------------------------------------------ */
/* Relocation plan                                                     */
/* ------------------------------------------------------------------ */
async function refreshPlan() {
  let data;
  try {
    data = await api("/api/relocation-plan");
  } catch (e) {
    return;
  }
  const moves = data.moves || [];
  $("planSub").textContent = moves.length ? `${moves.length} move(s)` : "no plan yet";

  const tbody = document.querySelector("#tablePlan tbody");
  tbody.innerHTML = "";
  if (!moves.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-msg">Run the pipeline to compute a relocation plan.</td></tr>`;
    return;
  }
  moves.forEach((m) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${shortId(m.record_id)}</td>
      <td><span class="chip chip-shard" style="background:${shardColor(m.source_shard)}">${m.source_shard}</span></td>
      <td><span class="chip chip-shard" style="background:${shardColor(m.destination_shard)}">${m.destination_shard}</span></td>
      <td class="num">${fmt(m.expected_value)}</td>
      <td class="num">${fmt(m.predicted_benefit)}</td>
      <td class="num">${fmt(m.predicted_cost)}</td>
      <td class="num">${fmt(m.confidence)}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ------------------------------------------------------------------ */
/* Dependency graph (hand-rolled SVG)                                  */
/* ------------------------------------------------------------------ */
async function refreshGraph() {
  let data;
  try {
    data = await api("/api/dependency-graph");
  } catch (e) {
    return;
  }
  const svg = $("depGraph");
  svg.innerHTML = "";
  $("graphSub").textContent = `${data.nodes.length} records · ${data.num_components} triangle(s)`;

  if (!data.nodes.length) {
    return;
  }

  // connected components via union-find over the edge list
  const parent = {};
  const find = (x) => (parent[x] === x || !parent[x] ? (parent[x] = parent[x] || x) : (parent[x] = find(parent[x])));
  const union = (a, b) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  data.nodes.forEach((n) => (parent[n.id] = n.id));
  data.edges.forEach((e) => union(e.source, e.target));

  const groups = {};
  data.nodes.forEach((n) => {
    const root = find(n.id);
    (groups[root] = groups[root] || []).push(n.id);
  });

  const components = Object.values(groups).slice(0, 24);
  const cols = 6;
  const cellW = 100, cellH = 85, pad = 30;
  const positions = {};

  components.forEach((comp, idx) => {
    const cx = pad + (idx % cols) * cellW + cellW / 2;
    const cy = pad + Math.floor(idx / cols) * cellH + cellH / 2;
    const n = comp.length;
    comp.forEach((id, i) => {
      const angle = (2 * Math.PI * i) / Math.max(n, 1) - Math.PI / 2;
      const radius = n === 1 ? 0 : 26;
      positions[id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
    });
  });

  const ns = "http://www.w3.org/2000/svg";
  const rows = Math.ceil(components.length / cols);
  svg.setAttribute("viewBox", `0 0 ${cols * cellW + pad} ${Math.max(rows * cellH + pad, 200)}`);

  data.edges.forEach((e) => {
    const a = positions[e.source], b = positions[e.target];
    if (!a || !b) return;
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    line.setAttribute("stroke", "rgba(255,255,255,0.15)");
    line.setAttribute("stroke-width", "1.5");
    svg.appendChild(line);
  });

  const tooltip = $("graphTooltip");
  data.nodes.forEach((n) => {
    const pos = positions[n.id];
    if (!pos) return;
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", pos.x); circle.setAttribute("cy", pos.y);
    circle.setAttribute("r", 7);
    circle.setAttribute("fill", COLORS.roles[n.role] || "#888");
    circle.setAttribute("stroke", "#0a0e14");
    circle.setAttribute("stroke-width", "1.5");
    circle.style.cursor = "pointer";
    circle.addEventListener("mousemove", (ev) => {
      tooltip.hidden = false;
      tooltip.textContent = shortId(n.id);
      tooltip.style.left = `${ev.offsetX + 12}px`;
      tooltip.style.top = `${ev.offsetY + 4}px`;
    });
    circle.addEventListener("mouseleave", () => (tooltip.hidden = true));
    svg.appendChild(circle);
  });
}

/* ------------------------------------------------------------------ */
/* Evaluation                                                          */
/* ------------------------------------------------------------------ */
async function refreshEvaluation() {
  let data;
  try {
    data = await api("/api/evaluation");
  } catch (e) {
    return;
  }
  if (!data.available) {
    $("evalHeadline").innerHTML = `<div class="empty-msg">${data.reason || "Not enough data yet."}</div>`;
    return;
  }

  const results = data.results;
  const byName = Object.fromEntries(results.map((r) => [r.name, r]));
  const leadTime = data.lead_time_seconds;

  $("evalHeadline").innerHTML = `
    <div class="eval-stat accent"><div class="v">${leadTime !== null ? fmt(leadTime, 0) + "s" : "–"}</div><div class="l">HeatShard lead time</div></div>
    <div class="eval-stat"><div class="v">${data.ground_truth_hot_count}</div><div class="l">records actually hot</div></div>
    <div class="eval-stat good"><div class="v">${fmt(byName.heatshard.precision * 100, 0)}%</div><div class="l">HeatShard precision</div></div>
    <div class="eval-stat bad"><div class="v">${fmt(byName.reactive.data_movement, 0)}×</div><div class="l">reactive data movement</div></div>
  `;

  const labels = results.map((r) => r.name);
  const colors = results.map((r) => COLORS.systems[r.name]);

  ensureChart("chartMovement", barConfig(labels, results.map((r) => r.data_movement), colors, "records moved"));
  ensureChart("chartFpRate", barConfig(labels, results.map((r) => r.false_positive_rate), colors, "FP rate", 1));

  ensureChart("chartPrecRecall", {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "precision", data: results.map((r) => r.precision), backgroundColor: COLORS.brand, borderRadius: 4 },
        { label: "recall", data: results.map((r) => r.recall), backgroundColor: COLORS.accent, borderRadius: 4 },
      ],
    },
    options: chartOptions("precision / recall", 1),
  });

  ensureChart("chartVariance", {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "before", data: results.map((r) => r.variance_before), backgroundColor: "#5b6572", borderRadius: 4 },
        { label: "after", data: results.map((r) => r.variance_after), backgroundColor: COLORS.warn, borderRadius: 4 },
      ],
    },
    options: chartOptions("load variance"),
  });
}

function barConfig(labels, values, colors, title, max) {
  return {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4 }] },
    options: chartOptions(title, max, true),
  };
}

function chartOptions(title, max, hideLegend) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: !hideLegend, position: "top" },
      title: { display: true, text: title, color: COLORS.faint, font: { size: 10.5, weight: "600" } },
    },
    scales: {
      x: { grid: { display: false } },
      y: { grid: { color: COLORS.grid }, beginAtZero: true, max },
    },
  };
}

/* ------------------------------------------------------------------ */
/* Self-tuning (weights + adaptive threshold)                          */
/* ------------------------------------------------------------------ */
async function refreshAdaptation() {
  let weightData, thresholdData;
  try {
    [weightData, thresholdData] = await Promise.all([api("/api/weight-history"), api("/api/threshold-history")]);
  } catch (e) {
    return;
  }

  $("emptyWeights").hidden = weightData.history.length > 0;
  $("emptyThreshold").hidden = thresholdData.history.length > 0;

  if (weightData.history.length) {
    const featureNames = Object.keys(weightData.history[0].weights);
    const palette = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#22d3ee", "#f97316"];
    ensureChart("chartWeights", {
      type: "line",
      data: {
        labels: weightData.history.map((h) => `w${h.window_index}`),
        datasets: featureNames.map((name, i) => ({
          label: name,
          data: weightData.history.map((h) => h.weights[name]),
          borderColor: palette[i % palette.length],
          backgroundColor: "transparent",
          tension: 0.3,
          pointRadius: 2,
        })),
      },
      options: chartOptions("heat index weights over refits"),
    });
  }

  if (thresholdData.history.length) {
    ensureChart("chartThreshold", {
      type: "line",
      data: {
        labels: thresholdData.history.map((h) => `w${h.window_index}`),
        datasets: [
          {
            label: "confidence threshold",
            data: thresholdData.history.map((h) => h.threshold),
            borderColor: COLORS.accent,
            backgroundColor: COLORS.accent + "22",
            fill: true,
            tension: 0.3,
            pointRadius: 2,
          },
        ],
      },
      options: chartOptions("adaptive confidence threshold", 1, true),
    });
  }
}

/* ------------------------------------------------------------------ */
/* Orchestration                                                       */
/* ------------------------------------------------------------------ */
async function refreshAll() {
  const status = await refreshStatus();
  await refreshShardLoad();
  if (status && status.db_exists) {
    await refreshRecords();
    await refreshPlan();
    await refreshGraph();
    await refreshEvaluation();
    await refreshAdaptation();
  }
}

$("btnLaunch").addEventListener("click", async () => {
  const payload = {
    seed: $("seedInput").value ? parseInt($("seedInput").value, 10) : null,
    spike_num_records: parseInt($("spikeCountInput").value, 10) || 3,
    spike_magnitude: parseFloat($("magnitudeInput").value) || 8,
  };
  $("btnLaunch").disabled = true;
  $("logBox").textContent = "Starting scenario…";
  try {
    await api("/api/scenario/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  } catch (e) {
    $("logBox").textContent = `Failed to start: ${e.message}`;
    $("btnLaunch").disabled = false;
    return;
  }
  await refreshStatus();
});

$("btnPipeline").addEventListener("click", async () => {
  $("btnPipeline").disabled = true;
  $("btnPipeline").textContent = "Running…";
  try {
    await api("/api/pipeline/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
  } catch (e) {
    alert(`Pipeline failed: ${e.message}`);
  }
  $("btnPipeline").textContent = "Run Prediction & Planning";
  await refreshAll();
});

refreshAll();
setInterval(refreshStatus, 4000);
setInterval(() => {
  if (!scenarioPolling) refreshAll();
}, 10000);
