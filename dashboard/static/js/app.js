/* HeatShard AI dashboard — polls the FastAPI backend and renders every panel.
   No build step, no framework: plain fetch + Chart.js + hand-rolled SVG for
   the charts, matching the rest of this project's "no unnecessary
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

// On a fresh page load after a scenario has already finished, show its last
// log instead of the "Launch a flash sale" placeholder.
async function showLastScenarioLog() {
  try {
    const status = await api("/api/scenario/status");
    if (!status.running && status.lines.length) {
      $("logBox").textContent = status.lines.join("\n");
      $("logSub").textContent = `last run (code ${status.returncode})`;
    }
  } catch (e) {
    /* leave the placeholder */
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
// null = follow the PEAK window (highest P(hotspot)) -- the latest window of a
// finished scenario is its cooldown, where everything is quiet and the chart
// would look empty. Moving the slider pins a specific window.
let selectedWindow = null;
let recordWindows = [];

function syncWindowPicker(data) {
  const slider = $("windowSlider");
  recordWindows = data.windows || [];
  const n = recordWindows.length;
  slider.disabled = n === 0;
  slider.max = Math.max(0, n - 1);
  const idx = Math.max(0, recordWindows.findIndex((w) => w.window_start === data.window_start));
  slider.value = idx;
  if (!n) {
    $("windowLabel").textContent = "–";
    $("recordsSub").textContent = "size = access count, color = flagged";
    return;
  }
  const w = recordWindows[idx];
  const isPeak = data.window_start === data.peak_window_start;
  $("windowLabel").textContent =
    `${idx + 1}/${n} · ${w.phase || "?"} · max P ${fmt(w.max_p)} · ${w.flagged} flagged${isPeak ? " · PEAK" : ""}`;
  $("recordsSub").textContent = `window ${idx + 1} of ${n}${isPeak ? " (peak)" : ""} · size = access count, color = flagged`;
}

$("windowSlider").addEventListener("input", () => {
  const w = recordWindows[parseInt($("windowSlider").value, 10)];
  if (!w) return;
  selectedWindow = w.window_start;
  refreshRecords();
});
$("btnPeak").addEventListener("click", () => {
  selectedWindow = null;
  refreshRecords();
});

async function refreshRecords() {
  let data;
  try {
    data = await api("/api/records" + (selectedWindow !== null ? `?window_start=${selectedWindow}` : ""));
  } catch (e) {
    return;
  }
  syncWindowPicker(data);
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
/* P(hotspot) over time                                                */
/* ------------------------------------------------------------------ */
const PHASE_TINT = {
  baseline: "rgba(255,255,255,0.025)",
  pre_spike: "rgba(251,191,36,0.07)",
  spike: "rgba(249,115,22,0.10)",
  cooldown: "rgba(96,165,250,0.05)",
};

// Draws the scenario's phase bands behind the lines, plus a vertical marker
// for when the flash-sale event was announced.
const phaseBandsPlugin = {
  id: "phaseBands",
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.phases) return;
    const { ctx, chartArea, scales } = chart;
    const x = scales.x;
    ctx.save();
    ctx.font = "600 10px -apple-system, sans-serif";
    ctx.textBaseline = "top";
    opts.phases.forEach((ph) => {
      const left = Math.max(chartArea.left, x.getPixelForValue(ph.start));
      const right = Math.min(chartArea.right, x.getPixelForValue(ph.end));
      if (right <= left) return;
      ctx.fillStyle = PHASE_TINT[ph.name] || "rgba(255,255,255,0.03)";
      ctx.fillRect(left, chartArea.top, right - left, chartArea.bottom - chartArea.top);
      ctx.fillStyle = COLORS.faint;
      ctx.fillText(ph.name.toUpperCase(), left + 6, chartArea.top + 5);
    });
    if (opts.eventAt !== null && opts.eventAt !== undefined) {
      const ex = x.getPixelForValue(opts.eventAt);
      if (ex >= chartArea.left && ex <= chartArea.right) {
        ctx.strokeStyle = COLORS.brand;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(ex, chartArea.top);
        ctx.lineTo(ex, chartArea.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.brand;
        ctx.fillText("event announced", ex + 5, chartArea.bottom - 14);
      }
    }
    ctx.restore();
  },
};

async function refreshTimeline() {
  let data;
  try {
    data = await api("/api/prediction-timeline");
  } catch (e) {
    return;
  }
  $("emptyTimeline").hidden = !!data.available;
  if (!data.available) {
    if (charts.chartTimeline) {
      charts.chartTimeline.destroy();
      delete charts.chartTimeline;
    }
    return;
  }

  const palette = ["#22d3ee", "#f97316", "#a78bfa", "#34d399", "#f472b6", "#fbbf24"];
  const kindLabel = { announced: "announced sale", surprise: "surprise spike", decoy: "decoy event", other: "" };
  const datasets = data.series.map((sr, i) => ({
    label: `${shortId(sr.record_id)}${kindLabel[sr.kind] ? " · " + kindLabel[sr.kind] : ""}`,
    data: sr.points.map((p) => ({ x: p.t, y: p.p })),
    borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length],
    borderWidth: 2,
    tension: 0.25,
    pointRadius: sr.points.map((p) => (p.flagged ? 5 : 1.5)),
    pointHoverRadius: 6,
  }));
  datasets.push({
    label: "acting threshold",
    data: data.thresholds.map((t) => ({ x: t.t, y: t.threshold })),
    borderColor: COLORS.dim,
    borderDash: [6, 4],
    borderWidth: 1.5,
    pointRadius: 0,
    stepped: true,
  });

  ensureChart("chartTimeline", {
    type: "line",
    data: { datasets },
    plugins: [phaseBandsPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        phaseBands: { phases: data.phases, eventAt: data.event_announced_at },
        legend: { position: "top" },
        tooltip: { callbacks: { title: (items) => `t = ${fmt(items[0].parsed.x, 0)} s`, label: (c) => `${c.dataset.label}: P ${fmt(c.parsed.y)}` } },
      },
      scales: {
        x: { type: "linear", title: { display: true, text: "seconds into scenario" }, grid: { color: COLORS.grid } },
        y: { title: { display: true, text: "P(hotspot next window)" }, min: 0, max: 1, grid: { color: COLORS.grid } },
      },
    },
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
  const ranWithoutMoves = !moves.length && data.predictions_ready;
  $("planSub").textContent = moves.length ? `${moves.length} move(s)` : ranWithoutMoves ? "no move recommended" : "no plan yet";

  const tbody = document.querySelector("#tablePlan tbody");
  tbody.innerHTML = "";
  if (!moves.length) {
    const msg = ranWithoutMoves
      ? "The pipeline ran, but no flagged record had a positive expected value (for example its shard is already lightly loaded), so no relocation is recommended."
      : "Run the pipeline to compute a relocation plan.";
    tbody.innerHTML = `<tr><td colspan="7" class="empty-msg">${msg}</td></tr>`;
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
  const leadVsHot = data.lead_vs_first_hot_seconds;
  const timing =
    leadVsHot === null || leadVsHot === undefined
      ? { value: "–", label: "HeatShard timing vs first hotspot" }
      : leadVsHot > 0
        ? { value: fmt(leadVsHot, 0) + "s", label: "HeatShard acted BEFORE the first hotspot" }
        : { value: fmt(-leadVsHot, 0) + "s", label: "HeatShard acted AFTER the first hotspot appeared" };

  const hs = byName.heatshard;
  const re = byName.reactive;
  const hits = (r) => Math.round(r.precision * r.data_movement); // moves that were truly hot
  const movesText = (r) => (r.data_movement ? `${hits(r)} of ${r.data_movement}` : "none");

  $("evalHeadline").innerHTML = `
    <div class="eval-stat accent"><div class="v">${timing.value}</div><div class="l">${timing.label}</div></div>
    <div class="eval-stat"><div class="v">${data.ground_truth_hot_count}</div><div class="l">records actually hot</div></div>
    <div class="eval-stat good"><div class="v">${movesText(hs)}</div><div class="l">HeatShard moves that were truly hot</div><div class="s">recall ${fmt(hs.recall * 100, 0)}% of the hot records</div></div>
    <div class="eval-stat bad"><div class="v">${movesText(re)}</div><div class="l">reactive moves that were truly hot</div><div class="s">${re.data_movement} records moved in total</div></div>
  `;

  const labels = results.map((r) => r.name);
  const colors = results.map((r) => COLORS.systems[r.name]);

  // "static" never moves anything, so precision / recall / FP rate are undefined
  // for it -- showing three empty bars for it only added noise.
  const acting = results.filter((r) => r.name !== "static");
  const actingLabels = acting.map((r) => r.name);
  const actingColors = acting.map((r) => COLORS.systems[r.name]);

  ensureChart("chartMovement", barConfig(labels, results.map((r) => r.data_movement), colors, "records moved"));
  ensureChart("chartFpRate", barConfig(actingLabels, acting.map((r) => r.false_positive_rate), actingColors, "false-positive rate", 1));

  ensureChart("chartPrecRecall", {
    type: "bar",
    data: {
      labels: actingLabels,
      datasets: [
        { label: "precision", data: acting.map((r) => r.precision), backgroundColor: COLORS.brand, borderRadius: 4 },
        { label: "recall", data: acting.map((r) => r.recall), backgroundColor: COLORS.accent, borderRadius: 4 },
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

/* ------------------------------------------------------------------ */
/* Aggregate evaluation: per-scenario dots + mean ± std                */
/* ------------------------------------------------------------------ */
const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const std = (a) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((v) => (v - m) ** 2)));
};
const median = (a) => {
  const b = [...a].sort((x, y) => x - y);
  const h = Math.floor(b.length / 2);
  return b.length % 2 ? b[h] : (b[h - 1] + b[h]) / 2;
};

// Draws a mean ± std bar over each group of dots.
const errorBarsPlugin = {
  id: "errorBars",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.groups) return;
    const { ctx, chartArea, scales } = chart;
    ctx.save();
    ctx.lineWidth = 2;
    opts.groups.forEach((g, i) => {
      const x = scales.x.getPixelForValue(i);
      const lo = Math.max(scales.y.min, g.mean - g.std);
      const hi = Math.min(scales.y.max, g.mean + g.std);
      const y1 = scales.y.getPixelForValue(lo);
      const y2 = scales.y.getPixelForValue(hi);
      ctx.strokeStyle = g.color;
      ctx.beginPath();
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y2);
      ctx.moveTo(x - 7, y1);
      ctx.lineTo(x + 7, y1);
      ctx.moveTo(x - 7, y2);
      ctx.lineTo(x + 7, y2);
      ctx.stroke();
    });
    ctx.restore();
  },
};

function stripChart(canvasId, title, groups, max) {
  const dots = groups.map((g, i) => ({
    label: g.name,
    data: g.values.map((v, k) => ({ x: i + (((k * 37) % 100) / 100 - 0.5) * 0.45, y: v })),
    backgroundColor: g.color + "88",
    borderColor: g.color,
    pointRadius: 3.5,
    order: 2,
  }));
  const means = groups.map((g, i) => ({
    label: `${g.name} mean`,
    data: [{ x: i, y: mean(g.values) }],
    backgroundColor: g.color,
    borderColor: "#fff",
    borderWidth: 1.5,
    pointStyle: "rectRot",
    pointRadius: 8,
    order: 1,
  }));
  ensureChart(canvasId, {
    type: "scatter",
    data: { datasets: [...dots, ...means] },
    plugins: [errorBarsPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: title, color: COLORS.faint, font: { size: 10.5, weight: "600" } },
        errorBars: { groups: groups.map((g) => ({ mean: mean(g.values), std: std(g.values), color: g.color })) },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, max === 1 ? 2 : 0)}` } },
      },
      scales: {
        x: {
          min: -0.6,
          max: groups.length - 0.4,
          grid: { display: false },
          ticks: { stepSize: 1, callback: (v) => (Number.isInteger(v) && groups[v] ? groups[v].name : "") },
        },
        y: { beginAtZero: true, max, grid: { color: COLORS.grid } },
      },
    },
  });
}

async function refreshAggregate() {
  let data;
  try {
    data = await api("/api/aggregate-evaluation");
  } catch (e) {
    return;
  }
  const empty = $("aggEmpty");
  if (!data.available) {
    empty.hidden = false;
    empty.textContent = data.reason || "No multi-scenario results yet.";
    $("aggHeadline").innerHTML = "";
    $("aggNote").textContent = "";
    return;
  }
  empty.hidden = true;

  const run = data.per_run;
  const n = data.n_runs;
  const c = COLORS.systems;
  const hsPrecision = run.heatshard.precision;
  const nPerfect = hsPrecision.filter((v) => v >= 0.999).length;
  const nNoMove = run.heatshard.data_movement.filter((v) => v === 0).length;
  const peakChange = run.heatshard.peak_variance_after.map((a, i) => 1 - a / run.heatshard.peak_variance_before[i]);
  const nImproved = peakChange.filter((v) => v > 1e-9).length;
  const lead = (data.lead_vs_first_hot || {}).heatshard || [];
  const nEarly = lead.filter((v) => v > 0).length;

  $("aggSub").textContent = `${n} never-trained-on scenarios · each dot is one scenario · diamond = mean · bar = ±1 std`;
  $("aggHeadline").innerHTML = `
    <div class="eval-stat accent"><div class="v">${fmt(mean(run.heatshard.data_movement), 1)} vs ${fmt(mean(run.reactive.data_movement), 1)}</div><div class="l">records moved: HeatShard vs reactive</div></div>
    <div class="eval-stat good"><div class="v">${fmt(mean(hsPrecision), 2)} ± ${fmt(std(hsPrecision), 2)}</div><div class="l">HeatShard precision</div><div class="s">${nPerfect} of ${n} runs perfect · ${nNoMove} made no move</div></div>
    <div class="eval-stat"><div class="v">${fmt(mean(run.heatshard.recall), 2)} ± ${fmt(std(run.heatshard.recall), 2)}</div><div class="l">HeatShard recall</div><div class="s">reactive ${fmt(mean(run.reactive.recall), 2)}</div></div>
    <div class="eval-stat"><div class="v">${fmt(-100 * median(peakChange), 0)}%</div><div class="l">median per-run change in peak-load variance</div><div class="s">improved in ${nImproved} of ${n} runs</div></div>
  `;

  stripChart("aggMoved", "records moved per scenario", [
    { name: "reactive", color: c.reactive, values: run.reactive.data_movement },
    { name: "heatshard", color: c.heatshard, values: run.heatshard.data_movement },
  ]);
  stripChart("aggPrecision", "precision per scenario", [
    { name: "reactive", color: c.reactive, values: run.reactive.precision },
    { name: "heatshard", color: c.heatshard, values: run.heatshard.precision },
  ], 1);
  stripChart("aggRecall", "recall per scenario", [
    { name: "reactive", color: c.reactive, values: run.reactive.recall },
    { name: "heatshard", color: c.heatshard, values: run.heatshard.recall },
  ], 1);
  stripChart("aggVariance", "shard-load variance at the load peak", [
    { name: "static", color: c.static, values: run.static.peak_variance_after },
    { name: "reactive", color: c.reactive, values: run.reactive.peak_variance_after },
    { name: "heatshard", color: c.heatshard, values: run.heatshard.peak_variance_after },
  ]);

  $("aggNote").textContent =
    `HeatShard acted before the first hotspot in ${nEarly} of ${lead.length} scenarios, so no lead-time advantage is claimed. ` +
    `Precision is bimodal (perfect or no move), which is why its std is large. Variance is lower-is-better; static = doing nothing.`;
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
    // Bar, not line: with only one or two refits (typical for a short demo
    // scenario -- refits need --refit-every windows, default 10), a line
    // chart just draws isolated dots, and several weights routinely
    // collapse to exactly 0 after clip+renormalize, so multiple dots stack
    // invisibly on the same point. Grouped bars stay legible at any refit
    // count, including one.
    ensureChart("chartWeights", {
      type: "bar",
      data: {
        labels: weightData.history.map((h) => `w${h.window_index}`),
        datasets: featureNames.map((name, i) => ({
          label: name,
          data: weightData.history.map((h) => h.weights[name]),
          backgroundColor: palette[i % palette.length],
          borderRadius: 3,
        })),
      },
      options: chartOptions("heat index weights over refits", 1),
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
    await refreshTimeline();
    await refreshPlan();
    await refreshEvaluation();
    await refreshAggregate();
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
  selectedWindow = null; // a new scenario has new windows
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
  selectedWindow = null;
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
showLastScenarioLog();
setInterval(refreshStatus, 4000);
setInterval(() => {
  if (!scenarioPolling) refreshAll();
}, 10000);
