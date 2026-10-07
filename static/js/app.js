/* Boiler monitoring UI: tabs, live polling, per-panel history graphs, controls.
   Vanilla JS only — no framework (project rule). */
(() => {
  "use strict";

  const CFG = window.BHEL || {};
  const POLL_LIVE_MS = 1200;
  const POLL_SITUATION_MS = 4000;
  const HISTORY = 300;          // samples kept per series
  const TOAST_MS = { ok: 6000, warn: 8000, err: 10000 };
  const AXIS_MARGIN = { l: 52, r: 10, t: 10, b: 16 };

  const COLOURS = CFG.levelColours || {
    NOMINAL: "#1f9d55", WATCH: "#d9a406", ALERT: "#e2701a", CRITICAL: "#c62828",
  };
  const SERIES_COLOURS = ["#4f8ef7", "#2dd4bf", "#c084fc", "#f472b6", "#facc15", "#38bdf8", "#fb923c"];

  const state = {
    parts: {},            // part key -> {name, criticality, tags}
    partTags: {},
    scoreHistory: {},     // part key -> [{t, v}]
    tagHistory: {},       // tag -> [{t, v}]
    levelHistory: [],     // [{t, level}]
    thresholds: {},
    selected: CFG.partKey || null,
    lastLive: null,
    tab: "live",
    busy: false,
  };

  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const fmt = (value, digits) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    const abs = Math.abs(value);
    if (digits !== undefined) return value.toFixed(digits);
    if (abs >= 1000) return value.toFixed(0);
    if (abs >= 100) return value.toFixed(1);
    return value.toFixed(2);
  };

  const fmtDuration = (seconds) => {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "–";
    if (seconds < 90) return `${seconds.toFixed(0)} s`;
    if (seconds < 5400) return `${(seconds / 60).toFixed(1)} min`;
    return `${(seconds / 3600).toFixed(1)} h`;
  };

  /* ------------------------------------------------------------------ */
  /* toasts + button busy state                                          */
  /* ------------------------------------------------------------------ */

  function toast(title, body, kind = "ok") {
    const host = $("toast-host");
    if (!host) return;
    const node = document.createElement("div");
    node.className = `toast ${kind}`;
    node.innerHTML = `<div class="toast-title"></div><div class="toast-body"></div>`;
    node.querySelector(".toast-title").textContent = title;
    node.querySelector(".toast-body").textContent = body || "";
    host.appendChild(node);
    const ttl = TOAST_MS[kind] || 6000;
    setTimeout(() => node.remove(), ttl);
  }

  function setBusy(busy, label) {
    state.busy = busy;
    const button = $("btn-start");
    if (!button) return;
    button.classList.toggle("busy", busy);
    button.disabled = busy;
    const text = button.querySelector(".btn-label");
    if (text) text.textContent = label || (busy ? "Working…" : "Inject fault");
  }

  async function postJSON(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `${response.status} ${response.statusText}`);
    }
    return payload;
  }

  function resetHistories() {
    state.scoreHistory = {};
    state.tagHistory = {};
    state.levelHistory = [];
  }

  /* ------------------------------------------------------------------ */
  /* canvas helpers: every panel owns its axes                           */
  /* ------------------------------------------------------------------ */

  function prepareCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(120, canvas.clientWidth || 280);
    const height = canvas.clientHeight || canvas.height || 120;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "10px system-ui, sans-serif";
    return { ctx, width, height };
  }

  function drawPanel(canvas, options) {
    const { series = [], threshold = null, thresholdLabel = "threshold", unitLabel = "" } = options;
    const { ctx, width, height } = prepareCanvas(canvas);
    const usable = series.filter((s) => s.points.length > 1);
    const m = AXIS_MARGIN;

    if (!usable.length) {
      ctx.fillStyle = "#64748b";
      ctx.fillText("collecting samples…", m.l + 4, height / 2);
      return { min: null, max: null, current: null };
    }

    let minT = Infinity, maxT = -Infinity, minV = Infinity, maxV = -Infinity;
    usable.forEach((s) => s.points.forEach((p) => {
      if (p.t < minT) minT = p.t;
      if (p.t > maxT) maxT = p.t;
      if (p.v < minV) minV = p.v;
      if (p.v > maxV) maxV = p.v;
    }));
    if (threshold !== null && Number.isFinite(threshold)) {
      minV = Math.min(minV, threshold);
      maxV = Math.max(maxV, threshold);
    }
    const pad = (maxV - minV) * 0.08 || Math.max(1e-6, Math.abs(maxV) * 0.02) || 1;
    minV -= pad;
    maxV += pad;
    const spanT = Math.max(1, maxT - minT);
    const spanV = Math.max(1e-9, maxV - minV);
    const px = (t) => m.l + ((t - minT) / spanT) * (width - m.l - m.r);
    const py = (v) => height - m.b - ((v - minV) / spanV) * (height - m.t - m.b);

    // grid + y labels (this panel's own scale)
    ctx.strokeStyle = "#1b2740";
    ctx.fillStyle = "#7f8ea8";
    ctx.lineWidth = 1;
    [0, 0.5, 1].forEach((frac) => {
      const value = minV + frac * spanV;
      const y = py(value);
      ctx.beginPath();
      ctx.moveTo(m.l, y);
      ctx.lineTo(width - m.r, y);
      ctx.stroke();
      ctx.fillText(fmt(value), 4, y + 3);
    });
    ctx.fillText(`samples ${minT} → ${maxT}`, m.l + 2, height - 4);
    if (unitLabel) ctx.fillText(unitLabel, width - m.r - 46, height - 4);

    if (threshold !== null && Number.isFinite(threshold)) {
      const y = py(threshold);
      ctx.strokeStyle = COLOURS.ALERT;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(m.l, y);
      ctx.lineTo(width - m.r, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLOURS.ALERT;
      ctx.fillText(`${thresholdLabel} ${fmt(threshold)}`, m.l + 4, Math.max(10, y - 4));
    }

    usable.forEach((s, index) => {
      const colour = s.colour || SERIES_COLOURS[index % SERIES_COLOURS.length];
      ctx.strokeStyle = colour;
      ctx.lineWidth = s.width || 1.8;
      ctx.setLineDash(s.dashed ? [4, 4] : []);
      ctx.beginPath();
      s.points.forEach((p, i) => {
        const x = px(p.t), y = py(p.v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      const last = s.points[s.points.length - 1];
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(px(last.t), py(last.v), 2.2, 0, Math.PI * 2);
      ctx.fill();
    });

    const first = usable[0].points;
    return { min: minV, max: maxV, current: first[first.length - 1].v };
  }

  function drawPlantStrip() {
    const canvas = $("plant-strip");
    if (!canvas) return;
    const { ctx, width, height } = prepareCanvas(canvas);
    const cells = state.levelHistory.slice(-HISTORY);
    if (!cells.length) {
      ctx.fillStyle = "#64748b";
      ctx.fillText("collecting samples…", 8, height / 2);
      return;
    }
    const cellW = Math.max(1, (width - 4) / HISTORY);
    cells.forEach((cell, index) => {
      ctx.fillStyle = COLOURS[cell.level] || "#334155";
      ctx.globalAlpha = 0.85;
      ctx.fillRect(2 + index * cellW, 8, Math.max(1, cellW - 1), height - 22);
    });
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#7f8ea8";
    ctx.fillText(`oldest sample ${cells[0].t} → newest ${cells[cells.length - 1].t}`, 4, height - 4);
  }

  /* ------------------------------------------------------------------ */
  /* history tab                                                         */
  /* ------------------------------------------------------------------ */

  function drawScorePanels() {
    $$(".history-panel[data-part] .panel-canvas").forEach((canvas) => {
      const key = canvas.dataset.part;
      const series = [{ points: state.scoreHistory[key] || [], colour: SERIES_COLOURS[0] }];
      const bounds = drawPanel(canvas, {
        series,
        threshold: state.thresholds[key],
        thresholdLabel: "thr",
        unitLabel: "score",
      });
      const panel = canvas.closest(".history-panel");
      const info = state.lastLive && state.lastLive.parts[key];
      const level = info ? info.level : "NOMINAL";
      const pill = panel.querySelector('[data-role="level"]');
      if (pill) {
        pill.textContent = level;
        pill.className = `level-pill ${level}`;
      }
      const score = panel.querySelector('[data-role="score"]');
      if (score) score.textContent = info ? `score ${fmt(info.score)} / thr ${fmt(info.threshold)}` : "–";
      const axis = panel.querySelector('[data-role="axis"]');
      if (axis && bounds.min !== null) {
        axis.textContent = `axis ${fmt(bounds.min)} … ${fmt(bounds.max)}`;
      }
      panel.classList.toggle("is-selected", key === state.selected);
    });
  }

  function buildTagPanels() {
    const host = $("tag-panels");
    if (!host || !state.selected) return;
    const tags = state.partTags[state.selected] || [];
    host.innerHTML = tags
      .map(
        (tag, index) => `
        <div class="history-panel" data-tag="${tag}">
          <div class="panel-head">
            <span class="panel-title mono">${tag}</span>
            <span class="panel-meta" data-role="value">–</span>
          </div>
          <canvas class="tag-canvas" data-tag="${tag}" height="110"></canvas>
          <div class="panel-foot">
            <span class="mono" data-role="axis">–</span>
            <span class="mono" data-role="deviation"></span>
          </div>
        </div>`
      )
      .join("");
    const label = $("tag-panel-part");
    if (label && state.parts[state.selected]) label.textContent = state.parts[state.selected].name;
    const focus = $("focus-select");
    if (focus && focus.value !== state.selected) focus.value = state.selected;
  }

  function drawTagPanels() {
    $$("#tag-panels .tag-canvas").forEach((canvas, index) => {
      const tag = canvas.dataset.tag;
      const bounds = drawPanel(canvas, {
        series: [{ points: state.tagHistory[tag] || [], colour: SERIES_COLORS(index) }],
        unitLabel: "value",
      });
      const panel = canvas.closest(".history-panel");
      const info = state.lastLive && state.lastLive.parts[state.selected];
      const value = info && info.values ? info.values[tag] : null;
      const deviation = info && info.deviations ? info.deviations[tag] : null;
      const valueNode = panel.querySelector('[data-role="value"]');
      if (valueNode) {
        valueNode.textContent = `now ${fmt(value)}`;
        valueNode.style.color = deviation !== null && Math.abs(deviation) > 0.5 ? COLOURS.ALERT : "";
      }
      const axis = panel.querySelector('[data-role="axis"]');
      if (axis && bounds.min !== null) axis.textContent = `axis ${fmt(bounds.min)} … ${fmt(bounds.max)}`;
      const dev = panel.querySelector('[data-role="deviation"]');
      if (dev) dev.textContent = deviation === null || deviation === undefined
        ? ""
        : `${deviation >= 0 ? "+" : ""}${fmt(deviation)} IQR`;
    });
  }

  const SERIES_COLORS = (index) => SERIES_COLOURS[(index + 1) % SERIES_COLOURS.length];

  function drawHistory() {
    if (state.tab !== "history") return;
    drawPlantStrip();
    drawScorePanels();
    drawTagPanels();
  }

  /* ------------------------------------------------------------------ */
  /* live tab                                                            */
  /* ------------------------------------------------------------------ */

  function renderBanner(live) {
    const banner = $("banner");
    const level = live.situation.level;
    if (banner) {
      banner.dataset.level = level;
      const label = $("banner-level");
      if (label) {
        label.textContent = CFG.singlePart
          ? `${state.parts[state.selected]?.name || "part"} — ${level}`
          : `PLANT ${level}`;
      }
      const summary = $("banner-summary");
      if (summary) summary.textContent = live.situation.summary;
      const counts = live.situation.counts || {};
      const countsNode = $("banner-counts");
      if (countsNode) {
        countsNode.textContent =
          `nominal ${counts.NOMINAL || 0} · watch ${counts.WATCH || 0} · ` +
          `alert ${counts.ALERT || 0} · critical ${counts.CRITICAL || 0}`;
      }
      const mode = $("banner-mode");
      if (mode) mode.textContent = `mode: ${live.mode}`;
    }
    const stamp = $("live-timestamp");
    if (stamp) stamp.textContent = `${live.timestamp} · sample ${live.position}/${live.total_samples}`;
    const tabsHint = $("tabs-hint");
    if (tabsHint) tabsHint.textContent = `${live.situation.summary}`;
  }

  function renderDiagram(live) {
    $$("#boiler-svg .part").forEach((group) => {
      const key = group.dataset.part;
      const info = live.parts[key];
      if (!info) return;
      group.dataset.level = info.level;
      const injected = !!(live.labels && live.labels.fault_active) && live.scenario && live.scenario.part === key;
      group.classList.toggle("fault", injected);
      group.classList.toggle("selected", state.selected === key);
      const tags = state.partTags[key] || [];
      const primary = tags.length ? tags[0] : null;
      const valueNode = group.querySelector(".part-value");
      if (valueNode) {
        valueNode.textContent = primary && info.values ? fmt(info.values[primary]) : fmt(info.score);
      }
      group.setAttribute("aria-label", `${info.name}: ${info.level}, score ${fmt(info.score)}`);
      let title = group.querySelector("title");
      if (!title) {
        title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        group.appendChild(title);
      }
      title.textContent = tags
        .map((tag) => `${tag}: ${fmt((info.values || {})[tag])}`)
        .join("\n");
    });
  }

  function renderPartsTable(live) {
    const body = $("parts-body");
    if (!body) return;
    const order = { CRITICAL: 0, ALERT: 1, WATCH: 2, NOMINAL: 3 };
    const rows = Object.entries(live.parts).sort((a, b) => {
      const diff = order[a[1].level] - order[b[1].level];
      return diff !== 0 ? diff : (b[1].score || 0) - (a[1].score || 0);
    });
    body.innerHTML = rows
      .map(([key, info]) => `
        <tr data-part="${key}" class="${state.selected === key ? "is-selected" : ""}">
          <td>${info.name}</td>
          <td><span class="level-pill ${info.level}">${info.level}</span></td>
          <td class="mono">${fmt(info.score)} <span class="hint">/ ${fmt(info.threshold)}</span></td>
          <td class="trend ${info.trend}">${info.trend}</td>
          <td class="mono">${fmt(info.severity_iqr, 2)} IQR</td>
        </tr>`)
      .join("");
    $$("tr[data-part]", body).forEach((row) => {
      row.style.cursor = "pointer";
      row.addEventListener("click", () => selectPart(row.dataset.part));
    });
  }

  function renderScenario(live) {
    const card = $("scenario-card");
    if (!card) return;
    const name = $("scenario-name");
    const meta = $("scenario-meta");
    const truth = $("scenario-truth");
    const hint = $("scenario-hint");
    const fill = $("sev-fill");
    const scenario = live.scenario;
    const labels = live.labels || {};

    card.dataset.active = String(Boolean(scenario));
    if (!scenario) {
      name.textContent = "No fault injected";
      meta.textContent = "Pick a scenario and press Inject fault — the fault is held until you stop it.";
      truth.textContent = `mode: ${live.mode} · simulator idle`;
      if (fill) fill.style.width = "0%";
      hint.textContent = "nothing injected";
      const badge = $("truth-badge");
      if (badge) badge.classList.add("hidden");
      return;
    }

    const severity = typeof labels.fault_severity === "number" ? labels.fault_severity : 0;
    const detected = live.parts[scenario.part] ? live.parts[scenario.part].level : "NOMINAL";
    const detectedFlag = detected !== "NOMINAL";
    name.textContent = scenario.name;
    const lifetime = scenario.held === false
      ? `runs for ${fmtDuration(scenario.duration_s)}`
      : "held until you press Stop";
    meta.innerHTML =
      `ground truth: <b>${scenario.part_name}</b> · elapsed ${fmtDuration(scenario.elapsed_s)}` +
      ` · intensity ${fmt(scenario.intensity, 2)} · ${lifetime}`;
    truth.textContent =
      `fault ${scenario.fault_key} · severity ${fmt(severity, 2)} IQR · ` +
      `detector level on that part: ${detected}${detectedFlag ? " ✓" : ""}`;
    if (fill) fill.style.width = `${Math.min(100, severity * 50).toFixed(0)}%`;
    hint.textContent = labels.fault_active ? "fault active" : "fault ramping / finished";

    const badge = $("truth-badge");
    if (badge) {
      badge.classList.remove("hidden");
      badge.classList.toggle("active", detectedFlag);
      badge.textContent = `injected: ${scenario.part_name}${detectedFlag ? " · detected ✓" : " · not flagged"}`;
    }
  }

  function renderAgent(situation) {
    const summary = $("agent-summary");
    if (summary) {
      summary.textContent = situation.situation.summary;
      summary.dataset.level = situation.situation.level;
    }
    const list = $("agent-list");
    if (!list) return;
    const attention = situation.attention || [];
    list.innerHTML = attention.length
      ? attention
          .map((item) => `
            <li data-level="${item.level}">
              <div class="who">${item.name} · criticality ${item.criticality} · ${item.level}</div>
              <div>${item.explanation}</div>
            </li>`)
          .join("")
      : `<li><div class="who">all parts nominal</div>
         <div>No part is deviating beyond its normal spread.</div></li>`;
  }

  /* ------------------------------------------------------------------ */
  /* selection                                                           */
  /* ------------------------------------------------------------------ */

  function selectPart(key) {
    if (!key) return;
    state.selected = key;
    $$("#boiler-svg .part").forEach((group) => {
      group.classList.toggle("selected", group.dataset.part === key);
    });
    if (state.lastLive) renderPartsTable(state.lastLive);
    buildTagPanels();
    drawHistory();
  }

  /* ------------------------------------------------------------------ */
  /* controls                                                            */
  /* ------------------------------------------------------------------ */

  async function injectFault() {
    if (state.busy) return;
    const select = $("scenario-select");
    const key = select ? select.value : null;
    if (!key) return;
    const option = select.options[select.selectedIndex];
    const name = option ? option.textContent.trim() : key;
    setBusy(true, "Injecting…");
    try {
      await postJSON("/api/mode", { mode: "simulation" });
      await postJSON("/api/scenarios/start", { scenario_key: key, hold: true });
      const live = await refreshLive();
      resetHistories();
      if (live) {
        renderScenario(live);
        toast("Fault injected", `${name} → ground truth part: ${live.scenario ? live.scenario.part_name : "?"}. Held until you press Stop.`, "ok");
      } else {
        toast("Fault injected", name, "ok");
      }
      switchTab("live");
    } catch (error) {
      toast("Injection failed", String(error.message || error), "err");
    } finally {
      setBusy(false, "Inject fault");
    }
  }

  async function stopFault() {
    try {
      await postJSON("/api/scenarios/stop", {});
      const live = await refreshLive();
      if (live) renderScenario(live);
      toast("Fault stopped", "Playback continues in simulation mode.", "warn");
    } catch (error) {
      toast("Stop failed", String(error.message || error), "err");
    }
  }

  function wireControls() {
    const mode = $("mode-select");
    if (mode) {
      mode.addEventListener("change", async () => {
        try {
          await postJSON("/api/mode", { mode: mode.value });
          resetHistories();
          toast("Mode changed", `Now in ${mode.value} mode.`, "ok");
          await refreshLive();
        } catch (error) {
          toast("Mode change failed", String(error.message || error), "err");
        }
      });
    }
    const start = $("btn-start");
    if (start) start.addEventListener("click", injectFault);
    ["btn-stop", "btn-stop-2"].forEach((id) => {
      const button = $(id);
      if (button) button.addEventListener("click", stopFault);
    });
    const reset = $("btn-reset");
    if (reset) {
      reset.addEventListener("click", async () => {
        try {
          await postJSON("/api/reset", { start_index: 0 });
          resetHistories();
          toast("Rewound", "Playback restarted from the first sample.", "ok");
          await refreshLive();
        } catch (error) {
          toast("Rewind failed", String(error.message || error), "err");
        }
      });
    }
    const speed = $("speed-range");
    if (speed) {
      speed.addEventListener("input", () => {
        const label = $("speed-value");
        if (label) label.textContent = speed.value;
      });
      speed.addEventListener("change", async () => {
        try {
          await postJSON("/api/speed", { speed: Number(speed.value) });
        } catch (error) {
          toast("Speed change failed", String(error.message || error), "err");
        }
      });
    }
    const focus = $("focus-select");
    if (focus) focus.addEventListener("change", () => selectPart(focus.value));
    $$("#boiler-svg .part").forEach((group) => {
      group.addEventListener("click", () => selectPart(group.dataset.part));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") selectPart(group.dataset.part);
      });
    });
    $$(".tab").forEach((tab) => {
      tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    });
  }

  function switchTab(tab) {
    state.tab = tab;
    $$(".tab").forEach((node) => node.classList.toggle("active", node.dataset.tab === tab));
    $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${tab}`));
    if (tab === "history") buildTagPanels();
    drawHistory();
  }

  /* ------------------------------------------------------------------ */
  /* polling                                                             */
  /* ------------------------------------------------------------------ */

  function remember(live) {
    state.lastLive = live;
    state.levelHistory.push({ t: live.position, level: live.situation.level });
    if (state.levelHistory.length > HISTORY) state.levelHistory.shift();
    Object.entries(live.parts || {}).forEach(([key, info]) => {
      if (typeof info.score === "number" && Number.isFinite(info.score)) {
        const buffer = (state.scoreHistory[key] = state.scoreHistory[key] || []);
        buffer.push({ t: live.position, v: info.score });
        if (buffer.length > HISTORY) buffer.shift();
      }
      state.thresholds[key] = info.threshold;
      Object.entries(info.values || {}).forEach(([tag, value]) => {
        const buffer = (state.tagHistory[tag] = state.tagHistory[tag] || []);
        buffer.push({ t: live.position, v: value });
        if (buffer.length > HISTORY) buffer.shift();
      });
    });
    // default focus: the most severe part, until the user picks one
    if (!state.selected && CFG.partKeys && CFG.partKeys.length) selectPart(CFG.partKeys[0]);
  }

  async function refreshLive() {
    try {
      const live = await (await fetch("/api/live")).json();
      remember(live);
      renderBanner(live);
      renderDiagram(live);
      renderPartsTable(live);
      renderScenario(live);
      const mode = $("mode-select");
      if (mode && live.mode !== mode.value) mode.value = live.mode;
      drawHistory();
      return live;
    } catch (error) {
      console.warn("live poll failed", error);
      return null;
    }
  }

  async function refreshSituation() {
    try {
      const situation = await (await fetch("/api/situation")).json();
      renderAgent(situation);
    } catch (error) {
      console.warn("situation poll failed", error);
    }
  }

  async function boot() {
    try {
      const meta = await (await fetch("/api/scenarios")).json();
      state.parts = meta.parts || {};
      state.partTags = Object.fromEntries(
        Object.entries(state.parts).map(([key, info]) => [key, info.tags])
      );
      if (meta.state && meta.state.speed) {
        const speed = $("speed-range");
        if (speed) {
          speed.value = Math.min(Number(speed.max), Math.max(Number(speed.min), meta.state.speed));
          const label = $("speed-value");
          if (label) label.textContent = speed.value;
        }
      }
      if (!state.selected) state.selected = (CFG.partKeys && CFG.partKeys[0]) || Object.keys(state.parts)[0];
      buildTagPanels();
    } catch (error) {
      toast("Could not load scenario metadata", String(error.message || error), "err");
    }
    wireControls();
    const clock = $("clock");
    if (clock) setInterval(() => { clock.textContent = new Date().toLocaleTimeString(); }, 1000);
    await refreshLive();
    await refreshSituation();
    setInterval(refreshLive, POLL_LIVE_MS);
    setInterval(refreshSituation, POLL_SITUATION_MS);
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
