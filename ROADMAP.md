# ROADMAP.md — The Bible

**Project:** Development of a Micro Language Model for Defence, Aerospace &
Public Enterprises (HAL, ISRO, DRDO) using TensorFlow:
BHEL-OptimizeTensorFlow Agents for Autonomous Boiler Lifecycle Monitoring in
Thermal Power Plants.

**Plan:** see `plan.md`. **Rules:** see `AGENT.md`. This file tracks all
checkpoints, tasks, and subtasks. Mark items `[x]` the moment they are done.
Commit at every major-task boundary with a ≤15-word message. Never delete
completed items.

Legend: `[x]` done · `[ ]` pending · `⚠ blocked:` reason · `⚠ deferred:` planned
but postponed.

---

## Milestone 1 — Foundation (skeleton, agent docs, research)

### 1. Skeleton Setup
- [x] Create project skeleton (git repo, AGENT.md operating manual)
- [x] Add workflow rule: never jump ahead, ask when confused, wait for user
- [x] Create .agents/ working-notes directory
- [x] Add .gitignore (data, venv, caches, model weights)

### 2. Research (done — see .agents/boiler.md, .agents/dataset.md)
- [x] Locate and analyse primary dataset (Kaggle industrial boiler operations —
      Nature Scientific Data long-tailed boiler dataset)
- [x] Verify anomaly labelling (global rule: TE_8332A outside 530–545 °C = anomaly, ≈8.6%)
- [x] Decide fault-injection recipe (IDF fan degradation via FOPDT dynamics —
      own synthetic faults, since original data has no labelled events)
- [x] Discover secondary dataset (NPPAD nuclear multi-accident time series)
- [x] Compile 30 sensor tags → boiler parts → failure modes map
- [x] Research boiler creeping problems & structural errors (fouling, erosion,
      creep, tube leaks, bearing wear, thermal fatigue, weld failures)

### 3. Planning
- [x] Write PLAN.md (basic demonstration plan + future additions)
- [x] Create ROADMAP.md (this file)
- [x] User review/approval of PLAN.md (approved with amendments:
      hybrid ensemble, Overview Agent rename, livestream/simulation dropdown,
      live panel + historical graphs, basic server)

---

## Milestone 2 — Data Layer (Phase A) ✅ complete

### 1. Project & Environment Setup
- [x] Initialize uv project (pyproject.toml, Python 3.13 via `uv`)
- [x] Add dependencies (tensorflow 2.21 **GPU build verified** (RTX 4050), NumPy,
      Pandas, Flask, scikit-learn, matplotlib, joblib, pytest) — `uv.lock` committed
- [x] Create src/tests/templates/static package layout (package `bhel_optimize`)
- [x] Set up pytest suite skeleton (43 tests passing)

### 2. Dataset Download
- [x] Dataset obtained — **user supplied `data/data.csv`, `data/data_AutoReg.csv`,
      `data/columns.csv`** (no Kaggle credentials / figshare needed)
- [x] Verified on shipped files: 86,400 × 30, 27 Mar–1 Apr 2022 (5 days exactly),
      every interval 5 s, raw has 30 NaNs in `YJJWSLL.AV_0`, imputed has 0

### 3. Load & Clean
- [x] Write loader for data.csv (30 tags; `#` suffix stripped; map by name, not
      position; descriptions in `config.TAG_INFO`)
- [x] Handle missing values (time-interpolation / linear / drop implemented and
      tested; AutoReg-imputed file used as canonical source)
- [x] Sanity checks (`sanity_report`: shape, missing per tag, time range,
      monotonicity, interval histogram)

### 4. Preprocess & Split
- [x] Per-tag normalization (train-fit `RobustScaler`/`StandardScaler` per tag,
      joblib persistence)
- [x] Chronological train/val/test split (70/15/15 = 60480/12960/12960, no shuffle)
- [x] Produce "normal" slice (TE_8332A within 530–545 °C) and "tailed" slice
      (measured tailed share 8.14%)
- [x] Resampling for playback/simulation rates (`resample`, 1 min → 7201 rows)

### 5. Part–Tag Map (single source of truth)
- [x] Encode 30 tags → 10 boiler parts in `src/bhel_optimize/config.py`
- [x] Encode part criticality weights (1–5; furnace/drum/main-steam = 5)
- [x] Unit tests for part map coverage (all 30 tags assigned, no orphans/dupes)

**Checkpoint:** data/ populated, part map in config, all tests pass. ✅

---

## Milestone 3 — Simulation Layer (Phase B) ✅ complete

### 1. Scenario Controller
- [x] Scenario API: `start_scenario(key, duration_s, intensity, target_tag)`,
      `start_fault`, `stop_scenario`, `current_fault`, `list_scenarios`
- [x] Real-time playback of data.csv samples at configurable rate
      (`speed` in samples/s, wall-clock `tick()` for UI + deterministic
      `advance()` for tests/offline, optional loop)
- [x] Two demo modes: **livestream** (pure real-data replay) and
      **anomaly simulation** (real replay + injected fault scenarios);
      UI dropdown switches via `set_mode` / `start_scenario`
- [x] Per-sample "true fault part" labels (`fault_part`, `fault_key`,
      `fault_severity`, `fault_active`, `tailed_real`, `tailed_observed`)

### 2. Fault Injection (dynamical, FOPDT-style)
- [x] **Fault propagation matrix:** every process fault drives 3+ correlated
      tags across 2+ parts (e.g. IDF degradation moves vibration, current,
      furnace pressures, flue-gas temps, O2, steam temp); gains are in
      multiples of each tag's IQR (units are unpublished)
- [x] Creeping faults: economizer fouling, IDF degradation, preheater fouling,
      hearth/layer ΔP drift (ramp dynamics)
- [x] Abrupt/structural faults: water-wall tube leak, superheater tube leak,
      IDF bearing failure, attemperator valve stuck/oscillation, furnace draft
      disturbance/air in-leak, sensor outlier spike / stuck (step, oscillate,
      spike, stuck dynamics)
- [x] Global anomaly injection: drives TE_8332A outside 530–545 °C ("tailed")
- [x] Scenario library documented and generated: `.agents/scenarios.md`
      (`python -m bhel_optimize.simulation.report`), 4 demo presets + 8 more

### 3. Tests
- [x] Unit tests for scenario controller (modes, playback, injection, expiry,
      labels, payloads)
- [x] Scenario outputs produce labelled time series (`run_series` → 30 tags +
      8 label columns, deterministic)

**Checkpoint:** scenarios run end-to-end and produce labelled data. ✅

---

## Milestone 4 — Detection Layer (Phase C) ✅ complete (accuracy caveat noted)

### 1. Per-Part Ensemble Models (TensorFlow/Keras)
- [x] LSTM prediction-residual model (`lstm_residual`)
- [x] Dense autoencoder reconstruction (`dense_ae`)
- [x] 1D-TCN autoencoder (`tcn_ae`, Opt-TCAE-style)
- [x] Transformer forecaster residual (`transformer_residual`) — implemented,
      **excluded from the default candidate set** as heavy (enable with
      `--include-heavy`); it never won on validation
- [x] Statistical/robust baselines: `ewma_mad`, `mean_shift` (control chart),
      `staleness` (stuck-sensor/variance collapse), Isolation Forest, One-Class SVM
- [x] First-difference feature channels added alongside tag levels (levels alone
      were weak; most fault signatures are rate-of-change phenomena)

### 2. Ensemble Fusion & Thresholds
- [x] Per-part weighted fusion (weights from validation AUROC; weak detectors
      dropped) — `weights_from_auroc`
- [x] Score calibration: clipped robust z against the normal-only validation
      distribution (percentile mapping was tried and rejected: it saturates at
      1.0 and destroys tail resolution)
- [x] Alarm smoothing (NaN-aware moving average, 12 samples = 1 min)
- [x] Selection = **best of {all detectors, fused ensemble} on validation**;
      fusion is only used when it actually wins (fusion was frequently worse
      than its best member)
- [x] Threshold: accuracy-optimal on validation scenarios but never more
      permissive than the false-alarm budget (default 5%)
- [x] Per-part leaderboard with AUROC + bootstrap CI, F1, precision, recall,
      FAR(clean), severity-banded recall, latency → `.agents/leaderboard.md`

### 3. Evaluation
- [x] Against synthetic per-part faults (per-part detection accuracy) — labels
      are **local per-part severity** in IQR units, pooled over 4 start offsets
- [x] Against the real 8.6% "tailed" class (global check on `main_steam`)
- [x] Leaderboard saved + documented (`.agents/leaderboard.md`, JSON at
      `artifacts/detection/leaderboard.json`)

**Checkpoint:** models saved, leaderboard produced, evaluation metrics reported ✅

**⚠ Accuracy caveat (documented in the report):** the dataset ships no per-part
fault labels, so per-part accuracy is measured against *injected simulated
faults*. Test AUROC ranges ≈0.31–0.85 by part; faults that reach ≥1 IQR of a
tag's spread are usually detected (recall up to 0.82–1.00), while injections
below ~0.5 IQR are largely missed — creeping faults are only detectable once
they exceed the local noise floor. Part-level numbers moved by >0.1 AUROC
between training configs, so they indicate rank rather than precise accuracy.

### 4. Deferred detection improvements ⚠ deferred
- [ ] Hyperparameter search per part (window length, model width, epochs)
- [ ] Richer features (spectral/rolling statistics, cross-tag correlations)
- [ ] More scenarios per part + longer evaluation series (tighter error bars)
- [ ] Physics-informed residuals (model the expected boiler response instead of
      purely data-driven reconstruction)

---

## Milestone 5 — Overview Layer — Overview Agent (Phase D) ✅ complete

### 1. Situation Levels
- [x] Per-part situation NOMINAL / WATCH / ALERT / CRITICAL from **measured
      clean-data bands** (watch = q95 of normal scores, alert = tuned threshold
      or q99.5 of normal — whichever is more conservative, critical = far tail)
- [x] Guards against alarm flapping: persistence gates (ALERT 3-of-5, WATCH
      6-of-10), one-step-per-update rate limiting, trend (rising/falling/steady)
- [x] Boiler-level situation = worst part level; **criticality drives emphasis**
      (sustained watch on a safety-critical part is called out in the summary)
      rather than inflating the level — inflating it made normal livestream data
      read PLANT ALERT
- [x] Probable-pattern matcher: compares observed tag deviations against the
      fault library's propagation maps, combining *coverage* of a fault's local
      signature with its *share* of the whole signature (so a single coupled tag
      cannot make another part's fault look local); distinguishes creep from
      abrupt faults by deviation magnitude
- [x] Warm-up safe: reports NOMINAL with `score: null` until a full window exists

### 2. Overview Strings & API
- [x] Template-based explanation strings per part (level, score vs threshold,
      trend, duration, most deviating tags with IQR deviation, matched pattern,
      recommended attention) + plant headline summary
- [x] `GET /api/situation` (JSON) via `create_overview_app` — includes plant
      level/counts/summary, per-part levels, trends, top tags, matched patterns,
      and the simulator's own labels (`fault_part`, `fault_active`) so the demo
      can show truth vs. detection
- [x] `GET /api/health` for liveness + part/threshold listing

**Checkpoint:** `/api/situation` produces coherent overviews ✅

**Notes / known limits (accepted for the demo):** the live monitor can still
show WATCH/ALERT during ordinary operation because the per-part detectors are
sensitive (up to ~5% of clean samples per part cross the tuned threshold); the
M4 leaderboard reports the tuned-threshold accuracy separately. Per-part
detection quality is the limiting factor, not the level logic.

---

## Milestone 6 — UI Layer (Phase E) ✅ complete

### 1. Boiler Recreation (SVG/CSS)
- [x] CFB schematic: furnace, fluidized bed/hearth, steam drum, superheater +
      attemperator, main steam line, economizer, air preheaters, PA/SA fans,
      ID fan, return-air chambers — plus unmonitored decoration (cyclone
      separator, refeeder, dust collector, stack, flue-gas/steam/feedwater paths)
- [x] Live sensor readouts on each part (`values[tag]` from the agent payload),
      situation colours (NOMINAL green → WATCH amber → ALERT orange → CRITICAL
      red), pulse animation on the injected part, hover tooltips with all tags

### 2. Flask App & API
- [x] Flask server, Jinja templates (`base`, `boiler`, `part`, `not_found`),
      static CSS + vanilla JS poller (no SPA framework)
- [x] `GET /api/live` — per-part level/score/threshold/trend/values/deviations
      plus plant status, mode, labels and scenario state (fast polling)
- [x] `GET /api/situation` — full Overview Agent payload
- [x] `GET /api/scenarios` — scenario list, part/tag metadata, current state
- [x] `POST /api/scenarios/start|stop`, `POST /api/mode`, `POST /api/speed`,
      `POST /api/reset`, `GET /api/health`
- [x] Faults started from the UI are **held until stopped** (`hold=True`) so a
      fast playback speed does not end the demo in two seconds

### 3. Demo Pages
- [x] Main demo page: boiler schematic + scenario colours, live panel (parts
      table sorted by severity), Overview Agent panel with explanations,
      truth-vs-detection chip
- [x] **Historical graphs:** per-part anomaly-score traces with the selected
      part's threshold line, plus live per-tag traces (canvas)
- [x] **Mode dropdown/control:** livestream ↔ anomaly simulation, scenario
      picker, inject/stop, speed slider, rewind
- [x] Per-part detail page: tags table with live values and IQR deviations,
      level bands, ensemble mode/weights
- [x] Polling every 1.2 s (live) and 4 s (situation) via fetch/JSON

**Checkpoint:** full demo runs end-to-end in browser ✅ (verified via API:
pages 200, /api/live updates, injected fault held and reflected in levels;
covered by 16 server tests)

### 4. Schematic — ✅ rebuilt from the user's reference design
- [x] Rebuilt the boiler schematic as a real left-to-right process flow with
      lanes (steam/water on top, combustion + flue gas in the middle, combustion
      air at the bottom) instead of overlapping boxes with stray ducts
- [x] Every part now carries label + live value *inside* its shape, so ducts and
      text cannot collide; flue path cycles furnace → cyclone separator →
      economizer → air preheater → dust collector → ID fan → stack
> **Adopted:** the user supplied `.agents/cfb_boiler.html` (CFB cutaway with
> circuit-based colours). `templates/_boiler_svg.html` is now that design —
> geometry, palette, typography and 1–8 numbering preserved — with live-value
> hooks and situation-level strokes added. The Mermaid source in
> `.agents/diagram.md` stays as process documentation.

- [x] `tests/test_diagram.py` (10 tests) guards the drawing: parses the SVG and
      asserts every part is drawn with a label + value, is keyboard reachable,
      ducts exist and show direction, no two part shapes overlap, no text lands
      inside another part, and no duct ends inside a part. Checks are
      **shape-wise** (parts may own disjoint pieces: return-air chambers, fans)
      with one documented exception (the bed is drawn inside the furnace shell)

### 4b. Schematic polish (user sign-off round)
- [x] Live value moved next to the part name inside each box (right-aligned on
      the name line, per-box font size; return-air readout sits just outside
      because that box is too narrow for both)
- [x] Box fill now follows NOMINAL / WATCH / ALERT / CRITICAL with a brighter
      fill plus pulsing outline for the injected fault; pipes keep their
      circuit colours (steam / feedwater / flue gas / combustion air / solids)
- [x] Legend removed from the SVG; the page legend now lists levels *and* pipe
      colours
- [x] `data-level="NOMINAL"` in the markup so boxes are tinted before the first
      poll (prevents a light-box/light-text flash in dark mode)
- [x] New tests: name+value must share a line and fit inside their box, no
      legend text in the SVG, and the CSS must colour boxes by level

### 5. UI feedback & history redesign (user feedback round)
- [x] **Fault injection is now visibly responsive:** the Inject button enters a
      busy state (spinner + disabled + "Injecting…"), a toast confirms the
      injected scenario and its ground-truth part, and the mode select follows
      the server's real mode on every poll
- [x] **Active-scenario panel:** scenario name, ground-truth part, simulated
      elapsed time, intensity, a severity bar (IQR units), whether the detector
      flagged that part, and a second Stop button
- [x] **Historical view is its own tab** (`data-tab="history"`), with the Live
      tab holding the schematic, controls and Overview Agent
- [x] **One panel per series, each with its own axes:** every part gets a
      server-rendered score panel (own min/max labels, grid, threshold line,
      sample range) instead of two shared canvases; sensor traces get one panel
      per tag with independent scales
- [x] **Plant level strip:** a colour-coded timeline of the plant situation
- [x] **Focus-part selector** in the history tab, synced with clicking the
      schematic or a table row
- [x] `/api/live` now also returns per-tag `deviations` for the tag panels
- [x] Tests: 4 new server tests (tabs, one panel per part, inject-feedback
      hooks, front-end wiring) — suite at 181 tests

### 6. Deferred UI items ⚠ deferred
- [ ] Dataset analytics/exploration page
- [ ] Browser-side visual regression test (no headless browser available here)

---

## Milestone 7 — Polish & Presentation (Phase F) ⏳ not started

- [ ] Analytics/exploration page (dataset plots) — optional
- [ ] Documentation (README, setup instructions)
- [ ] Presentation scenario script (creeping → WATCH, abrupt → CRITICAL)
- [ ] Final test suite run + demo walkthrough
- [ ] Git commit history clean at every milestone boundary

**Checkpoint:** college project demonstration ready.

---

## Milestone 8 — Future Additions (post-demo) ⚠ deferred

- [ ] Physics-informed modelling (PINN-style physics loss on steam temperature)
- [ ] Geospatial awareness (plant layout, adjacency-aware fusion)
- [ ] Advice/suggestions from micro-LM (TensorFlow micro-LM on playbook text)
- [ ] RAG over maintenance manuals (BHEL/ASME-style)
- [ ] Predictive maintenance / remaining-useful-life estimation
- [ ] NPPAD dataset as evaluation/transfer dataset
- [ ] Fleet/multi-boiler monitoring scale-out
- [ ] Lifecycle-aware advice (start-up/steady/shutdown supervision)
- [ ] Aerospace extension (HAL/ISRO/DRDO test-stand monitoring)
