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
- [ ] User review/approval of PLAN.md

---

## Milestone 2 — Data Layer (Phase A) ⏳ not started

### 1. Project & Environment Setup
- [ ] Initialize uv project (pyproject.toml, Python 3)
- [ ] Add dependencies (TensorFlow, NumPy, Pandas, Flask, pytest, kagglehub)
- [ ] Create src/tests/templates/static package layout
- [ ] Set up pytest suite skeleton

### 2. Dataset Download
- [ ] Download Kaggle data.csv via kagglehub (needs Kaggle credentials — ask user)
- [ ] Fallback: figshare original dataset (public, no key) if Kaggle fails
- [ ] Verify download: shape 86,400 × 30, date range 27 Mar–1 Apr 2022

### 3. Load & Clean
- [ ] Write loader for data.csv (30 tags with descriptions in config)
- [ ] Handle missing values (interpolation fallback; AutoReg already applied)
- [ ] Sanity checks: value ranges, monotonic timestamps/indices

### 4. Preprocess & Split
- [ ] Per-tag normalization (train-fit scalers only)
- [ ] Chronological train/val/test split (time-series aware, no shuffle)
- [ ] Produce "normal" slice (TE_8332A within 530–545 °C) and "tailed" test slice
- [ ] Resampling for playback/simulation rates

### 5. Part–Tag Map (single source of truth)
- [ ] Encode 30 tags → boiler parts in src/config.py
- [ ] Encode part criticality weights (safety-critical parts higher)
- [ ] Unit tests for part map coverage (all 30 tags assigned, no orphans)

**Checkpoint:** data/ populated, part map in config, all tests pass.

---

## Milestone 3 — Simulation Layer (Phase B) ⏳ not started

### 1. Scenario Controller
- [ ] Scenario API: start_scenario(name, ramp, duration), stop, list
- [ ] Real-time playback of data.csv samples at configurable rate
- [ ] Per-sample "true fault part" labels for evaluation

### 2. Fault Injection (dynamical, FOPDT-style)
- [ ] Creeping faults: economizer fouling, IDF degradation, preheater fouling,
      hearth/layer ΔP drift
- [ ] Abrupt/structural faults: water-wall tube leak, superheater tube leak,
      IDF bearing failure, attemperator valve stuck/oscillation, furnace draft
      disturbance/air in-leak, outlier sensor fault (spike/stuck)
- [ ] Global anomaly injection: drive TE_8332A outside 530–545 °C ("tailed" samples)
- [ ] Scenario library with documented parameters per scenario

### 3. Tests
- [ ] Unit tests for scenario controller
- [ ] Scenario outputs produce labelled time series

**Checkpoint:** scenarios run end-to-end and produce labelled data.

---

## Milestone 4 — Detection Layer (Phase C) ⏳ not started

### 1. Per-Part Ensemble Models (TensorFlow/Keras)
- [ ] LSTM prediction-residual model (drift + fast faults)
- [ ] Dense autoencoder reconstruction (correlated multivariate window)
- [ ] 1D-TCN autoencoder (Opt-TCAE-style, tube-leak candidate)
- [ ] Transformer forecaster residual (only where it wins)
- [ ] Statistical/robust baselines: EWMA, MAD, Isolation Forest, One-Class SVM

### 2. Ensemble Fusion & Thresholds
- [ ] Per-part weighted voting/averaging (weights learned on validation)
- [ ] Score calibration: percentile thresholds on normal val slice
- [ ] Select best-per-model when a single model dominates per part
- [ ] Per-part leaderboard (AUROC, F1, latency, false-alarm rate)

### 3. Evaluation
- [ ] Against real 8.6% "tailed" anomalies (global detection accuracy)
- [ ] Against synthetic per-part faults (per-part detection accuracy)
- [ ] Leaderboard saved + documented

**Checkpoint:** models saved, leaderboard produced, evaluation metrics reported.

---

## Milestone 5 — Assimilation Layer (Phase D) ⏳ not started

### 1. Situation Levels
- [ ] Per-part situation: NOMINAL / WATCH / ALERT / CRITICAL from calibrated
      thresholds + trend
- [ ] Boiler-level situation = worst level + criticality weighting

### 2. Overview Strings & API
- [ ] Template-based explanation strings (situation, affected tags, values)
- [ ] /api/situation endpoint (JSON) for dashboard polling

**Checkpoint:** /api/situation produces coherent overviews.

---

## Milestone 6 — UI Layer (Phase E) ⏳ not started

### 1. Boiler Recreation (SVG/CSS)
- [ ] Draw CFB boiler: furnace, cyclone, refeeder, PA/SA/ID fans, air
      preheaters, economizer, drum, superheater, desuperheater, main steam line
- [ ] Live sensor readouts on each part (values, situation colour, trend arrow)

### 2. Flask App & API
- [ ] Flask server, Jinja templates, static css/vanilla js
- [ ] /api/live endpoint (sensor values + per-part situation levels)
- [ ] /api/scenarios endpoint (list/start/stop simulation scenarios)

### 3. Demo Pages
- [ ] Boiler recreation page (main demo page) with scenario sidebar +
      situation-overview panel
- [ ] Per-part detail page (tags, sparklines, ensemble leaderboard,
      detection history)
- [ ] Polling every 1–2 s via fetch/JSON

**Checkpoint:** full demo runs end-to-end in browser.

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
