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

## Milestone 3 — Simulation Layer (Phase B) ⏳ not started

### 1. Scenario Controller
- [ ] Scenario API: start_scenario(name, ramp, duration), stop, list
- [ ] Real-time playback of data.csv samples at configurable rate
- [ ] Two demo modes: **livestream** (pure real-data replay) and
      **anomaly simulation** (real replay + injected fault scenarios); UI
      dropdown switches between them
- [ ] Per-sample "true fault part" labels for evaluation

### 2. Fault Injection (dynamical, FOPDT-style)
- [ ] **Fault propagation matrix:** every injected fault drives *all*
      physically-correlated sensors appropriately (per FOPDT coupling and the
      process cause-chain), not just the primary tag — e.g. IDF degradation
      shifts furnace pressure, flue-gas O₂, and economizer outlet temperature
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
- [x] **Decision:** hybrid ensemble strategy (user-approved): 2–3 candidates
      per part (TF autoencoder + LSTM-residual, plus Isolation Forest /
      One-Class SVM baselines), weighted fusion with percentile thresholds,
      fall back to single best model when one clearly dominates
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

## Milestone 5 — Overview Layer — Overview Agent (Phase D) ⏳ not started

### 1. Situation Levels
- [ ] Per-part situation: NOMINAL / WATCH / ALERT / CRITICAL from calibrated
      thresholds + trend
- [ ] Boiler-level situation = worst level + criticality weighting

### 2. Overview Strings & API
- [ ] Template-based explanation strings (situation, affected tags, values)
- [ ] /api/situation endpoint (JSON) for dashboard polling

**Checkpoint:** /api/situation produces coherent overviews (Overview Agent).

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
- [ ] Main demo page: boiler recreation with live sensor labels + scenario
      colours (NOMINAL green → WATCH amber → ALERT orange → CRITICAL red)
- [ ] **Live panel screen:** current sensor values, per-part situation levels,
      Overview Agent situation summary
- [ ] **Historical graphs:** per-tag time-series charts with anomaly flags /
      scenario event overlay
- [ ] **Mode dropdown/control:** switch between livestream mode (real replay)
      and real anomaly simulations (fault scenario injection)
- [ ] Per-part detail page (tags, sparklines, ensemble leaderboard,
      detection history)
- [ ] Polling every 1–2 s via fetch/JSON (basic server)

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
