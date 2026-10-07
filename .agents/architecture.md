# Onboarding Knowledge — System Architecture & Status

> Agent-facing architecture map. Keep this file updated as modules land.
> Read alongside `AGENT.md` (rules), `.agents/dataset.md` (data facts) and
> `.agents/boiler.md` (domain knowledge).

## Purpose

A micro-language-model / agent stack for **autonomous boiler lifecycle
monitoring**: it watches a boiler's sensor stream, detects anomalies **per
boiler part** with a hybrid ensemble, and feeds those results to an **Overview
Agent** that produces a clear situation summary — all shown in a Flask UI that
recreates the boiler with live values and scenario colours.

## Layered flow

```
            data/ (real DCS CSVs: data.csv, data_AutoReg.csv, columns.csv)
                              │
                    data loader + preprocessing
              (schema check, missing values, scalers, splits)
                              │
                 simulation layer  (M3 — DONE)
        playback modes + fault injection + propagation matrix
        livestream  |  anomaly-simulation  →  labelled samples
                              │
                 detection layer   (M4 — NEXT)
        per-part hybrid ensembles → calibrated scores, leaderboard
                              │
                 overview layer    (M5)
        situation levels NOMINAL/WATCH/ALERT/CRITICAL → summary text
                              │
                    Flask UI      (M6)
        boiler diagram + live panel + historical graphs + mode dropdown
```

## Module map (current)

| Path | Role | Status |
|---|---|---|
| `src/bhel_optimize/config.py` | Single source of truth: 30 tags, 10 parts, criticality, 530–545 °C rule, split fractions, paths | done |
| `src/bhel_optimize/data/loader.py` | Load raw/imputed CSVs, strip `#`, schema validation, `sanity_report` | done |
| `src/bhel_optimize/data/preprocess.py` | Missing values, per-tag scalers, chronological split, tailed slicing, resampling | done |
| `src/bhel_optimize/simulation/faults.py` | Fault library (FOPDT dynamics) + propagation matrix + scenario presets | done |
| `src/bhel_optimize/simulation/controller.py` | `ScenarioController`: modes, wall-clock playback, injection, labels, `run_series` | done |
| `src/bhel_optimize/simulation/report.py` | Generates `.agents/scenarios.md` from the fault library | done |
| `src/bhel_optimize/detection/windows.py` | Sliding windows, score scatter/sub-sampling, level+first-difference features | done |
| `src/bhel_optimize/detection/detectors.py` | 9 detectors: `ewma_mad`, `mean_shift`, `staleness`, `dense_ae`, `lstm_residual`, `tcn_ae`, `transformer_residual`, `isolation_forest`, `ocsvm` | done |
| `src/bhel_optimize/detection/ensembles.py` | Clipped-z normalisation, AUROC-weighted fusion, threshold calibration, `PartEnsemble` save/load | done |
| `src/bhel_optimize/detection/evaluate.py` | AUROC (+bootstrap CI), PR/F1, FAR, latency, severity-banded recall, alarm smoothing | done |
| `src/bhel_optimize/detection/data.py` | Labelled evaluation series from injected scenarios (per-part local severity) | done |
| `src/bhel_optimize/detection/train.py` | Trainer + CLI, leaderboard JSON + generated Markdown | done |
| `src/bhel_optimize/overview/situation.py` | Level bands (measured clean-data quantiles), persistence, trend, plant roll-up, fault-pattern matcher | done |
| `src/bhel_optimize/overview/agent.py` | `OverviewAgent`: scores every part on the live sample, tracks state, writes explanations | done |
| `src/bhel_optimize/overview/api.py` | Flask app: `/api/situation`, `/api/health` | done |
| `src/bhel_optimize/server/app.py` | Flask app: pages + `/api/live`, `/api/situation`, `/api/scenarios`, `/api/mode`, `/api/speed`, `/api/reset`, `/api/health` | done |
| `templates/` | `base.html`, `boiler.html` (main demo), `_boiler_svg.html` (schematic), `part.html`, `not_found.html` | done |
| `static/css/app.css`, `static/js/app.js` | Dark control-room theme; vanilla-JS poller, SVG renderer, canvas graphs, controls | done |

| `tests/` | 169 tests (config, loader, preprocess, faults, controller, report, detection, overview, server) | passing |

## Key decisions (with rationale)

- **Package `bhel_optimize`**, uv-managed, `uv.lock` committed.
- **TensorFlow 2.21 GPU** — verified `GPU:0` (RTX 4050) under nix-ld on NixOS.
- **Dataset:** shipped CSVs are the canonical source; `data_AutoReg.csv`
  (imputed) is the default. CSV tags carry a `#` suffix and column order
  differs from `columns.csv`, so **always map by name**.
- **Anomaly labels:** the dataset only labels a global "tailed" class via the
  steam-temperature rule (530–545 °C, measured 8.14% of samples). There are
  **no per-part fault labels**, so per-part ground truth comes from the
  simulation layer (`fault_part`, `fault_severity`, `fault_active`).
- **Detection strategy:** hybrid ensemble per part (TF autoencoder +
  LSTM-residual, plus Isolation Forest / One-Class SVM baselines), weighted
  fusion with percentile thresholds, falling back to a single best model when
  one dominates (user-approved).
- **Fault magnitudes are IQR-normalised** because the dataset publishes no
  engineering units — effects are "N x typical spread", not physical units.
- **Fault expiry is index-based**, so a fault started for 600 s stops affecting
  samples beyond that window even if the position is moved directly (UI polls).

## Simulation layer (M3) at a glance

- `ScenarioController(frame, mode=..., speed=..., start_index=..., loop=...)`
  - `mode`: `livestream` (pure real replay) or `simulation` (replay + fault)
  - `speed`: samples per second (1.0 default; 5.0 = real time)
  - `advance(n)` deterministic stepping (tests/offline),
    `tick(now)` wall-clock stepping (UI)
  - `start_scenario(key, duration_s=…, intensity=…, target_tag=…)`,
    `stop_scenario()`, `current_fault(index)`
  - `snapshot(index)` → values, baseline, deltas, affected tags, labels,
    scenario info (the UI payload)
  - `run_series(scenario_key, samples=…, start_index=…)` → DataFrame with all
    30 tags + label columns (`fault_key`, `fault_part`, `fault_category`,
    `fault_elapsed_s`, `fault_severity`, `fault_active`, `tailed_real`,
    `tailed_observed`) — the input for detection/evaluation
- 12 faults across 4 categories: 4 creeping, 5 abrupt, 2 sensor, 1 global;
  every process fault drives 3+ tags spanning 2+ parts
- Documented matrix: `.agents/scenarios.md` (generated)

## Detection layer (M4) at a glance

- Train: detectors are fitted on **normal-only** training windows (the 530–545 °C
  rule is applied *before* scaling — applying it to z-scored data silently
  yields zero training windows).
- Features: each part's windows carry tag **levels + first differences**
  (`part_features`); levels alone separated poorly.
- Scores: per-detector **clipped robust z** against the normal-validation
  distribution (NOT percentile mapping — that saturates at 1.0 and destroys the
  tail where detections live).
- Fusion: weights ∝ (validation AUROC − 0.5), weak detectors dropped; the
  **fused ensemble competes with individual detectors** and is only selected
  when it wins on validation (fusion was often worse than its best member).
- Threshold: accuracy-optimal on validation scenarios, capped by the
  false-alarm budget (default 5%) via the clean-sample score quantile.
- Alarms are **smoothed** (12-sample NaN-aware moving average = 1 min).
- Evaluation: injected scenarios pooled over 4 start offsets; positives =
  samples with ≥0.1 IQR *local* injection; recall reported per severity band;
  real "tailed" rows and <0.1 IQR injections excluded.
- Run: `uv run python -m bhel_optimize.detection.train` (≈4 min, all 10 parts)
  → `artifacts/detection/` + `.agents/leaderboard.md`.

### Accuracy reality check (do not overstate)

- There are **no per-part fault labels** in the dataset, so per-part accuracy is
  measured against our own injected faults (user-approved).
- Test AUROC ≈0.31–0.85 by part; ≥1 IQR faults are usually caught, <0.5 IQR
  injections mostly are not. Part ranks are more trustworthy than absolute
  values (config changes moved some parts by >0.1 AUROC).
- Two bugs worth remembering: window-local MAD denominators must be floored at
  a fraction of a per-tag reference spread (otherwise flat stretches produce
  1e7 scores on *normal* data), and severity banding must use each part's own
  local injection rather than the fault's global severity.

## Overview layer (M5) at a glance

- `OverviewAgent.from_artifacts()` loads the fitted ensembles, the persisted
  scalers and the per-tag baselines, then measures each part's **clean-data
  score distribution** to set level bands (~25 s startup for 10 parts).
- Levels: NOMINAL < q95 of normal scores → WATCH → ALERT (tuned threshold, or
  q99.5 of normal if that is stricter) → CRITICAL (far tail). Persistence
  (WATCH 6-of-10, ALERT 3-of-5) and a one-step-per-update rate limit stop
  flapping; a sustained watch on a criticality-4/5 part is *reported* but does
  not inflate the plant level.
- Explanations quote measurable facts (score vs threshold, trend, duration,
  tag deviations in IQR units) and name the best-matching fault pattern from the
  propagation matrix. Phase 2 replaces these templates with the micro-LM.
- `GET /api/situation` returns plant level + counts + summary, per-part details,
  attention list, matched patterns, and the simulator's ground-truth labels.

## UI layer (M6) at a glance

- Run: `uv run python -m bhel_optimize.server` (options `--speed`, `--port`,
  `--host`). First start builds the Overview Agent (~25 s: load ensembles +
  measure clean-data level bands).
- Pages: `/` main demo (schematic + live panel + graphs + agent panel),
  `/part/<key>` detail, `/api/*` JSON APIs.
- The JS poller keeps its own ring buffers (300 samples) for the graphs, so no
  server-side history endpoint is needed; clicking a part switches the traces.
- Faults started from the UI pass `hold=True` (active until Stop) — at 40×
  playback a 600 s preset fault would otherwise finish in ~2 s of wall clock.
- Scenario/preset metadata for the dropdown comes from the same generated fault
  library (`/api/scenarios`), so UI and simulation cannot drift.
- Two tabs: **Live** (schematic, controls, Active-scenario panel, Overview
  Agent) and **Historical view** (plant level strip + one score panel per part +
  one panel per tag of the focused part). Every panel draws its **own axes**
  (min/max labels, grid, threshold line, sample range) — sharing canvases caused
  two series with different units to be plotted on one scale.
- Fault injection feedback: busy button (spinner/disabled), toast with the
  injected scenario + ground-truth part, and an Active-scenario panel showing
  elapsed simulated time, intensity, severity in IQR units and whether the
  detector flagged the injected part. `injectFault()` always sends
  `hold: true`, so a fault stays active until Stop regardless of playback speed.
- **The diagram is parked** pending the user's Mermaid version — see
  `.agents/diagram.md` for the canonical Mermaid source, the part-key ↔ node ↔
  tag mapping and the hooks the UI needs. The stopgap SVG description below is
  kept only so the existing wiring stays understood.
- The stopgap schematic (`templates/_boiler_svg.html`) is a hand-drawn SVG in three
  lanes: steam/water (top), combustion + flue gas (middle: furnace → cyclone
  separator → economizer → air preheater → dust collector → ID fan → stack),
  combustion air (bottom). Labels and live values sit *inside* their shapes so
  ducts cannot cross text. `tests/test_diagram.py` parses the SVG and enforces
  the layout invariants (no shape overlaps, no stray text, ducts all have
  direction and terminate outside parts) — exceedance there means the picture is
  wrong, not just a failing assertion.

## JSON transport rule (learned the hard way)

- **Never let a non-finite float reach the wire.** A fault held until stopped has
  `duration_s = inf`; Python's `json` emits bare `Infinity`, which is *invalid
  JSON* — browsers fail with "JSON.parse: unexpected character" while
  `json.loads` (and Flask's test client) happily accept it, so tests can pass
  while the UI is completely broken. Symptoms looked like a UI bug ("live poll
  failed") but were a backend serialisation bug.
- Fix: `bhel_optimize.jsonio.json_safe()` maps non-finite floats to `null` and is
  applied to every payload (`live_payload`, `situation_payload`,
  `scenario_state`, scenarios, health); both app factories set
  `app.json.allow_nan = False` so any future leak raises in tests instead of
  breaking a browser. Held faults are reported as `"held": true` with
  `duration_s: null`.
- Regression test: `tests/test_server.py::test_payloads_never_emit_infinity_or_nan`
  fetches the **raw bytes** and parses them with a strict `parse_constant` that
  rejects `Infinity`/`NaN` — `get_json()` alone would not catch this class of bug.

## Environment pitfall found (GPU training)

- TensorFlow 2.21 needs `libdevice.10.bc` for XLA GPU kernel compilation. The
  generic wheel does not ship it, so GPU training failed with
  `INTERNAL: JIT compilation failed` / `libdevice not found` even though
  `tf.config.list_physical_devices("GPU")` succeeded and plain matmuls worked.
- Fix: `uv add nvidia-cuda-nvcc-cu12` (TF's loader already searches
  `site-packages/nvidia/cuda_nvcc/nvvm/libdevice`). nix-ld is enabled, which is
  what lets these pip CUDA libraries load on NixOS.
- `jit_compile=False` does **not** avoid it (TF still autotunes GPU kernels).

## Verified facts / pitfalls

- raw `data.csv`: **30** NaNs, all `YJJWSLL.AV_0`; imputed file: 0 NaNs
- exactly 86,400 rows, 5 s apart, 2022-03-27 14:28:54 → 2022-04-01 14:28:49
- root `/data/` and `/artifacts/` are git-ignored (anchor them in .gitignore;
  an unanchored `data/` also hides the Python package `bhel_optimize/data/`)
- `tests/conftest.py` provides `requires_data` so the suite skips gracefully
  when the (git-ignored) dataset is absent
