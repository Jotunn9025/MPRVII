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
| `src/bhel_optimize/detection/` | Per-part hybrid ensembles, thresholds, leaderboard | not started (M4) |
| `src/bhel_optimize/overview/` | Overview Agent: situation levels + summaries + `/api/situation` payload | not started (M5) |
| `src/bhel_optimize/server/` | Flask app, `/api/live`, `/api/scenarios` | not started (M6) |
| `templates/`, `static/` | Jinja templates, CSS, vanilla JS | placeholders only |
| `tests/` | 90 tests (config, loader, preprocess, faults, controller) | passing |

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

## Verified facts / pitfalls

- raw `data.csv`: **30** NaNs, all `YJJWSLL.AV_0`; imputed file: 0 NaNs
- exactly 86,400 rows, 5 s apart, 2022-03-27 14:28:54 → 2022-04-01 14:28:49
- root `/data/` and `/artifacts/` are git-ignored (anchor them in .gitignore;
  an unanchored `data/` also hides the Python package `bhel_optimize/data/`)
- `tests/conftest.py` provides `requires_data` so the suite skips gracefully
  when the (git-ignored) dataset is absent
