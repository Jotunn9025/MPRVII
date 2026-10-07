# plan.md — Project Plan (Phase 1 Demo + Future Extensions)

**Status:** plan only — nothing implemented yet.
**Approved by:** user (this plan drafted from user's instructions + dataset research).
**Rule:** AGENT.md governs workflow. ROADMAP.md (created after this plan is
reviewed) will turn the Phase 1 section into checkpoints/tasks/subtasks.

---

## 1. Understanding the Boiler (from the dataset's source paper)

The dataset comes from a real **coal-fired circulating fluidized bed (CFB)
boiler** at a chemical plant in Zhejiang, China. How it works (paper's process
description):

- **Fuel path:** coal bunker → crusher → coal feeder → combustion chamber
  (furnace). High air velocity carries solid particles out; a **cyclone
  separator** returns them via a **refeeder** to sustain combustion.
- **Flue-gas path:** separator releases hot flue gas → tail flue → convection
  heat surface → dust collector → chimney.
- **Steam path ("pot"):** feed water → economizer → steam drum → drop tube →
  water-cooled wall (vaporization) → back to steam drum → steam-water
  separation → superheater → main steam pipeline to the user.
- Sensor network is integrated with the plant's **Distributed Control System
  (DCS)**; temperature, pressure, and flow sensors at key process points
  (furnace, steam drum, superheater, flue-gas duct, feedwater system).

## 2. Dataset: what it is and how we use it

**Primary dataset (Kaggle, confirmed):**
`https://www.kaggle.com/datasets/nikitamanaenkov/time-series-of-industrial-boiler-operations`
(`data.csv`), mirrored on Figshare (DOI 10.6084/m9.figshare.28868849), described
in Nature Scientific Data (s41597-025-05096-4).

- 86,400 samples, 27 Mar–1 Apr 2022, **5-second intervals**, **30 variables**
  (upper furnace pressure, container outlet vapour pressure, pot pressure,
  flue-gas temperature at upper economizer outlet, primary fan outlet flow,
  boiler outlet steam temperature, desuperheating water flow `YJJWSLL.AV_0`,
  etc.). Exact 30 variable names live in the paper's Table 1 → **locate and map
  every column to a boiler part after downloading data.csv.**
- **Anomalies are labelled, but only globally:** boiler outlet steam
  temperature in **530–545 °C = normal** ("headed", 91.4%); outside that range
  = **anomaly** ("tailed", **8.6%** of samples). Long-tailed distribution.
- **No per-part fault-type labels** exist in the data.
- Missing values were imputed with AutoReg; dataset ships as
  `xinan_completed_data.csv` / `xinan_uncompleted_data.csv` (we use the
  completed version; verify actual CSV name inside the Kaggle download).
- Paper baseline: 7 Transformer variants for prediction + **weighted ensemble
  integration** (best-fit result) — supports our ensemble approach.

**Training usage decision:**
- Train **unsupervised anomaly-detection models on normal ("headed") data
  only**, then flag deviations.
- Use the **530–545 °C rule** as ground truth for evaluation of the global
  normal/anomaly label.
- Derive **per-part anomaly ground truth** by (a) rule-based process logic
  (see §3) and (b) **simulated fault injections** into the data stream for demo
  and evaluation purposes.

**Secondary datasets (optional / future):**
- NASA C-MAPSS turbofan degradation dataset (creep/structural degradation
  domain, fully labelled) — possible transfer-learning or demo fallback.
- SECOM, Tennessee Eastman process — generic industrial backup.

## 3. Boiler parts monitored, sensors, and susceptibility to creeping/structural problems

| Boiler part | Sensors (dataset variables to map) | Why it creeps / structural risks |
|---|---|---|
| Furnace (combustion chamber) | upper furnace pressure, flue-gas O₂, temperatures | furnace pressure instability, combustion imbalance, refractory wear, ash fouling |
| Steam drum | container outlet vapour pressure, pot pressure, drum level | drum level swell/shrink, carryover, fatigue cracking at nozzles, corrosion |
| Water-cooled wall + drop tube | wall temps, circulation flows | tube leakage (most common boiler failure), scale build-up, erosion by particles |
| Superheater | boiler outlet steam temperature, desuperheating water flow | creep rupture from over-temperature, fouling, spray attemperation faults |
| Economizer | flue-gas temperature at upper economizer outlet | low-temperature corrosion, ash fouling, blockage |
| Cyclone separator + refeeder | fan flows, separator temperatures, return flows | erosion from high-velocity solids, blockage, return-flow instability |
| Coal feeder / bunker / crusher | feeder speeds, flow rates | feeder blockage, coal quality variation, crusher wear |
| Primary fan / air system | primary fan outlet flow rate | fan degradation, duct leakage, air-flow imbalance |
| Main steam pipeline | pipeline pressure/temperature | creep, water hammer, weld failures |
| Dust collector / tail flue | outlet temps, dust readings | bag wear, pressure-drop creep, blockage |

*(Table to be finalized with the exact 30 variable names from data.csv +
paper Table 1; parts not covered by the dataset's variables will be simulated.)*

## 4. Phase 1 — Basic Demonstration (approved scope)

**Goal:** a working end-to-end demo: simulated live boiler with per-part
anomaly detection, an assimilation agent that gives clear situation overviews,
and a Flask UI that recreates the boiler diagram with live sensor labels.

### 4.1 Data layer
1. Download data.csv (Kaggle/Figshare mirror). Map all 30 columns → boiler
   parts. EDA notebook/script: distributions, missing values, the headed/tailed
   split, per-part normal ranges.
2. Build a **data simulator**: replay real data as a live 5-second stream
   (normal mode) and inject **simulated anomalies** (drift/creep ramps, sudden
   step faults, oscillation, sensor dropout) per part for demo and validation.

### 4.2 Anomaly detection per part (ensemble for best accuracy)
- One or more detectors per part, trained on normal data only:
  - TensorFlow-based: dense autoencoder, LSTM/temporal autoencoder,
    (optionally) small TCN or one-class network.
  - Non-TF baselines for the ensemble: Isolation Forest, One-Class SVM,
    statistical thresholds on residuals.
- **Ensemble fusion:** combine detector scores via weighted fusion / voting /
    stacking on the validation stream (paper itself found weighted ensembles
    best). Optimize thresholds for precision/recall vs. the simulated faults.
- Evaluate per part: precision, recall, F1, detection latency, false-alarm
  rate on normal stream.

### 4.3 Assimilation / Organization agent
- Ingests per-part ensemble results + sensor values.
- Produces a **clear structured overview**: plant status (NORMAL / ALERT /
  CRITICAL), which parts are anomalous, severity, duration/creep trajectory
  (rising vs. sudden), and a concise natural-language situation summary.
- Phase 1: rule/template-driven summarizer (the "micro-LM stub"). Phase 1 demo
  does **not** need a trained LM — that is explicitly a future phase.

### 4.4 Flask UI (basic demonstration)
- Recreates the boiler (CFB layout: furnace, drum, water wall, superheater,
  economizer, separator, fans, feeder, pipeline) as a diagram with **current
  sensor values labelled on each part** and per-part status colors.
- Panels: live time-series sparklines per part, ensemble score gauges,
  anomaly flags, simulated fault-injection controls (to trigger faults in the
  demo), and the assimilation agent's situation overview page.
- Basic Flask server, Jinja templates, minimal vanilla JS (polling for live
  updates). No SPA frameworks.

### 4.5 Deliverables for the demo
- `data.csv` ingested + column-to-part mapping documented.
- Per-part ensemble anomaly-detection models (TensorFlow + baselines) with
  evaluation results.
- Data simulator with normal + injected-anomaly modes.
- Assimilation agent producing structured overviews + summaries.
- Flask UI: boiler diagram with live sensor labels + overview page.
- Tests for data, models (basic), agent, and UI pages.

## 5. Future phases (explicitly out of Phase 1 demo)

1. **Micro-LM (TensorFlow):** train a small language model on boiler/ops
   corpora to power the assimilation agent's *suggestions and advice*
   (maintenance recommendations, next-step guidance) — the "micro language
   model" part of the topic.
2. **Physics-informed awareness:** inject thermodynamic/flow constraints
   (energy balance, pressure-temperature relations, creep-life models) so the
   agent's overviews respect process physics and can reason about cause-chains.
3. **Geospatial awareness:** spatial model of part locations (furnace vs. drum
   vs. flue path) so the agent understands propagation paths (e.g., upstream
   fault affecting downstream parts) and gives location-informed overviews.
4. **Advice/case-based reasoning:** maintenance playbooks, case library of
   past incidents, prioritized action lists.
5. Optional: real CFB process simulation for richer physics-informed data,
   scaling to a multi-boiler plant view, alerting/notification UI.

## 6. Open questions to resolve before/at implementation start

1. Confirm the actual CSV filename(s) and 30 variable names after downloading
   the dataset; finalize the part↔column mapping table (§3).
2. Confirm simulated anomaly scenarios wanted in the demo (which parts, which
   fault types, how visible).
3. Confirm UI style/level (single-page diagram + overview, or multiple pages).
4. Confirm whether secondary datasets (NASA C-MAPSS, Tennessee Eastman) are in
   scope for Phase 1 or deferred.
5. Python/uv project setup details (package name, TensorFlow version pinning).
