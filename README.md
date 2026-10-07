# BHEL-OptimizeTensorFlow — Boiler Lifecycle Monitoring Demo

Autonomous boiler monitoring demo for the MPRVII project: a **per-part hybrid
ensemble** anomaly-detection stack over real industrial boiler data, feeding an
**Overview Agent** that explains the situation, shown in a Flask UI that
recreates the CFB boiler with live sensor values.

## Quick start

```bash
uv sync                                     # Python 3.13 + TensorFlow 2.21 (GPU)
uv run python -m bhel_optimize.server       # http://127.0.0.1:5000
```

The UI loads the trained ensembles from `artifacts/detection/` (first start also
measures each part's clean-data level bands, ~25 s). Options:

```bash
uv run python -m bhel_optimize.server --speed 40 --port 8000
```

## Demo script (for the presentation)

1. Open <http://127.0.0.1:5000> — the furnace/drum/ID-fan are drawn with live
   values; the banner shows **PLANT NOMINAL/WATCH** and the Overview Agent panel
   explains any part that is deviating.
2. Set **Mode → anomaly simulation**, pick a scenario (the ★ presets are the
   curated demo ones), press **Inject fault**. The affected part pulses, its
   colour moves NOMINAL → WATCH → ALERT, the truth chip shows which fault was
   injected, and the agent names the **most consistent fault pattern**.
3. Press **Stop fault** to clear it. The score history and per-tag traces keep
   drawing on the two canvases; click any part (diagram or table row) to switch
   the traces to that part.
4. Switch **Mode → livestream** to show honest behaviour on unmodified real
   data, and open a **part detail** page (click a part then the link, or
   `/part/id_fan`) for bands, ensemble weights and per-tag deviations.

## Overview Agent narration (optional Groq)

The situation summary is short plain English. Without an API key it is built
from templates; with a key the agent asks a small hosted model instead (the raw
numbers stay in the API payload for the panels):

```bash
export GROQ_API_KEY=gsk_...            # enables narration
export BHEL_LLM_MODEL=llama-3.1-8b-instant   # default; any Groq chat model
export BHEL_LLM=off                    # force templates even with a key
```

Safety: one call per situation at most every 5 s, identical situations are
cached, the request times out after 6 s, and **any** failure (missing key,
timeout, bad response) silently falls back to the template text — the panel
shows whether the text came from the model (`written by Groq · <model>`) or the
templates. `python -m bhel_optimize.overview.llm` is not needed; tests stub the
HTTP call, so the suite never touches the network.

## Retraining the detectors

```bash
uv run python -m bhel_optimize.detection.train                 # ~4 min, all 10 parts
uv run python -m bhel_optimize.detection.train --include-heavy  # + transformer forecaster
```

Writes `artifacts/detection/` (models, scalers, leaderboard JSON) and the
generated report `.agents/leaderboard.md`.

## Docs map

| File | Contents |
|---|---|
| `plan.md` | approved Phase-1 plan + future phases (micro-LM, physics, geospatial) |
| `ROADMAP.md` | the task bible (M1–M8) with status |
| `AGENT.md` | workflow rules for coding agents |
| `.agents/boiler.md` | boiler anatomy, monitored parts, failure modes |
| `.agents/dataset.md` | dataset facts, all 30 tags, labelling rule |
| `.agents/scenarios.md` | generated fault library + propagation matrix |
| `.agents/leaderboard.md` | generated per-part detection accuracy |
| `.agents/architecture.md` | module map, decisions, pitfalls |

## Honest limitations

- The dataset has **no per-part fault labels**, so per-part accuracy is measured
  against *injected simulated faults* (user-approved). Test AUROC ≈ 0.3–0.85 by
  part; faults reaching ≥ 1 IQR of a tag's spread are usually caught, smaller
  ones often are not.
- The live monitor can show WATCH on ordinary data, because per-part detectors
  are sensitive and the ALERT band must also respect a clean-data false-alarm
  limit.
- Demo project only — not for real plant use.
