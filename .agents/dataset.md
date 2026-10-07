# Onboarding Knowledge — Primary Dataset

> Agent-facing dataset facts. Read this before touching data code.
> Source paper: Cao, Z. et al. "A long-tailed distribution time-series dataset
> in boiler equipment." *Scientific Data* 12 (2025).
> https://www.nature.com/articles/s41597-025-05096-4

## Chosen dataset (primary)

- **Kaggle:** https://www.kaggle.com/datasets/nikitamanaenkov/time-series-of-industrial-boiler-operations
  (file: `data.csv`, by Nikita Manaenkov, published 2025-05-06)
- **Original full archive (imputed + unimputed CSVs + AutoReg.py):**
  figshare https://doi.org/10.6084/m9.figshare.28868849
  - `xinan_completed_data.csv` (AutoReg-imputed) / `xinan_uncompleted_data.csv`

## Facts

- Coal-fired **circulating fluidized bed (CFB)** boiler in a **chemical plant,
  Zhejiang, China**.
- Collected **27 Mar 2022 – 1 Apr 2022** (≈6 days), **5-second intervals**,
  **86,400 samples**, **30 sensor variables** via DCS real-time database.
- Data quality: 2 missing values in `YJJWSLL.AV_0`, filled by AutoReg in the
  completed version. Otherwise clean.

## Anomaly labelling (important)

- The **boiler outlet steam temperature** (`TE_8332A.AV_0`) is the
  state-discriminating parameter.
  - 530 °C ≤ T ≤ 545 °C → **normal** ("headed data")
  - outside that range → **anomaly** ("tailed data")
  - anomalies ≈ **8.6%** of samples → **long-tailed distribution**.
- Anomalies ARE labelled, but by a **single global rule on one tag** — not
  per-boiler-part fault labels. Per-part anomalies must come from simulation
  (dynamical fault injection) or synthetic generation for the demo.

## Full 30-variable list (Kaggle `data.csv`)

| # | Tag | Description |
|---|-----|-------------|
| 1 | PT_8313A.AV_0 | Upper furnace pressure (A) |
| 2 | PT_8313B.AV_0 | Upper furnace pressure (B) |
| 3 | PT_8313C.AV_0 | Upper furnace pressure (C) |
| 4 | PT_8313D.AV_0 | Upper furnace pressure (D) |
| 5 | PT_8313E.AV_0 | Upper furnace pressure (E) |
| 6 | PT_8313F.AV_0 | Upper furnace pressure (F) |
| 7 | PTCA_8324.AV_0 | Container outlet vapour pressure (drum outlet / main steam) |
| 8 | PTCA_8322A.AV_0 | Pot pressure (left) — steam drum pressure (left) |
| 9 | TE_8319A.AV_0 | Flue gas temperature at upper economizer outlet (left) |
| 10 | TE_8319B.AV_0 | Flue gas temperature at upper economizer outlet (right) |
| 11 | TE_8313B.AV_0 | Temperature in the upper part of the furnace chamber (right) |
| 12 | TE_8303.AV_0 | Primary air preheater outlet air temperature |
| 13 | TE_8304.AV_0 | Secondary air preheater outlet air temperature |
| 14 | TE_8332A.AV_0 | Boiler outlet steam temperature (state-discriminating tag) |
| 15 | TV_8329ZC.AV_0 | Primary desuperheater outlet steam temperature regulating valve |
| 16 | FT_8301.AV_0 | Primary fan outlet flow rate |
| 17 | FT_8302.AV_0 | Secondary fan outlet flow |
| 18 | FT_8306A.AV_0 | Return air chamber air flow (left) |
| 19 | AIR_8301A.AV_0 | Upper economiser inlet flue gas oxygen (left) |
| 20 | FT_8306B.AV_0 | Return air chamber air flow (right) |
| 21 | AIR_8301B.AV_0 | Upper economiser inlet flue gas oxygen (right) |
| 22 | YFJ3_AI.AV_0 | Induced draft fan motor current |
| 23 | YFJ3_ZD1.AV_0 | Vibration of induced draft fan bearing shell (A) |
| 24 | YFJ3_ZD2.AV_0 | Vibration of induced draft fan bearing shell (B) |
| 25 | SXLTCYZ.AV_0 | Differential pressure between upper and lower hearth (left) |
| 26 | SXLTCYY.AV_0 | Differential pressure between upper and lower hearth (right) |
| 27 | ZCLCCY.AV_0 | Differential pressure in the left layer |
| 28 | YCLCCY.AV_0 | Differential pressure in the right layer |
| 29 | YJJWSLL.AV_0 | Primary desuperheating water flow output |
| 30 | ZZQBCHLL.AV_0 | Main steam flow rate after compensation |

## Fault injection recipe (own synthetic faults)

> The original Kaggle data contains **no labelled fault events** — we will
> inject our own faults.

- Primary injected fault: **Induced Draft Fan (IDF) degradation** — gradual fan
  efficiency reduction visible in motor current (YFJ3_AI), bearing vibration
  (YFJ3_ZD1/ZD2), and flow/pressure impacts. Physically plausible.
- Also inject: outlier anomalies, creeping/structural error modes, failures.
- Boiler dynamics modelled with **FOPDT** (First-Order Plus Dead-Time) — a
  simple, good recipe for fault injection.

## Related dataset (future alternative/addition)

- **NPPAD:** Qi, B. et al. "An open time-series simulated dataset covering
  various accidents for nuclear power plants." *Sci. Data* 9 (2022).
  https://doi.org/10.1038/s41597-022-01879-1 — simulated multi-sensor time
  series covering various nuclear plant accidents (labelled accidents,
  5-second; analogous monitoring problem for later evaluation/transfer demos).
- **Mendeley 300 MW CFBB dataset** (5760 obs, 26 attrs, 3 prediction labels) —
  power-plant CFBB alternative.

## Figshare extras

- Trained networks available: 7 Transformer forecasting variants + 3 generation
  models (Table 6 of the paper). Best forecasters: Fedformer (96/192),
  Pathformer (336/720). Paper uses model-weighted integration across
  prediction lengths.
