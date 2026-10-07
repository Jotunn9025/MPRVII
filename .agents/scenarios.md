# Onboarding Knowledge — Scenario & Fault Library

> **Generated** by `uv run python -m bhel_optimize.simulation.report` — do not hand-edit; change `simulation/faults.py` instead.

Gains are **multiples of each tag's IQR** measured from the real data (engineering units are not published), so `0.9` means "0.9 x typical spread of that tag".

## Demo scenarios (UI dropdown)

| Key | Name | Category | Part | Duration | Description |
|---|---|---|---|---|---|
| `demo_idf_degradation` | DEMO — IDF degradation (creeping) | creeping | Induced Draft Fan | 2700 s | Slow fan degradation: vibration/current creep, gentle draft loss. |
| `demo_sensor_spike` | DEMO — sensor outlier spike | sensor | Economizer (Upper) | 60 s | Single-tag spike on an economizer gas-temperature sensor. |
| `demo_tailed_steam_temp` | DEMO — steam temperature out of band | global | Main Steam Line & Outlet | 900 s | Reproduces the dataset's 'tailed' anomaly definition. |
| `demo_water_wall_leak` | DEMO — water-wall tube leak (abrupt) | abrupt | Furnace Chamber (Upper) | 600 s | Fast leak: drum pressure, steam temperature and steam flow fall. |

## Fault library and propagation matrix

### `economizer_fouling` — Economizer fouling

- **Category:** creeping
- **Primary part:** Economizer (Upper) (criticality 3)
- **Default duration:** 1800 s
- **Description:** Ash/soot deposits on economizer surfaces reduce heat pickup.
- **Physics:** Less heat extracted in the economizer -> flue-gas outlet temperature rises, O2 trims down, steam temperature sags, draft resistance loads the ID fan.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `TE_8319A.AV_0` | Flue gas temperature at the upper economizer outlet (left) | +1.00 | ramp | 0 s | 60 s | primary |
| `TE_8319B.AV_0` | Flue gas temperature at the upper economizer outlet (right) | +1.00 | ramp | 0 s | 60 s | primary |
| `AIR_8301A.AV_0` | Upper economiser inlet flue gas oxygen (left) | -0.35 | ramp | 0 s | 60 s | primary |
| `AIR_8301B.AV_0` | Upper economiser inlet flue gas oxygen (right) | -0.35 | ramp | 0 s | 60 s | primary |
| `TE_8332A.AV_0` | Boiler outlet steam temperature | -0.35 | ramp | 0 s | 60 s | primary |
| `YFJ3_AI.AV_0` | Induced draft fan motor current | +0.25 | ramp | 0 s | 60 s | primary |

### `id_fan_degradation` — Induced draft fan degradation

- **Category:** creeping
- **Primary part:** Induced Draft Fan (criticality 4)
- **Default duration:** 2700 s
- **Description:** Progressive bearing wear / impeller erosion reduces fan efficiency.
- **Physics:** Bearing vibration and motor current creep up; weaker draft makes furnace pressures less negative and leaves flue gas hotter; O2 drifts.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `YFJ3_ZD1.AV_0` | Vibration of induced draft fan bearing shell (A) | +0.90 | ramp | 0 s | 60 s | bearing shell A |
| `YFJ3_ZD2.AV_0` | Vibration of induced draft fan bearing shell (B) | +0.80 | ramp | 0 s | 60 s | bearing shell B |
| `YFJ3_AI.AV_0` | Induced draft fan motor current | +0.60 | ramp | 0 s | 60 s | current rises as efficiency drops |
| `PT_8313A.AV_0` | Upper furnace pressure (A) | -0.35 | ramp | 30 s | 60 s | primary |
| `PT_8313B.AV_0` | Upper furnace pressure (B) | -0.35 | ramp | 30 s | 60 s | primary |
| `PT_8313C.AV_0` | Upper furnace pressure (C) | -0.35 | ramp | 30 s | 60 s | primary |
| `PT_8313D.AV_0` | Upper furnace pressure (D) | -0.35 | ramp | 30 s | 60 s | primary |
| `PT_8313E.AV_0` | Upper furnace pressure (E) | -0.35 | ramp | 30 s | 60 s | primary |
| `PT_8313F.AV_0` | Upper furnace pressure (F) | -0.35 | ramp | 30 s | 60 s | primary |
| `TE_8319A.AV_0` | Flue gas temperature at the upper economizer outlet (left) | +0.30 | ramp | 30 s | 60 s | primary |
| `TE_8319B.AV_0` | Flue gas temperature at the upper economizer outlet (right) | +0.30 | ramp | 30 s | 60 s | primary |
| `AIR_8301A.AV_0` | Upper economiser inlet flue gas oxygen (left) | -0.20 | ramp | 30 s | 60 s | primary |
| `AIR_8301B.AV_0` | Upper economiser inlet flue gas oxygen (right) | -0.20 | ramp | 30 s | 60 s | primary |
| `TE_8332A.AV_0` | Boiler outlet steam temperature | -0.20 | ramp | 30 s | 60 s | primary |

### `air_preheater_fouling` — Air preheater fouling

- **Category:** creeping
- **Primary part:** Air Preheaters (Primary & Secondary) (criticality 3)
- **Default duration:** 1800 s
- **Description:** Deposits on preheater elements cut combustion-air heating.
- **Physics:** Preheated air temperatures fall, flue gas leaves hotter, furnace temperature shifts and O2 trims drift.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `TE_8303.AV_0` | Primary air preheater outlet air temperature | -0.80 | ramp | 0 s | 60 s | primary |
| `TE_8304.AV_0` | Secondary air preheater outlet air temperature | -0.80 | ramp | 0 s | 60 s | primary |
| `TE_8319A.AV_0` | Flue gas temperature at the upper economizer outlet (left) | +0.35 | ramp | 30 s | 60 s | primary |
| `TE_8319B.AV_0` | Flue gas temperature at the upper economizer outlet (right) | +0.35 | ramp | 30 s | 60 s | primary |
| `AIR_8301A.AV_0` | Upper economiser inlet flue gas oxygen (left) | -0.25 | ramp | 0 s | 60 s | primary |
| `AIR_8301B.AV_0` | Upper economiser inlet flue gas oxygen (right) | -0.25 | ramp | 0 s | 60 s | primary |
| `TE_8313B.AV_0` | Temperature in the upper part of the furnace chamber (right) | +0.20 | ramp | 30 s | 60 s | primary |

### `bed_dp_drift` — Fluidized-bed circulation blockage

- **Category:** creeping
- **Primary part:** Fluidized Bed (Hearth & Layers) (criticality 4)
- **Default duration:** 2700 s
- **Description:** Bed material circulation degrades; hearth/layer differential pressures drift.
- **Physics:** Hearth and layer delta-P rise as the bed densifies/blockages build; return-air flow falls, furnace pressure and temperature shift.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `SXLTCYZ.AV_0` | Differential pressure between upper and lower hearth (left) | +0.70 | ramp | 0 s | 60 s | primary |
| `SXLTCYY.AV_0` | Differential pressure between upper and lower hearth (right) | +0.70 | ramp | 0 s | 60 s | primary |
| `ZCLCCY.AV_0` | Differential pressure in the left layer | +0.80 | ramp | 0 s | 60 s | primary |
| `YCLCCY.AV_0` | Differential pressure in the right layer | +0.80 | ramp | 0 s | 60 s | primary |
| `FT_8306A.AV_0` | Return air chamber air flow (left) | -0.40 | ramp | 30 s | 60 s | primary |
| `FT_8306B.AV_0` | Return air chamber air flow (right) | -0.40 | ramp | 30 s | 60 s | primary |
| `PT_8313A.AV_0` | Upper furnace pressure (A) | -0.30 | ramp | 30 s | 60 s | primary |
| `PT_8313B.AV_0` | Upper furnace pressure (B) | -0.30 | ramp | 30 s | 60 s | primary |
| `PT_8313C.AV_0` | Upper furnace pressure (C) | -0.30 | ramp | 30 s | 60 s | primary |
| `PT_8313D.AV_0` | Upper furnace pressure (D) | -0.30 | ramp | 30 s | 60 s | primary |
| `PT_8313E.AV_0` | Upper furnace pressure (E) | -0.30 | ramp | 30 s | 60 s | primary |
| `PT_8313F.AV_0` | Upper furnace pressure (F) | -0.30 | ramp | 30 s | 60 s | primary |
| `TE_8313B.AV_0` | Temperature in the upper part of the furnace chamber (right) | +0.25 | ramp | 30 s | 60 s | primary |

### `water_wall_tube_leak` — Furnace water-wall tube leak

- **Category:** abrupt
- **Primary part:** Furnace Chamber (Upper) (criticality 5)
- **Default duration:** 600 s
- **Description:** Tube rupture in the furnace water wall (the most common boiler failure).
- **Physics:** Furnace pressure is disturbed, drum pressure collapses, steam temperature and main steam flow drop together, bed delta-P reacts.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `PT_8313A.AV_0` | Upper furnace pressure (A) | +0.60 | step | 10 s | 45 s | primary |
| `PT_8313B.AV_0` | Upper furnace pressure (B) | +0.60 | step | 10 s | 45 s | primary |
| `PT_8313C.AV_0` | Upper furnace pressure (C) | +0.60 | step | 10 s | 45 s | primary |
| `PT_8313D.AV_0` | Upper furnace pressure (D) | +0.60 | step | 10 s | 45 s | primary |
| `PT_8313E.AV_0` | Upper furnace pressure (E) | +0.60 | step | 10 s | 45 s | primary |
| `PT_8313F.AV_0` | Upper furnace pressure (F) | +0.60 | step | 10 s | 45 s | primary |
| `PTCA_8322A.AV_0` | Pot pressure (left) | -0.80 | step | 10 s | 45 s | primary |
| `PTCA_8324.AV_0` | Container outlet vapour pressure | -0.70 | step | 10 s | 45 s | primary |
| `TE_8332A.AV_0` | Boiler outlet steam temperature | -0.90 | step | 15 s | 60 s | primary |
| `ZZQBCHLL.AV_0` | Main steam flow rate after compensation | -0.50 | step | 15 s | 60 s | primary |
| `SXLTCYZ.AV_0` | Differential pressure between upper and lower hearth (left) | -0.35 | step | 20 s | 60 s | primary |
| `SXLTCYY.AV_0` | Differential pressure between upper and lower hearth (right) | -0.35 | step | 20 s | 60 s | primary |

### `superheater_tube_leak` — Superheater tube leak

- **Category:** abrupt
- **Primary part:** Superheater & Desuperheater (criticality 4)
- **Default duration:** 480 s
- **Description:** Superheater tube rupture losing high-grade steam.
- **Physics:** Steam temperature falls sharply, attemperator spray and valve react, main steam flow and drum pressure read low/uneven (leak flow).

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `TE_8332A.AV_0` | Boiler outlet steam temperature | -1.20 | step | 10 s | 45 s | primary |
| `YJJWSLL.AV_0` | Primary desuperheating water flow output | +0.50 | step | 15 s | 60 s | spray compensates |
| `TV_8329ZC.AV_0` | Primary desuperheater outlet steam temperature regulating valve | +0.60 | step | 15 s | 60 s | primary |
| `ZZQBCHLL.AV_0` | Main steam flow rate after compensation | +0.40 | step | 20 s | 60 s | leak flow |
| `PTCA_8324.AV_0` | Container outlet vapour pressure | -0.50 | step | 15 s | 60 s | primary |

### `id_fan_bearing_failure` — Induced draft fan bearing failure

- **Category:** abrupt
- **Primary part:** Induced Draft Fan (criticality 4)
- **Default duration:** 420 s
- **Description:** Sudden bearing shell failure on the induced draft fan.
- **Physics:** Vibration spikes on both shells, motor current surges, draft weakens (furnace pressure, flue-gas temperature, O2 all shift).

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `YFJ3_ZD1.AV_0` | Vibration of induced draft fan bearing shell (A) | +2.00 | step | 5 s | 20 s | primary |
| `YFJ3_ZD2.AV_0` | Vibration of induced draft fan bearing shell (B) | +1.60 | step | 5 s | 20 s | primary |
| `YFJ3_AI.AV_0` | Induced draft fan motor current | +1.00 | step | 5 s | 20 s | primary |
| `PT_8313A.AV_0` | Upper furnace pressure (A) | -0.50 | step | 15 s | 45 s | primary |
| `PT_8313B.AV_0` | Upper furnace pressure (B) | -0.50 | step | 15 s | 45 s | primary |
| `PT_8313C.AV_0` | Upper furnace pressure (C) | -0.50 | step | 15 s | 45 s | primary |
| `PT_8313D.AV_0` | Upper furnace pressure (D) | -0.50 | step | 15 s | 45 s | primary |
| `PT_8313E.AV_0` | Upper furnace pressure (E) | -0.50 | step | 15 s | 45 s | primary |
| `PT_8313F.AV_0` | Upper furnace pressure (F) | -0.50 | step | 15 s | 45 s | primary |
| `TE_8319A.AV_0` | Flue gas temperature at the upper economizer outlet (left) | +0.35 | step | 15 s | 45 s | primary |
| `TE_8319B.AV_0` | Flue gas temperature at the upper economizer outlet (right) | +0.35 | step | 15 s | 45 s | primary |

### `attemperator_valve_stuck` — Attemperator valve stuck / oscillating

- **Category:** abrupt
- **Primary part:** Superheater & Desuperheater (criticality 4)
- **Default duration:** 600 s
- **Description:** Desuperheater regulating valve jams, breaking temperature control.
- **Physics:** Spray-water flow freezes while steam temperature oscillates around setpoint; drum/steam flow pick up the control fight.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `TV_8329ZC.AV_0` | Primary desuperheater outlet steam temperature regulating valve | +0.00 | stuck | 5 s | 60 s |  |
| `YJJWSLL.AV_0` | Primary desuperheating water flow output | -0.50 | step | 5 s | 30 s | primary |
| `TE_8332A.AV_0` | Boiler outlet steam temperature | +0.60 | oscillate | 10 s | 60 s | primary |
| `ZZQBCHLL.AV_0` | Main steam flow rate after compensation | +0.20 | oscillate | 10 s | 60 s | primary |

### `furnace_draft_disturbance` — Furnace draft disturbance / air in-leak

- **Category:** abrupt
- **Primary part:** Furnace Chamber (Upper) (criticality 5)
- **Default duration:** 600 s
- **Description:** Draft upset with cold-air in-leakage into the furnace/flue path.
- **Physics:** Furnace pressures swing, flue-gas O2 rises from in-leakage, furnace temperature and ID fan current react.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `PT_8313A.AV_0` | Upper furnace pressure (A) | +0.90 | step | 5 s | 30 s | primary |
| `PT_8313B.AV_0` | Upper furnace pressure (B) | +0.90 | step | 5 s | 30 s | primary |
| `PT_8313C.AV_0` | Upper furnace pressure (C) | +0.90 | step | 5 s | 30 s | primary |
| `PT_8313D.AV_0` | Upper furnace pressure (D) | +0.90 | step | 5 s | 30 s | primary |
| `PT_8313E.AV_0` | Upper furnace pressure (E) | +0.90 | step | 5 s | 30 s | primary |
| `PT_8313F.AV_0` | Upper furnace pressure (F) | +0.90 | step | 5 s | 30 s | primary |
| `AIR_8301A.AV_0` | Upper economiser inlet flue gas oxygen (left) | +0.60 | step | 20 s | 60 s | primary |
| `AIR_8301B.AV_0` | Upper economiser inlet flue gas oxygen (right) | +0.60 | step | 20 s | 60 s | primary |
| `TE_8313B.AV_0` | Temperature in the upper part of the furnace chamber (right) | -0.35 | step | 20 s | 60 s | primary |
| `YFJ3_AI.AV_0` | Induced draft fan motor current | +0.35 | step | 10 s | 45 s | primary |
| `TE_8319A.AV_0` | Flue gas temperature at the upper economizer outlet (left) | -0.20 | step | 20 s | 60 s | primary |
| `TE_8319B.AV_0` | Flue gas temperature at the upper economizer outlet (right) | -0.20 | step | 20 s | 60 s | primary |

### `sensor_outlier_spike` — Sensor outlier spike

- **Category:** sensor
- **Primary part:** Economizer (Upper) (criticality 3)
- **Default duration:** 60 s
- **Description:** Short spurious spike on one sensor (single-tag fault).
- **Default target tag:** `TE_8319A.AV_0` (overridable per run)
- **Physics:** Instrument/EMI fault — must be distinguishable from a process fault because only the target tag moves.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `TE_8319A.AV_0` | Flue gas temperature at the upper economizer outlet (left) | +3.00 | spike | 0 s | 60 s | primary |

### `sensor_stuck` — Sensor stuck-at value

- **Category:** sensor
- **Primary part:** Furnace Chamber (Upper) (criticality 5)
- **Default duration:** 600 s
- **Description:** Sensor freezes at its last reading (single-tag fault).
- **Default target tag:** `PT_8313A.AV_0` (overridable per run)
- **Physics:** Instrument failure — the tag stops tracking the process.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `PT_8313A.AV_0` | Upper furnace pressure (A) | +0.00 | stuck | 0 s | 60 s |  |

### `steam_temp_anomaly` — Steam temperature out of band (tailed data)

- **Category:** global
- **Primary part:** Main Steam Line & Outlet (criticality 5)
- **Default duration:** 900 s
- **Description:** Drives the state-discriminating tag outside 530-545 degC, reproducing the dataset's own 'tailed' anomaly definition.
- **Physics:** Reproduces the long-tailed anomaly class shipped with the dataset.

| Tag | Sensor | Gain (IQR) | Dynamics | Dead time | tau | Note |
|---|---|---|---|---|---|---|
| `TE_8332A.AV_0` | Boiler outlet steam temperature | -2.50 | step | 30 s | 180 s | primary |
| `ZZQBCHLL.AV_0` | Main steam flow rate after compensation | -0.40 | step | 30 s | 180 s | primary |

## Scenario presets (all)

| Key | Name | Category | Duration | Demo |
|---|---|---|---|---|
| `demo_idf_degradation` | DEMO — IDF degradation (creeping) | creeping | 2700 s | yes |
| `demo_sensor_spike` | DEMO — sensor outlier spike | sensor | 60 s | yes |
| `demo_tailed_steam_temp` | DEMO — steam temperature out of band | global | 900 s | yes |
| `demo_water_wall_leak` | DEMO — water-wall tube leak (abrupt) | abrupt | 600 s | yes |
| `air_preheater_fouling` | Air preheater fouling | creeping | 1800 s |  |
| `attemperator_valve_stuck` | Attemperator valve stuck / oscillating | abrupt | 600 s |  |
| `economizer_fouling` | Economizer fouling | creeping | 1800 s |  |
| `bed_dp_drift` | Fluidized-bed circulation blockage | creeping | 2700 s |  |
| `furnace_draft_disturbance` | Furnace draft disturbance / air in-leak | abrupt | 600 s |  |
| `id_fan_bearing_failure` | Induced draft fan bearing failure | abrupt | 420 s |  |
| `sensor_stuck` | Sensor stuck-at value | sensor | 600 s |  |
| `superheater_tube_leak` | Superheater tube leak | abrupt | 480 s |  |

