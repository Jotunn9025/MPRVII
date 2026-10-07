# Onboarding Knowledge — Industrial Boiler Anatomy & Failure Modes

> Agent-facing domain knowledge. Read this before touching boiler-specific code.
> Source research trail: original session notes (Nature Sci. Data paper,
> ASME/BHEL monitoring literature, Opt-TCAE paper). Facts only — no plans here.

## Boiler type in scope

**Coal-fired circulating fluidized bed (CFB) boiler** in a chemical plant
(Zhejiang, China) — matches our primary dataset. CFB identifiers present in
data: cyclone separator, refeeder, hearth/layer differential pressures, return
air chambers. Conceptual mapping to a generic BHEL-style thermal power plant
boiler is a 1:1 part correspondence (furnace, drum, banks, fans, preheaters).

## Main parts and their monitored parameters

| Part | Monitored parameters (dataset tags) | Sensor types |
|------|-------------------------------------|--------------|
| Furnace (upper chamber) | 6 furnace pressures A–F (PT_8313A–F), chamber temperature (TE_8313B) | pressure, thermocouple |
| Steam drum / pot | Drum pressure left (PTCA_8322A), container outlet vapour pressure (PTCA_8324 = main steam) | pressure |
| Steam outlet / main steam line | Boiler outlet steam temperature (TE_8332A) — **state-discriminating tag**, main steam flow (ZZQBCHLL) | thermocouple, flow |
| Superheater + primary desuperheater (attemperator) | Desuperheater outlet temp regulating valve (TV_8329ZC), desuperheating water flow (YJJWSLL) | valve position, flow |
| Economizer (upper) | Flue gas temp outlet left/right (TE_8319A/B), flue gas O₂ inlet left/right (AIR_8301A/B) | thermocouple, O₂ analyzer |
| Air preheaters (primary + secondary) | Preheater outlet air temps (TE_8303, TE_8304) | thermocouple |
| Combustion air fans (primary FT_8301, secondary FT_8302) | Fan outlet flows | flow |
| Induced draft fan (IDF, YFJ3) | Motor current (YFJ3_AI), bearing vibrations A/B (YFJ3_ZD1/ZD2) | current, vibration |
| Return air chambers | Air flows left/right (FT_8306A/B) | flow |
| Fluidized bed (hearth/layers) | ΔP upper–lower hearth left/right (SXLTCYZ/YY), ΔP left/right layers (ZCLCCY/YCLCCY) | differential pressure |

**30 sensor variables total**, 5-second sampling — full tag table lives in
`dataset.md`.

## Creeping problems (gradual, developing over hours–weeks)

- **Fouling / slagging:** economizer, air preheater, furnace walls → flue-gas
  temp creep, ΔP drift, efficiency loss. (Creep fault candidates: economizer
  fouling, preheater fouling, hearth/layer ΔP drift.)
- **Creep (metallurgical):** sustained tube overtemperature → tube bulging,
  eventual burst. Water-walls, superheaters, reheaters most susceptible.
- **IDF/PA/SA fan degradation:** gradual bearing wear → slow vibration creep,
  motor current rise, flow rate loss.
- **Erosion:** fly-ash erosion of tubes, preheater elements (backfire erosion).
- **Corrosion:** water-side (caustic gouging), fire-side (sulphur/chlorine in
  coal) → thinning over months.
- **Air in-leak:** slow furnace draft drift, O₂ reading anomalies.

## Abrupt / structural errors (sudden faults)

- **Tube leakage / burst:** water-wall, superheater, economizer → sudden ΔP
  shifts, drum level instability, steam flow anomaly. #1 cause of forced
  outages in coal plants (>30% per ASME literature).
- **Thermal fatigue:** rapid cycling → weld cracks (welds are structural weak
  points).
- **Weld failures:** sudden tube separation at weld joints.
- **Bearing failure (IDF):** abrupt vibration spike, motor current anomaly.
- **Attemperator / control valve faults:** stuck, oscillating, or failed valve
  position (TV_8329ZC, YJJWSLL).
- **Furnace draft disturbance:** abrupt pressure swing across furnace A–F.
- **Sensor faults:** outlier spikes, stuck-at values (must be distinguishable
  from process faults).

## Criticality notes

- Safety-critical: drum level/pressure, furnace draft, steam outlet
  temperature, tube integrity (leak), IDF integrity.
- Efficiency-critical: economizer, preheaters, air/O₂ trim, bed fluidization
  ΔP.
- ASME/BHEL part correspondence: boiler drum, water-walls, superheaters,
  economizers, desuperheaters/attemperators, PA/SA/ID fans, air preheaters,
  ESP/dust collector (ESP not instrumented in our dataset).

## Key failure-mode literature anchor

- Opt-TCAE (Kim et al., Expert Syst. Appl. 215, 119377, 2023): boiler tube
  leakage detection from multi-sensor data via temporal convolutional
  autoencoder — direct candidate for the tube-leak detection agent.
- ASME boiler monitoring system paper (2024): predictive boiler monitoring,
  boiler tube failures as dominant outage driver.
