"""Fault model: FOPDT-style dynamics and the sensor propagation matrix.

Why a propagation matrix
------------------------
A real boiler fault never moves a single sensor. Fouling raises flue-gas
temperature *and* slightly lowers O2 *and* loads the ID fan; a tube leak drops
drum pressure, steam temperature and steam flow together. So every fault here
defines a **primary** effect on the tags of the affected part plus **coupled**
effects on the physically-correlated tags, each with its own gain, time
constant and dead time.

Magnitude convention (honest limitation)
---------------------------------------
The dataset's DCS values are raw and their engineering units are not
published, so physical magnitudes cannot be claimed. Effects are therefore
expressed as **multiples of each tag's IQR** (robust spread measured from the
real data): ``gain=0.5`` means "half of this tag's typical spread". Signs and
relative sizes follow the process cause-chain in ``.agents/boiler.md``; the
absolute size stays dataset-grounded and detectable.

Dynamics (first-order plus dead time)
-------------------------------------
``step``       ``1 - exp(-(t - dead) / tau)``   — abrupt faults (leak, failure)
``ramp``       ``min(1, (t - dead) / ramp)``    — creeping faults (fouling, wear)
``oscillate``  ``sin(2*pi*t / period)``         — valve/control oscillation
``spike``      1 inside a short window           — sensor outlier
``stuck``      freeze the tag at its onset value — sensor stuck / valve stuck
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from bhel_optimize import config as C

# --------------------------------------------------------------------------
# Recurring tag groups (keeps the propagation maps readable)
# --------------------------------------------------------------------------

FURNACE_PRESSURES = (
    "PT_8313A.AV_0",
    "PT_8313B.AV_0",
    "PT_8313C.AV_0",
    "PT_8313D.AV_0",
    "PT_8313E.AV_0",
    "PT_8313F.AV_0",
)
ECON_FLUE_TEMPS = ("TE_8319A.AV_0", "TE_8319B.AV_0")
ECON_O2 = ("AIR_8301A.AV_0", "AIR_8301B.AV_0")
HEARTH_DP = ("SXLTCYZ.AV_0", "SXLTCYY.AV_0")
LAYER_DP = ("ZCLCCY.AV_0", "YCLCCY.AV_0")
RETURN_AIR_FLOWS = ("FT_8306A.AV_0", "FT_8306B.AV_0")
ID_FAN_VIBRATION = ("YFJ3_ZD1.AV_0", "YFJ3_ZD2.AV_0")
PREHEATER_AIR_TEMPS = ("TE_8303.AV_0", "TE_8304.AV_0")

DEFAULT_TAU_S = 60.0


# --------------------------------------------------------------------------
# Effect / fault specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultEffect:
    """One tag's response to a fault (gain in multiples of that tag's IQR)."""

    tag: str
    gain: float
    tau_s: float = DEFAULT_TAU_S
    dead_time_s: float = 0.0
    kind: str = "step"  # step | ramp | oscillate | spike
    period_s: float = 120.0  # for oscillate
    width_s: float = 3.0  # for spike
    note: str = ""


@dataclass(frozen=True)
class FaultSpec:
    """A fault: metadata plus its coupled effects on sensor tags."""

    key: str
    name: str
    category: str  # creeping | abrupt | sensor | global
    part: str  # primary boiler part key (config.PARTS)
    description: str
    effects: tuple[FaultEffect, ...]
    default_duration_s: float = 900.0
    ramp_s: float = 300.0  # used by ramp/creeping effects
    target_tag: str | None = None  # for sensor faults (set at start time)
    physics: str = ""

    def effects_for(self, target_tag: str | None = None) -> tuple[FaultEffect, ...]:
        """Return effects, substituting the target tag for sensor faults."""
        if self.target_tag is None:
            return self.effects
        tag = target_tag or self.target_tag
        if tag not in C.TAGS:
            raise ValueError(f"unknown target tag: {tag}")
        return tuple(
            FaultEffect(
                tag=tag,
                gain=e.gain,
                tau_s=e.tau_s,
                dead_time_s=e.dead_time_s,
                kind=e.kind,
                period_s=e.period_s,
                width_s=e.width_s,
                note=e.note,
            )
            for e in self.effects
        )


def _eff(tag: str, gain: float, kind: str = "step", **kw) -> FaultEffect:
    return FaultEffect(tag=tag, gain=gain, kind=kind, **kw)


def _group(tags: Iterable[str], gain: float, kind: str = "step", **kw):
    return tuple(_eff(t, gain, kind, **kw) for t in tags)


# --------------------------------------------------------------------------
# Fault library — see .agents/boiler.md for the failure-mode knowledge
# --------------------------------------------------------------------------

FAULTS: dict[str, FaultSpec] = {
    # ---------------- creeping (slow onset, hours-scale in reality) --------
    "economizer_fouling": FaultSpec(
        key="economizer_fouling",
        name="Economizer fouling",
        category="creeping",
        part="economizer",
        description="Ash/soot deposits on economizer surfaces reduce heat pickup.",
        physics=(
            "Less heat extracted in the economizer -> flue-gas outlet temperature "
            "rises, O2 trims down, steam temperature sags, draft resistance loads "
            "the ID fan."
        ),
        ramp_s=600.0,
        default_duration_s=1800.0,
        effects=(
            *_group(ECON_FLUE_TEMPS, +1.0, "ramp"),
            *_group(ECON_O2, -0.35, "ramp"),
            _eff("TE_8332A.AV_0", -0.35, "ramp"),
            _eff("YFJ3_AI.AV_0", +0.25, "ramp"),
        ),
    ),
    "id_fan_degradation": FaultSpec(
        key="id_fan_degradation",
        name="Induced draft fan degradation",
        category="creeping",
        part="id_fan",
        description="Progressive bearing wear / impeller erosion reduces fan efficiency.",
        physics=(
            "Bearing vibration and motor current creep up; weaker draft makes furnace "
            "pressures less negative and leaves flue gas hotter; O2 drifts."
        ),
        ramp_s=900.0,
        default_duration_s=2700.0,
        effects=(
            _eff("YFJ3_ZD1.AV_0", +0.9, "ramp", note="bearing shell A"),
            _eff("YFJ3_ZD2.AV_0", +0.8, "ramp", note="bearing shell B"),
            _eff("YFJ3_AI.AV_0", +0.6, "ramp", note="current rises as efficiency drops"),
            *_group(FURNACE_PRESSURES, -0.35, "ramp", dead_time_s=30.0),
            *_group(ECON_FLUE_TEMPS, +0.30, "ramp", dead_time_s=30.0),
            *_group(ECON_O2, -0.20, "ramp", dead_time_s=30.0),
            _eff("TE_8332A.AV_0", -0.20, "ramp", dead_time_s=30.0),
        ),
    ),
    "air_preheater_fouling": FaultSpec(
        key="air_preheater_fouling",
        name="Air preheater fouling",
        category="creeping",
        part="air_preheaters",
        description="Deposits on preheater elements cut combustion-air heating.",
        physics=(
            "Preheated air temperatures fall, flue gas leaves hotter, furnace "
            "temperature shifts and O2 trims drift."
        ),
        ramp_s=600.0,
        default_duration_s=1800.0,
        effects=(
            *_group(PREHEATER_AIR_TEMPS, -0.8, "ramp"),
            *_group(ECON_FLUE_TEMPS, +0.35, "ramp", dead_time_s=30.0),
            *_group(ECON_O2, -0.25, "ramp"),
            _eff("TE_8313B.AV_0", +0.2, "ramp", dead_time_s=30.0),
        ),
    ),
    "bed_dp_drift": FaultSpec(
        key="bed_dp_drift",
        name="Fluidized-bed circulation blockage",
        category="creeping",
        part="fluidized_bed",
        description="Bed material circulation degrades; hearth/layer differential pressures drift.",
        physics=(
            "Hearth and layer delta-P rise as the bed densifies/blockages build; "
            "return-air flow falls, furnace pressure and temperature shift."
        ),
        ramp_s=900.0,
        default_duration_s=2700.0,
        effects=(
            *_group(HEARTH_DP, +0.7, "ramp"),
            *_group(LAYER_DP, +0.8, "ramp"),
            *_group(RETURN_AIR_FLOWS, -0.4, "ramp", dead_time_s=30.0),
            *_group(FURNACE_PRESSURES, -0.3, "ramp", dead_time_s=30.0),
            _eff("TE_8313B.AV_0", +0.25, "ramp", dead_time_s=30.0),
        ),
    ),
    # ---------------- abrupt / structural --------------------------------
    "water_wall_tube_leak": FaultSpec(
        key="water_wall_tube_leak",
        name="Furnace water-wall tube leak",
        category="abrupt",
        part="furnace",
        description="Tube rupture in the furnace water wall (the most common boiler failure).",
        physics=(
            "Furnace pressure is disturbed, drum pressure collapses, steam "
            "temperature and main steam flow drop together, bed delta-P reacts."
        ),
        default_duration_s=600.0,
        effects=(
            *_group(FURNACE_PRESSURES, +0.6, dead_time_s=10.0, tau_s=45.0),
            _eff("PTCA_8322A.AV_0", -0.8, dead_time_s=10.0, tau_s=45.0),
            _eff("PTCA_8324.AV_0", -0.7, dead_time_s=10.0, tau_s=45.0),
            _eff("TE_8332A.AV_0", -0.9, dead_time_s=15.0, tau_s=60.0),
            _eff("ZZQBCHLL.AV_0", -0.5, dead_time_s=15.0, tau_s=60.0),
            *_group(HEARTH_DP, -0.35, dead_time_s=20.0, tau_s=60.0),
        ),
    ),
    "superheater_tube_leak": FaultSpec(
        key="superheater_tube_leak",
        name="Superheater tube leak",
        category="abrupt",
        part="superheater",
        description="Superheater tube rupture losing high-grade steam.",
        physics=(
            "Steam temperature falls sharply, attemperator spray and valve react, "
            "main steam flow and drum pressure read low/uneven (leak flow)."
        ),
        default_duration_s=480.0,
        effects=(
            _eff("TE_8332A.AV_0", -1.2, dead_time_s=10.0, tau_s=45.0),
            _eff("YJJWSLL.AV_0", +0.5, dead_time_s=15.0, tau_s=60.0, note="spray compensates"),
            _eff("TV_8329ZC.AV_0", +0.6, dead_time_s=15.0, tau_s=60.0),
            _eff("ZZQBCHLL.AV_0", +0.4, dead_time_s=20.0, tau_s=60.0, note="leak flow"),
            _eff("PTCA_8324.AV_0", -0.5, dead_time_s=15.0, tau_s=60.0),
        ),
    ),
    "id_fan_bearing_failure": FaultSpec(
        key="id_fan_bearing_failure",
        name="Induced draft fan bearing failure",
        category="abrupt",
        part="id_fan",
        description="Sudden bearing shell failure on the induced draft fan.",
        physics=(
            "Vibration spikes on both shells, motor current surges, draft weakens "
            "(furnace pressure, flue-gas temperature, O2 all shift)."
        ),
        default_duration_s=420.0,
        effects=(
            _eff("YFJ3_ZD1.AV_0", +2.0, dead_time_s=5.0, tau_s=20.0),
            _eff("YFJ3_ZD2.AV_0", +1.6, dead_time_s=5.0, tau_s=20.0),
            _eff("YFJ3_AI.AV_0", +1.0, dead_time_s=5.0, tau_s=20.0),
            *_group(FURNACE_PRESSURES, -0.5, dead_time_s=15.0, tau_s=45.0),
            *_group(ECON_FLUE_TEMPS, +0.35, dead_time_s=15.0, tau_s=45.0),
        ),
    ),
    "attemperator_valve_stuck": FaultSpec(
        key="attemperator_valve_stuck",
        name="Attemperator valve stuck / oscillating",
        category="abrupt",
        part="superheater",
        description="Desuperheater regulating valve jams, breaking temperature control.",
        physics=(
            "Spray-water flow freezes while steam temperature oscillates around "
            "setpoint; drum/steam flow pick up the control fight."
        ),
        default_duration_s=600.0,
        effects=(
            _eff("TV_8329ZC.AV_0", 0.0, kind="stuck", dead_time_s=5.0),
            _eff("YJJWSLL.AV_0", -0.5, dead_time_s=5.0, tau_s=30.0),
            _eff("TE_8332A.AV_0", +0.6, kind="oscillate", period_s=90.0, dead_time_s=10.0),
            _eff("ZZQBCHLL.AV_0", +0.2, kind="oscillate", period_s=90.0, dead_time_s=10.0),
        ),
    ),
    "furnace_draft_disturbance": FaultSpec(
        key="furnace_draft_disturbance",
        name="Furnace draft disturbance / air in-leak",
        category="abrupt",
        part="furnace",
        description="Draft upset with cold-air in-leakage into the furnace/flue path.",
        physics=(
            "Furnace pressures swing, flue-gas O2 rises from in-leakage, furnace "
            "temperature and ID fan current react."
        ),
        default_duration_s=600.0,
        effects=(
            *_group(FURNACE_PRESSURES, +0.9, dead_time_s=5.0, tau_s=30.0),
            *_group(ECON_O2, +0.6, dead_time_s=20.0, tau_s=60.0),
            _eff("TE_8313B.AV_0", -0.35, dead_time_s=20.0, tau_s=60.0),
            _eff("YFJ3_AI.AV_0", +0.35, dead_time_s=10.0, tau_s=45.0),
            *_group(ECON_FLUE_TEMPS, -0.2, dead_time_s=20.0, tau_s=60.0),
            # draft upset makes the fans hunt and return-air flow shift, which
            # is what the combustion-air and return-air parts actually see
            *_group(("FT_8301.AV_0", "FT_8302.AV_0"), +0.55, dead_time_s=10.0, tau_s=45.0),
            *_group(RETURN_AIR_FLOWS, +0.45, dead_time_s=15.0, tau_s=45.0),
        ),
    ),
    # ---------------- sensor faults (target configurable) ----------------
    "sensor_outlier_spike": FaultSpec(
        key="sensor_outlier_spike",
        name="Sensor outlier spike",
        category="sensor",
        part="economizer",
        description="Short spurious spike on one sensor (single-tag fault).",
        physics="Instrument/EMI fault — must be distinguishable from a process fault "
        "because only the target tag moves.",
        default_duration_s=60.0,
        target_tag="TE_8319A.AV_0",
        effects=(_eff("TE_8319A.AV_0", +3.0, kind="spike", width_s=10.0),),
    ),
    "sensor_stuck": FaultSpec(
        key="sensor_stuck",
        name="Sensor stuck-at value",
        category="sensor",
        part="furnace",
        description="Sensor freezes at its last reading (single-tag fault).",
        physics="Instrument failure — the tag stops tracking the process.",
        default_duration_s=600.0,
        target_tag="PT_8313A.AV_0",
        effects=(_eff("PT_8313A.AV_0", 0.0, kind="stuck"),),
    ),
    # ---------------- global (dataset's own anomaly definition) -----------
    "steam_temp_anomaly": FaultSpec(
        key="steam_temp_anomaly",
        name="Steam temperature out of band (tailed data)",
        category="global",
        part="main_steam",
        description=(
            "Drives the state-discriminating tag outside 530-545 degC, reproducing "
            "the dataset's own 'tailed' anomaly definition."
        ),
        physics="Reproduces the long-tailed anomaly class shipped with the dataset.",
        default_duration_s=900.0,
        effects=(
            _eff("TE_8332A.AV_0", -2.5, dead_time_s=30.0, tau_s=180.0),
            _eff("ZZQBCHLL.AV_0", -0.4, dead_time_s=30.0, tau_s=180.0),
        ),
    ),
}


# --------------------------------------------------------------------------
# Scenario presets (what the UI dropdown lists)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A runnable preset: one fault with concrete duration/intensity."""

    key: str
    name: str
    fault_key: str
    duration_s: float
    intensity: float = 1.0
    target_tag: str | None = None
    description: str = ""
    demo: bool = False

    def spec(self) -> FaultSpec:
        return FAULTS[self.fault_key]


SCENARIOS: dict[str, Scenario] = {}


def _add_scenario(scenario: Scenario) -> None:
    SCENARIOS[scenario.key] = scenario


# demo presets (curated: one creeping, one abrupt, one sensor, one global)
_add_scenario(
    Scenario(
        key="demo_idf_degradation",
        name="DEMO — IDF degradation (creeping)",
        fault_key="id_fan_degradation",
        duration_s=2700.0,
        description="Slow fan degradation: vibration/current creep, gentle draft loss.",
        demo=True,
    )
)
_add_scenario(
    Scenario(
        key="demo_water_wall_leak",
        name="DEMO — water-wall tube leak (abrupt)",
        fault_key="water_wall_tube_leak",
        duration_s=600.0,
        description="Fast leak: drum pressure, steam temperature and steam flow fall.",
        demo=True,
    )
)
_add_scenario(
    Scenario(
        key="demo_sensor_spike",
        name="DEMO — sensor outlier spike",
        fault_key="sensor_outlier_spike",
        duration_s=60.0,
        description="Single-tag spike on an economizer gas-temperature sensor.",
        demo=True,
    )
)
_add_scenario(
    Scenario(
        key="demo_tailed_steam_temp",
        name="DEMO — steam temperature out of band",
        fault_key="steam_temp_anomaly",
        duration_s=900.0,
        description="Reproduces the dataset's 'tailed' anomaly definition.",
        demo=True,
    )
)

# every fault also gets a preset with its own defaults
for _key, _fault in FAULTS.items():
    if any(s.fault_key == _key and s.demo for s in SCENARIOS.values()):
        continue
    _add_scenario(
        Scenario(
            key=_key,
            name=_fault.name,
            fault_key=_key,
            duration_s=_fault.default_duration_s,
            target_tag=_fault.target_tag,
            description=_fault.description,
        )
    )


def list_scenarios() -> list[dict]:
    """Metadata for the UI dropdown (demo presets first)."""
    scenarios = sorted(SCENARIOS.values(), key=lambda s: (not s.demo, s.name))
    return [
        {
            "key": s.key,
            "name": s.name,
            "fault_key": s.fault_key,
            "category": s.spec().category,
            "part": s.spec().part,
            "duration_s": s.duration_s,
            "target_tag": s.target_tag,
            "description": s.description,
            "demo": s.demo,
        }
        for s in scenarios
    ]


def describe_faults() -> list[dict]:
    """Documentation view of the fault library and its propagation matrix."""
    out = []
    for fault in FAULTS.values():
        out.append(
            {
                "key": fault.key,
                "name": fault.name,
                "category": fault.category,
                "part": fault.part,
                "part_name": C.PARTS[fault.part].name,
                "description": fault.description,
                "physics": fault.physics,
                "default_duration_s": fault.default_duration_s,
                "target_tag": fault.target_tag,
                "effects": [
                    {
                        "tag": e.tag,
                        "tag_description": C.TAG_INFO[e.tag],
                        "gain_iqr": e.gain,
                        "kind": e.kind,
                        "tau_s": e.tau_s,
                        "dead_time_s": e.dead_time_s,
                        "note": e.note,
                    }
                    for e in fault.effects
                ],
            }
        )
    return out


# --------------------------------------------------------------------------
# Magnitude scales (IQR per tag) and response dynamics
# --------------------------------------------------------------------------


def tag_scales(frame: pd.DataFrame, tags: Iterable[str] | None = None) -> dict[str, float]:
    """Per-tag robust spread (IQR) used to convert gains into data units."""
    tags = tuple(C.TAGS if tags is None else tags)
    out: dict[str, float] = {}
    for tag in tags:
        q1, q3 = frame[tag].quantile([0.25, 0.75])
        spread = float(q3 - q1)
        out[tag] = spread if spread > 0 else 1.0
    return out


def default_tag_scales() -> dict[str, float]:
    """IQR scales measured from the shipped dataset."""
    from bhel_optimize.data import load_dataset

    return tag_scales(load_dataset())


def response(
    kind: str,
    elapsed_s: float,
    *,
    tau_s: float = DEFAULT_TAU_S,
    ramp_s: float = 300.0,
    dead_time_s: float = 0.0,
    period_s: float = 120.0,
    width_s: float = 3.0,
) -> float:
    """Normalised fault response in ``[0, 1]`` (or ``[-1, 1]`` for oscillate)."""
    t = elapsed_s - dead_time_s
    if t <= 0:
        return 0.0
    if kind == "step":
        return 1.0 - math.exp(-t / max(tau_s, 1e-6))
    if kind == "ramp":
        return min(1.0, t / max(ramp_s, 1e-6))
    if kind == "oscillate":
        return math.sin(2.0 * math.pi * t / max(period_s, 1e-6))
    if kind == "spike":
        return 1.0 if t <= width_s else 0.0
    if kind == "stuck":
        return 1.0
    raise ValueError(f"unknown fault response kind: {kind}")
