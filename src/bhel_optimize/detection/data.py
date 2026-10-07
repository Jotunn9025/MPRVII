"""Build labelled evaluation series from the simulation layer.

The dataset has no per-part fault labels, so per-part ground truth is produced
by injecting the simulated scenarios: each generated row carries a boolean
column per boiler part (was *that* part affected?) plus the fault metadata.

Scripted per scenario: a quiet ``gap`` of normal playback, then a ``block`` of
injected fault, repeated so that every run has a clean onset for latency
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bhel_optimize import config as C
from bhel_optimize.simulation import faults as F
from bhel_optimize.simulation.controller import ACTIVE_SEVERITY_THRESHOLD, ScenarioController

#: Injection is treated as a material fault for a part at this local severity.
MATERIAL_SEVERITY_IQR = ACTIVE_SEVERITY_THRESHOLD


@dataclass
class EvalSeries:
    """Observed sensor values plus per-part ground-truth labels."""

    values: pd.DataFrame  # date + 30 tags (observed, fault-injected)
    labels: pd.DataFrame  # part_<key> booleans + fault metadata
    scenarios: list[str]

    def __len__(self) -> int:
        return len(self.values)

    def part_labels(self, part_key: str) -> pd.Series:
        return self.labels[f"part_{part_key}"]


def affected_parts(scenario_key: str) -> set[str]:
    """Parts whose tags move when this scenario runs."""
    scenario = F.SCENARIOS[scenario_key]
    effects = scenario.spec().effects_for(scenario.target_tag)
    return {C.part_of(effect.tag) for effect in effects}


def build_eval_series(
    frame: pd.DataFrame,
    scenario_keys: list[str],
    *,
    start_index: int = 0,
    block: int = 240,
    gap: int = 120,
) -> EvalSeries:
    """Create a labelled series by injecting each scenario in turn."""
    controller = ScenarioController(frame, mode="simulation", start_index=start_index)
    n_samples = len(controller)
    value_rows: list[dict] = []
    label_rows: list[dict] = []
    used: list[str] = []

    def append(position: int) -> None:
        values = controller.values_at(position)
        values[C.DATE_COLUMN] = frame.iloc[position][C.DATE_COLUMN]
        value_rows.append(values)

    position = start_index
    for scenario_key in scenario_keys:
        if position >= n_samples - 2:
            break
        used.append(scenario_key)
        # quiet normal stretch
        for _ in range(gap):
            if position >= n_samples - 1:
                break
            append(position)
            blank = _blank_labels(position)
            blank["tailed_real"] = bool(controller.labels_at(position)["tailed_real"])
            label_rows.append(blank)
            position += 1
        # injected fault stretch
        controller.position = min(position, n_samples - 2)
        controller.start_scenario(scenario_key)
        active = controller.active
        assert active is not None  # just started
        touched = sorted({C.part_of(e.tag) for e in active.effects})
        for _ in range(block):
            if position >= n_samples - 1:
                break
            append(position)
            labels = controller.labels_at(position)
            row = _blank_labels(position)
            row["fault_key"] = labels["fault_key"]
            row["fault_severity"] = labels["fault_severity"]
            row["fault_active"] = labels["fault_active"]
            row["tailed_real"] = labels["tailed_real"]
            # per-part severity: the injected magnitude *in that part's own
            # tags*, in IQR units. A fault's global severity is set by its
            # strongest effect, which may be a different part entirely, so
            # banding recall on the global number would be misleading.
            deltas = controller.effects_at(position)
            local: dict[str, float] = {}
            for tag, delta in deltas.items():
                part_of_tag = C.part_of(tag)
                local[part_of_tag] = max(
                    local.get(part_of_tag, 0.0),
                    abs(delta) / controller.scales[tag],
                )
            for part_key in touched:
                severity = local.get(part_key, 0.0)
                row[f"severity_{part_key}"] = severity
                row[f"part_{part_key}"] = severity >= MATERIAL_SEVERITY_IQR
            label_rows.append(row)
            position += 1

    values = pd.DataFrame(value_rows)
    labels = pd.DataFrame(label_rows)
    return EvalSeries(values=values, labels=labels, scenarios=used)


def _blank_labels(position: int) -> dict:
    row: dict = {
        "position": position,
        "fault_key": None,
        "fault_severity": 0.0,
        "fault_active": False,
        "tailed_real": False,
    }
    for part_key in C.PARTS:
        row[f"part_{part_key}"] = False
        row[f"severity_{part_key}"] = 0.0
    return row


#: A scenario counts as a fault *for this part* only if its local effect on the
#: part's own tags reaches this magnitude (in IQR units) at full response.
MATERIAL_PART_SEVERITY = 0.5


def part_scenario_severity(scenario_key: str, part_key: str) -> float:
    """Full-response severity of a scenario on a part's own tags (IQR units)."""
    scenario = F.SCENARIOS[scenario_key]
    effects = scenario.spec().effects_for(scenario.target_tag)
    worst = 0.0
    for effect in effects:
        if C.part_of(effect.tag) != part_key:
            continue
        # a frozen/stuck sensor is a full-severity fault even with zero gain
        worst = max(worst, 1.0 if effect.kind == "stuck" else abs(effect.gain))
    return worst


def scenarios_for_part(part_key: str, limit: int = 6, prefer_demo: bool = True) -> list[str]:
    """Scenario keys that *materially* affect ``part_key`` (demo presets first).

    Parts that are only downstream bystanders of a fault (tiny coupled effect)
    are not scored on it — otherwise a part is judged on a signal that is below
    its own noise floor by construction.
    """
    matching = [
        key
        for key in F.SCENARIOS
        if part_scenario_severity(key, part_key) >= MATERIAL_PART_SEVERITY
    ]
    if not matching:
        # No simulated fault is materially sized for this part: fall back to
        # every scenario that touches its tags at all (weakest coupling), so the
        # part is still evaluated and the limitation is recorded in the report.
        matching = [key for key in F.SCENARIOS if part_key in affected_parts(key)]
    if prefer_demo:
        matching.sort(key=lambda key: (not F.SCENARIOS[key].demo, key))
    else:
        matching.sort()
    # keep category diversity: demo first, then one per category
    chosen: list[str] = []
    seen_categories: set[str] = set()
    for key in matching:
        category = F.SCENARIOS[key].spec().category
        if F.SCENARIOS[key].demo:
            chosen.append(key)
        elif category not in seen_categories:
            chosen.append(key)
        seen_categories.add(category)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def part_label_alignment(labels: pd.DataFrame, part_key: str) -> pd.Series:
    """Ground truth for a part (True once any material injection is present)."""
    return labels[f"part_{part_key}"].astype(bool)


def severity_alignment(labels: pd.DataFrame, part_key: str) -> np.ndarray:
    """Per-sample *local* injected severity for a part, in IQR units."""
    import numpy as np

    return labels[f"severity_{part_key}"].to_numpy(dtype="float64")


def part_scenario_ceiling(part_key: str, scenario_keys: list[str]) -> float:
    """Strongest local severity available for a part across given scenarios."""
    return max((part_scenario_severity(k, part_key) for k in scenario_keys), default=0.0)


def build_repeated_eval_series(
    frame: pd.DataFrame,
    scenario_keys: list[str],
    *,
    repeats: int = 4,
    block: int = 240,
    gap: int = 120,
) -> EvalSeries:
    """Concatenate several eval series at different offsets.

    A single short block drawn from a non-stationary series gives a very noisy
    AUROC (validation and test disagreed by >0.25), so the same scenarios are
    injected at several start offsets and pooled.
    """
    if not scenario_keys:
        empty = pd.DataFrame()
        return EvalSeries(values=empty, labels=empty, scenarios=[])
    span = (gap + block) * max(1, len(scenario_keys))
    max_start = max(0, len(frame) - span - 2)
    if repeats <= 1 or max_start == 0:
        return build_eval_series(frame, scenario_keys, block=block, gap=gap)
    starts = np.linspace(0, max_start, repeats).astype(int)
    values_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    for start in starts:
        series = build_eval_series(
            frame, scenario_keys, start_index=int(start), block=block, gap=gap
        )
        if not len(series):
            continue
        values_parts.append(series.values)
        label_parts.append(series.labels)
    if not values_parts:
        empty = pd.DataFrame()
        return EvalSeries(values=empty, labels=empty, scenarios=[])
    values = pd.concat(values_parts, ignore_index=True)
    labels = pd.concat(label_parts, ignore_index=True)
    return EvalSeries(values=values, labels=labels, scenarios=list(scenario_keys))
