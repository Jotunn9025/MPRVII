"""Scenario controller: mode switching, real-data playback, fault injection.

Two demo modes (the UI dropdown switches between them):

- ``livestream``  — pure replay of the real dataset, no injection. The only
  anomalies present are the dataset's own "tailed" samples.
- ``simulation``  — the same real replay with an injected fault scenario
  active; every injected fault drives its whole propagation matrix.

Playback is time-driven (samples per second) for the Flask UI, but also
exposes deterministic :meth:`ScenarioController.advance` stepping for tests
and offline labelled-series generation (:meth:`run_series`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from bhel_optimize import config as C
from bhel_optimize.simulation import faults as F

MODES = ("livestream", "simulation")

#: A fault counts as "materially active" once its strongest effect reaches
#: this fraction of a tag's IQR — used for objective ground-truth labels.
ACTIVE_SEVERITY_THRESHOLD = 0.10
#: "Established" = the fault has reached half a typical spread of the tag, the
#: point at which detection is a fair expectation. Early (0.1-0.5 IQR) detection
#: is reported separately rather than being required.
ESTABLISHED_SEVERITY_THRESHOLD = 0.50


@dataclass
class ActiveFault:
    """A fault that has been started and is currently running."""

    scenario_key: str
    fault_key: str
    name: str
    part: str
    onset_index: int
    duration_s: float
    intensity: float
    target_tag: str | None
    effects: tuple[F.FaultEffect, ...]

    def elapsed_s(self, index: int) -> float:
        return max(0.0, (index - self.onset_index) * C.SAMPLE_INTERVAL_S)

    def finished(self, index: int) -> bool:
        return self.elapsed_s(index) >= self.duration_s


class ScenarioController:
    """Drives playback and fault injection over the real boiler dataset."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        *,
        mode: str = "livestream",
        speed: float = 1.0,
        start_index: int = 0,
        scales: dict[str, float] | None = None,
        loop: bool = False,
    ) -> None:
        if frame is None:
            from bhel_optimize.data import load_dataset

            frame = load_dataset()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if speed <= 0:
            raise ValueError("speed must be > 0 (samples per second)")
        self.frame = frame.reset_index(drop=True)
        self.mode = mode
        self.speed = float(speed)
        self.loop = loop
        self.start_index = int(start_index)
        self.position = int(start_index)
        self.scales = scales if scales is not None else F.tag_scales(self.frame)
        self.active: ActiveFault | None = None
        self._last_tick: float | None = None

    # ------------------------------------------------------------------
    # playback
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def at_end(self) -> bool:
        return self.position >= len(self.frame) - 1

    def set_mode(self, mode: str) -> str:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        if mode == "livestream":
            self.active = None
        return self.mode

    def set_speed(self, speed: float) -> float:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = float(speed)
        return self.speed

    def reset(self, start_index: int | None = None) -> int:
        self.position = int(self.start_index if start_index is None else start_index)
        self.active = None
        self._last_tick = None
        return self.position

    def advance(self, samples: int = 1) -> int:
        """Deterministically move ``samples`` forward (used by tests/offline)."""
        if samples < 0:
            raise ValueError("samples must be >= 0")
        self._expire_faults(self.position)
        self.position = min(self.position + samples, len(self.frame) - 1)
        return self.position

    def tick(self, now: float | None = None) -> int:
        """Advance by wall-clock time; returns how many samples moved."""
        now = time.monotonic() if now is None else now
        if self._last_tick is None:
            self._last_tick = now
            return 0
        elapsed = now - self._last_tick
        steps = int(elapsed * self.speed)
        if steps <= 0:
            return 0
        moved = 0
        for _ in range(steps):
            if self.at_end:
                if self.loop:
                    self.position = self.start_index
                else:
                    break
            else:
                self.advance(1)
                moved += 1
        self._last_tick = now
        return moved

    # ------------------------------------------------------------------
    # fault control
    # ------------------------------------------------------------------

    def start_scenario(
        self,
        scenario_key: str,
        *,
        duration_s: float | None = None,
        intensity: float = 1.0,
        target_tag: str | None = None,
    ) -> ActiveFault:
        """Start a preset scenario (implies simulation mode)."""
        if scenario_key not in F.SCENARIOS:
            raise KeyError(f"unknown scenario: {scenario_key}")
        scenario = F.SCENARIOS[scenario_key]
        spec = scenario.spec()
        resolved_target = target_tag or scenario.target_tag or spec.target_tag
        self.mode = "simulation"
        self.active = ActiveFault(
            scenario_key=scenario.key,
            fault_key=spec.key,
            name=scenario.name,
            part=spec.part,
            onset_index=self.position,
            duration_s=(
                spec.default_duration_s if duration_s is None else float(duration_s)
            ),
            intensity=float(intensity),
            target_tag=resolved_target,
            effects=spec.effects_for(resolved_target),
        )
        return self.active

    def start_fault(
        self,
        fault_key: str,
        *,
        duration_s: float | None = None,
        intensity: float = 1.0,
        target_tag: str | None = None,
    ) -> ActiveFault:
        """Start a raw fault from the fault library."""
        if fault_key not in F.FAULTS:
            raise KeyError(f"unknown fault: {fault_key}")
        self.mode = "simulation"
        spec = F.FAULTS[fault_key]
        resolved_target = target_tag or spec.target_tag
        self.active = ActiveFault(
            scenario_key=fault_key,
            fault_key=fault_key,
            name=spec.name,
            part=spec.part,
            onset_index=self.position,
            duration_s=(
                spec.default_duration_s if duration_s is None else float(duration_s)
            ),
            intensity=float(intensity),
            target_tag=resolved_target,
            effects=spec.effects_for(resolved_target),
        )
        return self.active

    def stop_scenario(self) -> None:
        """Cancel any active fault (playback continues)."""
        self.active = None

    def _expire_faults(self, index: int) -> None:
        if self.active is not None and self.active.finished(index):
            self.active = None

    def current_fault(self, index: int | None = None) -> ActiveFault | None:
        """The running fault at ``index``, or ``None`` if none/expired."""
        index = self.position if index is None else index
        if self.mode != "simulation" or self.active is None:
            return None
        return None if self.active.finished(index) else self.active

    # ------------------------------------------------------------------
    # state inspection
    # ------------------------------------------------------------------

    def baseline_row(self, index: int | None = None) -> pd.Series:
        return self.frame.iloc[self.position if index is None else index]

    def effects_at(self, index: int) -> dict[str, float]:
        """Injected deltas per tag for the given index (empty in livestream)."""
        fault = self.current_fault(index)
        if fault is None:
            return {}
        elapsed = fault.elapsed_s(index)
        baseline = self.frame.iloc[index]
        onset = self.frame.iloc[fault.onset_index]
        deltas: dict[str, float] = {}
        for effect in fault.effects:
            if effect.kind == "stuck":
                deltas[effect.tag] = float(onset[effect.tag] - baseline[effect.tag])
                continue
            shape = F.response(
                effect.kind,
                elapsed,
                tau_s=effect.tau_s,
                ramp_s=F.FAULTS[fault.fault_key].ramp_s,
                dead_time_s=effect.dead_time_s,
                period_s=effect.period_s,
                width_s=effect.width_s,
            )
            delta = effect.gain * fault.intensity * self.scales[effect.tag] * shape
            if delta:
                deltas[effect.tag] = deltas.get(effect.tag, 0.0) + float(delta)
        return deltas

    def severity_at(self, index: int) -> float:
        """Strongest active effect, in IQR units (0 when nothing is injected)."""
        fault = self.current_fault(index)
        if fault is None:
            return 0.0
        elapsed = fault.elapsed_s(index)
        ramp_s = F.FAULTS[fault.fault_key].ramp_s
        worst = 0.0
        for effect in fault.effects:
            if effect.kind == "stuck":
                # a frozen sensor is fully "active" from its onset
                worst = max(worst, 1.0)
                continue
            shape = F.response(
                effect.kind,
                elapsed,
                tau_s=effect.tau_s,
                ramp_s=ramp_s,
                dead_time_s=effect.dead_time_s,
                period_s=effect.period_s,
                width_s=effect.width_s,
            )
            worst = max(worst, abs(effect.gain * fault.intensity * shape))
        return worst

    def values_at(self, index: int) -> dict[str, float]:
        """Observed sensor values (baseline + injected deltas)."""
        row = self.frame.iloc[index]
        values = {tag: float(row[tag]) for tag in C.TAGS}
        for tag, delta in self.effects_at(index).items():
            values[tag] = values[tag] + delta
        return values

    def labels_at(self, index: int) -> dict:
        """Ground-truth labels for one sample (used by detection/evaluation)."""
        baseline = self.frame.iloc[index]
        observed_steam = self.values_at(index)[C.STATE_TAG]
        low, high = C.NORMAL_STEAM_TEMP_C
        fault = self.current_fault(index)
        severity = self.severity_at(index)
        elapsed = fault.elapsed_s(index) if fault else None
        return {
            "fault_key": fault.fault_key if fault else None,
            "fault_part": fault.part if fault else None,
            "fault_category": F.FAULTS[fault.fault_key].category if fault else None,
            "fault_elapsed_s": elapsed,
            "fault_severity": severity,
            "fault_active": bool(fault and severity >= ACTIVE_SEVERITY_THRESHOLD),
            "fault_established": bool(
                fault and severity >= ESTABLISHED_SEVERITY_THRESHOLD
            ),
            "tailed_real": bool(
                baseline[C.STATE_TAG] < low or baseline[C.STATE_TAG] > high
            ),
            "tailed_observed": bool(observed_steam < low or observed_steam > high),
        }

    def snapshot(self, index: int | None = None) -> dict:
        """Live payload for the UI: values, deltas, labels, scenario state."""
        index = self.position if index is None else index
        baseline = self.frame.iloc[index]
        deltas = self.effects_at(index)
        labels = self.labels_at(index)
        fault = self.current_fault(index)
        return {
            "mode": self.mode,
            "position": int(index),
            "total_samples": len(self.frame),
            "speed": self.speed,
            "timestamp": baseline[C.DATE_COLUMN].isoformat(),
            "at_end": bool(index >= len(self.frame) - 1),
            "values": self.values_at(index),
            "baseline_values": {tag: float(baseline[tag]) for tag in C.TAGS},
            "deltas": deltas,
            "affected_tags": sorted(deltas),
            "labels": labels,
            "scenario": (
                {
                    "scenario_key": fault.scenario_key,
                    "fault_key": fault.fault_key,
                    "name": fault.name,
                    "part": fault.part,
                    "part_name": C.PARTS[fault.part].name,
                    "elapsed_s": fault.elapsed_s(index),
                    "duration_s": fault.duration_s,
                    "intensity": fault.intensity,
                    "target_tag": fault.target_tag,
                }
                if fault
                else None
            ),
        }

    def status(self) -> dict:
        """Cheap status without sensor values (for polls)."""
        fault = self.current_fault()
        return {
            "mode": self.mode,
            "position": self.position,
            "total_samples": len(self.frame),
            "speed": self.speed,
            "at_end": self.at_end,
            "scenario_key": fault.scenario_key if fault else None,
            "fault_key": fault.fault_key if fault else None,
            "fault_part": fault.part if fault else None,
            "elapsed_s": fault.elapsed_s(self.position) if fault else None,
        }


def parts_metadata() -> dict:
    """Static boiler-part metadata for the UI diagram."""
    return {
        key: {
            "key": key,
            "name": part.name,
            "description": part.description,
            "criticality": part.criticality,
            "tags": list(part.tags),
        }
        for key, part in C.PARTS.items()
    }


def run_series(
    scenario_key: str,
    *,
    frame: pd.DataFrame | None = None,
    samples: int = 600,
    start_index: int = 0,
    intensity: float = 1.0,
    target_tag: str | None = None,
) -> pd.DataFrame:
    """Generate a deterministic labelled series for one scenario.

    Returns the observed tag columns plus the ground-truth label columns
    (``fault_key``, ``fault_part``, ``fault_severity``, ``fault_active``,
    ``tailed_real``, ``tailed_observed``) — the training/evaluation input for
    the detection layer.
    """
    controller = ScenarioController(frame, mode="simulation", start_index=start_index)
    controller.start_scenario(scenario_key, intensity=intensity, target_tag=target_tag)
    rows = []
    for offset in range(samples):
        index = min(start_index + offset, len(controller) - 1)
        row = {C.DATE_COLUMN: controller.frame.iloc[index][C.DATE_COLUMN]}
        row.update(controller.values_at(index))
        row.update(controller.labels_at(index))
        rows.append(row)
        if index >= len(controller) - 1:
            break
    return pd.DataFrame(rows)
