"""Situation levels, trends and the pattern matcher used by the Overview Agent.

Level model
-----------
Per boiler part the agent tracks a **level** derived from that part's calibrated
ensemble score:

``NOMINAL``  score below the watch band
``WATCH``    rising or mildly elevated — worth a human eye
``ALERT``    the part's own calibrated decision threshold is exceeded
``CRITICAL`` well above threshold (or escalating while already severe)

Two safeguards stop the display from flapping: a *persistence* requirement
(an elevated instant level must hold for several samples) and a *rate limit*
(the displayed level moves at most one step per update).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from bhel_optimize import config as C

NOMINAL, WATCH, ALERT, CRITICAL = 0, 1, 2, 3
LEVEL_NAMES = ("NOMINAL", "WATCH", "ALERT", "CRITICAL")

#: Fallback bands as multiples of the calibrated threshold, used only when a
#: part has no measured clean-data quantiles. The primary bands come from the
#: part's own score distribution on clean normal data (see LevelBands), because
#: a fixed fraction of the threshold made perfectly normal livestream data
#: report WATCH/ALERT.
WATCH_FRACTION = 0.60
CRITICAL_FRACTION = 2.00
#: Trend is called rising/falling when the recent slope moves this fraction of
#: the threshold over the trend window.
TREND_FRACTION = 0.25
#: ALERT/CRITICAL must persist this many of the last samples...
PERSISTENCE_WINDOW = 5
PERSISTENCE_MIN = 3
#: ...and WATCH needs a longer, stricter gate, because the WATCH cut is the
#: 90-95th percentile of *normal* behaviour and autocorrelated normal
#: excursions would otherwise keep parts permanently "watched".
WATCH_PERSISTENCE_WINDOW = 10
WATCH_PERSISTENCE_MIN = 6
#: Levels on safety-critical parts (>= this criticality) are bumped one step
#: when they sit at WATCH, so a watch on the drum is not reported as calmly as
#: a watch on a return-air duct.
CRITICALITY_ESCALATION = 4
#: A criticality-escalated WATCH must have held this long before the summary
#: calls it out (avoids single-sample blips being shouted about).
CRITICAL_WATCH_HOLD_S = 120.0
#: Default WATCH cut = this quantile of the part's clean-normal score law.
WATCH_QUANTILE = 0.95
#: Minimum clean-data quantile the live ALERT band must respect (alarm limit).
ALERT_QUANTILE = 0.995
#: Minimum matcher confidence to name a probable fault pattern.
PATTERN_MIN_CONFIDENCE = 0.45


@dataclass(frozen=True)
class LevelBands:
    """Score cut-points for a part, measured on clean normal data."""

    watch: float
    alert: float
    critical: float

    @classmethod
    def from_threshold(cls, threshold: float) -> "LevelBands":
        """Fallback bands relative to the decision threshold."""
        return cls(
            watch=WATCH_FRACTION * threshold,
            alert=threshold,
            critical=CRITICAL_FRACTION * threshold,
        )

    @classmethod
    def from_clean_scores(
        cls, clean_scores, threshold: float, watch_q: float = 0.90, critical_q: float = 0.999
    ) -> "LevelBands":
        """Bands from the part's own clean-normal score distribution.

        ``watch`` = 90th percentile of normal behaviour (so ~10% of clean
        samples sit in the band before persistence is applied), ``alert`` = the
        calibrated decision threshold, ``critical`` = far tail of normal.
        """
        values = np.asarray([v for v in clean_scores if np.isfinite(v)], dtype="float64")
        if values.size == 0 or not np.isfinite(threshold):
            return cls.from_threshold(threshold if np.isfinite(threshold) else 1.0)
        watch = float(np.quantile(values, watch_q))
        critical = max(float(np.quantile(values, critical_q)), 1.5 * float(threshold))
        return cls(watch=watch, alert=float(threshold), critical=critical)

    def level_for(self, score: float) -> int:
        if not np.isfinite(score):
            return NOMINAL
        if score >= self.critical:
            return CRITICAL
        if score >= self.alert:
            return ALERT
        if score >= self.watch:
            return WATCH
        return NOMINAL


def level_for_score(score: float, threshold: float) -> int:
    """Instant level from a score and a bare threshold (convenience wrapper).

    A missing/non-positive threshold means "no calibrated bands yet", which must
    read NOMINAL rather than CRITICAL.
    """
    if not np.isfinite(score) or not np.isfinite(threshold) or threshold <= 0:
        return NOMINAL
    return LevelBands.from_threshold(threshold).level_for(score)


def trend_for_history(history: list[float] | deque[float], threshold: float) -> str:
    """``rising`` / ``falling`` / ``steady`` from the recent score slope."""
    values = np.asarray([v for v in history if np.isfinite(v)], dtype="float64")
    if len(values) < 3 or threshold <= 0:
        return "steady"
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
    change_over_window = slope * (len(values) - 1)
    if change_over_window >= TREND_FRACTION * threshold:
        return "rising"
    if change_over_window <= -TREND_FRACTION * threshold:
        return "falling"
    return "steady"


@dataclass
class PartState:
    """Tracked situation state for one boiler part."""

    part_key: str
    level: int = NOMINAL
    since_index: int | None = None
    history: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    above_flags: deque[bool] = field(
        default_factory=lambda: deque(maxlen=PERSISTENCE_WINDOW)
    )
    watch_flags: deque[bool] = field(
        default_factory=lambda: deque(maxlen=WATCH_PERSISTENCE_WINDOW)
    )

    def reset(self) -> None:
        self.level = NOMINAL
        self.since_index = None
        self.history.clear()
        self.above_flags.clear()
        self.watch_flags.clear()


def apply_persistence(instant: int, state: PartState) -> int:
    """Downgrade a fresh elevated instant level until it persists.

    ALERT/CRITICAL need :data:`PERSISTENCE_MIN` of the last
    :data:`PERSISTENCE_WINDOW` samples above the decision threshold; WATCH needs
    its own, stricter gate so ordinary autocorrelated noise does not park parts
    in the watch band.
    """
    if instant <= WATCH:
        return instant
    if instant == WATCH:
        persistent = sum(state.watch_flags) >= WATCH_PERSISTENCE_MIN
        return WATCH if persistent else NOMINAL
    flags = state.above_flags
    persistent = sum(flags) >= PERSISTENCE_MIN
    return instant if persistent else max(WATCH, instant - 1)


def step_level(previous: int, instant: int) -> int:
    """Rate-limit the displayed level to one step per update."""
    if instant > previous:
        return previous + 1
    if instant < previous:
        return previous - 1
    return previous


def criticality_adjusted(level: int, criticality: int, held_s: float = float("inf")) -> int:
    """Bump a *sustained* WATCH on a safety-critical part to ALERT.

    ``held_s`` guards against a single-sample blip on a critical part rolling
    the whole plant up to ALERT.
    """
    if (
        level == WATCH
        and criticality >= CRITICALITY_ESCALATION
        and held_s >= CRITICAL_WATCH_HOLD_S
    ):
        return ALERT
    return level


def plant_level(part_levels: dict[str, int], held_s: dict[str, float] | None = None) -> int:
    """Boiler-level situation: the worst part level.

    Criticality weighting deliberately does **not** change the level itself —
    an ALERT must mean "this part exceeded its calibrated decision threshold",
    otherwise ordinary noise on a criticality-5 part made a perfectly normal
    plant read ALERT. Criticality instead drives attention ordering and the
    ``criticality_escalated`` flag that the summary calls out.
    """
    if not part_levels:
        return NOMINAL
    return max(part_levels.values())


def criticality_escalated(part_levels: dict[str, int], held_s: dict[str, float]) -> list[str]:
    """Critical parts sitting at WATCH long enough to deserve emphasis."""
    return [
        key
        for key, level in part_levels.items()
        if criticality_adjusted(level, C.PARTS[key].criticality, held_s.get(key, 0.0)) > level
    ]


def top_deviating_tags(
    deviations: dict[str, float], limit: int = 3
) -> list[tuple[str, float]]:
    """Tags furthest from their normal baseline, by absolute IQR deviation."""
    ranked = sorted(
        ((tag, value) for tag, value in deviations.items() if np.isfinite(value)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return ranked[:limit]


def match_fault_patterns(
    part_key: str, deviations: dict[str, float], min_confidence: float = PATTERN_MIN_CONFIDENCE
) -> list[dict]:
    """Match observed tag deviations against the fault library's propagation maps.

    For every fault that couples into this part, the expected effect tags (with
    their sign and gain in IQR units) are compared with what actually deviated.
    Two things are combined, geometrically, because either one alone is
    misleading:

    * **coverage** — how much of the fault's *local* signature is reproduced
      (else a fault that touches this part only weakly scores perfectly);
    * **share** — how much of the fault's *whole* signature is visible here
      (else a fault whose main story is in another part wins this part's vote).
    """
    from bhel_optimize.simulation import faults as F

    def magnitude(effect) -> float:
        # a frozen sensor is a full-severity fault even though its "gain" is 0
        return 1.0 if effect.kind == "stuck" else abs(effect.gain)

    matches: list[dict] = []
    for key, fault in F.FAULTS.items():
        local_effects = [e for e in fault.effects if C.part_of(e.tag) == part_key]
        if not local_effects:
            continue
        local_total = sum(magnitude(e) for e in local_effects) or 1.0
        global_total = sum(magnitude(e) for e in fault.effects) or 1.0
        matched = 0.0
        for effect in local_effects:
            if effect.kind == "stuck":
                # shows up as a collapsed spread rather than a deviation
                matched += 0.5 * magnitude(effect)
                continue
            deviation = deviations.get(effect.tag)
            if deviation is None or not np.isfinite(deviation) or abs(deviation) < 0.2:
                continue
            if np.sign(deviation) != np.sign(effect.gain):
                continue
            agreement = min(1.0, abs(deviation) / max(abs(effect.gain), 0.2))
            matched += magnitude(effect) * agreement
        coverage = matched / local_total
        share = matched / global_total
        confidence = float(np.sqrt(max(0.0, coverage) * max(0.0, share)))
        if confidence >= min_confidence:
            matches.append(
                {
                    "fault_key": key,
                    "name": fault.name,
                    "category": fault.category,
                    "confidence": round(float(confidence), 3),
                }
            )
    return sorted(matches, key=lambda item: -item["confidence"])
