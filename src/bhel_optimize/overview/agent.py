"""The Overview Agent: turns per-part detection scores into a situation report.

Input  : live sensor values from the simulation layer + the fitted per-part
         ensembles from the detection layer.
Output : a JSON-ready overview — plant level, per-part levels/trends, the tags
         that are actually deviating, a probable fault pattern, and short
         human-readable explanations.

This is the Phase-1 (template-based) version: the strings are generated from
measured numbers, not by a language model. The micro-LM that will eventually
write advice is Phase 2 (see ``plan.md``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bhel_optimize import config as C
from bhel_optimize.data import chronological_split, fit_scalers, load_scalers, save_scalers, transform_tags
from bhel_optimize.detection.ensembles import PartEnsemble, ensemble_dir, load_part_ensemble
from bhel_optimize.detection.windows import augment_with_differences, build_windows, part_tags
from bhel_optimize.overview.situation import (
    ALERT,
    LEVEL_NAMES,
    NOMINAL,
    WATCH,
    LevelBands,
    PartState,
    apply_persistence,
    criticality_escalated,
    match_fault_patterns,
    plant_level,
    step_level,
    top_deviating_tags,
    trend_for_history,
)

SCALERS_ARTIFACT = C.ARTIFACTS_DIR / "detection" / "scalers.joblib"


@dataclass
class OverviewAgent:
    """Scores every part on the current sample and reports the situation."""

    ensembles: dict[str, PartEnsemble]
    scalers: dict
    baselines: dict[str, tuple[float, float]]  # tag -> (median, IQR) of normal data
    states: dict[str, PartState] = field(default_factory=dict)
    bands: dict[str, LevelBands] = field(default_factory=dict)
    interval_s: float = C.SAMPLE_INTERVAL_S
    history_limit: int = 240  # samples of per-part score kept for the UI
    #: windows sampled per part when measuring the clean-data score distribution
    band_sample_windows: int = 1500

    def __post_init__(self) -> None:
        for key, ensemble in self.ensembles.items():
            self.states.setdefault(key, PartState(part_key=key))
            self.bands.setdefault(
                key, LevelBands.from_threshold(float(ensemble.threshold))
            )

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        frame: pd.DataFrame | None = None,
        *,
        parts: tuple[str, ...] | None = None,
    ) -> "OverviewAgent":
        """Load fitted ensembles (+ scalers) and the normal-data baselines."""
        if frame is None:
            from bhel_optimize.data import load_dataset

            frame = load_dataset()
        splits = chronological_split(frame)
        if SCALERS_ARTIFACT.exists():
            scalers = load_scalers(SCALERS_ARTIFACT)
        else:  # deterministic fallback: same fit as training
            scalers = fit_scalers(splits["train"])
            save_scalers(scalers, SCALERS_ARTIFACT)

        keys = parts or tuple(
            key for key in C.part_keys() if (ensemble_dir(key) / "ensemble.json").exists()
        )
        if not keys:
            raise FileNotFoundError(
                "no fitted ensembles found — run "
                "`uv run python -m bhel_optimize.detection.train` first"
            )
        ensembles = {key: load_part_ensemble(key) for key in keys}
        baselines = cls._baselines(splits["train"], scalers)
        agent = cls(ensembles=ensembles, scalers=scalers, baselines=baselines)
        agent.calibrate_bands(splits["train"], scalers)
        return agent

    def calibrate_bands(self, train_frame: pd.DataFrame, scalers: dict) -> dict[str, LevelBands]:
        """Measure each part's score distribution on clean normal data.

        Level bands come from *measured* clean-data quantiles rather than a
        fixed fraction of the threshold: the latter reported WATCH on ordinary
        livestream data, which is exactly the kind of false alarm a monitoring
        UI must not show.
        """
        from bhel_optimize.data import steam_temp_labels

        normal_mask = (steam_temp_labels(train_frame).to_numpy() == 0)
        scaled = transform_tags(train_frame, scalers).reset_index(drop=True)
        normal = scaled[normal_mask]
        for key, ensemble in self.ensembles.items():
            tags = ensemble.tags
            frame, diff_names = augment_with_differences(normal, tags)
            values = frame[list(tags) + diff_names].to_numpy(dtype="float32")
            windows, _ = build_windows(values, ensemble.lookback)
            if len(windows) > self.band_sample_windows:
                idx = np.linspace(0, len(windows) - 1, self.band_sample_windows).astype(int)
                windows = windows[idx]
            clean_scores = np.asarray(ensemble.score(windows), dtype="float64")
            self.bands[key] = LevelBands.from_clean_scores(
                clean_scores, float(ensemble.threshold)
            )
        return self.bands

    @staticmethod
    def _baselines(scaled_train: pd.DataFrame, scalers: dict) -> dict[str, tuple[float, float]]:
        """Per-tag ``(median, IQR)`` in *scaled* units, on normal training data.

        Used to express how far a tag has moved in units of its own typical
        spread ("+1.4 IQR") — much more meaningful to an operator than a raw
        z-score.
        """
        scaled = transform_tags(scaled_train, scalers)
        out: dict[str, tuple[float, float]] = {}
        for tag in C.TAGS:
            column = scaled[tag]
            median = float(np.nanmedian(column))
            iqr = float(np.nanquantile(column, 0.75) - np.nanquantile(column, 0.25))
            out[tag] = (median, iqr if iqr > 0 else 1.0)
        return out

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _window_for(self, part_key: str, scaled_suffix: pd.DataFrame) -> np.ndarray:
        """Feature window (levels + differences) ending at the latest sample."""
        ensemble = self.ensembles[part_key]
        tags = ensemble.tags
        frame, diff_names = augment_with_differences(scaled_suffix, tags)
        values = frame[list(tags) + diff_names].to_numpy(dtype="float32")
        windows, _ = build_windows(values, ensemble.lookback)
        if len(windows) == 0:
            raise ValueError("not enough samples for a window")
        return windows[-1:]

    def part_scores(self, frame: pd.DataFrame, index: int) -> dict[str, float]:
        """Calibrated ensemble score for every part at ``index``.

        During warm-up (fewer samples than the longest lookback) every part
        reports NaN and therefore NOMINAL — a live monitor must not raise just
        because it was switched on a minute ago.
        """
        lookback = max(e.lookback for e in self.ensembles.values())
        if index + 1 < lookback + 1:
            return {key: float("nan") for key in self.ensembles}
        start = max(0, index - lookback)  # one extra row so diffs are real
        suffix = frame.iloc[start : index + 1]
        scaled = transform_tags(suffix, self.scalers).reset_index(drop=True)
        scores: dict[str, float] = {}
        for key, ensemble in self.ensembles.items():
            window = self._window_for(key, scaled)
            scores[key] = float(np.asarray(ensemble.score(window)).ravel()[-1])
        return scores

    def part_deviations(self, frame: pd.DataFrame, index: int) -> dict[str, dict[str, float]]:
        """Per-tag deviation from the normal baseline, in IQR units."""
        row = frame.iloc[index]
        out: dict[str, dict[str, float]] = {}
        for key, ensemble in self.ensembles.items():
            deviations: dict[str, float] = {}
            for tag in ensemble.tags:
                scaled_value = float(self.scalers[tag].transform([[row[tag]]])[0][0])
                median, iqr = self.baselines[tag]
                deviations[tag] = (scaled_value - median) / iqr
            out[key] = deviations
        return out

    # ------------------------------------------------------------------
    # situation
    # ------------------------------------------------------------------

    def update(self, frame: pd.DataFrame, index: int) -> dict:
        """Advance the agent by one sample and return the situation overview."""
        scores = self.part_scores(frame, index)
        deviations = self.part_deviations(frame, index)
        return self._build_overview(index, scores, deviations, frame)

    def reset(self) -> None:
        for state in self.states.values():
            state.reset()

    def _build_overview(
        self,
        index: int,
        scores: dict[str, float],
        deviations: dict[str, dict[str, float]],
        frame: pd.DataFrame,
    ) -> dict:
        parts: dict[str, dict] = {}
        for key, ensemble in self.ensembles.items():
            state = self.states.setdefault(key, PartState(part_key=key))
            score = scores[key]
            threshold = float(ensemble.threshold)

            bands = self.bands.get(key) or LevelBands.from_threshold(threshold)
            state.history.append(score)
            state.above_flags.append(bool(np.isfinite(score) and score >= threshold))
            state.watch_flags.append(bool(np.isfinite(score) and score >= bands.watch))

            instant = apply_persistence(bands.level_for(score), state)
            new_level = step_level(state.level, instant)
            if new_level != state.level:
                state.since_index = index
            state.level = new_level

            trend = trend_for_history(state.history, threshold)
            deviation_map = deviations[key]
            top_tags = top_deviating_tags(deviation_map, limit=3)
            pattern_matches = match_fault_patterns(key, deviation_map)
            duration_s = (
                (index - state.since_index) * self.interval_s
                if state.since_index is not None
                else 0.0
            )
            parts[key] = {
                "key": key,
                "name": C.PARTS[key].name,
                "criticality": C.PARTS[key].criticality,
                "level": LEVEL_NAMES[state.level],
                "level_index": state.level,
                "score": round(score, 4) if np.isfinite(score) else None,
                "threshold": round(threshold, 4),
                "headroom": (
                    round(score / threshold, 3) if threshold > 0 and np.isfinite(score) else None
                ),
                "trend": trend,
                "duration_s": duration_s,
                "severity_iqr": round(max((abs(v) for v in deviation_map.values()), default=0.0), 3),
                "top_tags": [
                    {
                        "tag": tag,
                        "description": C.TAG_INFO[tag],
                        "value": float(frame.iloc[index][tag]),
                        "deviation_iqr": round(float(deviation), 3),
                        "direction": "above" if deviation >= 0 else "below",
                    }
                    for tag, deviation in top_tags
                ],
                "pattern": pattern_matches[0] if pattern_matches else None,
                "pattern_candidates": pattern_matches[:3],
                "explanation": self._explain_part(
                    key, state.level, score, threshold, trend, top_tags, deviation_map,
                    pattern_matches[0] if pattern_matches else None, duration_s,
                ),
            }

        levels = {key: info["level_index"] for key, info in parts.items()}
        held = {key: info["duration_s"] for key, info in parts.items()}
        boiler = plant_level(levels, held)
        escalated = criticality_escalated(levels, held)
        counts = {
            name: sum(1 for info in parts.values() if info["level"] == name)
            for name in LEVEL_NAMES
        }
        attention = sorted(
            (info for info in parts.values() if info["level_index"] >= WATCH),
            key=lambda info: (-info["level_index"], -info["criticality"]),
        )
        return {
            "position": int(index),
            "timestamp": frame.iloc[index][C.DATE_COLUMN].isoformat(),
            "situation": {
                "level": LEVEL_NAMES[boiler],
                "level_index": boiler,
                "label": f"PLANT {LEVEL_NAMES[boiler]}",
                "counts": counts,
                "summary": self._plant_summary(boiler, attention, counts, escalated),
                "worst_parts": [info["key"] for info in attention[:3]],
                "criticality_escalated": escalated,
            },
            "parts": parts,
            "attention": [
                {
                    "key": info["key"],
                    "name": info["name"],
                    "level": info["level"],
                    "criticality": info["criticality"],
                    "explanation": info["explanation"],
                }
                for info in attention
            ],
        }

    # ------------------------------------------------------------------
    # language (template-based; the micro-LM replaces this in Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _explain_part(
        part_key: str,
        level: int,
        score: float,
        threshold: float,
        trend: str,
        top_tags: list[tuple[str, float]],
        deviations: dict[str, float],
        pattern: dict | None,
        duration_s: float,
    ) -> str:
        name = C.PARTS[part_key].name
        if level == NOMINAL and not top_tags:
            return f"{name}: nominal, no tag deviating beyond its normal spread."
        tag_text = ", ".join(
            f"{C.TAG_INFO[tag]} {value:+.1f} IQR" for tag, value in top_tags
        ) or "no significant deviation"
        trend_text = {"rising": "worsening", "falling": "improving", "steady": "steady"}[trend]
        head = (
            f"{name}: {LEVEL_NAMES[level]} (score {score:.2f} vs threshold {threshold:.2f},"
            f" {trend_text}"
            + (f" for {duration_s / 60:.0f} min" if duration_s >= 60 else "")
            + ")."
        )
        body = f" Most deviating tags — {tag_text}."
        if pattern:
            body += (
                f" Pattern matches {pattern['name']}"
                f" ({pattern['category']}, confidence {pattern['confidence']:.2f})."
            )
        if level >= ALERT:
            body += " Recommend operator review."
        elif level == WATCH:
            body += " Trend worth watching."
        return head + body

    @staticmethod
    def _plant_summary(
        boiler: int, attention: list[dict], counts: dict, escalated: list[str] | None = None
    ) -> str:
        if boiler == NOMINAL:
            note = ""
            if escalated:
                names = ", ".join(C.PARTS[key].name for key in escalated[:2])
                note = f" Sustained watch on safety-critical part(s): {names}."
            return (
                "Plant nominal: all monitored parts within their normal operating"
                " envelope." + note
            )
        names = ", ".join(f"{info['name']} ({info['level']})" for info in attention[:3])
        extra = (
            f" +{len(attention) - 3} more" if len(attention) > 3 else ""
        )
        top = attention[0]
        pattern = top.get("pattern")
        pattern_text = f" Most consistent with {pattern['name']}." if pattern else ""
        return (
            f"PLANT {LEVEL_NAMES[boiler]}: {counts['CRITICAL']} critical,"
            f" {counts['ALERT']} alert, {counts['WATCH']} watch. Attention —"
            f" {names}{extra}.{pattern_text}"
        )

    # ------------------------------------------------------------------
    # payload helpers
    # ------------------------------------------------------------------

    def to_json(self, overview: dict) -> str:
        """Serialise an overview payload."""
        return json.dumps(overview, default=str)

    def save(self, path: Path | None = None) -> Path:
        """Persist baselines and level bands used for the situation roll-up."""
        path = path or (C.ARTIFACTS_DIR / "detection" / "overview_baselines.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "baselines": {tag: list(v) for tag, v in self.baselines.items()},
                    "bands": {
                        key: {
                            "watch": band.watch,
                            "alert": band.alert,
                            "critical": band.critical,
                        }
                        for key, band in self.bands.items()
                    },
                    "parts": sorted(self.ensembles),
                    "interval_s": self.interval_s,
                },
                indent=2,
            )
        )
        return path
