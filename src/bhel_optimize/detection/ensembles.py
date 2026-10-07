"""Per-part hybrid ensemble: score normalisation, fusion, thresholds, storage.

Design (user-approved hybrid strategy)
--------------------------------------
1. Each candidate detector is trained on **normal data only** (unsupervised),
   so no fault labels are needed to train.
2. Raw detector scores are not comparable, and their distributions have very
   heavy tails (a z-score detector on a near-constant tag can explode). Scores
   are therefore mapped to **percentiles of their calibration-normal
   distribution** — bounded in ``[0, 1]``, monotone, and immune to one
   pathological detector dominating the fusion.
3. Fusion weights come from how well each detector separates normal from
   injected faults on the validation scenario series (AUROC-based — the
   model-weighted-integration idea from the dataset's source paper).
4. If one detector clearly dominates, the ensemble **falls back to that single
   model** rather than diluting it with weaker voters.
5. The decision threshold is a percentile of the *normal* calibration score
   distribution, so the false-alarm rate is a chosen input, not an accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bhel_optimize import config as C
from bhel_optimize.detection import detectors as D
from bhel_optimize.detection.windows import part_tags

EPS = 1e-9
N_KNOTS = 201

#: Robust z-scores are clipped here so a single pathological detector cannot
#: dominate the fusion, while the tail keeps its ordering (a percentile map
#: would saturate everything above the calibration maximum at 1.0 and destroy
#: resolution exactly where detection happens).
Z_CLIP = 20.0

#: All detection artifacts live under ``artifacts/detection/``.
DETECTION_DIR = C.ARTIFACTS_DIR / "detection"
MODELS_SUBDIR = DETECTION_DIR / "models"

#: A detector this much better than the runner-up counts as dominant.
SINGLE_MODEL_MARGIN = 0.05
#: Detectors this far below the best contribute no weight (they only add noise).
MIN_RELATIVE_AUROC = 0.85


def percentile_knots(scores: np.ndarray, n_knots: int = N_KNOTS) -> list[float]:
    """Quantile knots summarising a score distribution (for percentile mapping)."""
    scores = np.asarray(scores, dtype="float64")
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return [0.0, 1.0]
    return [float(v) for v in np.quantile(scores, np.linspace(0.0, 1.0, n_knots))]


def percentile_transform(scores: np.ndarray, knots: list[float]) -> np.ndarray:
    """Map raw scores to their percentile within the stored calibration law."""
    probs = np.linspace(0.0, 1.0, len(knots))
    return np.interp(np.asarray(scores, dtype="float64"), np.asarray(knots), probs)


def normalize_scores(
    scores: np.ndarray, reference: np.ndarray, clip: float = Z_CLIP
) -> np.ndarray:
    """Robust median/IQR z-score, clipped to ``[-clip, +clip]``."""
    median = float(np.nanmedian(reference))
    iqr = float(np.nanquantile(reference, 0.75) - np.nanquantile(reference, 0.25))
    z = (np.asarray(scores, dtype="float64") - median) / (iqr + EPS)
    return np.clip(z, -clip, clip)


def reference_stats(scores: np.ndarray) -> tuple[float, float]:
    """``(median, IQR)`` of a score distribution (diagnostics, NaN-safe)."""
    scores = np.asarray(scores, dtype="float64")
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.median(scores)),
        float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25)),
    )


def weights_from_auroc(aucs: dict[str, float]) -> dict[str, float]:
    """Turn per-detector AUROCs into fusion weights (drop weak/no-signal ones)."""
    if not aucs:
        return {}
    best = max(aucs.values())
    if best <= 0.5:
        return {key: 1.0 / len(aucs) for key in aucs}
    usable = {
        key: max(0.0, value - 0.5)
        for key, value in aucs.items()
        if value >= max(0.5, best * MIN_RELATIVE_AUROC)
    }
    total = sum(usable.values())
    if total <= 0:
        return {key: 1.0 / len(aucs) for key in aucs}
    return {key: value / total for key, value in usable.items()}


def fuse(normalized: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of normalised detector scores."""
    if not normalized:
        raise ValueError("no scores to fuse")
    keys = [k for k in normalized if k in weights]
    if not keys:  # pragma: no cover - defensive
        raise ValueError("no overlapping weights")
    total = sum(weights[k] for k in keys) or 1.0
    reference = next(iter(normalized.values()))
    stacked = np.zeros(np.shape(reference), dtype="float64")
    for key in keys:
        stacked += normalized[key] * (weights[key] / total)
    return stacked


def dominant_detector(aucs: dict[str, float], margin: float = SINGLE_MODEL_MARGIN) -> str | None:
    """Return the clearly-dominant detector key, or ``None`` to use fusion."""
    if len(aucs) < 2:
        return next(iter(aucs), None)
    ordered = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)
    (best_key, best_auc), (_, second_auc) = ordered[0], ordered[1]
    if best_auc - second_auc >= margin:
        return best_key
    return None


def calibrate_threshold(normal_scores: np.ndarray, false_alarm_rate: float = 0.01) -> float:
    """Percentile threshold achieving the requested false-alarm rate.

    NaN-safe: smoothed score series leave NaN at their edges, and a single NaN
    would otherwise make ``np.quantile`` return NaN.
    """
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false_alarm_rate must be in (0, 1)")
    scores = np.asarray(normal_scores, dtype="float64")
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return float("nan")
    return float(np.quantile(scores, 1.0 - false_alarm_rate))


@dataclass
class PartEnsemble:
    """Fitted ensemble for one boiler part."""

    part_key: str
    lookback: int
    tags: tuple[str, ...]
    detectors: dict[str, D.BaseDetector] = field(default_factory=dict)
    references: dict[str, tuple[float, float]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    mode: str = "fused"  # fused | single
    best_key: str | None = None
    threshold: float = 0.0
    false_alarm_rate: float = 0.01
    aurocs: dict[str, float] = field(default_factory=dict)

    # -- scoring ---------------------------------------------------------
    def normalized_scores(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        """Clipped robust z-scores per detector (comparable, tail-preserving)."""
        out: dict[str, np.ndarray] = {}
        for key, detector in self.detectors.items():
            raw = np.asarray(detector.score(windows), dtype="float64")
            out[key] = _normalise_with(raw, self.references[key])
        return out

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Ensemble score per window (higher == more anomalous)."""
        if not self.detectors:
            raise RuntimeError("ensemble has no detectors")
        normalised = self.normalized_scores(windows)
        if self.mode == "single" and self.best_key in normalised:
            return normalised[self.best_key]
        return fuse(normalised, self.weights)

    def decide(self, windows: np.ndarray) -> np.ndarray:
        """Boolean decisions at the calibrated threshold."""
        return self.score(windows) >= self.threshold

    # -- persistence -----------------------------------------------------
    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for detector in self.detectors.values():
            detector.save(directory)
        path = directory / "ensemble.json"
        path.write_text(json.dumps(self.metadata(), indent=2))
        return path

    def load(self, directory) -> "PartEnsemble":
        directory = Path(directory)
        meta = json.loads((directory / "ensemble.json").read_text())
        for key, detector in self.detectors.items():
            detector.load(directory)
        self.references = {k: tuple(v) for k, v in meta["references"].items()}
        self.weights = meta["weights"]
        self.threshold = meta["threshold"]
        self.mode = meta["mode"]
        self.best_key = meta["best_key"]
        self.false_alarm_rate = meta["false_alarm_rate"]
        self.aurocs = meta["aurocs"]
        return self

    def metadata(self) -> dict:
        return {
            "part_key": self.part_key,
            "lookback": self.lookback,
            "tags": list(self.tags),
            "detectors": sorted(self.detectors),
            "references": {k: list(v) for k, v in self.references.items()},
            "weights": self.weights,
            "mode": self.mode,
            "best_key": self.best_key,
            "threshold": self.threshold,
            "false_alarm_rate": self.false_alarm_rate,
            "aurocs": self.aurocs,
        }


def _normalise_with(scores: np.ndarray, reference: tuple[float, float]) -> np.ndarray:
    """Clipped robust z-score using a stored ``(median, IQR)`` reference."""
    median, iqr = reference
    z = (np.asarray(scores, dtype="float64") - median) / (iqr + EPS)
    return np.clip(z, -Z_CLIP, Z_CLIP)


def base_ensemble(part_key: str, lookback: int) -> PartEnsemble:
    """Empty ensemble wired to a part's tags."""
    return PartEnsemble(part_key=part_key, lookback=lookback, tags=part_tags(part_key))


def ensemble_dir(part_key: str) -> Path:
    """Artifact directory for a part's ensemble."""
    return MODELS_SUBDIR / part_key


def load_part_ensemble(part_key: str, detector_keys: tuple[str, ...] | None = None) -> PartEnsemble:
    """Rebuild a saved ensemble for inference (used by the UI/overview layers)."""
    directory = ensemble_dir(part_key)
    meta = json.loads((directory / "ensemble.json").read_text())
    lookback = int(meta["lookback"])
    tags = tuple(meta["tags"])
    ensemble = base_ensemble(part_key, lookback)
    for key in (detector_keys or tuple(meta["detectors"])):
        ensemble.detectors[key] = D.build_detector(key, lookback, len(tags))
    ensemble.load(directory)
    return ensemble
