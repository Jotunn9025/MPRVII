"""Detection metrics: AUROC, precision/recall/F1, false-alarm rate, latency."""

from __future__ import annotations

import numpy as np


def _valid(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype="float64")
    mask = ~np.isnan(scores)
    return labels[mask].astype(bool), scores[mask]


def _valid_decisions(labels: np.ndarray, decisions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    decisions = np.asarray(decisions).astype(bool)
    mask = ~np.isnan(np.asarray(decisions, dtype="float64"))
    mask &= ~np.isnan(np.asarray(labels, dtype="float64"))
    return labels[mask].astype(bool), decisions[mask]


def smooth_series(scores: np.ndarray, window: int = 12) -> np.ndarray:
    """NaN-aware centred moving average (alarm smoothing).

    Industrial alarms are raised on *sustained* exceedance, not single-sample
    spikes; smoothing also removes most single-sample false alarms. Edge
    samples inside the first/last half-window stay NaN so no partially filled
    window is scored.
    """
    scores = np.asarray(scores, dtype="float64")
    if window <= 1:
        return scores.copy()
    valid = ~np.isnan(scores)
    out = np.full_like(scores, np.nan)
    if valid.sum() < window:
        return out
    filled = np.where(valid, scores, 0.0)
    counts = np.convolve(valid.astype("float64"), np.ones(window), mode="same")
    totals = np.convolve(filled, np.ones(window), mode="same")
    half = window // 2
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = totals / counts
    smoothed[counts < window] = np.nan
    out[half : len(out) - half] = smoothed[half : len(out) - half]
    return out


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve; ``nan`` when only one class is present."""
    from sklearn.metrics import roc_auc_score

    y, s = _valid(labels, scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auroc_ci(
    labels: np.ndarray, scores: np.ndarray, n_boot: int = 200, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap confidence interval for AUROC.

    With only ~100-200 positive samples per part, a single AUROC number has a
    wide error bar (two training configs differed by >0.1 on the same part), so
    the interval is reported alongside the point estimate instead of pretending
    the estimate is precise.
    """
    from sklearn.metrics import roc_auc_score

    y, s = _valid(labels, scores)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(float(roc_auc_score(y[idx], s[idx])))
    if not values:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def confusion(labels: np.ndarray, decisions: np.ndarray) -> dict:
    """Confusion-matrix counts and derived rates."""
    y, d = _valid_decisions(labels, decisions)
    tp = int(np.sum(y & d))
    fp = int(np.sum(~y & d))
    fn = int(np.sum(y & ~d))
    tn = int(np.sum(~y & ~d))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "false_alarm_rate": far,
    }


def best_f1(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Best achievable F1 over all thresholds (threshold-free quality check)."""
    from sklearn.metrics import precision_recall_curve

    y, s = _valid(labels, scores)
    if len(np.unique(y)) < 2:
        return {
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": float("nan"),
        }
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    idx = int(np.nanargmax(f1))
    threshold = float(thresholds[min(idx, len(thresholds) - 1)]) if len(thresholds) else float("nan")
    return {
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "threshold": threshold,
    }


def mean_detection_latency(labels: np.ndarray, decisions: np.ndarray) -> float:
    """Average samples from each fault-run onset to its first detection."""
    y, d = _valid_decisions(labels, decisions)
    latencies: list[int] = []
    index = 0
    while index < len(y):
        if not y[index]:
            index += 1
            continue
        start = index
        while index < len(y) and y[index]:
            index += 1
        hits = np.flatnonzero(d[start:index])
        if hits.size:
            latencies.append(int(hits[0]))
    return float(np.mean(latencies)) if latencies else float("nan")


def detection_rate(labels: np.ndarray, decisions: np.ndarray) -> float:
    """Fraction of fault runs that are detected at least once."""
    y, d = _valid_decisions(labels, decisions)
    runs = 0
    hits = 0
    index = 0
    while index < len(y):
        if not y[index]:
            index += 1
            continue
        start = index
        while index < len(y) and y[index]:
            index += 1
        runs += 1
        hits += int(bool(np.any(d[start:index])))
    return (hits / runs) if runs else float("nan")


SEVERITY_BANDS: tuple[tuple[float, float], ...] = (
    (0.1, 0.5),
    (0.5, 1.0),
    (1.0, float("inf")),
)


def recall_by_severity(
    severity: np.ndarray,
    decisions: np.ndarray,
    scores: np.ndarray | None = None,
    bands: tuple[tuple[float, float], ...] = SEVERITY_BANDS,
) -> dict:
    """Detection rate per injected-severity band (IQR units).

    Creeping faults are only detectable once they exceed the noise floor, so a
    single recall number hides the interesting behaviour. Bands make the
    detectability limit explicit instead of tuning labels.
    """
    severity = np.asarray(severity, dtype="float64")
    decisions = np.asarray(decisions).astype(bool)
    if scores is not None:
        valid = ~np.isnan(np.asarray(scores, dtype="float64"))
        severity, decisions = severity[valid], decisions[valid]
    out: dict[str, float] = {}
    for low, high in bands:
        mask = (severity >= low) & (severity < high) if high != float("inf") else severity >= low
        out[f"recall_{low:g}_{'inf' if high == float('inf') else format(high, 'g')}_iqr"] = (
            float(decisions[mask].mean()) if mask.any() else float("nan")
        )
        out[f"n_{low:g}_{'inf' if high == float('inf') else format(high, 'g')}_iqr"] = int(mask.sum())
    return out


def false_alarm_rate_clean(
    severity: np.ndarray, decisions: np.ndarray, scores: np.ndarray | None = None
) -> float:
    """FAR measured only on completely clean samples (zero injection)."""
    severity = np.asarray(severity, dtype="float64")
    decisions = np.asarray(decisions).astype(bool)
    if scores is not None:
        valid = ~np.isnan(np.asarray(scores, dtype="float64"))
        severity, decisions = severity[valid], decisions[valid]
    clean = severity <= 0.0
    return float(decisions[clean].mean()) if clean.any() else float("nan")


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray | None = None,
) -> dict:
    """Full metric bundle for one detector/ensemble on one part."""
    result = {
        "auroc": auroc(labels, scores),
        "best_f1": best_f1(labels, scores)["f1"],
    }
    if decisions is not None:
        result.update(confusion(labels, decisions))
        result["detection_rate"] = detection_rate(labels, decisions)
        # latency is only meaningful in seconds: caller converts via interval
        result["latency_samples"] = mean_detection_latency(labels, decisions)
    return result


def latency_to_seconds(samples: float, interval_s: float) -> float:
    """Convert a latency in samples to seconds."""
    if samples is None or np.isnan(samples):
        return float("nan")
    return float(samples) * interval_s
