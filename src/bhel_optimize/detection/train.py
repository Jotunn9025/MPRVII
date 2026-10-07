"""Train per-part hybrid ensembles and build the accuracy leaderboard.

Usage::

    uv run python -m bhel_optimize.detection.train                 # all parts
    uv run python -m bhel_optimize.detection.train --parts furnace --epochs 5
    uv run python -m bhel_optimize.detection.train --include-heavy

Artifacts:
- ``artifacts/detection/models/<part>/<detector>.keras`` (+ ``ensemble.json``)
- ``artifacts/detection/leaderboard.json``
- ``.agents/leaderboard.md`` (generated report)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bhel_optimize import config as C
from bhel_optimize.data import (
    chronological_split,
    fit_scalers,
    save_scalers,
    steam_temp_labels,
    transform_tags,
)
from bhel_optimize.detection import detectors as D
from bhel_optimize.detection.data import (
    build_eval_series,
    build_repeated_eval_series,
    part_label_alignment,
    part_scenario_ceiling,
    scenarios_for_part,
    severity_alignment,
)
from bhel_optimize.detection.ensembles import (
    DETECTION_DIR,
    _normalise_with,
    PartEnsemble,
    base_ensemble,
    calibrate_threshold,
    dominant_detector,
    ensemble_dir,
    fuse,
    reference_stats,
    weights_from_auroc,
)
from bhel_optimize.detection.evaluate import (
    auroc,
    auroc_ci,
    best_f1,
    evaluate_scores,
    false_alarm_rate_clean,
    recall_by_severity,
    smooth_series,
)
from bhel_optimize.detection.windows import (
    augment_with_differences,
    build_windows,
    frame_values,
    part_tags,
    scatter_scores,
    subsample_windows,
)

ARTIFACT_ROOT = DETECTION_DIR
LEADERBOARD_JSON = ARTIFACT_ROOT / "leaderboard.json"
LEADERBOARD_MD = C.PROJECT_ROOT / ".agents" / "leaderboard.md"


@dataclass
class TrainConfig:
    """Knobs for a detection training run."""

    lookback: int = 60  # 5 minutes at 5 s sampling
    epochs: int = 20
    batch_size: int = 256
    max_windows: int = 8000
    false_alarm_rate: float = 0.05  # false-alarm budget on clean samples
    include_heavy: bool = False
    parts: tuple[str, ...] | None = None
    scenario_limit: int = 6
    #: alarm smoothing window in samples (12 = 1 minute at 5 s sampling)
    smooth_window: int = 12
    #: how many start offsets to pool per split (stabilises AUROC estimates)
    eval_repeats: int = 4
    block: int = 240
    gap: int = 120
    seed: int = 0
    #: "f1" = pick the accuracy-maximising threshold on the validation scenario
    #: series (reported FAR is then whatever that threshold achieves);
    #: "target-far" = pick the threshold for the requested false-alarm rate.
    threshold_mode: str = "f1"
    detector_keys: tuple[str, ...] = field(default_factory=D.detector_keys)


def _fit_detectors_only(
    part_key: str, scalers: dict, cfg: TrainConfig, splits: dict
) -> PartEnsemble:
    """Train detectors for a part that has no scenarios (no metrics)."""
    tags = part_tags(part_key)
    train_values, _ = part_features(transform_tags(splits["train"], scalers), tags)
    windows, _ = build_windows(train_values, cfg.lookback)
    windows, _ = subsample_windows(windows, np.arange(len(windows)), cfg.max_windows)
    ensemble = base_ensemble(part_key, cfg.lookback)
    for key in cfg.detector_keys:
        detector = D.build_detector(
            key,
            cfg.lookback,
            len(tags) * 2,
            seed=cfg.seed,
            **(
                {"epochs": cfg.epochs, "batch_size": cfg.batch_size}
                if key in {"dense_ae", "lstm_residual", "tcn_ae", "transformer_residual"}
                else {}
            ),
        )
        detector.fit(windows)
        ensemble.detectors[key] = detector
    return ensemble


def _masked(scores: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Set excluded samples to NaN so the metric helpers drop them."""
    out = np.asarray(scores, dtype="float64").copy()
    out[exclude] = np.nan
    return out


def _build_cfg(cfg: TrainConfig) -> None:
    cfg.detector_keys = D.detector_keys(include_heavy=cfg.include_heavy)


def part_features(scaled: pd.DataFrame, tags: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for a part: tag levels plus their first differences."""
    frame, diff_names = augment_with_differences(scaled, tags)
    feature_names = list(tags) + diff_names
    return frame_values(frame, feature_names), feature_names


def part_windows(scaled: pd.DataFrame, part_key: str, cfg: TrainConfig) -> np.ndarray:
    """Sliding windows for one part over a scaled frame (levels + differences)."""
    values, _ = part_features(scaled, part_tags(part_key))
    windows, _ = build_windows(values, cfg.lookback)
    return windows


def prepare_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological 70/15/15 split."""
    return chronological_split(frame)


def normal_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows the dataset itself marks as normal (steam temperature in band)."""
    labels = steam_temp_labels(frame)
    return frame.loc[labels == 0].reset_index(drop=True)


def train_part(
    part_key: str,
    frame: pd.DataFrame,
    scalers: dict,
    cfg: TrainConfig,
) -> tuple[PartEnsemble, dict]:
    """Fit, weight, calibrate and evaluate every detector for one part."""
    splits = prepare_splits(frame)
    tags = part_tags(part_key)

    scaled_train_all = transform_tags(splits["train"], scalers)
    scaled_val_all = transform_tags(splits["val"], scalers)
    scaled_test = transform_tags(splits["test"], scalers)

    # 1. train on normal data only (mask taken from the UNSCALED steam-temp rule)
    train_normal_mask = (steam_temp_labels(splits["train"]).to_numpy() == 0)
    train_normal = scaled_train_all[train_normal_mask]
    train_windows = part_windows(train_normal, part_key, cfg)
    train_windows, _ = subsample_windows(train_windows, np.arange(len(train_windows)), cfg.max_windows)

    # 2. calibration windows: normal validation data only (kept with the frame
    #    so calibration scores can be smoothed on the same sample grid)
    val_normal_mask = (steam_temp_labels(splits["val"]).to_numpy() == 0)
    calibration_frame = scaled_val_all[val_normal_mask].reset_index(drop=True)
    calibration_values, feature_names = part_features(calibration_frame, tags)
    calibration_windows, calibration_ends = build_windows(calibration_values, cfg.lookback)

    # 3. validation scenario series (weights) and test scenario series (final)
    scenario_keys = scenarios_for_part(part_key, limit=cfg.scenario_limit)
    val_series = build_repeated_eval_series(
        splits["val"], scenario_keys, repeats=cfg.eval_repeats, block=cfg.block, gap=cfg.gap
    )
    test_series = build_repeated_eval_series(
        splits["test"], scenario_keys, repeats=cfg.eval_repeats, block=cfg.block, gap=cfg.gap
    )
    if len(val_series) == 0 or len(test_series) == 0:
        # No simulated fault touches this part at all: keep the trained
        # detectors but record that per-part accuracy cannot be measured.
        report["evaluable"] = False
        report["evaluation_note"] = (
            "not scored: no simulated fault affects this part's tags"
        )
        return _fit_detectors_only(part_key, scalers, cfg, splits), report

    val_scaled = transform_tags(val_series.values, scalers)
    test_scaled = transform_tags(test_series.values, scalers)

    val_values, _ = part_features(val_scaled, tags)
    test_values, _ = part_features(test_scaled, tags)
    val_windows, val_ends = build_windows(val_values, cfg.lookback)
    test_windows, test_ends = build_windows(test_values, cfg.lookback)
    # Ground truth: positive = any material injection (>= 0.1 IQR). Severity is
    # kept so recall can be reported per severity band (the honest way to score
    # creeping faults).
    val_labels = part_label_alignment(val_series.labels, part_key).to_numpy()
    test_labels = part_label_alignment(test_series.labels, part_key).to_numpy()
    val_severity = severity_alignment(val_series.labels, part_key)
    test_severity = severity_alignment(test_series.labels, part_key)
    scenario_ceiling = part_scenario_ceiling(part_key, scenario_keys)
    # Rows the dataset itself marks "tailed" are real global anomalies, not
    # part faults: exclude them from per-part metrics (covered by the global
    # check instead) so we do not score the detector against unlabelled faults.
    # Samples with a negligible injection (0 < severity < 0.1 IQR) are excluded
    # too: they are neither clean nor fairly detectable.
    val_exclude = val_series.labels["tailed_real"].to_numpy().astype(bool) | (
        (val_severity > 0) & (val_severity < 0.1)
    )
    test_exclude = test_series.labels["tailed_real"].to_numpy().astype(bool) | (
        (test_severity > 0) & (test_severity < 0.1)
    )

    # 4. fit detectors + score every stage
    ensemble = base_ensemble(part_key, cfg.lookback)
    report: dict = {
        "part_key": part_key,
        "part_name": C.PARTS[part_key].name,
        "criticality": C.PARTS[part_key].criticality,
        "tags": list(tags),
        "n_features": len(feature_names),
        "lookback": cfg.lookback,
        "scenarios": scenario_keys,
        "n_train_windows": int(len(train_windows)),
        "scenario_ceiling_iqr": scenario_ceiling,
        "evaluable": bool(
            len(test_series) and test_labels.any() and (~test_labels).any()
        ),
        "detectors": [],
    }

    val_raw: dict[str, np.ndarray] = {}
    calibration_raw: dict[str, np.ndarray] = {}
    test_raw: dict[str, np.ndarray] = {}
    for key in cfg.detector_keys:
        started = time.time()
        detector = D.build_detector(
            key,
            cfg.lookback,
            len(feature_names),
            seed=cfg.seed,
            **(
                {"epochs": cfg.epochs, "batch_size": cfg.batch_size}
                if key in {"dense_ae", "lstm_residual", "tcn_ae", "transformer_residual"}
                else {}
            ),
        )
        detector.fit(train_windows)
        ensemble.detectors[key] = detector
        calibration_raw[key] = np.asarray(detector.score(calibration_windows), dtype="float64")
        val_raw[key] = np.asarray(detector.score(val_windows), dtype="float64")
        test_raw[key] = np.asarray(detector.score(test_windows), dtype="float64")
        report["detectors"].append(
            {
                "key": key,
                "name": detector.name,
                "kind": detector.kind,
                "fit_seconds": round(time.time() - started, 2),
            }
        )

    # 5. normalise every detector against the normal-calibration distribution
    for key, raw in calibration_raw.items():
        ensemble.references[key] = reference_stats(raw)

    def normalise(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            key: _normalise_with(values, ensemble.references[key])
            for key, values in raw.items()
        }

    val_norm = normalise(val_raw)
    test_norm = normalise(test_raw)
    calibration_norm = normalise(calibration_raw)

    # 6. AUROC on the validation scenario series -> weights / single-model mode
    def sample_scores(norm: dict, ends, n_samples: int, exclude: np.ndarray) -> dict:
        return {
            key: smooth_series(
                _masked(scatter_scores(ends, scores, n_samples), exclude),
                cfg.smooth_window,
            )
            for key, scores in norm.items()
        }

    val_samples = sample_scores(val_norm, val_ends, len(val_series), val_exclude)
    test_samples = sample_scores(test_norm, test_ends, len(test_series), test_exclude)

    val_aucs = {key: auroc(val_labels, scores) for key, scores in val_samples.items()}
    ensemble.aurocs = {k: float(v) for k, v in val_aucs.items()}
    ensemble.weights = weights_from_auroc({k: v for k, v in val_aucs.items() if not np.isnan(v)})

    # Candidate selection on the validation scenario series. The fused ensemble
    # competes with every individual detector, and fusion is only used when it
    # actually wins — otherwise weak members dilute the best detector.
    val_ensemble_auroc = auroc(val_labels, fuse(val_samples, ensemble.weights))
    candidates = dict(val_aucs)
    candidates["ensemble"] = val_ensemble_auroc
    scored_candidates = {k: v for k, v in candidates.items() if not np.isnan(v)}
    selected_key = (
        max(scored_candidates, key=scored_candidates.get) if scored_candidates else "ensemble"
    )
    best = None if selected_key == "ensemble" else selected_key
    ensemble.mode = "fused" if best is None else "single"
    ensemble.best_key = best
    report["val_ensemble_auroc"] = float(val_ensemble_auroc)

    # 7. threshold selection (cv = calibration-normal sample-level scores)
    cv_normal_samples = smooth_series(
        _masked(
            scatter_scores(
                calibration_ends,
                calibration_norm[best] if best else fuse(calibration_norm, ensemble.weights),
                len(calibration_frame),
            ),
            np.zeros(len(calibration_frame), dtype=bool),
        ),
        cfg.smooth_window,
    )
    val_sample_scores = val_samples[best] if best else fuse(val_samples, ensemble.weights)

    threshold_far = calibrate_threshold(cv_normal_samples, cfg.false_alarm_rate)
    threshold_f1 = float(best_f1(val_labels, val_sample_scores)["threshold"])
    if cfg.threshold_mode == "f1":
        # accuracy-optimal on the validation scenarios, but never more
        # permissive than the false-alarm budget allows
        ensemble.threshold = (
            threshold_far if np.isnan(threshold_f1) else max(threshold_f1, threshold_far)
        )
    elif cfg.threshold_mode == "target-far":
        ensemble.threshold = threshold_far
    else:
        raise ValueError(f"unknown threshold_mode: {cfg.threshold_mode}")
    ensemble.false_alarm_rate = cfg.false_alarm_rate

    # 8. final metrics on the held-out test scenario series
    for entry in report["detectors"]:
        key = entry["key"]
        sample_scores = test_samples[key]
        decisions = sample_scores >= ensemble.threshold
        entry.update(
            {
                "auroc": auroc(test_labels, sample_scores),
                "val_auroc": float(val_aucs[key]),
                "weight": float(ensemble.weights.get(key, 0.0)),
                **evaluate_scores(test_labels, sample_scores, decisions),
                "far_clean": false_alarm_rate_clean(test_severity, decisions, sample_scores),
                **recall_by_severity(test_severity, decisions, sample_scores),
            }
        )
        entry["latency_s"] = entry["latency_samples"] * C.SAMPLE_INTERVAL_S
        entry.pop("latency_samples", None)

    fused_sample = fuse(test_samples, ensemble.weights)
    ensemble_entry = {
        "key": "ensemble",
        "name": f"Ensemble ({'single: ' + best if best else 'weighted fusion'})",
        "kind": "ensemble",
        **evaluate_scores(test_labels, fused_sample, fused_sample >= ensemble.threshold),
        "far_clean": false_alarm_rate_clean(
            test_severity, fused_sample >= ensemble.threshold, fused_sample
        ),
        **recall_by_severity(test_severity, fused_sample >= ensemble.threshold, fused_sample),
    }
    ensemble_entry["latency_s"] = ensemble_entry["latency_samples"] * C.SAMPLE_INTERVAL_S
    ensemble_entry.pop("latency_samples", None)
    report["ensemble"] = ensemble_entry
    report["mode"] = ensemble.mode
    report["threshold_mode"] = cfg.threshold_mode
    report["smooth_window"] = cfg.smooth_window
    report["best_key"] = best
    report["threshold"] = ensemble.threshold
    report["threshold_far"] = threshold_far
    report["threshold_f1"] = threshold_f1
    report["weights"] = ensemble.weights
    report["selected"] = selected_key
    ensemble_entry["val_auroc"] = float(val_ensemble_auroc)
    selected_metrics = (
        ensemble_entry
        if selected_key == "ensemble"
        else next(e for e in report["detectors"] if e["key"] == selected_key)
    )
    selected_scores = (
        fused_sample if selected_key == "ensemble" else test_samples[selected_key]
    )
    report["selected_metrics"] = selected_metrics
    report["selected_auroc_ci"] = auroc_ci(test_labels, selected_scores)
    report["test_auroc_ci_all"] = {
        key: auroc_ci(test_labels, scores) for key, scores in test_samples.items()
    }
    report["n_val_positives"] = int(val_labels[~val_exclude].sum())
    report["excluded_real_tailed"] = {
        "val": int(val_exclude.sum()),
        "test": int(test_exclude.sum()),
    }
    report["n_test_positives"] = int(test_labels[~test_exclude].sum())
    report["tags_scored"] = list(tags)
    if not report["evaluable"]:
        report["evaluation_note"] = (
            "not scored: no simulated fault produces labelled positives for this"
            " part (no material scenario available)"
        )
    return ensemble, report


def global_tailed_report(ensemble: PartEnsemble, split: pd.DataFrame, scalers: dict) -> dict:
    """Score the real 'tailed' class (steam temperature rule) on held-out data."""
    scaled = transform_tags(split, scalers)
    values, _ = part_features(scaled, ensemble.tags)
    windows, ends = build_windows(values, ensemble.lookback)
    scores = scatter_scores(ends, ensemble.score(windows), len(split))
    labels = steam_temp_labels(split).to_numpy()
    decisions = scores >= ensemble.threshold
    metrics = evaluate_scores(labels, scores, decisions)
    metrics["latency_s"] = metrics.pop("latency_samples", float("nan")) * C.SAMPLE_INTERVAL_S
    metrics.update(
        {
            "positives": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "threshold": ensemble.threshold,
        }
    )
    return metrics


def train_all(cfg: TrainConfig, frame: pd.DataFrame | None = None) -> dict:
    """Train every requested part and assemble the leaderboard."""
    _build_cfg(cfg)
    if frame is None:
        from bhel_optimize.data import load_dataset

        frame = load_dataset()
    splits = prepare_splits(frame)
    scalers = fit_scalers(splits["train"])
    parts = cfg.parts or C.part_keys()

    leaderboard: dict = {
        "config": {
            "lookback": cfg.lookback,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "max_windows": cfg.max_windows,
            "smooth_window": cfg.smooth_window,
            "eval_repeats": cfg.eval_repeats,
            "false_alarm_rate": cfg.false_alarm_rate,
            "include_heavy": cfg.include_heavy,
            "detectors": list(cfg.detector_keys),
            "threshold_mode": cfg.threshold_mode,
            "parts": list(parts),
        },
        "parts": {},
    }

    # persist the train-fit scalers so inference (UI/overview) uses exactly the
    # same preprocessing the detectors were trained with
    save_scalers(scalers, ARTIFACT_ROOT / "scalers.joblib")

    for part_key in parts:
        started = time.time()
        ensemble, report = train_part(part_key, frame, scalers, cfg)
        if part_key == "main_steam":  # holds the state-discriminating tag
            report["global_tailed"] = global_tailed_report(ensemble, splits["test"], scalers)
        ensemble.save(ensemble_dir(part_key))
        report["total_seconds"] = round(time.time() - started, 1)
        leaderboard["parts"][part_key] = report

    return leaderboard


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def leaderboard_markdown(leaderboard: dict) -> str:
    """Human-readable leaderboard (generated; do not hand-edit)."""
    cfg = leaderboard["config"]
    lines: list[str] = []
    lines.append("# Onboarding Knowledge — Detection Leaderboard (M4)")
    lines.append("")
    lines.append(
        "> **Generated** by `uv run python -m bhel_optimize.detection.train` — do"
        " not hand-edit. Metrics measured on the held-out test scenario series"
        " (injected faults) unless stated otherwise."
    )
    lines.append("")
    lines.append(
        f"- Lookback: {cfg['lookback']} samples ({cfg['lookback'] * C.SAMPLE_INTERVAL_S} s)"
        f" | epochs: {cfg['epochs']} | max windows: {cfg['max_windows']}"
        f" | target false-alarm rate: {cfg['false_alarm_rate']:.1%}"
    )
    lines.append(f"- Candidate detectors: {', '.join(cfg['detectors'])}")
    lines.append(
        "- Accuracy caveat: the dataset has **no per-part fault labels**, so"
        " per-part accuracy is measured against **injected simulated faults**"
        " (plus the one real global label below)."
    )
    lines.append(
        "- Positives = samples with any material injection (>= 0.1 IQR); FAR is"
        " measured only on completely clean samples; recall is reported per"
        " severity band because creeping faults are only detectable once they"
        " exceed the noise floor. Real \'tailed\' rows and negligible injections"
        " are excluded from per-part metrics."
    )
    lines.append("")

    lines.append(
        "- AUROC confidence intervals are **within-series bootstrap** intervals:"
        " they capture sampling noise inside one evaluation series only. Running"
        " the trainer with a different block length moved some parts by >0.1"
        " AUROC (average across parts changed by <0.02), so treat part-level"
        " numbers as indicative of rank, not as precise constants."
    )
    lines.append("")
    lines.append("## Summary (chosen detector/ensemble per part)")
    lines.append("")
    lines.append("| Part | Crit | Chosen | val AUROC | test AUROC (95% CI) | F1 | Prec | Rec | FAR(clean) | Rec≥0.1 | Rec≥0.5 | Rec≥1.0 | Latency |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for part_key, report in leaderboard["parts"].items():
        chosen = report["selected_metrics"]
        lines.append(
            f"| {report['part_name']} | {report['criticality']} | `{report['selected']}` |"
            f" {_fmt(chosen.get('val_auroc'))} |"
            f" {_fmt(chosen['auroc'])} [{_fmt(report['selected_auroc_ci'][0], 2)}-{_fmt(report['selected_auroc_ci'][1], 2)}] |"
            f" {_fmt(chosen['f1'])} | {_fmt(chosen['precision'])} |"
            f" {_fmt(chosen['recall'])} | {_fmt(chosen.get('far_clean'))} |"
            f" {_fmt(chosen.get('recall_0.1_0.5_iqr'))} | {_fmt(chosen.get('recall_0.5_1_iqr'))} |"
            f" {_fmt(chosen.get('recall_1_inf_iqr'))} | {_fmt(chosen['latency_s'], 0)} s |"
        )
    lines.append("")

    lines.append("## Per-part detector comparison")
    for part_key, report in leaderboard["parts"].items():
        lines.append("")
        lines.append(
            f"### {report['part_name']} (`{part_key}`, criticality"
            f" {report['criticality']}, tags: {', '.join(report['tags'])})"
        )
        lines.append("")
        lines.append(
            f"Mode: **{report['mode']}**"
            + (f" (dominant model: `{report['best_key']}`)" if report["best_key"] else "")
            + f" | threshold: {report['threshold']:.2f} | scenarios: {', '.join(report['scenarios'])}"
        )
        lines.append("")
        lines.append("| Detector | Kind | AUROC | val AUROC | F1 | Prec | Rec | FAR(clean) | Rec≥0.5 | Rec≥1.0 | Latency | Weight |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for entry in sorted(report["detectors"], key=lambda e: -_num(e["auroc"])):
            lines.append(
                f"| `{entry['key']}` | {entry['kind']} | {_fmt(entry['auroc'])} |"
                f" {_fmt(entry['val_auroc'])} | {_fmt(entry['f1'])} | {_fmt(entry['precision'])} |"
                f" {_fmt(entry['recall'])} | {_fmt(entry.get('far_clean'))} |"
                f" {_fmt(entry.get('recall_0.5_1_iqr'))} | {_fmt(entry.get('recall_1_inf_iqr'))} |"
                f" {_fmt(entry['latency_s'], 0)} s | {entry['weight']:.2f} |"
            )
        ens = report["ensemble"]
        lines.append(
            f"| **ensemble** | fusion | {_fmt(ens['auroc'])} | – | {_fmt(ens['f1'])} |"
            f" {_fmt(ens['precision'])} | {_fmt(ens['recall'])} | {_fmt(ens.get('far_clean'))} |"
            f" {_fmt(ens.get('recall_0.5_1_iqr'))} | {_fmt(ens.get('recall_1_inf_iqr'))} |"
            f" {_fmt(ens['latency_s'], 0)} s | – |"
        )

    lines.append("")
    lines.append("## Global check against the dataset's real 'tailed' anomalies")
    lines.append("")
    lines.append("Only the `main_steam` part holds the state-discriminating tag (TE_8332A).")
    lines.append("")
    lines.append("| Part | AUROC | F1 | Precision | Recall | FAR | Positives | Prevalence |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for part_key, report in leaderboard["parts"].items():
        gt = report.get("global_tailed")
        if not gt:
            continue
        lines.append(
            f"| {report['part_name']} | {_fmt(gt['auroc'])} | {_fmt(gt['f1'])} |"
            f" {_fmt(gt['precision'])} | {_fmt(gt['recall'])} | {_fmt(gt['far'])} |"
            f" {gt['positives']} | {gt['prevalence']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def _num(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return -1.0
    return -1.0 if np.isnan(value) else value


def _fmt(value, digits: int = 3) -> str:
    number = _num(value)
    return "–" if number < 0 else f"{number:.{digits}f}"


def write_artifacts(leaderboard: dict) -> dict[str, Path]:
    """Write the leaderboard JSON and the generated Markdown report."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_JSON.write_text(json.dumps(leaderboard, indent=2, default=str))
    LEADERBOARD_MD.write_text(leaderboard_markdown(leaderboard))
    return {"json": LEADERBOARD_JSON, "markdown": LEADERBOARD_MD}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train per-part detection ensembles")
    parser.add_argument("--parts", nargs="*", default=None, help="part keys (default: all)")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=8000)
    parser.add_argument("--false-alarm-rate", type=float, default=0.05)
    parser.add_argument("--scenario-limit", type=int, default=6)
    parser.add_argument("--eval-repeats", type=int, default=4)
    parser.add_argument("--block", type=int, default=240, help="fault block length (samples)")
    parser.add_argument("--gap", type=int, default=120, help="clean stretch length (samples)")
    parser.add_argument("--include-heavy", action="store_true", help="also train the transformer forecaster")
    parser.add_argument(
        "--threshold-mode", choices=("f1", "target-far"), default="f1",
        help="f1: maximise accuracy on validation scenarios; target-far: hit a false-alarm target",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        false_alarm_rate=args.false_alarm_rate,
        include_heavy=args.include_heavy,
        parts=tuple(args.parts) if args.parts else None,
        scenario_limit=args.scenario_limit,
        eval_repeats=args.eval_repeats,
        block=args.block,
        gap=args.gap,
        threshold_mode=args.threshold_mode,
        seed=args.seed,
    )
    started = time.time()
    leaderboard = train_all(cfg)
    paths = write_artifacts(leaderboard)
    print(f"trained {len(leaderboard['parts'])} parts in {time.time() - started:.1f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
