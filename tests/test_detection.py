"""Tests for the detection layer: windows, detectors, ensembles, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bhel_optimize import config as C
from bhel_optimize.detection import detectors as D
from bhel_optimize.detection.ensembles import (
    PartEnsemble,
    base_ensemble,
    calibrate_threshold,
    dominant_detector,
    fuse,
    normalize_scores,
    percentile_knots,
    percentile_transform,
    reference_stats,
    weights_from_auroc,
)
from bhel_optimize.detection.evaluate import (
    auroc,
    best_f1,
    confusion,
    detection_rate,
    evaluate_scores,
    false_alarm_rate_clean,
    mean_detection_latency,
    recall_by_severity,
    smooth_series,
)
from bhel_optimize.detection.windows import (
    augment_with_differences,
    build_windows,
    scatter_scores,
    subsample_windows,
)


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------


def test_build_windows_shapes_and_ends():
    values = np.arange(30, dtype="float32").reshape(10, 3)
    windows, ends = build_windows(values, lookback=4)
    assert windows.shape == (7, 4, 3)
    assert ends.tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert windows[0].tolist() == values[:4].tolist()


def test_build_windows_stride_and_short_input():
    values = np.zeros((10, 2), dtype="float32")
    windows, ends = build_windows(values, lookback=4, stride=3)
    assert windows.shape[0] == len(ends) == 3
    windows, ends = build_windows(values[:3], lookback=4)
    assert windows.shape[0] == 0 and ends.shape[0] == 0


def test_build_windows_rejects_bad_params():
    with pytest.raises(ValueError):
        build_windows(np.zeros((5, 2)), lookback=1)
    with pytest.raises(ValueError):
        build_windows(np.zeros((5, 2)), lookback=3, stride=0)


def test_scatter_scores_aligns_to_last_sample():
    out = scatter_scores(np.array([2, 4]), np.array([1.0, 2.0]), n_samples=6)
    assert out[0] is np.nan or np.isnan(out[0])
    assert out[2] == 1.0 and out[4] == 2.0 and np.isnan(out[5])


def test_subsample_windows_keeps_order_and_size():
    windows = np.arange(100, dtype="float32").reshape(20, 5, 1)
    ends = np.arange(20)
    out, out_ends = subsample_windows(windows, ends, max_windows=5)
    assert out.shape[0] == 5 and out_ends.shape[0] == 5
    assert np.all(np.diff(out_ends) > 0)
    same, _ = subsample_windows(windows, ends, max_windows=0)
    assert same.shape[0] == 20


def test_augment_with_differences_adds_change_channels():
    frame = pd.DataFrame({"a": [1.0, 3.0, 6.0]})
    out, names = augment_with_differences(frame, ["a"])
    assert names == ["d_a"]
    assert out["a"].tolist() == [1.0, 3.0, 6.0]
    assert out["d_a"].tolist() == [0.0, 2.0, 3.0]


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------


def _flat_windows(n=60, lookback=20, n_tags=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, lookback, n_tags)).astype("float32")


def test_registry_keys_and_build_errors():
    keys = D.detector_keys()
    assert "mean_shift" in keys and "staleness" in keys
    assert "transformer_residual" not in keys
    assert "transformer_residual" in D.detector_keys(include_heavy=True)
    with pytest.raises(KeyError):
        D.build_detector("nope", 10, 2)


def test_ewma_mad_flags_last_sample_jump():
    windows = _flat_windows()
    detector = D.build_detector("ewma_mad", 20, 2).fit(windows)
    normal = detector.score(windows[:10])
    spiked = windows.copy()
    spiked[:, -1, :] += 8.0
    assert detector.score(spiked).mean() > normal.mean() * 3


def test_mean_shift_flags_sustained_shift():
    windows = _flat_windows()
    detector = D.build_detector("mean_shift", 20, 2).fit(windows)
    shifted = windows.copy()
    shifted[:, -5:, :] += 0.8  # sustained, below single-sample noise
    assert detector.score(shifted).mean() > detector.score(windows).mean()


def test_mad_detectors_do_not_explode_on_flat_windows():
    """Regression: dividing by a ~0 window MAD produced 1e7 scores."""
    rng = np.random.default_rng(1)
    windows = np.repeat(rng.normal(0, 1, (50, 20, 2)), 3, axis=1).astype("float32")
    flat = np.zeros((20, 20, 2), dtype="float32")  # perfectly flat stretch
    for key in ("ewma_mad", "mean_shift", "staleness"):
        detector = D.build_detector(key, 20, 2).fit(windows)
        assert np.isfinite(detector.score(flat)).all()
        assert np.abs(detector.score(flat)).max() < 1e6


def test_staleness_flags_frozen_sensor():
    windows = _flat_windows()
    detector = D.build_detector("staleness", 20, 2).fit(windows)
    frozen = windows.copy()
    frozen[:, -15:, 0] = frozen[:, -15, 0][:, None]
    assert detector.score(frozen).mean() > detector.score(windows).mean()


def test_sklearn_detectors_shapes_and_flags():
    windows = _flat_windows(80)
    for key in ("isolation_forest", "ocsvm"):
        detector = D.build_detector(key, 20, 2, seed=0).fit(windows)
        scores = detector.score(windows[:20])
        assert scores.shape == (20,) and np.isfinite(scores).all()


def test_tensorflow_detectors_need_fit_first():
    for key in ("dense_ae", "lstm_residual", "tcn_ae"):
        detector = D.build_detector(key, 20, 2)
        with pytest.raises(RuntimeError):
            detector.score(_flat_windows(5))


def test_dense_autoencoder_scores_anomaly_higher():
    """Tiny end-to-end TF check (1 epoch) — keeps the suite fast."""
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, (256, 20, 2)).astype("float32")
    detector = D.build_detector("dense_ae", 20, 2, seed=0, epochs=1, batch_size=64)
    detector.fit(train)
    normal = detector.score(train[:16])
    anomalous = train[:16].copy()
    anomalous[:, -1, :] += 10.0
    assert detector.score(anomalous).mean() > normal.mean()


# --------------------------------------------------------------------------
# ensembles
# --------------------------------------------------------------------------


def test_normalize_scores_is_clipped_z():
    reference = np.arange(100, dtype="float64")
    z = normalize_scores(np.array([1000.0]), reference)
    assert z[0] <= 20.0  # clipped, not 1e9


def test_reference_stats_and_percentile_helpers_are_nan_safe():
    with_nan = np.array([1.0, np.nan, 3.0, 5.0])
    median, iqr = reference_stats(with_nan)
    assert np.isfinite(median) and np.isfinite(iqr)
    knots = percentile_knots(with_nan)
    assert len(knots) == 201
    assert percentile_transform(np.array([0.0, 6.0]), knots).tolist() == [0.0, 1.0]


def test_fuse_respects_weights():
    a = np.array([1.0, 1.0])
    b = np.array([3.0, 3.0])
    assert fuse({"a": a, "b": b}, {"a": 1.0, "b": 0.0}).tolist() == [1.0, 1.0]
    assert fuse({"a": a, "b": b}, {"a": 0.5, "b": 0.5}).tolist() == [2.0, 2.0]
    with pytest.raises(ValueError):
        fuse({}, {})


def test_weights_from_auroc_drops_weak_detectors():
    weights = weights_from_auroc({"good": 0.9, "weak": 0.52, "bad": 0.4})
    assert weights.get("good", 0) > 0
    assert "bad" not in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    equal = weights_from_auroc({"a": 0.4, "b": 0.3})
    assert equal == {"a": 0.5, "b": 0.5}


def test_dominant_detector_margin():
    assert dominant_detector({"a": 0.9, "b": 0.6}) == "a"
    assert dominant_detector({"a": 0.9, "b": 0.89}) is None
    assert dominant_detector({"only": 0.7}) == "only"


def test_calibrate_threshold_hits_target_rate_and_ignores_nan():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(0, 1, 5000), [np.nan, np.nan]])
    threshold = calibrate_threshold(scores, false_alarm_rate=0.05)
    assert np.isfinite(threshold)
    realised = float(np.mean(scores[~np.isnan(scores)] >= threshold))
    assert 0.03 < realised < 0.07
    with pytest.raises(ValueError):
        calibrate_threshold(scores, 1.5)


def test_part_ensemble_score_and_save_load(tmp_path):
    windows = _flat_windows(50)
    ensemble = base_ensemble("drum", 20)
    ensemble.detectors["ewma_mad"] = D.build_detector("ewma_mad", 20, 2).fit(windows)
    ensemble.references["ewma_mad"] = reference_stats(
        ensemble.detectors["ewma_mad"].score(windows)
    )
    ensemble.weights = {"ewma_mad": 1.0}
    ensemble.threshold = 1.0

    scores = ensemble.score(windows[:5])
    decisions = ensemble.decide(windows[:5])
    assert scores.shape == (5,) and decisions.dtype == bool

    path = ensemble.save(tmp_path / "drum")
    assert path.exists()

    restored = base_ensemble("drum", 20)
    restored.detectors["ewma_mad"] = D.build_detector("ewma_mad", 20, 2)
    restored.load(tmp_path / "drum")
    np.testing.assert_allclose(restored.score(windows[:5]), scores)
    assert restored.threshold == ensemble.threshold


def test_part_ensemble_requires_detectors():
    with pytest.raises(RuntimeError):
        base_ensemble("drum", 20).score(np.zeros((1, 20, 2)))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_auroc_perfect_inverted_and_single_class():
    labels = np.array([0, 0, 1, 1])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert np.isnan(auroc(np.zeros(4), np.arange(4)))


def test_auroc_ignores_nan_scores():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, np.nan, 0.8])
    assert auroc(labels, scores) == 1.0


def test_confusion_counts_and_rates():
    labels = np.array([0, 0, 1, 1])
    decisions = np.array([False, True, True, False])
    result = confusion(labels, decisions)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["far"] == 0.5


def test_best_f1_finds_separable_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.1, 0.8, 0.9])
    result = best_f1(labels, scores)
    assert result["f1"] == 1.0
    assert 0.1 <= result["threshold"] <= 0.8


def test_latency_and_detection_rate_over_runs():
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    decisions = np.array([False, False, True, True, False, False, False, False])
    assert mean_detection_latency(labels, decisions) == 1.0  # (1 + never)/1 run
    assert detection_rate(labels, decisions) == 0.5


def test_recall_by_severity_bands():
    severity = np.array([0.0, 0.2, 0.7, 1.5])
    decisions = np.array([False, True, True, True])
    scores = np.ones(4)
    result = recall_by_severity(severity, decisions, scores)
    assert result["recall_0.1_0.5_iqr"] == 1.0
    assert result["recall_0.5_1_iqr"] == 1.0
    assert result["recall_1_inf_iqr"] == 1.0
    assert result["n_0.1_0.5_iqr"] == 1


def test_false_alarm_rate_clean_ignores_injected_samples():
    severity = np.array([0.0, 0.0, 0.6, 0.0])
    decisions = np.array([False, True, True, False])
    assert false_alarm_rate_clean(severity, decisions) == pytest.approx(1 / 3)


def test_smooth_series_reduces_single_sample_spikes_and_edges_nan():
    scores = np.zeros(40)
    scores[20] = 100.0
    smoothed = smooth_series(scores, window=5)
    assert smoothed[20] < 100.0
    assert np.isnan(smoothed[0]) and np.isnan(smoothed[-1])
    assert smooth_series(scores, window=1)[20] == 100.0


def test_evaluate_scores_bundle():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    bundle = evaluate_scores(labels, scores, scores >= 0.5)
    assert bundle["auroc"] == 1.0
    assert bundle["f1"] == 1.0
    assert "latency_samples" in bundle


# --------------------------------------------------------------------------
# end-to-end (tiny) training
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_train_part_produces_leaderboard(boiler_frame):
    from bhel_optimize.detection.train import (
        TrainConfig,
        leaderboard_markdown,
        train_all,
    )

    cfg = TrainConfig(
        parts=("furnace",),
        epochs=1,
        max_windows=200,
        scenario_limit=2,
        eval_repeats=2,
        block=60,
        gap=30,
        smooth_window=3,
    )
    leaderboard = train_all(cfg, frame=boiler_frame)
    report = leaderboard["parts"]["furnace"]
    assert report["part_key"] == "furnace"
    assert report["n_features"] > len(report["tags"])  # difference channels added
    assert report["selected"] in {e["key"] for e in report["detectors"]} | {"ensemble"}
    assert np.isfinite(report["threshold"])
    markdown = leaderboard_markdown(leaderboard)
    assert "Furnace Chamber (Upper)" in markdown
    assert "injected simulated faults" in markdown
