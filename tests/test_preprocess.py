"""Tests for preprocessing: missing values, splits, labels, scalers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bhel_optimize import config as C
from bhel_optimize.data import (
    chronological_split,
    fit_scalers,
    handle_missing,
    label_summary,
    load_scalers,
    normal_tailed_frames,
    resample,
    save_scalers,
    split_bounds,
    steam_temp_labels,
    transform_tags,
)
from conftest import requires_data


def _synthetic(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="5s")
    frame = pd.DataFrame({C.DATE_COLUMN: dates})
    for tag in C.TAGS:
        frame[tag] = np.arange(n, dtype=float)
    return frame


# --- splits ---------------------------------------------------------------


def test_split_bounds_contiguous_and_complete():
    bounds = split_bounds(1000)
    assert bounds["train"][0] == 0
    assert bounds["test"][1] == 1000
    assert bounds["train"][1] == bounds["val"][0]
    assert bounds["val"][1] == bounds["test"][0]


def test_split_bounds_sizes():
    bounds = split_bounds(1000)
    assert bounds["train"] == (0, 700)
    assert bounds["val"] == (700, 850)
    assert bounds["test"] == (850, 1000)


def test_split_bounds_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_bounds(10, {"a": 0.5, "b": 0.4})


def test_chronological_split_preserves_order():
    frame = _synthetic(100)
    splits = chronological_split(frame)
    assert [len(v) for v in splits.values()] == [70, 15, 15]
    joined = pd.concat(splits.values())
    assert joined[C.DATE_COLUMN].is_monotonic_increasing
    assert len(joined) == len(frame)


# --- labelling ------------------------------------------------------------


def test_steam_temp_labels_rule():
    frame = _synthetic(5)
    frame[C.STATE_TAG] = [529.9, 530.0, 537.5, 545.0, 545.1]
    assert steam_temp_labels(frame).tolist() == [1, 0, 0, 0, 1]


def test_label_summary_math():
    labels = pd.Series([0, 0, 0, 1], name="is_anomaly")
    summary = label_summary(labels)
    assert summary == {
        "normal": 3,
        "tailed": 1,
        "total": 4,
        "tailed_fraction": 0.25,
    }


@requires_data
def test_real_tailed_fraction_near_paper_value(imputed_df):
    summary = label_summary(steam_temp_labels(imputed_df))
    assert summary["total"] == C.EXPECTED_ROWS
    assert 0.07 < summary["tailed_fraction"] < 0.09


@requires_data
def test_normal_tailed_frames_partition_data(imputed_df):
    normal, tailed = normal_tailed_frames(imputed_df)
    assert len(normal) + len(tailed) == len(imputed_df)
    low, high = C.NORMAL_STEAM_TEMP_C
    assert normal[C.STATE_TAG].between(low, high).all()
    assert not tailed[C.STATE_TAG].between(low, high).any()


# --- missing values -------------------------------------------------------


@requires_data
def test_time_interpolation_removes_missing(raw_df):
    filled = handle_missing(raw_df, "time_interpolate")
    assert int(filled.isna().sum().sum()) == 0
    assert filled.shape[1] == raw_df.shape[1]


@requires_data
def test_linear_interpolation_removes_missing(raw_df):
    filled = handle_missing(raw_df, "linear")
    assert int(filled.isna().sum().sum()) == 0


@requires_data
def test_drop_missing_removes_rows(raw_df):
    dropped = handle_missing(raw_df, "drop")
    assert int(dropped.isna().sum().sum()) == 0
    assert len(dropped) == len(raw_df) - 30


def test_unknown_missing_method_raises():
    with pytest.raises(ValueError):
        handle_missing(_synthetic(5), "magic")


# --- scalers --------------------------------------------------------------


def test_fit_transform_standard_scaler():
    frame = _synthetic(50)
    scalers = fit_scalers(frame, "standard")
    scaled = transform_tags(frame, scalers)
    for tag in C.TAGS:
        assert abs(float(scaled[tag].mean())) < 1e-9
        # StandardScaler uses population std (ddof=0), pandas defaults to ddof=1
        assert abs(float(scaled[tag].std(ddof=0)) - 1.0) < 1e-6


def test_fit_transform_robust_scaler_centres_median():
    frame = _synthetic(50)
    scalers = fit_scalers(frame, "robust")
    scaled = transform_tags(frame, scalers)
    assert abs(float(scaled[C.TAGS[0]].median())) < 1e-9


def test_transform_does_not_mutate_input():
    frame = _synthetic(20)
    before = frame.copy()
    scalers = fit_scalers(frame, "robust")
    transform_tags(frame, scalers)
    pd.testing.assert_frame_equal(frame, before)


def test_unknown_scaler_method_raises():
    with pytest.raises(ValueError):
        fit_scalers(_synthetic(5), "magic")


def test_scaler_save_load_roundtrip(tmp_path):
    frame = _synthetic(30)
    scalers = fit_scalers(frame, "robust")
    path = save_scalers(scalers, tmp_path / "scalers.joblib")
    assert path.exists()
    restored = load_scalers(path)
    assert set(restored) == set(scalers)
    pd.testing.assert_frame_equal(
        transform_tags(frame, scalers), transform_tags(frame, restored)
    )


@requires_data
def test_scalers_fit_on_train_only(imputed_df):
    splits = chronological_split(imputed_df)
    scalers = fit_scalers(splits["train"], "robust")
    scaled = transform_tags(imputed_df, scalers)
    assert int(scaled.isna().sum().sum()) == 0


# --- resampling -----------------------------------------------------------


def test_resample_10s_halves_rows():
    frame = _synthetic(100)
    assert len(resample(frame, "10s")) == 50


def test_resample_1min_aggregates():
    frame = _synthetic(240)  # 240 x 5 s = 1200 s = 20 minutes
    out = resample(frame, "1min")
    assert len(out) == 20
    assert list(out.columns) == [C.DATE_COLUMN, *C.TAGS]


@requires_data
def test_resample_real_data(imputed_df):
    assert len(resample(imputed_df, "1min")) == 7201
