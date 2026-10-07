"""Preprocessing: missing values, per-tag scaling, splits and labels."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from bhel_optimize import config as C

SCALER_METHODS = {"robust": RobustScaler, "standard": StandardScaler}


def handle_missing(frame: pd.DataFrame, method: str = "time_interpolate") -> pd.DataFrame:
    """Fill missing tag values.

    Methods: ``time_interpolate`` (default, time-aware linear interpolation with
    edge fill), ``linear``, ``drop`` (drop rows with any missing tag).
    """
    out = frame.copy()
    if method == "drop":
        return out.dropna(subset=list(C.TAGS)).reset_index(drop=True)
    if method not in {"time_interpolate", "linear"}:
        raise ValueError(f"unknown missing-value method: {method}")
    indexed = out.set_index(C.DATE_COLUMN)
    if method == "time_interpolate":
        indexed = indexed.interpolate(method="time", limit_direction="both")
    else:
        indexed = indexed.interpolate(limit_direction="both")
    return indexed.reset_index()


def fit_scalers(
    frame: pd.DataFrame, method: str = "robust", tags=None
) -> dict[str, object]:
    """Fit one scaler per tag (call on the train slice only)."""
    if method not in SCALER_METHODS:
        raise ValueError(f"unknown scaler method: {method}")
    tags = tuple(C.TAGS if tags is None else tags)
    scalers: dict[str, object] = {}
    for tag in tags:
        scaler = SCALER_METHODS[method]()
        scaler.fit(frame[[tag]].to_numpy())
        scalers[tag] = scaler
    return scalers


def transform_tags(frame: pd.DataFrame, scalers: dict[str, object]) -> pd.DataFrame:
    """Return a copy with every scaled tag column replaced by its scaled value."""
    out = frame.copy()
    for tag, scaler in scalers.items():
        out[tag] = scaler.transform(out[[tag]].to_numpy()).ravel()
    return out


def save_scalers(scalers: dict[str, object], path=None) -> Path:
    """Persist fitted scalers with joblib."""
    path = Path(C.SCALERS_PATH if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, path)
    return path


def load_scalers(path=None) -> dict[str, object]:
    """Load scalers persisted by :func:`save_scalers`."""
    path = Path(C.SCALERS_PATH if path is None else path)
    return joblib.load(path)


def split_bounds(n_rows: int, fractions: dict[str, float] | None = None) -> dict[str, tuple[int, int]]:
    """Return contiguous ``[start, stop)`` index bounds per split."""
    fractions = dict(C.SPLIT_FRACTIONS if fractions is None else fractions)
    keys = list(fractions)
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")
    bounds: dict[str, tuple[int, int]] = {}
    start = 0
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            stop = n_rows
        else:
            stop = start + int(round(n_rows * fractions[key]))
        bounds[key] = (start, stop)
        start = stop
    return bounds


def chronological_split(
    frame: pd.DataFrame, fractions: dict[str, float] | None = None
) -> dict[str, pd.DataFrame]:
    """Split chronologically (no shuffling) into contiguous slices."""
    bounds = split_bounds(len(frame), fractions)
    return {
        key: frame.iloc[start:stop].reset_index(drop=True)
        for key, (start, stop) in bounds.items()
    }


def steam_temp_labels(frame: pd.DataFrame) -> pd.Series:
    """Label rows by the dataset's anomaly rule (1 = tailed, 0 = headed)."""
    low, high = C.NORMAL_STEAM_TEMP_C
    temp = frame[C.STATE_TAG]
    labels = ((temp < low) | (temp > high)).astype("int8")
    labels.name = "is_anomaly"
    return labels


def normal_tailed_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(normal, tailed)`` frames per the steam-temperature rule."""
    labels = steam_temp_labels(frame)
    return (
        frame.loc[labels == 0].reset_index(drop=True),
        frame.loc[labels == 1].reset_index(drop=True),
    )


def label_summary(labels: pd.Series) -> dict:
    """Count ``{normal, tailed, total, tailed_fraction}`` for a label series."""
    total = int(len(labels))
    tailed = int(labels.sum())
    return {
        "normal": total - tailed,
        "tailed": tailed,
        "total": total,
        "tailed_fraction": (tailed / total) if total else 0.0,
    }


def resample(frame: pd.DataFrame, rule: str = "10s") -> pd.DataFrame:
    """Mean-resample the tag columns onto a coarser time grid."""
    out = frame.set_index(C.DATE_COLUMN)
    out = out[list(C.TAGS)].resample(rule).mean().dropna(how="all")
    return out.reset_index()
