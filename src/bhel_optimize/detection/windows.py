"""Sliding-window helpers shared by every detector.

All detectors consume windows shaped ``(n_windows, lookback, n_tags)`` and
return one anomaly score per window. Scores are causal (a window ending at
sample ``t`` only sees samples ``<= t``) and are mapped back to samples with
:func:`scatter_scores`, which assigns each window's score to its **last**
sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bhel_optimize import config as C


def frame_values(frame: pd.DataFrame, tags: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return tag values as a contiguous ``float32`` array ``(n_samples, n_tags)``."""
    return frame[list(tags)].to_numpy(dtype="float32")


def build_windows(
    values: np.ndarray, lookback: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows and the index of each window's last sample."""
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    n_samples = len(values)
    if n_samples < lookback:
        empty = np.empty((0, lookback, values.shape[1]), dtype="float32")
        return empty, np.empty((0,), dtype="int64")
    starts = np.arange(0, n_samples - lookback + 1, stride, dtype="int64")
    ends = starts + lookback - 1
    windows = np.stack([values[s : s + lookback] for s in starts]).astype("float32")
    return windows, ends


def scatter_scores(
    end_indices: np.ndarray, scores: np.ndarray, n_samples: int, fill: float = np.nan
) -> np.ndarray:
    """Map per-window scores to per-sample scores (fill for unseen samples)."""
    out = np.full(n_samples, fill, dtype="float64")
    out[end_indices] = scores
    return out


def subsample_windows(
    windows: np.ndarray, ends: np.ndarray, max_windows: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly subsample windows for training time control (keeps order)."""
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows, ends
    idx = np.linspace(0, len(windows) - 1, max_windows).astype("int64")
    return windows[idx], ends[idx]


def part_tags(part_key: str) -> tuple[str, ...]:
    """Tags monitored for a boiler part."""
    return C.PARTS[part_key].tags


def augment_with_differences(
    frame: pd.DataFrame, tags: list[str] | tuple[str, ...]
) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame with each tag plus its first difference.

    Most boiler fault signatures are *rate-of-change* or sustained-shift
    phenomena (creep, drift, load transitions), not level anomalies. Giving the
    models both the level and the per-sample change of every tag markedly
    improves separation compared with levels alone.
    """
    tags = list(tags)
    out = frame[tags].copy()
    diff_names: list[str] = []
    for tag in tags:
        name = f"d_{tag}"
        out[name] = frame[tag].diff().fillna(0.0)
        diff_names.append(name)
    return out, diff_names
