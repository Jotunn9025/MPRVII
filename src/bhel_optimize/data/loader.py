"""Load and validate the industrial-boiler time-series dataset.

The published files live in ``data/`` (git-ignored, user-supplied):

- ``data.csv``          — as published; 30 missing values in ``YJJWSLL.AV_0``
- ``data_AutoReg.csv``  — same data with those gaps AutoReg-imputed
- ``columns.csv``       — tag descriptions from paper Table 1

Headers carry a trailing ``#`` on every DCS tag; tags are normalised on load
so the returned frame always uses :data:`bhel_optimize.config.TAGS`.
"""

from __future__ import annotations

import pandas as pd

from bhel_optimize import config as C


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset CSV is not present in ``data/``."""


def _require(path) -> None:
    if not path.exists():
        raise DatasetNotFoundError(
            f"dataset file not found: {path} — place data.csv / "
            "data_AutoReg.csv in the data/ directory"
        )


def load_columns(path=None) -> dict[str, str]:
    """Return ``{tag: description}`` from ``columns.csv``.

    The shipped file declares a ``Description`` column in its header but the
    data rows only carry two fields, so the third is treated as optional.
    """
    path = C.COLUMNS_CSV if path is None else path
    _require(path)
    frame = pd.read_csv(path)
    columns = [str(c).strip() for c in frame.columns]
    frame.columns = columns
    variant_col = columns[0]
    attribute_col = columns[1] if len(columns) > 1 else columns[0]
    out: dict[str, str] = {}
    for _, row in frame.iterrows():
        tag = C.normalise_tag(str(row[variant_col]).strip())
        description = str(row[attribute_col]).strip()
        out[tag] = description
    return out


def _load_csv(path) -> pd.DataFrame:
    """Read a dataset CSV into a date-indexed frame with normalised tags."""
    _require(path)
    frame = pd.read_csv(path)
    frame.columns = [C.normalise_tag(str(c).strip()) for c in frame.columns]

    if C.DATE_COLUMN not in frame.columns:
        raise ValueError(f"missing '{C.DATE_COLUMN}' column in {path}")
    frame[C.DATE_COLUMN] = pd.to_datetime(frame[C.DATE_COLUMN], errors="raise")
    frame = frame.sort_values(C.DATE_COLUMN).reset_index(drop=True)

    missing = [t for t in C.TAGS if t not in frame.columns]
    extra = [c for c in frame.columns if c not in (*C.TAGS, C.DATE_COLUMN)]
    if missing or extra:
        raise ValueError(
            f"unexpected dataset columns — missing={missing}, extra={extra}"
        )
    return frame[[C.DATE_COLUMN, *C.TAGS]]


def load_raw(path=None) -> pd.DataFrame:
    """Load the as-published dataset (contains the YJJWSLL.AV_0 gaps)."""
    return _load_csv(C.RAW_CSV if path is None else path)


def load_imputed(path=None) -> pd.DataFrame:
    """Load the AutoReg-imputed dataset (no missing values)."""
    return _load_csv(C.IMPUTED_CSV if path is None else path)


def load_dataset(imputed: bool = True) -> pd.DataFrame:
    """Load the canonical dataset (imputed by default, per the project plan)."""
    return load_imputed() if imputed else load_raw()


def sanity_report(frame: pd.DataFrame) -> dict:
    """Summarise shape, missing values, time range and sampling interval."""
    missing = {t: int(frame[t].isna().sum()) for t in C.TAGS}
    missing = {t: n for t, n in missing.items() if n}
    dates = frame[C.DATE_COLUMN]
    deltas = dates.diff().dropna().dt.total_seconds()
    interval_counts = deltas.value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "tags": len(C.TAGS),
        "missing_total": int(sum(missing.values())),
        "missing_by_tag": missing,
        "start": dates.min(),
        "end": dates.max(),
        "monotonic_increasing": bool(dates.is_monotonic_increasing),
        "interval_s_counts": interval_counts,
        "expected_rows_match": len(frame) == C.EXPECTED_ROWS,
    }
