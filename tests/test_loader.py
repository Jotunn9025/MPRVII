"""Tests for the dataset loader."""

from __future__ import annotations

import pandas as pd
import pytest

from bhel_optimize import config as C
from bhel_optimize.data import (
    DatasetNotFoundError,
    load_columns,
    load_dataset,
    load_imputed,
    load_raw,
    sanity_report,
)
from conftest import requires_data


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        load_raw(tmp_path / "nope.csv")
    with pytest.raises(DatasetNotFoundError):
        load_imputed(tmp_path / "nope.csv")


@requires_data
def test_columns_file_lists_all_tags():
    columns = load_columns()
    assert set(columns) == set(C.TAGS)
    assert columns["TE_8332A.AV_0"].lower().startswith("boiler outlet steam")


@requires_data
def test_raw_shape_and_schema(raw_df):
    assert raw_df.shape == (C.EXPECTED_ROWS, len(C.TAGS) + 1)
    assert list(raw_df.columns) == [C.DATE_COLUMN, *C.TAGS]


@requires_data
def test_imputed_shape_and_schema(imputed_df):
    assert imputed_df.shape == (C.EXPECTED_ROWS, len(C.TAGS) + 1)
    assert list(imputed_df.columns) == [C.DATE_COLUMN, *C.TAGS]


@requires_data
def test_raw_has_known_missing_values(raw_df):
    missing = raw_df.isna().sum()
    assert int(missing.sum()) == 30
    assert missing.idxmax() == "YJJWSLL.AV_0"
    assert int(missing["YJJWSLL.AV_0"]) == 30


@requires_data
def test_imputed_is_complete(imputed_df):
    assert int(imputed_df.isna().sum().sum()) == 0


@requires_data
def test_timestamps_monotonic_and_five_second(raw_df):
    dates = raw_df[C.DATE_COLUMN]
    assert dates.is_monotonic_increasing
    assert isinstance(dates.dtype, pd.DatetimeTZDtype) is False
    deltas = dates.diff().dropna().dt.total_seconds()
    assert set(deltas.unique()) == {float(C.SAMPLE_INTERVAL_S)}


@requires_data
def test_date_range(raw_df):
    report = sanity_report(raw_df)
    assert report["start"] == pd.Timestamp("2022-03-27 14:28:54")
    assert report["end"] == pd.Timestamp("2022-04-01 14:28:49")


@requires_data
def test_sanity_report_fields(raw_df):
    report = sanity_report(raw_df)
    assert report["rows"] == C.EXPECTED_ROWS
    assert report["tags"] == len(C.TAGS)
    assert report["expected_rows_match"] is True
    assert report["monotonic_increasing"] is True
    assert report["missing_by_tag"] == {"YJJWSLL.AV_0": 30}


@requires_data
def test_load_dataset_defaults_to_imputed(imputed_df):
    default = load_dataset()
    assert int(default.isna().sum().sum()) == 0
    assert default.shape == imputed_df.shape


@requires_data
def test_load_dataset_raw_option(raw_df):
    assert int(load_dataset(imputed=False).isna().sum().sum()) == 30
