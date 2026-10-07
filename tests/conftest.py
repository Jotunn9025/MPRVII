"""Shared fixtures. Dataset-dependent tests skip when data/ is absent."""

from __future__ import annotations

import pytest

from bhel_optimize import config as C

requires_data = pytest.mark.skipif(
    not (C.RAW_CSV.exists() and C.IMPUTED_CSV.exists()),
    reason="dataset CSVs not present in data/ (git-ignored user download)",
)


@pytest.fixture(scope="session")
def raw_df():
    from bhel_optimize.data import load_raw

    return load_raw()


@pytest.fixture(scope="session")
def imputed_df():
    from bhel_optimize.data import load_imputed

    return load_imputed()
