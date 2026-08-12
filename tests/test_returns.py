"""Tests offline para signals/returns.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.returns import log_returns, simple_returns


def test_simple_returns_first_value_is_nan_and_rest_correct() -> None:
    close = pd.Series([100.0, 110.0, 99.0])
    r = simple_returns(close)

    assert pd.isna(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(99.0 / 110.0 - 1.0)


def test_log_returns_first_value_is_nan_and_matches_formula() -> None:
    close = pd.Series([100.0, 110.0, 99.0])
    r = log_returns(close)

    assert pd.isna(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(110.0 / 100.0))
    assert r.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


def test_log_returns_close_to_simple_returns_for_small_changes() -> None:
    close = pd.Series([100.0, 100.5, 101.0, 100.7, 101.2])
    simple = simple_returns(close)
    log = log_returns(close)

    diff = (simple - log).dropna().abs()
    assert (diff < 1e-3).all()


def test_returns_preserve_index() -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0], index=idx)

    assert simple_returns(close).index.equals(idx)
    assert log_returns(close).index.equals(idx)
