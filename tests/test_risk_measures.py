"""Tests offline para metrics/risk_measures.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics.risk_measures import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    equity_curve,
    expected_shortfall,
    max_drawdown,
    sharpe_lo_adjusted,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

# --------------------------------------------------------------------------
# Equity curve / drawdown
# --------------------------------------------------------------------------


def test_equity_curve_starts_at_initial_value_and_compounds() -> None:
    r = pd.Series([np.nan, 0.10, -0.10])
    eq = equity_curve(r, initial_value=100.0)

    assert eq.iloc[0] == pytest.approx(100.0)
    assert eq.iloc[1] == pytest.approx(110.0)
    assert eq.iloc[2] == pytest.approx(99.0)


def test_max_drawdown_of_always_increasing_series_is_zero() -> None:
    r = pd.Series([0.01] * 50)
    assert max_drawdown(r) == pytest.approx(0.0)


def test_max_drawdown_detects_known_drop() -> None:
    # equity: 1 -> 1.10 -> 0.99  (caída del 10% desde el pico de 1.10)
    r = pd.Series([0.10, -0.10])
    assert max_drawdown(r) == pytest.approx(-0.10, abs=1e-9)


def test_calmar_ratio_is_inf_when_no_drawdown() -> None:
    r = pd.Series([0.01] * 50)
    assert calmar_ratio(r) == np.inf


# --------------------------------------------------------------------------
# Retorno / volatilidad anualizados
# --------------------------------------------------------------------------


def test_annualized_return_matches_compounding_formula() -> None:
    r = pd.Series([0.001] * 252)
    assert annualized_return(r, periods_per_year=252) == pytest.approx(1.001**252 - 1.0)


def test_annualized_volatility_scales_by_sqrt_periods_per_year() -> None:
    r = pd.Series(np.random.default_rng(5).normal(0, 0.01, 252))
    vol_daily = r.std(ddof=1)
    assert annualized_volatility(r, periods_per_year=252) == pytest.approx(vol_daily * np.sqrt(252))


# --------------------------------------------------------------------------
# Sharpe / Sortino
# --------------------------------------------------------------------------


def test_sharpe_ratio_constant_positive_returns_is_inf() -> None:
    r = pd.Series([0.001] * 100)
    assert sharpe_ratio(r) == np.inf


def test_sharpe_ratio_constant_zero_returns_is_zero() -> None:
    r = pd.Series([0.0] * 100)
    assert sharpe_ratio(r) == 0.0


def test_sharpe_ratio_constant_negative_returns_is_negative_inf() -> None:
    r = pd.Series([-0.001] * 100)
    assert sharpe_ratio(r) == -np.inf


def test_sharpe_ratio_of_iid_normal_returns_is_finite() -> None:
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0005, 0.02, 500))
    assert np.isfinite(sharpe_ratio(r))


def test_sortino_ratio_no_downside_is_inf() -> None:
    r = pd.Series([0.01, 0.05, 0.001, 0.02])
    assert sortino_ratio(r) == np.inf


def test_sortino_ratio_ge_sharpe_when_returns_are_positively_skewed() -> None:
    # Muchas subidas chicas + pocas bajadas grandes: el downside deviation
    # (solo mira la cola negativa) es menor que el desvío total, por lo que
    # el Sortino debería ser mayor o igual al Sharpe.
    r = pd.Series([0.01] * 18 + [-0.05, -0.03])
    assert sortino_ratio(r) >= sharpe_ratio(r)


def test_sharpe_lo_adjusted_shrinks_naive_sharpe_under_positive_autocorrelation() -> None:
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.01, 500)
    trend = 0.0008
    raw = pd.Series(trend + noise)
    # Suavizar induce autocorrelación serial positiva.
    smoothed = raw.rolling(window=5, min_periods=1).mean()

    naive = sharpe_ratio(smoothed)
    lo = sharpe_lo_adjusted(smoothed, max_lags=10)

    assert np.isfinite(lo)
    assert lo < naive


def test_sharpe_lo_adjusted_close_to_naive_for_iid_data() -> None:
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0003, 0.015, 2000))

    naive = sharpe_ratio(r)
    lo = sharpe_lo_adjusted(r, max_lags=10)

    assert np.isfinite(lo)
    assert lo == pytest.approx(naive, rel=0.5, abs=0.5)


# --------------------------------------------------------------------------
# VaR / Expected Shortfall
# --------------------------------------------------------------------------


def test_value_at_risk_and_expected_shortfall_are_positive_and_es_ge_var() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0, 0.02, 2000))

    var95 = value_at_risk(r, level=0.95)
    es95 = expected_shortfall(r, level=0.95)

    assert var95 > 0
    assert es95 > 0
    assert es95 >= var95


def test_value_at_risk_higher_confidence_is_more_conservative() -> None:
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.0, 0.02, 2000))

    var95 = value_at_risk(r, level=0.95)
    var99 = value_at_risk(r, level=0.99)

    assert var99 >= var95
