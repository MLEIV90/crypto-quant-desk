"""Tests offline para metrics/risk_measures.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import PERIODS_PER_YEAR
from metrics.risk_measures import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    drawdown_series,
    equity_curve,
    expected_shortfall,
    historical_percentile,
    max_drawdown,
    rolling_expected_shortfall,
    rolling_value_at_risk,
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


def test_drawdown_series_matches_max_drawdown_at_its_minimum() -> None:
    # Fase 21 (gráfico underwater): drawdown_series es la curva completa
    # detrás del escalar max_drawdown — su mínimo tiene que coincidir.
    r = pd.Series([0.10, -0.10, 0.05, -0.20, 0.30])
    dd = drawdown_series(r)
    assert dd.min() == pytest.approx(max_drawdown(r))


def test_drawdown_series_is_never_positive() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.02, 300))
    dd = drawdown_series(r)
    assert (dd <= 1e-12).all()


def test_drawdown_series_is_zero_at_new_peaks() -> None:
    r = pd.Series([0.10, 0.10, -0.05, 0.20])  # picos nuevos en índices 0, 1 y 3
    dd = drawdown_series(r)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.0)
    assert dd.iloc[3] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Retorno / volatilidad anualizados
# --------------------------------------------------------------------------


def test_annualized_return_matches_compounding_formula() -> None:
    r = pd.Series([0.001] * PERIODS_PER_YEAR)
    assert annualized_return(r, periods_per_year=PERIODS_PER_YEAR) == pytest.approx(
        1.001**PERIODS_PER_YEAR - 1.0
    )


def test_annualized_volatility_scales_by_sqrt_periods_per_year() -> None:
    r = pd.Series(np.random.default_rng(5).normal(0, 0.01, PERIODS_PER_YEAR))
    vol_daily = r.std(ddof=1)
    assert annualized_volatility(r, periods_per_year=PERIODS_PER_YEAR) == pytest.approx(
        vol_daily * np.sqrt(PERIODS_PER_YEAR)
    )


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


# --------------------------------------------------------------------------
# historical_percentile / rolling VaR-ES (Fase 20a)
# --------------------------------------------------------------------------


def test_historical_percentile_known_case() -> None:
    # 10 valores 1..10: el último (10) es el máximo -> percentil 100
    # (el 100% de la serie, incluido él mismo, es <= 10).
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert historical_percentile(series) == pytest.approx(100.0)

    # El mínimo (1) es <= solo a sí mismo -> percentil 10 (1 de 10 valores).
    assert historical_percentile(series, value=1.0) == pytest.approx(10.0)

    # La mediana exacta de un rango 1..10 par no cae en un solo valor, pero
    # un valor conocido (5.0) es <= a 5 de los 10 -> percentil 50.
    assert historical_percentile(series, value=5.0) == pytest.approx(50.0)


def test_historical_percentile_is_always_between_0_and_100() -> None:
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(size=500))
    for value in [series.min(), series.max(), series.median(), 0.0, 1e6, -1e6]:
        p = historical_percentile(series, value=float(value))
        assert 0.0 <= p <= 100.0


def test_historical_percentile_empty_series_is_nan() -> None:
    assert np.isnan(historical_percentile(pd.Series(dtype=float)))


def test_historical_percentile_defaults_to_last_value() -> None:
    series = pd.Series([10.0, 1.0, 2.0, 3.0])  # último valor (3.0) es <= a 3 de 4 -> percentil 75
    assert historical_percentile(series) == pytest.approx(75.0)


def test_rolling_value_at_risk_matches_manual_window_computation() -> None:
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0, 0.02, 100))
    window = 30

    rolling = rolling_value_at_risk(r, window=window, level=0.95)

    assert rolling.iloc[: window - 1].isna().all()  # warmup
    manual = value_at_risk(r.iloc[0:window], level=0.95)  # primera ventana completa
    assert rolling.iloc[window - 1] == pytest.approx(manual)

    last_window = r.iloc[-window:]
    assert rolling.iloc[-1] == pytest.approx(value_at_risk(last_window, level=0.95))


def test_rolling_expected_shortfall_matches_manual_window_computation() -> None:
    # Fase 25: restaurada (se había quitado en Fase 20c por quedar sin uso
    # tras ese rediseño) — ahora GET /api/risk la usa para unificar el
    # "ES actual" con el mismo método empírico que ya usaba /api/risk-summary
    # para el VaR.
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0, 0.02, 100))
    window = 30

    rolling = rolling_expected_shortfall(r, window=window, level=0.95)

    assert rolling.iloc[: window - 1].isna().all()  # warmup
    manual = expected_shortfall(r.iloc[0:window], level=0.95)
    assert rolling.iloc[window - 1] == pytest.approx(manual)

    last_window = r.iloc[-window:]
    assert rolling.iloc[-1] == pytest.approx(expected_shortfall(last_window, level=0.95))


def test_rolling_expected_shortfall_is_always_ge_rolling_var() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0002, 0.02, 500))
    window = 60

    rolling_var = rolling_value_at_risk(r, window=window, level=0.95)
    rolling_es = rolling_expected_shortfall(r, window=window, level=0.95)

    valid = rolling_var.notna() & rolling_es.notna()
    assert (rolling_es[valid] >= rolling_var[valid]).all()
