"""Tests offline para analysis/statistics.py (Fase 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.statistics import (
    BTC_HALVING_DATES,
    autocorrelation,
    hourly_seasonality,
    monthly_seasonality,
    spectral_periodogram,
    weekday_seasonality,
)


# --------------------------------------------------------------------------
# monthly_seasonality
# --------------------------------------------------------------------------


def test_monthly_seasonality_computes_correct_mean_per_month() -> None:
    idx = pd.to_datetime(["2021-01-05", "2021-01-15", "2021-02-10", "2022-01-20"], utc=True)
    returns = pd.Series([0.01, 0.03, 0.05, 0.02], index=idx)

    out = monthly_seasonality(returns)

    assert list(out.columns) == ["retorno_medio", "mediana", "desvio", "n"]
    assert out.index.name == "mes"
    assert out.loc[1, "retorno_medio"] == pytest.approx((0.01 + 0.03 + 0.02) / 3)
    assert out.loc[1, "n"] == 3
    assert out.loc[2, "retorno_medio"] == pytest.approx(0.05)
    assert out.loc[2, "n"] == 1
    # un solo dato -> desvío muestral indefinido, no se inventa un 0.
    assert pd.isna(out.loc[2, "desvio"])


def test_monthly_seasonality_omits_months_with_no_observations() -> None:
    idx = pd.to_datetime(["2021-03-01", "2021-03-15"], utc=True)
    returns = pd.Series([0.01, -0.01], index=idx)

    out = monthly_seasonality(returns)

    assert list(out.index) == [3]


# --------------------------------------------------------------------------
# weekday_seasonality
# --------------------------------------------------------------------------


def test_weekday_seasonality_computes_correct_mean() -> None:
    # 2021-01-04 y 2021-01-11 son lunes; 2021-01-05 es martes.
    idx = pd.to_datetime(["2021-01-04", "2021-01-11", "2021-01-05"], utc=True)
    returns = pd.Series([0.02, 0.04, 0.01], index=idx)

    out = weekday_seasonality(returns)

    assert list(out.columns) == ["retorno_medio", "mediana", "desvio", "n"]
    assert out.index.name == "dia_semana"
    assert out.loc[0, "retorno_medio"] == pytest.approx(0.03)
    assert out.loc[0, "n"] == 2
    assert out.loc[1, "retorno_medio"] == pytest.approx(0.01)


def test_weekday_seasonality_computes_desvio_and_leaves_it_nan_for_single_observation() -> None:
    # Fase 26: desvio/mediana agregados para poder comparar el "ruido" (el
    # desvío) contra la diferencia de medias entre días en el frontend.
    idx = pd.to_datetime(["2021-01-04", "2021-01-11", "2021-01-05"], utc=True)
    returns = pd.Series([0.02, 0.04, 0.01], index=idx)

    out = weekday_seasonality(returns)

    assert out.loc[0, "desvio"] == pytest.approx(returns.iloc[[0, 1]].std(ddof=1))
    assert pd.isna(out.loc[1, "desvio"])  # martes: una sola observación


# --------------------------------------------------------------------------
# hourly_seasonality
# --------------------------------------------------------------------------


def test_hourly_seasonality_computes_correct_mean() -> None:
    idx = pd.to_datetime(
        ["2021-01-01T05:00:00Z", "2021-01-02T05:00:00Z", "2021-01-01T10:00:00Z"], utc=True
    )
    returns = pd.Series([0.01, 0.03, 0.05], index=idx)

    out = hourly_seasonality(returns)

    assert list(out.columns) == ["retorno_medio", "mediana", "desvio", "n"]
    assert out.index.name == "hora"
    assert out.loc[5, "retorno_medio"] == pytest.approx(0.02)
    assert out.loc[5, "n"] == 2
    assert out.loc[10, "retorno_medio"] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# autocorrelation
# --------------------------------------------------------------------------


def test_autocorrelation_lag_zero_is_one_and_expected_shape() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, 200))

    out = autocorrelation(r, lags=10)

    assert list(out.columns) == ["acf_retornos", "acf_retornos2"]
    assert out.index.name == "lag"
    assert len(out) == 11
    assert out.loc[0, "acf_retornos"] == pytest.approx(1.0)
    assert out.loc[0, "acf_retornos2"] == pytest.approx(1.0)


def test_autocorrelation_detects_strong_positive_autocorrelation() -> None:
    n = 500
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + noise[t]
    r = pd.Series(ar)

    out = autocorrelation(r, lags=5)

    assert out.loc[1, "acf_retornos"] > 0.5


def test_autocorrelation_squared_detects_volatility_clustering() -> None:
    n = 400
    rng = np.random.default_rng(2)
    # Bloques alternos de 20 observaciones de alta/baja volatilidad: el
    # NIVEL de retorno no está autocorrelacionado (es ruido puro), pero la
    # VARIANZA sí (clustering de volatilidad).
    vol = np.where((np.arange(n) // 20) % 2 == 0, 0.5, 5.0)
    r = pd.Series(rng.normal(0.0, 1.0, n) * vol)

    out = autocorrelation(r, lags=5)

    assert out.loc[1, "acf_retornos2"] > 0.1


def test_autocorrelation_clamps_lags_for_short_series() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])

    out = autocorrelation(r, lags=30)

    assert len(out) <= len(r)


# --------------------------------------------------------------------------
# spectral_periodogram
# --------------------------------------------------------------------------


def test_spectral_periodogram_detects_known_sine_period() -> None:
    n = 500
    period_days = 20.0
    t = np.arange(n)
    signal = 0.02 * np.sin(2.0 * np.pi * t / period_days)
    r = pd.Series(signal)

    result = spectral_periodogram(r, periods_per_day=1.0, top_n=3)

    assert len(result["top_periodos_dias"]) == 3
    top_period = result["top_periodos_dias"][0][0]
    assert top_period == pytest.approx(period_days, rel=0.15)


def test_spectral_periodogram_top_periods_sorted_by_power_descending() -> None:
    n = 400
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0, 0.02, n))

    result = spectral_periodogram(r, periods_per_day=1.0, top_n=3)

    powers = [p for _, p in result["top_periodos_dias"]]
    assert powers == sorted(powers, reverse=True)


def test_spectral_periodogram_returns_empty_for_too_short_series() -> None:
    r = pd.Series([0.01])

    result = spectral_periodogram(r)

    assert result == {"frecuencias": [], "potencia": [], "top_periodos_dias": []}


def test_spectral_periodogram_hourly_periods_expressed_in_days() -> None:
    # periods_per_day=24 (retornos horarios): un ciclo de 48 muestras
    # horarias = 2 días.
    n = 600
    period_hours = 48.0
    t = np.arange(n)
    signal = 0.01 * np.sin(2.0 * np.pi * t / period_hours)
    r = pd.Series(signal)

    result = spectral_periodogram(r, periods_per_day=24.0, top_n=1)

    top_period_days = result["top_periodos_dias"][0][0]
    assert top_period_days == pytest.approx(2.0, rel=0.15)


# --------------------------------------------------------------------------
# BTC_HALVING_DATES
# --------------------------------------------------------------------------


def test_btc_halving_dates_are_sorted_valid_iso_dates() -> None:
    assert len(BTC_HALVING_DATES) == 4
    assert BTC_HALVING_DATES == sorted(BTC_HALVING_DATES)
    for date_str in BTC_HALVING_DATES:
        pd.Timestamp(date_str)  # no debe lanzar
