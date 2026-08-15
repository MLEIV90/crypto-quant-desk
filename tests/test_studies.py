"""Tests offline para signals/studies.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.studies import (
    adx,
    all_studies,
    fibonacci_levels,
    pivot_points,
    stochastic,
    support_resistance,
    swing_points,
)


def _synthetic_ohlcv_oscillating(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """OHLCV sintético oscilante (picos ~150, valles ~100): pensado para
    ejercitar swing highs/lows, soporte/resistencia, estocástico y ADX con
    suficiente variación de rango (evita el caso degenerado high==low).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 125.0 + 25.0 * np.sin(2 * np.pi * t / 20.0)
    noise = rng.normal(0.0, 0.5, n)
    close = base + noise
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1000.0}, index=idx
    )


# --------------------------------------------------------------------------
# fibonacci_levels
# --------------------------------------------------------------------------


def test_fibonacci_levels_uptrend_gives_correct_ratios() -> None:
    levels = fibonacci_levels(high=200.0, low=100.0, uptrend=True)

    assert levels["0.0%"] == pytest.approx(200.0)
    assert levels["23.6%"] == pytest.approx(200.0 - 0.236 * 100.0)
    assert levels["38.2%"] == pytest.approx(200.0 - 0.382 * 100.0)
    assert levels["50.0%"] == pytest.approx(150.0)
    assert levels["61.8%"] == pytest.approx(200.0 - 0.618 * 100.0)
    assert levels["78.6%"] == pytest.approx(200.0 - 0.786 * 100.0)
    assert levels["100.0%"] == pytest.approx(100.0)
    assert levels["127.2%"] == pytest.approx(100.0 + 1.272 * 100.0)
    assert levels["161.8%"] == pytest.approx(100.0 + 1.618 * 100.0)


def test_fibonacci_levels_downtrend_gives_correct_ratios() -> None:
    levels = fibonacci_levels(high=200.0, low=100.0, uptrend=False)

    assert levels["0.0%"] == pytest.approx(100.0)
    assert levels["50.0%"] == pytest.approx(150.0)
    assert levels["100.0%"] == pytest.approx(200.0)
    assert levels["127.2%"] == pytest.approx(200.0 - 1.272 * 100.0)
    assert levels["161.8%"] == pytest.approx(200.0 - 1.618 * 100.0)


def test_fibonacci_levels_rejects_high_not_greater_than_low() -> None:
    with pytest.raises(ValueError):
        fibonacci_levels(high=100.0, low=100.0)
    with pytest.raises(ValueError):
        fibonacci_levels(high=90.0, low=100.0)


# --------------------------------------------------------------------------
# swing_points
# --------------------------------------------------------------------------


def test_swing_points_detects_known_single_peak_and_valley() -> None:
    idx = pd.date_range("2020-01-01", periods=41, freq="D", tz="UTC")
    high_values = list(range(100, 121)) + list(range(119, 99, -1))  # pico único en pos 20 (valor 120)
    low_values = list(range(50, 29, -1)) + list(range(31, 51))  # valle único en pos 20 (valor 30)
    high = pd.Series(high_values, index=idx, dtype=float)
    low = pd.Series(low_values, index=idx, dtype=float)
    close = (high + low) / 2.0

    swings = swing_points(high, low, close, window=10)

    assert swings["is_swing_high"].iloc[20]
    assert swings["swing_high"].iloc[20] == pytest.approx(120.0)
    assert not swings["is_swing_high"].iloc[10]

    assert swings["is_swing_low"].iloc[20]
    assert swings["swing_low"].iloc[20] == pytest.approx(30.0)
    assert not swings["is_swing_low"].iloc[10]


# --------------------------------------------------------------------------
# support_resistance
# --------------------------------------------------------------------------


def test_support_resistance_returns_sorted_levels_within_n_levels() -> None:
    df = _synthetic_ohlcv_oscillating(n=200, seed=1)
    sr = support_resistance(df["close"], df["high"], df["low"], n_levels=3, window=5)

    assert len(sr["resistencia"]) <= 3
    assert len(sr["soporte"]) <= 3
    assert sr["resistencia"] == sorted(sr["resistencia"])
    assert sr["soporte"] == sorted(sr["soporte"])
    assert sr["precio_actual"] == pytest.approx(df["close"].iloc[-1])
    # picos ~150, valles ~100: la resistencia más floja debería seguir por
    # encima del soporte más alto.
    if sr["resistencia"] and sr["soporte"]:
        assert min(sr["resistencia"]) > max(sr["soporte"])


def test_support_resistance_empty_when_no_swings_detected() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0], index=idx)
    sr = support_resistance(close, close, close, n_levels=3, window=20)

    assert sr["resistencia"] == []
    assert sr["soporte"] == []


# --------------------------------------------------------------------------
# stochastic
# --------------------------------------------------------------------------


def test_stochastic_is_bounded_in_0_100() -> None:
    df = _synthetic_ohlcv_oscillating(n=150, seed=2)
    stoch_df = stochastic(df["high"], df["low"], df["close"], k=14, d=3)

    valid_k = stoch_df["stoch_k"].dropna()
    valid_d = stoch_df["stoch_d"].dropna()
    assert len(valid_k) > 0
    assert len(valid_d) > 0
    assert ((valid_k >= 0.0) & (valid_k <= 100.0)).all()
    assert ((valid_d >= 0.0) & (valid_d <= 100.0)).all()


def test_stochastic_is_nan_on_degenerate_zero_range() -> None:
    idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    flat = pd.Series(100.0, index=idx)

    stoch_df = stochastic(flat, flat, flat, k=5, d=2)

    assert stoch_df["stoch_k"].dropna().empty


# --------------------------------------------------------------------------
# adx
# --------------------------------------------------------------------------


def test_adx_is_non_negative() -> None:
    df = _synthetic_ohlcv_oscillating(n=150, seed=3)
    adx_series = adx(df["high"], df["low"], df["close"], window=14)

    valid = adx_series.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all()


# --------------------------------------------------------------------------
# pivot_points
# --------------------------------------------------------------------------


def test_pivot_points_matches_classic_formula() -> None:
    pivots = pivot_points(high=110.0, low=90.0, close=100.0)

    expected_p = (110.0 + 90.0 + 100.0) / 3.0
    assert pivots["P"] == pytest.approx(expected_p)
    assert pivots["R1"] == pytest.approx(2.0 * expected_p - 90.0)
    assert pivots["S1"] == pytest.approx(2.0 * expected_p - 110.0)
    assert pivots["R2"] == pytest.approx(expected_p + 20.0)
    assert pivots["S2"] == pytest.approx(expected_p - 20.0)
    assert pivots["R3"] == pytest.approx(110.0 + 2.0 * (expected_p - 90.0))
    assert pivots["S3"] == pytest.approx(90.0 - 2.0 * (110.0 - expected_p))


# --------------------------------------------------------------------------
# all_studies
# --------------------------------------------------------------------------


def test_all_studies_returns_expected_structure() -> None:
    df = _synthetic_ohlcv_oscillating(n=200, seed=4)
    summary = all_studies(df)

    assert set(summary.keys()) == {
        "fecha", "precio", "indicadores", "estocastico", "adx",
        "soporte_resistencia", "pivotes", "fibonacci",
    }
    assert set(summary["indicadores"].keys()) == {
        "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pct_b", "bb_zscore", "sma_20", "sma_50", "atr_14",
    }
    assert set(summary["estocastico"].keys()) == {"k", "d"}
    assert set(summary["pivotes"].keys()) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}
    assert summary["fecha"] == df.index[-1]
    assert summary["precio"] == pytest.approx(df["close"].iloc[-1])
