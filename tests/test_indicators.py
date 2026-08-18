"""Tests offline para signals/indicators.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.indicators import add_all_indicators, atr, bollinger_bands, ema, macd, obv, rsi, sma, vwap


def _monotonic_close(n: int = 60, start: float = 100.0, step: float = 1.0) -> pd.Series:
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.Series(start + step * np.arange(n), index=idx)


# --------------------------------------------------------------------------
# RSI
# --------------------------------------------------------------------------


def test_rsi_bounded_between_0_and_100() -> None:
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    r = rsi(close, window=14).dropna()

    assert (r >= 0).all()
    assert (r <= 100).all()


def test_rsi_monotonic_increasing_series_approaches_100() -> None:
    close = _monotonic_close()
    r = rsi(close, window=14).dropna()

    assert r.iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_rsi_flat_price_is_neutral_50() -> None:
    close = pd.Series([100.0] * 30)
    r = rsi(close, window=14).dropna()

    assert np.allclose(r.to_numpy(), 50.0)


# --------------------------------------------------------------------------
# SMA / EMA
# --------------------------------------------------------------------------


def test_sma_matches_manual_rolling_mean() -> None:
    close = pd.Series(np.arange(1, 11, dtype=float))
    result = sma(close, window=3)
    expected = close.rolling(window=3).mean()

    pd.testing.assert_series_equal(result, expected)


def test_ema_reacts_faster_than_sma_to_a_level_shift() -> None:
    close = pd.Series([100.0] * 20 + [110.0] * 5)
    s = sma(close, window=10)
    e = ema(close, span=10)

    assert abs(e.iloc[-1] - 110.0) < abs(s.iloc[-1] - 110.0)


# --------------------------------------------------------------------------
# MACD
# --------------------------------------------------------------------------


def test_macd_columns_and_hist_equals_macd_minus_signal() -> None:
    close = _monotonic_close(n=80)
    out = macd(close)

    assert list(out.columns) == ["macd", "signal", "hist"]
    diff = (out["hist"] - (out["macd"] - out["signal"])).dropna()
    assert (diff.abs() < 1e-9).all()


# --------------------------------------------------------------------------
# Bollinger bands
# --------------------------------------------------------------------------


def test_bollinger_zscore_is_zero_when_price_equals_moving_average() -> None:
    # Si el último precio es exactamente el promedio de los `window - 1`
    # precios anteriores, entonces también es el promedio de toda la ventana
    # (incluyéndose a sí mismo): mean_new = (sum_prev + x)/w, y con
    # x = sum_prev/(w-1) se cumple mean_new = x. Así evitamos el problema de
    # que cambiar el último precio también mueve la media que lo contiene.
    window = 20
    rng = np.random.default_rng(4)
    prev_values = 100.0 + rng.normal(0.0, 2.0, window - 1)
    last_value = prev_values.mean()
    close = pd.Series(np.append(prev_values, last_value))

    bb = bollinger_bands(close, window=window, num_std=2.0)

    assert bb["zscore"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_bollinger_bands_columns() -> None:
    close = _monotonic_close(n=40)
    bb = bollinger_bands(close)

    assert list(bb.columns) == ["mid", "upper", "lower", "pct_b", "bandwidth", "zscore"]
    assert (bb["upper"].dropna() >= bb["mid"].dropna()).all()
    assert (bb["lower"].dropna() <= bb["mid"].dropna()).all()


# --------------------------------------------------------------------------
# ATR
# --------------------------------------------------------------------------


def test_atr_degrades_to_close_diff_when_high_low_close_are_equal() -> None:
    close = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 2, 40)))
    high = close.copy()
    low = close.copy()

    result = atr(high, low, close, window=14)

    expected_tr = (close - close.shift(1)).abs()
    expected_atr = expected_tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected_atr, check_names=False)


def test_atr_is_non_negative() -> None:
    idx = pd.date_range("2021-01-01", periods=40, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 40)), index=idx)
    high = close + 1.0
    low = close - 1.0

    result = atr(high, low, close, window=14).dropna()
    assert (result >= 0).all()


# --------------------------------------------------------------------------
# VWAP
# --------------------------------------------------------------------------


def test_vwap_matches_manual_calculation_on_constant_volume() -> None:
    n = 30
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(10).normal(0, 1, n)), index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1000.0, index=idx)

    result = vwap(high, low, close, volume, window=10)

    typical_price = (high + low + close) / 3.0
    expected = typical_price.rolling(window=10).mean()  # con volumen constante, VWAP == SMA del precio típico
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_vwap_weights_high_volume_candles_more() -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    high = pd.Series([10.0, 10.0, 20.0], index=idx)
    low = high.copy()
    close = high.copy()
    volume = pd.Series([1.0, 1.0, 100.0], index=idx)  # última vela domina por volumen

    result = vwap(high, low, close, volume, window=3)

    assert result.iloc[-1] == pytest.approx(19.80392156862745, rel=1e-9)


def test_vwap_is_nan_when_window_volume_is_zero() -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    close = pd.Series(100.0, index=idx)
    volume = pd.Series(0.0, index=idx)

    result = vwap(close, close, close, volume, window=3)
    assert result.dropna().empty


# --------------------------------------------------------------------------
# OBV
# --------------------------------------------------------------------------


def test_obv_accumulates_signed_volume() -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    close = pd.Series([100.0, 101.0, 99.0, 99.0, 105.0], index=idx)  # sube, baja, plano, sube
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)

    result = obv(close, volume)

    expected = pd.Series(
        [10.0, 10.0 + 20.0, 30.0 - 30.0, 0.0, 0.0 + 50.0], index=idx
    )
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_obv_first_value_is_first_volume() -> None:
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    close = pd.Series([50.0, 51.0, 52.0, 53.0], index=idx)
    volume = pd.Series([5.0, 6.0, 7.0, 8.0], index=idx)

    result = obv(close, volume)
    assert result.iloc[0] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# add_all_indicators
# --------------------------------------------------------------------------


def test_add_all_indicators_returns_copy_with_expected_columns() -> None:
    n = 60
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, n)), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0},
        index=idx,
    )

    out = add_all_indicators(df)

    expected_new_cols = {
        "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "bb_mid", "bb_upper", "bb_lower", "bb_pct_b", "bb_bandwidth", "bb_zscore",
        "atr_14", "vwap", "obv",
    }
    assert expected_new_cols.issubset(set(out.columns))
    assert set(df.columns).issubset(set(out.columns))
    # No debe mutar el DataFrame original.
    assert "sma_20" not in df.columns
