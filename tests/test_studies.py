"""Tests offline para signals/studies.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.studies import (
    adx,
    all_studies,
    fibonacci_levels,
    ichimoku,
    pivot_points,
    stochastic,
    support_resistance,
    swing_points,
    volume_profile,
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
# ichimoku
# --------------------------------------------------------------------------


def test_ichimoku_returns_expected_columns() -> None:
    df = _synthetic_ohlcv_oscillating(n=200, seed=5)
    out = ichimoku(df["high"], df["low"], df["close"])

    assert list(out.columns) == ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"]
    assert len(out) == len(df)


def test_ichimoku_tenkan_matches_manual_high_low_average() -> None:
    df = _synthetic_ohlcv_oscillating(n=60, seed=6)
    out = ichimoku(df["high"], df["low"], df["close"], tenkan_window=9)

    expected_tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2.0
    pd.testing.assert_series_equal(out["tenkan"], expected_tenkan, check_names=False)


def test_ichimoku_senkou_a_is_shifted_forward_and_not_lookahead() -> None:
    # senkou_a[t] debe ser exactamente (tenkan+kijun)/2 calculado en
    # t - kijun_window (dato ya disponible en ese momento) — no algo nuevo
    # inventado en la posición desplazada.
    df = _synthetic_ohlcv_oscillating(n=120, seed=7)
    kijun_window = 26
    out = ichimoku(df["high"], df["low"], df["close"], kijun_window=kijun_window)

    raw_senkou_a = (out["tenkan"] + out["kijun"]) / 2.0
    shifted = raw_senkou_a.shift(kijun_window)
    pd.testing.assert_series_equal(out["senkou_a"], shifted, check_names=False)


def test_ichimoku_chikou_uses_future_close_and_tail_is_nan() -> None:
    # chikou[t] = close[t + kijun_window]: comparar contra un shift(-kijun_window)
    # manual, y confirmar que los últimos kijun_window valores son NaN (no
    # hay close futuro más allá del final de la serie).
    df = _synthetic_ohlcv_oscillating(n=60, seed=8)
    kijun_window = 26
    out = ichimoku(df["high"], df["low"], df["close"], kijun_window=kijun_window)

    expected_chikou = df["close"].shift(-kijun_window)
    pd.testing.assert_series_equal(out["chikou"], expected_chikou, check_names=False)
    assert out["chikou"].iloc[-kijun_window:].isna().all()


# --------------------------------------------------------------------------
# all_studies
# --------------------------------------------------------------------------


def test_all_studies_returns_expected_structure() -> None:
    df = _synthetic_ohlcv_oscillating(n=200, seed=4)
    summary = all_studies(df)

    assert set(summary.keys()) == {
        "fecha", "precio", "indicadores", "estocastico", "adx", "ichimoku",
        "soporte_resistencia", "pivotes", "fibonacci",
    }
    assert set(summary["indicadores"].keys()) == {
        "rsi_14", "macd", "macd_signal", "macd_hist", "bb_pct_b", "bb_zscore", "sma_20", "sma_50", "atr_14",
        "vwap", "obv",
    }
    assert set(summary["estocastico"].keys()) == {"k", "d"}
    assert set(summary["ichimoku"].keys()) == {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou"}
    assert set(summary["pivotes"].keys()) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}
    assert summary["fecha"] == df.index[-1]
    assert summary["precio"] == pytest.approx(df["close"].iloc[-1])
    # chikou en la última vela es siempre None por construcción (usa close
    # futuro, ver ichimoku()) — no es un bug, está documentado.
    assert summary["ichimoku"]["chikou"] is None


# --------------------------------------------------------------------------
# volume_profile (Fase 13a)
# --------------------------------------------------------------------------


def _flat_ohlcv(prices: list[float], volumes: list[float], price_width: float = 0.5) -> pd.DataFrame:
    """OHLCV sintético con rango de precio angosto por vela (`high =
    price + width/2`, `low = price - width/2`) — cada vela reparte su
    volumen casi enteramente en el nivel que contiene `price`, útil para
    controlar a mano DÓNDE debería caer el volumen en `volume_profile`.
    """
    n = len(prices)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    prices_arr = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": prices_arr,
            "close": prices_arr,
            "high": prices_arr + price_width / 2.0,
            "low": prices_arr - price_width / 2.0,
            "volume": volumes,
        },
        index=idx,
    )


def test_volume_profile_poc_falls_where_volume_concentrated() -> None:
    # 90 velas cerca de 100 con volumen chico, 10 velas cerca de 200 con
    # volumen ENORME -> el POC debe caer cerca de 200, no de 100, aunque
    # haya muchas más VELAS cerca de 100 (el POC es por VOLUMEN, no por
    # cantidad de velas).
    rng = np.random.default_rng(0)
    low_prices = 100.0 + rng.normal(0, 0.3, 90)
    high_prices = 200.0 + rng.normal(0, 0.3, 10)
    prices = np.concatenate([low_prices, high_prices])
    volumes = np.concatenate([np.full(90, 10.0), np.full(10, 5000.0)])
    df = _flat_ohlcv(prices.tolist(), volumes.tolist())

    profile = volume_profile(df, bins=50)

    assert profile["poc"] == pytest.approx(200.0, abs=2.0)


def test_volume_profile_conserves_total_volume() -> None:
    # El volumen total repartido entre niveles no puede perderse ni
    # inflarse respecto del volumen real de las velas.
    df = _synthetic_ohlcv_oscillating(n=200, seed=1)
    profile = volume_profile(df, bins=40)

    assert sum(profile["volumenes"]) == pytest.approx(profile["volumen_total"])
    assert profile["volumen_total"] == pytest.approx(df["volume"].sum())


def test_volume_profile_value_area_covers_target_fraction() -> None:
    df = _synthetic_ohlcv_oscillating(n=200, seed=2)
    profile = volume_profile(df, bins=50, value_area_fraction=0.70)

    niveles = np.array(profile["niveles_precio"])
    volumenes = np.array(profile["volumenes"])
    in_value_area = (niveles >= profile["value_area_low"]) & (niveles <= profile["value_area_high"])
    coverage = volumenes[in_value_area].sum() / volumenes.sum()

    assert coverage >= 0.70
    # con la distribución sintética (oscilante, no uniforme), el 70% del
    # volumen tiene que caber en MENOS niveles que el total, si no el value
    # area no estaría concentrando nada.
    assert in_value_area.sum() < len(niveles)


def test_volume_profile_value_area_contains_poc() -> None:
    df = _synthetic_ohlcv_oscillating(n=200, seed=3)
    profile = volume_profile(df, bins=50)

    assert profile["value_area_low"] <= profile["poc"] <= profile["value_area_high"]


def test_volume_profile_returns_requested_bins_ascending() -> None:
    df = _synthetic_ohlcv_oscillating(n=100, seed=5)
    profile = volume_profile(df, bins=30)

    assert len(profile["niveles_precio"]) == 30
    assert len(profile["volumenes"]) == 30
    assert profile["niveles_precio"] == sorted(profile["niveles_precio"])


def test_volume_profile_handles_some_zero_range_candles() -> None:
    # Mezcla velas con rango 0 (high==low) y velas con rango real: no debe
    # perder volumen ni romper (ver la rama "degenerate" del cálculo).
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 150.0, 150.0, 200.0],
            "close": [100.0, 100.0, 150.0, 150.0, 200.0],
            "high": [100.0, 100.0, 160.0, 150.0, 210.0],
            "low": [100.0, 100.0, 140.0, 150.0, 190.0],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC"),
    )

    profile = volume_profile(df, bins=20)

    assert sum(profile["volumenes"]) == pytest.approx(150.0)


def test_volume_profile_degenerate_single_price_does_not_crash() -> None:
    df = _flat_ohlcv([100.0] * 20, [10.0] * 20, price_width=0.0)

    profile = volume_profile(df, bins=10)

    assert profile["poc"] == pytest.approx(100.0)
    assert profile["value_area_low"] == pytest.approx(100.0)
    assert profile["value_area_high"] == pytest.approx(100.0)
    assert profile["volumen_total"] == pytest.approx(200.0)


def test_volume_profile_empty_df_raises() -> None:
    df = _synthetic_ohlcv_oscillating(n=10, seed=0).iloc[0:0]
    with pytest.raises(ValueError):
        volume_profile(df)


def test_volume_profile_invalid_bins_raises() -> None:
    df = _synthetic_ohlcv_oscillating(n=10, seed=0)
    with pytest.raises(ValueError):
        volume_profile(df, bins=0)
