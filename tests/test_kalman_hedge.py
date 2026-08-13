"""Tests offline para pairs/kalman_hedge.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.kalman_hedge import kalman_hedge_ratio


def test_kalman_hedge_ratio_returns_expected_columns_and_index() -> None:
    rng = np.random.default_rng(0)
    n = 300
    log_x = 5.0 + np.cumsum(rng.normal(0.0, 0.01, n))
    log_y = 1.5 * log_x + rng.normal(0.0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.exp(log_x), index=idx)
    y = pd.Series(np.exp(log_y), index=idx)

    result = kalman_hedge_ratio(y, x)

    assert list(result.columns) == ["beta", "alpha", "spread"]
    assert result.index.equals(idx)
    assert not result["beta"].isna().any()


def test_kalman_hedge_ratio_tracks_a_step_change_in_beta() -> None:
    # y = beta_true * x + ruido, con beta_true en escalón: 1.0 -> 2.5 a
    # mitad de la muestra. El beta estimado debería moverse claramente hacia
    # el nuevo nivel tras el cambio.
    rng = np.random.default_rng(0)
    n = 1000
    log_x = 5.0 + np.cumsum(rng.normal(0.0, 0.03, n))
    x = np.exp(log_x)

    beta_true = np.where(np.arange(n) < n // 2, 1.0, 2.5)
    noise = rng.normal(0.0, 0.01, n)
    log_y = beta_true * log_x + noise
    y = np.exp(log_y)

    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    y = pd.Series(y, index=idx)
    x = pd.Series(x, index=idx)

    result = kalman_hedge_ratio(y, x, delta=1e-2, r=1e-3)

    beta_before = result["beta"].iloc[n // 2 - 50 : n // 2].mean()
    beta_after = result["beta"].iloc[-50:].mean()

    assert beta_before == pytest.approx(1.0, abs=0.2)
    # No se exige convergencia exacta a 2.5: cuando log(x) tiene un nivel
    # alto y poco variable (el caso realista, ver docstring de
    # kalman_hedge_ratio), beta y alpha quedan correlacionados en la
    # covarianza del filtro y la convergencia post-cambio es más lenta.
    # Alcanza con que se haya movido claramente hacia el nuevo nivel.
    assert beta_after > 2.0
    assert abs(beta_after - 2.5) < abs(beta_before - 2.5)


def test_kalman_hedge_ratio_beta_matches_ols_when_beta_is_constant() -> None:
    # Con beta constante (sin escalón), el promedio del beta filtrado
    # debería quedar razonablemente cerca del beta OLS estático.
    rng = np.random.default_rng(2)
    n = 600
    log_x = 5.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    log_y = 1.8 * log_x + rng.normal(0.0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.exp(log_x), index=idx)
    y = pd.Series(np.exp(log_y), index=idx)

    result = kalman_hedge_ratio(y, x, delta=1e-4, r=1e-3)

    beta_mean_second_half = result["beta"].iloc[n // 2 :].mean()
    assert beta_mean_second_half == pytest.approx(1.8, abs=0.15)


def test_kalman_hedge_ratio_spread_has_no_lookahead() -> None:
    # El spread (innovación) del día t no debería cambiar si se recorta la
    # serie después de t: es una propiedad de "solo pasado", igual que en
    # backtest/engine.py y signals/engine.py.
    rng = np.random.default_rng(3)
    n = 400
    log_x = 5.0 + np.cumsum(rng.normal(0.0, 0.01, n))
    log_y = 1.2 * log_x + rng.normal(0.0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.exp(log_x), index=idx)
    y = pd.Series(np.exp(log_y), index=idx)

    full_result = kalman_hedge_ratio(y, x)
    cutoff = n - 50
    truncated_result = kalman_hedge_ratio(y.iloc[:cutoff], x.iloc[:cutoff])

    pd.testing.assert_frame_equal(full_result.iloc[:cutoff], truncated_result)
