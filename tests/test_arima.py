"""Tests offline para models/arima.py (series simuladas, sin red ni datos de mercado)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.arima import (
    directional_forecast,
    fit_arima,
    forecast_returns,
    select_arima_order,
    walk_forward_directional_accuracy,
)


def _simulate_ar1_returns(n: int = 1200, phi: float = 0.6, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    """Serie AR(1) con coeficiente FUERTE y conocido: señal genuinamente
    predecible a 1 paso, para verificar que el módulo la detecta.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0.0, sigma)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.Series(r, index=idx)


def _simulate_white_noise_returns(n: int = 900, sigma: float = 0.01, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.0, sigma, n), index=idx)


# --------------------------------------------------------------------------
# select_arima_order
# --------------------------------------------------------------------------


def test_select_arima_order_recovers_ar_structure_on_strong_ar1() -> None:
    returns = _simulate_ar1_returns()
    order = select_arima_order(returns, max_p=3, max_q=1, criterion="aic")

    p, d, q = order
    assert d == 0
    # No exigimos recuperar exactamente (1,0,0): alcanza con que detecte
    # ALGO de estructura AR, no que se quede en un modelo sin componente AR.
    assert p >= 1


def test_select_arima_order_rejects_invalid_criterion() -> None:
    returns = _simulate_ar1_returns(n=300)
    with pytest.raises(ValueError):
        select_arima_order(returns, max_p=1, max_q=1, criterion="not_a_criterion")


# --------------------------------------------------------------------------
# fit_arima / forecast_returns / directional_forecast
# --------------------------------------------------------------------------


def test_fit_arima_with_explicit_order_skips_selection() -> None:
    returns = _simulate_ar1_returns(n=400)
    result = fit_arima(returns, order=(1, 0, 0))
    assert "ar.L1" in result.params.index


def test_forecast_returns_has_expected_columns_and_length() -> None:
    returns = _simulate_ar1_returns(n=400)
    result = fit_arima(returns, order=(1, 0, 0))

    forecast = forecast_returns(result, horizon=5)

    assert list(forecast.columns) == ["media", "lower", "upper"]
    assert len(forecast) == 5
    assert (forecast["lower"] <= forecast["media"]).all()
    assert (forecast["media"] <= forecast["upper"]).all()


def test_directional_forecast_returns_plus_or_minus_one() -> None:
    returns = _simulate_ar1_returns(n=400)
    result = fit_arima(returns, order=(1, 0, 0))

    out = directional_forecast(result, horizon=1)

    assert out["signo"] in (1, -1)
    assert out["conviccion"] >= 0
    assert out["horizonte"] == 1


# --------------------------------------------------------------------------
# walk_forward_directional_accuracy — el entregable clave
# --------------------------------------------------------------------------


def test_walk_forward_accuracy_beats_50_on_strong_predictable_ar1() -> None:
    returns = _simulate_ar1_returns(n=1200, phi=0.6)

    result = walk_forward_directional_accuracy(
        returns, order=(1, 0, 0), train_window=400, step=3, refit_every=40
    )

    assert result["n_predicciones"] > 0
    assert result["accuracy_baseline"] == 0.5
    # Señal AR fuerte y real: el acierto debería quedar claramente por
    # encima del azar, y el test binomial debería rechazar H0.
    assert result["accuracy"] > 0.55
    assert result["rechaza_h0"] is True
    assert result["p_valor"] < 0.05


def test_walk_forward_accuracy_hovers_around_50_on_white_noise() -> None:
    returns = _simulate_white_noise_returns(n=900)

    result = walk_forward_directional_accuracy(
        returns, order=(1, 0, 0), train_window=400, step=3, refit_every=40
    )

    assert result["n_predicciones"] > 0
    # Sin señal real, el acierto debería quedar cerca del 50% y el test
    # binomial NO debería rechazar H0 (resultado esperado y honesto).
    assert 0.35 < result["accuracy"] < 0.65
    assert result["rechaza_h0"] is False


def test_walk_forward_accuracy_raises_when_series_too_short() -> None:
    returns = _simulate_white_noise_returns(n=100)
    with pytest.raises(ValueError):
        walk_forward_directional_accuracy(returns, order=(1, 0, 0), train_window=400)
