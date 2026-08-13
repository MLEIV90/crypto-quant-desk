"""Tests offline para models/garch.py (todo simulado, sin red ni datos de mercado)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from models.garch import (
    conditional_volatility,
    fit_garch_variant,
    forecast_volatility,
    garch_var,
    kupiec_test,
    select_best_model,
    volatility_regime,
)


def _simulate_garch_returns(n: int = 1000, seed: int = 0) -> pd.Series:
    """Simula un GARCH(1,1) con innovaciones t-Student (6 grados de libertad),
    en escala DECIMAL, para tener una serie con clustering de volatilidad
    real y poder testear el módulo sin depender de datos de mercado.
    """
    rng = np.random.default_rng(seed)
    omega, alpha1, beta1 = 0.00001, 0.1, 0.85
    eps = np.zeros(n)
    sigma2 = np.full(n, omega / (1 - alpha1 - beta1))
    for t in range(1, n):
        sigma2[t] = omega + alpha1 * eps[t - 1] ** 2 + beta1 * sigma2[t - 1]
        eps[t] = rng.standard_t(6) * np.sqrt(sigma2[t])
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(eps, index=idx)


@pytest.fixture(scope="module")
def simulated_returns() -> pd.Series:
    return _simulate_garch_returns()


@pytest.fixture(scope="module")
def fitted_garch(simulated_returns: pd.Series):
    return fit_garch_variant(simulated_returns, vol="Garch", dist="t")


# --------------------------------------------------------------------------
# fit_garch_variant / select_best_model
# --------------------------------------------------------------------------


def test_fit_garch_variant_rejects_unknown_vol(simulated_returns: pd.Series) -> None:
    with pytest.raises(ValueError):
        fit_garch_variant(simulated_returns, vol="not_a_variant")


def test_fit_garch_variant_gjr_forces_asymmetric_term(simulated_returns: pd.Series) -> None:
    result = fit_garch_variant(simulated_returns, vol="GJR", dist="t")
    # GJR sin término asimétrico no sería GJR: debe aparecer gamma[1] en los parámetros.
    assert "gamma[1]" in result.params.index


def test_select_best_model_runs_and_returns_finite_aic_bic(simulated_returns: pd.Series) -> None:
    best = select_best_model(simulated_returns, criterion="aic")

    assert best["vol"] in ("Garch", "EGarch", "GJR")
    assert best["dist"] in ("normal", "t")
    assert np.isfinite(best["aic"])
    assert np.isfinite(best["bic"])
    assert best["result"] is not None


def test_select_best_model_rejects_invalid_criterion(simulated_returns: pd.Series) -> None:
    with pytest.raises(ValueError):
        select_best_model(simulated_returns, criterion="not_a_criterion")


# --------------------------------------------------------------------------
# conditional_volatility
# --------------------------------------------------------------------------


def test_conditional_volatility_is_positive_and_aligned(
    simulated_returns: pd.Series, fitted_garch
) -> None:
    cond_vol = conditional_volatility(fitted_garch)

    assert len(cond_vol) == len(simulated_returns)
    assert cond_vol.index.equals(simulated_returns.index)
    non_nan = cond_vol.dropna()
    assert len(non_nan) > 0
    assert (non_nan > 0).all()


# --------------------------------------------------------------------------
# forecast_volatility
# --------------------------------------------------------------------------


def test_forecast_volatility_returns_n_positive_values(fitted_garch) -> None:
    forecast = forecast_volatility(fitted_garch, horizon=5)

    assert len(forecast) == 5
    assert (forecast > 0).all()
    assert list(forecast.index) == [1, 2, 3, 4, 5]


def test_forecast_volatility_multi_step_egarch_falls_back_to_simulation(
    simulated_returns: pd.Series,
) -> None:
    # EGARCH no tiene pronóstico analítico multi-paso en `arch` (ValueError
    # interna); forecast_volatility debe recurrir a simulación en vez de
    # propagar el error.
    egarch_result = fit_garch_variant(simulated_returns, vol="EGarch", dist="t")

    forecast = forecast_volatility(egarch_result, horizon=5)

    assert len(forecast) == 5
    assert (forecast > 0).all()
    assert np.isfinite(forecast).all()


# --------------------------------------------------------------------------
# volatility_regime
# --------------------------------------------------------------------------


def test_volatility_regime_only_has_expected_labels(fitted_garch) -> None:
    cond_vol = conditional_volatility(fitted_garch)
    regime = volatility_regime(cond_vol, lookback=100)

    labels = set(regime.dropna().unique())
    assert labels.issubset({"calma", "normal", "tension"})
    assert len(labels) > 0


def test_volatility_regime_rejects_invalid_thresholds(fitted_garch) -> None:
    cond_vol = conditional_volatility(fitted_garch)
    with pytest.raises(ValueError):
        volatility_regime(cond_vol, low=0.7, high=0.3)


# --------------------------------------------------------------------------
# garch_var
# --------------------------------------------------------------------------


def test_garch_var_is_positive_and_aligned(simulated_returns: pd.Series, fitted_garch) -> None:
    var_series = garch_var(fitted_garch, alpha=0.05)

    assert len(var_series) == len(simulated_returns)
    assert (var_series > 0).all()


def test_garch_var_higher_confidence_is_more_conservative(fitted_garch) -> None:
    var_95 = garch_var(fitted_garch, alpha=0.05)
    var_99 = garch_var(fitted_garch, alpha=0.01)

    assert (var_99 >= var_95).all()


# --------------------------------------------------------------------------
# kupiec_test
# --------------------------------------------------------------------------


def test_kupiec_test_accepts_well_calibrated_var_and_rejects_optimistic_var() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    sigma = 0.02
    returns = pd.Series(rng.normal(0.0, sigma, n), index=pd.date_range("2020-01-01", periods=n, freq="D"))

    alpha = 0.05
    true_var = -scipy_stats.norm.ppf(alpha, loc=0.0, scale=sigma)
    well_calibrated_var = pd.Series(true_var, index=returns.index)

    good_result = kupiec_test(returns, well_calibrated_var, alpha=alpha)
    assert good_result["rechaza_h0"] is False
    assert good_result["n_obs"] == n

    # VaR demasiado optimista (mucho más chico que el real): exceso de excepciones.
    optimistic_var = pd.Series(true_var * 0.2, index=returns.index)
    bad_result = kupiec_test(returns, optimistic_var, alpha=alpha)
    assert bad_result["rechaza_h0"] is True
    assert bad_result["n_excepciones"] > good_result["n_excepciones"]


def test_kupiec_test_on_own_garch_var_is_reasonably_calibrated(
    simulated_returns: pd.Series, fitted_garch
) -> None:
    # El VaR paramétrico calculado a partir del propio modelo ajustado sobre
    # los datos que lo generaron debería estar razonablemente bien calibrado.
    var_series = garch_var(fitted_garch, alpha=0.05)
    result = kupiec_test(simulated_returns, var_series, alpha=0.05)

    assert result["rechaza_h0"] is False


def test_kupiec_test_raises_when_no_aligned_observations() -> None:
    returns = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-01", periods=2, freq="D"))
    var_series = pd.Series([0.01, 0.02], index=pd.date_range("2030-01-01", periods=2, freq="D"))
    with pytest.raises(ValueError):
        kupiec_test(returns, var_series)
