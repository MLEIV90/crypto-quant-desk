"""Tests offline para pairs/cointegration.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs import cointegration
from pairs.cointegration import engle_granger, find_cointegrated_pairs, half_life, hedge_ratio_ols, johansen_test


def _random_walk_price(n: int, seed: int, start_level: float = 5.0, step_std: float = 0.01) -> pd.Series:
    """Precio tipo random walk geométrico: log-precio = random walk, precio
    siempre positivo (exp del log-precio) — así se puede tomar np.log() sin
    problemas, como cualquier serie de precios real.
    """
    rng = np.random.default_rng(seed)
    log_price = start_level + np.cumsum(rng.normal(0.0, step_std, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.exp(log_price), index=idx)


def _cointegrated_pair(
    n: int = 1000, seed: int = 0, beta: float = 2.0, noise_std: float = 0.01
) -> tuple[pd.Series, pd.Series]:
    """x = random walk geométrico; y = x^beta * ruido ESTACIONARIO chico, es
    decir log(y) = beta*log(x) + ruido i.i.d. (estacionario por construcción).
    Un caso de cointegración de manual: log(y) - beta*log(x) es estacionario.
    """
    rng = np.random.default_rng(seed)
    log_x = 5.0 + np.cumsum(rng.normal(0.0, 0.01, n))
    stationary_noise = rng.normal(0.0, noise_std, n)
    log_y = beta * log_x + stationary_noise

    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.exp(log_x), index=idx)
    y = pd.Series(np.exp(log_y), index=idx)
    return y, x


# --------------------------------------------------------------------------
# engle_granger / hedge_ratio_ols
# --------------------------------------------------------------------------


def test_engle_granger_detects_cointegration_and_recovers_beta() -> None:
    y, x = _cointegrated_pair(n=1000, seed=0, beta=2.0, noise_std=0.01)

    result = engle_granger(y, x)

    assert result["es_cointegrado"] is True
    assert result["p_valor_adf"] < 0.05
    assert result["beta"] == pytest.approx(2.0, abs=0.05)
    assert len(result["spread"]) == 1000


def test_hedge_ratio_ols_matches_engle_granger_beta() -> None:
    y, x = _cointegrated_pair(n=500, seed=1, beta=1.5)

    beta_direct = hedge_ratio_ols(y, x)
    beta_from_eg = engle_granger(y, x)["beta"]

    assert beta_direct == pytest.approx(beta_from_eg)
    assert beta_direct == pytest.approx(1.5, abs=0.05)


def test_engle_granger_does_not_detect_cointegration_for_independent_random_walks() -> None:
    # Semilla fija: dos random walks geométricos SIN relación entre sí. El
    # residuo de regresionar uno sobre el otro no tiene por qué ser
    # estacionario (y con esta semilla, no lo es).
    x = _random_walk_price(n=1000, seed=10)
    y = _random_walk_price(n=1000, seed=20)

    result = engle_granger(y, x)

    assert result["es_cointegrado"] is False
    assert result["p_valor_adf"] > 0.05


def test_engle_granger_raises_no_error_but_aligns_by_common_dates() -> None:
    y, x = _cointegrated_pair(n=500, seed=2)
    y_shorter = y.iloc[100:]  # simula un activo con menos historia (p. ej. SOL)

    result = engle_granger(y_shorter, x)

    assert len(result["spread"]) == 400  # solo el rango en común


# --------------------------------------------------------------------------
# johansen_test
# --------------------------------------------------------------------------


def test_johansen_test_finds_at_least_one_relation_for_cointegrated_pair() -> None:
    y, x = _cointegrated_pair(n=800, seed=7, beta=2.0)
    df_prices = pd.DataFrame({"x": x, "y": y})

    result = johansen_test(df_prices)

    assert set(result.keys()) == {
        "n_relaciones_cointegracion", "estadisticos_traza", "valores_criticos_95", "vectores_cointegracion",
    }
    assert result["n_relaciones_cointegracion"] >= 1
    assert len(result["estadisticos_traza"]) == 2
    assert result["vectores_cointegracion"].shape == (2, 2)


def test_johansen_test_rejects_single_column() -> None:
    with pytest.raises(ValueError):
        johansen_test(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


# --------------------------------------------------------------------------
# half_life
# --------------------------------------------------------------------------


def test_half_life_of_known_ar1_spread_is_positive_and_reasonable() -> None:
    # Simulación directa de un OU discreto con theta conocido:
    # spread_t = spread_{t-1} * (1 + theta) + ruido, theta < 0 (revierte).
    rng = np.random.default_rng(3)
    n = 2000
    theta_true = -0.1
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1.0 + theta_true) + rng.normal(0.0, 0.5)
    spread_series = pd.Series(spread, index=pd.date_range("2020-01-01", periods=n, freq="D"))

    hl = half_life(spread_series)

    expected_hl = -np.log(2.0) / theta_true  # ~6.93
    assert hl > 0
    assert hl == pytest.approx(expected_hl, rel=0.3)


def test_half_life_returns_inf_when_no_mean_reversion() -> None:
    # Proceso EXPLOSIVO (theta > 0 por construcción, diverge en vez de
    # revertir). Nota: no se usa un random walk puro (theta poblacional =
    # 0) para este caso porque el estimador OLS de theta sobre una serie
    # integrada tiene un sesgo NEGATIVO conocido en muestras finitas (es la
    # razón de ser del test de Dickey-Fuller, que usa una distribución
    # especial en vez de un t-test normal) — con un random walk, `theta`
    # suele salir negativo por ese sesgo, no por reversión real, lo que
    # haría este test frágil/no determinístico.
    rng = np.random.default_rng(4)
    n = 500
    theta_true = 0.02
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1.0 + theta_true) + rng.normal(0.0, 0.5)
    spread_series = pd.Series(spread)

    hl = half_life(spread_series)

    assert hl == float("inf")


# --------------------------------------------------------------------------
# find_cointegrated_pairs (get_prices mockeado, offline)
# --------------------------------------------------------------------------


def test_find_cointegrated_pairs_returns_expected_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # A y B cointegrados (B ~ 2*A); C independiente. 3 combinaciones en total.
    b, a = _cointegrated_pair(n=800, seed=5, beta=2.0)
    c = _random_walk_price(n=800, seed=6)

    close_by_asset = {"A": a, "B": b, "C": c}

    def fake_get_prices(asset: str, source: str = "store", start=None, end=None, use_cache: bool = True):
        return pd.DataFrame({"close": close_by_asset[asset]})

    monkeypatch.setattr(cointegration, "get_prices", fake_get_prices)

    table = find_cointegrated_pairs(assets=["A", "B", "C"], source="store")

    assert list(table.columns) == ["par", "direccion", "beta", "p_valor_adf", "half_life", "es_cointegrado"]
    assert len(table) == 3  # combinaciones de a 2 entre 3 activos: A-B, A-C, B-C
    assert table["p_valor_adf"].is_monotonic_increasing
    # El par A-B (genuinamente cointegrado) debería quedar primero, con el p-valor más bajo.
    assert table.iloc[0]["par"] == "A-B"
    assert bool(table.iloc[0]["es_cointegrado"]) is True


def test_find_cointegrated_pairs_rejects_single_asset() -> None:
    with pytest.raises(ValueError):
        find_cointegrated_pairs(assets=["A"])
