"""Tests offline para eda/eda_report.py (solo las funciones no gráficas)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda.eda_report import DESCRIPTIVE_QUANTILES, adf_test, correlation_matrix, descriptive_stats, ljung_box_squared


def test_descriptive_stats_returns_expected_columns_and_single_row() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.02, 500))

    out = descriptive_stats(r, name="BTC")

    expected_cols = {"n", "media", "std", "skew", "kurtosis_exceso", "min", "max"}
    expected_cols |= {f"q{int(round(q * 100)):02d}" for q in DESCRIPTIVE_QUANTILES}
    assert expected_cols.issubset(set(out.columns))
    assert list(out.index) == ["BTC"]
    assert out.loc["BTC", "n"] == 500


def test_descriptive_stats_uses_series_name_when_no_name_given() -> None:
    r = pd.Series([0.01, -0.02, 0.03], name="ETH")
    out = descriptive_stats(r)
    assert list(out.index) == ["ETH"]


def test_descriptive_stats_defaults_to_returns_label() -> None:
    r = pd.Series([0.01, -0.02, 0.03])
    out = descriptive_stats(r)
    assert list(out.index) == ["returns"]


def test_adf_test_distinguishes_stationary_series_from_random_walk() -> None:
    rng = np.random.default_rng(1)
    n = 1000

    # Serie claramente estacionaria: ruido blanco alrededor de una media constante.
    stationary = pd.Series(rng.normal(0.0, 1.0, n))

    # Random walk: suma acumulada de ruido blanco (raíz unitaria por construcción).
    random_walk = pd.Series(np.cumsum(rng.normal(0.0, 1.0, n)))

    stationary_result = adf_test(stationary)
    random_walk_result = adf_test(random_walk)

    assert stationary_result["es_estacionaria"] is True
    assert random_walk_result["es_estacionaria"] is False
    assert stationary_result["p_valor"] < random_walk_result["p_valor"]


def test_adf_test_returns_expected_keys() -> None:
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0, 1.0, 200))
    result = adf_test(r)

    assert set(result.keys()) == {
        "estadistico", "p_valor", "n_lags", "n_obs", "valores_criticos", "es_estacionaria",
    }
    assert set(result["valores_criticos"].keys()) == {"1%", "5%", "10%"}


def test_ljung_box_squared_detects_volatility_clustering() -> None:
    # Simulación simple tipo ARCH(1): la varianza de hoy depende del retorno
    # al cuadrado de ayer, lo que genera clustering de volatilidad y
    # autocorrelación fuerte en los retornos al cuadrado.
    rng = np.random.default_rng(3)
    n = 1000
    r = np.zeros(n)
    sigma2 = np.ones(n)
    for t in range(1, n):
        sigma2[t] = 0.05 + 0.9 * r[t - 1] ** 2
        r[t] = rng.normal(0.0, np.sqrt(sigma2[t]))
    returns = pd.Series(r)

    result = ljung_box_squared(returns, lags=10)

    assert result["hay_clustering_volatilidad"] is True
    assert result["p_valor"] < 0.05
    assert len(result["tabla"]) == 10


def test_ljung_box_squared_no_clustering_for_iid_noise() -> None:
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.0, 1.0, 500))

    result = ljung_box_squared(returns, lags=10)

    assert result["hay_clustering_volatilidad"] is False


def test_correlation_matrix_aligns_by_date_and_diagonal_is_one() -> None:
    idx_a = pd.date_range("2021-01-01", periods=10, freq="D")
    idx_b = pd.date_range("2021-01-03", periods=10, freq="D")  # desplazado, menos overlap

    a = pd.Series(np.arange(10, dtype=float), index=idx_a)
    b = pd.Series(np.arange(10, dtype=float) * 2.0, index=idx_b)  # perfectamente correlacionado donde solapa

    corr = correlation_matrix({"A": a, "B": b})

    assert list(corr.columns) == ["A", "B"]
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_correlation_matrix_spearman_is_robust_to_monotonic_nonlinearity() -> None:
    # b = a**3 no es una relación LINEAL (pearson < 1) pero SÍ es monótona
    # (spearman = 1) -- ejercita que method="spearman" de verdad cambia el
    # resultado, no solo se ignora.
    idx = pd.date_range("2021-01-01", periods=30, freq="D")
    a = pd.Series(np.linspace(-1.0, 1.0, 30), index=idx)
    b = a**3

    pearson = correlation_matrix({"A": a, "B": b}, method="pearson")
    spearman = correlation_matrix({"A": a, "B": b}, method="spearman")

    assert spearman.loc["A", "B"] == pytest.approx(1.0)
    assert pearson.loc["A", "B"] < 0.999
