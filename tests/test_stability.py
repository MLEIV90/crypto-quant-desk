"""Tests offline para pairs/stability.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs import stability
from pairs.stability import rolling_cointegration, screen_pairs_stability, stability_summary


def _random_walk_price(n: int, seed: int, start_level: float = 5.0, step_std: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_price = start_level + np.cumsum(rng.normal(0.0, step_std, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.exp(log_price), index=idx)


def _cointegrated_pair(
    n: int = 1500, seed: int = 0, beta: float = 2.0, noise_std: float = 0.01
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    log_x = 5.0 + np.cumsum(rng.normal(0.0, 0.01, n))
    stationary_noise = rng.normal(0.0, noise_std, n)
    log_y = beta * log_x + stationary_noise

    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.Series(np.exp(log_x), index=idx)
    y = pd.Series(np.exp(log_y), index=idx)
    return y, x


# --------------------------------------------------------------------------
# rolling_cointegration / stability_summary
# --------------------------------------------------------------------------


def test_stable_cointegrated_pair_has_high_cointegrated_fraction() -> None:
    y, x = _cointegrated_pair(n=1500, seed=0, beta=2.0, noise_std=0.01)

    rolling_df = rolling_cointegration(y, x, window=365, step=30)
    summary = stability_summary(rolling_df)

    assert list(rolling_df.columns) == ["fecha_fin", "p_valor_adf", "beta", "es_cointegrado"]
    assert summary["fraccion_cointegrada"] > 0.8
    assert summary["estable"] is True
    assert summary["beta_medio"] == pytest.approx(2.0, abs=0.1)


def test_independent_random_walks_have_low_cointegrated_fraction() -> None:
    # Semilla fija: dos random walks geométricos sin relación entre sí.
    x = _random_walk_price(n=1500, seed=10)
    y = _random_walk_price(n=1500, seed=20)

    rolling_df = rolling_cointegration(y, x, window=365, step=30)
    summary = stability_summary(rolling_df)

    assert summary["fraccion_cointegrada"] < 0.4
    assert summary["estable"] is False


def test_rolling_cointegration_raises_when_not_enough_history() -> None:
    y, x = _cointegrated_pair(n=100, seed=1)
    with pytest.raises(ValueError):
        rolling_cointegration(y, x, window=365, step=30)


def test_stability_summary_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        stability_summary(pd.DataFrame(columns=["fecha_fin", "p_valor_adf", "beta", "es_cointegrado"]))


# --------------------------------------------------------------------------
# screen_pairs_stability (get_prices mockeado, offline)
# --------------------------------------------------------------------------


def test_screen_pairs_stability_returns_expected_table(monkeypatch: pytest.MonkeyPatch) -> None:
    b, a = _cointegrated_pair(n=1500, seed=5, beta=2.0)
    c = _random_walk_price(n=1500, seed=6)
    close_by_asset = {"A": a, "B": b, "C": c}

    def fake_get_prices(asset: str, source: str = "store", start=None, end=None, use_cache: bool = True):
        return pd.DataFrame({"close": close_by_asset[asset]})

    monkeypatch.setattr(stability, "get_prices", fake_get_prices)

    table = screen_pairs_stability(assets=["A", "B", "C"], window=365, step=30)

    assert list(table.columns) == [
        "par", "direccion", "n_ventanas", "fraccion_cointegrada", "beta_medio", "beta_std", "estable",
    ]
    assert len(table) == 3
    assert table["fraccion_cointegrada"].is_monotonic_decreasing
    assert table.iloc[0]["par"] == "A-B"


def test_screen_pairs_stability_rejects_single_asset() -> None:
    with pytest.raises(ValueError):
        screen_pairs_stability(assets=["A"])
