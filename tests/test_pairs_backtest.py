"""Tests offline para pairs/backtest.py (Fase 15b, series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import TRANSACTION_COST_BPS
from pairs.backtest import backtest_pair, backtest_pair_rotation
from pairs.signals import generate_pair_signals
from signals.returns import simple_returns


def _prices_from_spread(spread_true: np.ndarray, growth: float = 1.0002) -> tuple[pd.Series, pd.Series]:
    """Arma un par (y_prices, x_prices) con hedge_ratio=1.0 EXACTO a partir
    de un spread NIVEL elegido a mano: `x` es una referencia suave y
    creciente, `y = x * exp(spread_true)` así `log(y) - 1.0*log(x) ==
    spread_true` por construcción — permite controlar el spread que verá
    `backtest_pair` sin pasar por una estimación de hedge_ratio.
    """
    n = len(spread_true)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    x = 100.0 * (growth ** np.arange(n))
    y = x * np.exp(spread_true)
    return pd.Series(y, index=idx), pd.Series(x, index=idx)


def _reverting_spread(n: int = 250, seed: int = 0) -> np.ndarray:
    """Spread con 3 shocks (caídas) que cada uno REVIERTE de vuelta a la
    zona plana original antes del próximo — para dar a la estrategia
    varias oportunidades reales de entrar y salir en ganancia.
    """
    spread = np.zeros(n)
    rng = np.random.default_rng(seed)

    def add_shock(start: int, depth: float, revert_len: int) -> None:
        for i in range(revert_len):
            frac = i / revert_len
            spread[start + i] += depth * np.exp(-3 * frac)

    add_shock(20, -1.0, 40)
    add_shock(100, -1.0, 40)
    add_shock(180, -1.0, 40)
    return spread + rng.normal(0, 0.01, n)


def _persistently_diverging_spread(n: int = 250, seed: int = 1) -> np.ndarray:
    """Spread que se mantiene plano al principio y después arranca un
    RANDOM WALK con drift persistente hacia abajo, sin volver nunca cerca
    de su nivel original — no hay reversión real a la que apostarle.
    """
    spread = np.zeros(n)
    rng = np.random.default_rng(seed)
    drift = -0.02
    for i in range(20, n):
        spread[i] = spread[i - 1] + drift + rng.normal(0, 0.03)
    return spread


# --------------------------------------------------------------------------
# backtest_pair
# --------------------------------------------------------------------------


def test_backtest_pair_profits_on_a_spread_that_genuinely_reverts() -> None:
    y_prices, x_prices = _prices_from_spread(_reverting_spread())

    result = backtest_pair(y_prices, x_prices, hedge_ratio=1.0)

    assert result["metrics"]["total_return"] > 0
    assert result["metrics"]["sharpe"] > 0
    assert result["metrics"]["n_trades"] > 0


def test_backtest_pair_loses_on_a_spread_that_never_reverts() -> None:
    y_prices, x_prices = _prices_from_spread(_persistently_diverging_spread())

    result = backtest_pair(y_prices, x_prices, hedge_ratio=1.0)

    assert result["metrics"]["total_return"] < 0
    assert result["metrics"]["sharpe"] < 0
    assert result["metrics"]["n_trades"] > 0  # sí entra (el z-score cruza el umbral), pero pierde


def test_backtest_pair_returns_expected_keys_and_aligned_series() -> None:
    y_prices, x_prices = _prices_from_spread(_reverting_spread())

    result = backtest_pair(y_prices, x_prices, hedge_ratio=1.0)

    assert set(result.keys()) == {"equity_curve", "returns", "metrics", "spread", "zscore", "eventos"}
    expected_metric_keys = {
        "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
        "turnover_total", "turnover_medio_diario", "n_trades", "exposicion_media", "pct_tiempo_fuera",
        "hit_rate",
    }
    assert set(result["metrics"].keys()) == expected_metric_keys
    # "spread"/"zscore"/"eventos" están alineados entre sí (misma cantidad de fechas).
    assert len(result["spread"]) == len(result["zscore"]) == len(result["eventos"])
    # la equity curve arranca en 1.0 (base) por construcción de equity_curve().
    assert result["equity_curve"].iloc[0] == pytest.approx(1.0, abs=0.05)


def test_backtest_pair_respects_entry_exit_stop_and_cost_bps_overrides() -> None:
    y_prices, x_prices = _prices_from_spread(_reverting_spread())

    loose = backtest_pair(y_prices, x_prices, hedge_ratio=1.0, entry=5.0, exit=0.5, stop=6.0)
    # Con un umbral de entrada mucho más exigente (5.0, nunca alcanzado por
    # este spread sintético), la estrategia no debería operar en absoluto.
    assert loose["metrics"]["n_trades"] == 0
    assert loose["metrics"]["total_return"] == pytest.approx(0.0)

    cheap = backtest_pair(y_prices, x_prices, hedge_ratio=1.0, cost_bps=0.0)
    expensive = backtest_pair(y_prices, x_prices, hedge_ratio=1.0, cost_bps=500.0)
    # Mismas señales, pero costos mucho más altos -> retorno total menor.
    assert expensive["metrics"]["total_return"] < cheap["metrics"]["total_return"]


def test_backtest_pair_raises_for_insufficient_overlap() -> None:
    idx_y = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    idx_x = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")  # sin fechas en común
    y_prices = pd.Series([100.0, 101, 102, 103, 104], index=idx_y)
    x_prices = pd.Series([50.0, 51, 52, 53, 54], index=idx_x)

    with pytest.raises(ValueError):
        backtest_pair(y_prices, x_prices, hedge_ratio=1.0)


# --------------------------------------------------------------------------
# backtest_pair_rotation (Fase 30: variante long-only/rotación)
# --------------------------------------------------------------------------


def test_backtest_pair_rotation_returns_expected_keys_and_aligned_series() -> None:
    y_prices, x_prices = _prices_from_spread(_reverting_spread())

    result = backtest_pair_rotation(y_prices, x_prices, hedge_ratio=1.0)

    assert set(result.keys()) == {"equity_curve", "returns", "metrics", "spread", "zscore", "eventos"}
    expected_metric_keys = {
        "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
        "turnover_total", "turnover_medio_diario", "n_trades", "exposicion_media", "pct_tiempo_fuera",
        "hit_rate",
    }
    assert set(result["metrics"].keys()) == expected_metric_keys
    assert len(result["spread"]) == len(result["zscore"]) == len(result["eventos"])
    assert result["equity_curve"].iloc[0] == pytest.approx(1.0, abs=0.05)


def test_backtest_pair_rotation_matches_manual_long_only_formula() -> None:
    """Reconstruye el retorno esperado a mano (sin pasar por la función) a
    partir de la MISMA señal de `generate_pair_signals` — verifica que
    `backtest_pair_rotation` de verdad rota 100% entre Y/X sin nunca ir en
    corto (nunca las dos patas activas a la vez) ni apalancar (pesos nunca
    superan 1.0).
    """
    y_prices, x_prices = _prices_from_spread(_reverting_spread())
    hedge_ratio = 1.0

    result = backtest_pair_rotation(y_prices, x_prices, hedge_ratio=hedge_ratio)

    aligned = pd.concat([y_prices.rename("y"), x_prices.rename("x")], axis=1).dropna()
    spread = np.log(aligned["y"]) - hedge_ratio * np.log(aligned["x"])
    # mismos defaults que `backtest_pair_rotation` (DEFAULT_PAIR_*, no los
    # de `pairs.signals` — difieren en el stop, ver docstring del módulo).
    signals_df = generate_pair_signals(spread, entry=2.0, exit=0.5, stop=3.0)
    y_returns = simple_returns(aligned["y"])
    x_returns = simple_returns(aligned["x"])
    posicion = signals_df["posicion_spread"].reindex(y_returns.index)
    position_effective = posicion.shift(1).fillna(0.0)
    peso_y = position_effective.clip(lower=0.0)
    peso_x = (-position_effective).clip(lower=0.0)
    turnover = position_effective.diff().fillna(0.0).abs()
    expected_returns = (peso_y * y_returns + peso_x * x_returns) - (TRANSACTION_COST_BPS / 1e4) * turnover

    pd.testing.assert_series_equal(result["returns"], expected_returns.rename("strategy_return"))
    # Nunca las dos patas a la vez, y nunca apalancado por encima de 1.0.
    assert ((peso_y > 0) & (peso_x > 0)).sum() == 0
    assert peso_y.max() <= 1.0
    assert peso_x.max() <= 1.0


def test_backtest_pair_rotation_respects_entry_exit_stop_and_cost_bps_overrides() -> None:
    y_prices, x_prices = _prices_from_spread(_reverting_spread())

    loose = backtest_pair_rotation(y_prices, x_prices, hedge_ratio=1.0, entry=5.0, exit=0.5, stop=6.0)
    assert loose["metrics"]["n_trades"] == 0
    assert loose["metrics"]["total_return"] == pytest.approx(0.0)

    cheap = backtest_pair_rotation(y_prices, x_prices, hedge_ratio=1.0, cost_bps=0.0)
    expensive = backtest_pair_rotation(y_prices, x_prices, hedge_ratio=1.0, cost_bps=500.0)
    assert expensive["metrics"]["total_return"] < cheap["metrics"]["total_return"]


def test_backtest_pair_rotation_raises_for_insufficient_overlap() -> None:
    idx_y = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    idx_x = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
    y_prices = pd.Series([100.0, 101, 102, 103, 104], index=idx_y)
    x_prices = pd.Series([50.0, 51, 52, 53, 54], index=idx_x)

    with pytest.raises(ValueError):
        backtest_pair_rotation(y_prices, x_prices, hedge_ratio=1.0)
