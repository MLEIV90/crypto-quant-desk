"""Tests offline para backtest/engine.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import compare_to_buy_and_hold, run_backtest
from metrics.risk_measures import equity_curve, sharpe_ratio


def _synthetic_returns(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.0005, 0.02, n), index=idx)


# --------------------------------------------------------------------------
# Siempre-largo == buy & hold (con el desfase anti-lookahead)
# --------------------------------------------------------------------------


def test_always_long_with_zero_cost_matches_asset_returns_after_warmup_day() -> None:
    asset_returns = _synthetic_returns()
    positions = pd.Series(1.0, index=asset_returns.index)

    result = run_backtest(asset_returns, positions, cost_bps=0.0)

    # positions=1 es una serie CONSTANTE, así que shift(1) no la cambia (solo
    # introduce el 0 inicial): la estrategia "siempre largo" debería
    # reducirse exactamente al retorno del activo tal cual, salvo el primer
    # día (sin posición previa, retorno de estrategia = 0 ese día).
    expected_returns = asset_returns.copy()
    expected_returns.iloc[0] = 0.0
    expected_returns.name = "strategy_return"
    pd.testing.assert_series_equal(result.returns, expected_returns, check_names=False)

    expected_equity = equity_curve(expected_returns)
    pd.testing.assert_series_equal(result.equity_curve, expected_equity, check_names=False)


def test_compare_to_buy_and_hold_matches_when_strategy_is_buy_and_hold() -> None:
    asset_returns = _synthetic_returns()
    positions = pd.Series(1.0, index=asset_returns.index)
    result = run_backtest(asset_returns, positions, cost_bps=0.0)

    comparison = compare_to_buy_and_hold(asset_returns, result, cost_bps=0.0)

    assert set(comparison.keys()) == {"estrategia", "buy_and_hold"}
    assert comparison["estrategia"] == pytest.approx(comparison["buy_and_hold"])


# --------------------------------------------------------------------------
# Previsión perfecta: el motor premia el acierto
# --------------------------------------------------------------------------


def test_perfect_foresight_position_gives_high_sharpe_and_zero_drawdown() -> None:
    asset_returns = _synthetic_returns()

    # SOLO PARA EL TEST: usar el retorno FUTURO (shift(-1)) para construir la
    # posición es un cheat de lookahead deliberado. Como run_backtest aplica
    # shift(1) internamente, positions[t-1] = sign(asset_returns[t]) queda
    # efectivo justo en el día t -> previsión perfecta e ilegítima, útil acá
    # solo para verificar que el motor premia el acierto direccional.
    perfect_positions = np.sign(asset_returns.shift(-1)).fillna(0.0)

    result = run_backtest(asset_returns, perfect_positions, cost_bps=0.0)

    # Con previsión perfecta, posicion_efectiva_t * asset_returns_t =
    # sign(asset_returns_t) * asset_returns_t = |asset_returns_t| >= 0
    # SIEMPRE: la equity curve nunca puede caer.
    assert result.metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["sharpe"] > 5.0


# --------------------------------------------------------------------------
# Costos de transacción restados vía turnover
# --------------------------------------------------------------------------


def test_cost_bps_is_subtracted_via_turnover_exactly() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    asset_returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    positions = pd.Series([1.0, 1.0, -1.0, -1.0], index=idx)
    cost_bps = 100.0  # 1% = 0.01 decimal por unidad de turnover

    result = run_backtest(asset_returns, positions, cost_bps=cost_bps)

    # posicion_efectiva (shift(1), primer día en 0): [0, 1, 1, -1]
    # turnover (|diff|):                             [0, 1, 0, 2]
    # retorno bruto (pos_efectiva * asset_returns):  [0, 0.02, -0.01, -0.03]
    # costo (0.01 * turnover):                       [0, 0.01, 0, 0.02]
    expected = pd.Series([0.0, 0.02 - 0.01, -0.01 - 0.0, -0.03 - 0.02], index=idx, name="strategy_return")
    pd.testing.assert_series_equal(result.returns, expected, check_names=False)
    assert result.metrics["turnover_total"] == pytest.approx(3.0)
    assert result.metrics["n_trades"] == 2  # 0->1 (día 1) y 1->-1 (día 3)


def test_frequent_flipping_underperforms_with_positive_cost_bps() -> None:
    asset_returns = _synthetic_returns()
    flipping_positions = pd.Series(
        np.where(np.arange(len(asset_returns)) % 2 == 0, 1.0, -1.0), index=asset_returns.index
    )

    result_no_cost = run_backtest(asset_returns, flipping_positions, cost_bps=0.0)
    result_with_cost = run_backtest(asset_returns, flipping_positions, cost_bps=50.0)

    assert result_with_cost.metrics["turnover_total"] == pytest.approx(result_no_cost.metrics["turnover_total"])
    assert result_with_cost.metrics["turnover_total"] > 0
    assert result_with_cost.metrics["total_return"] < result_no_cost.metrics["total_return"]
    assert result_with_cost.metrics["cagr"] < result_no_cost.metrics["cagr"]


# --------------------------------------------------------------------------
# El desfase (shift) evita el lookahead
# --------------------------------------------------------------------------


def test_shift_prevents_lookahead_bias() -> None:
    """Si se pudiera usar el retorno del MISMO día para decidir la posición
    de ESE día (lookahead), el resultado sería irrealmente bueno. Por eso
    `run_backtest` aplica `positions.shift(1)` internamente: la posición que
    llega en `positions[t]` recién se aplica al retorno de t+1, nunca al de
    t. Acá se contrasta el resultado REAL del motor (con el shift) contra un
    cálculo manual SIN el shift (lookahead), para mostrar la diferencia.
    """
    asset_returns = _synthetic_returns()
    same_day_sign_positions = np.sign(asset_returns)

    # Motor real: el signo de HOY recién se aplica al retorno de MAÑANA.
    engine_result = run_backtest(asset_returns, same_day_sign_positions, cost_bps=0.0)

    # Lookahead deliberado, a mano, SIN el shift: el signo de HOY sobre el
    # retorno de HOY mismo (equivalente a la "previsión perfecta" de arriba).
    lookahead_returns = np.sign(asset_returns) * asset_returns
    lookahead_sharpe = sharpe_ratio(lookahead_returns)

    assert lookahead_sharpe > engine_result.metrics["sharpe"]


# --------------------------------------------------------------------------
# Validaciones
# --------------------------------------------------------------------------


def test_run_backtest_rejects_positions_outside_bounds() -> None:
    asset_returns = _synthetic_returns(n=50)
    bad_positions = pd.Series(1.5, index=asset_returns.index)
    with pytest.raises(ValueError):
        run_backtest(asset_returns, bad_positions)


def test_run_backtest_raises_when_no_overlapping_dates() -> None:
    asset_returns = _synthetic_returns(n=50)
    other_idx = pd.date_range("2099-01-01", periods=50, freq="D")
    positions = pd.Series(1.0, index=other_idx)
    with pytest.raises(ValueError):
        run_backtest(asset_returns, positions)


def test_metrics_exposicion_media_and_hit_rate_are_sane() -> None:
    asset_returns = _synthetic_returns()
    positions = pd.Series(1.0, index=asset_returns.index)

    result = run_backtest(asset_returns, positions, cost_bps=0.0)

    # positions=1 constante: la exposición efectiva promedio debería ser
    # prácticamente 1 (salvo el primer día, en 0 por el warmup del shift).
    assert result.metrics["exposicion_media"] == pytest.approx(1.0, abs=0.01)
    assert 0.0 <= result.metrics["hit_rate"] <= 1.0
