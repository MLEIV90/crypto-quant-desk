"""Tests offline para backtest/engine.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import backtest_from_prices, compare_to_buy_and_hold, run_backtest
from metrics.risk_measures import equity_curve, sharpe_ratio
from signals.returns import simple_returns


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


def test_run_backtest_accepts_scalar_positions() -> None:
    # positions=1 (escalar, no pd.Series) debe comportarse igual que pasar
    # una serie constante de 1.0 en todas las fechas de asset_returns.
    asset_returns = _synthetic_returns()

    scalar_result = run_backtest(asset_returns, positions=1, cost_bps=0.0)
    series_result = run_backtest(asset_returns, pd.Series(1.0, index=asset_returns.index), cost_bps=0.0)

    pd.testing.assert_series_equal(scalar_result.returns, series_result.returns)
    assert scalar_result.metrics == series_result.metrics


# --------------------------------------------------------------------------
# backtest_from_prices: contrato de retornos SIMPLES
# --------------------------------------------------------------------------


def test_backtest_from_prices_matches_run_backtest_with_simple_returns() -> None:
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(2)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 200))), index=idx)

    result_from_prices = backtest_from_prices(prices, positions=1, cost_bps=10.0)

    manual_returns = simple_returns(prices)
    result_manual = run_backtest(manual_returns, positions=1, cost_bps=10.0)

    pd.testing.assert_series_equal(result_from_prices.returns, result_manual.returns)
    assert result_from_prices.metrics == result_manual.metrics


def test_run_backtest_warns_when_returns_look_logarithmic(caplog: pytest.LogCaptureFixture) -> None:
    # Un log-retorno de una caída fuerte (< -100%) hace que (1 + r) sea
    # negativo, imposible para un retorno SIMPLE válido (el precio no puede
    # caer más del 100%). run_backtest debe loguear un warning claro, NO abortar.
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    fake_log_returns = pd.Series([0.01, -1.5, 0.02, -0.01, 0.03], index=idx)
    positions = pd.Series(1.0, index=idx)

    with caplog.at_level("WARNING", logger="backtest.engine"):
        result = run_backtest(fake_log_returns, positions, cost_bps=0.0)

    assert any("LOGARÍTMICOS" in record.message for record in caplog.records)
    assert len(result.returns) == 5  # no abortó


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


# --------------------------------------------------------------------------
# Fase 23: `positions` expuesta en BacktestResult + "pct_tiempo_fuera"
# --------------------------------------------------------------------------


def test_backtest_result_exposes_the_effective_position_series() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    asset_returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    positions = pd.Series([1.0, 1.0, -1.0, -1.0], index=idx)

    result = run_backtest(asset_returns, positions, cost_bps=0.0)

    # Igual que en test_cost_bps_is_subtracted_via_turnover_exactly: la
    # posición EFECTIVA es positions.shift(1) con el primer día en 0.
    expected = pd.Series([0.0, 1.0, 1.0, -1.0], index=idx)
    pd.testing.assert_series_equal(result.positions, expected, check_names=False)


def test_pct_tiempo_fuera_is_zero_for_always_invested_strategy() -> None:
    asset_returns = _synthetic_returns()
    positions = pd.Series(1.0, index=asset_returns.index)

    result = run_backtest(asset_returns, positions, cost_bps=0.0)

    # Solo el primer día (warmup del shift, sin decisión previa) cuenta como
    # "afuera" — sobre una serie larga, la fracción es prácticamente 0.
    assert result.metrics["pct_tiempo_fuera"] == pytest.approx(1.0 / len(asset_returns), abs=1e-9)


def test_pct_tiempo_fuera_is_higher_when_strategy_spends_time_flat() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    asset_returns = pd.Series(0.01, index=idx)
    # Afuera del mercado (posición 0) la mitad de los días.
    positions = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=idx)

    always_invested = run_backtest(asset_returns, pd.Series(1.0, index=idx), cost_bps=0.0)
    half_flat = run_backtest(asset_returns, positions, cost_bps=0.0)

    assert half_flat.metrics["pct_tiempo_fuera"] > always_invested.metrics["pct_tiempo_fuera"]
