"""Backtester vectorizado con costos de transacción.

REGLA ANTI-LOOKAHEAD (léase antes de usar este módulo): la posición que
llega a `run_backtest` en `positions[t]` se interpreta como la decisión
tomada con información disponible hasta el CIERRE del día t, y por lo tanto
recién puede aplicarse al retorno del día t+1 — nunca al retorno del día t
mismo (eso sería mirar el futuro). Internamente, `run_backtest` calcula
`posicion_efectiva = positions.shift(1)` y opera SIEMPRE sobre esa serie
desplazada, nunca sobre `positions` cruda. Cualquier señal que asuma que su
posición del día t se aplica al retorno del día t invalida el backtest (da
métricas irrealmente buenas — ver `tests/test_backtest.py` para una
demostración concreta del efecto).

Este módulo NO reimplementa métricas de riesgo/performance: reutiliza
`metrics.risk_measures` (Sharpe, Sortino, Calmar, max drawdown, equity
curve, retorno/vol anualizados) tal cual, y solo agrega las métricas propias
del backtest (turnover, n° de trades, exposición, hit rate).

Convención del proyecto: `asset_returns`/`positions.returns` en escala
DECIMAL, frecuencia diaria, `PERIODS_PER_YEAR=252` por defecto (ver
config.py). `cost_bps` sigue `config.TRANSACTION_COST_BPS` (en basis points,
1 bp = 0.01% = 0.0001 decimal) aplicado sobre el turnover en la convención
`config.TURNOVER_CONVENTION` ("one_way": el turnover de un rebalanceo cuenta
un solo lado, no ida+vuelta).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import PERIODS_PER_YEAR, TRANSACTION_COST_BPS
from metrics.risk_measures import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

logger = logging.getLogger(__name__)

# Tolerancia numérica para validar que `positions` está en [-1, 1] (evita
# rechazar por error de punto flotante un valor como 1.0000000000000002).
_POSITION_BOUND_TOLERANCE = 1e-9


@dataclass
class BacktestResult:
    """Resultado de `run_backtest`.

    Atributos
    ---------
    equity_curve:
        Curva de capital de la estrategia (base 1.0), vía
        `metrics.risk_measures.equity_curve` sobre `returns`.
    returns:
        Retornos DECIMALES diarios de la estrategia (ya netos de costos,
        con la posición efectiva desplazada un día — ver regla
        anti-lookahead del docstring del módulo).
    metrics:
        Dict de métricas: "total_return", "cagr", "ann_vol", "sharpe",
        "sortino", "max_drawdown", "calmar" (reutilizadas de
        `metrics.risk_measures`), más "turnover_total",
        "turnover_medio_diario", "n_trades", "exposicion_media", "hit_rate"
        (propias del backtest).
    """

    equity_curve: pd.Series
    returns: pd.Series
    metrics: dict[str, float]


def run_backtest(
    asset_returns: pd.Series,
    positions: pd.Series,
    cost_bps: float | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> BacktestResult:
    """Corre un backtest vectorizado de una estrategia direccional.

    Parámetros
    ----------
    asset_returns:
        Retornos DECIMALES del activo, indexados por fecha.
    positions:
        Posición objetivo en [-1, 1] (o {-1, 0, 1} para long/flat/short),
        indexada por fecha, alineada (por índice) con `asset_returns`.
        Se interpreta como la decisión tomada con información hasta el
        cierre de cada fecha (ver regla anti-lookahead del docstring del
        módulo) — `run_backtest` es responsable de aplicar el desfase, NO
        hay que pre-desplazar `positions` antes de llamar a esta función.
    cost_bps:
        Costo de transacción en basis points por unidad de turnover
        one-way. Si es None, se usa `config.TRANSACTION_COST_BPS`.
    periods_per_year:
        Para anualizar métricas (252 por defecto).

    Lógica (por día t, ya con `posicion_efectiva = positions.shift(1)`):
        turnover_t = |posicion_efectiva_t - posicion_efectiva_{t-1}|
        retorno_estrategia_t = posicion_efectiva_t * asset_returns_t
                                - (cost_bps / 1e4) * turnover_t

    El primer día de la serie alineada siempre tiene posición efectiva 0
    (no hay decisión previa antes del inicio de los datos) y por lo tanto
    turnover 0 — no se asume ninguna posición "heredada" de fuera de la
    muestra.

    Devuelve un `BacktestResult`. Lanza `ValueError` si `positions` tiene
    valores fuera de [-1, 1] o si no hay fechas en común con `asset_returns`.
    """
    if cost_bps is None:
        cost_bps = TRANSACTION_COST_BPS

    aligned = pd.concat(
        [asset_returns.rename("asset_return"), positions.rename("position")], axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError("run_backtest: no hay fechas en común entre 'asset_returns' y 'positions'")

    if (aligned["position"].abs() > 1.0 + _POSITION_BOUND_TOLERANCE).any():
        raise ValueError("run_backtest: 'positions' debe estar en [-1, 1]")

    asset_returns_aligned = aligned["asset_return"]
    positions_aligned = aligned["position"]

    # REGLA ANTI-LOOKAHEAD: la posición decidida en t recién es efectiva en t+1.
    position_effective = positions_aligned.shift(1).fillna(0.0)

    turnover = position_effective.diff().fillna(0.0).abs()

    strategy_returns = position_effective * asset_returns_aligned - (cost_bps / 1e4) * turnover
    strategy_returns.name = "strategy_return"

    equity = equity_curve(strategy_returns)

    sign = np.sign(position_effective)
    n_trades = int((sign.diff().fillna(0.0) != 0.0).sum())

    metrics: dict[str, float] = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": annualized_return(strategy_returns, periods_per_year=periods_per_year),
        "ann_vol": annualized_volatility(strategy_returns, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(strategy_returns, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(strategy_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(strategy_returns),
        "calmar": calmar_ratio(strategy_returns, periods_per_year=periods_per_year),
        "turnover_total": float(turnover.sum()),
        "turnover_medio_diario": float(turnover.mean()),
        "n_trades": n_trades,
        "exposicion_media": float(position_effective.abs().mean()),
        "hit_rate": float((strategy_returns > 0.0).mean()),
    }

    logger.info(
        "run_backtest: %d días, cost_bps=%.1f, sharpe=%.2f, cagr=%.2f%%, max_drawdown=%.2f%%, n_trades=%d",
        len(strategy_returns), cost_bps, metrics["sharpe"], metrics["cagr"] * 100,
        metrics["max_drawdown"] * 100, n_trades,
    )

    return BacktestResult(equity_curve=equity, returns=strategy_returns, metrics=metrics)


def compare_to_buy_and_hold(
    asset_returns: pd.Series,
    result: BacktestResult,
    cost_bps: float | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """Compara las métricas de `result` (un backtest ya corrido) contra un
    benchmark buy & hold (posición constante = 1) sobre el mismo
    `asset_returns`, corrido con la misma convención de costos (mismo
    `cost_bps`, para que la comparación sea justa: ambos pagan el costo de
    entrar a la posición una vez).

    Devuelve un dict {"estrategia": result.metrics, "buy_and_hold": <dict de
    métricas del buy & hold>}, listo para comparar lado a lado.
    """
    buy_and_hold_positions = pd.Series(1.0, index=asset_returns.index)
    buy_and_hold_result = run_backtest(
        asset_returns, buy_and_hold_positions, cost_bps=cost_bps, periods_per_year=periods_per_year
    )
    return {"estrategia": result.metrics, "buy_and_hold": buy_and_hold_result.metrics}
