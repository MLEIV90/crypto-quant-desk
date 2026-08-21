"""Backtest de la estrategia de reversión a la media sobre UN par (Fase
15b — esto es lo que se había archivado en la Fase 2c del proyecto).

Reutiliza `pairs.signals.generate_pair_signals` (la máquina de estados
entrada/salida/stop sobre el z-score del spread — NO se reimplementa acá)
y `backtest.engine.run_backtest` (el backtester vectorizado con costos que
ya usa el resto del proyecto) — este módulo solo arma el "activo sintético"
que hace falta para poder correr un par de dos patas a través de un
backtester pensado para UN activo.

SUPUESTOS — léase antes de interpretar el resultado:

1. DOLLAR-NEUTRAL, REBALANCEADO A DIARIO: "estar largo el spread" se
   interpreta como estar largo $1 de `y` y corto `hedge_ratio` dólares de
   `x`, con el tamaño de las dos patas reajustado TODOS los días para
   mantener esa proporción (no se deja "correr" el desbalance que
   introduce cada pata moviéndose a su propio precio). Es la simplificación
   estándar de un backtest de pairs trading sin ejecución real — en la
   práctica, rebalancear todos los días tiene un costo de transacción
   adicional que este backtest NO carga aparte (ver el punto 3).

2. RETORNOS SIMPLES, NO LOG-RETORNOS: el retorno diario del spread se mide
   como `simple_returns(y) - hedge_ratio * simple_returns(x)` — NO como la
   diferencia de LOG-retornos, aunque `hedge_ratio` salga de una regresión
   sobre log-precios (`pairs.cointegration.engle_granger`). Es una
   aproximación: log-retorno ≈ retorno simple para variaciones diarias
   moderadas (la discrepancia crece en días de movimientos extremos), pero
   es la que hace falta para reutilizar `backtest.engine.run_backtest` tal
   cual — ese backtester exige retornos SIMPLES porque compone la equity
   curve vía `(1 + r)`, algo que solo es matemáticamente correcto para
   retornos simples (ver su docstring).

3. UN SOLO `cost_bps` PARA LAS DOS PATAS: el costo de transacción se aplica
   sobre el turnover DEL SPREAD combinado (`|Δposición_spread|`), no sobre
   cada pata por separado. En la práctica, dos patas separadas suelen pagar
   MÁS costo total (dos comisiones, dos spreads bid-ask) que una sola pata
   — este backtest probablemente SUBESTIMA el costo real, un sesgo a favor
   de la estrategia que hay que tener presente al leer el resultado.

4. `DEFAULT_PAIR_EXIT = 0.5`, no 0.0: con `exit=0.0` exacto, la condición
   de cierre normal de `generate_pair_signals` (`abs(z) < exit`) nunca se
   cumple (ningún número real tiene valor absoluto negativo) — la posición
   solo podría cerrarse por stop-loss. 0.5 hace que "cerrar cuando el
   spread volvió cerca de su media" sea un mecanismo que de verdad dispara.

HONESTIDAD (mismo criterio que `pairs/stability.py`): este backtest existe
para PONER A PRUEBA cuantitativamente si un par es operable, no para
recomendar operarlo. Sobre un par que la Fase 2/13b ya mostró que NO está
establemente cointegrado, lo ESPERABLE es que este backtest muestre
pérdidas o resultados inestables — eso CONFIRMA la falta de edge, no es un
error del backtest.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from config import TRANSACTION_COST_BPS
from pairs.signals import generate_pair_signals
from signals.returns import simple_returns

logger = logging.getLogger(__name__)

# NOTA sobre `exit` (Fase 15b): con `exit=0.0` EXACTO, la condición de
# cierre normal de `generate_pair_signals` (`abs(z) < exit`) nunca se
# cumple — ningún número real tiene valor absoluto negativo, así que la
# posición solo podría cerrarse por stop-loss, nunca por reversión
# efectiva. Se usa 0.5 acá (en vez de 0.0) para que "cerrar cuando el
# spread volvió cerca de su media" sea un mecanismo que de verdad dispara.
DEFAULT_PAIR_ENTRY: float = 2.0
DEFAULT_PAIR_EXIT: float = 0.5
DEFAULT_PAIR_STOP: float = 3.0


def backtest_pair(
    y_prices: pd.Series,
    x_prices: pd.Series,
    hedge_ratio: float,
    entry: float = DEFAULT_PAIR_ENTRY,
    exit: float = DEFAULT_PAIR_EXIT,
    stop: float = DEFAULT_PAIR_STOP,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> dict:
    """Backtestea la estrategia de reversión de `pairs.signals.generate_pair_signals`
    sobre el par (`y_prices`, `x_prices`) con hedge ratio `hedge_ratio` (el
    de `pairs.cointegration.engle_granger`/`hedge_ratio_ols`, NO se
    re-estima acá — se recibe como parámetro).

    Pasos (ningún cálculo de bajo nivel nuevo, ver docstring del módulo):
    1. Arma el spread NIVEL: `log(y) - hedge_ratio * log(x)`, alineado por
       fecha (inner join) entre `y_prices`/`x_prices`.
    2. Corre `generate_pair_signals(spread, entry, exit, stop)` — reutilizado
       tal cual: da el z-score (`pairs.signals.zscore`, expansivo) y la
       posición día a día (+1/-1/0), sin lookahead.
    3. Arma el retorno DIARIO del spread (`simple_returns(y) - hedge_ratio *
       simple_returns(x)` — ver SUPUESTO 2 del docstring del módulo) y lo
       corre por `backtest.engine.run_backtest` con esa posición.

    Lanza `ValueError` si `y_prices`/`x_prices` no tienen al menos 2 fechas
    en común.

    Devuelve un dict:
    - "equity_curve": `pd.Series` (base 1.0), de `run_backtest`.
    - "returns": `pd.Series`, retornos diarios de la estrategia (netos de costos).
    - "metrics": dict de `run_backtest` — "total_return", "cagr", "ann_vol",
      "sharpe", "sortino", "max_drawdown", "calmar", "turnover_total",
      "turnover_medio_diario", "n_trades", "exposicion_media", "hit_rate".
    - "spread": `pd.Series`, el spread NIVEL (para graficarlo).
    - "zscore": `pd.Series`, el z-score expansivo del spread (mismo índice que "spread").
    - "eventos": `pd.Series` de strings ("entrada_larga"/"entrada_corta"/"cierre"/"stop_loss"/"").
    """
    aligned = pd.concat([y_prices.rename("y"), x_prices.rename("x")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError(
            "backtest_pair: se necesitan al menos 2 observaciones alineadas entre 'y_prices' y 'x_prices'"
        )

    log_y = np.log(aligned["y"])
    log_x = np.log(aligned["x"])
    spread = (log_y - hedge_ratio * log_x).rename("spread")

    signals_df = generate_pair_signals(spread, entry=entry, exit=exit, stop=stop)

    spread_returns = simple_returns(aligned["y"]) - hedge_ratio * simple_returns(aligned["x"])
    spread_returns = spread_returns.dropna()
    spread_returns.name = "spread_return"

    result = run_backtest(spread_returns, signals_df["posicion_spread"], cost_bps=cost_bps)

    logger.info(
        "backtest_pair: hedge_ratio=%.4f, entry=%.1f/exit=%.1f/stop=%.1f, sharpe=%.2f, "
        "total_return=%.2f%%, n_trades=%d",
        hedge_ratio, entry, exit, stop, result.metrics["sharpe"], result.metrics["total_return"] * 100,
        result.metrics["n_trades"],
    )

    return {
        "equity_curve": result.equity_curve,
        "returns": result.returns,
        "metrics": result.metrics,
        "spread": spread,
        "zscore": signals_df["z"],
        "eventos": signals_df["evento"],
    }
