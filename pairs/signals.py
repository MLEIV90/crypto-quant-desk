"""Señales de trading sobre el spread de un par (z-score + entrada/salida/stop).

NO implementa el backtest de P&L de las dos patas del par — eso es Fase 2c.
Acá solo se decide la POSICIÓN SOBRE EL SPREAD (+1 largo spread, -1 corto
spread, 0 flat) día a día, a partir de un `spread` ya calculado (por
`pairs.cointegration.engle_granger` o `pairs.kalman_hedge.kalman_hedge_ratio`).

REGLA ANTI-LOOKAHEAD: igual que en `signals.engine`, la posición del día t
usa solo información hasta el cierre de t (el z-score expansivo/rolling y
las reglas de entrada/salida son todas trailing) y este módulo NO aplica
ningún `shift` — es responsabilidad del backtester de Fase 2c desplazarla un
día antes de aplicarla a un retorno.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ENTRY: float = 2.0
DEFAULT_EXIT: float = 0.5
DEFAULT_STOP: float = 4.0


def zscore(spread: pd.Series, window: int | None = None) -> pd.Series:
    """Z-score del `spread`: z_t = (spread_t - media_t) / desvío_t.

    Si `window` es None (default): media y desvío EXPANSIVOS (`expanding()`,
    con toda la historia hasta cada fecha, incluyéndola — igual que el resto
    de los indicadores del proyecto, ver `signals.indicators`). Es la opción
    más honesta para trading en vivo: no asume una ventana arbitraria ni que
    el régimen de volatilidad del spread sea estacionario.

    Si se da `window`: media y desvío ROLLING de esa ventana (trailing,
    tampoco mira el futuro) — más reactivo a cambios de régimen recientes en
    el spread, a costa de más varianza de estimación (ventanas cortas) o de
    quedar desactualizado (ventanas largas).

    Los primeros valores (antes de tener al menos 2 observaciones, o dentro
    del warmup de `window`) quedan en NaN por `std` indefinido; si el
    desvío da exactamente 0 (spread constante), el z-score queda en NaN en
    vez de +-inf.
    """
    if window is None:
        mean = spread.expanding().mean()
        std = spread.expanding().std(ddof=1)
    else:
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std(ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (spread - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def generate_pair_signals(
    spread: pd.Series, entry: float = DEFAULT_ENTRY, exit: float = DEFAULT_EXIT, stop: float = DEFAULT_STOP
) -> pd.DataFrame:
    """Reglas de posición sobre el spread, día a día, SIN lookahead: la
    posición del día t depende solo del z-score hasta t y de la posición
    del día anterior — una máquina de estados secuencial (no vectorizable
    sin loop, porque "hay posición abierta" es un estado que depende de la
    historia de decisiones, no solo del dato de hoy).

    Reglas (evaluadas en este orden, día a día):
    - Si NO hay posición abierta:
        * z > entry  -> abre CORTO-spread (posicion_spread = -1): el spread
          está anormalmente alto, se apuesta a que revierte hacia abajo.
        * z < -entry -> abre LARGO-spread (posicion_spread = +1): el spread
          está anormalmente bajo, se apuesta a que revierte hacia arriba.
    - Si HAY posición abierta:
        * |z| > stop  -> STOP-LOSS, cierra (posicion_spread = 0): la
          reversión no llegó y el spread se siguió alejando; se corta la
          pérdida en vez de esperar indefinidamente.
        * |z| < exit  -> cierre normal (posicion_spread = 0): el spread
          volvió cerca de su media, se toma la ganancia.
        * si no, se mantiene la posición sin cambios.
    Días con z-score NaN (warmup) se tratan como "sin señal": se mantiene la
    posición previa (0 si todavía no se abrió ninguna).

    `entry`/`exit`/`stop` deben cumplir 0 <= exit < entry < stop (si no, no
    hay una zona de entrada ni de stop-loss coherente).

    Devuelve un `pd.DataFrame` indexado igual que `spread`, con columnas:
    "z" (`zscore(spread)`, expansivo), "posicion_spread" (+1/-1/0),
    "evento" (string: "entrada_corta", "entrada_larga", "cierre",
    "stop_loss", o "" si no pasó nada ese día).
    """
    if not (0.0 <= exit < entry < stop):
        raise ValueError(
            f"generate_pair_signals: se espera 0 <= exit < entry < stop, recibido "
            f"exit={exit}, entry={entry}, stop={stop}"
        )

    z = zscore(spread)

    position = np.zeros(len(z))
    event = [""] * len(z)
    current_position = 0

    for i, z_t in enumerate(z.to_numpy()):
        if not np.isnan(z_t):
            if current_position == 0:
                if z_t > entry:
                    current_position = -1
                    event[i] = "entrada_corta"
                elif z_t < -entry:
                    current_position = 1
                    event[i] = "entrada_larga"
            else:
                if abs(z_t) > stop:
                    current_position = 0
                    event[i] = "stop_loss"
                elif abs(z_t) < exit:
                    current_position = 0
                    event[i] = "cierre"

        position[i] = current_position

    return pd.DataFrame({"z": z, "posicion_spread": position, "evento": event}, index=spread.index)
