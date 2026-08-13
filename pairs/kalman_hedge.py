"""Hedge ratio DINÁMICO vía filtro de Kalman.

`pairs.cointegration.hedge_ratio_ols` (Fase 2a) estima un hedge ratio
ESTÁTICO: un único beta para toda la muestra, vía OLS. Es simple y estable,
pero si la relación entre los dos activos cambia con el tiempo (lo típico en
cripto: la relación BTC-ETH de 2019 no es la de 2024), un beta fijo queda
desactualizado y el spread que se computa con él deja de ser el spread
"correcto" — empieza a mezclar la reversión real con el drift del hedge
ratio que cambió.

Este módulo trata beta (y alpha) como un RANDOM WALK y los estima con un
filtro de Kalman: en cada fecha, primero PREDICE beta/alpha a partir de la
estimación del día anterior (sin usar el precio de hoy) y calcula el spread
del día como el error de esa predicción (la "innovación"); recién después
ACTUALIZA la estimación de beta/alpha incorporando el precio de hoy. Por
construcción, el spread reportado nunca usa el propio dato del día para
"corregirse a sí mismo" antes de reportarse.

Trabaja sobre log-precios, igual que `pairs.cointegration` (recibe precios
RAW y aplica `np.log()` internamente).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_DELTA: float = 1e-4
DEFAULT_R: float = 1e-3


def kalman_hedge_ratio(
    y: pd.Series, x: pd.Series, delta: float = DEFAULT_DELTA, r: float = DEFAULT_R
) -> pd.DataFrame:
    """Filtro de Kalman para el modelo de espacio de estados:

        log(y_t) = beta_t * log(x_t) + alpha_t + v_t,     v_t ~ N(0, r)
        [beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + w_t,   w_t ~ N(0, Q)

    con `Q = delta/(1-delta) * I` (parametrización estándar de Kalman para
    pairs trading, ver Chan, "Algorithmic Trading", 2013).

    Parámetros
    ----------
    delta:
        Controla la varianza del proceso (cuánto puede moverse `beta`/`alpha`
        de un día a otro). Mayor `delta` => más varianza de proceso => el
        hedge ratio se ADAPTA más rápido a cambios reales, pero con más
        ruido día a día. Menor `delta` => hedge ratio más SUAVE y estable,
        pero más lento para reflejar un cambio de régimen genuino.
    r:
        Varianza de observación (cuánto ruido se asume en la relación
        y~x en sí). Mayor `r` => el filtro confía menos en cada observación
        individual y reacciona más despacio a ella (más suavizado también,
        pero del lado de la observación, no del proceso). Menor `r` => cada
        observación pesa más en la actualización, más reactivo pero más
        sensible a ruido puntual.
    Juntos, `delta` y `r` son el trade-off central del filtro:
    suavidad/estabilidad vs. adaptabilidad/reactividad. Comparado con
    `pairs.cointegration.hedge_ratio_ols` (un solo beta fijo, el extremo
    "toda suavidad, cero adaptabilidad"), este filtro permite elegir un
    punto intermedio.

    Nota sobre convergencia tras un cambio de régimen: como el modelo usa
    `log(x_t)` directamente (no centrado) como regresor junto a una
    constante, cuando `log(x)` se mantiene en un nivel alto y poco variable
    (el caso típico de precios reales: log(30000) ~ 10.3 para BTC, no
    log-retornos cerca de 0), `beta` y `alpha` quedan correlacionados en la
    covarianza del filtro — son parcialmente intercambiables para explicar
    el mismo ajuste. Esto no es un error: es una propiedad conocida de
    regresionar sobre el NIVEL en vez de sobre una versión centrada, y
    también afecta a la regresión OLS estática (`hedge_ratio_ols`) si se
    reajusta por sub-períodos. En la práctica, significa que tras un cambio
    real y grande del hedge ratio, `beta` se mueve rápido en la dirección
    correcta pero puede tardar bastante en asentarse muy cerca del nuevo
    valor exacto — no hay que esperar una convergencia instantánea ni
    exacta, solo una tendencia clara hacia el nuevo nivel (ver el test
    correspondiente en `tests/test_kalman_hedge.py`).

    Devuelve un `pd.DataFrame` indexado igual que la muestra alineada de
    `y`/`x`, con columnas:
    - "beta", "alpha": la estimación FILTRADA (posterior, ya incorporando el
      precio del propio día t) del hedge ratio y el intercepto — análogo a
      cómo el resto de los indicadores del proyecto (RSI, SMA, etc.) usan el
      cierre del propio día t para el valor de ese día.
    - "spread": la INNOVACIÓN del día t (log(y_t) menos la predicción hecha
      con la estimación de beta/alpha del día ANTERIOR, antes de actualizar
      con el dato de hoy) — es el spread que efectivamente se puede observar
      "en vivo" en el momento de decidir, y el que debe alimentar
      `pairs.signals.generate_pair_signals` en vez de un residuo calculado
      con el beta ya actualizado (que subestimaría el spread real).
    """
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    log_y = np.log(aligned["y"]).to_numpy()
    log_x = np.log(aligned["x"]).to_numpy()
    n = len(aligned)

    process_var = delta / (1.0 - delta)
    process_cov = process_var * np.eye(2)

    # Estado = [beta, alpha]. Covarianza inicial GRANDE (prior difuso): sin
    # esto, el filtro arranca "muy seguro" de beta=alpha=0 y tarda muchas
    # observaciones en corregirse.
    state = np.zeros(2)
    state_cov = np.eye(2) * 1e6

    betas = np.empty(n)
    alphas = np.empty(n)
    spread = np.empty(n)

    for t in range(n):
        h = np.array([log_x[t], 1.0])

        # Predicción a partir del estado del día ANTERIOR (sin usar log_y[t]).
        state_pred = state
        cov_pred = state_cov + process_cov

        # Innovación: el spread "en vivo" del día t.
        y_pred = h @ state_pred
        innovation = log_y[t] - y_pred
        innovation_var = h @ cov_pred @ h.T + r

        # Actualización con el dato de hoy (esto sí puede usar log_y[t]:
        # queda reflejado en beta/alpha de HOY, no en el spread ya reportado).
        kalman_gain = cov_pred @ h / innovation_var
        state = state_pred + kalman_gain * innovation
        state_cov = cov_pred - np.outer(kalman_gain, h) @ cov_pred

        betas[t] = state[0]
        alphas[t] = state[1]
        spread[t] = innovation

    return pd.DataFrame({"beta": betas, "alpha": alphas, "spread": spread}, index=aligned.index)
