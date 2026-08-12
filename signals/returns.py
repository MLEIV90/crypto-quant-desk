"""Cálculo de retornos a partir de series de precios de cierre.

Convención del proyecto: los retornos se expresan siempre en escala DECIMAL
(0.01 == 1%), nunca en porcentaje (ver config.py). La frecuencia de trabajo
por defecto es diaria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(close: pd.Series) -> pd.Series:
    """Retorno simple período a período: r_t = (P_t / P_{t-1}) - 1.

    Es la convención correcta para agregar en el tiempo por composición
    (equity curves, retorno acumulado), a diferencia del retorno logarítmico.
    El primer valor de la serie resultante queda en NaN (no existe P_{t-1}
    para el primer punto) y se conserva así deliberadamente, en vez de
    rellenarlo con 0, para no introducir un retorno falso al inicio de la serie.
    """
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    """Retorno logarítmico período a período: r_t = ln(P_t / P_{t-1}).

    Para variaciones pequeñas es aproximadamente igual al retorno simple, pero
    es aditivo en el tiempo (sumar log-retornos de varios períodos da el
    log-retorno del período agregado), lo que lo hace conveniente para
    ventanas móviles y para modelado estadístico (p. ej. GARCH en fases
    posteriores). Al igual que `simple_returns`, el primer valor queda en NaN.
    """
    return np.log(close / close.shift(1))
