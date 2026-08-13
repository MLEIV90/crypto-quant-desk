"""Matriz de features para el pipeline de ML de crypto-quant-desk.

Fase 3a: solo construcción de features y alineación con las etiquetas de
`ml.labeling`. El entrenamiento de un modelo es Fase 3c; la validación
walk-forward es Fase 3b.

REGLA ANTI-LOOKAHEAD: todas las features son TRAILING (calculadas con
información hasta el cierre de cada fecha t, nunca con datos posteriores;
ver el detalle de cada una en `build_feature_matrix`, incluida la única
excepción documentada: `vol_garch`). Esto es tan importante acá como en
`signals/engine.py` — una feature con lookahead invalida cualquier modelo
entrenado sobre ella, aunque el LABEL (`ml.labeling.triple_barrier_labels`,
que sí mira el futuro por diseño porque es el target, no una feature) esté
perfectamente bien.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import PERIODS_PER_YEAR
from ml.labeling import get_sample_weights
from models.garch import conditional_volatility, select_best_model
from signals.indicators import add_all_indicators
from signals.returns import log_returns, simple_returns

logger = logging.getLogger(__name__)

_RETURN_LAGS: tuple[int, ...] = (1, 5, 10, 20)
_REALIZED_VOL_WINDOW: int = 20


def build_feature_matrix(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de features X a partir de un OHLCV estandarizado
    (el de `data.loaders.get_prices`). No modifica `df_ohlcv` in-place.

    Reutiliza `signals.indicators.add_all_indicators` tal cual (agrega
    sma_20/50, ema_12/26, rsi_14, macd/macd_signal/macd_hist,
    bb_mid/upper/lower/pct_b/bandwidth/zscore, atr_14 — todos trailing por
    construcción, ver ese módulo) y agrega:

    - "ret_1", "ret_5", "ret_10", "ret_20": retorno simple de los últimos
      1/5/10/20 días (`close_t / close_{t-k} - 1`, vía `Series.pct_change`)
      — momentum a distintos horizontes. Trailing: usa solo precios <= t.
    - "vol_realizada_20": desvío estándar rolling de 20 días de los
      retornos simples (`signals.returns.simple_returns`), ANUALIZADO
      (`* sqrt(config.PERIODS_PER_YEAR)`) — volatilidad reciente realizada,
      trailing.
    - "vol_garch": volatilidad condicional GARCH (`models.garch`,
      `select_best_model` + `conditional_volatility`, sobre log-retornos)
      como feature de RÉGIMEN de volatilidad.

      CAVEAT IMPORTANTE (única excepción a la regla anti-lookahead de este
      módulo): el modelo GARCH se ajusta UNA SOLA VEZ sobre TODA la serie de
      `df_ohlcv` a la vez, por eficiencia (re-ajustar un GARCH día por día
      sería prohibitivamente lento para una matriz de features). Como el
      ajuste es conjunto, técnicamente la vol GARCH estimada para una fecha
      temprana está informada, a través de los parámetros del modelo, por
      datos de fechas POSTERIORES — un lookahead real, aunque sutil. Es un
      trade-off deliberado para Fase 3a. En Fase 3b/3c, cuando se arme la
      validación walk-forward, esta feature debe recalcularse re-ajustando
      el GARCH solo con datos hasta cada fecha de corte, no con este ajuste
      único in-sample. Si el GARCH no converge (`select_best_model` agota
      su grilla), se loguea un warning y la columna queda en NaN en vez de
      romper toda la matriz de features.

    Devuelve una copia de `df_ohlcv` con todas las columnas de
    `add_all_indicators` más las agregadas acá.
    """
    out = add_all_indicators(df_ohlcv)
    close = out["close"]

    for lag in _RETURN_LAGS:
        out[f"ret_{lag}"] = close.pct_change(periods=lag)

    realized_vol = simple_returns(close).rolling(window=_REALIZED_VOL_WINDOW).std(ddof=1)
    out["vol_realizada_20"] = realized_vol * np.sqrt(PERIODS_PER_YEAR)

    try:
        garch_returns = log_returns(close).dropna()
        best = select_best_model(garch_returns, criterion="aic")
        cond_vol = conditional_volatility(best["result"])
        out["vol_garch"] = cond_vol.reindex(out.index)
    except Exception as exc:  # noqa: BLE001 - un GARCH que no converge no debe tumbar toda la matriz
        logger.warning("build_feature_matrix: GARCH no convergió, 'vol_garch' queda en NaN: %s", exc)
        out["vol_garch"] = np.nan

    return out


def align_features_labels(X: pd.DataFrame, labels_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Alinea la matriz de features `X` (ver `build_feature_matrix`) con las
    etiquetas de `labels_df` (ver `ml.labeling.triple_barrier_labels`) y sus
    pesos por solapamiento (`ml.labeling.get_sample_weights`, calculados
    acá mismo a partir de `labels_df`), por índice común, y elimina las
    filas con NaN en cualquiera de los tres:
    - warmup de los indicadores/features al principio de la serie (p. ej.
      `sma_50` necesita 50 observaciones previas),
    - cola sin etiqueta completa al final (las últimas `max_holding` filas
      de `triple_barrier_labels`).

    Devuelve `(X_limpio, y, sample_weights)`, las tres alineadas 1 a 1 por
    índice, listas para entrenar un modelo (Fase 3c).
    """
    weights = get_sample_weights(labels_df)
    combined = pd.concat([X, labels_df["label"].rename("__label__"), weights.rename("__weight__")], axis=1)
    combined = combined.dropna()

    X_clean = combined[X.columns]
    y = combined["__label__"].rename("label")
    sample_weights = combined["__weight__"].rename("weight")
    return X_clean, y, sample_weights
