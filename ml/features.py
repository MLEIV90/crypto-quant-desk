"""Matriz de features para el pipeline de ML de crypto-quant-desk.

Fase 3a: solo construcción de features y alineación con las etiquetas de
`ml.labeling`. El entrenamiento de un modelo es Fase 3c; la validación
walk-forward es Fase 3b.

REGLA ANTI-LOOKAHEAD: todas las features son TRAILING (calculadas con
información hasta el cierre de cada fecha t, nunca con datos posteriores;
ver el detalle de cada una en `build_feature_matrix`) — SIN EXCEPCIONES.
Esto es tan importante acá como en `signals/engine.py` — una feature con
lookahead invalida cualquier modelo entrenado sobre ella, aunque el LABEL
(`ml.labeling.triple_barrier_labels`, que sí mira el futuro por diseño
porque es el target, no una feature) esté perfectamente bien.

HOTFIX (post Fase 3a): se removió la feature `vol_garch`. Ajustaba UN GARCH
sobre TODA la serie a la vez (in-sample) y se documentó en su momento como
"la única excepción" a la regla anti-lookahead — pero un data leakage
documentado sigue siendo data leakage: se confirmó truncando la serie en una
fecha t y viendo que la fila de `vol_garch` en t cambiaba según cuánto futuro
hubiera disponible al momento del ajuste (las otras 25 features no cambiaban).
GARCH in-sample sigue siendo la herramienta correcta como MOTOR DE RIESGO
para la foto actual (ver `models.garch` y `signals.engine.latest_recommendation`,
que ajustan un GARCH sobre toda la historia disponible HASTA HOY para estimar
el régimen de volatilidad vigente — eso no es leakage, es exactamente lo que
se quiere: la mejor estimación posible con toda la información disponible al
momento de decidir). El problema es específicamente usarlo como FEATURE de
entrenamiento sobre una serie histórica completa, donde "hoy" varía fila a
fila pero el ajuste no. Ver `garch_feature_walkforward` más abajo para la
alternativa causal (no implementada todavía).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import PERIODS_PER_YEAR
from ml.labeling import DEFAULT_VOLATILITY_SPAN, get_daily_volatility, get_sample_weights
from signals.indicators import add_all_indicators
from signals.returns import simple_returns

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
    - "vol_ewma": volatilidad diaria vía EWMA de retornos
      (`ml.labeling.get_daily_volatility`, span=`DEFAULT_VOLATILITY_SPAN`,
      reutilizada tal cual, NO reimplementada acá) — una segunda medida de
      volatilidad, con memoria más larga y suavizado exponencial en vez de
      ventana rolling fija, pero igual de CAUSAL/trailing que
      `vol_realizada_20`. Es el sustituto causal de la ex-feature
      `vol_garch` (ver HOTFIX en el docstring del módulo): no captura
      asimetría ni clustering tan bien como un GARCH, pero no tiene
      leakage.

    Devuelve una copia de `df_ohlcv` con todas las columnas de
    `add_all_indicators` más las agregadas acá.
    """
    out = add_all_indicators(df_ohlcv)
    close = out["close"]

    for lag in _RETURN_LAGS:
        out[f"ret_{lag}"] = close.pct_change(periods=lag)

    realized_vol = simple_returns(close).rolling(window=_REALIZED_VOL_WINDOW).std(ddof=1)
    out["vol_realizada_20"] = realized_vol * np.sqrt(PERIODS_PER_YEAR)

    out["vol_ewma"] = get_daily_volatility(close, span=DEFAULT_VOLATILITY_SPAN)

    return out


def garch_feature_walkforward(df_ohlcv: pd.DataFrame, *args, **kwargs) -> pd.Series:
    """STUB — NO IMPLEMENTADO. Placeholder documentado para una futura
    feature de volatilidad GARCH que sí sea causal.

    Para que una vol GARCH sirva como FEATURE de entrenamiento (a diferencia
    de su uso legítimo como motor de riesgo para la foto actual, ver
    `models.garch`), habría que re-ajustar el modelo en cada fecha de corte
    usando SOLO los datos disponibles hasta ese momento (ventana expansiva o
    rolling), no una vez sobre toda la serie — literalmente lo mismo que ya
    hace `models.arima.walk_forward_directional_accuracy` para el baseline
    ARIMA, pero para GARCH. Es costoso (un ajuste de grilla de 6 modelos por
    cada fecha, no uno solo para toda la serie) y por eso se difiere a una
    fase futura de este proyecto en vez de resolverse acá. No debe llamarse
    esta función todavía: existe solo para documentar la intención y dejar
    la puerta abierta.
    """
    raise NotImplementedError(
        "garch_feature_walkforward todavía no está implementada — ver su docstring. "
        "Usá 'vol_realizada_20' o 'vol_ewma' de build_feature_matrix mientras tanto."
    )


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
