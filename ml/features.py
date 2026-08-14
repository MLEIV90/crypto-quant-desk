"""Matriz de features para el pipeline de ML de crypto-quant-desk.

Fase 3a: solo construcción de features y alineación con las etiquetas de
`ml.labeling`. El entrenamiento de un modelo es Fase 3c; la validación
walk-forward es Fase 3b. Fase 5b agrega la opción `include_onchain` de
`build_feature_matrix`, que mergea las features de `data/onchain.py`
(ver ese módulo — también estrictamente trailing) — el resto del pipeline
(alineación, pesos por solapamiento) es idéntico, con o sin on-chain.

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

HOTFIX (Fase 6a, bug de raíz de Fase 3a): `build_feature_matrix` incluía 12
NIVELES de precio crudos como feature —
open/high/low/close/sma_20/sma_50/ema_12/ema_26/bb_mid/bb_upper/bb_lower/atr_14
(más macd/macd_signal, en unidades de precio, aunque no eran niveles per
se) — que NO son estacionarios: su escala absoluta crece con el precio del
activo a lo largo de los años (p. ej. `close` pasó de ~$3.000 a ~$100.000
en la historia de BTC). Un árbol de decisión (el modelo primario, Fase 3c)
aprende reglas del tipo "si close > X entonces...", y ese umbral X, calibrado
en un rango de precios de una época, no tiene ningún motivo para seguir
siendo relevante en otra — no es lookahead (no mira el futuro), pero SÍ es
un problema de generalización: el modelo memoriza el rango de precios visto
en train en vez de aprender un patrón que se repita. Esto era consistente
con el resultado nulo de Fase 3c/5b (el modelo no le ganaba a los baselines
triviales). La corrección: ninguna feature de `build_feature_matrix` es ya
un nivel de precio en unidades absolutas — todas son ratios, diferencias
normalizadas o indicadores ya acotados (ver el detalle de cada una en
`build_feature_matrix`), invariantes a la escala del precio.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import PERIODS_PER_YEAR
from data.onchain import build_onchain_features, load_onchain, merge_onchain
from ml.labeling import DEFAULT_VOLATILITY_SPAN, get_daily_volatility, get_sample_weights
from signals.indicators import add_all_indicators
from signals.returns import simple_returns

logger = logging.getLogger(__name__)

_RETURN_LAGS: tuple[int, ...] = (1, 5, 10, 20)
_REALIZED_VOL_WINDOW: int = 20


_VOLUME_REL_WINDOW: int = 20


def build_feature_matrix(
    df_ohlcv: pd.DataFrame, include_onchain: bool = False, asset: str | None = None
) -> pd.DataFrame:
    """Construye la matriz de features X a partir de un OHLCV estandarizado
    (el de `data.loaders.get_prices`). No modifica `df_ohlcv` in-place.

    Usa `signals.indicators.add_all_indicators` como INSUMO interno (todos
    sus indicadores son trailing, ver ese módulo) pero NINGUNA de sus
    columnas en unidades de precio absolutas queda como feature final — ver
    el HOTFIX de Fase 6a en el docstring del módulo: la matriz que devuelve
    esta función es SIEMPRE estacionaria/escala-invariante, columna por
    columna. Las features son:

    Momentum y volatilidad (Fase 3a, sin cambios):
    - "ret_1"/"ret_5"/"ret_10"/"ret_20": retorno simple de los últimos
      1/5/10/20 días (`close_t / close_{t-k} - 1`, vía `Series.pct_change`).
    - "vol_realizada_20": desvío estándar rolling de 20 días de los
      retornos simples, ANUALIZADO (`* sqrt(config.PERIODS_PER_YEAR)`).
    - "vol_ewma": volatilidad diaria vía EWMA de retornos
      (`ml.labeling.get_daily_volatility`, span=`DEFAULT_VOLATILITY_SPAN`).

    Osciladores ya acotados/estacionarios por construcción (Fase 3a, de
    `add_all_indicators`, sin cambios): "rsi_14" (∈[0,100]), "bb_pct_b"
    (posición relativa dentro de la banda), "bb_bandwidth" (ancho relativo
    de la banda), "bb_zscore" (desvíos estándar respecto de la media móvil).

    Transformaciones estacionarias de niveles (Fase 6a — reemplazan a los
    niveles crudos, ver HOTFIX del módulo):
    - "dist_sma20"/"dist_sma50" = `(close - sma) / sma`: distancia
      PORCENTUAL del precio a su media móvil de 20/50 — el mismo valor
      (p. ej. "5% por encima de la SMA20") significa lo mismo hoy que hace
      cinco años, a diferencia del nivel crudo de la SMA.
    - "ma_cross" = `(ema_12 - ema_26) / ema_26`: cruce de medias EMA
      normalizado por el nivel de la EMA lenta — mismo gap relativo, sea
      cual sea la escala de precio del momento (equivalente a la lógica de
      "trend" de `signals.engine.compute_signal_components`, pero expuesto
      acá como feature de ML en vez de componente de score).
    - "intraday_range" = `(high - low) / close`: rango intradía como
      fracción del precio de cierre — un proxy de volatilidad/actividad del
      día, ya relativo.
    - "body" = `(close - open) / open`: retorno intradía (apertura a
      cierre) — mismo tipo de magnitud que `ret_1`, pero dentro del día.
    - "atr_norm" = `atr_14 / close`: ATR (rango verdadero promedio, en
      unidades de precio) normalizado por el precio actual — vuelve
      comparable la volatilidad de rango entre épocas de precio distinto.
    - "macd_norm"/"macd_signal_norm"/"macd_hist_norm" = cada componente del
      MACD (en unidades de precio, porque es una diferencia de EMAs)
      dividido por `close` — mismo racional que `atr_norm`.
    - "rel_volume" = `volume / volume.rolling(20).mean()`: volumen del día
      relativo a su propio promedio móvil reciente — un volumen "alto" o
      "bajo" relativo a lo normal de ESE activo en ESE momento, en vez de
      un número crudo cuya escala también cambia con la liquidez del
      mercado a lo largo de los años.

    Todas las transformaciones de arriba son TRAILING por construcción
    (usan `close`/`open`/`high`/`low`/`volume` y los indicadores de
    `add_all_indicators`, todos trailing, solo hasta la fecha t) — dividir
    o restar dos cantidades de la misma fecha no introduce ningún
    lookahead nuevo.

    Si `include_onchain=True` (Fase 5b), además mergea las features on-chain
    de `asset` (obligatorio en ese caso): `data.onchain.load_onchain` +
    `data.onchain.build_onchain_features` + `data.onchain.merge_onchain`,
    reutilizadas tal cual — no se reimplementa ningún cálculo on-chain acá.
    Todas esas features ya son TRAILING/causales y estacionarias por
    construcción (ver `data/onchain.py`). Si `asset` no tiene snapshot
    on-chain exportado (p. ej. SOL, ver `data/onchain.py`), se loguea un
    warning y la matriz devuelta queda IGUAL que con `include_onchain=False`
    — no se aborta. Si el activo tiene cobertura PARCIAL (p. ej. BNB/LTC,
    sin flujos de exchange), se agregan las que estén disponibles y se
    loguea cuáles fueron.

    Devuelve un DataFrame con el mismo índice que `df_ohlcv` y SOLO las
    columnas de feature descriptas arriba (más las on-chain, si aplica) —
    a propósito, NUNCA las columnas OHLCV crudas ni los indicadores en
    unidades de precio absolutas (ver HOTFIX de Fase 6a). El warmup de las
    ventanas rolling (NaN al principio de la serie) es responsabilidad de
    `align_features_labels`, igual que con cualquier otra feature.
    """
    raw = add_all_indicators(df_ohlcv)
    close = raw["close"]
    open_ = raw["open"]
    high = raw["high"]
    low = raw["low"]
    volume = raw["volume"]

    out = pd.DataFrame(index=raw.index)

    for lag in _RETURN_LAGS:
        out[f"ret_{lag}"] = close.pct_change(periods=lag)

    realized_vol = simple_returns(close).rolling(window=_REALIZED_VOL_WINDOW).std(ddof=1)
    out["vol_realizada_20"] = realized_vol * np.sqrt(PERIODS_PER_YEAR)
    out["vol_ewma"] = get_daily_volatility(close, span=DEFAULT_VOLATILITY_SPAN)

    out["rsi_14"] = raw["rsi_14"]
    out["bb_pct_b"] = raw["bb_pct_b"]
    out["bb_bandwidth"] = raw["bb_bandwidth"]
    out["bb_zscore"] = raw["bb_zscore"]

    out["dist_sma20"] = (close - raw["sma_20"]) / raw["sma_20"]
    out["dist_sma50"] = (close - raw["sma_50"]) / raw["sma_50"]
    out["ma_cross"] = (raw["ema_12"] - raw["ema_26"]) / raw["ema_26"]
    out["intraday_range"] = (high - low) / close
    out["body"] = (close - open_) / open_
    out["atr_norm"] = raw["atr_14"] / close
    out["macd_norm"] = raw["macd"] / close
    out["macd_signal_norm"] = raw["macd_signal"] / close
    out["macd_hist_norm"] = raw["macd_hist"] / close
    out["rel_volume"] = volume / volume.rolling(window=_VOLUME_REL_WINDOW).mean()

    if include_onchain:
        if asset is None:
            raise ValueError("build_feature_matrix: 'asset' es obligatorio si include_onchain=True")
        try:
            onchain_raw = load_onchain(asset)
        except FileNotFoundError as exc:
            logger.warning(
                "build_feature_matrix: '%s' sin on-chain disponible (%s) — se sigue solo con features técnicas.",
                asset, exc,
            )
        else:
            onchain_feats = build_onchain_features(onchain_raw)
            out = merge_onchain(out, onchain_feats)
            logger.info(
                "build_feature_matrix: '%s' — features on-chain agregadas: %s",
                asset, list(onchain_feats.columns),
            )

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
