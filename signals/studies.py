"""Motor de estudios técnicos de crypto-quant-desk (Fase 7a).

CONTEXTO HONESTO — léase antes de usar este módulo: el proyecto ya
demostró, en varias fases (ver `ml/models.py::evaluate_primary`,
`ml/models.py::evaluate_with_without_onchain`, la comparación diario vs.
horario de Fase 6c), que ni las features técnicas clásicas ni el on-chain
le ganan de forma consistente a un baseline trivial. Los estudios de este
módulo NO son una excepción a ese hallazgo ni pretenden serlo: son
herramientas de ANÁLISIS VISUAL y de APOYO a la decisión del usuario, no
un sistema con edge direccional demostrado. Se incluyen deliberadamente
estudios populares SIN respaldo estadístico fuerte (el caso más claro es
Fibonacci: no hay evidencia robusta de que los mercados "respeten" esos
niveles más de lo que respetarían un nivel redondo cualquiera) porque el
usuario los pidió explícitamente para su propio análisis — se documentan
así, sin maquillaje, en cada función correspondiente.

Reutiliza `signals.indicators` (RSI, MACD, Bollinger, ATR, medias) tal
cual — no reimplementa ninguno de esos cálculos.

CAUSALIDAD — nota importante y distinta al resto del proyecto: `stochastic`
y `adx` son TRAILING (rolling hacia atrás, igual que todo `signals.indicators`)
y sí son aptos como insumo de una regla/feature en tiempo real. `swing_points`,
en cambio, usa una ventana CENTRADA a propósito (necesita conocer velas
POSTERIORES para confirmar que un máximo/mínimo local fue efectivamente un
swing) — es información solo reconstruible en retrospectiva, útil para
análisis histórico y para anclar `fibonacci_levels`/`support_resistance`,
pero NUNCA debe usarse como feature de un modelo entrenado ni como señal de
trading en tiempo real sin rediseñarla para ser trailing. `fibonacci_levels`
y `pivot_points` toman niveles ya elegidos (escalares) y no tienen problema
de causalidad en sí mismas — el problema de causalidad, si existe, está en
CÓMO se elige el swing/período de referencia que se les pasa. `ichimoku`
(Fase 10b) se suma a la lista de casos NO trailing: su línea `chikou` usa
`close` FUTURO respecto de cada fecha (ver su propio docstring) — mismo
criterio que `swing_points`, solo para lectura visual retrospectiva.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from signals.indicators import atr

logger = logging.getLogger(__name__)

# Ratios de Fibonacci convencionales (no derivados de ningún ajuste
# estadístico — son, literalmente, la convención de mercado tal cual la usa
# la mayoría de las plataformas de trading).
FIBONACCI_RETRACEMENT_RATIOS: tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
FIBONACCI_EXTENSION_RATIOS: tuple[float, ...] = (1.272, 1.618)

DEFAULT_SWING_WINDOW: int = 10
DEFAULT_SR_N_LEVELS: int = 5
DEFAULT_SR_WINDOW: int = 20
DEFAULT_STOCH_K: int = 14
DEFAULT_STOCH_D: int = 3
DEFAULT_ADX_WINDOW: int = 14
DEFAULT_ICHIMOKU_TENKAN: int = 9
DEFAULT_ICHIMOKU_KIJUN: int = 26
DEFAULT_ICHIMOKU_SENKOU_B: int = 52


def _format_ratio(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def fibonacci_levels(high: float, low: float, uptrend: bool = True) -> dict[str, float]:
    """Retrocesos y extensiones de Fibonacci entre un swing high y un swing
    low (ambos escalares — ver `swing_points` para detectarlos).

    HONESTIDAD: Fibonacci NO tiene respaldo predictivo probado — no hay
    evidencia estadística robusta de que el precio "reaccione" en estos
    niveles más de lo que reaccionaría en cualquier nivel redondo o
    arbitrario. Se incluye acá por pedido explícito del usuario, como
    herramienta de análisis visual, no como señal con edge demostrado (ver
    el CONTEXTO HONESTO del docstring del módulo).

    Convención (`uptrend=True`, el swing fue de `low` a `high`):
    - Retrocesos (`ratio` en [0, 1]): miden el retroceso hacia ABAJO desde
      el `high`, `nivel = high - ratio * (high - low)`. 0% = en el high (sin
      retroceso), 100% = de vuelta en el low (retroceso completo).
    - Extensiones (`ratio` > 1): proyectan continuación hacia ARRIBA más
      allá del `high`, `nivel = low + ratio * (high - low)`.

    Convención espejada si `uptrend=False` (el swing fue de `high` a
    `low`, se espera una reacción/rebote hacia arriba antes de continuar
    bajando): retrocesos miden el rebote hacia ARRIBA desde `low`
    (`nivel = low + ratio * (high - low)`), extensiones proyectan
    continuación hacia ABAJO más allá del `low`
    (`nivel = high - ratio * (high - low)`).

    Devuelve un dict `{"0.0%": nivel, "23.6%": nivel, ..., "161.8%": nivel}`.
    Lanza `ValueError` si `high <= low`.
    """
    if high <= low:
        raise ValueError(f"fibonacci_levels: 'high' ({high}) debe ser mayor que 'low' ({low})")

    price_range = high - low
    levels: dict[str, float] = {}

    for ratio in FIBONACCI_RETRACEMENT_RATIOS:
        level = (high - ratio * price_range) if uptrend else (low + ratio * price_range)
        levels[_format_ratio(ratio)] = float(level)

    for ratio in FIBONACCI_EXTENSION_RATIOS:
        level = (low + ratio * price_range) if uptrend else (high - ratio * price_range)
        levels[_format_ratio(ratio)] = float(level)

    return levels


def swing_points(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = DEFAULT_SWING_WINDOW
) -> pd.DataFrame:
    """Detecta swing highs/lows locales: una vela es swing high si su `high`
    es el máximo dentro de una ventana CENTRADA de `2*window + 1` velas (la
    propia vela más `window` velas antes y `window` después); análogamente
    para swing low con `low` y el mínimo.

    NO ES TRAILING A PROPÓSITO (ver CAUSALIDAD en el docstring del módulo):
    confirmar que una vela fue un swing requiere ver las `window` velas
    POSTERIORES — es información solo disponible en retrospectiva, pensada
    para análisis histórico (identificar soportes/resistencias pasados,
    anclar Fibonacci), no para una señal en tiempo real. Con empates
    (varias velas con el mismo máximo/mínimo dentro de la ventana) pueden
    marcarse varias como swing — no se resuelve a una sola, se documenta
    tal cual.

    Devuelve un `pd.DataFrame` con el mismo índice que `high`/`low`/`close`,
    columnas:
    - "swing_high": el valor de `high` en las velas marcadas como swing
      high, NaN en el resto.
    - "swing_low": ídem con `low` en las velas de swing low.
    - "is_swing_high"/"is_swing_low": máscaras booleanas equivalentes.
    """
    window_size = 2 * window + 1
    rolling_max = high.rolling(window=window_size, center=True).max()
    rolling_min = low.rolling(window=window_size, center=True).min()

    is_swing_high = (high == rolling_max) & rolling_max.notna()
    is_swing_low = (low == rolling_min) & rolling_min.notna()

    return pd.DataFrame(
        {
            "swing_high": high.where(is_swing_high),
            "swing_low": low.where(is_swing_low),
            "is_swing_high": is_swing_high,
            "is_swing_low": is_swing_low,
        },
        index=close.index,
    )


def _cluster_levels(values: list[float], n_levels: int, tolerance: float) -> list[float]:
    """Clustering 1D simple (enlace único): agrupa `values` ORDENADOS en
    clusters contiguos donde cada valor está a `tolerance` o menos del
    anterior, ordena los clusters por CANTIDAD de puntos agrupados (más
    puntos = nivel "más respetado" por el precio = más fuerte), y devuelve
    el promedio de los `n_levels` clusters más fuertes, ordenados
    ascendentemente por precio.
    """
    if not values:
        return []

    sorted_values = sorted(values)
    clusters: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    strongest = sorted(clusters, key=len, reverse=True)[:n_levels]
    representative_levels = [float(np.mean(cluster)) for cluster in strongest]
    return sorted(representative_levels)


def support_resistance(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    n_levels: int = DEFAULT_SR_N_LEVELS,
    window: int = DEFAULT_SR_WINDOW,
) -> dict:
    """Niveles de soporte y resistencia por clustering de swing points
    (`swing_points`, reutilizada tal cual — ver su nota de causalidad: esto
    hereda el mismo carácter "solo para análisis histórico/visual").

    Solo considera swings dentro de una ventana RECIENTE (las últimas
    `window * 5` velas) — no toda la historia disponible: en un activo con
    años de historia (p. ej. BTC, que pasó de ~$3.000 a seis cifras), los
    niveles más "fuertes" en términos de cantidad de swings acumulados
    suelen ser zonas de precio antiguas y ya irrelevantes para el precio
    actual si se mira toda la serie. Dentro de esa ventana reciente, agrupa
    los swing highs en clusters de precios cercanos entre sí (tolerancia =
    1% del rango de precio de esa misma ventana) y toma los `n_levels`
    clusters con MÁS swings agrupados como resistencia (más veces que el
    precio "respetó" esa zona al mirar hacia atrás = nivel más fuerte);
    análogamente con los swing lows para soporte.

    Devuelve un dict:
    - "resistencia": lista de niveles (ascendente), como mucho `n_levels`.
    - "soporte": lista de niveles (ascendente), como mucho `n_levels`.
    - "precio_actual": `close` de la última vela, para poder ubicar los
      niveles relativos al precio vigente.
    """
    swings = swing_points(high, low, close, window=window)

    lookback = window * 5
    recent_swings = swings.tail(lookback)
    recent_high = high.tail(lookback)
    recent_low = low.tail(lookback)
    price_range = float(recent_high.max() - recent_low.min())
    tolerance = price_range * 0.01 if price_range > 0 else 0.0

    resistance_values = recent_swings["swing_high"].dropna().tolist()
    support_values = recent_swings["swing_low"].dropna().tolist()

    return {
        "resistencia": _cluster_levels(resistance_values, n_levels=n_levels, tolerance=tolerance),
        "soporte": _cluster_levels(support_values, n_levels=n_levels, tolerance=tolerance),
        "precio_actual": float(close.iloc[-1]),
    }


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = DEFAULT_STOCH_K, d: int = DEFAULT_STOCH_D
) -> pd.DataFrame:
    """Oscilador estocástico: %K y %D.

        %K_t = 100 * (close_t - min(low, k)) / (max(high, k) - min(low, k))
        %D_t = SMA(%K, d)

    TRAILING por construcción (rolling hacia atrás) — sin lookahead. Como
    `close` siempre está en `[min(low,k), max(high,k)]` por definición,
    %K queda matemáticamente acotado en [0, 100] salvo en el caso
    degenerado de rango cero (`high == low` en toda la ventana), donde el
    denominador es 0 y el valor queda en NaN (no se inventa un 50 "neutral"
    ni se fuerza una división por cero).
    """
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    denom = highest_high - lowest_low

    percent_k = pd.Series(
        np.where(denom > 0.0, 100.0 * (close - lowest_low) / denom, np.nan), index=close.index
    )
    percent_d = percent_k.rolling(window=d).mean()

    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = DEFAULT_ADX_WINDOW) -> pd.Series:
    """ADX (Average Directional Index) de Wilder: mide la FUERZA de la
    tendencia (no su dirección) — útil como filtro (p. ej. desconfiar de
    señales direccionales cuando ADX es bajo, mercado sin tendencia clara).

    Reutiliza `signals.indicators.atr` (ya calcula el True Range suavizado
    a la Wilder) como denominador normalizador de +DI/-DI, en vez de
    reimplementar el suavizado del TR acá.

        +DM_t = max(high_t - high_{t-1}, 0) si supera a -DM_t, si no 0
        -DM_t = max(low_{t-1} - low_t, 0) si supera a +DM_t, si no 0
        +DI_t = 100 * EWMA_Wilder(+DM) / ATR_t ; -DI_t análogo
        DX_t  = 100 * |+DI_t - -DI_t| / (+DI_t + -DI_t)
        ADX_t = EWMA_Wilder(DX)

    TRAILING por construcción — sin lookahead. Siempre >= 0 (es un
    promedio de una cantidad no-negativa por definición), salvo NaN en el
    warmup o en el caso degenerado +DI=-DI=0 (0/0).
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=high.index)

    smoothed_atr = atr(high, low, close, window=window)
    plus_dm_smooth = plus_dm.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    plus_di = 100.0 * plus_dm_smooth / smoothed_atr
    minus_di = 100.0 * minus_dm_smooth / smoothed_atr

    di_sum = plus_di + minus_di
    dx = pd.Series(
        np.where(di_sum > 0.0, 100.0 * (plus_di - minus_di).abs() / di_sum, np.nan), index=high.index
    )
    return dx.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def pivot_points(high: float, low: float, close: float) -> dict[str, float]:
    """Pivotes clásicos (fórmula estándar, la más común entre plataformas
    de trading), a partir del high/low/close de UN período de referencia
    (p. ej. la vela anterior) — para proyectar niveles del período
    siguiente.

        P  = (high + low + close) / 3
        R1 = 2*P - low        S1 = 2*P - high
        R2 = P + (high - low) S2 = P - (high - low)
        R3 = high + 2*(P - low)   S3 = low - 2*(high - P)

    Devuelve un dict `{"P":..., "R1":..., "R2":..., "R3":..., "S1":...,
    "S2":..., "S3":...}`.
    """
    pivot = (high + low + close) / 3.0
    price_range = high - low
    return {
        "P": float(pivot),
        "R1": float(2.0 * pivot - low),
        "R2": float(pivot + price_range),
        "R3": float(high + 2.0 * (pivot - low)),
        "S1": float(2.0 * pivot - high),
        "S2": float(pivot - price_range),
        "S3": float(low - 2.0 * (high - pivot)),
    }


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_window: int = DEFAULT_ICHIMOKU_TENKAN,
    kijun_window: int = DEFAULT_ICHIMOKU_KIJUN,
    senkou_b_window: int = DEFAULT_ICHIMOKU_SENKOU_B,
) -> pd.DataFrame:
    """Sistema Ichimoku Kinko Hyo (períodos convencionales 9/26/52).

    HONESTIDAD (ver CONTEXTO HONESTO del módulo): Ichimoku es POPULAR
    entre traders técnicos, pero SIN respaldo estadístico de edge probado
    en este proyecto — es una herramienta de lectura visual de tendencia y
    soporte/resistencia, no una señal predictiva.

    Cinco líneas:
    - tenkan (línea de conversión): promedio del máximo y el mínimo de los
      últimos `tenkan_window` períodos. La más rápida de las cinco.
    - kijun (línea base): igual que tenkan pero con `kijun_window`
      períodos — más lenta, funciona como referencia de tendencia de
      mediano plazo.
    - senkou_a (span A, un borde de la nube): promedio de tenkan y kijun,
      DESPLAZADO `kijun_window` períodos hacia ADELANTE.
    - senkou_b (span B, el otro borde de la nube): promedio del máximo y
      el mínimo de los últimos `senkou_b_window` períodos, desplazado
      `kijun_window` períodos hacia adelante.
    - chikou (línea rezagada): el `close` actual, desplazado
      `kijun_window` períodos hacia ATRÁS.

    LA NUBE ("kumo") es el área entre senkou_a y senkou_b — cuando
    senkou_a > senkou_b algunos traders la leen como alcista (y viceversa);
    el frontend la dibuja rellenando el área entre ambas líneas.

    CAUSALIDAD de senkou_a/senkou_b: el desplazamiento hacia adelante NO
    es lookahead — `senkou_a[t]` es el promedio de tenkan/kijun calculado
    con datos disponibles hasta `t - kijun_window`, solo reubicado
    `kijun_window` posiciones más adelante (la convención estándar del
    indicador). Esta implementación reubica esos valores DENTRO del rango
    de fechas ya existente — a diferencia de un gráfico Ichimoku
    tradicional, que suele extender la nube más allá de la última vela
    real (fechas futuras), acá no se fabrican fechas nuevas: no se ve esa
    proyección más allá de la última fecha disponible.

    CAUSALIDAD de chikou: es EXACTAMENTE LO OPUESTO — `chikou[t] =
    close[t + kijun_window]`, información del FUTURO respecto de `t`. Es
    deliberado (la convención estándar de la línea rezagada, para comparar
    el precio actual contra el de `kijun_window` períodos atrás) pero NO
    es trailing: mismo criterio que `swing_points` (ver CAUSALIDAD del
    módulo) — solo para lectura visual retrospectiva, nunca como feature
    de un modelo ni señal en tiempo real. Por construcción, sus últimos
    `kijun_window` valores son siempre NaN (no hay close futuro todavía).

    Devuelve un `pd.DataFrame` con columnas ["tenkan", "kijun", "senkou_a",
    "senkou_b", "chikou"], mismo índice que `high`/`low`/`close`.
    """
    tenkan = (high.rolling(window=tenkan_window).max() + low.rolling(window=tenkan_window).min()) / 2.0
    kijun = (high.rolling(window=kijun_window).max() + low.rolling(window=kijun_window).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(kijun_window)
    senkou_b = (
        (high.rolling(window=senkou_b_window).max() + low.rolling(window=senkou_b_window).min()) / 2.0
    ).shift(kijun_window)
    chikou = close.shift(-kijun_window)

    return pd.DataFrame(
        {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou},
        index=close.index,
    )


def all_studies(df: pd.DataFrame) -> dict:
    """Corre TODOS los estudios de este módulo más los de
    `signals.indicators.add_all_indicators` (reutilizado, no reimplementado)
    sobre un OHLCV estandarizado (`data.loaders.get_prices`), y arma un
    resumen estructurado de los valores ACTUALES (última vela).

    El swing más reciente de cada tipo (`swing_points`) determina la
    dirección que se le pasa a `fibonacci_levels`: si el último swing high
    es más reciente que el último swing low, el movimiento más reciente fue
    ALCISTA (`uptrend=True`); si no, fue bajista. Si todavía no hay swings
    detectados (serie corta), "fibonacci" queda en `None`.

    Los pivotes (`pivot_points`) se calculan con el high/low/close de la
    ÚLTIMA vela disponible, como referencia para la vela SIGUIENTE (la
    convención estándar de pivotes: se calculan con el período ya cerrado
    para proyectar el que viene).

    Devuelve un dict con las claves "fecha", "precio", "indicadores"
    (rsi_14/macd/macd_signal/macd_hist/bb_pct_b/bb_zscore/sma_20/sma_50/
    atr_14/vwap/obv — las últimas dos, Fase 10b), "estocastico"
    ({"k":..., "d":...}), "adx", "ichimoku" (Fase 10b, últimos valores de
    tenkan/kijun/senkou_a/senkou_b/chikou — "chikou" suele salir `None`
    acá, ver su docstring: sus últimos `kijun_window` valores son siempre
    NaN por construcción), "soporte_resistencia", "pivotes", "fibonacci"
    (o `None`).
    """
    from signals.indicators import add_all_indicators

    high, low, close = df["high"], df["low"], df["close"]
    indicators_df = add_all_indicators(df)
    stoch_df = stochastic(high, low, close)
    adx_series = adx(high, low, close)
    ichimoku_df = ichimoku(high, low, close)
    swings = swing_points(high, low, close)

    last_swing_high = swings["swing_high"].dropna()
    last_swing_low = swings["swing_low"].dropna()
    fibonacci = None
    if len(last_swing_high) > 0 and len(last_swing_low) > 0:
        high_idx, low_idx = last_swing_high.index[-1], last_swing_low.index[-1]
        uptrend = high_idx > low_idx
        swing_high_price = float(last_swing_high.iloc[-1])
        swing_low_price = float(last_swing_low.iloc[-1])
        if swing_high_price > swing_low_price:
            fibonacci = fibonacci_levels(swing_high_price, swing_low_price, uptrend=uptrend)

    last_row = df.iloc[-1]
    pivots = pivot_points(float(last_row["high"]), float(last_row["low"]), float(last_row["close"]))

    def _last_or_none(series: pd.Series) -> float | None:
        value = series.iloc[-1]
        return float(value) if pd.notna(value) else None

    return {
        "fecha": df.index[-1],
        "precio": float(close.iloc[-1]),
        "indicadores": {
            "rsi_14": _last_or_none(indicators_df["rsi_14"]),
            "macd": _last_or_none(indicators_df["macd"]),
            "macd_signal": _last_or_none(indicators_df["macd_signal"]),
            "macd_hist": _last_or_none(indicators_df["macd_hist"]),
            "bb_pct_b": _last_or_none(indicators_df["bb_pct_b"]),
            "bb_zscore": _last_or_none(indicators_df["bb_zscore"]),
            "sma_20": _last_or_none(indicators_df["sma_20"]),
            "sma_50": _last_or_none(indicators_df["sma_50"]),
            "atr_14": _last_or_none(indicators_df["atr_14"]),
            "vwap": _last_or_none(indicators_df["vwap"]),
            "obv": _last_or_none(indicators_df["obv"]),
        },
        "estocastico": {"k": _last_or_none(stoch_df["stoch_k"]), "d": _last_or_none(stoch_df["stoch_d"])},
        "adx": _last_or_none(adx_series),
        "ichimoku": {
            "tenkan": _last_or_none(ichimoku_df["tenkan"]),
            "kijun": _last_or_none(ichimoku_df["kijun"]),
            "senkou_a": _last_or_none(ichimoku_df["senkou_a"]),
            "senkou_b": _last_or_none(ichimoku_df["senkou_b"]),
            "chikou": _last_or_none(ichimoku_df["chikou"]),
        },
        "soporte_resistencia": support_resistance(close, high, low),
        "pivotes": pivots,
        "fibonacci": fibonacci,
    }
