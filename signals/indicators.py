"""Indicadores técnicos clásicos, vectorizados con pandas (sin TA-Lib ni
dependencias externas de análisis técnico).

Todas las funciones esperan `pd.Series`/`pd.DataFrame` con índice temporal ya
estandarizado (ver data.loaders.get_prices) y no imponen ninguna frecuencia
específica: los `window`/`span` se interpretan en cantidad de observaciones,
no en tiempo calendario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_VWAP_WINDOW = 20


def sma(series: pd.Series, window: int) -> pd.Series:
    """Media móvil simple (SMA) de `window` observaciones."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Media móvil exponencial (EMA) de `span` observaciones.

    Usa la forma recursiva `adjust=False`:
        EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1},  alpha = 2 / (span + 1)
    que es la convención estándar en plataformas de trading, a diferencia del
    promedio ponderado que usa pandas por defecto (`adjust=True`).
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) de Wilder.

    Las ganancias y pérdidas período a período se suavizan con la media móvil
    exponencial de Wilder (alpha = 1/window, forma recursiva), no con una
    media simple:
        avg_gain_t = avg_gain_{t-1} + (1/window) * (gain_t - avg_gain_{t-1})

    RSI = 100 - 100 / (1 + avg_gain / avg_loss)

    Rango: [0, 100]. Si avg_loss es 0 (racha alcista pura dentro de la
    ventana), la división da +inf y el RSI resultante es 100 de forma
    natural. Si tanto avg_gain como avg_loss son 0 (precio plano), el
    resultado sería NaN por 0/0; ese caso se define explícitamente como 50
    (neutral).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))

    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    out = out.where(~flat, 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence).

    macd   = EMA(close, fast) - EMA(close, slow)
    signal = EMA(macd, signal)
    hist   = macd - signal   (histograma: mide el momentum de la convergencia)
    """
    ema_fast = ema(close, span=fast)
    ema_slow = ema(close, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger.

    mid       = SMA(close, window)
    std       = desvío estándar móvil de `close` (poblacional, ddof=0)
    upper/low = mid +/- num_std * std
    pct_b     = (close - lower) / (upper - lower)  → posición relativa del
                precio dentro de la banda (0 = banda inferior, 1 = superior;
                puede salir de [0,1] si el precio rompe la banda)
    bandwidth = (upper - lower) / mid  → ancho relativo de la banda (proxy de
                volatilidad reciente)
    zscore    = (close - mid) / std  → cuántos desvíos estándar está el
                precio por encima/debajo de su media móvil
    """
    mid = sma(close, window=window)
    std = close.rolling(window=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    zscore = (close - mid) / std
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": bandwidth, "zscore": zscore}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (ATR) de Wilder.

    True Range (TR) = max(high - low, |high - close_prev|, |low - close_prev|)
    ATR = media móvil exponencial de Wilder (alpha = 1/window) del TR.

    Nota sobre datos de solo-cierre (p. ej. CoinMetrics/CoinGecko vía
    data.loaders, donde open=high=low=close): ahí high-low=0 y los otros dos
    términos del TR coinciden con |close - close_prev|, así que el TR (y por
    lo tanto el ATR) degrada a una media móvil de la variación absoluta del
    cierre. No representa un rango intradía real en ese caso, solo la
    magnitud del cambio día a día.

    El primer valor de la serie no tiene close_prev disponible, así que el TR
    queda en NaN en ese punto (no se aproxima con high-low solo) para no
    introducir un valor artificialmente bajo al inicio de la serie.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = DEFAULT_VWAP_WINDOW
) -> pd.Series:
    """VWAP (Volume Weighted Average Price) en VENTANA MÓVIL de `window`
    velas — NO acumulado desde el inicio de la serie.

    ELECCIÓN DE DISEÑO (documentada, ver Fase 10a/10b): el VWAP "clásico"
    intradía se acumula desde la apertura de una sesión de mercado y se
    reinicia cada día — no aplica tal cual acá, porque este proyecto
    trabaja con velas diarias/horarias de un mercado (cripto) que opera
    24/7, sin sesiones que reiniciar. Un VWAP acumulado desde el inicio de
    TODA la serie (años de historia en algunos activos) se volvería casi
    plano con el tiempo y perdería utilidad como referencia de precio
    reciente. Esta versión es una ventana móvil: funciona como una media
    móvil ponderada por volumen (en vez de una media simple como la SMA),
    dándole más peso a las velas de mayor actividad dentro de la ventana.

        precio_típico_t = (high_t + low_t + close_t) / 3
        VWAP_t = sum(precio_típico * volume, últimas `window` velas)
                 / sum(volume, últimas `window` velas)

    TRAILING por construcción (rolling hacia atrás) — sin lookahead. Si
    `volume` viene en NaN (fuentes de solo-cierre como CoinMetrics/
    CoinGecko, ver data.loaders) o la suma de volumen de la ventana es 0,
    el resultado queda en NaN — no se inventa un peso uniforme como
    sustituto.
    """
    typical_price = (high + low + close) / 3.0
    sum_price_volume = (typical_price * volume).rolling(window=window).sum()
    sum_volume = volume.rolling(window=window).sum()
    return pd.Series(np.where(sum_volume > 0.0, sum_price_volume / sum_volume, np.nan), index=close.index)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume (OBV): acumula el volumen con signo según haya
    subido o bajado el cierre respecto a la vela anterior.

        signo_t = +1 si close_t > close_{t-1}
                  -1 si close_t < close_{t-1}
                   0 si close_t == close_{t-1}
        OBV_t = OBV_{t-1} + signo_t * volume_t     (OBV en la primera vela = volume_0)

    Es un ACUMULADO, no un oscilador con rango fijo — se interpreta por su
    PENDIENTE/tendencia ("¿el volumen acompaña al movimiento del precio?"),
    no por su nivel absoluto (que no tiene una escala "buena" o "mala" en
    sí misma). Si `volume` viene en NaN (fuentes de solo-cierre, ver
    data.loaders), esa vela queda en NaN y el acumulado sigue desde la
    última vela válida (`Series.cumsum` con `skipna=True`, el default).
    """
    direction = np.sign(close.diff())
    signed_volume = direction * volume
    signed_volume.iloc[0] = volume.iloc[0]
    return signed_volume.cumsum()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de indicadores técnicos a un DataFrame OHLCV
    estandarizado (el que devuelve `data.loaders.get_prices`).

    No modifica `df` in-place: devuelve una copia con las columnas agregadas
    ["sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "macd", "macd_signal",
    "macd_hist", "bb_mid", "bb_upper", "bb_lower", "bb_pct_b", "bb_bandwidth",
    "bb_zscore", "atr_14", "vwap", "obv"] (las últimas dos, Fase 10b).
    """
    out = df.copy()
    close = out["close"]

    out["sma_20"] = sma(close, window=20)
    out["sma_50"] = sma(close, window=50)
    out["ema_12"] = ema(close, span=12)
    out["ema_26"] = ema(close, span=26)
    out["rsi_14"] = rsi(close, window=14)

    macd_df = macd(close)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    out["macd_hist"] = macd_df["hist"]

    bb_df = bollinger_bands(close)
    out["bb_mid"] = bb_df["mid"]
    out["bb_upper"] = bb_df["upper"]
    out["bb_lower"] = bb_df["lower"]
    out["bb_pct_b"] = bb_df["pct_b"]
    out["bb_bandwidth"] = bb_df["bandwidth"]
    out["bb_zscore"] = bb_df["zscore"]

    out["atr_14"] = atr(out["high"], out["low"], out["close"], window=14)

    out["vwap"] = vwap(out["high"], out["low"], out["close"], out["volume"])
    out["obv"] = obv(out["close"], out["volume"])

    return out
