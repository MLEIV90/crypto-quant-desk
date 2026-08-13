"""Loaders unificados de precios OHLCV para crypto-quant-desk.

Punto de entrada público: `get_prices`. Todo el resto de funciones son
implementación interna (prefijo `_`).

Contrato de salida de `get_prices` (y de todos los loaders internos tras pasar
por `_standardize`):
    - Índice: DatetimeIndex en UTC, tz-aware, ordenado ascendentemente y sin
      duplicados.
    - Columnas: exactamente ["open", "high", "low", "close", "volume"], dtype float.
    - Fuentes que solo reportan precio de cierre (CoinMetrics, CoinGecko)
      rellenan open/high/low con el close; volume queda en NaN salvo que la
      fuente lo reporte explícitamente (CoinGecko sí reporta total_volumes).

No se implementa ningún modelo (GARCH/ARIMA/ML) en este módulo: es Fase 0,
solo andamiaje de datos.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (
    CACHE_DIR,
    DEFAULT_INTERVAL,
    DEFAULT_SOURCE,
    FALLBACK_SOURCES,
    RAW_START_DATE,
    SNAPSHOT_DIR,
    UNIVERSE,
)

logger = logging.getLogger(__name__)

STANDARD_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
COINMETRICS_CSV_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{symbol}.csv"
COINGECKO_RANGE_URL = "https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range"

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
BINANCE_MAX_CANDLES = 1000
HTTP_TIMEOUT_SECONDS = 30

# Duración de una vela de Binance en milisegundos, usada para paginar.
_BINANCE_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

# Alias de pandas.date_range para detectar gaps por intervalo.
_INTERVAL_TO_PANDAS_FREQ: dict[str, str] = {
    "1m": "min",
    "5m": "5min",
    "15m": "15min",
    "1h": "h",
    "4h": "4h",
    "1d": "D",
}


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------


def get_prices(
    asset: str,
    source: str = DEFAULT_SOURCE,
    interval: str = DEFAULT_INTERVAL,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Devuelve precios OHLCV estandarizados para `asset`.

    Parámetros
    ----------
    asset:
        Ticker interno definido en `config.UNIVERSE` (p. ej. "BTC").
    source:
        Fuente primaria a intentar ("binance", "coinmetrics", "coingecko", o
        "store" para leer un snapshot local ya exportado por
        `scripts/export_snapshot.py` desde `config.SNAPSHOT_DIR`, útil para
        verificar el pipeline offline con un dataset ya compartido).
    interval:
        Intervalo de las velas (p. ej. "1d"). Ver `config.DEFAULT_INTERVAL`.
    start, end:
        Fechas ISO ("YYYY-MM-DD"), interpretadas en UTC. Si se omiten, se usa
        `config.RAW_START_DATE` como inicio y "hoy" (UTC) como fin.
    use_cache:
        Si es True, intenta leer de `data/cache/*.parquet` antes de golpear la
        red, y escribe en caché tras una descarga exitosa.

    Devuelve
    -------
    pd.DataFrame con índice UTC tz-aware, ordenado, sin duplicados, y columnas
    exactas ["open", "high", "low", "close", "volume"] en float.

    Si `source` falla (red, HTTP, símbolo inválido, etc.), se prueban en orden
    las fuentes de `config.FALLBACK_SOURCES` que no sean igual a `source`.
    Lanza `RuntimeError` si todas las fuentes fallan.
    """
    if asset not in UNIVERSE:
        raise ValueError(f"Activo '{asset}' no está definido en config.UNIVERSE")

    start = start or RAW_START_DATE
    end = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    cache_path = _cache_path(asset, source, interval)
    if use_cache and cache_path.exists():
        df = _read_cache(cache_path)
        return _slice(df, start, end)

    sources_to_try = [source] + [s for s in FALLBACK_SOURCES if s != source]
    last_exc: Exception | None = None
    for src in sources_to_try:
        try:
            logger.info("Descargando %s desde fuente '%s' (%s a %s)", asset, src, start, end)
            raw = _load_from_source(asset, src, interval, start, end)
            df = _standardize(raw, interval=interval, source=src)
        except Exception as exc:  # noqa: BLE001 - queremos degradar a la siguiente fuente ante cualquier falla
            logger.warning("Fuente '%s' falló para %s: %s", src, asset, exc)
            last_exc = exc
            continue

        if use_cache:
            _write_cache(df, _cache_path(asset, src, interval))
        return _slice(df, start, end)

    raise RuntimeError(
        f"No se pudo obtener precios de '{asset}' desde ninguna fuente {sources_to_try}"
    ) from last_exc


# --------------------------------------------------------------------------
# Dispatch de fuentes
# --------------------------------------------------------------------------


def _load_from_source(asset: str, source: str, interval: str, start: str, end: str) -> pd.DataFrame:
    if source == "store":
        # No usa config.UNIVERSE: el snapshot se nombra con el ticker interno
        # directamente (ver scripts/export_snapshot.py), no con un símbolo de exchange.
        return _load_store(asset, interval, start, end)

    symbols = UNIVERSE[asset]
    if source not in symbols:
        raise ValueError(f"Fuente '{source}' no está definida para '{asset}' en config.UNIVERSE")
    symbol = symbols[source]

    if source == "binance":
        return _load_binance(symbol, interval, start, end)
    if source == "coinmetrics":
        return _load_coinmetrics(symbol, start, end)
    if source == "coingecko":
        return _load_coingecko(symbol, start, end)
    raise ValueError(f"Fuente desconocida: '{source}'")


# --------------------------------------------------------------------------
# Loaders internos por fuente
# --------------------------------------------------------------------------


def _load_binance(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Descarga velas OHLCV desde Binance (`GET /api/v3/klines`), sin API key.

    Binance devuelve máximo 1000 velas por request, así que se pagina
    avanzando `startTime` hasta cubrir `end`. Los errores de red/HTTP se
    reintentan con backoff exponencial (ver `_http_get_with_retry`).
    """
    if interval not in _BINANCE_INTERVAL_MS:
        raise ValueError(f"Intervalo '{interval}' no soportado por _load_binance: {list(_BINANCE_INTERVAL_MS)}")

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    interval_ms = _BINANCE_INTERVAL_MS[interval]

    all_rows: list[list] = []
    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": BINANCE_MAX_CANDLES,
        }
        response = _http_get_with_retry(BINANCE_KLINES_URL, params)
        rows = response.json()
        if not rows:
            break
        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        next_start = last_open_time + interval_ms
        if next_start <= current_start:
            # Salvaguarda ante una respuesta que no avanza (evita loop infinito).
            break
        current_start = next_start
        if len(rows) < BINANCE_MAX_CANDLES:
            break

    if not all_rows:
        raise ValueError(f"Binance no devolvió velas para '{symbol}' en [{start}, {end}]")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df[STANDARD_COLUMNS].astype(float)


def _load_coinmetrics(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Descarga histórico diario de CoinMetrics Community (CSV público en GitHub).

    Esta fuente solo reporta precio de cierre: open/high/low se rellenan con
    ese close y volume queda en NaN porque CoinMetrics community no lo
    reporta. Es ideal como fallback y para verificar el pipeline offline
    porque cubre el histórico diario completo del activo.

    La columna de precio se llama "PriceUSD" para la mayoría de los activos,
    pero algunos (p. ej. SOL) solo publican "ReferenceRateUSD"; se usa esa
    como alternativa cuando "PriceUSD" no está presente.
    """
    url = COINMETRICS_CSV_URL.format(symbol=symbol)
    response = _http_get_with_retry(url, params={})
    df = pd.read_csv(StringIO(response.text))

    if "time" not in df.columns:
        raise ValueError(f"CSV de CoinMetrics para '{symbol}' no tiene la columna 'time' esperada")
    price_col = "PriceUSD" if "PriceUSD" in df.columns else "ReferenceRateUSD"
    if price_col not in df.columns:
        raise ValueError(
            f"CSV de CoinMetrics para '{symbol}' no tiene columnas de precio esperadas (PriceUSD/ReferenceRateUSD)"
        )

    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("timestamp").sort_index()
    close = df[price_col].astype(float).dropna()

    out = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": np.nan},
        index=close.index,
    )
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return out.loc[(out.index >= start_ts) & (out.index <= end_ts)]


def _load_coingecko(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Descarga precios desde el endpoint público `market_chart/range` de CoinGecko.

    LIMITACIÓN conocida del plan gratuito de CoinGecko: el histórico
    disponible vía API pública está acotado a aproximadamente los últimos 365
    días. Pedir rangos más antiguos puede devolver datos vacíos o parciales;
    para histórico profundo, usar CoinMetrics. Además, CoinGecko viene
    exigiendo cada vez más una API key incluso para este endpoint "público"
    (responde 401 sin ella); si eso ocurre, esta fuente fallará y
    `get_prices` seguirá probando el resto de FALLBACK_SOURCES.

    Esta fuente tampoco reporta OHLC real: open/high/low se rellenan con el
    close ("prices"). El volume sí está disponible ("total_volumes") y se usa
    tal cual lo reporta la API (volumen agregado, no por vela).
    """
    url = COINGECKO_RANGE_URL.format(symbol=symbol)
    params = {
        "vs_currency": "usd",
        "from": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "to": int(pd.Timestamp(end, tz="UTC").timestamp()),
    }
    response = _http_get_with_retry(url, params)
    payload = response.json()

    prices = payload.get("prices", [])
    if not prices:
        raise ValueError(f"CoinGecko no devolvió precios para '{symbol}' en [{start}, {end}]")

    price_df = pd.DataFrame(prices, columns=["ts_ms", "close"])
    price_df["timestamp"] = pd.to_datetime(price_df["ts_ms"], unit="ms", utc=True)
    price_df = price_df.set_index("timestamp")[["close"]]

    volumes = payload.get("total_volumes", [])
    if volumes:
        vol_df = pd.DataFrame(volumes, columns=["ts_ms", "volume"])
        vol_df["timestamp"] = pd.to_datetime(vol_df["ts_ms"], unit="ms", utc=True)
        vol_df = vol_df.set_index("timestamp")[["volume"]]
        price_df = price_df.join(vol_df, how="left")
    else:
        price_df["volume"] = np.nan

    close = price_df["close"].astype(float)
    out = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": price_df["volume"].astype(float)},
        index=close.index,
    )
    return out.sort_index()


def _load_store(asset: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Lee un snapshot local ya exportado por `scripts/export_snapshot.py`
    (`config.SNAPSHOT_DIR/{asset}_{interval}.parquet`).

    Pensado para verificar el pipeline offline con un dataset ya compartido,
    sin depender de la disponibilidad de Binance en el entorno. Si el
    archivo no existe, lanza un error claro indicando cómo generarlo.
    """
    path = SNAPSHOT_DIR / f"{asset}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el snapshot local '{path}'. Corré 'python scripts/export_snapshot.py' "
            "para generarlo antes de usar source='store'."
        )

    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def _load_csv(path: str | Path) -> pd.DataFrame:
    """Carga un CSV local ya estandarizado.

    Se espera una primera columna de fecha/hora (parseable por pandas) como
    índice, y columnas ["open", "high", "low", "close", "volume"]. Si el
    índice viene sin timezone, se asume UTC.
    """
    path = Path(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df[STANDARD_COLUMNS].astype(float)


# --------------------------------------------------------------------------
# HTTP con reintentos
# --------------------------------------------------------------------------


def _http_get_with_retry(url: str, params: dict) -> requests.Response:
    """GET con reintentos y backoff exponencial ante errores de red o 429/5xx."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            wait_seconds = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Intento %d/%d falló para %s: %s. Reintentando en %.1fs",
                attempt, MAX_RETRIES, url, exc, wait_seconds,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait_seconds)

    raise ConnectionError(f"No se pudo completar el request a '{url}' tras {MAX_RETRIES} intentos") from last_exc


# --------------------------------------------------------------------------
# Estandarización y validaciones
# --------------------------------------------------------------------------


def _standardize(df: pd.DataFrame, *, interval: str, source: str) -> pd.DataFrame:
    """Normaliza un DataFrame crudo al contrato de salida de `get_prices`.

    - Fuerza índice DatetimeIndex tz-aware en UTC.
    - Elimina duplicados de índice (conserva la última observación) y ordena.
    - Verifica que estén las columnas exactas ["open","high","low","close","volume"]
      y las castea a float.
    - Loguea (sin lanzar excepción) si detecta gaps respecto a la frecuencia
      esperada del intervalo.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"El índice devuelto por la fuente '{source}' no es DatetimeIndex")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    n_before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    if len(df) < n_before:
        logger.warning("Fuente '%s': se eliminaron %d filas duplicadas por índice", source, n_before - len(df))

    missing_cols = set(STANDARD_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas {missing_cols} en los datos de '{source}'")
    df = df[STANDARD_COLUMNS].astype(float)

    _log_gaps(df, interval=interval, source=source)
    return df


def _log_gaps(df: pd.DataFrame, *, interval: str, source: str) -> None:
    freq = _INTERVAL_TO_PANDAS_FREQ.get(interval)
    if freq is None or df.empty:
        return
    expected_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    missing = expected_index.difference(df.index)
    if len(missing) > 0:
        logger.warning(
            "Fuente '%s' (%s, %d filas): %d timestamps faltantes en [%s, %s]",
            source, interval, len(df), len(missing), df.index.min(), df.index.max(),
        )


# --------------------------------------------------------------------------
# Caché en parquet
# --------------------------------------------------------------------------


def _cache_path(asset: str, source: str, interval: str) -> Path:
    return CACHE_DIR / f"{asset}_{source}_{interval}.parquet"


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Caché escrita en '%s' (%d filas)", path, len(df))


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    logger.info("Caché leída desde '%s' (%d filas)", path, len(df))
    return df


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
