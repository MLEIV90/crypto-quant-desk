"""Actualización incremental de snapshots OHLCV locales (Fase 9a).

`update_snapshot()` es el complemento "en caliente" de
`scripts/export_snapshot.py` (que arma el snapshot completo desde
`config.RAW_START_DATE`, de una sola vez, pensado para correrse a mano de
vez en cuando): en vez de rehacer todo el histórico, lee el parquet YA
existente en `config.SNAPSHOT_DIR`, detecta la última fecha guardada, y
baja de Binance SOLO las velas nuevas desde ahí hasta la última vela YA
CERRADA — nunca la vela en formación, que todavía puede cambiar de precio.

Reutiliza `data.loaders._load_binance` (la paginación de klines ya
existente) y `data.quality.validate_ohlcv` — no reimplementa ninguna de las
dos cosas. A propósito NO pasa por `data.loaders.get_prices`: esa función
además de Binance encadena CoinMetrics/CoinGecko como fallback y usa una
caché separada (`data/cache/`) que no aplica acá — para un refresco
incremental de un snapshot ya construido con datos reales de Binance,
mezclar silenciosamente una fuente de solo-cierre (OHLC degenerado) sería
peor que fallar con un error claro, que es lo que hace esta función
(`GeoblockedError` con mensaje explícito ante 451/403).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from config import SNAPSHOT_DIR, UNIVERSE
from data import loaders
from data.quality import validate_ohlcv

logger = logging.getLogger(__name__)

# Códigos HTTP típicos del geobloqueo de Binance en ciertas regiones (451 =
# Unavailable For Legal Reasons, 403 = Forbidden). `scripts/export_snapshot.py`
# importa este mismo valor y `is_geoblocked_error` de acá en vez de duplicarlos.
GEOBLOCK_STATUS_CODES = (451, 403)

# Intervalos para los que este proyecto efectivamente mantiene snapshots
# locales (ver scripts/export_snapshot.py::DEFAULT_START_BY_INTERVAL y
# api/main.py::SUPPORTED_INTERVALS). Binance soporta más intervalos (1m/5m/
# 15m/4h, ver data.loaders._BINANCE_INTERVAL_MS), pero esta función es solo
# para refrescar snapshots que ya existen en uno de estos dos.
SUPPORTED_INTERVALS: tuple[str, ...] = ("1d", "1h")


class GeoblockedError(RuntimeError):
    """Binance devolvió 451/403 (geobloqueo regional) al pedir velas nuevas."""


def is_geoblocked_error(exc: BaseException) -> bool:
    """Recorre la cadena de excepciones (`__cause__`) buscando un status HTTP
    451 o 403 — la señal típica de geobloqueo de Binance en ciertas regiones.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(getattr(current, "response", None), "status_code", None)
        if status_code in GEOBLOCK_STATUS_CODES:
            return True
        current = current.__cause__
    return False


def _snapshot_path(asset: str, interval: str) -> Path:
    return SNAPSHOT_DIR / f"{asset}_{interval}.parquet"


def last_closed_candle_open_time(interval: str) -> pd.Timestamp:
    """Open-time (UTC) de la última vela YA CERRADA de `interval`, a partir de "ahora".

    La vela EN FORMACIÓN en este momento abrió en `floor(ahora, interval)` y
    todavía no cerró (cierra recién en `floor(ahora, interval) + interval`).
    La última vela CERRADA es la anterior a esa.
    """
    freq = loaders._INTERVAL_TO_PANDAS_FREQ[interval]
    now = pd.Timestamp.now(tz="UTC")
    current_candle_open = now.floor(freq)
    interval_ms = loaders._BINANCE_INTERVAL_MS[interval]
    return current_candle_open - pd.Timedelta(milliseconds=interval_ms)


def update_snapshot(asset: str, interval: str) -> dict:
    """Actualiza en caliente el snapshot local de `asset`/`interval` contra Binance.

    Requiere que el snapshot ya exista (`FileNotFoundError` con el mismo
    mensaje que `data.loaders._load_store` si no fue exportado todavía, ver
    `scripts/export_snapshot.py`) — esta función es solo para refrescos
    incrementales, no para el volcado inicial.

    Pasos: lee el parquet existente, detecta `last_saved` (última fecha
    guardada) y `last_closed` (última vela ya cerrada en este momento). Si
    `last_saved >= last_closed`, no hay nada nuevo que bajar. Si no, pide a
    Binance las velas en `[last_saved, last_closed]` (el propio
    `last_saved` se re-pide a propósito, por si esa vela estaba incompleta
    cuando se guardó; el dedup de abajo la reemplaza sin duplicarla),
    concatena, deduplica (conserva la versión nueva), valida con
    `data.quality.validate_ohlcv` y reescribe el parquet.

    Devuelve {"filas_agregadas": int, "ultima_fecha": pd.Timestamp,
    "ya_actualizado": bool}.

    Lanza `GeoblockedError` si Binance devuelve 451/403. Cualquier otra
    falla de red/HTTP se propaga tal cual (ver `data.loaders._http_get_with_retry`).
    """
    if asset not in UNIVERSE:
        raise ValueError(f"Activo '{asset}' no está definido en config.UNIVERSE")
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Intervalo '{interval}' no soportado: {list(SUPPORTED_INTERVALS)}")

    path = _snapshot_path(asset, interval)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el snapshot local '{path}'. Corré 'python scripts/export_snapshot.py "
            f"--interval {interval}' para generar el volcado inicial antes de actualizar."
        )

    existing = pd.read_parquet(path)
    if existing.empty:
        raise ValueError(
            f"El snapshot '{path}' existe pero está vacío. Regeneralo con "
            f"'python scripts/export_snapshot.py --interval {interval}'."
        )
    if existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")
    else:
        existing.index = existing.index.tz_convert("UTC")
    existing = existing.sort_index()

    last_saved = existing.index.max()
    last_closed = last_closed_candle_open_time(interval)

    if last_saved >= last_closed:
        logger.info("update_snapshot(%s, %s): ya está al día (última vela %s)", asset, interval, last_saved)
        return {"filas_agregadas": 0, "ultima_fecha": last_saved, "ya_actualizado": True}

    symbol = UNIVERSE[asset]["binance"]
    logger.info(
        "update_snapshot(%s, %s): bajando velas nuevas de Binance desde %s hasta %s",
        asset, interval, last_saved, last_closed,
    )
    try:
        raw = loaders._load_binance(symbol, interval, last_saved.isoformat(), last_closed.isoformat())
    # F-03 (auditoría, Fase 14): acotado a lo que realmente puede propagar
    # `_load_binance` — `requests.RequestException` (un fallo HTTP directo,
    # p. ej. el 451/403 de geobloqueo antes de agotar reintentos),
    # `ConnectionError` (agotó los reintentos de `loaders._http_get_with_retry`,
    # con el `RequestException` original encadenado en `__cause__`, que es
    # justo lo que recorre `is_geoblocked_error`) o `ValueError` (sin velas /
    # intervalo inválido).
    except (requests.RequestException, ConnectionError, ValueError) as exc:
        if is_geoblocked_error(exc):
            raise GeoblockedError(
                f"Binance bloqueó la descarga de '{asset}' por geolocalización (HTTP 451/403) — "
                "tu región probablemente no tiene acceso a api.binance.com."
            ) from exc
        raise

    new_data = loaders._standardize(raw, interval=interval, source="binance")
    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    report = validate_ohlcv(combined, activo=asset)
    logger.info("update_snapshot(%s, %s): %s", asset, interval, report.resumen())

    combined.to_parquet(path)
    filas_agregadas = len(combined) - len(existing)
    logger.info(
        "update_snapshot(%s, %s): +%d filas, última fecha %s", asset, interval, filas_agregadas, combined.index.max()
    )

    return {
        "filas_agregadas": int(filas_agregadas),
        "ultima_fecha": combined.index.max(),
        "ya_actualizado": False,
    }
