"""Configuración centralizada de crypto-quant-desk.

Convenciones globales del proyecto:
    - Todos los retornos se expresan en escala DECIMAL (0.01 == 1%), nunca en
      porcentaje ni en basis points, salvo que el nombre de la variable lo
      indique explícitamente (p. ej. TRANSACTION_COST_BPS).
    - La frecuencia de trabajo por defecto es DIARIA, por lo que
      PERIODS_PER_YEAR se usa para anualizar métricas. A diferencia de las
      acciones (252 días hábiles), el mercado cripto opera los 365 días del
      año sin feriados ni fines de semana, así que PERIODS_PER_YEAR = 365.
    - Todas las series temporales de precios usan índice datetime en UTC
      (tz-aware). No se asume timezone local en ningún punto del pipeline.
    - Los paths se resuelven siempre vía pathlib a partir de BASE_DIR, nunca
      hardcodeados, para que el proyecto sea reproducible en cualquier máquina.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
# Snapshot local exportado por scripts/export_snapshot.py (fuente "store" de
# data/loaders.py). Se comparte manualmente entre máquinas, no se versiona.
SNAPSHOT_DIR: Path = DATA_DIR / "snapshot"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Universo de activos
# --------------------------------------------------------------------------
# Cada activo mapea a los símbolos/identificadores usados por cada fuente de
# datos soportada. Esto permite que data/loaders.py traduzca un ticker interno
# (p. ej. "BTC") al identificador específico que espera cada API.

UNIVERSE: dict[str, dict[str, str]] = {
    "BTC": {"binance": "BTCUSDT", "coinmetrics": "btc", "coingecko": "bitcoin"},
    "ETH": {"binance": "ETHUSDT", "coinmetrics": "eth", "coingecko": "ethereum"},
    "SOL": {"binance": "SOLUSDT", "coinmetrics": "sol", "coingecko": "solana"},
    "BNB": {"binance": "BNBUSDT", "coinmetrics": "bnb", "coingecko": "binancecoin"},
    "LTC": {"binance": "LTCUSDT", "coinmetrics": "ltc", "coingecko": "litecoin"},
}

# --------------------------------------------------------------------------
# Parámetros de retornos y trading
# --------------------------------------------------------------------------

# 365, no 252: cripto opera todos los días del año, no solo días hábiles
# bursátiles (esa es la convención de acciones, no la de este proyecto).
PERIODS_PER_YEAR: int = 365
DEFAULT_INTERVAL: str = "1d"
RAW_START_DATE: str = "2017-01-01"

# Costo de transacción round-trip (ida + vuelta) en basis points. 1 bp = 0.01%.
TRANSACTION_COST_BPS: float = 10.0

# Convención usada para medir turnover de una estrategia: "one_way" cuenta
# solo el lado de compra (o venta) de cada rebalanceo, no ambos.
TURNOVER_CONVENTION: str = "one_way"

# --------------------------------------------------------------------------
# Fuentes de datos
# --------------------------------------------------------------------------

DEFAULT_SOURCE: str = "binance"
FALLBACK_SOURCES: list[str] = ["coinmetrics", "coingecko"]

# --------------------------------------------------------------------------
# Motor de señales / decisión (ver signals/engine.py)
# --------------------------------------------------------------------------

# Pesos de cada componente en el score compuesto. DEBEN sumar 1 (si se pasan
# pesos custom a composite_score() que no sumen 1, se loguea un warning).
SIGNAL_WEIGHTS: dict[str, float] = {"trend": 0.5, "momentum": 0.25, "mean_reversion": 0.25}

# Zona muerta del score compuesto (en [-1,1]): por encima de
# SCORE_THRESHOLD_LONG => LONG, por debajo de SCORE_THRESHOLD_SHORT => SHORT,
# en el medio => FLAT (posición 0, señal demasiado débil/ambigua para operar).
SCORE_THRESHOLD_LONG: float = 0.2
SCORE_THRESHOLD_SHORT: float = -0.2

# Vol targeting para el sizing de la posición: TARGET_VOL es la volatilidad
# ANUAL objetivo del book (alta, 50%, acorde a la volatilidad típica de
# cripto); MAX_LEVERAGE acota el apalancamiento máximo permitido (1.0 = sin
# apalancamiento, la posición nunca excede el 100% del capital).
TARGET_VOL: float = 0.50
MAX_LEVERAGE: float = 1.0

# Si es False, las señales negativas se recortan a 0 (solo largo/flat, sin
# cortos) — útil si el usuario no puede o no quiere shortear.
ALLOW_SHORT: bool = True

# Ventana (en observaciones) de la volatilidad realizada usada para el
# sizing por vol targeting en el backtest (rolling std de retornos).
VOL_WINDOW: int = 30
