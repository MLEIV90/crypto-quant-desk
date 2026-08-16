"""Esquemas Pydantic de respuesta de la API (Fase 8a).

Solo describen la FORMA de lo que ya calcula el backend (tipado,
autodocumentado vía /docs) — ningún modelo de acá hace ningún cálculo.
Convención: fechas siempre `datetime` tz-aware UTC (FastAPI las serializa a
ISO 8601 con offset, p. ej. "2026-08-12T00:00:00+00:00"); los valores
faltantes (NaN de pandas) se convierten a `None` ANTES de construir estos
modelos (ver `api/main.py::_none_if_nan`/`_series_to_list`), nunca se les
pasa un `float('nan')` directo — un NaN de Python serializado a JSON da
`NaN` literal, que no es JSON válido; `None` sí da `null`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AssetsResponse(BaseModel):
    """Respuesta de `GET /api/assets`."""

    activos: list[str] = Field(description="Tickers disponibles (config.UNIVERSE)")
    timeframes: list[str] = Field(description="Intervalos soportados por source='store'")


class Candle(BaseModel):
    """Una vela OHLCV."""

    fecha: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class OHLCVResponse(BaseModel):
    """Respuesta de `GET /api/ohlcv`."""

    asset: str
    interval: str
    velas: list[Candle]


class StudiesResponse(BaseModel):
    """Respuesta de `GET /api/studies`: series alineadas a las mismas
    fechas (`fechas`, mismo orden e índice que cada lista) más los niveles
    puntuales (Fibonacci/soporte-resistencia/pivotes) de la última vela.
    """

    asset: str
    interval: str
    fechas: list[datetime]
    sma_20: list[float | None]
    sma_50: list[float | None]
    ema_12: list[float | None]
    ema_26: list[float | None]
    bb_upper: list[float | None]
    bb_mid: list[float | None]
    bb_lower: list[float | None]
    rsi_14: list[float | None]
    macd: list[float | None]
    macd_signal: list[float | None]
    macd_hist: list[float | None]
    stoch_k: list[float | None]
    stoch_d: list[float | None]
    fibonacci: dict[str, float] | None = Field(
        default=None, description="Retrocesos/extensiones de Fibonacci — SIN respaldo predictivo probado, ver signals/studies.py"
    )
    soporte_resistencia: dict = Field(description='{"resistencia": [...], "soporte": [...], "precio_actual": ...}')
    pivotes: dict[str, float] = Field(description="Pivotes clásicos: P, R1/R2/R3, S1/S2/S3")


class DesempenoHistorico(BaseModel):
    """Desempeño histórico REAL de la regla de consenso del sugeridor —
    ver `signals/suggester.py`: viaja SIEMPRE junto con la sugerencia, la
    API no la omite ni la esconde.
    """

    cagr_sugeridor: float
    sharpe_sugeridor: float
    max_drawdown_sugeridor: float
    n_trades_sugeridor: int
    cagr_buy_and_hold: float
    sharpe_buy_and_hold: float
    max_drawdown_buy_and_hold: float


class SuggesterResponse(BaseModel):
    """Respuesta de `GET /api/suggester` — tal cual devuelve
    `signals.suggester.suggest()`, con su honestidad intacta (ver
    `DesempenoHistorico`).
    """

    sugerencia: str = Field(description="COMPRAR / VENDER / ESPERAR")
    votos_alcistas: int
    votos_bajistas: int
    votos_neutrales: int
    confianza: float = Field(description="|alcistas - bajistas| / total_estudios, en [0, 1]")
    detalle: dict[str, str] = Field(description="Voto de cada estudio: alcista / bajista / neutral")
    desempeno_historico: DesempenoHistorico


class RiskResponse(BaseModel):
    """Respuesta de `GET /api/risk` — mismas magnitudes que calcula
    `app.workers.AnalysisWorker` para la pestaña "Riesgo" del cockpit,
    siempre en diario (el modelo GARCH del proyecto es diario).
    """

    asset: str
    vol_realizada: float = Field(description="Volatilidad realizada anualizada (rolling)")
    modelo_garch: str = Field(description='Modelo GARCH ganador, formato "vol/dist" (p. ej. "EGarch/t")')
    vol_garch: float = Field(description="Volatilidad condicional GARCH anualizada, última vela")
    regimen: str | None = Field(description='"calma" / "normal" / "tension", o null si no hay suficiente historia')
    var95: float = Field(description="Value at Risk histórico al 95% (pérdida positiva)")
    es95: float = Field(description="Expected Shortfall histórico al 95% (pérdida positiva)")
    accion: str = Field(description="LONG / FLAT / SHORT (signals.engine)")
    score: float
    tamano_sugerido: float = Field(description="Tamaño de posición por vol targeting, en [-1, 1]")
    ultima_fecha: datetime


class EquityPoint(BaseModel):
    """Un punto de una curva de equity (base 1.0)."""

    fecha: datetime
    valor: float


class BacktestResponse(BaseModel):
    """Respuesta de `GET /api/backtest`: métricas + curvas de equity de la
    estrategia del engine vs. buy & hold, ambas con costos de transacción
    (ver `backtest/engine.py`).
    """

    asset: str
    metrics_estrategia: dict[str, float]
    metrics_buy_and_hold: dict[str, float]
    equity_curve_estrategia: list[EquityPoint]
    equity_curve_buy_and_hold: list[EquityPoint]


class GarchSeriesResponse(BaseModel):
    """Respuesta de `GET /api/garch-series`: serie temporal completa de
    volatilidad condicional GARCH, para graficarla (p. ej. el panel de
    riesgo del futuro frontend).
    """

    asset: str
    fechas: list[datetime]
    vol_condicional: list[float | None]
    modelo_garch: str
    regimen_actual: str | None
