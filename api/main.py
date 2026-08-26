"""API REST (FastAPI) de crypto-quant-desk (Fase 8a) — ver `api/__init__.py`.

Cada endpoint es una capa FINA: llama a funciones ya existentes del
backend y serializa el resultado con los esquemas de `api/models.py`. No
reimplementa NINGÚN cálculo — ver el comentario de cada endpoint para qué
módulo/función reutiliza.

RENDIMIENTO — léase antes de integrar un frontend: `/api/risk` y
`/api/garch-series` ajustan un modelo GARCH (grid search sobre varias
especificaciones, `models.garch.select_best_model`) en cada request —
varios segundos en activos con mucha historia. `/api/prediction` es el
endpoint MÁS LENTO de todos: entrena/evalúa el modelo primario de ML con
validación purgeada (varios ajustes de XGBoost) en cada request, del orden
de 15-30 segundos — un frontend NUNCA debería dispararlo automáticamente
al cargar una vista, solo ante una acción explícita del usuario (un botón
"correr predicción"). `/api/report` (Fase 16b) es LENTO por una razón
distinta: ajusta un GARCH Y corre cointegración rolling sobre todos los
pares (30-90 segundos), también pensado para un botón explícito, nunca
automático — ver `reports/pdf_report.py`. `/api/backtest` corre un
backtest vectorizado completo (rápido). El resto (`/api/assets`,
`/api/ohlcv`, `/api/studies`, `/api/suggester`, `/api/data-status`,
`/api/stats`, `/api/compare`) son livianos (indicadores/estadística
vectorizados, sin ajuste de modelos).
Ninguno cachea entre requests — cada llamada recalcula desde cero (simple y
correcto; una capa de caché queda para una fase futura si hiciera falta).

RED — `/api/refresh` (Fase 9a) es el ÚNICO endpoint de toda la API que
toca la red (Binance): todos los demás solo leen `data/snapshot/*.parquet`
(estático hasta que se llama a `/api/refresh`). Puede tardar varios
segundos; pensado para dispararse desde un botón explícito ("Actualizar
datos"), nunca automáticamente.

Uso (desarrollo):
    uvicorn api.main:app --reload
    # Swagger interactivo en http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from analysis.comparison import align_common_dates, compare_assets
from analysis.cycles import drawdown_analysis, halving_cycles, market_phases, monthly_yearly_heatmap
from analysis.statistics import (
    BTC_HALVING_DATES,
    autocorrelation,
    hourly_seasonality,
    monthly_seasonality,
    weekday_seasonality,
)
from api.models import (
    AdfResult,
    AssetsResponse,
    AutocorrelationPoint,
    BacktestResponse,
    Candle,
    CompareResponse,
    CorrelationResponse,
    DataStatusResponse,
    DrawdownEpisode,
    EquityPoint,
    GarchSeriesResponse,
    HalvingCycle,
    HalvingCyclesInfo,
    MarketPhase,
    MonthlyYearlyHeatmap,
    OHLCVResponse,
    PairBacktestMetrics,
    PairBacktestResult,
    PairDetailResponse,
    PairScreeningResponse,
    PairScreeningRow,
    PairStabilitySummary,
    BacktestStrategiesResponse,
    BacktestStrategyInfo,
    PairZscoreExtreme,
    PredictionResponse,
    RefreshResponse,
    ResearchExperimentsResponse,
    ReturnHistogram,
    RiskPercentiles,
    RiskResponse,
    RiskSummaryResponse,
    RiskSummaryRow,
    RlResearchResult,
    RotationResearchResult,
    SeasonalityBucket,
    StatsResponse,
    StudiesResponse,
    SuggesterResponse,
    VolumeProfileResponse,
)
from backtest.engine import backtest_from_prices, compare_to_buy_and_hold
from config import BASE_DIR, PERIODS_PER_YEAR, TARGET_VOL, TRANSACTION_COST_BPS, UNIVERSE, VOL_WINDOW
from data.loaders import RESAMPLED_INTERVALS, get_prices
from data.snapshot import GeoblockedError, last_closed_candle_open_time, update_snapshot
from eda.eda_report import adf_test, correlation_matrix
from metrics.risk_measures import (
    drawdown_series,
    expected_shortfall,
    historical_percentile,
    rolling_expected_shortfall,
    rolling_value_at_risk,
    value_at_risk,
)
from ml.features import align_features_labels, build_feature_matrix
from ml.labeling import get_daily_volatility, triple_barrier_labels
from ml.models import evaluate_primary_with_roc_auc, feature_importances, fit_final, latest_oos_prediction, t1_from_labels
from models.garch import conditional_volatility, select_best_model, volatility_regime
from pairs.backtest import DEFAULT_PAIR_ENTRY, DEFAULT_PAIR_EXIT, DEFAULT_PAIR_STOP, backtest_pair
from pairs.cointegration import engle_granger, half_life
from pairs.signals import zscore
from pairs.stability import rolling_cointegration, screen_pairs_stability, stability_summary
from reports.pdf_report import build_report
from signals.engine import (
    generate_positions,
    generate_positions_engine_signal,
    generate_positions_vol_targeting,
    latest_recommendation,
)
from signals.indicators import add_all_indicators
from signals.returns import log_returns, simple_returns
from signals.studies import all_studies, ichimoku, stochastic, volume_profile
from signals.suggester import suggest

logger = logging.getLogger(__name__)

# Intervalos que acepta la API de LECTURA. "1d"/"1h" son los que el
# snapshot local tiene exportados de verdad (`source="store"`, ver
# scripts/export_snapshot.py y Fase 6b); "4h"/"1w" (Fase 17b) son
# DERIVADOS por resampleo on-the-fly desde esos dos (ver
# `data.loaders.RESAMPLED_INTERVALS`/`_load_store`) — no hace falta bajar
# datos nuevos ni exportar un snapshot propio para ellos. Todos los
# endpoints de LECTURA solo leen del snapshot local — el único que golpea
# Binance en vivo es `POST /api/refresh` (Fase 9a, ver
# data.snapshot.update_snapshot), que a propósito NO acepta "4h"/"1w"
# directamente (ver `refresh_snapshot`): no hay nada propio que refrescar
# en un intervalo derivado, se refresca su base.
SUPPORTED_INTERVALS: tuple[str, ...] = ("1d", "1h", "4h", "1w")
DEFAULT_RISK_INTERVAL = "1d"  # el modelo GARCH del proyecto es diario, ver models/garch.py

# Rótulos de base para /api/risk y /api/risk-summary (Fase 20c, coherencia
# del panel de riesgo) — texto legible que documenta, EN LA RESPUESTA
# misma, contra qué ventana/método se compara cada número, para que el
# frontend nunca tenga que adivinar ni hardcodear una explicación que
# podría desincronizarse del cálculo real. Ver el docstring de
# `RiskResponse`/`RiskSummaryRow` en `api/models.py` para el porqué.
RISK_ROLLING_WINDOW_LABEL = "último año (365 días)"
RISK_HISTORICO_BASIS_LABEL = "toda la serie histórica"
RISK_REGIMEN_GARCH_BASIS_LABEL = f"{RISK_ROLLING_WINDOW_LABEL}, volatilidad condicional GARCH"
RISK_SUMMARY_REGIMEN_BASIS_LABEL = f"{RISK_ROLLING_WINDOW_LABEL}, volatilidad realizada (no GARCH)"
# Fase 25: UN solo rótulo para el VaR/ES "actual", compartido a propósito
# entre `RiskResponse.actual_basis` (GET /api/risk) y
# `RiskSummaryRow.var95_basis` (GET /api/risk-summary) — antes de esta fase
# eran dos constantes de texto DISTINTAS porque los dos endpoints usaban
# métodos DISTINTOS (GARCH paramétrico vs. empírico rolling) para la misma
# idea de "VaR actual", y un mismo activo podía mostrar dos números
# distintos sin que el usuario supiera por qué. Ahora ambos usan
# `rolling_value_at_risk`/`rolling_expected_shortfall` (empírico, sin
# GARCH) y por lo tanto la MISMA etiqueta de método.
RISK_ACTUAL_VAR_BASIS_LABEL = f"{RISK_ROLLING_WINDOW_LABEL}, empírico (ventana móvil, no GARCH)"

# Catálogo de estrategias de /api/backtest (Fase 21) — única fuente de
# verdad para el selector del frontend: describe en criollo qué hace cada
# una, para qué sirve y cuál es su trade-off explícito, evitando que la
# pestaña "Backtest" muestre una curva sin explicar qué la generó. "combo"
# es la estrategia HISTÓRICA de este endpoint (dirección del engine +
# tamaño por vol targeting, vía `signals.engine.generate_positions`), que
# sigue siendo el default de `GET /api/backtest` para no romper el resumen
# de `RiskView` (sección "El valor de gestionar el riesgo", que la consume
# sin pasar `strategy` desde antes de esta fase) — en la pestaña Backtest
# rediseñada se ofrece como una cuarta opción, rotulada como lo que
# realmente es, junto a las 3 pedidas explícitamente (vol targeting /
# engine solo / buy & hold).
DEFAULT_BACKTEST_STRATEGY = "combo"
_TARGET_VOL_MIN = 0.10
_TARGET_VOL_MAX = 1.50
_BACKTEST_STRATEGY_CATALOG: dict[str, dict[str, str | bool]] = {
    "vol_targeting": {
        "nombre": "Vol targeting",
        "descripcion": (
            "Ajusta el tamaño de la posición según la volatilidad realizada reciente: invierte "
            "menos cuando el mercado está más agitado de lo normal, buscando mantener el riesgo "
            "estable. Siempre está LARGO (nunca corto) y no intenta adivinar la dirección del precio."
        ),
        "objetivo": "Mantener una exposición al riesgo pareja en el tiempo, no maximizar el retorno.",
        "tradeoff": (
            "Fase 23: en TEORÍA, reducir posición cuando sube la volatilidad tiende a costar algo de "
            "retorno en mercados alcistas fuertes y sostenidos (es el trade-off esperado del mecanismo, "
            "reduce posición justo cuando el mercado sube con volatilidad). En la PRÁCTICA, para este "
            "activo y este período, el efecto fue el contrario — la tabla de abajo es la verdad de este "
            "caso concreto, no esta descripción general del mecanismo: mirala antes de sacar conclusiones."
        ),
        "tiene_target_vol": True,
    },
    "engine": {
        "nombre": "Señal del engine de consenso",
        "descripcion": (
            "Combina tendencia (EMA/MACD), momentum (RSI) y reversión a la media (Bollinger) en un "
            "score compuesto que INTENTA anticipar la dirección del precio (largo, corto o afuera). "
            "El tamaño de la posición sigue la convicción del score, sin ajustar por volatilidad."
        ),
        "objetivo": (
            "Intentar anticipar la dirección del precio combinando varios indicadores técnicos — el "
            "mismo enfoque de base que el sugeridor de la pestaña Análisis Técnico, no un modelo nuevo."
        ),
        "tradeoff": (
            "Fase 23: el proyecto ya mostró que anticipar la dirección de forma consistente es difícil "
            "— el sugeridor de Análisis Técnico está etiquetado 'sin edge' porque no le gana con "
            "claridad a buy & hold. Esta estrategia parte de la misma lógica de base: no asumas que sí "
            "tiene ventaja solo por combinar más indicadores, y además no reduce el tamaño cuando el "
            "mercado se pone más volátil. Mirá los números de abajo para este caso concreto."
        ),
        "tiene_target_vol": False,
    },
    "combo": {
        "nombre": "Engine + vol targeting (combinado)",
        "descripcion": (
            "La misma señal direccional del engine de consenso, pero con el tamaño de la posición "
            "ajustado por vol targeting en vez de fijo por la convicción del score."
        ),
        "objetivo": "Sumar la dirección del engine con el sizing de vol targeting en una sola posición.",
        "tradeoff": (
            "Hereda los errores de dirección del engine Y paga el costo de operar menos en "
            "momentos volátiles — combina los dos trade-offs, no los evita."
        ),
        "tiene_target_vol": True,
    },
    "buy_and_hold": {
        "nombre": "Buy & hold puro (control)",
        "descripcion": "Siempre 100% largo, sin señal ni ajuste de tamaño de ningún tipo.",
        "objetivo": "Servir de referencia: si una estrategia no le gana a esto en lo que dice buscar, no aporta valor.",
        "tradeoff": "Ninguno propio — es la vara de medir, no una estrategia con compromiso a evaluar.",
        "tiene_target_vol": False,
    },
}

# Tope de `limit` en /api/ohlcv y /api/studies (Fase 11): por encima del
# histórico horario real más largo (~58.000 velas desde 2020, ver
# scripts/export_snapshot.py::DEFAULT_START_BY_INTERVAL), para que "Todo"
# pueda pedir el histórico COMPLETO sin que el tope lo recorte primero.
MAX_CANDLE_LIMIT: int = 60_000

# Activos con cobertura on-chain suficiente para el modelo enriquecido de
# /api/prediction (mismo criterio que `app.workers.ONCHAIN_ENABLED_ASSETS`
# — ver ese módulo para el detalle de por qué BTC/ETH sí y el resto no).
ONCHAIN_ENABLED_ASSETS: frozenset[str] = frozenset({"BTC", "ETH"})

# Umbral de antigüedad a partir del cual /api/data-status marca el snapshot
# como desactualizado (Fase 9a): una vela diaria recién cerrada tiene hasta
# 1 día de "antigüedad normal"; una horaria, hasta 2h (algo de margen sobre
# el intervalo mismo, para no marcar en ámbar el instante justo después de
# que cierra la última vela). "4h"/"1w" (Fase 17b) siguen el mismo criterio
# que su intervalo BASE respectivo: 2x la duración de la vela para "4h"
# (como "1h"), 1x para "1w" (como "1d") — ver `get_data_status` para cómo
# se calcula la frescura real de un intervalo derivado.
STALE_THRESHOLD_SECONDS: dict[str, float] = {
    "1d": 24 * 3600,
    "1h": 2 * 3600,
    "4h": 8 * 3600,
    "1w": 7 * 24 * 3600,
}

_PREDICTION_N_SPLITS = 5
_PREDICTION_EMBARGO_PCT = 0.01
_TOP_N_FEATURES = 8
_CLASS_DISPLAY: dict[float, str] = {-1.0: "SHORT", 0.0: "FLAT", 1.0: "LONG"}

app = FastAPI(
    title="crypto-quant-desk API",
    description=(
        "Capa REST fina sobre el backend de crypto-quant-desk (precios, estudios técnicos, "
        "sugeridor de consenso, riesgo GARCH/VaR/ES, backtest) para un futuro frontend web. "
        "No reimplementa cálculos: cada endpoint llama a los módulos ya existentes del backend."
    ),
    version="0.1.0",
)

# CORS habilitado para desarrollo: el frontend (p. ej. React con Vite en
# otro puerto) necesita poder consumir esta API desde otro origen. Abierto
# a cualquier origen a propósito (fase de desarrollo, sin autenticación
# todavía) — restringir a un origen concreto queda para cuando haya un
# despliegue real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------
# Helpers internos (validación + conversión NaN -> None)
# --------------------------------------------------------------------------


def _validate_asset(asset: str) -> None:
    if asset not in UNIVERSE:
        raise HTTPException(
            status_code=404,
            detail=f"Activo '{asset}' no está definido en config.UNIVERSE. Válidos: {sorted(UNIVERSE)}",
        )


def _validate_interval(interval: str) -> None:
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(
            status_code=400,
            detail=f"interval debe ser uno de {list(SUPPORTED_INTERVALS)}, recibido '{interval}'",
        )


def _load_df(asset: str, interval: str) -> pd.DataFrame:
    """Precios estandarizados desde el snapshot local (`data.loaders.get_prices`,
    `source="store"`) — el único punto de entrada a datos de toda la API.

    `use_cache=False` a propósito (Fase 9a): `get_prices` cachearía el
    resultado en `data/cache/{asset}_store_{interval}.parquet`, una copia
    SEPARADA del propio `data/snapshot/{asset}_{interval}.parquet` — sin
    esto, tras un `POST /api/refresh` que reescribe el snapshot, esta
    función seguiría devolviendo la copia vieja de `data/cache/` para
    siempre (el snapshot en sí ya es un archivo local: cachear una copia de
    una lectura de archivo local no ahorra nada y solo agrega una fuente de
    datos desactualizados).

    `end` explícito a "mañana" (Fase 9a), también a propósito: el default de
    `get_prices` cuando no se pasa `end` es la fecha de HOY sin hora
    ("YYYY-MM-DD" -> medianoche UTC), así que con `interval="1h"` recortaría
    silenciosamente cualquier vela horaria de HOY posterior a la
    medianoche — justo las velas que un refresco recién bajado agregaría.
    """
    _validate_asset(asset)
    _validate_interval(interval)
    tomorrow = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        return get_prices(asset, source="store", interval=interval, end=tomorrow, use_cache=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _csv_response(df: pd.DataFrame, filename: str, *, index: bool = False) -> Response:
    """Serializa `df` a CSV (`pandas.DataFrame.to_csv`, sin reimplementar
    ningún formateo propio) y lo envuelve en una respuesta descargable —
    mismo patrón de `Content-Disposition: attachment` que ya usa
    `GET /api/report` (Fase 16b) para el PDF, acá con `text/csv`.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=index)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _none_if_nan(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def _series_to_list(series: pd.Series) -> list[float | None]:
    return [_none_if_nan(v) for v in series.to_numpy(dtype=float)]


def _dates_to_list(index: pd.DatetimeIndex) -> list:
    return list(index.to_pydatetime())


def _regime_series_to_list(series: pd.Series) -> list[str | None]:
    """Como `_series_to_list`, pero para la serie de régimen (Fase 20a) —
    valores STRING ("calma"/"normal"/"tension"), no numéricos: NaN (warmup,
    ver `models.garch.volatility_regime`) se convierte a None, el resto a
    `str` tal cual.
    """
    return [None if pd.isna(v) else str(v) for v in series.to_numpy(dtype=object)]


def _drawdown_to_model(d: dict) -> DrawdownEpisode:
    return DrawdownEpisode(
        fecha_pico=d["fecha_pico"].to_pydatetime(),
        fecha_fondo=d["fecha_fondo"].to_pydatetime(),
        profundidad_pct=d["profundidad_pct"],
        fecha_recuperacion=d["fecha_recuperacion"].to_pydatetime() if d["fecha_recuperacion"] is not None else None,
        dias_caida=d["dias_caida"],
        dias_recuperacion=d["dias_recuperacion"],
    )


def _phase_to_model(p: dict) -> MarketPhase:
    return MarketPhase(
        tipo=p["tipo"],
        fecha_inicio=p["fecha_inicio"].to_pydatetime(),
        fecha_fin=p["fecha_fin"].to_pydatetime(),
        duracion_dias=p["duracion_dias"],
        retorno_pct=p["retorno_pct"],
        confirmada=p["confirmada"],
    )


def _halving_cycle_to_model(c: dict) -> HalvingCycle:
    return HalvingCycle(
        fecha_inicio=c["fecha_inicio"].to_pydatetime(),
        fecha_fin=c["fecha_fin"].to_pydatetime(),
        en_curso=c["en_curso"],
        duracion_dias=c["duracion_dias"],
        retorno_pct=c["retorno_pct"],
        drawdown_maximo_pct=c["drawdown_maximo_pct"],
    )


def _seasonality_to_buckets(df: pd.DataFrame) -> list[SeasonalityBucket]:
    """Convierte el DataFrame de `analysis.statistics.monthly_seasonality`/
    `weekday_seasonality`/`hourly_seasonality` a `list[SeasonalityBucket]`
    (Fase 11, simplificado en Fase 26: las tres funciones calculan
    "mediana"/"desvio" desde Fase 26 — antes solo `monthly_seasonality` lo
    hacía, y esta función tenía un parámetro `has_median_std` para
    distinguirlas).
    """
    buckets = []
    for bucket, row in df.iterrows():
        buckets.append(
            SeasonalityBucket(
                bucket=int(bucket),
                retorno_medio=float(row["retorno_medio"]),
                mediana=_none_if_nan(row["mediana"]),
                desvio=_none_if_nan(row["desvio"]),
                n=int(row["n"]),
            )
        )
    return buckets


def _none_if_nonfinite(value: float) -> float | None:
    """Como `_none_if_nan` pero también convierte +-inf a None (Fase 12b:
    `pairs.cointegration.half_life` devuelve `float('inf')` cuando el
    spread no revierte — JSON no tiene un literal válido para infinito).
    """
    return None if pd.isna(value) or value in (float("inf"), float("-inf")) else float(value)


_ZSCORE_EXTREME_THRESHOLD = 2.0


def _interpret_zscore(z: float | None) -> str:
    """Texto honesto sobre qué tan 'estirado' está el spread hoy (Fase 12b)."""
    if z is None or pd.isna(z):
        return "Sin suficiente historia todavía para calcular el z-score."
    if abs(z) > _ZSCORE_EXTREME_THRESHOLD:
        direccion = "por encima" if z > 0 else "por debajo"
        return f"Zona EXTREMA: el spread está muy {direccion} de su media histórica (z={z:.2f})."
    return f"Zona normal: el spread está dentro de su rango habitual (z={z:.2f})."


def _humanize_age(delta: pd.Timedelta) -> str:
    """Antigüedad legible en español, para /api/data-status ("hace 5 días", "hace 3 h")."""
    seconds = delta.total_seconds()
    if seconds < 60:
        return "hace instantes"
    minutes = seconds / 60
    if minutes < 60:
        return f"hace {int(minutes)} min"
    hours = minutes / 60
    if hours < 24:
        return f"hace {int(hours)} h"
    days = int(hours / 24)
    return f"hace {days} día" + ("" if days == 1 else "s")


# --------------------------------------------------------------------------
# GET /api/assets
# --------------------------------------------------------------------------


@router.get("/assets", response_model=AssetsResponse, summary="Activos y timeframes disponibles")
def get_assets() -> AssetsResponse:
    """Reutiliza `config.UNIVERSE` tal cual — no consulta ninguna fuente de datos."""
    return AssetsResponse(activos=list(UNIVERSE.keys()), timeframes=list(SUPPORTED_INTERVALS))


# --------------------------------------------------------------------------
# GET /api/ohlcv
# --------------------------------------------------------------------------


@router.get("/ohlcv", response_model=OHLCVResponse, summary="Velas OHLCV")
def get_ohlcv(
    asset: str,
    interval: str = "1d",
    limit: int = Query(
        500, gt=0, le=MAX_CANDLE_LIMIT, description="Cantidad de velas más recientes a devolver"
    ),
) -> OHLCVResponse:
    """Reutiliza `data.loaders.get_prices` tal cual; solo recorta a las
    últimas `limit` velas y serializa a JSON.

    `le=MAX_CANDLE_LIMIT` (Fase 11): antes tope en 5000 rechazaba el pedido
    de "Todo" el histórico horario (~58.000 velas) — ahora el tope alcanza
    para pedir la historia COMPLETA de cualquier activo/intervalo (el
    frontend, `PeriodSelector.tsx`, ya no necesita acotar "Todo" a un
    número arbitrario menor al histórico real).
    """
    df = _load_df(asset, interval)
    recent = df.tail(limit)

    velas = [
        Candle(
            fecha=fecha.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=_none_if_nan(row["volume"]),
        )
        for fecha, row in recent.iterrows()
    ]
    return OHLCVResponse(asset=asset, interval=interval, velas=velas)


# --------------------------------------------------------------------------
# GET /api/studies
# --------------------------------------------------------------------------


@router.get("/studies", response_model=StudiesResponse, summary="Estudios técnicos (series + niveles)")
def get_studies(
    asset: str,
    interval: str = "1d",
    limit: int = Query(
        500, gt=0, le=MAX_CANDLE_LIMIT, description="Cantidad de velas más recientes a devolver"
    ),
) -> StudiesResponse:
    """Reutiliza `signals.indicators.add_all_indicators` (incluye vwap/obv,
    Fase 10b), `signals.studies.stochastic`, `signals.studies.ichimoku`
    (Fase 10b) y `signals.studies.all_studies` (Fase 7a) tal cual — mismos
    cálculos que alimentan la pestaña "Análisis Técnico" del cockpit
    (`app.workers.StudiesWorker`), acá expuestos como series completas (no
    solo la última vela) para que el frontend las grafique/toggee.
    """
    df = _load_df(asset, interval)

    indicators_df = add_all_indicators(df)
    stoch_df = stochastic(df["high"], df["low"], df["close"])
    ichimoku_df = ichimoku(df["high"], df["low"], df["close"])
    resumen = all_studies(df)

    recent = df.tail(limit)
    recent_indicators = indicators_df.loc[recent.index]
    recent_stoch = stoch_df.loc[recent.index]
    recent_ichimoku = ichimoku_df.loc[recent.index]

    return StudiesResponse(
        asset=asset,
        interval=interval,
        fechas=_dates_to_list(recent.index),
        sma_20=_series_to_list(recent_indicators["sma_20"]),
        sma_50=_series_to_list(recent_indicators["sma_50"]),
        ema_12=_series_to_list(recent_indicators["ema_12"]),
        ema_26=_series_to_list(recent_indicators["ema_26"]),
        bb_upper=_series_to_list(recent_indicators["bb_upper"]),
        bb_mid=_series_to_list(recent_indicators["bb_mid"]),
        bb_lower=_series_to_list(recent_indicators["bb_lower"]),
        rsi_14=_series_to_list(recent_indicators["rsi_14"]),
        macd=_series_to_list(recent_indicators["macd"]),
        macd_signal=_series_to_list(recent_indicators["macd_signal"]),
        macd_hist=_series_to_list(recent_indicators["macd_hist"]),
        stoch_k=_series_to_list(recent_stoch["stoch_k"]),
        stoch_d=_series_to_list(recent_stoch["stoch_d"]),
        vwap=_series_to_list(recent_indicators["vwap"]),
        obv=_series_to_list(recent_indicators["obv"]),
        ichimoku_tenkan=_series_to_list(recent_ichimoku["tenkan"]),
        ichimoku_kijun=_series_to_list(recent_ichimoku["kijun"]),
        ichimoku_senkou_a=_series_to_list(recent_ichimoku["senkou_a"]),
        ichimoku_senkou_b=_series_to_list(recent_ichimoku["senkou_b"]),
        ichimoku_chikou=_series_to_list(recent_ichimoku["chikou"]),
        fibonacci=resumen["fibonacci"],
        soporte_resistencia=resumen["soporte_resistencia"],
        pivotes=resumen["pivotes"],
    )


# --------------------------------------------------------------------------
# GET /api/volume-profile
# --------------------------------------------------------------------------


@router.get(
    "/volume-profile",
    response_model=VolumeProfileResponse,
    summary="Volume Profile: volumen operado por nivel de precio (POC + Value Area)",
)
def get_volume_profile(
    asset: str,
    interval: str = "1d",
    limit: int = Query(
        365, gt=0, le=MAX_CANDLE_LIMIT, description="Cantidad de velas más recientes a incluir en el perfil"
    ),
    bins: int = Query(50, ge=10, le=200, description="Cantidad de niveles de precio en los que se divide el rango"),
) -> VolumeProfileResponse:
    """Reutiliza `signals.studies.volume_profile` tal cual — calcula el
    perfil sobre las últimas `limit` velas de `_load_df(asset, interval)`
    (el mismo recorte de "período" que ya usan `/api/ohlcv`/`/api/studies`,
    ver `PeriodSelector` del frontend): cambiar el período recalcula el
    perfil sobre ese rango, no sobre TODO el histórico disponible.
    """
    df = _load_df(asset, interval).tail(limit)
    profile = volume_profile(df, bins=bins)

    return VolumeProfileResponse(
        asset=asset,
        interval=interval,
        niveles_precio=profile["niveles_precio"],
        volumenes=profile["volumenes"],
        poc=profile["poc"],
        value_area_low=profile["value_area_low"],
        value_area_high=profile["value_area_high"],
        volumen_total=profile["volumen_total"],
    )


# --------------------------------------------------------------------------
# GET /api/suggester
# --------------------------------------------------------------------------


@router.get("/suggester", response_model=SuggesterResponse, summary="Sugeridor de consenso")
def get_suggester(asset: str, interval: str = "1d") -> SuggesterResponse:
    """Reutiliza `signals.suggester.suggest` tal cual — devuelve exactamente
    su dict (sugerencia, votos, confianza, desempeño histórico), sin
    ocultar ni recortar el campo `desempeno_historico` (ver `signals/suggester.py`:
    esa regla nunca viaja sin su propio historial de desempeño al lado).
    """
    df = _load_df(asset, interval)
    resultado = suggest(df)
    return SuggesterResponse(**resultado)


# --------------------------------------------------------------------------
# GET /api/risk
# --------------------------------------------------------------------------


@router.get("/risk", response_model=RiskResponse, summary="Panel de riesgo (GARCH/VaR/ES/sizing)")
def get_risk(asset: str) -> RiskResponse:
    """Reutiliza `models.garch.select_best_model`/`conditional_volatility`/
    `volatility_regime`, `metrics.risk_measures.value_at_risk`/
    `expected_shortfall`/`rolling_value_at_risk`/`rolling_expected_shortfall`/
    `historical_percentile` y `signals.engine.latest_recommendation` — la
    misma secuencia de llamadas que arma `app.workers.AnalysisWorker` para
    la pestaña "Riesgo" del cockpit, reescrita acá para no depender de la
    app de escritorio (PySide6) desde la API. LENTO: ajusta un modelo GARCH
    por request (ver docstring del módulo) — el GARCH sigue siendo
    necesario para `vol_garch`/`regimen`, aunque ya no para el VaR/ES
    "actual" (ver más abajo).

    COHERENCIA (Fase 20c, actualizado en Fase 25): ver el docstring de
    `RiskResponse` en `api/models.py` para por qué esta respuesta trae DOS
    VaR/ES distintos (`var95`/`es95` histórico vs. `var95_actual`/`es95_actual`
    de HOY) y por qué los 4 percentiles de `percentiles` comparten la MISMA
    ventana que `regimen` (último año) en vez de comparar contra toda la
    historia. Fase 25: `var95_actual`/`es95_actual` pasaron de ser el
    VaR/ES paramétrico implícito por GARCH a ser EMPÍRICO en ventana móvil
    (`rolling_value_at_risk`/`rolling_expected_shortfall`) — el MISMO
    método que ya usaba `GET /api/risk-summary`, para que el VaR "actual"
    de un activo sea el MISMO número en la tabla de 5 monedas y en esta
    vista de detalle.
    """
    df = _load_df(asset, DEFAULT_RISK_INTERVAL)
    close = df["close"]

    returns = simple_returns(close).dropna()
    garch_returns = log_returns(close).dropna()

    best = select_best_model(garch_returns, criterion="aic")
    garch_result = best["result"]
    cond_vol = conditional_volatility(garch_result)
    regime_series = volatility_regime(cond_vol)
    last_regime = regime_series.iloc[-1]

    # Histórico (toda la serie) — referencia de largo plazo, NO refleja el
    # régimen actual (ver docstring de RiskResponse).
    var95 = value_at_risk(returns, level=0.95)
    es95 = expected_shortfall(returns, level=0.95)

    # "Actual" (Fase 20c, unificado en Fase 25): EMPÍRICO en ventana móvil
    # de 1 año — el MISMO método que ya usaba GET /api/risk-summary para su
    # VaR "actual" (antes de Fase 25 acá se usaba el VaR/ES paramétrico
    # implícito por GARCH, un método DISTINTO que daba un número distinto
    # para el mismo activo/momento). Esto es lo que SÍ se mueve con el
    # régimen, y lo que el frontend muestra como titular.
    rolling_var_series = rolling_value_at_risk(returns, window=PERIODS_PER_YEAR, level=0.95)
    rolling_es_series = rolling_expected_shortfall(returns, window=PERIODS_PER_YEAR, level=0.95)
    var95_actual = float(rolling_var_series.iloc[-1])
    es95_actual = float(rolling_es_series.iloc[-1])

    recomendacion = latest_recommendation(df, garch_regime=False)

    # Fase 20a/20c: percentil de cada métrica "actual" vs su propia ventana
    # RECIENTE (último año, MISMA base que `regimen` — antes de Fase 20c
    # comparaban contra TODA la historia, una base distinta de la de
    # `regimen` que podía verse contradictoria, p. ej. "TENSIÓN" junto a un
    # percentil bajo). Vol realizada/GARCH/VaR/ES-actual ya son series
    # temporales — alcanza con recortar a los últimos PERIODS_PER_YEAR
    # valores antes de ubicar el último dato en esa ventana.
    realized_vol_series = (
        returns.rolling(window=VOL_WINDOW).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    )  # misma fórmula que signals.engine.generate_positions/latest_recommendation

    percentiles = RiskPercentiles(
        vol_realizada=_none_if_nan(historical_percentile(realized_vol_series.tail(PERIODS_PER_YEAR))),
        vol_garch=_none_if_nan(historical_percentile(cond_vol.tail(PERIODS_PER_YEAR))),
        var95=_none_if_nan(historical_percentile(rolling_var_series.tail(PERIODS_PER_YEAR))),
        es95=_none_if_nan(historical_percentile(rolling_es_series.tail(PERIODS_PER_YEAR))),
        base=RISK_ROLLING_WINDOW_LABEL,
    )

    # Fase 20a: distribución de retornos diarios + marcas de VaR/ES sobre
    # esa distribución, para el histograma del frontend (bins vía
    # numpy.histogram, no un cálculo propio del proyecto). Las marcas usan
    # el VaR/ES HISTÓRICO (no el actual): son el percentil/cola real de
    # ESTA MISMA distribución empírica que se está graficando, no una
    # proyección paramétrica de otro momento.
    counts, edges = np.histogram(returns.to_numpy(), bins=40)
    histograma = ReturnHistogram(
        bin_edges=[float(edge) for edge in edges],
        counts=[int(count) for count in counts],
        var95_return=float(-var95),
        es95_return=float(-es95),
    )

    return RiskResponse(
        asset=asset,
        vol_realizada=float(recomendacion["vol_realizada"]),
        modelo_garch=f"{best['vol']}/{best['dist']}",
        vol_garch=float(cond_vol.iloc[-1]),
        regimen=str(last_regime) if pd.notna(last_regime) else None,
        regimen_basis=RISK_REGIMEN_GARCH_BASIS_LABEL,
        var95=float(var95),
        es95=float(es95),
        var95_actual=var95_actual,
        es95_actual=es95_actual,
        historico_basis=RISK_HISTORICO_BASIS_LABEL,
        actual_basis=RISK_ACTUAL_VAR_BASIS_LABEL,
        accion=str(recomendacion["accion"]),
        score=float(recomendacion["score"]),
        tamano_sugerido=float(recomendacion["tamaño_sugerido"]),
        ultima_fecha=close.index[-1].to_pydatetime(),
        percentiles=percentiles,
        histograma=histograma,
    )


# --------------------------------------------------------------------------
# GET /api/risk-summary
# --------------------------------------------------------------------------


@router.get(
    "/risk-summary",
    response_model=RiskSummaryResponse,
    summary="Comparación de riesgo actual entre las 5 monedas (rápido, sin GARCH)",
)
def get_risk_summary() -> RiskSummaryResponse:
    """Reutiliza `signals.returns.simple_returns`, `metrics.risk_measures.value_at_risk`/
    `historical_percentile`/`rolling_value_at_risk` y `models.garch.volatility_regime`
    (Fase 20b, coherencia en Fase 20c) — la misma fórmula de vol realizada
    que `GET /api/risk` (`returns.rolling(VOL_WINDOW).std(ddof=1) * sqrt(PERIODS_PER_YEAR)`)
    y la MISMA función de régimen, pero aplicada a esa serie de vol
    REALIZADA en vez de a la volatilidad condicional GARCH.

    DECISIÓN DE RENDIMIENTO (léase antes de tocar esto): `GET /api/risk`
    ajusta un modelo GARCH (`models.garch.select_best_model`, un grid
    search de 6 especificaciones) — varios segundos por activo. Este
    endpoint evalúa las 5 monedas de `config.UNIVERSE` de una sola vez,
    pensado para cargar automáticamente al entrar a la vista "Riesgo" (sin
    un botón explícito de por medio, a diferencia de `/api/prediction`) —
    ajustar 5 modelos GARCH secuenciales haría esa carga demasiado lenta
    para una tabla comparativa que se supone se lee "de un vistazo". En vez
    de eso, se usa volatilidad REALIZADA (rápida: un rolling std) para
    clasificar el régimen — `regimen_basis` lo rotula explícitamente para
    que no se confunda con el régimen GARCH de la vista detallada de un
    activo (dos métodos DISTINTOS a propósito, por rendimiento). No
    devuelve `vol_garch` por el mismo motivo.

    COHERENCIA (Fase 20c, unificado con `GET /api/risk` en Fase 25): `var95`
    acá YA NO es el histórico de toda la serie (como quedó `GET /api/risk`'s
    `var95`) — es un VaR "actual" recalculado en ventana móvil
    (`rolling_value_at_risk`), el MISMO método (y la MISMA `var95_basis`)
    que `GET /api/risk`'s `var95_actual` desde Fase 25: antes de esa fase
    eran dos métodos DISTINTOS (acá empírico rolling, allá paramétrico
    GARCH) que daban un número distinto para el mismo activo/momento sin
    que el usuario supiera por qué — a diferencia del régimen (que sigue
    siendo, a propósito, dos métodos distintos por rendimiento), el VaR
    "actual" no tenía ninguna razón para ser distinto entre las dos vistas.
    El percentil de cada fila compara contra su propia ventana de
    RISK_ROLLING_WINDOW_LABEL, la misma base que usa `regimen` — ver el
    docstring de `RiskSummaryRow`.
    """
    filas: list[RiskSummaryRow] = []
    for asset in UNIVERSE:
        df = _load_df(asset, DEFAULT_RISK_INTERVAL)
        close = df["close"]
        returns = simple_returns(close).dropna()

        realized_vol_series = returns.rolling(window=VOL_WINDOW).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
        regime_series = volatility_regime(realized_vol_series)
        last_regime = regime_series.iloc[-1]

        rolling_var = rolling_value_at_risk(returns, window=PERIODS_PER_YEAR, level=0.95)
        var95_actual = float(rolling_var.iloc[-1])

        filas.append(
            RiskSummaryRow(
                asset=asset,
                vol_realizada=float(realized_vol_series.iloc[-1]),
                vol_realizada_percentil=_none_if_nan(historical_percentile(realized_vol_series.tail(PERIODS_PER_YEAR))),
                regimen=str(last_regime) if pd.notna(last_regime) else None,
                regimen_basis=RISK_SUMMARY_REGIMEN_BASIS_LABEL,
                var95=var95_actual,
                var95_percentil=_none_if_nan(historical_percentile(rolling_var.tail(PERIODS_PER_YEAR))),
                var95_basis=RISK_ACTUAL_VAR_BASIS_LABEL,
                ultima_fecha=close.index[-1].to_pydatetime(),
            )
        )

    return RiskSummaryResponse(filas=filas)


# --------------------------------------------------------------------------
# GET /api/backtest
# --------------------------------------------------------------------------


def _positions_for_strategy(df: pd.DataFrame, strategy: str, target_vol: float | None) -> pd.Series:
    """Despacha a la función de posiciones correcta según `strategy` (ver
    `_BACKTEST_STRATEGY_CATALOG`) — reutiliza tal cual las piezas ya
    existentes en `signals.engine`, no reimplementa ninguna.
    """
    if strategy == "vol_targeting":
        return generate_positions_vol_targeting(df, target_vol=target_vol)
    if strategy == "engine":
        return generate_positions_engine_signal(df)
    if strategy == "combo":
        return generate_positions(df, target_vol=target_vol)
    if strategy == "buy_and_hold":
        return pd.Series(1.0, index=df.index)
    raise ValueError(f"estrategia desconocida: {strategy}")  # pragma: no cover — ya validada antes de llegar acá


def _equity_points(series: pd.Series) -> list[EquityPoint]:
    return [EquityPoint(fecha=fecha.to_pydatetime(), valor=float(valor)) for fecha, valor in series.items()]


@router.get(
    "/backtest",
    response_model=BacktestResponse,
    summary="Backtest configurable: estrategia elegida vs. buy & hold",
)
def get_backtest(
    asset: str,
    strategy: str = DEFAULT_BACKTEST_STRATEGY,
    cost_bps: float | None = None,
    target_vol: float | None = None,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
) -> BacktestResponse:
    """Reutiliza `signals.engine.generate_positions`/`generate_positions_vol_targeting`/
    `generate_positions_engine_signal`, `backtest.engine.backtest_from_prices`
    y `compare_to_buy_and_hold` — no reimplementa nada del motor.

    `strategy` (Fase 21, default "combo" para no romper el resumen de
    `RiskView`, ver `_BACKTEST_STRATEGY_CATALOG` para el detalle de cada
    una): "vol_targeting" | "engine" | "combo" | "buy_and_hold".
    `target_vol` solo aplica a las estrategias con `tiene_target_vol=True`
    (se ignora en las demás). `fecha_inicio`/`fecha_fin` (ISO "YYYY-MM-DD",
    opcionales) recortan la VENTANA del backtest DESPUÉS de calcular las
    posiciones sobre el histórico completo — así el recorte no genera un
    período de "warmup" artificial al principio de la ventana elegida.
    """
    if strategy not in _BACKTEST_STRATEGY_CATALOG:
        raise HTTPException(
            status_code=400,
            detail=f"strategy debe ser una de {sorted(_BACKTEST_STRATEGY_CATALOG)}, recibido '{strategy}'",
        )
    if cost_bps is None:
        cost_bps = TRANSACTION_COST_BPS

    df_full = _load_df(asset, DEFAULT_RISK_INTERVAL)
    positions_full = _positions_for_strategy(df_full, strategy, target_vol)
    close_full = df_full["close"]

    if fecha_inicio is not None or fecha_fin is not None:
        try:
            close = close_full.loc[fecha_inicio:fecha_fin]
            positions = positions_full.loc[fecha_inicio:fecha_fin]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"fecha_inicio/fecha_fin inválidas: {exc}") from exc
        if close.empty:
            raise HTTPException(status_code=400, detail="No hay datos en el rango de fechas pedido")
    else:
        close = close_full
        positions = positions_full

    result_estrategia = backtest_from_prices(close, positions, cost_bps=cost_bps)
    asset_returns = simple_returns(close)
    comparacion = compare_to_buy_and_hold(asset_returns, result_estrategia, cost_bps=cost_bps)
    result_buy_and_hold = backtest_from_prices(close, 1.0, cost_bps=cost_bps)

    return BacktestResponse(
        asset=asset,
        strategy=strategy,
        cost_bps=cost_bps,
        fecha_inicio=pd.Timestamp(fecha_inicio).to_pydatetime() if fecha_inicio else None,
        fecha_fin=pd.Timestamp(fecha_fin).to_pydatetime() if fecha_fin else None,
        metrics_estrategia=comparacion["estrategia"],
        metrics_buy_and_hold=comparacion["buy_and_hold"],
        equity_curve_estrategia=_equity_points(result_estrategia.equity_curve),
        equity_curve_buy_and_hold=_equity_points(result_buy_and_hold.equity_curve),
        drawdown_curve_estrategia=_equity_points(drawdown_series(result_estrategia.returns)),
        drawdown_curve_buy_and_hold=_equity_points(drawdown_series(result_buy_and_hold.returns)),
        exposure_curve_estrategia=_equity_points(result_estrategia.positions),
    )


@router.get(
    "/backtest-strategies",
    response_model=BacktestStrategiesResponse,
    summary="Catálogo de estrategias backtesteables (para el selector del frontend)",
)
def get_backtest_strategies() -> BacktestStrategiesResponse:
    """Expone `_BACKTEST_STRATEGY_CATALOG` tal cual: la MISMA fuente que usa
    `get_backtest` para despachar, para que el selector/las descripciones/los
    límites de `target_vol` en el frontend nunca puedan desincronizarse de
    lo que el backend realmente calcula.
    """
    estrategias = [
        BacktestStrategyInfo(
            id=strategy_id,
            nombre=str(info["nombre"]),
            descripcion=str(info["descripcion"]),
            objetivo=str(info["objetivo"]),
            tradeoff=str(info["tradeoff"]),
            tiene_target_vol=bool(info["tiene_target_vol"]),
            target_vol_default=TARGET_VOL if info["tiene_target_vol"] else None,
            target_vol_min=_TARGET_VOL_MIN if info["tiene_target_vol"] else None,
            target_vol_max=_TARGET_VOL_MAX if info["tiene_target_vol"] else None,
        )
        for strategy_id, info in _BACKTEST_STRATEGY_CATALOG.items()
    ]
    return BacktestStrategiesResponse(estrategias=estrategias, cost_bps_default=TRANSACTION_COST_BPS)


# --------------------------------------------------------------------------
# GET /api/garch-series
# --------------------------------------------------------------------------


@router.get("/garch-series", response_model=GarchSeriesResponse, summary="Serie de volatilidad condicional GARCH")
def get_garch_series(asset: str) -> GarchSeriesResponse:
    """Reutiliza `models.garch.select_best_model`/`conditional_volatility`/
    `volatility_regime` tal cual — la serie completa (no solo la última
    vela, a diferencia de `/api/risk`) para graficar la volatilidad
    condicional en el tiempo. LENTO: ajusta un modelo GARCH por request
    (ver docstring del módulo).
    """
    df = _load_df(asset, DEFAULT_RISK_INTERVAL)
    close = df["close"]
    garch_returns = log_returns(close).dropna()

    best = select_best_model(garch_returns, criterion="aic")
    cond_vol = conditional_volatility(best["result"])
    regime_series = volatility_regime(cond_vol)
    last_regime = regime_series.iloc[-1]

    return GarchSeriesResponse(
        asset=asset,
        fechas=_dates_to_list(cond_vol.index),
        vol_condicional=_series_to_list(cond_vol),
        modelo_garch=f"{best['vol']}/{best['dist']}",
        regimen_actual=str(last_regime) if pd.notna(last_regime) else None,
        regimen_serie=_regime_series_to_list(regime_series),
    )


# --------------------------------------------------------------------------
# GET /api/prediction
# --------------------------------------------------------------------------


@router.get(
    "/prediction",
    response_model=PredictionResponse,
    summary="Predicción OOS del modelo primario de ML (investigación, sin edge demostrado)",
)
def get_prediction(asset: str) -> PredictionResponse:
    """Reutiliza `ml.features`/`ml.labeling`/`ml.models` tal cual — la misma
    secuencia que `app.workers.PredictionWorker` para la pestaña
    "Research (sin edge)" del cockpit, reescrita acá para no depender de la
    app de escritorio (PySide6) desde la API.

    MUY LENTO (ver docstring del módulo): entrena/evalúa XGBoost con
    validación purgeada en cada request. Este endpoint es investigación con
    resultado negativo — el modelo NUNCA superó de forma consistente a los
    baselines triviales en ninguna validación del proyecto — no una
    recomendación de operar.
    """
    df = _load_df(asset, DEFAULT_RISK_INTERVAL)
    close = df["close"]
    use_onchain = asset in ONCHAIN_ENABLED_ASSETS

    X_raw = build_feature_matrix(df, include_onchain=use_onchain, asset=asset if use_onchain else None)
    onchain_columns: list[str] = []
    if use_onchain:
        X_technical_raw = build_feature_matrix(df, include_onchain=False)
        onchain_columns = [c for c in X_raw.columns if c not in X_technical_raw.columns]

    volatility = get_daily_volatility(close)
    labels_df = triple_barrier_labels(close, volatility)
    X, y, sample_weight = align_features_labels(X_raw, labels_df)
    t1 = t1_from_labels(labels_df, X.index)

    evaluacion = evaluate_primary_with_roc_auc(
        X, y, t1, sample_weight=sample_weight,
        n_splits=_PREDICTION_N_SPLITS, embargo_pct=_PREDICTION_EMBARGO_PCT,
    )
    prediccion = latest_oos_prediction(
        X, y, t1, sample_weight=sample_weight,
        n_splits=_PREDICTION_N_SPLITS, embargo_pct=_PREDICTION_EMBARGO_PCT,
    )

    model = fit_final(X, y, sample_weight=sample_weight)
    importances = feature_importances(model, X.columns)
    top_features = [(str(name), float(val)) for name, val in importances.head(_TOP_N_FEATURES).items()]

    proba_por_texto = {_CLASS_DISPLAY[k]: v for k, v in prediccion["proba"].items()}

    return PredictionResponse(
        asset=asset,
        used_onchain=use_onchain,
        onchain_columns=onchain_columns,
        ultima_fecha=prediccion["fecha"].to_pydatetime(),
        prediccion_clase=_CLASS_DISPLAY[prediccion["clase"]],
        prediccion_confianza=prediccion["confianza"],
        prediccion_proba=proba_por_texto,
        accuracy_media=evaluacion["accuracy_media"],
        baseline_azar=evaluacion["baseline_azar"],
        baseline_mayoritaria=evaluacion["baseline_mayoritaria"],
        supera_azar=evaluacion["supera_azar"],
        supera_mayoritaria=evaluacion["supera_mayoritaria"],
        roc_auc_media=evaluacion["roc_auc_media"],
        top_features=top_features,
    )


# --------------------------------------------------------------------------
# GET /api/research-experiments (Fase 24)
# --------------------------------------------------------------------------

# Directorios donde `scripts/run_rl_experiment.py`/`scripts/run_rotation_experiment.py`
# guardan sus corridas (Fases 18/19a) — este endpoint SOLO LEE lo que ya está
# ahí, nunca corre un experimento nuevo (son lentos: RL tarda ~15 minutos,
# ver `rl/results/*.json::elapsed_seconds`). Rutas vía `config.BASE_DIR`,
# nunca relativas al directorio de trabajo del proceso (misma convención que
# `config.DATA_DIR`).
RL_RESULTS_DIR: Path = BASE_DIR / "rl" / "results"
ROTATION_RESULTS_DIR: Path = BASE_DIR / "strategies" / "results"

# Timestamp embebido en el nombre de archivo por los scripts de experimento
# (p. ej. "rl_experiment_20260822_182720.json") — se usa para (a) elegir el
# archivo más reciente por orden lexicográfico/cronológico y (b) exponer
# "fecha_experimento" en la respuesta sin tener que guardarla aparte.
_EXPERIMENT_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def _latest_result_json(directory: Path) -> Path | None:
    """El archivo `*.json` más reciente de `directory` (orden lexicográfico
    == cronológico, por el timestamp `YYYYMMDD_HHMMSS` en el nombre), o
    `None` si el directorio no existe o está vacío — el experimento
    correspondiente nunca se corrió.
    """
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*.json"))
    return candidates[-1] if candidates else None


def _experiment_timestamp_from_filename(path: Path) -> datetime:
    match = _EXPERIMENT_TIMESTAMP_RE.search(path.stem)
    if match is None:
        # No debería pasar con los nombres que generan los scripts de
        # experimento, pero no es motivo para que el endpoint falle.
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def _nan_to_none(value):
    """Los JSON de resultados guardados pueden tener `NaN` literal (Python
    los escribe así por defecto; no es JSON estándar) en columnas que no
    aplican a un par/fila dado — p. ej. `sharpe_buy_hold_SOL` en una fila
    del par BTC-ETH (ver `strategies/results/*.json`). Los convierte a
    `None` recursivamente ANTES de validar con Pydantic/serializar a JSON de
    verdad, para no devolver un `NaN` no estándar en la respuesta HTTP.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


@router.get(
    "/research-experiments",
    response_model=ResearchExperimentsResponse,
    summary="Resultados guardados de los experimentos de Deep RL y rotación por momentum",
)
def get_research_experiments() -> ResearchExperimentsResponse:
    """Lee (NO recalcula) el resultado más reciente de `rl/results/` y de
    `strategies/results/` — experimentos lentos de investigación (Fases
    18/19a) que se corren aparte con `scripts/run_rl_experiment.py`/
    `scripts/run_rotation_experiment.py`, nunca desde un request HTTP.

    Si un experimento nunca se corrió (directorio inexistente o vacío), el
    campo correspondiente (`rl`/`rotation`) viene en `None` — degrada con
    gracia, no lanza 404 ni 500: la pestaña Research de un checkout nuevo
    sin experimentos guardados todavía es un estado válido, no un error.
    """
    rl_result: RlResearchResult | None = None
    rl_path = _latest_result_json(RL_RESULTS_DIR)
    if rl_path is not None:
        with open(rl_path, encoding="utf-8") as f:
            raw = _nan_to_none(json.load(f))
        rl_result = RlResearchResult(fecha_experimento=_experiment_timestamp_from_filename(rl_path), **raw)

    rotation_result: RotationResearchResult | None = None
    rotation_path = _latest_result_json(ROTATION_RESULTS_DIR)
    if rotation_path is not None:
        with open(rotation_path, encoding="utf-8") as f:
            raw = _nan_to_none(json.load(f))
        rotation_result = RotationResearchResult(
            fecha_experimento=_experiment_timestamp_from_filename(rotation_path), **raw
        )

    return ResearchExperimentsResponse(rl=rl_result, rotation=rotation_result)


# --------------------------------------------------------------------------
# GET /api/data-status
# --------------------------------------------------------------------------


@router.get("/data-status", response_model=DataStatusResponse, summary="Antigüedad del snapshot local")
def get_data_status(asset: str, interval: str = "1d") -> DataStatusResponse:
    """Solo LEE el snapshot local (`_load_df`, sin red) y compara su última
    fecha contra "ahora" (UTC) — pensado para que el frontend muestre
    SIEMPRE de qué fecha son los datos que está mirando, sin tener que
    disparar `/api/refresh` (que sí toca la red) solo para chequear.

    `desactualizado` NO compara contra "ahora" directo: una vela diaria se
    indexa por su OPEN time (ver `data.loaders._load_binance`), así que la
    última vela CERRADA disponible en este instante siempre está entre 24 y
    48hs "vieja" respecto de "ahora" aunque el snapshot esté perfectamente
    al día — comparar contra "ahora" marcaría el dato como desactualizado
    casi todo el día, todos los días, incluso recién refrescado. En cambio
    se compara contra la última vela YA CERRADA que existe en este momento
    (`data.snapshot._last_closed_candle_open_time`, la misma cuenta que usa
    `update_snapshot` para decidir qué bajar): si coincide con lo que ya
    tenemos, antigüedad-vs-lo-más-fresco-posible es 0 y no es
    "desactualizado", sin importar la hora del día.

    Fase 17b: "4h"/"1w" son DERIVADOS por resampleo (`data.loaders.RESAMPLED_INTERVALS`)
    — no tienen un "closed candle" propio bien definido para pandas (`1w`
    en particular: `Timestamp.floor("W")` ni siquiera es una operación
    válida, "W" es una frecuencia anclada al calendario, no fija). La
    frescura real de un intervalo derivado la determina su snapshot BASE
    ("1h"/"1d"), así que acá se compara la última vela del snapshot base
    contra el último cierre YA CERRADO de ESE base — `ultima_fecha` en la
    respuesta sigue siendo la del intervalo pedido (lo que el usuario ve),
    pero `desactualizado` refleja la frescura real de los datos subyacentes.
    """
    df = _load_df(asset, interval)
    ultima_fecha = df.index.max()

    base_interval = RESAMPLED_INTERVALS.get(interval, interval)
    base_last = df.index.max() if base_interval == interval else _load_df(asset, base_interval).index.max()

    now = pd.Timestamp.now(tz="UTC")
    delta = now - ultima_fecha
    threshold = STALE_THRESHOLD_SECONDS.get(interval, STALE_THRESHOLD_SECONDS["1d"])
    last_closed_base = last_closed_candle_open_time(base_interval)
    gap_vs_freshest = (last_closed_base - base_last).total_seconds()

    return DataStatusResponse(
        asset=asset,
        interval=interval,
        ultima_fecha=ultima_fecha.to_pydatetime(),
        antiguedad_segundos=delta.total_seconds(),
        antiguedad_texto=_humanize_age(delta),
        desactualizado=gap_vs_freshest > threshold,
    )


# --------------------------------------------------------------------------
# POST /api/refresh
# --------------------------------------------------------------------------


@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Actualiza el snapshot local desde Binance (SÍ toca la red)",
)
def refresh_snapshot(asset: str, interval: str = "1d") -> RefreshResponse:
    """A diferencia de TODOS los demás endpoints de esta API (que solo leen
    `data/snapshot/*.parquet`, nunca la red), este SÍ pega contra
    api.binance.com (reutiliza `data.snapshot.update_snapshot`, que a su vez
    reutiliza `data.loaders._load_binance` — no reimplementa la descarga).
    Puede tardar varios segundos según cuántas velas nuevas haya. Pensado
    para dispararse desde una acción EXPLÍCITA del usuario (un botón
    "Actualizar datos"), nunca automáticamente al cargar una vista.

    "4h"/"1w" (Fase 17b) son DERIVADOS por resampleo — no tienen snapshot
    propio que bajar de Binance (`data.snapshot.SUPPORTED_INTERVALS` sigue
    siendo solo `("1d", "1h")`), así que acá se rechazan con un 400 claro
    en vez de dejar que `update_snapshot` explote con un `ValueError` crudo:
    refrescar el intervalo BASE (`data.loaders.RESAMPLED_INTERVALS`) ya deja
    al día también al derivado, que se recalcula al leer, no se guarda aparte.
    """
    _validate_asset(asset)
    _validate_interval(interval)
    if interval in RESAMPLED_INTERVALS:
        base_interval = RESAMPLED_INTERVALS[interval]
        raise HTTPException(
            status_code=400,
            detail=(
                f"interval='{interval}' se deriva por resampleo de '{base_interval}' — no hay nada propio "
                f"que refrescar. Refrescá interval='{base_interval}' y '{interval}' queda al día solo."
            ),
        )
    try:
        result = update_snapshot(asset, interval)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeoblockedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - cualquier otra falla de red/HTTP, mensaje claro al frontend
        logger.error("refresh_snapshot(%s, %s): falló la actualización desde Binance: %s", asset, interval, exc)
        raise HTTPException(status_code=502, detail=f"No se pudo actualizar '{asset}' desde Binance: {exc}") from exc

    return RefreshResponse(
        asset=asset,
        interval=interval,
        filas_agregadas=result["filas_agregadas"],
        ultima_fecha=result["ultima_fecha"].to_pydatetime(),
        ya_actualizado=result["ya_actualizado"],
    )


# --------------------------------------------------------------------------
# GET /api/stats
# --------------------------------------------------------------------------


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Análisis estadístico y de ciclos de mercado (drawdowns, fases, halving, estacionalidad)",
)
def get_stats(asset: str, interval: str = "1d") -> StatsResponse:
    """Reutiliza `analysis.statistics` (Fase 11, estacionalidad/ACF),
    `analysis.cycles` (Fase 15a, drawdowns/fases de mercado/ciclos de
    halving/heatmap mensual) y `eda.eda_report.adf_test` (Fase 1,
    estacionariedad) tal cual — ningún cálculo estadístico nuevo vive acá,
    solo se arma la respuesta. NADA de esto predice el precio (ver el
    docstring de `analysis/statistics.py`/`analysis/cycles.py`).

    `estacionalidad_horaria` solo se calcula si `interval == "1h"` (con
    velas diarias todas caerían en la hora 0, sin información — ver
    `analysis.statistics.hourly_seasonality`). `halvings_btc`/`ciclos_halving`
    solo se incluyen si `asset == "BTC"`.

    Fase 15a: se sacó el periodograma espectral de esta respuesta — sobre
    retornos diarios daba "ciclos" de 2-3 días (ruido de alta frecuencia,
    sin significado de mercado). Se reemplaza por ciclos con significado
    real: drawdowns históricos, fases bull/bear, y (para BTC) los ciclos
    entre halvings.
    """
    df = _load_df(asset, interval)
    close = df["close"]
    returns = simple_returns(close).dropna()

    monthly = _seasonality_to_buckets(monthly_seasonality(returns))
    weekly = _seasonality_to_buckets(weekday_seasonality(returns))
    hourly = _seasonality_to_buckets(hourly_seasonality(returns)) if interval == "1h" else None

    acf_df = autocorrelation(returns)
    autocorrelacion = [
        AutocorrelationPoint(
            lag=int(lag),
            acf_retornos=_none_if_nan(row["acf_retornos"]),
            acf_retornos2=_none_if_nan(row["acf_retornos2"]),
        )
        for lag, row in acf_df.iterrows()
    ]

    drawdowns = [_drawdown_to_model(d) for d in drawdown_analysis(close)]
    fases_mercado = [_phase_to_model(p) for p in market_phases(close)]

    ciclos_halving = None
    if asset == "BTC":
        hc = halving_cycles(close)
        ciclos_halving = HalvingCyclesInfo(
            ciclos=[_halving_cycle_to_model(c) for c in hc["ciclos"]],
            n_halvings_totales=hc["n_halvings_totales"],
            n_halvings_con_datos=hc["n_halvings_con_datos"],
        )

    heatmap = monthly_yearly_heatmap(returns)
    heatmap_mensual = MonthlyYearlyHeatmap(anios=heatmap["anios"], matriz=heatmap["matriz"])

    return StatsResponse(
        asset=asset,
        interval=interval,
        estacionalidad_mensual=monthly,
        estacionalidad_semanal=weekly,
        estacionalidad_horaria=hourly,
        autocorrelacion=autocorrelacion,
        adf_precio=AdfResult(**adf_test(close)),
        adf_retornos=AdfResult(**adf_test(returns)),
        halvings_btc=list(BTC_HALVING_DATES) if asset == "BTC" else None,
        drawdowns=drawdowns,
        fases_mercado=fases_mercado,
        ciclos_halving=ciclos_halving,
        heatmap_mensual=heatmap_mensual,
    )


# --------------------------------------------------------------------------
# GET /api/compare
# --------------------------------------------------------------------------


@router.get(
    "/compare",
    response_model=CompareResponse,
    summary="Comparación de rendimiento normalizado entre activos",
)
def get_compare(
    assets: str = Query(..., description="Tickers separados por coma, ej. 'BTC,ETH,SOL'"),
    interval: str = "1d",
    limit: int = Query(
        365, gt=0, le=MAX_CANDLE_LIMIT, description="Cantidad de fechas comunes más recientes a comparar"
    ),
) -> CompareResponse:
    """Reutiliza `data.loaders.get_prices` (vía `_load_df`) para cargar cada
    activo y `analysis.comparison.compare_assets` (Fase 12a) para alinear
    por fechas comunes, recortar a las últimas `limit` fechas y normalizar
    cada serie a base 100 — ningún cálculo nuevo vive acá.

    Comparación de DESEMPEÑO HISTÓRICO normalizado, no una predicción — ver
    `CompareResponse`/el texto que muestra el frontend. Si dos activos no
    tienen NINGUNA fecha en común (no pasa con `config.UNIVERSE` actual,
    pero es posible en teoría), devuelve `fechas`/`series` vacíos en vez de
    fallar.
    """
    asset_list = [a.strip().upper() for a in assets.split(",") if a.strip()]
    if not asset_list:
        raise HTTPException(status_code=400, detail="'assets' no puede estar vacío")

    _validate_interval(interval)
    prices: dict[str, pd.Series] = {asset: _load_df(asset, interval)["close"] for asset in asset_list}

    result = compare_assets(prices, limit=limit)

    return CompareResponse(
        assets=asset_list,
        interval=interval,
        fechas=_dates_to_list(result["fechas"]),
        series={asset: _series_to_list(result["normalizado"][asset]) for asset in asset_list},
        rendimiento_total_pct=result["rendimiento_total_pct"],
    )


# --------------------------------------------------------------------------
# GET /api/pairs/screening
# --------------------------------------------------------------------------


@router.get(
    "/pairs/screening",
    response_model=PairScreeningResponse,
    summary="Ranking de operabilidad de pares (arbitraje estadístico)",
)
def get_pairs_screening(interval: str = "1d") -> PairScreeningResponse:
    """Reutiliza `pairs.stability.screen_pairs_stability` tal cual (NO
    reimplementa nada de `pairs/`) — corre cointegración rolling sobre
    todas las combinaciones de `config.UNIVERSE` y arma el ranking honesto
    de operabilidad.

    `interval` solo acepta `"1d"` por ahora: `screen_pairs_stability` carga
    los precios con `get_prices(asset, source="store")` SIN pasar
    `interval` (usa el default de esa función, diario), y sus ventanas
    rolling (365 observaciones / paso de 30) están calibradas para datos
    diarios — no para horas. Pedir `interval="1h"` da 400 en vez de devolver
    silenciosamente un resultado diario disfrazado de horario.

    ENCUADRE: esto es arbitraje ESTADÍSTICO (pairs trading) sobre las 5
    monedas de `config.UNIVERSE`, no arbitraje entre exchanges. La Fase 2 ya
    mostró que la mayoría de estos pares NO están establemente cointegrados
    — este endpoint expone ese resultado tal cual, sin maquillarlo (ver
    `estable` por fila y el texto de la vista "Arbitraje" del frontend).
    """
    if interval != "1d":
        raise HTTPException(
            status_code=400,
            detail=(
                "El screening de pares (pairs.stability.screen_pairs_stability) solo soporta "
                "interval='1d': sus ventanas rolling (365/30 observaciones) están calibradas "
                "para velas diarias."
            ),
        )

    table = screen_pairs_stability()
    filas = [
        PairScreeningRow(
            par=str(row["par"]),
            direccion=str(row["direccion"]),
            n_ventanas=int(row["n_ventanas"]),
            fraccion_cointegrada=float(row["fraccion_cointegrada"]),
            beta_medio=float(row["beta_medio"]),
            beta_std=float(row["beta_std"]),
            estable=bool(row["estable"]),
        )
        for _, row in table.iterrows()
    ]
    return PairScreeningResponse(
        interval=interval,
        filas=filas,
        n_estables=int(table["estable"].sum()),
        n_total=len(table),
    )


# --------------------------------------------------------------------------
# GET /api/pairs/detail
# --------------------------------------------------------------------------


@router.get(
    "/pairs/detail",
    response_model=PairDetailResponse,
    summary="Detalle de un par: cointegración, spread, z-score, estabilidad rolling y backtest",
)
def get_pairs_detail(
    asset_y: str,
    asset_x: str,
    interval: str = "1d",
    bt_entry: float = DEFAULT_PAIR_ENTRY,
    bt_exit: float = DEFAULT_PAIR_EXIT,
    bt_stop: float = DEFAULT_PAIR_STOP,
    bt_cost_bps: float = TRANSACTION_COST_BPS,
) -> PairDetailResponse:
    """Reutiliza `pairs.cointegration.engle_granger`/`half_life`,
    `pairs.stability.rolling_cointegration`/`stability_summary` y (Fase 15b)
    `pairs.backtest.backtest_pair` tal cual (NO reimplementa nada de
    `pairs/`) sobre precios cargados vía `_load_df` (mismo punto de entrada
    que el resto de la API, respeta `interval` de verdad — a diferencia de
    `/api/pairs/screening`, acá SÍ se puede pedir `interval="1h"`).

    Si no hay historia suficiente para ni una ventana rolling completa
    (`pairs.stability.rolling_cointegration` pide 365 observaciones por
    defecto — con `interval="1h"` puede no alcanzar), `estabilidad` viaja en
    `null` con el motivo en `estabilidad_mensaje`, en vez de fallar el
    endpoint entero: el resto del análisis (cointegración in-sample, spread,
    z-score, backtest) sigue siendo válido igual.

    `bt_entry`/`bt_exit`/`bt_stop`/`bt_cost_bps` (Fase 15b): parámetros del
    backtest de `pairs.backtest.backtest_pair` — opcionales, con los mismos
    defaults que esa función (ver su docstring, incluida la nota de por qué
    `bt_exit` default es 0.5 y no 0.0).
    """
    _validate_asset(asset_y)
    _validate_asset(asset_x)
    if asset_y == asset_x:
        raise HTTPException(status_code=400, detail="asset_y y asset_x deben ser distintos")
    _validate_interval(interval)

    y = _load_df(asset_y, interval)["close"]
    x = _load_df(asset_x, interval)["close"]

    eg = engle_granger(y, x)
    spread = eg["spread"]
    hl = half_life(spread)
    z = zscore(spread)
    z_actual = _none_if_nan(float(z.iloc[-1])) if len(z) else None

    zscore_extremos = [
        PairZscoreExtreme(fecha=fecha.to_pydatetime(), z=float(value))
        for fecha, value in z.items()
        if pd.notna(value) and abs(value) >= _ZSCORE_EXTREME_THRESHOLD
    ]

    try:
        rolling = rolling_cointegration(y, x)
        summary = stability_summary(rolling)
        estabilidad = PairStabilitySummary(**summary)
        estabilidad_mensaje = None
    except ValueError as exc:
        estabilidad = None
        estabilidad_mensaje = f"No se pudo evaluar estabilidad rolling: {exc}"

    try:
        bt = backtest_pair(
            y, x, hedge_ratio=eg["beta"], entry=bt_entry, exit=bt_exit, stop=bt_stop, cost_bps=bt_cost_bps
        )
    except ValueError as exc:
        # p. ej. bt_entry/bt_exit/bt_stop no cumplen 0 <= exit < entry < stop
        # (ver pairs.signals.generate_pair_signals) — error del USUARIO vía
        # query params, no una falla del servidor.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    backtest = PairBacktestResult(
        fechas=_dates_to_list(bt["equity_curve"].index),
        equity_curve=[float(v) for v in bt["equity_curve"].to_numpy()],
        metrics=PairBacktestMetrics(**bt["metrics"]),
    )

    return PairDetailResponse(
        asset_y=asset_y,
        asset_x=asset_x,
        interval=interval,
        beta=eg["beta"],
        alpha=eg["alpha"],
        estadistico_adf=eg["estadistico_adf"],
        p_valor_adf=eg["p_valor_adf"],
        es_cointegrado=eg["es_cointegrado"],
        half_life_dias=_none_if_nonfinite(hl),
        fechas=_dates_to_list(spread.index),
        spread=_series_to_list(spread),
        zscore=_series_to_list(z),
        zscore_actual=z_actual,
        zscore_interpretacion=_interpret_zscore(z_actual),
        zscore_extremos=zscore_extremos,
        estabilidad=estabilidad,
        estabilidad_mensaje=estabilidad_mensaje,
        backtest=backtest,
    )


# --------------------------------------------------------------------------
# GET /api/correlation
# --------------------------------------------------------------------------

CORRELATION_METHODS: tuple[str, ...] = ("pearson", "spearman")


def _compute_correlation(interval: str, limit: int, method: str) -> tuple[pd.DataFrame, list[str], int]:
    """Cálculo compartido por `GET /api/correlation` y `GET /api/export/correlation`
    (Fase 17a): reutiliza `eda.eda_report.correlation_matrix` (Fase 1) y
    `analysis.comparison.align_common_dates` (Fase 12a) — alinea por fechas
    comunes y recorta a las últimas `limit` ANTES de correlacionar, así "el
    período elegido" corta la ventana antes del cálculo, no después.

    Devuelve `(corr_df, assets, fechas_n)` — `corr_df` ya indexado/columnado
    por activo, para que cada endpoint lo serialice a su formato (JSON o CSV)
    sin repetir el cálculo.
    """
    _validate_interval(interval)
    if method not in CORRELATION_METHODS:
        raise HTTPException(
            status_code=400, detail=f"method debe ser uno de {list(CORRELATION_METHODS)}, recibido '{method}'"
        )

    assets = list(UNIVERSE)
    returns = {asset: simple_returns(_load_df(asset, interval)["close"]).dropna() for asset in assets}
    aligned = align_common_dates(returns).tail(limit)
    trimmed_returns = {asset: aligned[asset] for asset in assets}

    corr_df = correlation_matrix(trimmed_returns, method=method)
    return corr_df, assets, len(aligned)


@router.get(
    "/correlation",
    response_model=CorrelationResponse,
    summary="Matriz de correlación entre activos (sobre RETORNOS, no precios)",
)
def get_correlation(
    interval: str = "1d",
    limit: int = Query(
        365, gt=0, le=MAX_CANDLE_LIMIT, description="Cantidad de fechas comunes más recientes a usar"
    ),
    method: str = "pearson",
) -> CorrelationResponse:
    """Reutiliza `_compute_correlation` (ver ahí para el detalle de qué
    calcula y por qué) y solo arma la respuesta JSON.

    Correlación de RETORNOS, no de precios: dos precios pueden estar muy
    correlacionados solo porque ambos tienen tendencia (no estacionariedad),
    sin que eso diga nada sobre si se mueven juntos día a día — ver
    `eda.eda_report.correlation_matrix` y el texto que muestra el frontend.
    """
    corr_df, assets, fechas_n = _compute_correlation(interval, limit, method)
    matriz = [[_none_if_nan(corr_df.loc[row, col]) for col in assets] for row in assets]

    return CorrelationResponse(
        interval=interval,
        method=method,
        fechas_n=fechas_n,
        activos=assets,
        matriz=matriz,
    )


# --------------------------------------------------------------------------
# GET /api/export/* (Fase 17a) — mismos cálculos que sus endpoints JSON
# hermanos, solo cambia el formato de salida a CSV descargable (ver
# `_csv_response`). Ninguno reimplementa ningún cálculo.
# --------------------------------------------------------------------------


@router.get("/export/ohlcv", summary="Velas OHLCV en CSV descargable")
def export_ohlcv(
    asset: str,
    interval: str = "1d",
    limit: int = Query(
        500, gt=0, le=MAX_CANDLE_LIMIT, description="Cantidad de velas más recientes a exportar"
    ),
) -> Response:
    """Reutiliza `data.loaders.get_prices` (vía `_load_df`) — misma fuente
    que `GET /api/ohlcv`, acá servida como CSV en vez de JSON.
    """
    df = _load_df(asset, interval).tail(limit)
    csv_df = df.reset_index().rename(columns={"timestamp": "fecha"})
    return _csv_response(csv_df, f"ohlcv_{asset}_{interval}.csv")


@router.get("/export/drawdowns", summary="Drawdowns históricos en CSV descargable")
def export_drawdowns(
    asset: str,
    interval: str = "1d",
    top_n: int = Query(10, gt=0, le=100, description="Cantidad de peores drawdowns a exportar"),
) -> Response:
    """Reutiliza `analysis.cycles.drawdown_analysis` tal cual — misma fuente
    que la tabla de "Drawdowns históricos" de la vista "Ciclos y Estadística"
    (`GET /api/stats`), acá servida como CSV en vez de JSON.
    """
    close = _load_df(asset, interval)["close"]
    episodios = drawdown_analysis(close, top_n=top_n)
    csv_df = pd.DataFrame(episodios)
    return _csv_response(csv_df, f"drawdowns_{asset}_{interval}.csv")


@router.get("/export/correlation", summary="Matriz de correlación en CSV descargable")
def export_correlation(
    interval: str = "1d",
    limit: int = Query(
        365, gt=0, le=MAX_CANDLE_LIMIT, description="Cantidad de fechas comunes más recientes a usar"
    ),
    method: str = "pearson",
) -> Response:
    """Reutiliza `_compute_correlation` (mismo cálculo que `GET /api/correlation`)
    — acá se exporta la matriz tal cual la devuelve `pandas.DataFrame.corr`,
    con el nombre de cada activo como índice de fila.
    """
    corr_df, _assets, _fechas_n = _compute_correlation(interval, limit, method)
    corr_df = corr_df.copy()
    corr_df.index.name = "activo"
    return _csv_response(corr_df, f"correlation_{interval}_{method}.csv", index=True)


# --------------------------------------------------------------------------
# GET /api/report
# --------------------------------------------------------------------------


@router.get("/report", summary="Informe PDF descargable (gráficas + explicaciones)")
def get_report(asset: str, interval: str = "1d") -> Response:
    """Reutiliza `reports.pdf_report.build_report` — arma un PDF con
    gráficas (matplotlib) y texto explicativo a partir de TODO el backend de
    análisis (riesgo/GARCH, estudios/sugeridor, ciclos/estadística,
    correlación, screening de pares), sin recalcular nada acá. El informe
    EXPLICA los análisis, no recomienda comprar/vender (mismo encuadre
    honesto que el resto de la API).

    MUY LENTO: ajusta un modelo GARCH y corre cointegración rolling sobre
    todos los pares de `config.UNIVERSE` (ver el docstring de rendimiento
    de `reports/pdf_report.py`) — del orden de 30 a 90 segundos. Pensado
    para dispararse desde un botón explícito del usuario, nunca automático.
    """
    _validate_asset(asset)
    _validate_interval(interval)

    try:
        pdf_bytes = build_report(asset, interval=interval)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = f"informe_{asset}_{interval}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


app.include_router(router)
