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
"correr predicción"). `/api/backtest` corre un backtest vectorizado
completo (rápido). El resto (`/api/assets`, `/api/ohlcv`, `/api/studies`,
`/api/suggester`, `/api/data-status`) son livianos (indicadores
vectorizados, sin ajuste de modelos). Ninguno cachea entre requests — cada
llamada recalcula desde cero (simple y correcto; una capa de caché queda
para una fase futura si hiciera falta).

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

import logging

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    AssetsResponse,
    BacktestResponse,
    Candle,
    DataStatusResponse,
    EquityPoint,
    GarchSeriesResponse,
    OHLCVResponse,
    PredictionResponse,
    RefreshResponse,
    RiskResponse,
    StudiesResponse,
    SuggesterResponse,
)
from backtest.engine import backtest_from_prices, compare_to_buy_and_hold
from config import UNIVERSE
from data.loaders import get_prices
from data.snapshot import GeoblockedError, last_closed_candle_open_time, update_snapshot
from metrics.risk_measures import expected_shortfall, value_at_risk
from ml.features import align_features_labels, build_feature_matrix
from ml.labeling import get_daily_volatility, triple_barrier_labels
from ml.models import evaluate_primary_with_roc_auc, feature_importances, fit_final, latest_oos_prediction, t1_from_labels
from models.garch import conditional_volatility, select_best_model, volatility_regime
from signals.engine import generate_positions, latest_recommendation
from signals.indicators import add_all_indicators
from signals.returns import log_returns, simple_returns
from signals.studies import all_studies, ichimoku, stochastic
from signals.suggester import suggest

logger = logging.getLogger(__name__)

# Intervalos que efectivamente tiene exportados el snapshot local
# (`source="store"`, ver scripts/export_snapshot.py y Fase 6b). Todos los
# endpoints de LECTURA solo leen de ahí — el único que golpea Binance en
# vivo es `POST /api/refresh` (Fase 9a, ver data.snapshot.update_snapshot).
SUPPORTED_INTERVALS: tuple[str, ...] = ("1d", "1h")
DEFAULT_RISK_INTERVAL = "1d"  # el modelo GARCH del proyecto es diario, ver models/garch.py

# Activos con cobertura on-chain suficiente para el modelo enriquecido de
# /api/prediction (mismo criterio que `app.workers.ONCHAIN_ENABLED_ASSETS`
# — ver ese módulo para el detalle de por qué BTC/ETH sí y el resto no).
ONCHAIN_ENABLED_ASSETS: frozenset[str] = frozenset({"BTC", "ETH"})

# Umbral de antigüedad a partir del cual /api/data-status marca el snapshot
# como desactualizado (Fase 9a): una vela diaria recién cerrada tiene hasta
# 1 día de "antigüedad normal"; una horaria, hasta 2h (algo de margen sobre
# el intervalo mismo, para no marcar en ámbar el instante justo después de
# que cierra la última vela).
STALE_THRESHOLD_SECONDS: dict[str, float] = {"1d": 24 * 3600, "1h": 2 * 3600}

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


def _none_if_nan(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def _series_to_list(series: pd.Series) -> list[float | None]:
    return [_none_if_nan(v) for v in series.to_numpy(dtype=float)]


def _dates_to_list(index: pd.DatetimeIndex) -> list:
    return list(index.to_pydatetime())


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
    limit: int = Query(500, gt=0, le=5000, description="Cantidad de velas más recientes a devolver"),
) -> OHLCVResponse:
    """Reutiliza `data.loaders.get_prices` tal cual; solo recorta a las
    últimas `limit` velas (para no mandar, p. ej., las ~58.000 velas
    horarias completas de un golpe, ver Fase 6b) y serializa a JSON.
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
    limit: int = Query(500, gt=0, le=5000, description="Cantidad de velas más recientes a devolver"),
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
    `expected_shortfall` y `signals.engine.latest_recommendation` — la misma
    secuencia de llamadas que arma `app.workers.AnalysisWorker` para la
    pestaña "Riesgo" del cockpit, reescrita acá para no depender de la app
    de escritorio (PySide6) desde la API. LENTO: ajusta un modelo GARCH por
    request (ver docstring del módulo).
    """
    df = _load_df(asset, DEFAULT_RISK_INTERVAL)
    close = df["close"]

    returns = simple_returns(close).dropna()
    garch_returns = log_returns(close).dropna()

    best = select_best_model(garch_returns, criterion="aic")
    cond_vol = conditional_volatility(best["result"])
    regime_series = volatility_regime(cond_vol)
    last_regime = regime_series.iloc[-1]

    var95 = value_at_risk(returns, level=0.95)
    es95 = expected_shortfall(returns, level=0.95)

    recomendacion = latest_recommendation(df, garch_regime=False)

    return RiskResponse(
        asset=asset,
        vol_realizada=float(recomendacion["vol_realizada"]),
        modelo_garch=f"{best['vol']}/{best['dist']}",
        vol_garch=float(cond_vol.iloc[-1]),
        regimen=str(last_regime) if pd.notna(last_regime) else None,
        var95=float(var95),
        es95=float(es95),
        accion=str(recomendacion["accion"]),
        score=float(recomendacion["score"]),
        tamano_sugerido=float(recomendacion["tamaño_sugerido"]),
        ultima_fecha=close.index[-1].to_pydatetime(),
    )


# --------------------------------------------------------------------------
# GET /api/backtest
# --------------------------------------------------------------------------


@router.get("/backtest", response_model=BacktestResponse, summary="Backtest estrategia vs. buy & hold")
def get_backtest(asset: str) -> BacktestResponse:
    """Reutiliza `signals.engine.generate_positions`, `backtest.engine.backtest_from_prices`
    y `compare_to_buy_and_hold` — la misma secuencia que
    `app.workers.BacktestWorker` para la pestaña "Backtest" del cockpit.
    """
    df = _load_df(asset, DEFAULT_RISK_INTERVAL)
    close = df["close"]
    positions = generate_positions(df)

    result_estrategia = backtest_from_prices(close, positions)
    asset_returns = simple_returns(close)
    comparacion = compare_to_buy_and_hold(asset_returns, result_estrategia)
    result_buy_and_hold = backtest_from_prices(close, 1.0)

    return BacktestResponse(
        asset=asset,
        metrics_estrategia=comparacion["estrategia"],
        metrics_buy_and_hold=comparacion["buy_and_hold"],
        equity_curve_estrategia=[
            EquityPoint(fecha=fecha.to_pydatetime(), valor=float(valor))
            for fecha, valor in result_estrategia.equity_curve.items()
        ],
        equity_curve_buy_and_hold=[
            EquityPoint(fecha=fecha.to_pydatetime(), valor=float(valor))
            for fecha, valor in result_buy_and_hold.equity_curve.items()
        ],
    )


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
    """
    df = _load_df(asset, interval)
    ultima_fecha = df.index.max()
    now = pd.Timestamp.now(tz="UTC")
    delta = now - ultima_fecha
    threshold = STALE_THRESHOLD_SECONDS.get(interval, STALE_THRESHOLD_SECONDS["1d"])
    last_closed = last_closed_candle_open_time(interval)
    gap_vs_freshest = (last_closed - ultima_fecha).total_seconds()

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
    """
    _validate_asset(asset)
    _validate_interval(interval)
    try:
        result = update_snapshot(asset, interval)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeoblockedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - cualquier otra falla de red/HTTP, mensaje claro al frontend
        raise HTTPException(status_code=502, detail=f"No se pudo actualizar '{asset}' desde Binance: {exc}") from exc

    return RefreshResponse(
        asset=asset,
        interval=interval,
        filas_agregadas=result["filas_agregadas"],
        ultima_fecha=result["ultima_fecha"].to_pydatetime(),
        ya_actualizado=result["ya_actualizado"],
    )


app.include_router(router)
