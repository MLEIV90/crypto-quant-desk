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
    vwap: list[float | None] = Field(description="VWAP en ventana móvil de 20 velas — ver signals/indicators.py::vwap")
    obv: list[float | None] = Field(description="On Balance Volume acumulado — ver signals/indicators.py::obv")
    ichimoku_tenkan: list[float | None] = Field(description="Línea de conversión Ichimoku (9 períodos)")
    ichimoku_kijun: list[float | None] = Field(description="Línea base Ichimoku (26 períodos)")
    ichimoku_senkou_a: list[float | None] = Field(description="Span A Ichimoku (borde de la nube), desplazado 26 períodos hacia adelante")
    ichimoku_senkou_b: list[float | None] = Field(description="Span B Ichimoku (borde de la nube), desplazado 26 períodos hacia adelante")
    ichimoku_chikou: list[float | None] = Field(
        description="Línea rezagada Ichimoku — usa close FUTURO (ver signals/studies.py::ichimoku), solo para lectura visual retrospectiva"
    )
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


class RiskPercentiles(BaseModel):
    """Percentil (0-100) de cada métrica de riesgo (Fase 20a, base
    unificada en Fase 20c) — ver `metrics.risk_measures.historical_percentile`:
    "¿qué fracción del ÚLTIMO AÑO de esta métrica tuvo un valor MENOR O
    IGUAL al de hoy?". Descriptivo, no predictivo — un percentil alto no
    anticipa nada sobre mañana, solo describe que hoy es inusual respecto
    de lo reciente.

    Fase 20c (coherencia): las CUATRO métricas de acá usan la MISMA
    ventana de comparación — `base` la nombra explícitamente, para que
    quede claro que es la MISMA base que usa `RiskResponse.regimen`
    (`models.garch.volatility_regime`, también rolling de 1 año) — antes de
    esta fase, estos percentiles comparaban contra TODA la historia
    mientras el régimen comparaba contra el último año, lo que podía
    mostrar, por ejemplo, "TENSIÓN" junto a un percentil bajo sin ninguna
    explicación de por qué no contradecían: eran dos lentes distintos sin
    rotular. `var95`/`es95` acá corresponden a `RiskResponse.var95_actual`/
    `es95_actual` (el VaR/ES EMPÍRICO rolling de HOY, Fase 25 — ver esos
    campos) — NO a `RiskResponse.var95`/`es95` (todo el historial), que al
    ser un único número sobre toda la serie no tiene un "percentil de hoy"
    que tenga sentido calcular.
    """

    vol_realizada: float | None
    vol_garch: float | None
    var95: float | None
    es95: float | None
    base: str = Field(description="Rótulo legible de la ventana de comparación compartida por las 4 métricas de acá")


class ReturnHistogram(BaseModel):
    """Distribución de retornos diarios del activo (Fase 20a), con las
    marcas de VaR 95% y ES 95% ya convertidas a RETORNO (negativo, para
    ubicarlas directamente sobre el eje del histograma) — ver
    `metrics.risk_measures.value_at_risk`/`expected_shortfall` (que
    devuelven la pérdida en positivo; acá el signo se invierte una sola vez
    para que el frontend no tenga que repetir esa conversión).
    """

    bin_edges: list[float] = Field(description="Bordes de los bins (longitud = len(counts) + 1)")
    counts: list[int] = Field(description="Cantidad de días en cada bin")
    var95_return: float = Field(description="Retorno correspondiente al VaR 95% (negativo)")
    es95_return: float = Field(description="Retorno correspondiente al ES 95% (negativo, <= var95_return)")


class RiskResponse(BaseModel):
    """Respuesta de `GET /api/risk` — mismas magnitudes que calcula
    `app.workers.AnalysisWorker` para la pestaña "Riesgo" del cockpit,
    siempre en diario (el modelo GARCH del proyecto es diario).

    COHERENCIA (Fase 20c, actualizado en Fase 25 — léase antes de tocar
    esto): `var95`/`es95` son puntuales sobre TODA la serie histórica — NO
    se mueven con el régimen actual (un activo en "tensión" y ese mismo
    activo en "calma" muestran el MISMO número), así que no sirven para
    responder "¿cuánto riesgo hay HOY?". Para eso están
    `var95_actual`/`es95_actual`: VaR/ES EMPÍRICO recalculado en ventana
    móvil de 1 año (`metrics.risk_measures.rolling_value_at_risk`/
    `rolling_expected_shortfall`) — es la magnitud que SÍ hay que mirar
    para "riesgo actual", y la que muestra el frontend como titular.
    `var95`/`es95` quedan como referencia secundaria de largo plazo,
    rotulados explícitamente (`historico_basis`) para no confundirlos.

    Fase 25 (unificación): antes de esta fase, `var95_actual`/`es95_actual`
    acá eran el VaR/ES PARAMÉTRICO implícito por el modelo GARCH ya
    ajustado (`models.garch.garch_var`/`garch_expected_shortfall`) — un
    método DISTINTO al que ya usaba `RiskSummaryRow.var95` (empírico
    rolling), así que un mismo activo mostraba dos números de "VaR actual"
    distintos sin que el usuario supiera que eran dos métodos diferentes.
    Ahora ambos endpoints usan el MISMO método empírico rolling, con la
    MISMA etiqueta de base (`actual_basis` acá == `RiskSummaryRow.var95_basis`)
    — para un mismo activo, `var95_actual` de acá == `var95` de la fila
    correspondiente de `GET /api/risk-summary`. La volatilidad condicional
    GARCH (`vol_garch`) NO se pierde: sigue siendo la métrica de
    volatilidad que se muestra, y `regimen` sigue basándose en ella — solo
    el VaR/ES "actual" dejó de ser paramétrico.
    """

    asset: str
    vol_realizada: float = Field(description="Volatilidad realizada anualizada (rolling)")
    modelo_garch: str = Field(description='Modelo GARCH ganador, formato "vol/dist" (p. ej. "EGarch/t")')
    vol_garch: float = Field(description="Volatilidad condicional GARCH anualizada, última vela")
    regimen: str | None = Field(description='"calma" / "normal" / "tension", o null si no hay suficiente historia')
    regimen_basis: str = Field(description="Rótulo legible de la ventana que usa 'regimen' (y los percentiles de 'percentiles')")
    var95: float = Field(description="VaR histórico al 95% sobre TODA la serie (pérdida positiva) — referencia de largo plazo, NO refleja el régimen actual")
    es95: float = Field(description="Expected Shortfall histórico al 95% sobre TODA la serie (pérdida positiva) — misma referencia que var95")
    var95_actual: float = Field(
        description="VaR 95% EMPÍRICO en ventana móvil de 1 año (pérdida positiva) — mismo método que RiskSummaryRow.var95, SÍ refleja el régimen actual"
    )
    es95_actual: float = Field(
        description="Expected Shortfall 95% EMPÍRICO en ventana móvil de 1 año (pérdida positiva)"
    )
    historico_basis: str = Field(description="Rótulo legible de la base de var95/es95")
    actual_basis: str = Field(description="Rótulo legible de la base de var95_actual/es95_actual")
    accion: str = Field(description="LONG / FLAT / SHORT (signals.engine)")
    score: float
    tamano_sugerido: float = Field(description="Tamaño de posición por vol targeting, en [-1, 1]")
    ultima_fecha: datetime
    percentiles: RiskPercentiles
    histograma: ReturnHistogram


class RiskSummaryRow(BaseModel):
    """Fila de `GET /api/risk-summary` (Fase 20b, coherencia en Fase 20c)
    para UNA moneda.

    A propósito NO incluye `vol_garch`: este endpoint evalúa las 5 monedas
    de `config.UNIVERSE` en una sola respuesta, y ajustar un modelo GARCH
    por activo (como hace `GET /api/risk`) sería demasiado lento para una
    tabla comparativa pensada para cargar de un vistazo, sin un botón
    explícito de por medio (ver el docstring de `get_risk_summary`).

    COHERENCIA (Fase 20c, léase antes de comparar esto con `GET /api/risk`):
    `regimen` usa la MISMA función que `GET /api/risk`
    (`models.garch.volatility_regime`), pero sobre volatilidad REALIZADA en
    vez de condicional GARCH (`regimen_basis` lo rotula explícitamente) —
    por eso una misma moneda puede aparecer "tensión" acá y "calma" en el
    panel detallado (o viceversa) sin que sea una inconsistencia: son dos
    métodos de medición distintos, no dos respuestas contradictorias sobre
    lo mismo. `var95` tampoco es el histórico de toda la serie (a
    diferencia de `RiskResponse.var95`): es un VaR "actual" recalculado en
    ventana móvil de 1 año (`metrics.risk_measures.rolling_value_at_risk`)
    — desde Fase 25, el MISMO método (y la MISMA `var95_basis`) que usa
    `RiskResponse.var95_actual`: antes de esa fase, `GET /api/risk` usaba
    en cambio un VaR paramétrico implícito por GARCH, un método DISTINTO
    que daba un número distinto para el mismo activo/momento sin rotularlo
    — ver el docstring de `RiskResponse` para el detalle de la unificación.
    """

    asset: str
    vol_realizada: float = Field(description="Volatilidad realizada anualizada (rolling)")
    vol_realizada_percentil: float | None = Field(description="Percentil de vol_realizada vs su propia ventana reciente, 0-100")
    regimen: str | None = Field(
        description='"calma"/"normal"/"tension" según la volatilidad REALIZADA (no GARCH, ver arriba), o null'
    )
    regimen_basis: str = Field(description="Rótulo legible de qué volatilidad y ventana usa 'regimen' acá")
    var95: float = Field(description="VaR 'actual' en ventana móvil de 1 año (pérdida positiva) — no el histórico de toda la serie")
    var95_percentil: float | None = Field(description="Percentil de var95 vs su propia ventana reciente, 0-100")
    var95_basis: str = Field(description="Rótulo legible de la base de var95 acá")
    ultima_fecha: datetime


class RiskSummaryResponse(BaseModel):
    """Respuesta de `GET /api/risk-summary` (Fase 20b): comparación de
    riesgo actual entre las 5 monedas de `config.UNIVERSE`, en el mismo
    orden que ese diccionario.
    """

    filas: list[RiskSummaryRow]


class EquityPoint(BaseModel):
    """Un punto de una curva de equity (base 1.0)."""

    fecha: datetime
    valor: float


class BacktestResponse(BaseModel):
    """Respuesta de `GET /api/backtest`: métricas + curvas de equity y de
    drawdown de la estrategia elegida vs. buy & hold, ambas con costos de
    transacción (ver `backtest/engine.py`).

    `strategy`/`cost_bps`/`fecha_inicio`/`fecha_fin` (Fase 21) son un ECO de
    los parámetros efectivamente aplicados (incluyendo los defaults que se
    hayan usado cuando no se pasaron) — así el frontend siempre puede
    mostrar exactamente qué se corrió sin tener que duplicar los defaults
    del backend. `drawdown_curve_*` (Fase 21): la curva "underwater"
    completa (`metrics.risk_measures.drawdown_series`) de cada lado, para el
    gráfico que muestra POR QUÉ una estrategia sufre menos que la otra, no
    solo su peor valor puntual (ya en `metrics_*["max_drawdown"]`).
    `exposure_curve_estrategia` (Fase 23): la posición EFECTIVA de la
    estrategia elegida en cada fecha (`backtest.engine.BacktestResult.positions`,
    ya con el desfase anti-lookahead aplicado) — en [-1, 1]. Solo del lado
    de la estrategia (no del buy & hold, que es trivialmente 1.0 todo el
    tiempo y no aporta nada graficado aparte).
    """

    asset: str
    strategy: str
    cost_bps: float
    fecha_inicio: datetime | None
    fecha_fin: datetime | None
    metrics_estrategia: dict[str, float]
    metrics_buy_and_hold: dict[str, float]
    equity_curve_estrategia: list[EquityPoint]
    equity_curve_buy_and_hold: list[EquityPoint]
    drawdown_curve_estrategia: list[EquityPoint]
    drawdown_curve_buy_and_hold: list[EquityPoint]
    exposure_curve_estrategia: list[EquityPoint]


class BacktestStrategyInfo(BaseModel):
    """Descripción de una estrategia backtesteable vía `GET /api/backtest`
    (Fase 21) — la MISMA fuente que usa el backend para calcular, expuesta
    como datos para que el selector del frontend nunca tenga que hardcodear
    (y arriesgarse a desincronizar) una explicación de lo que la estrategia
    realmente hace.
    """

    id: str
    nombre: str
    descripcion: str
    objetivo: str
    tradeoff: str
    tiene_target_vol: bool
    target_vol_default: float | None
    target_vol_min: float | None
    target_vol_max: float | None


class BacktestStrategiesResponse(BaseModel):
    """Respuesta de `GET /api/backtest-strategies`: catálogo de estrategias
    + el default de costos de transacción, para poblar el selector y los
    parámetros configurables de la pestaña Backtest sin hardcodear nada del
    backend en el frontend.
    """

    estrategias: list[BacktestStrategyInfo]
    cost_bps_default: float


class GarchSeriesResponse(BaseModel):
    """Respuesta de `GET /api/garch-series`: serie temporal completa de
    volatilidad condicional GARCH, para graficarla (p. ej. el panel de
    riesgo del futuro frontend).

    `regimen_serie` (Fase 20a): la clasificación calma/normal/tensión
    (`models.garch.volatility_regime`) de CADA fecha, no solo la actual —
    misma longitud e índice que `fechas`/`vol_condicional` — para pintar
    una franja de régimen a lo largo del tiempo, no solo mostrar el régimen
    de hoy.
    """

    asset: str
    fechas: list[datetime]
    vol_condicional: list[float | None]
    modelo_garch: str
    regimen_actual: str | None
    regimen_serie: list[str | None]


class PredictionResponse(BaseModel):
    """Respuesta de `GET /api/prediction` — INVESTIGACIÓN CON RESULTADO
    NEGATIVO, no una herramienta operativa: el modelo primario de ML nunca
    superó de forma consistente a los baselines triviales en ninguna
    validación del proyecto (ver `ml/models.py`, Fases 3c/5b/6a/6c). Se
    expone tal cual sale, sin maquillar, igual que en la pestaña
    "Research (sin edge)" de la app de escritorio.
    """

    asset: str
    used_onchain: bool = Field(description="True solo para BTC/ETH, ver data/onchain.py")
    onchain_columns: list[str]
    ultima_fecha: datetime
    prediccion_clase: str = Field(description="LONG / FLAT / SHORT")
    prediccion_confianza: float = Field(description="Probabilidad OOS de la clase predicha, en [0, 1]")
    prediccion_proba: dict[str, float]
    accuracy_media: float
    baseline_azar: float
    baseline_mayoritaria: float
    supera_azar: bool
    supera_mayoritaria: bool
    roc_auc_media: float
    top_features: list[tuple[str, float]]


# --------------------------------------------------------------------------
# GET /api/research-experiments (Fase 24) — LEE (no recalcula) los resultados
# YA guardados de los experimentos lentos de Deep RL (Fase 18) y rotación por
# momentum (Fase 19a). Cada campo espeja tal cual el JSON que ya guardan
# `scripts/run_rl_experiment.py`/`scripts/run_rotation_experiment.py` — ver
# `api/main.py::get_research_experiments` para el detalle de cómo se localiza
# el archivo más reciente y por qué degrada con gracia (campo en `None`) si
# el experimento nunca se corrió.
# --------------------------------------------------------------------------


class RlParams(BaseModel):
    """Parámetros con los que se corrió el experimento de RL guardado (`rl/results/*.json`)."""

    assets: list[str]
    min_train_days: int
    n_blocks: int
    seeds: list[int]
    total_timesteps: int
    cost_bps: float


class RlBlock(BaseModel):
    """Un bloque walk-forward: índices de barra (no fechas) de train/test."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


class RlSummaryRow(BaseModel):
    """Una fila de `summary_table`: una estrategia (RL o un baseline) con sus
    métricas promediadas sobre las semillas — `*_std` es 0.0 para los
    baselines determinísticos (buy & hold, vol targeting), que no dependen
    de una semilla aleatoria.
    """

    estrategia: str
    sharpe_media: float
    sharpe_std: float
    retorno_anualizado_media: float
    retorno_anualizado_std: float
    retorno_total_media: float
    retorno_total_std: float
    max_drawdown_media: float
    max_drawdown_std: float
    turnover_total_media: float
    turnover_total_std: float
    turnover_medio_diario_media: float
    turnover_medio_diario_std: float


class RlConclusion(BaseModel):
    """Veredicto honesto (`rl.evaluation.rl_beats_all_baselines`): exige que
    la PEOR semilla del agente, no el promedio, supere a TODOS los
    baselines — una sola semilla mala alcanza para responder "No".
    """

    supera_a_todos_los_baselines_consistentemente: bool
    sharpe_rl_peor_semilla: float
    sharpe_rl_mejor_semilla: float
    sharpe_baselines: dict[str, float]


class RlResearchResult(BaseModel):
    """Resultado completo del experimento de Deep RL más reciente guardado en
    `rl/results/` — `fecha_experimento` se deriva del timestamp en el
    nombre del archivo (`rl_experiment_YYYYMMDD_HHMMSS.json`), no está en
    el JSON en sí.
    """

    fecha_experimento: datetime
    params: RlParams
    elapsed_seconds: float
    n_ppo_runs: int
    oos_date_range: list[str]
    blocks: list[RlBlock]
    summary_table: list[RlSummaryRow]
    conclusion: RlConclusion


class RotationParams(BaseModel):
    """Parámetros del experimento de rotación guardado (`strategies/results/*.json`)."""

    pairs: list[str]
    lookback_grid: list[int]
    rebalance_grid: list[int]
    cost_bps: float | None
    primary_pair: str


class RotationSummaryRow(BaseModel):
    """Una fila de `summary_table`: un par x lookback x rebalanceo. Los
    `sharpe_buy_hold_<ASSET>` de monedas que NO son parte de `par` llegan
    en `None` (NaN en el JSON original, saneado antes de validar — ver
    `api/main.py::_nan_to_none`).
    """

    par: str
    lookback_days: int
    rebalance_days: int
    sharpe_rotacion: float | None
    cagr_rotacion: float | None
    retorno_total_rotacion: float | None
    max_drawdown_rotacion: float | None
    n_rotaciones: int
    mejor_baseline: str
    sharpe_mejor_baseline: float | None
    gana_al_mejor_baseline: bool
    sharpe_buy_hold_BTC: float | None = None
    sharpe_buy_hold_ETH: float | None = None
    sharpe_buy_hold_SOL: float | None = None
    sharpe_buy_hold_BNB: float | None = None
    sharpe_buy_hold_LTC: float | None = None
    sharpe_50_50_rebalanceado: float | None = None


class RotationConclusion(BaseModel):
    """Veredicto honesto (`strategies.rotation.rotation_beats_baselines_robustly`):
    exige que el par principal sea robusto en TODAS sus combinaciones de
    parámetros Y que al menos la mitad del resto de los pares también lo sea.
    """

    robusto_par_principal: bool
    par_principal: str
    fraccion_pares_robustos: float
    pares_robustos: list[str]
    veredicto_global: bool


class RotationResearchResult(BaseModel):
    """Resultado completo del experimento de rotación por momentum más
    reciente guardado en `strategies/results/` — `fecha_experimento` se
    deriva del timestamp en el nombre del archivo
    (`rotation_experiment_YYYYMMDD_HHMMSS.json`).
    """

    fecha_experimento: datetime
    params: RotationParams
    elapsed_seconds: float
    n_combos: int
    summary_table: list[RotationSummaryRow]
    per_pair_robusto: dict[str, bool]
    conclusion: RotationConclusion


class ResearchExperimentsResponse(BaseModel):
    """Respuesta de `GET /api/research-experiments`: los resultados YA
    guardados de RL y rotación, cada uno `None` si ese experimento nunca se
    corrió (degrada con gracia — el endpoint no falla, el frontend muestra
    "experimento no corrido aún").
    """

    rl: RlResearchResult | None
    rotation: RotationResearchResult | None


class RefreshResponse(BaseModel):
    """Respuesta de `POST /api/refresh` — resultado de `data.snapshot.update_snapshot`."""

    asset: str
    interval: str
    filas_agregadas: int = Field(description="Velas nuevas agregadas al snapshot local")
    ultima_fecha: datetime = Field(description="Última fecha disponible en el snapshot tras la actualización")
    ya_actualizado: bool = Field(description="True si no había velas nuevas para bajar (snapshot ya al día)")


class DataStatusResponse(BaseModel):
    """Respuesta de `GET /api/data-status` — antigüedad del snapshot local, para que la UI la muestre siempre."""

    asset: str
    interval: str
    ultima_fecha: datetime = Field(description="Última fecha disponible en el snapshot local")
    antiguedad_segundos: float = Field(description="Segundos transcurridos entre ultima_fecha y ahora (UTC)")
    antiguedad_texto: str = Field(description='Texto legible, p. ej. "hace 5 días" o "hace 3 horas"')
    desactualizado: bool = Field(
        description="True si la antigüedad supera el umbral esperado del intervalo (1 día para 1d, 2h para 1h)"
    )


class SeasonalityBucket(BaseModel):
    """Una fila de `analysis.statistics.monthly_seasonality`/`weekday_seasonality`/
    `hourly_seasonality` (Fase 11, `mediana`/`desvio` en las tres desde Fase
    26) — `bucket` es el mes (1-12), día de semana (0=lunes..6=domingo) u
    hora UTC (0-23) según el campo de `StatsResponse` que la contiene.
    `mediana`/`desvio` pueden ser `None` si ese bucket tiene una sola
    observación (desvío muestral con 1 dato no está definido).
    """

    bucket: int
    retorno_medio: float
    mediana: float | None = None
    desvio: float | None = None
    n: int


class AutocorrelationPoint(BaseModel):
    """Un rezago de `analysis.statistics.autocorrelation` (Fase 11)."""

    lag: int
    acf_retornos: float | None = Field(description="ACF del NIVEL de retorno — cerca de 0 es lo esperado")
    acf_retornos2: float | None = Field(
        description="ACF de los retornos al cuadrado — positiva y persistente indica clustering de volatilidad"
    )


class DrawdownEpisode(BaseModel):
    """Un episodio de drawdown (`analysis.cycles.drawdown_analysis`, Fase 15a) — reutilizado tal cual."""

    fecha_pico: datetime = Field(description="Último máximo histórico antes de empezar a caer")
    fecha_fondo: datetime = Field(description="Día del mínimo dentro de este episodio")
    profundidad_pct: float = Field(description="(precio_fondo / precio_pico - 1) * 100 — negativo")
    fecha_recuperacion: datetime | None = Field(
        description="Primer día en que el precio vuelve a igualar el pico — null si todavía no recuperó"
    )
    dias_caida: int = Field(description="Días corridos entre el pico y el fondo")
    dias_recuperacion: int | None = Field(description="Días corridos entre el fondo y la recuperación, o null")


class MarketPhase(BaseModel):
    """Una fase bull/bear (`analysis.cycles.market_phases`, Fase 15a) — reutilizado tal cual."""

    tipo: str = Field(description='"bull" o "bear"')
    fecha_inicio: datetime
    fecha_fin: datetime
    duracion_dias: int
    retorno_pct: float = Field(description="(precio_fin / precio_inicio - 1) * 100")
    confirmada: bool = Field(
        description="True si un movimiento opuesto de threshold (20%) ya cerró la fase; False = tramo más reciente, todavía 'en curso'"
    )


class HalvingCycle(BaseModel):
    """Un ciclo entre halvings de Bitcoin (`analysis.cycles.halving_cycles`, Fase 15a)."""

    fecha_inicio: datetime
    fecha_fin: datetime
    en_curso: bool = Field(description="True para el tramo desde el último halving disponible hasta hoy")
    duracion_dias: int
    retorno_pct: float
    drawdown_maximo_pct: float = Field(description="Peor caída pico-a-valle DENTRO de este ciclo")


class HalvingCyclesInfo(BaseModel):
    """Respuesta completa de `analysis.cycles.halving_cycles` (Fase 15a) — incluye el
    caveat de tamaño de muestra explícito (ver `n_halvings_totales`/`n_halvings_con_datos`).
    """

    ciclos: list[HalvingCycle]
    n_halvings_totales: int = Field(description="4 — la cantidad real de halvings en TODA la historia de Bitcoin")
    n_halvings_con_datos: int = Field(description="Cuántos de esos 4 caen dentro del histórico de precios disponible")


class MonthlyYearlyHeatmap(BaseModel):
    """Matriz mes x año de retornos compuestos (`analysis.cycles.monthly_yearly_heatmap`, Fase 15a)."""

    anios: list[int] = Field(description="Años presentes, ascendente — columnas de 'matriz'")
    matriz: list[list[float | None]] = Field(
        description="12 filas (índice 0=enero..11=diciembre), una columna por año en 'anios'; retorno_pct o null si no hay datos para ese mes-año"
    )


class AdfResult(BaseModel):
    """Resultado de `eda.eda_report.adf_test` (test Augmented Dickey-Fuller de raíz unitaria), reutilizado tal cual."""

    estadistico: float
    p_valor: float
    n_lags: int = Field(description="Rezagos elegidos automáticamente por criterio AIC")
    n_obs: int
    valores_criticos: dict[str, float] = Field(description="Valores críticos al 1%/5%/10%")
    es_estacionaria: bool = Field(description="True si se rechaza H0 (raíz unitaria) al 5% de significancia")


class StatsResponse(BaseModel):
    """Respuesta de `GET /api/stats` (Fase 11, rehecha en Fase 15a) —
    estacionalidad, autocorrelación, estacionariedad (ADF) y ciclos de
    mercado REALES (drawdowns/fases/halving/heatmap mensual). NADA de esto
    predice el precio — ver el docstring de `analysis/statistics.py`,
    `analysis/cycles.py` y los tooltips del frontend (`helpTexts.ts`).

    Fase 15a: se SACÓ el periodograma espectral de esta respuesta — sobre
    retornos diarios daba "ciclos" de 2-3 días (ruido de alta frecuencia,
    sin ningún significado de mercado). La función
    `analysis.statistics.spectral_periodogram` sigue existiendo en el
    backend (marcada deprecada), pero ya no se expone acá.
    """

    asset: str
    interval: str
    estacionalidad_mensual: list[SeasonalityBucket]
    estacionalidad_semanal: list[SeasonalityBucket]
    estacionalidad_horaria: list[SeasonalityBucket] | None = Field(
        default=None, description="Solo presente si interval='1h' — con velas diarias, todas caerían en la hora 0"
    )
    autocorrelacion: list[AutocorrelationPoint]
    adf_precio: AdfResult = Field(description="ADF sobre el NIVEL de precio — típicamente NO estacionario")
    adf_retornos: AdfResult = Field(description="ADF sobre los RETORNOS — típicamente SÍ estacionarios")
    halvings_btc: list[str] | None = Field(
        default=None, description="Fechas de halving de Bitcoin (ISO), solo presente si asset='BTC'"
    )
    drawdowns: list[DrawdownEpisode] = Field(description="Los peores drawdowns históricos, más profundo primero")
    fases_mercado: list[MarketPhase] = Field(description="Fases bull/bear delimitadas por un umbral de 20%, orden cronológico")
    ciclos_halving: HalvingCyclesInfo | None = Field(
        default=None, description="Solo presente si asset='BTC' — ciclos entre halvings, con el caveat de n chico"
    )
    heatmap_mensual: MonthlyYearlyHeatmap = Field(description="Retorno compuesto de cada mes de cada año")


class PairScreeningRow(BaseModel):
    """Una fila de `pairs.stability.screen_pairs_stability` (Fase 12b) —
    el ranking honesto de operabilidad de un par, reutilizado tal cual.
    """

    par: str = Field(description='Par, formato "A-B" (orden alfabético, no implica dirección)')
    direccion: str = Field(description='Dirección con mejor fracción cointegrada, formato "Y~X" (Y es el activo dependiente)')
    n_ventanas: int = Field(description="Cantidad de ventanas rolling testeadas")
    fraccion_cointegrada: float = Field(description="Proporción de ventanas con cointegración detectada, en [0, 1]")
    beta_medio: float = Field(description="Media del hedge ratio estático estimado por ventana")
    beta_std: float = Field(description="Desvío del hedge ratio entre ventanas — alto = relación inestable")
    estable: bool = Field(description="True si fraccion_cointegrada >= umbral (0.6 por defecto) — el veredicto operable/no operable")


class PairScreeningResponse(BaseModel):
    """Respuesta de `GET /api/pairs/screening` — reutiliza
    `pairs.stability.screen_pairs_stability` tal cual. Solo opera en velas
    DIARIAS (esa función carga precios sin parámetro de intervalo, y sus
    ventanas de 365/30 observaciones están calibradas para datos diarios) —
    ver el docstring del endpoint.
    """

    interval: str
    filas: list[PairScreeningRow] = Field(description="Ordenado por fraccion_cointegrada descendente (ranking de operabilidad)")
    n_estables: int = Field(description="Cantidad de pares con estable=True")
    n_total: int = Field(description="Cantidad total de pares evaluados")


class PairStabilitySummary(BaseModel):
    """`pairs.stability.stability_summary` para UN par, tal cual (mismos campos que `PairScreeningRow` sin par/dirección)."""

    n_ventanas: int
    fraccion_cointegrada: float
    beta_medio: float
    beta_std: float
    estable: bool


class PairZscoreExtreme(BaseModel):
    """Un punto donde `|z| >= 2` (Fase 15b) — para marcar visualmente los
    extremos históricos del spread en el gráfico del frontend.
    """

    fecha: datetime
    z: float


class PairBacktestMetrics(BaseModel):
    """Métricas de `backtest.engine.run_backtest` (modo `long_short`) o de
    `pairs.backtest.backtest_pair_rotation` (modo `long_only`, Fase 30) —
    misma forma en ambos casos, ver `PairBacktestResult.modo`."""

    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    turnover_total: float
    turnover_medio_diario: float
    n_trades: int
    exposicion_media: float
    pct_tiempo_fuera: float = Field(
        description="Fracción de días con posición efectiva EXACTAMENTE 0 (sin señal, o recién cerrada/warmup)"
    )
    hit_rate: float


class PairBacktestResult(BaseModel):
    """Resultado de `pairs.backtest.backtest_pair` (Fase 15b) o
    `pairs.backtest.backtest_pair_rotation` (Fase 30) sobre ESTE par, según
    `modo` — con sus supuestos (ver el docstring de `pairs/backtest.py`)
    documentados ahí, no acá.

    HONESTIDAD: este backtest existe para PONER A PRUEBA si el par es
    operable, no para recomendar operarlo — sobre un par no establemente
    cointegrado, lo esperable es que muestre pérdidas o resultados
    inestables (ver el texto de la vista "Arbitraje" del frontend).
    """

    modo: str = Field(description='"long_short" (dollar-neutral, backtest_pair) o "long_only" (rotación, backtest_pair_rotation)')
    fechas: list[datetime]
    equity_curve: list[float] = Field(description="Base 1.0")
    metrics: PairBacktestMetrics


class PairDetailResponse(BaseModel):
    """Respuesta de `GET /api/pairs/detail` (Fase 12b, ampliada en 15b) —
    análisis completo de UN par vía `pairs.cointegration.engle_granger`/
    `half_life`, `pairs.stability.rolling_cointegration`/`stability_summary`,
    y `pairs.backtest.backtest_pair`, todo tal cual. Arbitraje ESTADÍSTICO
    (pairs trading), no arbitraje entre exchanges — ver el texto de la
    vista "Arbitraje" del frontend.
    """

    asset_y: str = Field(description="Activo dependiente (\"y\" de la regresión log(y) = alpha + beta*log(x) + spread)")
    asset_x: str
    interval: str
    beta: float = Field(description="Hedge ratio estático")
    alpha: float = Field(description="Intercepto de la regresión")
    estadistico_adf: float
    p_valor_adf: float = Field(description="p-valor del test ADF sobre el spread IN-SAMPLE (toda la muestra a la vez)")
    es_cointegrado: bool = Field(description="Cointegración IN-SAMPLE — ver 'estabilidad' para el veredicto rolling, más honesto")
    half_life_dias: float | None = Field(
        description="Vida media de reversión, en velas del interval pedido. null si el spread no revierte (theta>=0, ver pairs.cointegration.half_life)"
    )
    fechas: list[datetime]
    ratio: list[float | None] = Field(
        default_factory=list,
        description="Fase 30: precio_y / precio_x cruda (sin logaritmo) — 'el tipo de cambio' entre las dos monedas, mismo índice que 'fechas'",
    )
    spread: list[float | None] = Field(description="log(y) - alpha - beta*log(x), residuo de la regresión de cointegración")
    banda_media: list[float | None] = Field(
        default_factory=list, description="Fase 30: media móvil del spread (ventana 'band_window') — el 'centro' de las bandas"
    )
    banda_superior: list[float | None] = Field(
        default_factory=list, description="Fase 30: banda_media + band_n_std * desvío móvil del spread"
    )
    banda_inferior: list[float | None] = Field(
        default_factory=list, description="Fase 30: banda_media - band_n_std * desvío móvil del spread"
    )
    kalman_beta: list[float | None] = Field(
        default_factory=list,
        description="Fase 30: hedge ratio ESTIMADO DÍA A DÍA por pairs.kalman_hedge.kalman_hedge_ratio (vs. 'beta', el único valor estático de toda la muestra)",
    )
    kalman_alpha: list[float | None] = Field(default_factory=list, description="Fase 30: intercepto día a día del mismo filtro de Kalman")
    zscore: list[float | None] = Field(description="Z-score expansivo del spread (pairs.signals.zscore)")
    zscore_actual: float | None = Field(description="Último valor del z-score — qué tan 'estirado' está el spread HOY")
    zscore_interpretacion: str = Field(description="Texto honesto: zona normal vs. extrema (|z|>2), o sin datos suficientes")
    zscore_extremos: list[PairZscoreExtreme] = Field(
        description="Todos los puntos históricos con |z| >= 2 — para marcarlos en el gráfico"
    )
    estabilidad: PairStabilitySummary | None = Field(
        default=None, description="null si no hay historia suficiente para ni una ventana rolling (ver estabilidad_mensaje)"
    )
    estabilidad_mensaje: str | None = Field(default=None, description="Explica por qué 'estabilidad' es null, si aplica")
    backtest: PairBacktestResult = Field(description="Backtest de la estrategia de reversión sobre este par (Fase 15b, modo configurable desde Fase 30)")


class VolumeProfileResponse(BaseModel):
    """Respuesta de `GET /api/volume-profile` (Fase 13a) — reutiliza
    `signals.studies.volume_profile` tal cual. Volume Profile: cuánto
    volumen se operó en cada NIVEL DE PRECIO del período (no en el tiempo,
    como el volumen normal). Herramienta de ANÁLISIS, no predice — ver el
    tooltip del frontend (`helpTexts.ts`): los nodos de alto volumen SUELEN
    actuar como soporte/resistencia, no es una regla garantizada.
    """

    asset: str
    interval: str
    niveles_precio: list[float] = Field(description="Precio (punto medio de cada nivel), ascendente")
    volumenes: list[float] = Field(description="Volumen acumulado por nivel, mismo orden que niveles_precio")
    poc: float = Field(description="Point of Control: el nivel de precio con más volumen del período")
    value_area_low: float = Field(description="Límite inferior del Value Area (rango que concentra ~70% del volumen)")
    value_area_high: float = Field(description="Límite superior del Value Area")
    volumen_total: float = Field(description="Suma de 'volumenes' — para calcular % sobre el total en el frontend")


class CorrelationResponse(BaseModel):
    """Respuesta de `GET /api/correlation` (Fase 13b) — reutiliza
    `eda.eda_report.correlation_matrix` tal cual, sobre los RETORNOS (no
    los precios: la correlación de precios es engañosa por la no
    estacionariedad, ver el docstring del endpoint) alineados por fechas
    comunes (`analysis.comparison.align_common_dates`, Fase 12a).
    """

    interval: str
    method: str = Field(description='"pearson" (lineal) o "spearman" (de rangos, monótona)')
    fechas_n: int = Field(
        description="Cantidad de fechas comunes efectivamente usadas — puede ser menor a 'limit' si no hay tanta historia superpuesta entre todos los activos"
    )
    activos: list[str] = Field(description="Orden de filas/columnas de 'matriz'")
    matriz: list[list[float | None]] = Field(
        description="matriz[i][j] = correlación entre activos[i] y activos[j] — simétrica, diagonal = 1.0. null si es indeterminada (p. ej. 'limit' demasiado chico)"
    )


class RollingCorrelationResponse(BaseModel):
    """Respuesta de `GET /api/correlation/rolling` (Fase 29) — reutiliza
    `eda.eda_report.rolling_pairwise_correlation` tal cual: correlación de
    Pearson en ventana móvil entre DOS activos a lo largo del tiempo, a
    diferencia de `CorrelationResponse` (una única foto estática sobre todo
    el período de las 5 monedas a la vez).

    `correlacion_actual`/`correlacion_promedio_historico` se calculan
    SIEMPRE sobre TODA la historia común disponible entre `asset_a`/
    `asset_b` (no sobre `fechas`/`correlacion`, que sí pueden venir
    recortadas por `limit` para no mandar de más al gráfico) — así
    "actual vs. promedio histórico" compara contra una base estable, sin
    importar cuánta historia se esté graficando en ese momento.
    """

    asset_a: str
    asset_b: str
    interval: str
    window: int = Field(description="Tamaño de la ventana móvil, en velas del intervalo elegido")
    fechas: list[datetime] = Field(description="Fechas comunes a ambos activos, recortadas a 'limit'")
    correlacion: list[float | None] = Field(
        description="Correlación rolling en cada fecha de 'fechas' — null en el warmup (primeras window-1 fechas)"
    )
    correlacion_actual: float | None = Field(
        description="Último valor no-nulo de la correlación rolling sobre TODA la historia común (no solo la recortada a 'limit')"
    )
    correlacion_promedio_historico: float | None = Field(
        description="Promedio de la correlación rolling sobre TODA la historia común disponible — la base contra la que comparar 'correlacion_actual'"
    )


class AssetRiskComparison(BaseModel):
    """Métricas de riesgo/performance de UN activo (Fase 27), calculadas
    sobre la MISMA ventana que `CompareResponse.rendimiento_total_pct` (no
    toda la historia del activo) — reutiliza `metrics.risk_measures` tal
    cual, vía `analysis.comparison.compare_assets`. Junto con el
    rendimiento, permite ver que "quién subió más" y "quién rindió mejor
    ajustado por riesgo" pueden ser respuestas distintas.
    """

    vol_anualizada: float = Field(description="Volatilidad anualizada de los retornos del activo en el período comparado")
    max_drawdown: float = Field(description="Máximo drawdown en el período comparado (negativo o cero)")
    sharpe: float = Field(description="Sharpe ratio (rf=0) de los retornos del activo en el período comparado")


class CompareResponse(BaseModel):
    """Respuesta de `GET /api/compare` (Fase 12a, riesgo agregado en Fase
    27) — reutiliza `analysis.comparison.compare_assets` tal cual.
    Comparación de DESEMPEÑO HISTÓRICO normalizado, no una predicción — el
    desempeño pasado no garantiza el futuro (ver el texto que muestra el
    frontend).
    """

    assets: list[str] = Field(description="Activos pedidos, en el orden en que se comparan")
    interval: str
    fechas: list[datetime] = Field(
        description="Fechas COMUNES a todos los activos (inner join) dentro de la ventana pedida"
    )
    fecha_base: datetime | None = Field(
        description="Primera fecha de 'fechas' — desde cuándo arranca la comparación (todos los activos tienen "
        "dato desde acá). None si no hay ninguna fecha común entre los activos pedidos."
    )
    series: dict[str, list[float | None]] = Field(
        description="Por activo: serie normalizada a base 100 en la primera fecha de 'fechas'"
    )
    rendimiento_total_pct: dict[str, float] = Field(
        description="Por activo: rendimiento total del período, en puntos porcentuales (equivalente a series[activo][-1] - 100)"
    )
    riesgo: dict[str, AssetRiskComparison] = Field(
        description="Por activo: volatilidad anualizada, máximo drawdown y Sharpe del MISMO período que rendimiento_total_pct"
    )
