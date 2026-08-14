"""Workers en segundo plano (`QThread`) del cockpit.

Cada worker corre TODO su cómputo pesado fuera del hilo de la UI y devuelve
el resultado empaquetado en un dataclass vía la señal `resultado_listo` —
nunca toca ningún widget directamente (esa es la regla: el worker calcula,
la ventana principal dibuja, ver `app/__init__.py`).

- `AnalysisWorker` (Fase 4a): precio/vol/GARCH/VaR-ES/señal para el panel de
  riesgo y el panel de señales.
- `BacktestWorker` (Fase 4b): backtest de la estrategia del engine vs. buy &
  hold para el panel de backtest.
- `PredictionWorker` (Fase 5b): predicción OOS del modelo primario de ML
  (enriquecido con on-chain para BTC/ETH) para la pestaña "Predicción (ML)".

Ninguno reimplementa cálculos: cada paso es una llamada directa a una
función ya existente del backend (`data.loaders.get_prices`,
`signals.returns`, `signals.engine`, `models.garch`, `metrics.risk_measures`,
`backtest.engine`, `ml.features`, `ml.labeling`, `ml.models`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from PySide6.QtCore import QThread, Signal

from backtest.engine import backtest_from_prices, compare_to_buy_and_hold
from data.loaders import get_prices
from metrics.risk_measures import expected_shortfall, value_at_risk
from ml.features import align_features_labels, build_feature_matrix
from ml.labeling import get_daily_volatility, triple_barrier_labels
from ml.models import evaluate_primary_with_roc_auc, feature_importances, fit_final, latest_oos_prediction, t1_from_labels
from models.garch import conditional_volatility, select_best_model, volatility_regime
from signals.engine import generate_positions, latest_recommendation
from signals.returns import log_returns, simple_returns

logger = logging.getLogger(__name__)

# Ventana de la media móvil de precio que se muestra en el gráfico (solo
# para el panel; no es la señal del engine, que usa sus propias medias).
PRICE_SMA_WINDOW = 20


@dataclass
class AnalysisResult:
    """Todo lo que necesita la UI para dibujar el panel de un activo — cada
    campo viene directo de una función del backend, ver `AnalysisWorker._analyze`.
    """

    asset: str
    close: pd.Series
    sma: pd.Series
    sma_window: int
    cond_vol: pd.Series
    garch_vol: str
    garch_dist: str
    vol_garch_actual: float
    vol_realizada_actual: float
    regimen_actual: str
    var95: float
    es95: float
    accion: str
    score: float
    tamano_sugerido: float
    desglose: dict[str, float]
    ultima_fecha: pd.Timestamp


class AnalysisWorker(QThread):
    """Ejecuta `AnalysisWorker._analyze` en un hilo separado del de la UI.

    Señales:
    - `resultado_listo(object)`: emite un `AnalysisResult` cuando termina bien.
    - `error(str)`: emite un mensaje si algo falla, en vez de tirar la
      excepción dentro del hilo (donde Qt no la propagaría a la UI).

    Uso: instanciar con el ticker, conectar las señales, llamar `.start()`
    (NUNCA `.run()` directamente — eso lo ejecutaría en el hilo actual,
    anulando el propósito del worker).
    """

    resultado_listo = Signal(object)
    error = Signal(str)

    def __init__(self, asset: str, parent=None) -> None:
        super().__init__(parent)
        self.asset = asset

    def run(self) -> None:
        try:
            resultado = self._analyze(self.asset)
        except Exception as exc:  # noqa: BLE001 - cualquier falla debe llegar a la UI como mensaje, no como crash
            logger.exception("AnalysisWorker: falló el análisis de %s", self.asset)
            self.error.emit(f"No se pudo analizar {self.asset}: {exc}")
            return
        self.resultado_listo.emit(resultado)

    def _analyze(self, asset: str) -> AnalysisResult:
        # 1) Precios (fuente local reproducible, ver Fase 1b).
        df = get_prices(asset, source="store")
        close = df["close"]

        # 2) Retornos: simples para VaR/ES (convención del proyecto),
        #    logarítmicos para GARCH (convención de models.garch).
        returns = simple_returns(close).dropna()
        garch_returns = log_returns(close).dropna()

        # 3) Modelo GARCH ganador + volatilidad condicional + régimen.
        best = select_best_model(garch_returns, criterion="aic")
        cond_vol = conditional_volatility(best["result"])
        regime_series = volatility_regime(cond_vol)
        last_regime = regime_series.iloc[-1]

        # 4) VaR / Expected Shortfall al 95%, históricos (pérdida positiva).
        var95 = value_at_risk(returns, level=0.95)
        es95 = expected_shortfall(returns, level=0.95)

        # 5) Score/acción/tamaño sugerido del engine (Fase 1e). garch_regime=False
        #    porque ya ajustamos el GARCH acá arriba: evita re-ajustarlo dos veces.
        recomendacion = latest_recommendation(df, garch_regime=False)

        sma = close.rolling(window=PRICE_SMA_WINDOW).mean()

        return AnalysisResult(
            asset=asset,
            close=close,
            sma=sma,
            sma_window=PRICE_SMA_WINDOW,
            cond_vol=cond_vol,
            garch_vol=best["vol"],
            garch_dist=best["dist"],
            vol_garch_actual=float(cond_vol.iloc[-1]),
            vol_realizada_actual=float(recomendacion["vol_realizada"]),
            regimen_actual=str(last_regime) if pd.notna(last_regime) else "sin datos",
            var95=float(var95),
            es95=float(es95),
            accion=str(recomendacion["accion"]),
            score=float(recomendacion["score"]),
            tamano_sugerido=float(recomendacion["tamaño_sugerido"]),
            desglose=dict(recomendacion["desglose"]),
            ultima_fecha=close.index[-1],
        )


@dataclass
class BacktestRunResult:
    """Todo lo que necesita `app.widgets.backtest_panel.BacktestPanel` para
    dibujar el backtest de un activo — ver `BacktestWorker._run_backtest`.
    """

    asset: str
    equity_curve_estrategia: pd.Series
    equity_curve_buy_and_hold: pd.Series
    metrics_estrategia: dict[str, float]
    metrics_buy_and_hold: dict[str, float]


class BacktestWorker(QThread):
    """Corre, en un hilo separado, el backtest de la estrategia del engine
    (`signals.engine.generate_positions` -> `backtest.engine.backtest_from_prices`)
    contra buy & hold (`backtest.engine.compare_to_buy_and_hold`) para un activo.

    Mismo patrón que `AnalysisWorker`: señales `resultado_listo(object)` /
    `error(str)`, se arranca con `.start()`, nunca con `.run()` directamente.
    """

    resultado_listo = Signal(object)
    error = Signal(str)

    def __init__(self, asset: str, parent=None) -> None:
        super().__init__(parent)
        self.asset = asset

    def run(self) -> None:
        try:
            resultado = self._run_backtest(self.asset)
        except Exception as exc:  # noqa: BLE001 - cualquier falla debe llegar a la UI como mensaje, no como crash
            logger.exception("BacktestWorker: falló el backtest de %s", self.asset)
            self.error.emit(f"No se pudo correr el backtest de {self.asset}: {exc}")
            return
        self.resultado_listo.emit(resultado)

    def _run_backtest(self, asset: str) -> BacktestRunResult:
        # 1) Precios + posiciones del engine (Fase 1e, sin re-shiftear acá:
        #    el desfase anti-lookahead lo aplica backtest_from_prices).
        df = get_prices(asset, source="store")
        close = df["close"]
        positions = generate_positions(df)

        # 2) Backtest de la estrategia y comparación contra buy & hold, mismo
        #    cost_bps para que la comparación sea justa (Fase 1d).
        result_estrategia = backtest_from_prices(close, positions)
        asset_returns = simple_returns(close)
        comparacion = compare_to_buy_and_hold(asset_returns, result_estrategia)

        # 3) Curva de equity de buy & hold para el gráfico (mismo cost_bps
        #    por defecto que usó compare_to_buy_and_hold internamente).
        result_buy_and_hold = backtest_from_prices(close, 1.0)

        return BacktestRunResult(
            asset=asset,
            equity_curve_estrategia=result_estrategia.equity_curve,
            equity_curve_buy_and_hold=result_buy_and_hold.equity_curve,
            metrics_estrategia=comparacion["estrategia"],
            metrics_buy_and_hold=comparacion["buy_and_hold"],
        )


# Activos con cobertura on-chain suficiente en CoinMetrics Community para
# alimentar el modelo enriquecido (BTC/ETH tienen el set completo, incluidos
# flujos de exchange — ver `data/onchain.py` y el resumen de
# `scripts/export_onchain.py`). SOL no tiene nada; BNB/LTC tienen cobertura
# PARCIAL y, en el caso de BNB, un histórico on-chain que corta en 2019 —
# demasiado viejo para alimentar una predicción de HOY. Para esos tres, la
# pestaña de predicción usa solo features técnicas y lo avisa en la UI.
ONCHAIN_ENABLED_ASSETS: frozenset[str] = frozenset({"BTC", "ETH"})

# n_splits/embargo de la validación purgeada que corre esta pestaña — los
# mismos usados en la validación de Fase 5b (ver ml/models.py).
_PREDICTION_N_SPLITS = 5
_PREDICTION_EMBARGO_PCT = 0.01
_TOP_N_FEATURES = 8

_CLASS_DISPLAY: dict[float, str] = {-1.0: "SHORT", 0.0: "FLAT", 1.0: "LONG"}


@dataclass
class PredictionResult:
    """Todo lo que necesita `app.widgets.prediction_panel.PredictionPanel`
    para pintar la pestaña "Predicción (ML)" — ver `PredictionWorker._predict`.
    """

    asset: str
    used_onchain: bool
    onchain_columns: list[str]
    ultima_fecha: pd.Timestamp
    prediccion_clase: str
    prediccion_confianza: float
    prediccion_proba: dict[str, float]
    accuracy_media: float
    baseline_azar: float
    baseline_mayoritaria: float
    supera_azar: bool
    supera_mayoritaria: bool
    roc_auc_media: float
    top_features: list[tuple[str, float]]


class PredictionWorker(QThread):
    """Ejecuta, en un hilo separado, el modelo primario de ML (Fase 3c) —
    enriquecido con on-chain para BTC/ETH (Fase 5b, `ONCHAIN_ENABLED_ASSETS`),
    solo técnico para el resto — y arma una `PredictionResult`: la
    predicción OOS honesta más reciente (con probabilidad real, no un
    proxy), si el modelo le gana a los baselines triviales, y qué features
    pesan más.

    No reimplementa NINGÚN cálculo de ML: cada paso es una llamada directa
    a `ml.features`/`ml.labeling`/`ml.models`, reutilizados tal cual.
    Mismo patrón que `AnalysisWorker`/`BacktestWorker`: señales
    `resultado_listo(object)` / `error(str)`, se arranca con `.start()`.
    """

    resultado_listo = Signal(object)
    error = Signal(str)

    def __init__(self, asset: str, parent=None) -> None:
        super().__init__(parent)
        self.asset = asset

    def run(self) -> None:
        try:
            resultado = self._predict(self.asset)
        except Exception as exc:  # noqa: BLE001 - cualquier falla debe llegar a la UI como mensaje, no como crash
            logger.exception("PredictionWorker: falló la predicción de %s", self.asset)
            self.error.emit(f"No se pudo predecir {self.asset}: {exc}")
            return
        self.resultado_listo.emit(resultado)

    def _predict(self, asset: str) -> PredictionResult:
        use_onchain = asset in ONCHAIN_ENABLED_ASSETS

        # 1) Precios + matriz de features (técnica, o técnica+on-chain).
        df = get_prices(asset, source="store")
        close = df["close"]

        X_raw = build_feature_matrix(df, include_onchain=use_onchain, asset=asset if use_onchain else None)
        onchain_columns: list[str] = []
        if use_onchain:
            X_technical_raw = build_feature_matrix(df, include_onchain=False)
            onchain_columns = [c for c in X_raw.columns if c not in X_technical_raw.columns]

        # 2) Target triple-barrier + alineación (Fase 3a) + t1 para la
        #    validación purgeada (Fase 3b).
        volatility = get_daily_volatility(close)
        labels_df = triple_barrier_labels(close, volatility)
        X, y, sample_weight = align_features_labels(X_raw, labels_df)
        t1 = t1_from_labels(labels_df, X.index)

        # 3) Evaluación purgeada (accuracy/F1/ROC-AUC vs. baselines, Fase 3c/5b).
        evaluacion = evaluate_primary_with_roc_auc(
            X, y, t1, sample_weight=sample_weight,
            n_splits=_PREDICTION_N_SPLITS, embargo_pct=_PREDICTION_EMBARGO_PCT,
        )

        # 4) Predicción OOS honesta (con probabilidad real) para la última fecha.
        prediccion = latest_oos_prediction(
            X, y, t1, sample_weight=sample_weight,
            n_splits=_PREDICTION_N_SPLITS, embargo_pct=_PREDICTION_EMBARGO_PCT,
        )

        # 5) Feature importances: ajuste FINAL sobre toda la data (Fase 3c,
        #    `fit_final`) — solo para interpretar qué pesa más, NUNCA para
        #    predecir (esa es la predicción OOS del paso 4).
        model = fit_final(X, y, sample_weight=sample_weight)
        importances = feature_importances(model, X.columns)
        top_features = [(str(name), float(val)) for name, val in importances.head(_TOP_N_FEATURES).items()]

        proba_por_texto = {_CLASS_DISPLAY[k]: v for k, v in prediccion["proba"].items()}

        return PredictionResult(
            asset=asset,
            used_onchain=use_onchain,
            onchain_columns=onchain_columns,
            ultima_fecha=prediccion["fecha"],
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
