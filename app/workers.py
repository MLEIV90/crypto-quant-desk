"""Worker en segundo plano (`QThread`) para el análisis de un activo.

Corre TODO el cómputo pesado (descarga/lectura de precios, ajuste GARCH,
VaR/ES, sizing) fuera del hilo de la UI, y devuelve el resultado empaquetado
en un `AnalysisResult` vía la señal `resultado_listo` — nunca toca ningún
widget directamente (esa es la regla: el worker calcula, la ventana
principal dibuja, ver `app/__init__.py`).

No reimplementa NINGÚN cálculo: cada paso de `AnalysisWorker._analyze` es
una llamada directa a una función ya existente del backend
(`data.loaders.get_prices`, `signals.returns`, `models.garch`,
`metrics.risk_measures`, `signals.engine`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from PySide6.QtCore import QThread, Signal

from data.loaders import get_prices
from metrics.risk_measures import expected_shortfall, value_at_risk
from models.garch import conditional_volatility, select_best_model, volatility_regime
from signals.engine import latest_recommendation
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
            ultima_fecha=close.index[-1],
        )
