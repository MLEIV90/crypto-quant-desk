"""Ventana principal de la terminal de crypto-quant-desk (el "cockpit").

Solo arma la UI y orquesta a los workers de `app.workers`: dispara
`AnalysisWorker` al hacer clic en "Analizar" y, cuando termina (por señal,
nunca por espera bloqueante), vuelca su `AnalysisResult` a los widgets de
gráficos, al panel de riesgo (pestaña "Riesgo", Fase 4a) y al panel de
señales (pestaña "Señales", Fase 4b) — mismo resultado, dos vistas, un solo
cómputo. Las pestañas "Backtest" (Fase 4b) y "Predicción (ML)" (Fase 5b)
son independientes: cada una tiene su propio selector/botón porque dispara
su propio worker (`BacktestWorker`/`PredictionWorker`), un cómputo aparte
del análisis de riesgo/señales. Ningún cálculo de negocio vive acá — ver
`app/__init__.py` para la regla de separación modelo/vista.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.theme import STYLESHEET  # noqa: E402
from app.widgets.backtest_panel import BacktestPanel  # noqa: E402
from app.widgets.plot_canvas import PriceCanvas, VolatilityCanvas  # noqa: E402
from app.widgets.prediction_panel import PredictionPanel  # noqa: E402
from app.widgets.risk_panel import RiskPanel  # noqa: E402
from app.widgets.signals_panel import SignalsPanel  # noqa: E402
from app.workers import AnalysisWorker  # noqa: E402
from config import UNIVERSE  # noqa: E402

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "Herramienta de análisis y gestión de riesgo. No es asesoramiento financiero "
    "ni predicción de precio."
)


class MainWindow(QMainWindow):
    """Ventana principal: selector de activo + botón "Analizar" + gráficos
    + panel de riesgo, con un `AnalysisWorker` corriendo el cómputo pesado
    en segundo plano.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("crypto-quant-desk — Terminal de riesgo")
        self.resize(1100, 720)
        self.setStyleSheet(STYLESHEET)

        self._worker: AnalysisWorker | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        root_layout.addLayout(self._build_header())

        disclaimer = QLabel(DISCLAIMER_TEXT)
        disclaimer.setObjectName("disclaimer")
        disclaimer.setWordWrap(True)
        root_layout.addWidget(disclaimer)

        root_layout.addWidget(self._build_tabs())

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()

        header.addWidget(QLabel("Activo:"))

        self.asset_combo = QComboBox()
        self.asset_combo.addItems(list(UNIVERSE.keys()))
        header.addWidget(self.asset_combo)

        self.analyze_button = QPushButton("Analizar")
        self.analyze_button.clicked.connect(self._on_analyze_clicked)
        header.addWidget(self.analyze_button)

        header.addStretch()

        self.status_label = QLabel("Listo.")
        self.status_label.setObjectName("statusLabel")
        header.addWidget(self.status_label)

        return header

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_risk_tab(), "Riesgo")

        self.signals_panel = SignalsPanel()
        self.tabs.addTab(self.signals_panel, "Señales")

        self.backtest_panel = BacktestPanel()
        self.tabs.addTab(self.backtest_panel, "Backtest")

        self.prediction_panel = PredictionPanel()
        self.tabs.addTab(self.prediction_panel, "Predicción (ML)")

        return self.tabs

    def _build_risk_tab(self) -> QWidget:
        """Pestaña "Riesgo" (Fase 4a): gráficos de precio/volatilidad +
        panel de riesgo, alimentados por `AnalysisWorker`.
        """
        risk_tab = QWidget()
        content = QHBoxLayout(risk_tab)

        plots_layout = QVBoxLayout()
        self.price_canvas = PriceCanvas()
        self.vol_canvas = VolatilityCanvas()
        plots_layout.addWidget(self.price_canvas)
        plots_layout.addWidget(self.vol_canvas)
        content.addLayout(plots_layout, stretch=3)

        self.risk_panel = RiskPanel()
        content.addWidget(self.risk_panel, stretch=1)

        return risk_tab

    # ------------------------------------------------------------------
    # Disparo del worker y manejo de resultados
    # ------------------------------------------------------------------

    def _on_analyze_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return  # ya hay un análisis en curso; no se apilan workers

        asset = self.asset_combo.currentText()
        self.analyze_button.setEnabled(False)
        self.asset_combo.setEnabled(False)
        self.status_label.setText(f"Calculando {asset}...")

        self._worker = AnalysisWorker(asset)
        self._worker.resultado_listo.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_analysis_finished(self, resultado) -> None:
        self.status_label.setText(f"{resultado.asset} — datos al {resultado.ultima_fecha.date()}")
        self.price_canvas.plot(resultado)
        self.vol_canvas.plot(resultado)
        self.risk_panel.update_values(resultado)
        self.signals_panel.update_values(resultado)

    def _on_analysis_error(self, mensaje: str) -> None:
        self.status_label.setText("Error en el análisis.")
        self.risk_panel.reset()
        self.signals_panel.reset()
        QMessageBox.critical(self, "Error al analizar", mensaje)

    def _on_worker_finished(self) -> None:
        self.analyze_button.setEnabled(True)
        self.asset_combo.setEnabled(True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
