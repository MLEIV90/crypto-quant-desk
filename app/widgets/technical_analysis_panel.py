"""Pestaña "Análisis Técnico" (Fase 7b): selector de activo + timeframe +
botón "Analizar", gráfico de velas con overlays/sub-paneles
(`app.widgets.technical_chart.TechnicalChartCanvas`) y el panel del
sugeridor de consenso (`app.widgets.suggester_panel.SuggesterPanel`), todo
alimentado por un único `app.workers.StudiesWorker`.

Tiene su propio selector/botón (independiente del de arriba, igual que
`BacktestPanel`/`PredictionPanel`), porque dispara su propio worker — ver
`app/__init__.py` para la regla de separación modelo/vista y el patrón de
worker.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.suggester_panel import SuggesterPanel
from app.widgets.technical_chart import TechnicalChartCanvas
from app.workers import StudiesWorker
from config import UNIVERSE

TIMEFRAMES: list[tuple[str, str]] = [("Diario (1d)", "1d"), ("Horario (1h)", "1h")]

HONESTY_TEXT = (
    "Estudios técnicos clásicos (medias, RSI, MACD, estocástico, Bollinger, Fibonacci, soporte/"
    "resistencia, pivotes) para ANÁLISIS y apoyo a tu propia decisión. El proyecto ya demostró que "
    "estas señales NO tienen edge direccional automático (ver ml/models.py) — incluyen incluso "
    "estudios sin respaldo estadístico fuerte (Fibonacci) porque se pidieron para análisis visual. "
    "El sugeridor de la derecha muestra siempre su desempeño histórico real al lado, sin ocultarlo."
)


class TechnicalAnalysisPanel(QWidget):
    """Selector de activo + timeframe + botón "Analizar" + gráfico de velas
    + panel del sugeridor, todo alimentado por un `StudiesWorker`.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("technicalAnalysisPanel")
        self._worker: StudiesWorker | None = None

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Activo:"))

        self.asset_combo = QComboBox()
        self.asset_combo.addItems(list(UNIVERSE.keys()))
        header.addWidget(self.asset_combo)

        header.addWidget(QLabel("Timeframe:"))

        self.timeframe_combo = QComboBox()
        for label, _ in TIMEFRAMES:
            self.timeframe_combo.addItem(label)
        header.addWidget(self.timeframe_combo)

        self.analyze_button = QPushButton("Analizar")
        self.analyze_button.clicked.connect(self._on_analyze_clicked)
        header.addWidget(self.analyze_button)

        header.addStretch()

        self.status_label = QLabel("Listo.")
        self.status_label.setObjectName("statusLabel")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        warning = QLabel(HONESTY_TEXT)
        warning.setObjectName("honestyWarning")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        content = QHBoxLayout()

        self.chart_canvas = TechnicalChartCanvas()
        content.addWidget(self.chart_canvas, stretch=3)

        self.suggester_panel = SuggesterPanel()
        content.addWidget(self.suggester_panel, stretch=1)

        layout.addLayout(content)

    # ------------------------------------------------------------------
    # Disparo del worker y manejo de resultados
    # ------------------------------------------------------------------

    def _selected_timeframe(self) -> str:
        return TIMEFRAMES[self.timeframe_combo.currentIndex()][1]

    def _on_analyze_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return  # ya hay un análisis en curso; no se apilan workers

        asset = self.asset_combo.currentText()
        timeframe = self._selected_timeframe()
        self.analyze_button.setEnabled(False)
        self.asset_combo.setEnabled(False)
        self.timeframe_combo.setEnabled(False)
        self.status_label.setText(f"Calculando {asset} ({timeframe})...")

        self._worker = StudiesWorker(asset, timeframe)
        self._worker.resultado_listo.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_finished(self, resultado) -> None:
        self.status_label.setText(
            f"{resultado.asset} ({resultado.timeframe}) — última vela: {resultado.ultima_fecha}"
        )
        self.chart_canvas.plot(resultado)
        self.suggester_panel.update_values(resultado)

    def _on_error(self, mensaje: str) -> None:
        self.status_label.setText("Error en el análisis técnico.")
        self.suggester_panel.reset()
        QMessageBox.critical(self, "Error al analizar", mensaje)

    def _on_worker_finished(self) -> None:
        self.analyze_button.setEnabled(True)
        self.asset_combo.setEnabled(True)
        self.timeframe_combo.setEnabled(True)
