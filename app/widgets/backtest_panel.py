"""Panel de backtest: corre la estrategia del engine contra buy & hold en un
`QThread` (`app.workers.BacktestWorker`) y muestra la curva de equity + una
tabla de métricas lado a lado.

Tiene su propio selector de activo y botón (independiente del de arriba),
porque dispara un cómputo propio (backtest completo) distinto del análisis
de riesgo/señales de las otras pestañas — ver `app/__init__.py` para la
regla de separación modelo/vista y el patrón de worker.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.widgets.plot_canvas import EquityCurveCanvas
from app.workers import BacktestWorker
from config import UNIVERSE

DISCLAIMER_TEXT = (
    "Los resultados incluyen costos de transacción (ver config.TRANSACTION_COST_BPS). "
    "El desempeño pasado NO garantiza resultados futuros."
)

# (clave en el dict de métricas, etiqueta de fila, formato de valor)
_METRIC_ROWS: list[tuple[str, str, str]] = [
    ("cagr", "CAGR", "{:.2%}"),
    ("sharpe", "Sharpe", "{:.2f}"),
    ("sortino", "Sortino", "{:.2f}"),
    ("max_drawdown", "Max drawdown", "{:.2%}"),
    ("calmar", "Calmar", "{:.2f}"),
    ("n_trades", "N° de trades", "{:.0f}"),
    ("turnover_total", "Turnover total", "{:.2f}"),
]


class BacktestPanel(QWidget):
    """Selector de activo + botón "Correr backtest" + gráfico de equity +
    tabla comparativa estrategia vs. buy & hold.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("backtestPanel")
        self._worker: BacktestWorker | None = None

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Activo:"))

        self.asset_combo = QComboBox()
        self.asset_combo.addItems(list(UNIVERSE.keys()))
        header.addWidget(self.asset_combo)

        self.run_button = QPushButton("Correr backtest")
        self.run_button.clicked.connect(self._on_run_clicked)
        header.addWidget(self.run_button)

        header.addStretch()

        self.status_label = QLabel("Listo.")
        self.status_label.setObjectName("statusLabel")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        disclaimer = QLabel(DISCLAIMER_TEXT)
        disclaimer.setObjectName("panelNote")
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)

        self.equity_canvas = EquityCurveCanvas()
        layout.addWidget(self.equity_canvas, stretch=2)

        self.metrics_table = QTableWidget(len(_METRIC_ROWS), 2)
        self.metrics_table.setHorizontalHeaderLabels(["Estrategia", "Buy & hold"])
        self.metrics_table.setVerticalHeaderLabels([label for _, label, _ in _METRIC_ROWS])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionMode(QTableWidget.NoSelection)
        self.metrics_table.setMinimumHeight(240)
        layout.addWidget(self.metrics_table, stretch=1)

    # ------------------------------------------------------------------
    # Disparo del worker y manejo de resultados
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return  # ya hay un backtest en curso; no se apilan workers

        asset = self.asset_combo.currentText()
        self.run_button.setEnabled(False)
        self.asset_combo.setEnabled(False)
        self.status_label.setText(f"Corriendo backtest de {asset}...")

        self._worker = BacktestWorker(asset)
        self._worker.resultado_listo.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_finished(self, resultado) -> None:
        self.status_label.setText(f"{resultado.asset} — backtest completo")
        self.equity_canvas.plot(resultado)
        for row, (key, _, fmt) in enumerate(_METRIC_ROWS):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(fmt.format(resultado.metrics_estrategia[key])))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(fmt.format(resultado.metrics_buy_and_hold[key])))

    def _on_error(self, mensaje: str) -> None:
        self.status_label.setText("Error en el backtest.")
        QMessageBox.critical(self, "Error al correr backtest", mensaje)

    def _on_worker_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.asset_combo.setEnabled(True)
