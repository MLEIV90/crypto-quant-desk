"""Pestaña "Research (sin edge)": corre el modelo primario (Fase 3c) —
enriquecido con on-chain para BTC/ETH (Fase 5b) — en un `QThread`
(`app.workers.PredictionWorker`) y muestra la predicción OOS más reciente,
si le gana o no a los baselines triviales, y qué features pesan más.

CONSOLIDACIÓN DE FASE 7b: esta pestaña era "Predicción (ML)" y vivía en la
barra principal del cockpit. Se movió acá, al final, y se renombró
explícitamente porque el proyecto NUNCA encontró un modelo que le gane de
forma consistente a los baselines triviales (ver `ml/models.py`,
Fases 3c/5b/6a/6c) — mantenerla junto a herramientas operativas (Riesgo,
Backtest, Análisis Técnico) sugeriría un edge que no existe. Es
investigación con resultado negativo, documentada con la misma honestidad
que el resto del proyecto — no una herramienta para operar.

Tiene su propio selector de activo y botón (independiente del de arriba,
igual que `BacktestPanel`), porque dispara un cómputo propio (varios
ajustes de XGBoost con validación purgeada) distinto del análisis de
riesgo/señales de las otras pestañas — ver `app/__init__.py` para la regla
de separación modelo/vista y el patrón de worker.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
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

from app.theme import DIRECTION_COLORS
from app.workers import ONCHAIN_ENABLED_ASSETS, PredictionWorker
from config import UNIVERSE

DISCLAIMER_TEXT = (
    "Herramienta de análisis y gestión de riesgo. No es asesoramiento financiero "
    "ni predicción de precio."
)

HONESTY_TEXT = (
    "INVESTIGACIÓN CON RESULTADO NEGATIVO, NO HERRAMIENTA OPERATIVA: este modelo de ML "
    "NUNCA superó de forma consistente a los baselines triviales (azar / clase mayoritaria) "
    "en ninguna validación del proyecto. Predicción OOS (out-of-sample, validación purgeada — "
    "nunca vio la fecha que predice) del modelo primario. Para BTC/ETH usa features técnicas + "
    "on-chain (flujos de exchange, MVRV, direcciones activas); para SOL/BNB/LTC, solo técnicas — "
    "sin cobertura on-chain suficiente en CoinMetrics. Se muestra tal cual sale, sin maquillar — "
    "no es una recomendación de operar."
)

_CLASS_ORDER = ["LONG", "FLAT", "SHORT"]
_SEMAFORO_STYLE = "font-size: 26pt; color: {color};"


class PredictionPanel(QWidget):
    """Selector de activo + botón "Predecir" + semáforo de la predicción +
    desglose de probabilidades + comparación contra baselines + ranking de
    feature importances.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("predictionPanel")
        self._worker: PredictionWorker | None = None

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Activo:"))

        self.asset_combo = QComboBox()
        self.asset_combo.addItems(list(UNIVERSE.keys()))
        header.addWidget(self.asset_combo)

        self.predict_button = QPushButton("Predecir")
        self.predict_button.clicked.connect(self._on_predict_clicked)
        header.addWidget(self.predict_button)

        header.addStretch()

        self.status_label = QLabel("Listo.")
        self.status_label.setObjectName("statusLabel")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        warning = QLabel(HONESTY_TEXT)
        warning.setObjectName("honestyWarning")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        semaforo_row = QHBoxLayout()
        self.semaforo_dot = QLabel("●")
        self.semaforo_dot.setObjectName("semaforoDot")
        self.semaforo_dot.setStyleSheet(_SEMAFORO_STYLE.format(color=DIRECTION_COLORS["FLAT"]))
        semaforo_row.addWidget(self.semaforo_dot)

        self.clase_label = QLabel("—")
        self.clase_label.setObjectName("accionLabel")
        semaforo_row.addWidget(self.clase_label)

        self.fecha_label = QLabel("")
        self.fecha_label.setObjectName("cardName")
        semaforo_row.addWidget(self.fecha_label)
        semaforo_row.addStretch()
        layout.addLayout(semaforo_row)

        content = QHBoxLayout()

        left_grid = QGridLayout()
        self._proba_labels: dict[str, QLabel] = {}
        for row, clase in enumerate(_CLASS_ORDER):
            name_label = QLabel(f"P({clase})")
            name_label.setObjectName("cardName")
            value_label = QLabel("—")
            value_label.setObjectName("cardValue")
            left_grid.addWidget(name_label, row, 0)
            left_grid.addWidget(value_label, row, 1)
            self._proba_labels[clase] = value_label

        self._metric_fields: list[tuple[str, str]] = [
            ("accuracy", "Accuracy purgeada"),
            ("baseline_azar", "Baseline azar"),
            ("baseline_mayoritaria", "Baseline mayoritaria"),
            ("roc_auc", "ROC-AUC (ovr)"),
            ("supera_azar", "¿Supera azar?"),
            ("supera_mayoritaria", "¿Supera mayoritaria?"),
            ("onchain", "On-chain usado"),
        ]
        self._metric_labels: dict[str, QLabel] = {}
        offset = len(_CLASS_ORDER) + 1
        for i, (key, label_text) in enumerate(self._metric_fields):
            row = offset + i
            name_label = QLabel(label_text)
            name_label.setObjectName("cardName")
            value_label = QLabel("—")
            value_label.setObjectName("cardValue")
            left_grid.addWidget(name_label, row, 0)
            left_grid.addWidget(value_label, row, 1)
            self._metric_labels[key] = value_label

        content.addLayout(left_grid, stretch=1)

        self.features_table = QTableWidget(0, 2)
        self.features_table.setHorizontalHeaderLabels(["Feature", "Importancia"])
        self.features_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.features_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.features_table.setSelectionMode(QTableWidget.NoSelection)
        self.features_table.setMinimumHeight(260)
        content.addWidget(self.features_table, stretch=1)

        layout.addLayout(content)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Disparo del worker y manejo de resultados
    # ------------------------------------------------------------------

    def _on_predict_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return  # ya hay una predicción en curso; no se apilan workers

        asset = self.asset_combo.currentText()
        self.predict_button.setEnabled(False)
        self.asset_combo.setEnabled(False)
        onchain_hint = "con on-chain" if asset in ONCHAIN_ENABLED_ASSETS else "solo técnicas"
        self.status_label.setText(f"Calculando {asset} ({onchain_hint})...")

        self._worker = PredictionWorker(asset)
        self._worker.resultado_listo.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_finished(self, resultado) -> None:
        self.status_label.setText(f"{resultado.asset} — última fecha con datos: {resultado.ultima_fecha.date()}")
        self.fecha_label.setText(f"({resultado.ultima_fecha.date()})")

        self.clase_label.setText(f"{resultado.prediccion_clase}  ({resultado.prediccion_confianza:.1%} conf.)")
        color = DIRECTION_COLORS.get(resultado.prediccion_clase, DIRECTION_COLORS["FLAT"])
        self.semaforo_dot.setStyleSheet(_SEMAFORO_STYLE.format(color=color))

        for clase in _CLASS_ORDER:
            self._proba_labels[clase].setText(f"{resultado.prediccion_proba.get(clase, 0.0):.1%}")

        self._metric_labels["accuracy"].setText(f"{resultado.accuracy_media:.1%}")
        self._metric_labels["baseline_azar"].setText(f"{resultado.baseline_azar:.1%}")
        self._metric_labels["baseline_mayoritaria"].setText(f"{resultado.baseline_mayoritaria:.1%}")
        self._metric_labels["roc_auc"].setText(f"{resultado.roc_auc_media:.3f}")
        self._metric_labels["supera_azar"].setText("SÍ" if resultado.supera_azar else "NO")
        self._metric_labels["supera_mayoritaria"].setText("SÍ" if resultado.supera_mayoritaria else "NO")
        onchain_text = f"SÍ ({len(resultado.onchain_columns)} cols.)" if resultado.used_onchain else "NO"
        self._metric_labels["onchain"].setText(onchain_text)

        self.features_table.setRowCount(len(resultado.top_features))
        for row, (name, importancia) in enumerate(resultado.top_features):
            self.features_table.setItem(row, 0, QTableWidgetItem(name))
            self.features_table.setItem(row, 1, QTableWidgetItem(f"{importancia:.4f}"))

    def _on_error(self, mensaje: str) -> None:
        self.status_label.setText("Error en la predicción.")
        QMessageBox.critical(self, "Error al predecir", mensaje)

    def _on_worker_finished(self) -> None:
        self.predict_button.setEnabled(True)
        self.asset_combo.setEnabled(True)
