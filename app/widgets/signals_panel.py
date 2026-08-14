"""Panel de señales: desglose del engine + "semáforo" del score compuesto.

Solo muestra valores ya calculados en un `app.workers.AnalysisResult`
(componentes trend/momentum/mean_reversion, score, accion) — no calcula nada
acá (ver `app/__init__.py`, regla de separación modelo/vista).

El texto de honestidad de abajo (`HONESTY_TEXT`) es fijo y no depende del
resultado: el proyecto (Fase 1e/3c) no encontró edge direccional
estadísticamente demostrado en estas señales técnicas, así que el panel las
muestra como insumo de análisis, nunca como recomendación de operar.
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

HONESTY_TEXT = (
    "Las señales técnicas de este engine NO demostraron edge direccional "
    "estadísticamente significativo en la validación del proyecto. Se muestran "
    "como INSUMO de análisis — desglose de indicadores y score compuesto — "
    "nunca como una recomendación de operar."
)

SEMAFORO_COLORS: dict[str, str] = {
    "LONG": "#22c55e",
    "FLAT": "#94a3b8",
    "SHORT": "#ef4444",
}
_SEMAFORO_STYLE = "font-size: 26pt; color: {color};"


class SignalsPanel(QWidget):
    """Semáforo (LONG=verde / FLAT=gris / SHORT=rojo) + desglose de los tres
    componentes del score compuesto + el aviso de honestidad del engine.
    """

    _COMPONENT_FIELDS: list[tuple[str, str]] = [
        ("trend", "Tendencia"),
        ("momentum", "Momentum"),
        ("mean_reversion", "Reversión a la media"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("signalsPanel")

        layout = QVBoxLayout(self)

        title = QLabel("Panel de señales")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        warning = QLabel(HONESTY_TEXT)
        warning.setObjectName("honestyWarning")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        semaforo_row = QHBoxLayout()
        self.semaforo_dot = QLabel("●")
        self.semaforo_dot.setObjectName("semaforoDot")
        self.semaforo_dot.setStyleSheet(_SEMAFORO_STYLE.format(color=SEMAFORO_COLORS["FLAT"]))
        semaforo_row.addWidget(self.semaforo_dot)

        self.accion_label = QLabel("—")
        self.accion_label.setObjectName("accionLabel")
        semaforo_row.addWidget(self.accion_label)

        self.score_label = QLabel("score: —")
        self.score_label.setObjectName("cardName")
        semaforo_row.addWidget(self.score_label)
        semaforo_row.addStretch()
        layout.addLayout(semaforo_row)

        grid = QGridLayout()
        self._value_labels: dict[str, QLabel] = {}
        for row, (key, label_text) in enumerate(self._COMPONENT_FIELDS):
            name_label = QLabel(label_text)
            name_label.setObjectName("cardName")
            value_label = QLabel("—")
            value_label.setObjectName("cardValue")
            grid.addWidget(name_label, row, 0)
            grid.addWidget(value_label, row, 1)
            self._value_labels[key] = value_label
        layout.addLayout(grid)
        layout.addStretch()

    def update_values(self, resultado) -> None:
        """Vuelca un `app.workers.AnalysisResult` al semáforo y al desglose."""
        self.accion_label.setText(resultado.accion)
        self.score_label.setText(f"score: {resultado.score:+.3f}")
        color = SEMAFORO_COLORS.get(resultado.accion, SEMAFORO_COLORS["FLAT"])
        self.semaforo_dot.setStyleSheet(_SEMAFORO_STYLE.format(color=color))
        for key, _ in self._COMPONENT_FIELDS:
            self._value_labels[key].setText(f"{resultado.desglose[key]:+.3f}")

    def reset(self) -> None:
        """Vuelve el panel a su estado vacío (nuevo análisis o error)."""
        self.accion_label.setText("—")
        self.score_label.setText("score: —")
        self.semaforo_dot.setStyleSheet(_SEMAFORO_STYLE.format(color=SEMAFORO_COLORS["FLAT"]))
        for key, _ in self._COMPONENT_FIELDS:
            self._value_labels[key].setText("—")
