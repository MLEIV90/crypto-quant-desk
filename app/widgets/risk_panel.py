"""Panel de tarjetas con las métricas de riesgo/decisión de la última foto.

Solo muestra valores ya calculados en un `app.workers.AnalysisResult` — no
calcula nada acá (ver `app/__init__.py`, regla de separación modelo/vista).
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget


class RiskPanel(QWidget):
    """Tarjetas: vol. realizada, modelo GARCH ganador, vol. GARCH, régimen,
    VaR95, ES95, señal del engine y tamaño sugerido — con la aclaración
    explícita de que son medidas de riesgo/decisión, no una predicción.
    """

    _FIELDS: list[tuple[str, str]] = [
        ("vol_realizada", "Vol. realizada anualizada"),
        ("modelo_garch", "Modelo GARCH ganador"),
        ("vol_garch", "Vol. condicional GARCH"),
        ("regimen", "Régimen de volatilidad"),
        ("var95", "VaR 95% (pérdida diaria)"),
        ("es95", "Expected Shortfall 95%"),
        ("accion", "Señal del engine"),
        ("tamano_sugerido", "Tamaño sugerido (vol targeting)"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("riskPanel")

        layout = QVBoxLayout(self)

        title = QLabel("Panel de riesgo")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        note = QLabel(
            "Estas son medidas de RIESGO y de SIZING calculadas sobre la historia — "
            "no son una predicción de precio."
        )
        note.setObjectName("panelNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        grid = QGridLayout()
        self._value_labels: dict[str, QLabel] = {}
        for row, (key, label_text) in enumerate(self._FIELDS):
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
        """Vuelca un `app.workers.AnalysisResult` a las tarjetas."""
        self._value_labels["vol_realizada"].setText(f"{resultado.vol_realizada_actual:.2%}")
        self._value_labels["modelo_garch"].setText(f"{resultado.garch_vol} / {resultado.garch_dist}")
        self._value_labels["vol_garch"].setText(f"{resultado.vol_garch_actual:.2%}")
        self._value_labels["regimen"].setText(resultado.regimen_actual.upper())
        self._value_labels["var95"].setText(f"{resultado.var95:.2%}")
        self._value_labels["es95"].setText(f"{resultado.es95:.2%}")
        self._value_labels["accion"].setText(f"{resultado.accion} (score {resultado.score:+.2f})")
        self._value_labels["tamano_sugerido"].setText(f"{resultado.tamano_sugerido:+.2f}x")

    def reset(self) -> None:
        """Vuelve todas las tarjetas a su estado vacío (p. ej. al empezar
        un nuevo análisis o si el anterior falló).
        """
        for label in self._value_labels.values():
            label.setText("—")
