"""Panel lateral del sugeridor de consenso (Fase 7b): la pestaña "Análisis
Técnico" muestra acá la lectura de `signals.suggester.suggest` para el
activo/timeframe elegido — la sugerencia, el desglose de votos por
estudio, y el desempeño histórico de esa misma regla SIEMPRE visible al
lado (nunca se muestra la sugerencia sin su historial real, ver
`signals/suggester.py` — este panel solo dibuja lo que ya viene calculado
en un `app.workers.StudiesResult`, no calcula nada).
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from app.theme import DIRECTION_COLORS, TEXT_MUTED

_SUGGESTION_STYLE = "font-size: 22pt; font-weight: bold; color: {color};"

_STUDY_LABELS: dict[str, str] = {
    "rsi": "RSI",
    "medias": "Medias móviles",
    "macd": "MACD",
    "estocastico": "Estocástico",
    "bollinger": "Bollinger",
    "pivotes": "Pivote",
}

_VOTE_TEXT: dict[str, str] = {"alcista": "▲ alcista", "bajista": "▼ bajista", "neutral": "— neutral"}
_VOTE_COLOR: dict[str, str] = {
    "alcista": DIRECTION_COLORS["COMPRAR"], "bajista": DIRECTION_COLORS["VENDER"], "neutral": TEXT_MUTED,
}

DEFAULT_PERFORMANCE_TEXT = "Corré un análisis para ver el desempeño histórico de esta regla."

PERFORMANCE_TEMPLATE = (
    "Esta sugerencia es un CONSENSO DE INDICADORES — no una señal con edge demostrado (ver "
    "signals/suggester.py). Su desempeño histórico medido: Sharpe {sharpe_sug:.2f} vs. buy & hold "
    "{sharpe_bh:.2f} · CAGR {cagr_sug:+.1%} vs. buy & hold {cagr_bh:+.1%} · Max. drawdown "
    "{mdd_sug:.1%} vs. buy & hold {mdd_bh:.1%}. Usala como APOYO a tu análisis, NO como orden de operar."
)


class SuggesterPanel(QWidget):
    """Sugerencia (COMPRAR/VENDER/ESPERAR) con color + confianza, desglose
    de votos por estudio, y el desempeño histórico destacado de la regla.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("suggesterPanel")

        layout = QVBoxLayout(self)

        title = QLabel("Sugeridor de consenso")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.suggestion_label = QLabel("—")
        self.suggestion_label.setStyleSheet(_SUGGESTION_STYLE.format(color=DIRECTION_COLORS["ESPERAR"]))
        layout.addWidget(self.suggestion_label)

        self.confidence_label = QLabel("Confianza: —")
        self.confidence_label.setObjectName("cardName")
        layout.addWidget(self.confidence_label)

        self.votes_summary_label = QLabel("Votos: —")
        self.votes_summary_label.setObjectName("cardName")
        self.votes_summary_label.setWordWrap(True)
        layout.addWidget(self.votes_summary_label)

        detail_title = QLabel("Detalle por estudio")
        detail_title.setObjectName("panelNote")
        layout.addWidget(detail_title)

        detail_grid = QGridLayout()
        self._detail_labels: dict[str, QLabel] = {}
        for row, (key, label_text) in enumerate(_STUDY_LABELS.items()):
            name_label = QLabel(label_text)
            name_label.setObjectName("cardName")
            value_label = QLabel("—")
            value_label.setStyleSheet(f"color: {TEXT_MUTED};")
            detail_grid.addWidget(name_label, row, 0)
            detail_grid.addWidget(value_label, row, 1)
            self._detail_labels[key] = value_label
        layout.addLayout(detail_grid)

        self.performance_label = QLabel(DEFAULT_PERFORMANCE_TEXT)
        self.performance_label.setObjectName("performanceHighlight")
        self.performance_label.setWordWrap(True)
        layout.addWidget(self.performance_label)

        layout.addStretch()

    def update_values(self, resultado) -> None:
        """Vuelca `resultado.sugerencia` (el dict que arma `signals.suggester.suggest`,
        ya calculado por `app.workers.StudiesWorker`) a este panel.
        """
        sugerencia = resultado.sugerencia
        color = DIRECTION_COLORS.get(sugerencia["sugerencia"], DIRECTION_COLORS["ESPERAR"])
        self.suggestion_label.setText(sugerencia["sugerencia"])
        self.suggestion_label.setStyleSheet(_SUGGESTION_STYLE.format(color=color))

        self.confidence_label.setText(f"Confianza: {sugerencia['confianza']:.0%}")
        self.votes_summary_label.setText(
            f"Votos — alcistas: {sugerencia['votos_alcistas']} · bajistas: {sugerencia['votos_bajistas']} · "
            f"neutrales: {sugerencia['votos_neutrales']}"
        )

        for key, label in self._detail_labels.items():
            vote = sugerencia["detalle"].get(key, "neutral")
            label.setText(_VOTE_TEXT.get(vote, vote))
            label.setStyleSheet(f"color: {_VOTE_COLOR.get(vote, TEXT_MUTED)}; font-weight: bold;")

        desempeno = sugerencia["desempeno_historico"]
        self.performance_label.setText(
            PERFORMANCE_TEMPLATE.format(
                sharpe_sug=desempeno["sharpe_sugeridor"], sharpe_bh=desempeno["sharpe_buy_and_hold"],
                cagr_sug=desempeno["cagr_sugeridor"], cagr_bh=desempeno["cagr_buy_and_hold"],
                mdd_sug=desempeno["max_drawdown_sugeridor"], mdd_bh=desempeno["max_drawdown_buy_and_hold"],
            )
        )

    def reset(self) -> None:
        """Vuelve el panel a su estado vacío (nuevo análisis o error)."""
        self.suggestion_label.setText("—")
        self.suggestion_label.setStyleSheet(_SUGGESTION_STYLE.format(color=DIRECTION_COLORS["ESPERAR"]))
        self.confidence_label.setText("Confianza: —")
        self.votes_summary_label.setText("Votos: —")
        for label in self._detail_labels.values():
            label.setText("—")
            label.setStyleSheet(f"color: {TEXT_MUTED};")
        self.performance_label.setText(DEFAULT_PERFORMANCE_TEXT)
