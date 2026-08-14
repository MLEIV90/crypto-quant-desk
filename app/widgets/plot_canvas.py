"""Gráficos matplotlib embebidos en Qt (`FigureCanvasQTAgg`).

Estos widgets SOLO dibujan lo que ya viene calculado en un
`app.workers.AnalysisResult` — no calculan nada (ni medias móviles nuevas,
ni volatilidades): eso es responsabilidad exclusiva del backend, llamado
desde `app.workers.AnalysisWorker`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")  # backend embebido en Qt (PySide6), no una ventana propia

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from app.theme import (
    BACKGROUND,
    BORDER,
    PANEL_BACKGROUND,
    PRICE_LINE,
    REGIME_COLOR_DEFAULT,
    REGIME_COLORS,
    SMA_LINE,
    TEXT,
    TEXT_MUTED,
    VOL_LINE,
)


class _DarkCanvas(FigureCanvasQTAgg):
    """Base común: figura con los colores del tema oscuro de la terminal."""

    def __init__(self, parent=None) -> None:
        figure = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(figure)
        self.setParent(parent)
        self.figure.set_facecolor(PANEL_BACKGROUND)
        self.ax = figure.add_subplot(111)
        self._style_axes()

    def _style_axes(self) -> None:
        self.ax.set_facecolor(BACKGROUND)
        self.ax.tick_params(labelsize=8, colors=TEXT_MUTED)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.ax.title.set_color(TEXT)
        self.ax.xaxis.label.set_color(TEXT_MUTED)
        self.ax.yaxis.label.set_color(TEXT_MUTED)


class PriceCanvas(_DarkCanvas):
    """Precio de cierre + su media móvil (solo para referencia visual; la
    media móvil que usa el engine de señales es otra, ver `signals.engine`).
    """

    def plot(self, resultado) -> None:
        self.ax.clear()
        self._style_axes()

        self.ax.plot(resultado.close.index, resultado.close.to_numpy(), color=PRICE_LINE, linewidth=1.2, label="Precio")
        self.ax.plot(
            resultado.sma.index, resultado.sma.to_numpy(), color=SMA_LINE, linewidth=1.0,
            label=f"Media móvil ({resultado.sma_window}d)",
        )
        self.ax.set_title(f"{resultado.asset} — precio de cierre")
        legend = self.ax.legend(loc="upper left", fontsize=8, facecolor=PANEL_BACKGROUND, edgecolor=BORDER)
        for text in legend.get_texts():
            text.set_color(TEXT_MUTED)
        self.draw()


class EquityCurveCanvas(_DarkCanvas):
    """Curvas de equity (base 1.0) de la estrategia del engine vs. buy &
    hold, ya calculadas por `app.workers.BacktestWorker` — solo dibuja.
    """

    def plot(self, resultado) -> None:
        self.ax.clear()
        self._style_axes()

        self.ax.plot(
            resultado.equity_curve_estrategia.index, resultado.equity_curve_estrategia.to_numpy(),
            color=PRICE_LINE, linewidth=1.2, label="Estrategia (engine)",
        )
        self.ax.plot(
            resultado.equity_curve_buy_and_hold.index, resultado.equity_curve_buy_and_hold.to_numpy(),
            color=SMA_LINE, linewidth=1.2, label="Buy & hold",
        )
        self.ax.set_ylabel("Equity (base 1.0)")
        self.ax.set_title(f"{resultado.asset} — equity: estrategia vs. buy & hold")
        legend = self.ax.legend(loc="upper left", fontsize=8, facecolor=PANEL_BACKGROUND, edgecolor=BORDER)
        for text in legend.get_texts():
            text.set_color(TEXT_MUTED)
        self.draw()


class VolatilityCanvas(_DarkCanvas):
    """Volatilidad condicional GARCH en el tiempo, con el régimen ACTUAL
    (último día) resaltado según su color (calma/normal/tensión).
    """

    def plot(self, resultado) -> None:
        self.ax.clear()
        self._style_axes()

        vol_pct = resultado.cond_vol.to_numpy() * 100.0
        self.ax.plot(resultado.cond_vol.index, vol_pct, color=VOL_LINE, linewidth=1.2)
        self.ax.set_ylabel("Vol. anualizada (%)")
        self.ax.set_title(f"{resultado.asset} — vol. condicional GARCH ({resultado.garch_vol}/{resultado.garch_dist})")

        color = REGIME_COLORS.get(resultado.regimen_actual, REGIME_COLOR_DEFAULT)
        self.ax.scatter(
            [resultado.cond_vol.index[-1]], [vol_pct[-1]],
            color=color, s=70, zorder=5, edgecolors=TEXT, linewidths=0.8,
            label=f"Hoy: {resultado.regimen_actual}",
        )
        legend = self.ax.legend(loc="upper left", fontsize=8, facecolor=PANEL_BACKGROUND, edgecolor=BORDER)
        for text in legend.get_texts():
            text.set_color(TEXT_MUTED)
        self.draw()
