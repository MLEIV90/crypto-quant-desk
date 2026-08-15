"""Gráfico de velas japonesas (mplfinance embebido en Qt) de la pestaña
"Análisis Técnico" (Fase 7b).

Solo dibuja lo que ya viene calculado en un `app.workers.StudiesResult` —
no calcula ningún indicador acá (ni medias, ni RSI, ni niveles): eso es
responsabilidad exclusiva del backend, llamado desde
`app.workers.StudiesWorker` (ver `app/__init__.py`, regla de separación
modelo/vista).

NOTA TÉCNICA sobre mplfinance: a diferencia de los demás canvases del
cockpit (que reutilizan una única `Figure`/`Axes` y solo la limpian y
redibujan, ver `app.widgets.plot_canvas`), mplfinance necesita construir su
propia `Figure` de punta a punta en cada llamada a `mpf.plot()` — no admite
"redibujar sobre" una figura ya armada. Por eso este widget es un
CONTENEDOR (`QWidget` con un `QVBoxLayout`) que, en cada `plot()`, tira el
`FigureCanvasQTAgg` anterior y crea uno nuevo envolviendo la figura que
acaba de generar mplfinance, en vez de heredar de `FigureCanvasQTAgg`
directamente como los otros canvases.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("QtAgg")  # backend embebido en Qt (PySide6), no una ventana propia

import mplfinance as mpf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.theme import (
    BACKGROUND,
    BOLLINGER_LINE,
    BORDER,
    CANDLE_DOWN,
    CANDLE_UP,
    EMA12_LINE,
    EMA26_LINE,
    FIBONACCI_LINE,
    MACD_HIST_DOWN,
    MACD_HIST_UP,
    MACD_LINE,
    MACD_SIGNAL_LINE,
    PANEL_BACKGROUND,
    PIVOT_LINE,
    RESISTANCE_LINE,
    RSI_LINE,
    SMA20_LINE,
    SMA50_LINE,
    STOCH_D_LINE,
    STOCH_K_LINE,
    SUPPORT_LINE,
    TEXT,
    TEXT_MUTED,
)

logger = logging.getLogger(__name__)

_PANEL_RATIOS = (6, 1.6, 1.6, 1.6, 1.6)  # precio, volumen, RSI, MACD, estocástico
_PIVOT_KEYS_SHOWN = ("P", "R1", "S1")  # ver Fase 7b: "Pivotes del día (P, R1, S1)"

_MPF_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    facecolor=BACKGROUND,
    figcolor=PANEL_BACKGROUND,
    edgecolor=BORDER,
    gridcolor=BORDER,
    gridstyle="--",
    marketcolors=mpf.make_marketcolors(
        up=CANDLE_UP, down=CANDLE_DOWN, edge="inherit", wick="inherit", volume="inherit",
    ),
    rc={
        "axes.labelcolor": TEXT_MUTED,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT,
        "axes.edgecolor": BORDER,
    },
)


def _horizontal_level(ax, level: float, color: str, label: str, linestyle: str = "--") -> None:
    """Línea horizontal + etiqueta pegada al borde derecho del panel de
    precio, en coordenadas MEZCLADAS (x en fracción de los ejes, y en
    unidades de precio) — necesario porque mplfinance arma su propio eje X
    POSICIONAL internamente (no son fechas reales), así que anotar con una
    fecha como coordenada X quedaría mal ubicado.
    """
    ax.axhline(level, color=color, linestyle=linestyle, linewidth=0.8, alpha=0.75, zorder=1)
    ax.annotate(
        label, xy=(1.005, level), xycoords=ax.get_yaxis_transform(),
        fontsize=6.5, color=color, va="center", annotation_clip=False,
    )


def _build_figure(resultado) -> Figure:
    df = resultado.ohlcv_recent
    price_min, price_max = float(df["Low"].min()), float(df["High"].max())
    visible_margin = (price_max - price_min) * 0.15

    def _in_range(level: float) -> bool:
        return (price_min - visible_margin) <= level <= (price_max + visible_margin)

    addplots = [
        mpf.make_addplot(resultado.sma_20, panel=0, color=SMA20_LINE, width=1.0),
        mpf.make_addplot(resultado.sma_50, panel=0, color=SMA50_LINE, width=1.0),
        mpf.make_addplot(resultado.ema_12, panel=0, color=EMA12_LINE, width=0.9),
        mpf.make_addplot(resultado.ema_26, panel=0, color=EMA26_LINE, width=0.9),
        mpf.make_addplot(resultado.bb_upper, panel=0, color=BOLLINGER_LINE, width=0.7, linestyle="--"),
        mpf.make_addplot(resultado.bb_mid, panel=0, color=BOLLINGER_LINE, width=0.5, linestyle=":"),
        mpf.make_addplot(resultado.bb_lower, panel=0, color=BOLLINGER_LINE, width=0.7, linestyle="--"),
        mpf.make_addplot(resultado.rsi_14, panel=2, color=RSI_LINE, width=1.0, ylabel="RSI"),
        mpf.make_addplot(resultado.macd, panel=3, color=MACD_LINE, width=1.0, ylabel="MACD"),
        mpf.make_addplot(resultado.macd_signal, panel=3, color=MACD_SIGNAL_LINE, width=1.0),
        mpf.make_addplot(
            resultado.macd_hist, panel=3, type="bar", width=0.7, alpha=0.6,
            color=[MACD_HIST_UP if v >= 0 else MACD_HIST_DOWN for v in resultado.macd_hist.fillna(0.0)],
        ),
        mpf.make_addplot(resultado.stoch_k, panel=4, color=STOCH_K_LINE, width=1.0, ylabel="Estocástico"),
        mpf.make_addplot(resultado.stoch_d, panel=4, color=STOCH_D_LINE, width=1.0),
    ]

    datetime_format = "%Y-%m-%d" if resultado.timeframe == "1d" else "%m-%d %H:%M"

    fig, axes = mpf.plot(
        df, type="candle", style=_MPF_STYLE, volume=True, volume_panel=1,
        panel_ratios=_PANEL_RATIOS, addplot=addplots, returnfig=True,
        figsize=(13, 11), scale_width_adjustment=dict(candle=1.4, volume=0.7),
        show_nontrading=False, datetime_format=datetime_format, xrotation=15,
    )
    # Margen derecho reservado a propósito (en vez de tight_layout, que lo
    # recorta): ahí van las etiquetas de Fibonacci/soporte-resistencia/
    # pivotes, ancladas por fuera del área de ejes (ver `_horizontal_level`).
    fig.subplots_adjust(left=0.07, right=0.90, top=0.96, bottom=0.06, hspace=0.15)
    price_ax, rsi_ax, macd_ax, stoch_ax = axes[0], axes[4], axes[6], axes[8]

    # --- Leyenda manual del panel de precio (mplfinance no arma una sola
    #     por sí mismo cuando se usan addplots superpuestos) ---
    legend_handles = [
        Line2D([], [], color=SMA20_LINE, linewidth=1.2, label="SMA 20"),
        Line2D([], [], color=SMA50_LINE, linewidth=1.2, label="SMA 50"),
        Line2D([], [], color=EMA12_LINE, linewidth=1.2, label="EMA 12"),
        Line2D([], [], color=EMA26_LINE, linewidth=1.2, label="EMA 26"),
        Line2D([], [], color=BOLLINGER_LINE, linewidth=1.2, linestyle="--", label="Bollinger"),
    ]
    legend = price_ax.legend(
        handles=legend_handles, loc="upper left", fontsize=7, facecolor=PANEL_BACKGROUND, edgecolor=BORDER,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_MUTED)

    # --- Fibonacci (Fase 7a: SIN respaldo predictivo probado, ver
    #     signals.studies.fibonacci_levels — se muestra igual, por pedido
    #     explícito, pero solo los retrocesos intermedios pedidos) ---
    fib_labels_to_show = {"23.6%", "38.2%", "50.0%", "61.8%", "78.6%"}
    if resultado.fibonacci:
        for etiqueta, nivel in resultado.fibonacci.items():
            if etiqueta not in fib_labels_to_show or not _in_range(nivel):
                continue
            _horizontal_level(price_ax, nivel, FIBONACCI_LINE, f"fib {etiqueta}", linestyle=":")

    # --- Soporte / resistencia (Fase 7a) ---
    for nivel in resultado.soporte_resistencia.get("resistencia", []):
        if _in_range(nivel):
            _horizontal_level(price_ax, nivel, RESISTANCE_LINE, "R", linestyle="-")
    for nivel in resultado.soporte_resistencia.get("soporte", []):
        if _in_range(nivel):
            _horizontal_level(price_ax, nivel, SUPPORT_LINE, "S", linestyle="-")

    # --- Pivotes del día: P, R1, S1 (Fase 7b) ---
    for clave in _PIVOT_KEYS_SHOWN:
        nivel = resultado.pivotes.get(clave)
        if nivel is not None and _in_range(nivel):
            _horizontal_level(price_ax, nivel, PIVOT_LINE, clave, linestyle="-.")

    # --- Líneas de referencia de los osciladores ---
    rsi_ax.axhline(70, color=TEXT_MUTED, linestyle="--", linewidth=0.6)
    rsi_ax.axhline(30, color=TEXT_MUTED, linestyle="--", linewidth=0.6)
    stoch_ax.axhline(80, color=TEXT_MUTED, linestyle="--", linewidth=0.6)
    stoch_ax.axhline(20, color=TEXT_MUTED, linestyle="--", linewidth=0.6)
    macd_ax.axhline(0, color=TEXT_MUTED, linestyle="-", linewidth=0.5)

    price_ax.set_title(
        f"{resultado.asset} ({resultado.timeframe}) — últimas {len(df)} velas, al {resultado.ultima_fecha.date()}",
        color=TEXT, fontsize=11,
    )

    return fig


class TechnicalChartCanvas(QWidget):
    """Contenedor del gráfico de velas + overlays + sub-paneles. Reemplaza
    su `FigureCanvasQTAgg` interno entero en cada `plot()` (ver nota
    técnica del docstring del módulo) — no hereda de `FigureCanvasQTAgg`
    como el resto de los canvases del cockpit.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas: FigureCanvasQTAgg | None = None

        self._placeholder = QLabel('Elegí un activo y un timeframe, y presioná "Analizar" para ver el gráfico.')
        self._placeholder.setObjectName("panelNote")
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)

    def plot(self, resultado) -> None:
        try:
            figure = _build_figure(resultado)
        except Exception:
            logger.exception("TechnicalChartCanvas: falló al construir el gráfico de %s", resultado.asset)
            raise

        new_canvas = FigureCanvasQTAgg(figure)
        self._swap_canvas(new_canvas)

    def reset(self) -> None:
        self._swap_canvas(None)

    def _swap_canvas(self, new_canvas: FigureCanvasQTAgg | None) -> None:
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None
        else:
            self._layout.removeWidget(self._placeholder)
            self._placeholder.setParent(None)

        if new_canvas is not None:
            self._canvas = new_canvas
            self._layout.addWidget(self._canvas)
        else:
            self._layout.addWidget(self._placeholder)
