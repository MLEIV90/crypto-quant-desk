"""Informe PDF descargable (Fase 16b): `build_report(asset, interval="1d")
-> bytes` arma un PDF con gráficas (matplotlib) + texto explicativo
(reportlab) a partir de TODO el backend de análisis ya existente
(`analysis/`, `signals/`, `models/garch.py`, `metrics/`, `pairs/`,
`eda/`) — este módulo NO reimplementa ningún cálculo, solo llama a las
mismas funciones que ya usa `api/main.py` para cada vista del frontend
(ver el comentario de cada sección de abajo, con el endpoint que reutiliza
como referencia) y las convierte en párrafos + tablas + figuras.

ENCUADRE HONESTO — igual que el resto del proyecto: el informe EXPLICA los
análisis y muestra los valores actuales, NUNCA recomienda comprar/vender.
Cada sección lleva un párrafo que dice qué muestra el dato, cómo leerlo, y
qué NO significa (mismo criterio que `frontend/src/helpTexts.ts`).

LIBRERÍA ELEGIDA: matplotlib (ya es dependencia del proyecto) para las
gráficas, renderizadas a PNG en memoria y embebidas como imágenes; reportlab
(agregado en esta fase, no estaba en requirements.txt) para la maquetación
del documento — título, tablas, párrafos con wrap de texto. Se descartó
`matplotlib.backends.backend_pdf.PdfPages` porque el informe necesita texto
largo con wrap automático y tablas con estilo (drawdowns, fases de mercado,
screening de pares) — hacerlo a mano sobre `Figure`/`Axes` de matplotlib
habría sido mucho más código y peor resultado que usar un motor de
maquetación de documentos (reportlab) ya pensado para eso.

RENDIMIENTO — léase antes de llamar a este módulo desde un request HTTP:
`build_report` ajusta UN modelo GARCH (`models.garch.select_best_model`,
grid search de 6 especificaciones, igual que `/api/risk`) y corre
`pairs.stability.screen_pairs_stability` (cointegración rolling sobre las
10 combinaciones de a pares de `config.UNIVERSE`, igual que
`/api/pairs/screening`) — la suma de ambos es, con diferencia, la parte más
lenta del informe: del orden de 30 a 90 segundos según el activo y la
cantidad de historia disponible. El GARCH se ajusta UNA SOLA VEZ (no dos,
a diferencia de tener `/api/risk` y `/api/garch-series` como endpoints
separados) y se reutiliza tanto para las métricas puntuales como para la
serie completa del gráfico — ver `_build_risk_section`.

`_load_df` duplica (a propósito, no por descuido) la lógica de
`api/main.py::_load_df` (`source="store"`, `use_cache=False`, `end=mañana`
para no truncar la vela de hoy) en vez de importarla: `api/main.py` importa
`build_report` de este módulo para `GET /api/report`, así que importar en
sentido contrario crearía un ciclo. Es la única lógica duplicada de este
archivo — todo lo demás (cálculos) se reutiliza tal cual, sin reimplementar
nada.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # backend sin display, server-side (Fase 16b)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analysis.comparison import align_common_dates
from analysis.cycles import drawdown_analysis, market_phases, monthly_yearly_heatmap
from config import UNIVERSE
from data.loaders import get_prices
from eda.eda_report import adf_test, correlation_matrix
from metrics.risk_measures import expected_shortfall, value_at_risk
from models.garch import conditional_volatility, select_best_model, volatility_regime
from pairs.stability import screen_pairs_stability
from signals.engine import latest_recommendation
from signals.indicators import add_all_indicators
from signals.returns import log_returns, simple_returns
from signals.studies import all_studies
from signals.suggester import RSI_OVERBOUGHT, RSI_OVERSOLD, suggest

logger = logging.getLogger(__name__)

SUPPORTED_INTERVALS: tuple[str, ...] = ("1d", "1h")
# El modelo GARCH/pairs screening de este proyecto son diarios (ver
# `api/main.py::DEFAULT_RISK_INTERVAL` y `pairs/stability.py`) — el informe
# sigue la misma convención sin importar el `interval` pedido.
REPORT_RISK_INTERVAL = "1d"
CORRELATION_LOOKBACK_DAYS = 365
MAX_PHASE_ROWS = 12
MAX_DRAWDOWN_ROWS = 5

MONTH_LABELS = ("Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic")


# --------------------------------------------------------------------------
# Carga de datos (duplica _load_df de api/main.py a propósito — ver
# docstring del módulo, es la única lógica no reutilizada tal cual).
# --------------------------------------------------------------------------


def _load_df(asset: str, interval: str) -> pd.DataFrame:
    if asset not in UNIVERSE:
        raise ValueError(f"Activo no soportado: {asset!r}. Elegí uno de {list(UNIVERSE)}.")
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Intervalo no soportado: {interval!r}. Elegí uno de {SUPPORTED_INTERVALS}.")

    tomorrow = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return get_prices(asset, source="store", interval=interval, end=tomorrow, use_cache=False)


# --------------------------------------------------------------------------
# Helpers de maquetación (reportlab)
# --------------------------------------------------------------------------


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=22, leading=26, fontName="Helvetica-Bold"))
    styles.add(
        ParagraphStyle(name="ReportSubtitle", fontSize=13, leading=16, textColor=colors.HexColor("#555555"))
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            fontSize=15,
            leading=18,
            spaceBefore=4,
            spaceAfter=8,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1a1a2e"),
        )
    )
    styles.add(
        ParagraphStyle(name="SubHeading", fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
    )
    styles.add(ParagraphStyle(name="Body2", fontSize=9.5, leading=13, spaceAfter=8, alignment=TA_JUSTIFY))
    styles.add(
        ParagraphStyle(
            name="HonestyNote",
            fontSize=9,
            leading=12.5,
            spaceAfter=8,
            textColor=colors.HexColor("#7a4a00"),
            backColor=colors.HexColor("#fff4da"),
            borderPadding=8,
        )
    )
    styles.add(ParagraphStyle(name="Cell", fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="CellBold", fontSize=8, leading=10, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Caption", fontSize=8, leading=10, textColor=colors.HexColor("#777777"), alignment=TA_CENTER))
    return styles


def _fig_to_image(fig: plt.Figure, width_cm: float = 16.0) -> Image:
    """Renderiza una `Figure` de matplotlib a PNG en memoria y la devuelve
    como flowable `Image` de reportlab, escalada a `width_cm` manteniendo la
    proporción original (alto/ancho) de la figura.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    reader = ImageReader(buf)
    natural_w, natural_h = reader.getSize()
    width = width_cm * cm
    height = width * natural_h / natural_w

    buf.seek(0)
    return Image(buf, width=width, height=height)


def _text_table(rows: list[list[str]], styles, col_widths: list[float]) -> Table:
    """Tabla con la primera fila como encabezado (fondo oscuro, texto
    blanco) y filas alternadas — cada celda es un `Paragraph` para que el
    texto largo (p. ej. explicaciones) haga wrap dentro de la columna en
    vez de desbordar la página.
    """
    header = [Paragraph(str(cell), styles["CellBold"]) for cell in rows[0]]
    data = [header]
    for row in rows[1:]:
        data.append([Paragraph(str(cell), styles["Cell"]) for cell in row])

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


# --------------------------------------------------------------------------
# Lecturas en texto plano de valores numéricos (presentación, no cálculo:
# reusa los mismos umbrales/reglas que ya definen signals/suggester.py).
# --------------------------------------------------------------------------


def _rsi_reading(rsi: float | None) -> str:
    if rsi is None:
        return "Sin datos suficientes (warmup)."
    if rsi > RSI_OVERBOUGHT:
        return f"Sobrecomprado (> {RSI_OVERBOUGHT:.0f})."
    if rsi < RSI_OVERSOLD:
        return f"Sobrevendido (< {RSI_OVERSOLD:.0f})."
    return "En rango neutral."


def _trend_reading(precio: float, sma_20: float | None, sma_50: float | None) -> str:
    if sma_20 is None or sma_50 is None:
        return "Sin datos suficientes (warmup)."
    if precio > sma_20 and precio > sma_50:
        return "Precio por encima de ambas medias (sesgo alcista de corto/mediano plazo)."
    if precio < sma_20 and precio < sma_50:
        return "Precio por debajo de ambas medias (sesgo bajista de corto/mediano plazo)."
    return "Medias mezcladas, sin sesgo claro de tendencia."


def _macd_reading(macd_hist: float | None) -> str:
    if macd_hist is None:
        return "Sin datos suficientes (warmup)."
    if macd_hist > 0:
        return "Histograma positivo: momentum alcista reciente."
    if macd_hist < 0:
        return "Histograma negativo: momentum bajista reciente."
    return "Histograma en cero: momentum neutral."


# --------------------------------------------------------------------------
# PORTADA
# --------------------------------------------------------------------------


def _build_cover(asset: str, interval: str, close: pd.Series, generated_at: datetime, styles) -> list:
    story: list = [
        Spacer(1, 2.5 * cm),
        Paragraph("crypto-quant-desk", styles["ReportTitle"]),
        Paragraph(f"Informe de análisis — {asset} ({interval})", styles["ReportSubtitle"]),
        Spacer(1, 1 * cm),
    ]

    meta_rows = [
        ["Activo", asset],
        ["Intervalo de los datos", interval],
        ["Última fecha con datos", close.index[-1].strftime("%Y-%m-%d %H:%M UTC")],
        ["Informe generado", generated_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Precio de cierre más reciente", f"${close.iloc[-1]:,.2f}"],
    ]
    table = Table(meta_rows, colWidths=[6 * cm, 9 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 1.5 * cm))

    disclaimer = (
        "Este informe es un resumen de ANÁLISIS histórico y estadístico — precio, volatilidad, indicadores "
        "técnicos, ciclos de mercado, correlación entre monedas y cointegración de pares. NO es asesoramiento "
        "financiero, NO es una recomendación de compra/venta, y ningún número de acá predice el precio futuro. "
        "Cada sección explica honestamente qué muestra el dato, cómo leerlo, y qué NO significa — el mismo "
        "criterio que rige toda la aplicación crypto-quant-desk."
    )
    story.append(Paragraph(disclaimer, styles["HonestyNote"]))
    return story


# --------------------------------------------------------------------------
# SECCIÓN 1: RIESGO — reutiliza la misma secuencia que api/main.py::get_risk
# y api/main.py::get_garch_series, pero ajustando el GARCH UNA sola vez.
# --------------------------------------------------------------------------


def _build_risk_section(asset: str, df_daily: pd.DataFrame, styles) -> list:
    story: list = [Paragraph("1. Riesgo", styles["SectionHeading"])]
    story.append(
        Paragraph(
            "Estas son medidas de RIESGO calculadas sobre la historia del activo — ninguna predice el precio "
            "de mañana. Sirven para decidir CUÁNTO exponerte (tamaño de posición), no CUÁNDO vas a acertar. "
            "Se calculan siempre sobre velas DIARIAS, sea cual sea el intervalo elegido para el resto del "
            "informe (el modelo GARCH de este proyecto es diario).",
            styles["Body2"],
        )
    )

    close = df_daily["close"]
    returns = simple_returns(close).dropna()
    garch_returns = log_returns(close).dropna()

    best = select_best_model(garch_returns, criterion="aic")
    cond_vol = conditional_volatility(best["result"])
    regime_series = volatility_regime(cond_vol)
    last_regime = regime_series.iloc[-1]

    var95 = value_at_risk(returns, level=0.95)
    es95 = expected_shortfall(returns, level=0.95)
    recomendacion = latest_recommendation(df_daily, garch_regime=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(close.index, close.values, color="#1f77b4", linewidth=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("Precio (USD, esc. log)")
    ax1.set_title(f"{asset} — precio y volatilidad condicional GARCH")
    ax1.grid(alpha=0.25)
    ax2.plot(cond_vol.index, cond_vol.values * 100, color="#d62728", linewidth=1)
    ax2.set_ylabel("Vol. anualizada (%)")
    ax2.set_xlabel("Fecha")
    ax2.grid(alpha=0.25)
    fig.autofmt_xdate()
    story.append(_fig_to_image(fig))
    story.append(Spacer(1, 0.3 * cm))

    regimen_label = str(last_regime) if pd.notna(last_regime) else "sin dato (warmup)"
    rows = [
        ["Métrica", "Valor", "Qué significa"],
        [
            "Vol. realizada (anualizada)",
            f"{recomendacion['vol_realizada'] * 100:.1f}%",
            "Qué tan movido estuvo el precio en el pasado reciente. Más alta = movimientos diarios más "
            "grandes, en ambas direcciones.",
        ],
        [
            "Modelo GARCH elegido",
            f"{best['vol']}/{best['dist']}",
            "El modelo (familia GARCH) que mejor ajustó la volatilidad histórica, elegido automáticamente "
            "por criterio AIC — no lo elige un humano a mano.",
        ],
        [
            "Vol. GARCH (anualizada)",
            f"{cond_vol.iloc[-1] * 100:.1f}%",
            "Estimación de volatilidad del modelo GARCH: pondera más los movimientos recientes que la vol. "
            "realizada simple. Proyección de corto plazo sobre datos pasados, no una certeza sobre el futuro.",
        ],
        [
            "Régimen de volatilidad",
            regimen_label,
            "CALMA = volatilidad baja respecto a la propia historia del activo. NORMAL = típica. TENSIÓN = "
            "alta. Útil para ajustar tamaño de posición, no para adivinar el momento de entrar o salir.",
        ],
        [
            "VaR 95%",
            f"{var95 * 100:.2f}%",
            "En un día malo típico (peor 5% histórico), la pérdida esperada ronda este valor. Medida "
            "estadística sobre el pasado, no una predicción de lo que va a pasar mañana.",
        ],
        [
            "Expected Shortfall 95%",
            f"{es95 * 100:.2f}%",
            "Si ese día malo (peor 5%) efectivamente ocurre, esta es la pérdida PROMEDIO esperada en ese "
            "escenario — siempre >= VaR, porque mira más adentro de la cola mala de la distribución.",
        ],
        [
            "Señal del motor",
            recomendacion["accion"],
            "Dirección sugerida por el motor de señales (tendencia + momentum + reversión a la media). Dato "
            "de apoyo, no una recomendación de inversión.",
        ],
        [
            "Tamaño sugerido (vol targeting)",
            f"{recomendacion['tamaño_sugerido'] * 100:.0f}%",
            "Tamaño de posición sugerido por 'vol targeting': apunta a una volatilidad objetivo del book, "
            "reduciendo la exposición cuando el activo está más volátil de lo normal.",
        ],
    ]
    story.append(_text_table(rows, styles, col_widths=[3.6 * cm, 2.6 * cm, 9.3 * cm]))
    return story


# --------------------------------------------------------------------------
# SECCIÓN 2: ANÁLISIS TÉCNICO — reutiliza api/main.py::get_studies y
# api/main.py::get_suggester tal cual.
# --------------------------------------------------------------------------


def _build_technical_section(asset: str, interval: str, df: pd.DataFrame, styles) -> list:
    story: list = [Paragraph("2. Análisis técnico", styles["SectionHeading"])]
    story.append(
        Paragraph(
            "Indicadores técnicos clásicos sobre el precio. El proyecto ya evaluó formalmente (ver la "
            "sección de Machine Learning del repo) que estas herramientas NO tienen una ventaja "
            "estadística consistente para predecir la dirección del precio — se muestran como ANÁLISIS "
            "VISUAL y de apoyo, no como un sistema con edge demostrado.",
            styles["Body2"],
        )
    )

    indicators_df = add_all_indicators(df)
    resumen = all_studies(df)
    sugerencia = suggest(df)

    plot_df = df.tail(500)
    plot_ind = indicators_df.loc[plot_df.index]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(plot_df.index, plot_df["close"], color="#1f77b4", linewidth=1, label="Precio")
    ax.plot(plot_ind.index, plot_ind["sma_20"], color="#ff7f0e", linewidth=0.9, label="SMA 20")
    ax.plot(plot_ind.index, plot_ind["sma_50"], color="#2ca02c", linewidth=0.9, label="SMA 50")
    ax.plot(plot_ind.index, plot_ind["bb_upper"], color="#999999", linewidth=0.6, linestyle="--", label="Bollinger")
    ax.plot(plot_ind.index, plot_ind["bb_lower"], color="#999999", linewidth=0.6, linestyle="--")
    ax.fill_between(plot_ind.index, plot_ind["bb_lower"], plot_ind["bb_upper"], color="#999999", alpha=0.08)
    ax.set_title(f"{asset} ({interval}) — últimas {len(plot_df)} velas: precio, medias y Bollinger")
    ax.set_ylabel("Precio (USD)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    story.append(_fig_to_image(fig))
    story.append(Spacer(1, 0.3 * cm))

    ind = resumen["indicadores"]
    estado_rows = [
        ["Estudio", "Valor actual", "Lectura"],
        ["RSI (14)", f"{ind['rsi_14']:.1f}" if ind["rsi_14"] is not None else "s/d", _rsi_reading(ind["rsi_14"])],
        [
            "MACD histograma",
            f"{ind['macd_hist']:.2f}" if ind["macd_hist"] is not None else "s/d",
            _macd_reading(ind["macd_hist"]),
        ],
        [
            "Precio vs. SMA20/SMA50",
            f"${resumen['precio']:,.2f}",
            _trend_reading(resumen["precio"], ind["sma_20"], ind["sma_50"]),
        ],
    ]
    story.append(_text_table(estado_rows, styles, col_widths=[3.6 * cm, 3 * cm, 8.9 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Sugeridor de consenso", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Es un VOTO de varios estudios técnicos (RSI, medias, MACD, estocástico, Bollinger, pivote) "
            "combinados por mayoría — no un modelo entrenado ni una IA que aprendió patrones. Por eso viaja "
            "siempre junto a su desempeño histórico real (Sharpe/CAGR/drawdown, con costos de transacción): "
            "si no le gana con claridad al buy & hold, tratalo como un dato más entre varios, no como una "
            "orden de operar.",
            styles["Body2"],
        )
    )
    perf = sugerencia["desempeno_historico"]
    sugg_rows = [
        ["Dato", "Valor"],
        ["Sugerencia actual", sugerencia["sugerencia"]],
        [
            "Votos alcistas / bajistas / neutrales",
            f"{sugerencia['votos_alcistas']} / {sugerencia['votos_bajistas']} / {sugerencia['votos_neutrales']}",
        ],
        ["Confianza", f"{sugerencia['confianza'] * 100:.0f}%"],
        [
            "CAGR sugeridor vs. buy & hold",
            f"{perf['cagr_sugeridor'] * 100:.1f}% vs. {perf['cagr_buy_and_hold'] * 100:.1f}%",
        ],
        [
            "Sharpe sugeridor vs. buy & hold",
            f"{perf['sharpe_sugeridor']:.2f} vs. {perf['sharpe_buy_and_hold']:.2f}",
        ],
        [
            "Máx. drawdown sugeridor vs. buy & hold",
            f"{perf['max_drawdown_sugeridor'] * 100:.1f}% vs. {perf['max_drawdown_buy_and_hold'] * 100:.1f}%",
        ],
        ["Cantidad de operaciones (sugeridor)", str(perf["n_trades_sugeridor"])],
    ]
    story.append(_text_table(sugg_rows, styles, col_widths=[7.5 * cm, 8 * cm]))
    return story


# --------------------------------------------------------------------------
# SECCIÓN 3: CICLOS Y ESTADÍSTICA — reutiliza api/main.py::get_stats tal cual.
# --------------------------------------------------------------------------


def _monthly_heatmap_figure(heatmap: dict) -> plt.Figure:
    anios = heatmap["anios"]
    matriz = np.array([[v if v is not None else np.nan for v in row] for row in heatmap["matriz"]])

    fig, ax = plt.subplots(figsize=(min(0.55 * len(anios) + 2, 12), 4.5))
    finite = matriz[np.isfinite(matriz)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    im = ax.imshow(matriz, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(anios)))
    ax.set_xticklabels(anios, rotation=45, fontsize=7)
    ax.set_yticks(range(12))
    ax.set_yticklabels(MONTH_LABELS, fontsize=7)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            val = matriz[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5.5, color="black")
    fig.colorbar(im, ax=ax, label="Retorno mensual (%)", fraction=0.035, pad=0.02)
    ax.set_title("Retorno compuesto por mes-año (%)")
    fig.tight_layout()
    return fig


def _build_cycles_section(close: pd.Series, returns: pd.Series, styles) -> list:
    story: list = [Paragraph("3. Ciclos y estadística", styles["SectionHeading"])]
    story.append(
        Paragraph(
            "Estos análisis describen patrones HISTÓRICOS del activo — ninguno predice el precio de mañana. "
            "En cripto, con pocos años de historia disponible, tratalos como observaciones para tu propio "
            "criterio, no como reglas para operar.",
            styles["Body2"],
        )
    )

    drawdowns = drawdown_analysis(close, top_n=MAX_DRAWDOWN_ROWS)
    fases = market_phases(close)  # default min_duration_days=30 (Fase 16a): sin whipsaws
    heatmap = monthly_yearly_heatmap(returns)
    adf_precio = adf_test(close)
    adf_retornos = adf_test(returns)

    story.append(Paragraph(f"Drawdowns históricos (peores {len(drawdowns)})", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Un drawdown es una caída desde un máximo histórico hasta el mínimo posterior, antes de volver "
            "a superar ese máximo. Es historia, no una garantía de que el próximo drawdown tenga una forma "
            "parecida.",
            styles["Body2"],
        )
    )
    dd_rows = [["Pico", "Fondo", "Profundidad", "Días de caída", "Recuperación"]]
    for d in drawdowns:
        recuperacion = (
            d["fecha_recuperacion"].strftime("%Y-%m-%d") if d["fecha_recuperacion"] is not None else "Todavía no recuperó"
        )
        dd_rows.append(
            [
                d["fecha_pico"].strftime("%Y-%m-%d"),
                d["fecha_fondo"].strftime("%Y-%m-%d"),
                f"{d['profundidad_pct']:.1f}%",
                str(d["dias_caida"]),
                recuperacion,
            ]
        )
    story.append(_text_table(dd_rows, styles, col_widths=[2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 4.3 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Fases de mercado (filtradas, mínimo 30 días)", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Una fase pasa a ser BAJISTA en cuanto el precio cae 20% o más desde un máximo, y ALCISTA en "
            "cuanto sube 20% o más desde un mínimo — regla mecánica y arbitraria (el 20% es convención de "
            "medios financieros, no un número mágico). Las fases más cortas que 30 días se fusionan con sus "
            "vecinas para no mostrar 'whipsaws' de pocos días como si fueran ciclos de mercado. La fase más "
            "reciente queda 'en curso' porque todavía no se confirmó el próximo cruce de 20% en sentido opuesto.",
            styles["Body2"],
        )
    )
    shown_fases = fases[-MAX_PHASE_ROWS:]
    if len(fases) > MAX_PHASE_ROWS:
        story.append(
            Paragraph(
                f"Mostrando las {MAX_PHASE_ROWS} fases más recientes de {len(fases)} totales.", styles["Caption"]
            )
        )
    fase_rows = [["Tipo", "Inicio", "Fin", "Duración", "Retorno", "Estado"]]
    for f in shown_fases:
        fase_rows.append(
            [
                "Alcista" if f["tipo"] == "bull" else "Bajista",
                f["fecha_inicio"].strftime("%Y-%m-%d"),
                f["fecha_fin"].strftime("%Y-%m-%d"),
                f"{f['duracion_dias']} días",
                f"{f['retorno_pct']:+.1f}%",
                "Confirmada" if f["confirmada"] else "En curso",
            ]
        )
    if fases:
        story.append(_text_table(fase_rows, styles, col_widths=[2.2 * cm, 2.6 * cm, 2.6 * cm, 2.3 * cm, 2.3 * cm, 3.5 * cm]))
    else:
        story.append(Paragraph("Ningún movimiento cruzó el umbral de 20% en el período disponible.", styles["Body2"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Estacionalidad mensual (mes x año)", styles["SubHeading"]))
    story.append(
        Paragraph(
            "Retorno compuesto de cada mes, año por año — a diferencia de un promedio único por mes (que "
            "mezcla todos los años en un solo número), acá se ve la estacionalidad real, con qué tan distinto "
            "fue cada año. Verde = mes positivo, rojo = negativo; casillero vacío = sin datos ese mes-año.",
            styles["Body2"],
        )
    )
    if heatmap["anios"]:
        story.append(_fig_to_image(_monthly_heatmap_figure(heatmap), width_cm=16))
    else:
        story.append(Paragraph("Sin suficiente historia para armar el mapa de calor.", styles["Body2"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Estacionariedad (test ADF)", styles["SubHeading"]))
    story.append(
        Paragraph(
            "El test ADF pregunta si una serie 'vuelve' hacia un nivel estable con el tiempo (estacionaria) "
            "o si puede alejarse sin límite (no estacionaria, un 'paseo aleatorio'). El PRECIO típicamente NO "
            "es estacionario (tiene tendencia de largo plazo) — por eso este proyecto modela sobre RETORNOS, "
            "que sí suelen serlo. p-valor < 0.05 se lee como 'sí es estacionaria'.",
            styles["Body2"],
        )
    )
    adf_rows = [
        ["Serie", "¿Estacionaria?", "p-valor"],
        ["Precio", "Sí" if adf_precio["es_estacionaria"] else "No", f"{adf_precio['p_valor']:.4f}"],
        ["Retornos", "Sí" if adf_retornos["es_estacionaria"] else "No", f"{adf_retornos['p_valor']:.4f}"],
    ]
    story.append(_text_table(adf_rows, styles, col_widths=[5 * cm, 5 * cm, 5 * cm]))
    return story


# --------------------------------------------------------------------------
# SECCIÓN 4: CORRELACIÓN — reutiliza api/main.py::get_correlation tal cual.
# --------------------------------------------------------------------------


def _build_correlation_section(asset: str, interval: str, close_asset: pd.Series, styles) -> list:
    story: list = [Paragraph("4. Correlación entre monedas", styles["SectionHeading"])]
    story.append(
        Paragraph(
            "Correlación entre los RETORNOS de cada par de monedas — no entre sus precios: dos precios "
            "pueden parecer muy correlacionados solo porque ambos vienen subiendo con el tiempo. Cercana a "
            "+1 = se mueven casi juntas; cercana a 0 = independientes; cercana a -1 = sentido opuesto. NO son "
            "fijas — cambian con el tiempo y el período elegido, esto es una foto, no una constante del mercado.",
            styles["Body2"],
        )
    )

    assets = list(UNIVERSE)
    returns: dict[str, pd.Series] = {}
    for a in assets:
        close_a = close_asset if a == asset else _load_df(a, interval)["close"]
        returns[a] = simple_returns(close_a).dropna()

    aligned = align_common_dates(returns).tail(CORRELATION_LOOKBACK_DAYS)
    trimmed = {a: aligned[a] for a in assets}
    corr_df = correlation_matrix(trimmed, method="pearson")

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(assets)))
    ax.set_xticklabels(assets)
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets)
    for i in range(len(assets)):
        for j in range(len(assets)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlación (Pearson)")
    ax.set_title(f"Correlación de retornos ({len(aligned)} fechas comunes)")
    fig.tight_layout()
    story.append(_fig_to_image(fig, width_cm=11))
    return story


# --------------------------------------------------------------------------
# SECCIÓN 5: ARBITRAJE — reutiliza api/main.py::get_pairs_screening tal cual.
# --------------------------------------------------------------------------


def _build_arbitrage_section(styles) -> list:
    story: list = [Paragraph("5. Arbitraje estadístico (pares)", styles["SectionHeading"])]
    story.append(
        Paragraph(
            "Esto es arbitraje ESTADÍSTICO ('pairs trading'), no arbitraje entre exchanges: se busca un PAR "
            "de monedas cuyos precios se mueven juntos de forma estable, para apostar a que un desvío "
            "temporal entre ellas revierte. Con las 5 monedas de este proyecto, la mayoría de los pares NO "
            "cumplen esa condición de forma estable — la tabla de abajo lo muestra tal cual, sin maquillar.",
            styles["Body2"],
        )
    )
    story.append(
        Paragraph(
            "Por cada par se re-testea cointegración en ventanas móviles de ~1 año y se mide en qué fracción "
            "de esas ventanas siguió cointegrado. 'Operable' requiere al menos 60% de esas ventanas — un par "
            "cointegrado en una sola foto histórica pero no de forma consistente se marca como NO operable. "
            "Siempre calculado sobre velas diarias.",
            styles["Body2"],
        )
    )

    table_df = screen_pairs_stability()
    n_estables = int(table_df["estable"].sum())
    n_total = len(table_df)
    story.append(
        Paragraph(f"<b>{n_estables} de {n_total} pares operables</b> (fracción cointegrada rolling &gt;= 60%).", styles["Body2"])
    )

    rows = [["Par", "Dirección", "Fracción cointegrada", "Beta medio", "¿Operable?"]]
    for _, row in table_df.iterrows():
        rows.append(
            [
                str(row["par"]),
                str(row["direccion"]),
                f"{row['fraccion_cointegrada'] * 100:.0f}%",
                f"{row['beta_medio']:.3f}",
                "Sí" if row["estable"] else "No",
            ]
        )
    story.append(_text_table(rows, styles, col_widths=[3.2 * cm, 3.2 * cm, 3.5 * cm, 2.8 * cm, 2.3 * cm]))
    return story


# --------------------------------------------------------------------------
# CIERRE
# --------------------------------------------------------------------------


def _build_closing_section(styles) -> list:
    story: list = [Paragraph("Hallazgos del proyecto", styles["SectionHeading"])]
    paragraphs = [
        "Este proyecto evaluó formalmente, con datos reales y validación fuera de muestra, si algún método "
        "(indicadores técnicos clásicos, un modelo de Machine Learning con features técnicas y on-chain, "
        "arbitraje estadístico entre las 5 monedas de este universo) predice la DIRECCIÓN del precio de "
        "forma consistente. En ningún caso encontró una ventaja robusta sobre un baseline trivial.",
        "Lo que SÍ mostró evidencia consistente es que la VOLATILIDAD es más predecible que la dirección: "
        "los modelos GARCH capturan bien el 'clustering' de volatilidad (a un día volátil le sigue otro día "
        "volátil), lo que es útil para gestionar RIESGO — cuánto exponerse — aunque no diga nada sobre hacia "
        "dónde va a moverse el precio.",
        "En síntesis: usá este informe para entender el contexto histórico y de riesgo del activo, no como "
        "una señal de compra o venta.",
    ]
    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, styles["Body2"]))
    return story


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------


def build_report(asset: str, interval: str = "1d") -> bytes:
    """Arma el informe PDF completo para `asset` y devuelve los bytes del
    archivo (para que el llamador — `api/main.py::get_report` o un test —
    lo escriba a disco o lo devuelva como respuesta HTTP directamente, sin
    archivos temporales de por medio).

    Orden de las secciones: portada, riesgo (GARCH/VaR/ES/sizing), análisis
    técnico (indicadores + sugeridor), ciclos y estadística (drawdowns/fases
    de mercado/heatmap mensual/ADF), correlación entre monedas, arbitraje
    (screening de pares) y un cierre con los hallazgos del proyecto. Ver el
    docstring del módulo para el motivo de cada elección de librería y la
    advertencia de rendimiento (30-90s típico).
    """
    df = _load_df(asset, interval)
    close = df["close"]
    returns = simple_returns(close).dropna()

    df_daily = df if interval == REPORT_RISK_INTERVAL else _load_df(asset, REPORT_RISK_INTERVAL)

    generated_at = datetime.now(timezone.utc)
    styles = _build_styles()

    story: list = []
    story += _build_cover(asset, interval, close, generated_at, styles)
    story.append(PageBreak())
    story += _build_risk_section(asset, df_daily, styles)
    story.append(PageBreak())
    story += _build_technical_section(asset, interval, df, styles)
    story.append(PageBreak())
    story += _build_cycles_section(close, returns, styles)
    story.append(PageBreak())
    story += _build_correlation_section(asset, interval, close, styles)
    story.append(PageBreak())
    story += _build_arbitrage_section(styles)
    story.append(PageBreak())
    story += _build_closing_section(styles)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        title=f"Informe crypto-quant-desk — {asset}",
    )
    doc.build(story)
    return buffer.getvalue()
