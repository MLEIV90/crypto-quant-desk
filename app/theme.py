"""Paleta y hoja de estilos (QSS) de la terminal — navy/oscuro, sobrio,
"tipo terminal". Un solo lugar para los colores, para que la ventana
principal y los widgets de gráficos usen exactamente los mismos.
"""

from __future__ import annotations

BACKGROUND = "#0f172a"
PANEL_BACKGROUND = "#1e293b"
BORDER = "#334155"
TEXT = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
ACCENT = "#38bdf8"
PRICE_LINE = "#38bdf8"
SMA_LINE = "#f59e0b"
VOL_LINE = "#a78bfa"

REGIME_COLORS: dict[str, str] = {
    "calma": "#22c55e",
    "normal": "#eab308",
    "tension": "#ef4444",
}
REGIME_COLOR_DEFAULT = "#94a3b8"

# Colores de dirección: LONG/COMPRAR = verde, FLAT/ESPERAR = gris, SHORT/VENDER
# = rojo — un solo mapeo para el semáforo de predicción (Fase 5b, valores
# LONG/FLAT/SHORT) y el sugeridor de consenso (Fase 7b, valores
# COMPRAR/VENDER/ESPERAR), para que ambos usen exactamente la misma paleta.
DIRECTION_COLORS: dict[str, str] = {
    "LONG": "#22c55e", "COMPRAR": "#22c55e",
    "FLAT": "#94a3b8", "ESPERAR": "#94a3b8",
    "SHORT": "#ef4444", "VENDER": "#ef4444",
}
DIRECTION_COLOR_DEFAULT = "#94a3b8"

# Paleta del panel de Análisis Técnico (Fase 7b): un color por línea/estudio
# superpuesto en el gráfico de velas, elegidos para distinguirse entre sí y
# del celeste/naranja ya usados en el resto del cockpit.
CANDLE_UP = "#22c55e"
CANDLE_DOWN = "#ef4444"
SMA20_LINE = "#f59e0b"
SMA50_LINE = "#eab308"
EMA12_LINE = "#38bdf8"
EMA26_LINE = "#818cf8"
BOLLINGER_LINE = "#a78bfa"
FIBONACCI_LINE = "#fbbf24"
SUPPORT_LINE = "#22c55e"
RESISTANCE_LINE = "#ef4444"
PIVOT_LINE = "#e2e8f0"
RSI_LINE = "#38bdf8"
MACD_LINE = "#38bdf8"
MACD_SIGNAL_LINE = "#f59e0b"
MACD_HIST_UP = "#22c55e"
MACD_HIST_DOWN = "#ef4444"
STOCH_K_LINE = "#38bdf8"
STOCH_D_LINE = "#f59e0b"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}}
QComboBox, QPushButton {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 12px;
    color: {TEXT};
}}
QComboBox:hover, QPushButton:hover {{
    background-color: {BORDER};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
}}
QLabel#disclaimer {{
    color: {TEXT_MUTED};
    font-style: italic;
    font-size: 9pt;
    padding: 6px 2px;
    border-top: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
}}
QLabel#statusLabel {{
    color: {ACCENT};
    font-weight: bold;
}}
QWidget#riskPanel {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QLabel#panelTitle {{
    font-size: 12pt;
    font-weight: bold;
    color: {ACCENT};
}}
QLabel#panelNote {{
    color: {TEXT_MUTED};
    font-size: 8pt;
    font-style: italic;
}}
QLabel#cardName {{
    color: {TEXT_MUTED};
    font-size: 9pt;
}}
QLabel#cardValue {{
    color: {TEXT};
    font-weight: bold;
    font-size: 10.5pt;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {BACKGROUND};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {PANEL_BACKGROUND};
    color: {TEXT_MUTED};
    padding: 8px 16px;
    border: 1px solid {BORDER};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background-color: {BACKGROUND};
    color: {ACCENT};
    font-weight: bold;
}}
QLabel#honestyWarning {{
    color: #fbbf24;
    font-size: 8.5pt;
    font-style: italic;
    padding: 6px;
    border: 1px solid #fbbf24;
    border-radius: 4px;
}}
QLabel#accionLabel {{
    font-size: 14pt;
    font-weight: bold;
}}
QLabel#semaforoDot {{
    padding-right: 4px;
}}
QTableWidget {{
    background-color: {PANEL_BACKGROUND};
    color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
}}
QTableWidget::item {{
    padding: 4px;
}}
QHeaderView::section {{
    background-color: {BACKGROUND};
    color: {TEXT_MUTED};
    padding: 4px;
    border: 1px solid {BORDER};
}}
QWidget#suggesterPanel {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QLabel#performanceHighlight {{
    background-color: {BACKGROUND};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 8px;
    font-size: 9pt;
}}
"""
