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
"""
