"""Smoke test offline para app/main.py: instancia la ventana principal en
modo "offscreen" (sin display real, vía el plugin de Qt `offscreen`) y
verifica que no explota. NO testea interacción real (clicks, threads) — eso
requeriría un entorno gráfico real y datos de mercado; acá solo se confirma
que la UI se arma sin excepciones y que el selector de activos está bien
poblado.
"""

from __future__ import annotations

import os

# Tiene que fijarse ANTES de importar PySide6 (acá o vía la variable de
# entorno al correr pytest) para que Qt no intente abrir una ventana real.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from config import UNIVERSE


@pytest.fixture(scope="module")
def qapp():
    """Una sola QApplication para todo el módulo: Qt no permite crear más
    de una instancia por proceso.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_instantiates_without_exception(qapp) -> None:
    from app.main import MainWindow

    window = MainWindow()
    assert window is not None
    window.close()


def test_asset_combo_has_all_five_coins(qapp) -> None:
    from app.main import MainWindow

    window = MainWindow()
    items = [window.asset_combo.itemText(i) for i in range(window.asset_combo.count())]

    assert window.asset_combo.count() == 5
    assert set(items) == set(UNIVERSE.keys())
    window.close()


def test_window_has_analyze_button_and_disclaimer(qapp) -> None:
    from PySide6.QtWidgets import QLabel

    from app.main import DISCLAIMER_TEXT, MainWindow

    window = MainWindow()

    assert window.analyze_button.text() == "Analizar"
    assert window.analyze_button.isEnabled()

    disclaimer_label = window.findChild(QLabel, "disclaimer")
    assert disclaimer_label is not None
    assert disclaimer_label.text() == DISCLAIMER_TEXT
    window.close()


def test_risk_panel_starts_empty_and_resets_cleanly(qapp) -> None:
    from app.main import MainWindow

    window = MainWindow()
    for label in window.risk_panel._value_labels.values():
        assert label.text() == "—"

    window.risk_panel.reset()
    for label in window.risk_panel._value_labels.values():
        assert label.text() == "—"
    window.close()
