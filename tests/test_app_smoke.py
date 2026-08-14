"""Smoke test offline para app/main.py: instancia la ventana principal en
modo "offscreen" (sin display real, vía el plugin de Qt `offscreen`) y
verifica que no explota. NO testea interacción real (clicks, threads) — eso
requeriría un entorno gráfico real; acá solo se confirma que la UI se arma
sin excepciones y que los selectores/pestañas están bien poblados.

La excepción es `BacktestWorker` (Fase 4b): SÍ se corre directamente (sin
UI ni QThread.start(), como ya se hacía para validar `AnalysisWorker`) sobre
un activo real (`source="store"`), para confirmar que el dict de métricas
tiene las claves que usa `app.widgets.backtest_panel.BacktestPanel` — esto
sí pega contra datos reales, no es un test puramente offline de UI.
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


def test_cockpit_has_the_three_tabs(qapp) -> None:
    from app.main import MainWindow

    window = MainWindow()
    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]

    assert tabs == ["Riesgo", "Señales", "Backtest"]
    window.close()


def test_signals_panel_has_semaforo_and_honesty_warning(qapp) -> None:
    from app.main import MainWindow
    from app.widgets.signals_panel import HONESTY_TEXT

    window = MainWindow()
    panel = window.signals_panel

    assert panel.accion_label.text() == "—"
    for label in panel._value_labels.values():
        assert label.text() == "—"

    warning_label = panel.findChild(type(panel.accion_label), "honestyWarning")
    assert warning_label is not None
    assert warning_label.text() == HONESTY_TEXT
    window.close()


def test_backtest_panel_has_asset_combo_and_metrics_table(qapp) -> None:
    from app.main import MainWindow
    from config import UNIVERSE

    window = MainWindow()
    panel = window.backtest_panel
    items = [panel.asset_combo.itemText(i) for i in range(panel.asset_combo.count())]

    assert set(items) == set(UNIVERSE.keys())
    assert panel.run_button.text() == "Correr backtest"
    assert panel.metrics_table.rowCount() > 0
    assert panel.metrics_table.columnCount() == 2
    window.close()


def test_backtest_worker_returns_expected_metric_keys_on_real_data() -> None:
    """Corre `BacktestWorker._run_backtest` directo (sin QThread ni UI, como
    ya se hace para `AnalysisWorker` en la validación manual de la Fase 4a)
    sobre BTC real, y verifica que el resultado tiene todo lo que pinta
    `BacktestPanel`.
    """
    from app.workers import BacktestWorker

    worker = BacktestWorker("BTC")
    resultado = worker._run_backtest("BTC")

    expected_keys = {"cagr", "sharpe", "sortino", "max_drawdown", "calmar", "n_trades", "turnover_total"}
    assert expected_keys.issubset(resultado.metrics_estrategia.keys())
    assert expected_keys.issubset(resultado.metrics_buy_and_hold.keys())
    assert len(resultado.equity_curve_estrategia) > 0
    assert len(resultado.equity_curve_buy_and_hold) > 0
    assert resultado.equity_curve_estrategia.iloc[0] == pytest.approx(1.0)
    assert resultado.equity_curve_buy_and_hold.iloc[0] == pytest.approx(1.0)
