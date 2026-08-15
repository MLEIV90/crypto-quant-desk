"""Smoke test offline para app/main.py: instancia la ventana principal en
modo "offscreen" (sin display real, vía el plugin de Qt `offscreen`) y
verifica que no explota. NO testea interacción real (clicks, threads) — eso
requeriría un entorno gráfico real; acá solo se confirma que la UI se arma
sin excepciones y que los selectores/pestañas están bien poblados.

La excepción son `BacktestWorker` (Fase 4b), `PredictionWorker` (Fase 5b) y
`StudiesWorker` (Fase 7b): SÍ se corren directamente (sin UI ni
`QThread.start()`, como ya se hacía para validar `AnalysisWorker`) sobre un
activo real (`source="store"`), para confirmar que sus resultados tienen
las claves que pintan `BacktestPanel`/`PredictionPanel`/
`TechnicalAnalysisPanel` — esto sí pega contra datos reales (locales, del
snapshot), no es un test puramente offline de UI.
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


def test_cockpit_has_the_four_tabs(qapp) -> None:
    from app.main import MainWindow

    window = MainWindow()
    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]

    assert tabs == ["Riesgo", "Análisis Técnico", "Backtest", "Research (sin edge)"]
    window.close()


def test_technical_analysis_panel_has_selectors_and_honesty_warning(qapp) -> None:
    from PySide6.QtWidgets import QLabel

    from app.main import MainWindow
    from app.widgets.technical_analysis_panel import HONESTY_TEXT, TIMEFRAMES
    from config import UNIVERSE

    window = MainWindow()
    panel = window.technical_analysis_panel

    asset_items = [panel.asset_combo.itemText(i) for i in range(panel.asset_combo.count())]
    assert set(asset_items) == set(UNIVERSE.keys())

    timeframe_items = [panel.timeframe_combo.itemText(i) for i in range(panel.timeframe_combo.count())]
    assert timeframe_items == [label for label, _ in TIMEFRAMES]

    assert panel.analyze_button.text() == "Analizar"

    warning_label = panel.findChild(QLabel, "honestyWarning")
    assert warning_label is not None
    assert warning_label.text() == HONESTY_TEXT

    # El panel del sugeridor arranca vacío.
    assert panel.suggester_panel.suggestion_label.text() == "—"
    for label in panel.suggester_panel._detail_labels.values():
        assert label.text() == "—"
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


def test_prediction_panel_has_asset_combo_and_features_table(qapp) -> None:
    from app.main import MainWindow
    from app.widgets.prediction_panel import HONESTY_TEXT
    from config import UNIVERSE

    window = MainWindow()
    panel = window.prediction_panel
    items = [panel.asset_combo.itemText(i) for i in range(panel.asset_combo.count())]

    assert set(items) == set(UNIVERSE.keys())
    assert panel.predict_button.text() == "Predecir"
    assert panel.clase_label.text() == "—"
    assert panel.features_table.columnCount() == 2

    warning_label = panel.findChild(type(panel.clase_label), "honestyWarning")
    assert warning_label is not None
    assert warning_label.text() == HONESTY_TEXT
    window.close()


def test_prediction_worker_returns_expected_fields_on_real_data() -> None:
    """Corre `PredictionWorker._predict` directo (sin QThread ni UI) sobre
    SOL real (técnicas solas — sin cobertura on-chain, más rápido que
    BTC/ETH con on-chain) y verifica que el resultado tiene todo lo que
    pinta `PredictionPanel`, incluida la aclaración honesta de si el modelo
    le gana o no a los baselines.
    """
    from app.workers import PredictionWorker

    worker = PredictionWorker("SOL")
    resultado = worker._predict("SOL")

    assert resultado.asset == "SOL"
    assert resultado.used_onchain is False
    assert resultado.onchain_columns == []
    assert resultado.prediccion_clase in {"LONG", "FLAT", "SHORT"}
    assert 0.0 <= resultado.prediccion_confianza <= 1.0
    assert set(resultado.prediccion_proba.keys()) == {"LONG", "FLAT", "SHORT"}
    assert sum(resultado.prediccion_proba.values()) == pytest.approx(1.0, abs=1e-6)
    assert isinstance(resultado.supera_azar, bool)
    assert isinstance(resultado.supera_mayoritaria, bool)
    assert len(resultado.top_features) > 0


def test_studies_worker_returns_expected_fields_on_real_data() -> None:
    """Corre `StudiesWorker._compute` directo (sin QThread ni UI) sobre BTC
    diario real y verifica que el resultado tiene todo lo que pintan
    `TechnicalChartCanvas`/`SuggesterPanel`: estudios (Fase 7a) + sugerencia
    de consenso, con su desempeño histórico siempre presente.
    """
    from app.workers import StudiesWorker

    worker = StudiesWorker("BTC", "1d")
    resultado = worker._compute("BTC", "1d")

    assert resultado.asset == "BTC"
    assert resultado.timeframe == "1d"
    assert list(resultado.ohlcv_recent.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(resultado.ohlcv_recent) > 0

    for series in (
        resultado.sma_20, resultado.sma_50, resultado.ema_12, resultado.ema_26,
        resultado.bb_upper, resultado.bb_mid, resultado.bb_lower, resultado.rsi_14,
        resultado.macd, resultado.macd_signal, resultado.macd_hist,
        resultado.stoch_k, resultado.stoch_d,
    ):
        assert series.index.equals(resultado.ohlcv_recent.index)

    assert set(resultado.pivotes.keys()) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}
    assert set(resultado.soporte_resistencia.keys()) == {"resistencia", "soporte", "precio_actual"}

    sugerencia = resultado.sugerencia
    assert sugerencia["sugerencia"] in {"COMPRAR", "VENDER", "ESPERAR"}
    assert set(sugerencia["desempeno_historico"].keys()) == {
        "cagr_sugeridor", "sharpe_sugeridor", "max_drawdown_sugeridor", "n_trades_sugeridor",
        "cagr_buy_and_hold", "sharpe_buy_and_hold", "max_drawdown_buy_and_hold",
    }


def test_studies_worker_works_on_hourly_timeframe_too() -> None:
    """Mismo worker, timeframe horario (Fase 6b) — confirma que el recorte
    a la ventana reciente (`TECHNICAL_CHART_RECENT_CANDLES`) funciona igual
    de bien sobre 58.000 velas horarias que sobre ~3.000 diarias.
    """
    from app.workers import TECHNICAL_CHART_RECENT_CANDLES, StudiesWorker

    worker = StudiesWorker("ETH", "1h")
    resultado = worker._compute("ETH", "1h")

    assert resultado.timeframe == "1h"
    assert len(resultado.ohlcv_recent) == TECHNICAL_CHART_RECENT_CANDLES
