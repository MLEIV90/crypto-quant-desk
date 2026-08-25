"""Tests para api/main.py, con `fastapi.testclient.TestClient` sobre datos
REALES del snapshot local (`source="store"`, sin red — mismo patrón que el
resto del proyecto: `tests/test_models.py`, `tests/test_app_smoke.py`).

`/api/risk` y `/api/garch-series` ajustan un modelo GARCH por request (ver
el docstring de rendimiento en `api/main.py`) — son tests lentos, no es un
problema del test, es inherente al endpoint. `/api/prediction` (Fase 8c)
es el MÁS lento de todos (entrena XGBoost con validación purgeada) — se
usa SOL a propósito (solo técnicas, sin on-chain) para que el test no
tarde más de lo necesario.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from data import loaders
from data.snapshot import GeoblockedError, last_closed_candle_open_time

client = TestClient(app)


def _write_fake_store_snapshot(directory: Path, asset: str, interval: str, ultima_fecha: pd.Timestamp) -> None:
    """Snapshot mínimo de fixture para /api/data-status, con última fecha controlada."""
    idx = pd.date_range(end=ultima_fecha, periods=3, freq="D" if interval == "1d" else "h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0],
            "volume": [10.0, 20.0, 30.0],
        },
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_{interval}.parquet")


# --------------------------------------------------------------------------
# /api/assets
# --------------------------------------------------------------------------


def test_get_assets_returns_universe_and_timeframes() -> None:
    response = client.get("/api/assets")

    assert response.status_code == 200
    body = response.json()
    assert set(body["activos"]) == {"BTC", "ETH", "SOL", "BNB", "LTC"}
    assert set(body["timeframes"]) == {"1d", "1h", "4h", "1w"}


# --------------------------------------------------------------------------
# /api/ohlcv
# --------------------------------------------------------------------------


def test_get_ohlcv_respects_limit_and_is_sorted_ascending() -> None:
    response = client.get("/api/ohlcv", params={"asset": "BTC", "interval": "1d", "limit": 50})

    assert response.status_code == 200
    body = response.json()
    assert body["asset"] == "BTC"
    assert body["interval"] == "1d"
    assert len(body["velas"]) == 50

    fechas = [vela["fecha"] for vela in body["velas"]]
    assert fechas == sorted(fechas)

    primera = body["velas"][0]
    assert set(primera.keys()) == {"fecha", "open", "high", "low", "close", "volume"}
    assert isinstance(primera["close"], float)


def test_get_ohlcv_accepts_high_limit_for_full_history() -> None:
    # Fase 11: "Todo" ya no debe rebotar con 422 al pedir el histórico
    # horario completo (~58.000 velas) — antes el tope era le=5000.
    response = client.get("/api/ohlcv", params={"asset": "BTC", "interval": "1h", "limit": 60_000})

    assert response.status_code == 200
    body = response.json()
    # .tail(60_000) sobre un histórico más chico devuelve TODO lo disponible,
    # no rellena con datos inventados — por eso no se compara contra 60_000.
    assert len(body["velas"]) > 5_000


def test_get_ohlcv_unknown_asset_returns_404() -> None:
    response = client.get("/api/ohlcv", params={"asset": "DOGE"})

    assert response.status_code == 404
    assert "DOGE" in response.json()["detail"]


def test_get_ohlcv_invalid_interval_returns_400() -> None:
    response = client.get("/api/ohlcv", params={"asset": "BTC", "interval": "5m"})

    assert response.status_code == 400


def test_get_ohlcv_accepts_4h_derived_interval() -> None:
    # Fase 17b: "4h" se deriva por resampleo del snapshot horario — no hace
    # falta un snapshot propio, /api/ohlcv debe aceptarlo igual que "1d"/"1h".
    response = client.get("/api/ohlcv", params={"asset": "BTC", "interval": "4h", "limit": 30})

    assert response.status_code == 200
    body = response.json()
    assert body["interval"] == "4h"
    assert len(body["velas"]) == 30
    fechas = [vela["fecha"] for vela in body["velas"]]
    assert fechas == sorted(fechas)


def test_get_ohlcv_accepts_1w_derived_interval() -> None:
    # Fase 17b: "1w" se deriva por resampleo del snapshot diario.
    response = client.get("/api/ohlcv", params={"asset": "BTC", "interval": "1w", "limit": 20})

    assert response.status_code == 200
    body = response.json()
    assert body["interval"] == "1w"
    assert len(body["velas"]) == 20


def test_get_ohlcv_invalid_limit_returns_422() -> None:
    response = client.get("/api/ohlcv", params={"asset": "BTC", "limit": 0})

    assert response.status_code == 422


# --------------------------------------------------------------------------
# /api/studies
# --------------------------------------------------------------------------


def test_get_studies_returns_series_aligned_to_fechas() -> None:
    response = client.get("/api/studies", params={"asset": "BTC", "interval": "1d", "limit": 120})

    assert response.status_code == 200
    body = response.json()

    n = len(body["fechas"])
    assert n == 120
    for campo in (
        "sma_20", "sma_50", "ema_12", "ema_26", "bb_upper", "bb_mid", "bb_lower",
        "rsi_14", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
        "vwap", "obv", "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
        "ichimoku_senkou_b", "ichimoku_chikou",
    ):
        assert len(body[campo]) == n

    assert set(body["pivotes"].keys()) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}
    assert set(body["soporte_resistencia"].keys()) == {"resistencia", "soporte", "precio_actual"}
    # con 120 velas ya pasó el warmup de rsi/ema/vwap/ichimoku: no debería haber NaN (null) al final
    assert body["rsi_14"][-1] is not None
    assert body["ema_12"][-1] is not None
    assert body["vwap"][-1] is not None
    assert body["obv"][-1] is not None
    assert body["ichimoku_tenkan"][-1] is not None
    assert body["ichimoku_senkou_a"][-1] is not None
    # chikou en la última vela es SIEMPRE None por construcción (usa close
    # futuro que todavía no existe) — no es un fallo, ver signals/studies.py::ichimoku.
    assert body["ichimoku_chikou"][-1] is None


def test_get_studies_accepts_4h_and_1w_derived_intervals() -> None:
    # Fase 17b: los indicadores son todos por CANTIDAD DE VELAS (RSI/SMA/MACD
    # ventanas en barras, no en tiempo calendario), así que funcionan sobre
    # cualquier intervalo sin cambios — esto confirma que 4h/1w no rompen
    # el endpoint (no hace falta reimplementar nada indicador por indicador).
    for interval in ("4h", "1w"):
        response = client.get("/api/studies", params={"asset": "BTC", "interval": interval, "limit": 100})
        assert response.status_code == 200
        body = response.json()
        assert len(body["fechas"]) == 100
        assert len(body["sma_20"]) == 100


def test_get_studies_unknown_asset_returns_404() -> None:
    response = client.get("/api/studies", params={"asset": "DOGE"})

    assert response.status_code == 404


# --------------------------------------------------------------------------
# /api/volume-profile (Fase 13a)
# --------------------------------------------------------------------------


def test_get_volume_profile_returns_poc_and_value_area() -> None:
    response = client.get("/api/volume-profile", params={"asset": "BTC", "interval": "1d", "limit": 365, "bins": 40})

    assert response.status_code == 200
    body = response.json()

    assert body["asset"] == "BTC"
    assert body["interval"] == "1d"
    assert len(body["niveles_precio"]) == 40
    assert len(body["volumenes"]) == 40
    assert body["niveles_precio"] == sorted(body["niveles_precio"])
    assert body["value_area_low"] <= body["poc"] <= body["value_area_high"]
    assert body["volumen_total"] == pytest.approx(sum(body["volumenes"]), rel=1e-6)


def test_get_volume_profile_default_bins_is_50() -> None:
    response = client.get("/api/volume-profile", params={"asset": "ETH", "limit": 200})

    assert response.status_code == 200
    body = response.json()
    assert len(body["niveles_precio"]) == 50


def test_get_volume_profile_unknown_asset_returns_404() -> None:
    response = client.get("/api/volume-profile", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_get_volume_profile_invalid_interval_returns_400() -> None:
    response = client.get("/api/volume-profile", params={"asset": "BTC", "interval": "5m"})

    assert response.status_code == 400


# --------------------------------------------------------------------------
# /api/suggester
# --------------------------------------------------------------------------


def test_get_suggester_includes_desempeno_historico() -> None:
    response = client.get("/api/suggester", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()

    assert body["sugerencia"] in {"COMPRAR", "VENDER", "ESPERAR"}
    assert body["votos_alcistas"] + body["votos_bajistas"] + body["votos_neutrales"] == len(body["detalle"])
    assert 0.0 <= body["confianza"] <= 1.0

    desempeno = body["desempeno_historico"]
    assert set(desempeno.keys()) == {
        "cagr_sugeridor", "sharpe_sugeridor", "max_drawdown_sugeridor", "n_trades_sugeridor",
        "cagr_buy_and_hold", "sharpe_buy_and_hold", "max_drawdown_buy_and_hold",
    }


def test_get_suggester_unknown_asset_returns_404() -> None:
    response = client.get("/api/suggester", params={"asset": "DOGE"})

    assert response.status_code == 404


# --------------------------------------------------------------------------
# /api/risk (LENTO: ajusta un modelo GARCH)
# --------------------------------------------------------------------------


def test_get_risk_returns_expected_fields() -> None:
    response = client.get("/api/risk", params={"asset": "BTC"})

    assert response.status_code == 200
    body = response.json()

    assert body["asset"] == "BTC"
    assert body["accion"] in {"LONG", "FLAT", "SHORT"}
    assert body["regimen"] in {"calma", "normal", "tension", None}
    assert body["es95"] >= body["var95"]  # Expected Shortfall siempre >= VaR (misma convención de pérdida positiva)
    assert "/" in body["modelo_garch"]

    # Fase 20c: VaR/ES "actual" (GARCH, hoy) además del histórico (toda la
    # serie) — misma relación ES >= VaR, y con rótulos de base no vacíos.
    assert body["es95_actual"] >= body["var95_actual"]
    assert body["historico_basis"]
    assert body["actual_basis"]
    assert body["historico_basis"] != body["actual_basis"]
    assert body["regimen_basis"]

    # Fase 20a/20c: percentiles, todos en [0, 100] o None, con un rótulo de
    # base compartido no vacío (misma base que usa 'regimen').
    percentiles = body["percentiles"]
    for key in ("vol_realizada", "vol_garch", "var95", "es95"):
        value = percentiles[key]
        assert value is None or 0.0 <= value <= 100.0
    assert percentiles["base"]
    assert percentiles["base"] == body["regimen_basis"].split(",")[0].strip()

    # Fase 20a: histograma de retornos + marcas de VaR/ES (histórico, no actual).
    histograma = body["histograma"]
    assert len(histograma["bin_edges"]) == len(histograma["counts"]) + 1
    assert sum(histograma["counts"]) > 0
    assert histograma["es95_return"] <= histograma["var95_return"]  # la cola del ES es más extrema (más negativa)
    assert histograma["var95_return"] == pytest.approx(-body["var95"])
    assert histograma["es95_return"] == pytest.approx(-body["es95"])


def test_get_risk_actual_var_differs_from_historico_when_regime_shifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fase 20c: caso conocido -- una serie con un tramo de vol BAJA seguido
    # de un tramo de vol ALTA (justo antes de "hoy"). El VaR histórico
    # (toda la serie) queda "promediado" entre ambos tramos, pero el VaR
    # "actual" (GARCH, condicionado a la vela de hoy) debe reflejar el
    # tramo de vol ALTA reciente -- por construcción, tienen que diferir.
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    rng = np.random.default_rng(7)
    n_calm, n_stressed = 700, 60
    calm = rng.normal(0.0, 0.005, n_calm)
    stressed = rng.normal(0.0, 0.05, n_stressed)
    log_rets = np.concatenate([calm, stressed])
    prices = 100.0 * np.exp(np.cumsum(log_rets))

    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("D"), periods=len(prices) + 1, freq="D")
    close = np.concatenate([[100.0], prices])
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    )
    df.index.name = "timestamp"
    df.to_parquet(tmp_path / "BTC_1d.parquet")

    response = client.get("/api/risk", params={"asset": "BTC"})

    assert response.status_code == 200
    body = response.json()
    # El tramo reciente es mucho más volátil que el promedio de toda la
    # serie -> el VaR actual (GARCH) debe ser sensiblemente mayor que el
    # histórico, no casi igual (que sería la señal de que no está
    # reaccionando al régimen reciente en absoluto).
    assert body["var95_actual"] > body["var95"] * 1.2


def test_get_risk_unknown_asset_returns_404() -> None:
    response = client.get("/api/risk", params={"asset": "DOGE"})

    assert response.status_code == 404


# --------------------------------------------------------------------------
# /api/risk-summary (Fase 20b) — rápido a propósito (sin GARCH, ver
# api/main.py::get_risk_summary), así que no hace falta el mismo cuidado de
# "test lento" que /api/risk.
# --------------------------------------------------------------------------


def test_get_risk_summary_returns_one_row_per_asset() -> None:
    response = client.get("/api/risk-summary")

    assert response.status_code == 200
    body = response.json()
    filas = body["filas"]
    assert [fila["asset"] for fila in filas] == ["BTC", "ETH", "SOL", "BNB", "LTC"]

    for fila in filas:
        assert fila["vol_realizada"] > 0
        assert fila["var95"] > 0
        assert fila["regimen"] in {"calma", "normal", "tension", None}
        for percentil_key in ("vol_realizada_percentil", "var95_percentil"):
            value = fila[percentil_key]
            assert value is None or 0.0 <= value <= 100.0
        # Fase 20b: a propósito NO expone vol_garch (ver docstring del endpoint).
        assert "vol_garch" not in fila
        # Fase 20c: rótulos de base no vacíos, que ACLARAN explícitamente
        # que esto NO es GARCH (para no sugerir que son comparables con el
        # régimen/VaR GARCH de /api/risk).
        assert fila["regimen_basis"]
        assert fila["var95_basis"]
        assert "no GARCH" in fila["regimen_basis"]
        assert "no GARCH" in fila["var95_basis"]


def test_get_risk_summary_is_fast_without_garch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Si alguien reintroduce un ajuste GARCH acá por error, este test lo
    # detecta: select_best_model NUNCA debería llamarse desde este endpoint.
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("get_risk_summary no debería ajustar un modelo GARCH (ver su docstring)")

    monkeypatch.setattr(api_main, "select_best_model", fail_if_called)

    response = client.get("/api/risk-summary")

    assert response.status_code == 200


# --------------------------------------------------------------------------
# /api/backtest
# --------------------------------------------------------------------------


def test_get_backtest_returns_metrics_and_equity_curves() -> None:
    response = client.get("/api/backtest", params={"asset": "BTC"})

    assert response.status_code == 200
    body = response.json()

    expected_metric_keys = {"cagr", "sharpe", "sortino", "max_drawdown", "calmar", "n_trades", "turnover_total"}
    assert expected_metric_keys.issubset(body["metrics_estrategia"].keys())
    assert expected_metric_keys.issubset(body["metrics_buy_and_hold"].keys())
    assert len(body["equity_curve_estrategia"]) > 0
    assert len(body["equity_curve_buy_and_hold"]) > 0
    assert body["equity_curve_estrategia"][0]["valor"] == pytest.approx(1.0)
    # Fase 21: default sin `strategy` explícito sigue siendo "combo" — no
    # debe cambiar el resumen que ya consume RiskView desde antes de esta fase.
    assert body["strategy"] == "combo"
    assert body["cost_bps"] == pytest.approx(10.0)
    assert body["fecha_inicio"] is None and body["fecha_fin"] is None


def test_get_backtest_drawdown_curves_are_never_positive_and_start_at_first_date() -> None:
    response = client.get("/api/backtest", params={"asset": "BTC", "strategy": "vol_targeting"})

    assert response.status_code == 200
    body = response.json()

    assert len(body["drawdown_curve_estrategia"]) == len(body["equity_curve_estrategia"])
    assert all(point["valor"] <= 1e-9 for point in body["drawdown_curve_estrategia"])
    assert all(point["valor"] <= 1e-9 for point in body["drawdown_curve_buy_and_hold"])


def test_get_backtest_exposure_curve_is_long_only_for_vol_targeting() -> None:
    response = client.get("/api/backtest", params={"asset": "BTC", "strategy": "vol_targeting"})

    assert response.status_code == 200
    body = response.json()

    assert len(body["exposure_curve_estrategia"]) == len(body["equity_curve_estrategia"])
    assert all(-1e-9 <= point["valor"] <= 1.0 + 1e-9 for point in body["exposure_curve_estrategia"])
    assert "pct_tiempo_fuera" in body["metrics_estrategia"]


def test_get_backtest_exposure_curve_can_be_negative_for_engine() -> None:
    response = client.get("/api/backtest", params={"asset": "BTC", "strategy": "engine"})

    assert response.status_code == 200
    body = response.json()

    valores = [point["valor"] for point in body["exposure_curve_estrategia"]]
    assert any(v < 0.0 for v in valores)  # el engine SÍ puede ir corto, a diferencia de vol targeting


def test_get_backtest_strategy_selector_gives_different_results_per_strategy() -> None:
    responses = {
        strategy: client.get("/api/backtest", params={"asset": "BTC", "strategy": strategy}).json()
        for strategy in ("vol_targeting", "engine", "buy_and_hold")
    }

    for strategy, body in responses.items():
        assert body["strategy"] == strategy

    # buy_and_hold como "control": la estrategia debe coincidir con su
    # propio benchmark (misma posición constante = 1 en ambos lados).
    bh = responses["buy_and_hold"]
    assert bh["metrics_estrategia"]["cagr"] == pytest.approx(bh["metrics_buy_and_hold"]["cagr"])

    # vol targeting (siempre largo, tamaño variable) y el engine (dirección
    # variable, tamaño fijo) son estrategias genuinamente distintas sobre el
    # mismo activo — sus curvas de equity finales no deberían coincidir.
    vt_final = responses["vol_targeting"]["equity_curve_estrategia"][-1]["valor"]
    engine_final = responses["engine"]["equity_curve_estrategia"][-1]["valor"]
    assert vt_final != pytest.approx(engine_final)


def test_get_backtest_invalid_strategy_returns_400() -> None:
    response = client.get("/api/backtest", params={"asset": "BTC", "strategy": "no_existe"})
    assert response.status_code == 400


def test_get_backtest_cost_bps_override_changes_metrics() -> None:
    cheap = client.get("/api/backtest", params={"asset": "BTC", "strategy": "engine", "cost_bps": 0.0}).json()
    expensive = client.get(
        "/api/backtest", params={"asset": "BTC", "strategy": "engine", "cost_bps": 500.0}
    ).json()

    assert cheap["cost_bps"] == pytest.approx(0.0)
    assert expensive["cost_bps"] == pytest.approx(500.0)
    assert expensive["metrics_estrategia"]["total_return"] < cheap["metrics_estrategia"]["total_return"]


def test_get_backtest_target_vol_changes_vol_targeting_sizing() -> None:
    low_target = client.get(
        "/api/backtest", params={"asset": "BTC", "strategy": "vol_targeting", "target_vol": 0.1}
    ).json()
    high_target = client.get(
        "/api/backtest", params={"asset": "BTC", "strategy": "vol_targeting", "target_vol": 1.0}
    ).json()

    assert low_target["metrics_estrategia"]["exposicion_media"] < high_target["metrics_estrategia"]["exposicion_media"]


def test_get_backtest_date_range_restricts_equity_curve() -> None:
    full = client.get("/api/backtest", params={"asset": "BTC", "strategy": "engine"}).json()
    fecha_inicio = full["equity_curve_estrategia"][len(full["equity_curve_estrategia"]) // 2]["fecha"][:10]

    windowed = client.get(
        "/api/backtest",
        params={"asset": "BTC", "strategy": "engine", "fecha_inicio": fecha_inicio},
    ).json()

    assert len(windowed["equity_curve_estrategia"]) < len(full["equity_curve_estrategia"])
    assert windowed["equity_curve_estrategia"][0]["valor"] == pytest.approx(1.0)
    assert windowed["fecha_inicio"] is not None


def test_get_backtest_invalid_date_range_returns_400() -> None:
    response = client.get(
        "/api/backtest", params={"asset": "BTC", "strategy": "engine", "fecha_inicio": "no-es-una-fecha"}
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# /api/backtest-strategies
# --------------------------------------------------------------------------


def test_get_backtest_strategies_lists_the_required_strategies() -> None:
    response = client.get("/api/backtest-strategies")

    assert response.status_code == 200
    body = response.json()

    ids = {row["id"] for row in body["estrategias"]}
    assert {"vol_targeting", "engine", "buy_and_hold"}.issubset(ids)
    assert body["cost_bps_default"] == pytest.approx(10.0)

    for row in body["estrategias"]:
        assert row["descripcion"]
        assert row["objetivo"]
        assert row["tradeoff"]
        if row["tiene_target_vol"]:
            assert row["target_vol_default"] is not None
            assert row["target_vol_min"] < row["target_vol_default"] < row["target_vol_max"]
        else:
            assert row["target_vol_default"] is None


# --------------------------------------------------------------------------
# /api/garch-series (LENTO: ajusta un modelo GARCH)
# --------------------------------------------------------------------------


def test_get_garch_series_returns_series() -> None:
    response = client.get("/api/garch-series", params={"asset": "ETH"})

    assert response.status_code == 200
    body = response.json()

    assert body["asset"] == "ETH"
    assert len(body["fechas"]) == len(body["vol_condicional"])
    assert len(body["fechas"]) > 0
    assert "/" in body["modelo_garch"]

    # Fase 20a: una etiqueta de régimen por fecha, mismo largo que la serie.
    assert len(body["regimen_serie"]) == len(body["fechas"])
    assert set(v for v in body["regimen_serie"] if v is not None) <= {"calma", "normal", "tension"}
    assert body["regimen_serie"][-1] == body["regimen_actual"]


# --------------------------------------------------------------------------
# /api/prediction (Fase 8c, EL MÁS LENTO: entrena XGBoost con validación purgeada)
# --------------------------------------------------------------------------


def test_get_prediction_returns_expected_fields_without_onchain() -> None:
    # SOL a propósito: sin cobertura on-chain (más rápido que BTC/ETH), ver
    # docstring del módulo.
    response = client.get("/api/prediction", params={"asset": "SOL"})

    assert response.status_code == 200
    body = response.json()

    assert body["asset"] == "SOL"
    assert body["used_onchain"] is False
    assert body["onchain_columns"] == []
    assert body["prediccion_clase"] in {"LONG", "FLAT", "SHORT"}
    assert 0.0 <= body["prediccion_confianza"] <= 1.0
    assert set(body["prediccion_proba"].keys()) == {"LONG", "FLAT", "SHORT"}
    assert sum(body["prediccion_proba"].values()) == pytest.approx(1.0, abs=1e-6)
    assert isinstance(body["supera_azar"], bool)
    assert isinstance(body["supera_mayoritaria"], bool)
    assert len(body["top_features"]) > 0
    # cada feature es un par [nombre, importancia]
    assert len(body["top_features"][0]) == 2


def test_get_prediction_unknown_asset_returns_404() -> None:
    response = client.get("/api/prediction", params={"asset": "DOGE"})

    assert response.status_code == 404


# --------------------------------------------------------------------------
# /api/research-experiments (Fase 24) — LEE resultados guardados, no entrena/corre nada
# --------------------------------------------------------------------------


def test_get_research_experiments_returns_saved_rl_and_rotation_results() -> None:
    # Usa los archivos REALES ya guardados en rl/results/ y strategies/results/
    # (mismo criterio que el resto del proyecto: sin mocks cuando hay datos
    # reales disponibles) — si algún día no existen, el otro test de acá
    # abajo cubre el degradado con gracia.
    response = client.get("/api/research-experiments")

    assert response.status_code == 200
    body = response.json()

    assert body["rl"] is not None
    assert body["rl"]["conclusion"]["supera_a_todos_los_baselines_consistentemente"] is False
    estrategias = {row["estrategia"] for row in body["rl"]["summary_table"]}
    assert "RL (PPO)" in estrategias
    assert body["rl"]["fecha_experimento"] is not None

    assert body["rotation"] is not None
    assert body["rotation"]["conclusion"]["veredicto_global"] is False
    assert 0.0 <= body["rotation"]["conclusion"]["fraccion_pares_robustos"] <= 1.0
    # NaN del JSON original saneado a None (Fase 24, ver _nan_to_none) — nunca
    # un NaN no estándar en la respuesta HTTP.
    for row in body["rotation"]["summary_table"]:
        for key, value in row.items():
            assert not (isinstance(value, float) and value != value), f"{key} vino NaN sin sanear"


def test_get_research_experiments_degrades_gracefully_when_no_results_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "RL_RESULTS_DIR", tmp_path / "rl_vacio")
    monkeypatch.setattr(api_main, "ROTATION_RESULTS_DIR", tmp_path / "rotation_vacio")

    response = client.get("/api/research-experiments")

    assert response.status_code == 200
    body = response.json()
    assert body["rl"] is None
    assert body["rotation"] is None


def test_get_research_experiments_picks_the_most_recent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rl_dir = tmp_path / "rl_results"
    rl_dir.mkdir()
    old_payload = {
        "params": {
            "assets": ["BTC"], "min_train_days": 730, "n_blocks": 1,
            "seeds": [0], "total_timesteps": 100, "cost_bps": 10.0,
        },
        "elapsed_seconds": 1.0,
        "n_ppo_runs": 1,
        "oos_date_range": ["2020-01-01", "2020-02-01"],
        "blocks": [{"train_start": 0, "train_end": 10, "test_start": 10, "test_end": 20}],
        "summary_table": [
            {
                "estrategia": "RL (PPO)", "sharpe_media": 0.1, "sharpe_std": 0.0,
                "retorno_anualizado_media": 0.0, "retorno_anualizado_std": 0.0,
                "retorno_total_media": 0.0, "retorno_total_std": 0.0,
                "max_drawdown_media": 0.0, "max_drawdown_std": 0.0,
                "turnover_total_media": 0.0, "turnover_total_std": 0.0,
                "turnover_medio_diario_media": 0.0, "turnover_medio_diario_std": 0.0,
            }
        ],
        "conclusion": {
            "supera_a_todos_los_baselines_consistentemente": False,
            "sharpe_rl_peor_semilla": 0.1, "sharpe_rl_mejor_semilla": 0.1,
            "sharpe_baselines": {},
        },
    }
    new_payload = json.loads(json.dumps(old_payload))
    new_payload["conclusion"]["sharpe_rl_peor_semilla"] = 0.99

    (rl_dir / "rl_experiment_20200101_000000.json").write_text(json.dumps(old_payload), encoding="utf-8")
    (rl_dir / "rl_experiment_20260101_000000.json").write_text(json.dumps(new_payload), encoding="utf-8")
    monkeypatch.setattr(api_main, "RL_RESULTS_DIR", rl_dir)
    monkeypatch.setattr(api_main, "ROTATION_RESULTS_DIR", tmp_path / "rotation_vacio")

    response = client.get("/api/research-experiments")

    assert response.status_code == 200
    body = response.json()
    assert body["rl"]["conclusion"]["sharpe_rl_peor_semilla"] == pytest.approx(0.99)
    assert body["rl"]["fecha_experimento"].startswith("2026-01-01")


# --------------------------------------------------------------------------
# /api/data-status (Fase 9a)
# --------------------------------------------------------------------------


def test_get_data_status_returns_expected_fields() -> None:
    response = client.get("/api/data-status", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()
    assert body["asset"] == "BTC"
    assert body["interval"] == "1d"
    assert body["antiguedad_segundos"] >= 0
    assert isinstance(body["desactualizado"], bool)
    assert body["antiguedad_texto"].startswith("hace")


def test_get_data_status_unknown_asset_returns_404() -> None:
    response = client.get("/api/data-status", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_get_data_status_flags_stale_daily_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    old_date = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=5)
    _write_fake_store_snapshot(tmp_path, "BTC", "1d", old_date)

    response = client.get("/api/data-status", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()
    assert body["desactualizado"] is True
    assert body["antiguedad_segundos"] >= 4 * 24 * 3600
    assert "día" in body["antiguedad_texto"]


def test_get_data_status_flags_freshly_refreshed_daily_snapshot_as_up_to_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un snapshot que YA tiene la última vela cerrada disponible (lo que
    dejaría un `POST /api/refresh` recién corrido) no debe marcarse
    desactualizado, sin importar la hora del día en que se consulte —
    ver el docstring de `get_data_status`: comparar contra "ahora" directo
    marcaría esto como viejo casi siempre, porque una vela diaria SIEMPRE
    tiene entre 24 y 48hs de antigüedad de reloj por construcción (se
    indexa por su open time).
    """
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    freshest_possible = last_closed_candle_open_time("1d")
    _write_fake_store_snapshot(tmp_path, "BTC", "1d", freshest_possible)

    response = client.get("/api/data-status", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()
    assert body["desactualizado"] is False
    # el texto legible SÍ refleja la antigüedad real de reloj (24-48hs) —
    # eso es correcto y esperado, es un dato distinto de "desactualizado".
    assert body["antiguedad_segundos"] > 0


def test_get_data_status_flags_stale_hourly_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    old_date = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=5)
    _write_fake_store_snapshot(tmp_path, "BTC", "1h", old_date)

    response = client.get("/api/data-status", params={"asset": "BTC", "interval": "1h"})

    assert response.status_code == 200
    assert response.json()["desactualizado"] is True  # umbral horario: 2h, no 1 día


def test_get_data_status_4h_uses_base_hourly_freshness_with_wider_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fase 17b: "4h" se deriva del snapshot horario y evalúa su frescura
    # contra ESE snapshot base con un umbral propio (8h, el doble de "1h")
    # — los mismos datos que arriba se marcan "desactualizado" para
    # interval="1h" (5h de antigüedad > umbral de 2h) siguen "al día" para
    # interval="4h" (5h < umbral de 8h): una vista más gruesa tolera más
    # demora del dato base.
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    old_date = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=5)
    _write_fake_store_snapshot(tmp_path, "BTC", "1h", old_date)

    response = client.get("/api/data-status", params={"asset": "BTC", "interval": "4h"})

    assert response.status_code == 200
    body = response.json()
    assert body["interval"] == "4h"
    assert body["desactualizado"] is False


def test_get_data_status_4h_flags_stale_when_base_hourly_gap_exceeds_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    old_date = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=10)
    _write_fake_store_snapshot(tmp_path, "BTC", "1h", old_date)

    response = client.get("/api/data-status", params={"asset": "BTC", "interval": "4h"})

    assert response.status_code == 200
    assert response.json()["desactualizado"] is True  # 10h > umbral de 8h


def test_get_data_status_1w_uses_base_daily_freshness_with_wider_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mismo criterio que "4h" arriba, pero "1w" se deriva del snapshot
    # DIARIO con un umbral de 7 días (vs. 1 día para "1d" solo).
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    old_date = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=3)
    _write_fake_store_snapshot(tmp_path, "BTC", "1d", old_date)

    response = client.get("/api/data-status", params={"asset": "BTC", "interval": "1w"})

    assert response.status_code == 200
    body = response.json()
    assert body["interval"] == "1w"
    assert body["desactualizado"] is False  # 3 días < umbral de 7 días para "1w"


# --------------------------------------------------------------------------
# POST /api/refresh (Fase 9a) — mockeado, NUNCA pega a Binance en tests
# --------------------------------------------------------------------------


def test_post_refresh_returns_expected_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_update_snapshot(asset: str, interval: str) -> dict:
        assert asset == "BTC"
        assert interval == "1d"
        return {
            "filas_agregadas": 3,
            "ultima_fecha": pd.Timestamp("2026-08-16", tz="UTC"),
            "ya_actualizado": False,
        }

    monkeypatch.setattr(api_main, "update_snapshot", fake_update_snapshot)

    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()
    assert body["asset"] == "BTC"
    assert body["interval"] == "1d"
    assert body["filas_agregadas"] == 3
    assert body["ya_actualizado"] is False
    assert body["ultima_fecha"].startswith("2026-08-16")


def test_post_refresh_already_up_to_date(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_update_snapshot(asset: str, interval: str) -> dict:
        return {"filas_agregadas": 0, "ultima_fecha": pd.Timestamp("2026-08-16", tz="UTC"), "ya_actualizado": True}

    monkeypatch.setattr(api_main, "update_snapshot", fake_update_snapshot)

    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()
    assert body["filas_agregadas"] == 0
    assert body["ya_actualizado"] is True


def test_post_refresh_geoblocked_returns_502_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_update_snapshot(asset: str, interval: str) -> dict:
        raise GeoblockedError("Binance bloqueó la descarga por geolocalización (HTTP 451/403)")

    monkeypatch.setattr(api_main, "update_snapshot", fake_update_snapshot)

    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 502
    assert "geolocaliz" in response.json()["detail"].lower()


def test_post_refresh_network_failure_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_update_snapshot(asset: str, interval: str) -> dict:
        raise ConnectionError("no se pudo conectar a Binance")

    monkeypatch.setattr(api_main, "update_snapshot", fake_update_snapshot)

    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 502


def test_post_refresh_missing_snapshot_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_update_snapshot(asset: str, interval: str) -> dict:
        raise FileNotFoundError(f"No existe el snapshot local para {asset}")

    monkeypatch.setattr(api_main, "update_snapshot", fake_update_snapshot)

    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 404


def test_post_refresh_rejects_derived_interval_4h(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fase 17b: "4h" no tiene snapshot propio que bajar de Binance (se
    # deriva por resampleo del horario) — debe rechazarse con un 400 claro
    # en vez de propagar el ValueError crudo de data.snapshot.update_snapshot.
    def fake_update_snapshot(asset: str, interval: str) -> dict:
        raise AssertionError("update_snapshot no debería llamarse para un intervalo derivado")

    monkeypatch.setattr(api_main, "update_snapshot", fake_update_snapshot)

    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "4h"})

    assert response.status_code == 400
    assert "1h" in response.json()["detail"]


def test_post_refresh_rejects_derived_interval_1w(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_update_snapshot(asset: str, interval: str) -> dict:
        raise AssertionError("update_snapshot no debería llamarse para un intervalo derivado")

    monkeypatch.setattr(api_main, "update_snapshot", fake_update_snapshot)

    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "1w"})

    assert response.status_code == 400
    assert "1d" in response.json()["detail"]


def test_post_refresh_unknown_asset_returns_404() -> None:
    response = client.post("/api/refresh", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_post_refresh_invalid_interval_returns_400() -> None:
    response = client.post("/api/refresh", params={"asset": "BTC", "interval": "5m"})

    assert response.status_code == 400


# --------------------------------------------------------------------------
# /api/stats (Fase 11)
# --------------------------------------------------------------------------


def test_get_stats_daily_returns_expected_structure() -> None:
    response = client.get("/api/stats", params={"asset": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()

    assert body["asset"] == "BTC"
    assert body["interval"] == "1d"
    assert len(body["estacionalidad_mensual"]) > 0
    assert set(body["estacionalidad_mensual"][0].keys()) == {"bucket", "retorno_medio", "mediana", "desvio", "n"}
    assert len(body["estacionalidad_semanal"]) > 0
    # diario: sin estacionalidad horaria (todas las velas caerían en la hora 0).
    assert body["estacionalidad_horaria"] is None

    assert len(body["autocorrelacion"]) > 0
    assert body["autocorrelacion"][0]["lag"] == 0
    assert body["autocorrelacion"][0]["acf_retornos"] == pytest.approx(1.0)

    # Fase 15a: el periodograma ya NO forma parte de la respuesta.
    assert "periodograma" not in body

    for adf_key in ("adf_precio", "adf_retornos"):
        assert set(body[adf_key].keys()) == {
            "estadistico", "p_valor", "n_lags", "n_obs", "valores_criticos", "es_estacionaria",
        }
    # el precio de BTC (con años de tendencia) típicamente NO es estacionario;
    # los retornos sí — el hallazgo motivador de modelar sobre retornos.
    assert body["adf_precio"]["es_estacionaria"] is False
    assert body["adf_retornos"]["es_estacionaria"] is True

    # BTC: fechas de halving incluidas.
    assert body["halvings_btc"] == ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]

    # Fase 15a: drawdowns históricos, ordenados del más profundo al menos profundo.
    assert len(body["drawdowns"]) > 0
    profundidades = [d["profundidad_pct"] for d in body["drawdowns"]]
    assert profundidades == sorted(profundidades)
    assert all(p <= 0 for p in profundidades)
    primer_dd = body["drawdowns"][0]
    assert set(primer_dd.keys()) == {
        "fecha_pico", "fecha_fondo", "profundidad_pct", "fecha_recuperacion", "dias_caida", "dias_recuperacion",
    }

    # Fase 15a: fases de mercado bull/bear.
    assert len(body["fases_mercado"]) > 0
    assert all(f["tipo"] in ("bull", "bear") for f in body["fases_mercado"])
    assert body["fases_mercado"][-1]["confirmada"] is False  # la más reciente siempre queda "en curso"

    # Fase 15a: BTC tiene ciclos de halving, con el caveat de n chico.
    assert body["ciclos_halving"] is not None
    assert body["ciclos_halving"]["n_halvings_totales"] == 4
    assert body["ciclos_halving"]["n_halvings_con_datos"] <= 4
    assert len(body["ciclos_halving"]["ciclos"]) == body["ciclos_halving"]["n_halvings_con_datos"]

    # Fase 15a: heatmap mes x año.
    assert len(body["heatmap_mensual"]["matriz"]) == 12
    assert len(body["heatmap_mensual"]["anios"]) > 0
    assert len(body["heatmap_mensual"]["matriz"][0]) == len(body["heatmap_mensual"]["anios"])


def test_get_stats_hourly_includes_hourly_seasonality() -> None:
    response = client.get("/api/stats", params={"asset": "BTC", "interval": "1h"})

    assert response.status_code == 200
    body = response.json()
    assert body["estacionalidad_horaria"] is not None
    assert len(body["estacionalidad_horaria"]) > 0
    horas = {bucket["bucket"] for bucket in body["estacionalidad_horaria"]}
    assert horas.issubset(set(range(24)))


def test_get_stats_non_btc_asset_has_no_halvings() -> None:
    response = client.get("/api/stats", params={"asset": "ETH", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()
    assert body["halvings_btc"] is None
    assert body["ciclos_halving"] is None


def test_get_stats_unknown_asset_returns_404() -> None:
    response = client.get("/api/stats", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_get_stats_invalid_interval_returns_400() -> None:
    response = client.get("/api/stats", params={"asset": "BTC", "interval": "5m"})

    assert response.status_code == 400


# --------------------------------------------------------------------------
# /api/compare (Fase 12a)
# --------------------------------------------------------------------------


def test_get_compare_all_series_start_at_100() -> None:
    response = client.get("/api/compare", params={"assets": "BTC,ETH,SOL", "interval": "1d", "limit": 365})

    assert response.status_code == 200
    body = response.json()

    assert body["assets"] == ["BTC", "ETH", "SOL"]
    assert body["interval"] == "1d"
    assert set(body["series"].keys()) == {"BTC", "ETH", "SOL"}

    n = len(body["fechas"])
    assert n > 0
    for asset in ("BTC", "ETH", "SOL"):
        assert len(body["series"][asset]) == n
        assert body["series"][asset][0] == pytest.approx(100.0)

    assert set(body["rendimiento_total_pct"].keys()) == {"BTC", "ETH", "SOL"}
    # el rendimiento total del período debe ser consistente con el último
    # valor de la serie normalizada (arranca en 100).
    for asset in ("BTC", "ETH", "SOL"):
        assert body["rendimiento_total_pct"][asset] == pytest.approx(body["series"][asset][-1] - 100.0, abs=1e-6)


def test_get_compare_respects_limit() -> None:
    response = client.get("/api/compare", params={"assets": "BTC,ETH", "interval": "1d", "limit": 30})

    assert response.status_code == 200
    body = response.json()
    assert len(body["fechas"]) == 30


def test_get_compare_lowercases_are_normalized_to_uppercase() -> None:
    response = client.get("/api/compare", params={"assets": "btc,eth", "interval": "1d", "limit": 10})

    assert response.status_code == 200
    assert response.json()["assets"] == ["BTC", "ETH"]


def test_get_compare_single_asset_is_allowed() -> None:
    response = client.get("/api/compare", params={"assets": "BTC", "interval": "1d", "limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert body["assets"] == ["BTC"]
    assert body["series"]["BTC"][0] == pytest.approx(100.0)


def test_get_compare_unknown_asset_returns_404() -> None:
    response = client.get("/api/compare", params={"assets": "BTC,DOGE", "interval": "1d"})

    assert response.status_code == 404


def test_get_compare_invalid_interval_returns_400() -> None:
    response = client.get("/api/compare", params={"assets": "BTC,ETH", "interval": "5m"})

    assert response.status_code == 400


def test_get_compare_empty_assets_returns_400() -> None:
    response = client.get("/api/compare", params={"assets": ""})

    assert response.status_code == 400


# --------------------------------------------------------------------------
# /api/correlation (Fase 13b)
# --------------------------------------------------------------------------


def test_get_correlation_diagonal_is_one_and_symmetric() -> None:
    response = client.get("/api/correlation", params={"interval": "1d", "limit": 365, "method": "pearson"})

    assert response.status_code == 200
    body = response.json()

    assert body["interval"] == "1d"
    assert body["method"] == "pearson"
    assert set(body["activos"]) == {"BTC", "ETH", "SOL", "BNB", "LTC"}
    n = len(body["activos"])
    matriz = body["matriz"]
    assert len(matriz) == n
    assert all(len(row) == n for row in matriz)

    for i in range(n):
        assert matriz[i][i] == pytest.approx(1.0, abs=1e-6)
        for j in range(n):
            assert matriz[i][j] == pytest.approx(matriz[j][i], abs=1e-6)

    assert body["fechas_n"] > 0


def test_get_correlation_spearman_method() -> None:
    response = client.get("/api/correlation", params={"method": "spearman"})

    assert response.status_code == 200
    assert response.json()["method"] == "spearman"


def test_get_correlation_invalid_method_returns_400() -> None:
    response = client.get("/api/correlation", params={"method": "kendall"})

    assert response.status_code == 400


def test_get_correlation_invalid_interval_returns_400() -> None:
    response = client.get("/api/correlation", params={"interval": "5m"})

    assert response.status_code == 400


def test_get_correlation_respects_limit() -> None:
    response = client.get("/api/correlation", params={"limit": 30})

    assert response.status_code == 200
    assert response.json()["fechas_n"] <= 30


# --------------------------------------------------------------------------
# /api/pairs/screening y /api/pairs/detail (Fase 12b)
# --------------------------------------------------------------------------


def test_get_pairs_screening_returns_ranked_table() -> None:
    response = client.get("/api/pairs/screening")

    assert response.status_code == 200
    body = response.json()
    assert body["interval"] == "1d"
    assert body["n_total"] > 0
    assert body["n_total"] == len(body["filas"])
    assert body["n_estables"] <= body["n_total"]

    fracciones = [fila["fraccion_cointegrada"] for fila in body["filas"]]
    assert fracciones == sorted(fracciones, reverse=True)

    for fila in body["filas"]:
        assert "-" in fila["par"]
        assert "~" in fila["direccion"]
        assert 0.0 <= fila["fraccion_cointegrada"] <= 1.0
        assert isinstance(fila["estable"], bool)
        assert fila["estable"] == (fila["fraccion_cointegrada"] >= 0.6)


def test_get_pairs_screening_rejects_non_daily_interval() -> None:
    response = client.get("/api/pairs/screening", params={"interval": "1h"})

    assert response.status_code == 400


def test_get_pairs_detail_returns_cointegration_spread_and_zscore() -> None:
    response = client.get("/api/pairs/detail", params={"asset_y": "ETH", "asset_x": "BTC", "interval": "1d"})

    assert response.status_code == 200
    body = response.json()

    assert body["asset_y"] == "ETH"
    assert body["asset_x"] == "BTC"
    assert body["interval"] == "1d"
    assert isinstance(body["beta"], float)
    assert isinstance(body["es_cointegrado"], bool)
    assert 0.0 <= body["p_valor_adf"] <= 1.0

    n = len(body["fechas"])
    assert n > 0
    assert len(body["spread"]) == n
    assert len(body["zscore"]) == n
    # el z-score expansivo tiene warmup en NaN/null al principio de la serie
    assert body["zscore"][0] is None
    assert body["zscore"][-1] is not None
    assert body["zscore_actual"] == pytest.approx(body["zscore"][-1])
    assert "z=" in body["zscore_interpretacion"]

    if body["estabilidad"] is not None:
        assert 0.0 <= body["estabilidad"]["fraccion_cointegrada"] <= 1.0
        assert body["estabilidad_mensaje"] is None
    else:
        assert body["estabilidad_mensaje"] is not None

    # Fase 15b: extremos de z-score, todos con |z| >= 2.
    for extremo in body["zscore_extremos"]:
        assert abs(extremo["z"]) >= 2.0

    # Fase 15b: backtest del par.
    backtest = body["backtest"]
    assert len(backtest["fechas"]) == len(backtest["equity_curve"])
    assert len(backtest["fechas"]) > 0
    assert backtest["equity_curve"][0] == pytest.approx(1.0)
    expected_metric_keys = {
        "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
        "turnover_total", "turnover_medio_diario", "n_trades", "exposicion_media", "hit_rate",
    }
    assert set(backtest["metrics"].keys()) == expected_metric_keys


def test_get_pairs_detail_backtest_params_change_the_result() -> None:
    default_response = client.get("/api/pairs/detail", params={"asset_y": "ETH", "asset_x": "BTC"})
    strict_response = client.get(
        "/api/pairs/detail", params={"asset_y": "ETH", "asset_x": "BTC", "bt_entry": 10.0, "bt_stop": 20.0}
    )

    assert default_response.status_code == 200
    assert strict_response.status_code == 200
    # con un umbral de entrada inalcanzable, el backtest no debería operar.
    assert strict_response.json()["backtest"]["metrics"]["n_trades"] == 0


def test_get_pairs_detail_invalid_backtest_params_returns_400() -> None:
    # exit < entry < stop no se cumple (entry=10 > stop=3, el default) -> 400, no 500.
    response = client.get(
        "/api/pairs/detail", params={"asset_y": "ETH", "asset_x": "BTC", "bt_entry": 10.0}
    )

    assert response.status_code == 400


def test_get_pairs_detail_same_asset_returns_400() -> None:
    response = client.get("/api/pairs/detail", params={"asset_y": "BTC", "asset_x": "BTC"})

    assert response.status_code == 400


def test_get_pairs_detail_unknown_asset_returns_404() -> None:
    response = client.get("/api/pairs/detail", params={"asset_y": "DOGE", "asset_x": "BTC"})

    assert response.status_code == 404


def test_get_pairs_detail_invalid_interval_returns_400() -> None:
    response = client.get("/api/pairs/detail", params={"asset_y": "ETH", "asset_x": "BTC", "interval": "5m"})

    assert response.status_code == 400


# --------------------------------------------------------------------------
# /api/report (Fase 16b, informe PDF descargable) — LENTO: ajusta un GARCH
# y corre cointegración rolling sobre todos los pares (ver
# reports/pdf_report.py). Un solo test end-to-end alcanza para confirmar
# que el archivo es un PDF válido y descargable; el detalle de cada
# sección se prueba directo sobre `build_report` en tests/test_pdf_report.py.
# --------------------------------------------------------------------------


def test_get_report_returns_downloadable_pdf() -> None:
    response = client.get("/api/report", params={"asset": "ETH", "interval": "1d"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment" in response.headers["content-disposition"]
    assert "informe_ETH_1d.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")


def test_get_report_unknown_asset_returns_404() -> None:
    response = client.get("/api/report", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_get_report_invalid_interval_returns_400() -> None:
    response = client.get("/api/report", params={"asset": "BTC", "interval": "5m"})

    assert response.status_code == 400


# --------------------------------------------------------------------------
# /api/export/* (Fase 17a, CSV descargable) — mismos cálculos que sus
# endpoints JSON hermanos (/api/ohlcv, /api/stats, /api/correlation), solo
# cambia el formato de salida.
# --------------------------------------------------------------------------


def test_export_ohlcv_returns_downloadable_csv_with_expected_columns() -> None:
    response = client.get("/api/export/ohlcv", params={"asset": "BTC", "interval": "1d", "limit": 50})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "ohlcv_BTC_1d.csv" in response.headers["content-disposition"]

    lines = response.text.strip().splitlines()
    assert lines[0] == "fecha,open,high,low,close,volume"
    assert len(lines) == 51  # encabezado + 50 velas


def test_export_ohlcv_unknown_asset_returns_404() -> None:
    response = client.get("/api/export/ohlcv", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_export_drawdowns_returns_downloadable_csv_with_expected_columns() -> None:
    response = client.get("/api/export/drawdowns", params={"asset": "BTC", "top_n": 5})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "drawdowns_BTC_1d.csv" in response.headers["content-disposition"]

    lines = response.text.strip().splitlines()
    assert lines[0] == "fecha_pico,fecha_fondo,profundidad_pct,fecha_recuperacion,dias_caida,dias_recuperacion"
    assert len(lines) - 1 <= 5


def test_export_drawdowns_unknown_asset_returns_404() -> None:
    response = client.get("/api/export/drawdowns", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_export_correlation_returns_downloadable_csv_with_all_assets() -> None:
    response = client.get("/api/export/correlation", params={"interval": "1d", "limit": 200})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    lines = response.text.strip().splitlines()
    assert lines[0] == "activo,BTC,ETH,SOL,BNB,LTC"
    assert len(lines) == 6  # encabezado + 5 activos


def test_export_correlation_invalid_method_returns_400() -> None:
    response = client.get("/api/export/correlation", params={"method": "kendall"})

    assert response.status_code == 400


# --------------------------------------------------------------------------
# Documentación automática (Swagger/OpenAPI)
# --------------------------------------------------------------------------


def test_docs_and_openapi_schema_are_available() -> None:
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    paths = openapi_response.json()["paths"]
    assert set(paths.keys()) == {
        "/api/assets", "/api/ohlcv", "/api/studies", "/api/suggester",
        "/api/risk", "/api/backtest", "/api/backtest-strategies", "/api/garch-series", "/api/prediction",
        "/api/research-experiments",
        "/api/data-status", "/api/refresh", "/api/stats", "/api/compare",
        "/api/pairs/screening", "/api/pairs/detail", "/api/volume-profile", "/api/correlation",
        "/api/report", "/api/export/ohlcv", "/api/export/drawdowns", "/api/export/correlation",
        "/api/risk-summary",
    }
