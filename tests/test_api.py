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

from pathlib import Path

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
    assert set(body["timeframes"]) == {"1d", "1h"}


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


def test_get_studies_unknown_asset_returns_404() -> None:
    response = client.get("/api/studies", params={"asset": "DOGE"})

    assert response.status_code == 404


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


def test_get_risk_unknown_asset_returns_404() -> None:
    response = client.get("/api/risk", params={"asset": "DOGE"})

    assert response.status_code == 404


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

    assert "top_periodos" in body["periodograma"]
    assert len(body["periodograma"]["frecuencias"]) == len(body["periodograma"]["potencia"])

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
    assert response.json()["halvings_btc"] is None


def test_get_stats_unknown_asset_returns_404() -> None:
    response = client.get("/api/stats", params={"asset": "DOGE"})

    assert response.status_code == 404


def test_get_stats_invalid_interval_returns_400() -> None:
    response = client.get("/api/stats", params={"asset": "BTC", "interval": "5m"})

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
        "/api/risk", "/api/backtest", "/api/garch-series", "/api/prediction",
        "/api/data-status", "/api/refresh", "/api/stats",
    }
