"""Tests para api/main.py, con `fastapi.testclient.TestClient` sobre datos
REALES del snapshot local (`source="store"`, sin red — mismo patrón que el
resto del proyecto: `tests/test_models.py`, `tests/test_app_smoke.py`).

`/api/risk` y `/api/garch-series` ajustan un modelo GARCH por request (ver
el docstring de rendimiento en `api/main.py`) — son los tests más lentos de
este archivo, no es un problema del test, es inherente al endpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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
    ):
        assert len(body[campo]) == n

    assert set(body["pivotes"].keys()) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}
    assert set(body["soporte_resistencia"].keys()) == {"resistencia", "soporte", "precio_actual"}
    # con 120 velas ya pasó el warmup de rsi/ema: no debería haber NaN (null) al final
    assert body["rsi_14"][-1] is not None
    assert body["ema_12"][-1] is not None


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
        "/api/risk", "/api/backtest", "/api/garch-series",
    }
