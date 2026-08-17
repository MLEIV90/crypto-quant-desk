"""Tests offline (sin red) para data/snapshot.py (Fase 9a)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests

from data import loaders, snapshot


def _write_fake_snapshot(directory: Path, asset: str, interval: str, index: pd.DatetimeIndex) -> None:
    closes = [float(i + 1) for i in range(len(index))]
    df = pd.DataFrame(
        {
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": [10.0 * (i + 1) for i in range(len(index))],
        },
        index=index,
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_{interval}.parquet")


# --------------------------------------------------------------------------
# update_snapshot: caso feliz, detecta desde dónde bajar
# --------------------------------------------------------------------------


def test_update_snapshot_downloads_only_new_candles_since_last_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)

    # Snapshot existente: última vela guardada hace 3 días (bien vieja para diario).
    last_saved = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=3)
    idx = pd.date_range(end=last_saved, periods=5, freq="D", tz="UTC")
    _write_fake_snapshot(tmp_path, "BTC", "1d", idx)

    captured: dict = {}

    def fake_load_binance(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        captured["symbol"] = symbol
        captured["interval"] = interval
        captured["start"] = pd.Timestamp(start)
        captured["end"] = pd.Timestamp(end)
        new_idx = pd.date_range(start=last_saved + pd.Timedelta(days=1), periods=2, freq="D", tz="UTC")
        return pd.DataFrame(
            {"open": [15.0, 16.0], "high": [15.0, 16.0], "low": [15.0, 16.0], "close": [15.0, 16.0], "volume": [1.0, 2.0]},
            index=new_idx,
        )

    monkeypatch.setattr(loaders, "_load_binance", fake_load_binance)

    result = snapshot.update_snapshot("BTC", "1d")

    # Pidió a Binance exactamente desde la última fecha guardada (no desde el
    # principio del histórico) hasta la última vela cerrada (~hoy - 1 día).
    assert captured["symbol"] == "BTCUSDT"
    assert captured["start"] == last_saved
    assert captured["end"] == pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)

    assert result["filas_agregadas"] == 2
    assert result["ya_actualizado"] is False
    assert result["ultima_fecha"] == last_saved + pd.Timedelta(days=2)

    updated = pd.read_parquet(tmp_path / "BTC_1d.parquet")
    assert len(updated) == 7  # 5 originales + 2 nuevas
    assert updated["close"].iloc[-1] == pytest.approx(16.0)
    assert updated.index.is_monotonic_increasing
    assert not updated.index.duplicated().any()


def test_update_snapshot_replaces_last_saved_candle_instead_of_duplicating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La última vela guardada se vuelve a pedir a propósito (por si estaba
    incompleta) — el resultado debe reemplazarla, no duplicarla.
    """
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)

    last_saved = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=2)
    idx = pd.date_range(end=last_saved, periods=3, freq="D", tz="UTC")
    _write_fake_snapshot(tmp_path, "BTC", "1d", idx)  # última vela: close=3.0

    def fake_load_binance(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        # Binance devuelve una versión actualizada de la última vela guardada
        # (close cambió de 3.0 a 3.5, como pasaría si esa vela no había
        # cerrado del todo cuando se guardó) más una vela nueva.
        new_idx = pd.date_range(start=last_saved, periods=2, freq="D", tz="UTC")
        return pd.DataFrame(
            {"open": [3.0, 4.0], "high": [3.5, 4.0], "low": [3.0, 4.0], "close": [3.5, 4.0], "volume": [30.0, 40.0]},
            index=new_idx,
        )

    monkeypatch.setattr(loaders, "_load_binance", fake_load_binance)

    result = snapshot.update_snapshot("BTC", "1d")

    assert result["filas_agregadas"] == 1  # solo la vela genuinamente nueva

    updated = pd.read_parquet(tmp_path / "BTC_1d.parquet")
    assert len(updated) == 4  # 3 originales, la última reemplazada (no +1), más 1 nueva
    assert updated["close"].iloc[-2] == pytest.approx(3.5)  # reemplazada, no duplicada
    assert updated["close"].iloc[-1] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# update_snapshot: ya está al día
# --------------------------------------------------------------------------


def test_update_snapshot_returns_ya_actualizado_when_no_new_closed_candles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)

    last_saved = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
    idx = pd.date_range(end=last_saved, periods=3, freq="D", tz="UTC")
    _write_fake_snapshot(tmp_path, "BTC", "1d", idx)

    def fake_load_binance(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("no debería pegarle a Binance si el snapshot ya está al día")

    monkeypatch.setattr(loaders, "_load_binance", fake_load_binance)

    result = snapshot.update_snapshot("BTC", "1d")

    assert result == {"filas_agregadas": 0, "ultima_fecha": last_saved, "ya_actualizado": True}


# --------------------------------------------------------------------------
# update_snapshot: geobloqueo y errores
# --------------------------------------------------------------------------


def test_update_snapshot_raises_geoblocked_error_on_http_451(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    last_saved = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=2)
    idx = pd.date_range(end=last_saved, periods=3, freq="D", tz="UTC")
    _write_fake_snapshot(tmp_path, "BTC", "1d", idx)

    class _FakeResponse:
        status_code = 451

    def fake_load_binance(*args: object, **kwargs: object) -> pd.DataFrame:
        raise requests.exceptions.HTTPError(response=_FakeResponse())

    monkeypatch.setattr(loaders, "_load_binance", fake_load_binance)

    with pytest.raises(snapshot.GeoblockedError, match="geoloc"):
        snapshot.update_snapshot("BTC", "1d")


def test_update_snapshot_propagates_non_geoblock_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    last_saved = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=2)
    idx = pd.date_range(end=last_saved, periods=3, freq="D", tz="UTC")
    _write_fake_snapshot(tmp_path, "BTC", "1d", idx)

    def fake_load_binance(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ConnectionError("simulación de fallo de red genérico")

    monkeypatch.setattr(loaders, "_load_binance", fake_load_binance)

    with pytest.raises(ConnectionError, match="fallo de red"):
        snapshot.update_snapshot("BTC", "1d")


def test_update_snapshot_raises_clear_error_when_snapshot_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="export_snapshot"):
        snapshot.update_snapshot("BTC", "1d")


def test_update_snapshot_raises_for_unknown_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)

    with pytest.raises(ValueError, match="UNIVERSE"):
        snapshot.update_snapshot("DOGE", "1d")


def test_update_snapshot_raises_for_unsupported_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)

    with pytest.raises(ValueError, match="Intervalo"):
        snapshot.update_snapshot("BTC", "5m")


# --------------------------------------------------------------------------
# is_geoblocked_error (predicado puro, reutilizado también por
# scripts/export_snapshot.py)
# --------------------------------------------------------------------------


def test_is_geoblocked_error_detects_403_and_451() -> None:
    class _FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    for code in (403, 451):
        exc = requests.exceptions.HTTPError(response=_FakeResponse(code))
        assert snapshot.is_geoblocked_error(exc) is True


def test_is_geoblocked_error_false_for_unrelated_errors() -> None:
    assert snapshot.is_geoblocked_error(ConnectionError("timeout")) is False

    class _FakeResponse:
        status_code = 500

    exc = requests.exceptions.HTTPError(response=_FakeResponse())
    assert snapshot.is_geoblocked_error(exc) is False


def test_is_geoblocked_error_walks_exception_chain() -> None:
    class _FakeResponse:
        status_code = 451

    inner = requests.exceptions.HTTPError(response=_FakeResponse())
    outer = RuntimeError("fallaron todas las fuentes")
    outer.__cause__ = inner

    assert snapshot.is_geoblocked_error(outer) is True
