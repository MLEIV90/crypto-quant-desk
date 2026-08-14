"""Tests offline (sin red) para data/loaders.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import loaders

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class _FakeResponse:
    """Sustituto mínimo de requests.Response para mockear `_http_get_with_retry`."""

    def __init__(self, text: str) -> None:
        self.text = text


# --------------------------------------------------------------------------
# _standardize
# --------------------------------------------------------------------------


def test_standardize_returns_expected_columns_and_utc_index() -> None:
    idx = pd.to_datetime(["2021-01-02", "2021-01-01", "2021-01-01", "2021-01-03"])
    raw = pd.DataFrame(
        {
            "open": [2.0, 1.0, 1.5, 3.0],
            "high": [2.0, 1.0, 1.5, 3.0],
            "low": [2.0, 1.0, 1.5, 3.0],
            "close": [2.0, 1.0, 1.5, 3.0],
            "volume": [20.0, 10.0, 15.0, 30.0],
        },
        index=idx,
    )

    out = loaders._standardize(raw, interval="1d", source="test")

    assert list(out.columns) == loaders.STANDARD_COLUMNS
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    # La fila duplicada en 2021-01-01 debe resolverse quedándose con la última (close=1.5).
    assert out.loc["2021-01-01", "close"].item() == pytest.approx(1.5)
    assert out.dtypes.eq(float).all()


def test_standardize_raises_on_missing_columns() -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    raw = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

    with pytest.raises(ValueError):
        loaders._standardize(raw, interval="1d", source="test")


# --------------------------------------------------------------------------
# CoinMetrics (fixture offline)
# --------------------------------------------------------------------------


def test_load_coinmetrics_uses_priceusd_as_close_and_fills_ohlc(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_text = (FIXTURES_DIR / "coinmetrics_btc_sample.csv").read_text(encoding="utf-8")

    def fake_http_get_with_retry(url: str, params: dict) -> _FakeResponse:
        assert "btc.csv" in url
        return _FakeResponse(fixture_text)

    monkeypatch.setattr(loaders, "_http_get_with_retry", fake_http_get_with_retry)

    df = loaders._load_coinmetrics("btc", start="2020-01-01", end="2020-01-05")

    assert list(df.columns) == loaders.STANDARD_COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert len(df) == 5
    # Sin OHLC real: open/high/low/close deben ser todos iguales al PriceUSD.
    assert (df["open"] == df["close"]).all()
    assert (df["high"] == df["close"]).all()
    assert (df["low"] == df["close"]).all()
    assert df["close"].iloc[0] == pytest.approx(7200.5)
    assert df["volume"].isna().all()


# --------------------------------------------------------------------------
# source="store" (snapshot local de scripts/export_snapshot.py)
# --------------------------------------------------------------------------


def _write_fake_snapshot(directory: Path, asset: str, interval: str = "1d") -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.0, 2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0, 5.0],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_{interval}.parquet")
    return df


def test_load_store_raises_clear_error_when_snapshot_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="export_snapshot"):
        loaders._load_store("BTC", "1d", "2020-01-01", "2020-01-05")


def test_load_store_reads_local_snapshot_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_snapshot(tmp_path, "BTC")

    df = loaders._load_store("BTC", "1d", "2020-01-01", "2020-01-05")

    assert list(df.columns) == loaders.STANDARD_COLUMNS
    assert len(df) == 5
    assert str(df.index.tz) == "UTC"
    assert df["close"].iloc[-1] == pytest.approx(5.0)


def test_get_prices_source_store_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_snapshot(tmp_path, "BTC")

    out = loaders.get_prices("BTC", source="store", start="2020-01-01", end="2020-01-05", use_cache=False)

    assert list(out.columns) == loaders.STANDARD_COLUMNS
    assert len(out) == 5
    assert out.index.tz is not None
    assert out.index.is_monotonic_increasing


def _write_fake_hourly_snapshot(directory: Path, asset: str) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.arange(1.0, 9.0),
            "high": np.arange(1.0, 9.0),
            "low": np.arange(1.0, 9.0),
            "close": np.arange(1.0, 9.0),
            "volume": np.arange(10.0, 90.0, 10.0),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_1h.parquet")
    return df


def test_load_store_reads_hourly_snapshot_when_interval_is_1h(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fase 6b: interval="1h" debe leer "{asset}_1h.parquet", un archivo
    # DISTINTO del diario ("{asset}_1d.parquet") — ambos pueden coexistir
    # en el mismo SNAPSHOT_DIR sin pisarse.
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_snapshot(tmp_path, "BTC", interval="1d")
    _write_fake_hourly_snapshot(tmp_path, "BTC")

    df = loaders._load_store("BTC", "1h", "2020-01-01", "2020-01-02")

    assert list(df.columns) == loaders.STANDARD_COLUMNS
    assert len(df) == 8  # las 8 velas horarias, no las 5 diarias
    assert df["close"].iloc[-1] == pytest.approx(8.0)


def test_get_prices_source_store_end_to_end_hourly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_snapshot(tmp_path, "BTC", interval="1d")  # coexiste, no debe mezclarse
    _write_fake_hourly_snapshot(tmp_path, "BTC")

    out = loaders.get_prices(
        "BTC", source="store", interval="1h", start="2020-01-01", end="2020-01-02", use_cache=False
    )

    assert list(out.columns) == loaders.STANDARD_COLUMNS
    assert len(out) == 8
    assert out.index.tz is not None
    assert out.index.is_monotonic_increasing


def test_get_prices_source_store_default_interval_still_reads_daily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """interval="1d" (default, sin pasar `interval=` explícito) sigue
    leyendo el snapshot diario — la parametrización por interval de Fase 6b
    no rompió el caso default preexistente.
    """
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_snapshot(tmp_path, "BTC", interval="1d")
    _write_fake_hourly_snapshot(tmp_path, "BTC")

    out = loaders.get_prices("BTC", source="store", start="2020-01-01", end="2020-01-05", use_cache=False)

    assert len(out) == 5
    assert out["close"].iloc[-1] == pytest.approx(5.0)  # vino del snapshot diario, no del horario


# --------------------------------------------------------------------------
# get_prices: fallback entre fuentes
# --------------------------------------------------------------------------


def test_get_prices_falls_back_when_primary_source_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    fallback_df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [np.nan, np.nan, np.nan],
        },
        index=idx,
    )

    calls: list[str] = []

    def fake_load_from_source(asset: str, source: str, interval: str, start: str, end: str) -> pd.DataFrame:
        calls.append(source)
        if source == "binance":
            raise ConnectionError("simulación de fallo de red")
        if source == "coinmetrics":
            return fallback_df
        raise AssertionError(f"No debería intentarse la fuente '{source}' en este test")

    monkeypatch.setattr(loaders, "_load_from_source", fake_load_from_source)

    out = loaders.get_prices("BTC", source="binance", start="2021-01-01", end="2021-01-03", use_cache=False)

    assert calls[0] == "binance"
    assert "coinmetrics" in calls
    assert list(out.columns) == loaders.STANDARD_COLUMNS
    assert len(out) == 3


def test_get_prices_raises_when_all_sources_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_from_source(asset: str, source: str, interval: str, start: str, end: str) -> pd.DataFrame:
        raise ConnectionError(f"fallo simulado en {source}")

    monkeypatch.setattr(loaders, "_load_from_source", fake_load_from_source)

    with pytest.raises(RuntimeError):
        loaders.get_prices("BTC", source="binance", start="2021-01-01", end="2021-01-03", use_cache=False)
