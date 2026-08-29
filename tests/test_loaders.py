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
# source="store": intervalos DERIVADOS por resampleo (Fase 17b) — "4h"
# desde el snapshot horario, "1w" desde el diario. Los valores de
# open/high/low/close/volume de cada fixture son MONÓTONAMENTE CRECIENTES
# con el índice de la vela (open=100+i, high=105+i, low=95+i, close=102+i,
# volume=10*(i+1)) a propósito: así el primer/último/máximo/mínimo/suma de
# cada bucket son triviales de calcular a mano y el test detecta con
# claridad si alguna columna usa la agregación equivocada (p. ej. "last" en
# vez de "first" para open).
# --------------------------------------------------------------------------


def _write_synthetic_snapshot(
    directory: Path, asset: str, interval: str, start: str, periods: int, freq: str
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    i = np.arange(periods)
    df = pd.DataFrame(
        {
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 95.0 + i,
            "close": 102.0 + i,
            "volume": 10.0 * (i + 1),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_{interval}.parquet")
    return df


def test_load_store_resamples_4h_from_hourly_snapshot_with_correct_ohlcv_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    # 8 velas horarias (i=0..7) alineadas a medianoche UTC -> exactamente 2
    # buckets de 4h: [00-04) con i=0..3, [04-08) con i=4..7.
    _write_synthetic_snapshot(tmp_path, "BTC", "1h", "2020-01-01 00:00", periods=8, freq="h")

    df = loaders._load_store("BTC", "4h", "2020-01-01", "2020-01-02")

    assert list(df.columns) == loaders.STANDARD_COLUMNS
    assert len(df) == 2  # 8 horarias / 4 = 2 velas de 4h, sin resto

    bucket1 = df.iloc[0]
    assert bucket1["open"] == pytest.approx(100.0)  # open[0]: PRIMER valor del período, no un promedio
    assert bucket1["high"] == pytest.approx(108.0)  # max(high[0..3]) = high[3] = 105+3
    assert bucket1["low"] == pytest.approx(95.0)  # min(low[0..3]) = low[0] = 95+0
    assert bucket1["close"] == pytest.approx(105.0)  # close[3]: ÚLTIMO valor del período
    assert bucket1["volume"] == pytest.approx(100.0)  # suma: 10*(1+2+3+4)

    bucket2 = df.iloc[1]
    assert bucket2["open"] == pytest.approx(104.0)  # open[4]
    assert bucket2["high"] == pytest.approx(112.0)  # high[7] = 105+7
    assert bucket2["low"] == pytest.approx(99.0)  # low[4] = 95+4
    assert bucket2["close"] == pytest.approx(109.0)  # close[7] = 102+7
    assert bucket2["volume"] == pytest.approx(260.0)  # 10*(5+6+7+8)


def test_get_prices_4h_candle_count_is_roughly_hourly_over_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_synthetic_snapshot(tmp_path, "BTC", "1h", "2020-01-01 00:00", periods=400, freq="h")

    hourly = loaders.get_prices("BTC", source="store", interval="1h", start="2020-01-01", end="2020-01-20", use_cache=False)
    four_hourly = loaders.get_prices("BTC", source="store", interval="4h", start="2020-01-01", end="2020-01-20", use_cache=False)

    assert len(hourly) == 400
    assert len(four_hourly) == pytest.approx(len(hourly) / 4, abs=1)


def test_load_store_resamples_1w_from_daily_snapshot_with_correct_ohlcv_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    # 14 velas diarias (i=0..13) empezando un LUNES (2024-01-01) -> con la
    # convención de pandas freq="W" (semana termina el DOMINGO), exactamente
    # 2 semanas completas: [01-01..01-07] (i=0..6) y [01-08..01-14] (i=7..13).
    _write_synthetic_snapshot(tmp_path, "BTC", "1d", "2024-01-01", periods=14, freq="D")

    df = loaders._load_store("BTC", "1w", "2024-01-01", "2024-01-14")

    assert list(df.columns) == loaders.STANDARD_COLUMNS
    assert len(df) == 2  # 14 diarias / 7 = 2 semanas exactas
    # Documentando la convención elegida (ver docstring de RESAMPLED_INTERVALS
    # en data/loaders.py): el índice de cada vela semanal es el DOMINGO en
    # que esa semana termina, no el lunes de inicio.
    assert df.index[0] == pd.Timestamp("2024-01-07", tz="UTC")
    assert df.index[0].day_name() == "Sunday"
    assert df.index[1] == pd.Timestamp("2024-01-14", tz="UTC")

    week1 = df.iloc[0]
    assert week1["open"] == pytest.approx(100.0)  # open[0]: lunes 01-01
    assert week1["high"] == pytest.approx(111.0)  # max(high[0..6]) = high[6] = 105+6
    assert week1["low"] == pytest.approx(95.0)  # min(low[0..6]) = low[0] = 95+0
    assert week1["close"] == pytest.approx(108.0)  # close[6]: domingo 01-07
    assert week1["volume"] == pytest.approx(280.0)  # 10*(1+2+...+7)

    week2 = df.iloc[1]
    assert week2["open"] == pytest.approx(107.0)  # open[7]: lunes 01-08
    assert week2["high"] == pytest.approx(118.0)  # high[13] = 105+13
    assert week2["low"] == pytest.approx(102.0)  # low[7] = 95+7
    assert week2["close"] == pytest.approx(115.0)  # close[13]: domingo 01-14
    assert week2["volume"] == pytest.approx(770.0)  # 10*(8+9+...+14)


def test_get_prices_1w_candle_count_is_roughly_daily_over_seven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_synthetic_snapshot(tmp_path, "BTC", "1d", "2024-01-01", periods=140, freq="D")

    daily = loaders.get_prices("BTC", source="store", interval="1d", start="2024-01-01", end="2024-12-31", use_cache=False)
    weekly = loaders.get_prices("BTC", source="store", interval="1w", start="2024-01-01", end="2024-12-31", use_cache=False)

    assert len(daily) == 140
    assert len(weekly) == pytest.approx(len(daily) / 7, abs=1)


def test_load_store_derived_interval_error_mentions_base_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sin snapshot horario disponible: pedir "4h" debe fallar mencionando
    # el archivo BASE ("BTC_1h.parquet"), no uno "BTC_4h.parquet" inexistente
    # por diseño (nunca se exporta un snapshot propio para un derivado).
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="BTC_1h.parquet"):
        loaders._load_store("BTC", "4h", "2020-01-01", "2020-01-02")


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


# --------------------------------------------------------------------------
# source="full" (Fase 31): serie de precio larga fusionada CoinMetrics
# (pre-corte, solo cierre) + Binance/store (desde el corte, OHLC real).
# --------------------------------------------------------------------------


def _write_fake_coinmetrics_snapshot(
    directory: Path, asset: str, dates: list[str], closes: list[float]
) -> pd.DataFrame:
    """Como `_write_fake_snapshot`, pero con la forma que produce
    `_load_coinmetrics` de verdad: open=high=low=close (solo cierre), sin
    volumen — y guardado con el nombre que espera `_load_full_history`
    (`{asset}_coinmetrics_1d.parquet`, NO `{asset}_1d.parquet`).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates), tz="UTC")
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [np.nan] * len(closes)},
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_coinmetrics_1d.parquet")
    return df


def _write_fake_modern_snapshot(
    directory: Path, asset: str, dates: list[str], closes: list[float]
) -> pd.DataFrame:
    """Snapshot "store" (Binance) sintético con OHLC REAL (no degenerado:
    high > close > low en cada fila) — a diferencia de `_write_fake_snapshot`
    (que sí usa open=high=low=close por simplicidad), esto permite verificar
    que `_load_full_history` deja el tramo Binance/store INTACTO, sin
    aplanarlo.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates), tz="UTC")
    opens = [c * 0.995 for c in closes]
    highs = [c * 1.02 for c in closes]
    lows = [c * 0.98 for c in closes]
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [100.0] * len(closes)},
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_1d.parquet")
    return df


def test_load_full_history_starts_before_the_merge_cutoff_not_at_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_coinmetrics_snapshot(
        tmp_path, "BTC", ["2015-01-01", "2015-01-02", "2017-12-31"], [100.0, 101.0, 102.0]
    )
    _write_fake_modern_snapshot(tmp_path, "BTC", ["2018-01-01", "2018-01-02"], [102.3, 103.0])

    df = loaders._load_full_history("BTC", "1d", "2015-01-01", "2018-01-02")

    # PARTE B: la serie fusionada arranca ANTES del corte, no en 2018.
    assert df.index.min() == pd.Timestamp("2015-01-01", tz="UTC")
    assert df.index.min() < pd.Timestamp(loaders.MERGE_CUTOFF_DATE, tz="UTC")


def test_load_full_history_seam_has_no_artificial_jump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    # Último cierre pre-corte y primer cierre post-corte deliberadamente
    # cercanos (0.3%) — así cualquier salto > 3% en el resultado sería un
    # artefacto de la FUSIÓN, no una variación real de precio de la fixture.
    _write_fake_coinmetrics_snapshot(tmp_path, "BTC", ["2017-12-30", "2017-12-31"], [100.0, 100.0])
    _write_fake_modern_snapshot(tmp_path, "BTC", ["2018-01-01", "2018-01-02"], [100.3, 101.0])

    df = loaders._load_full_history("BTC", "1d", "2017-12-30", "2018-01-02")

    seam_pct_change = df["close"].pct_change().abs()
    # PARTE B: sin salto de precio > 3% en la costura 2017-12-31 -> 2018-01-01.
    assert seam_pct_change.loc[pd.Timestamp(loaders.MERGE_CUTOFF_DATE, tz="UTC")] < 0.03


def test_load_full_history_no_duplicates_sorted_and_monotonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    # Fechas fuera de orden a propósito en ambos archivos — `_load_full_history`
    # debe devolver la serie ordenada igual.
    _write_fake_coinmetrics_snapshot(
        tmp_path, "BTC", ["2017-12-31", "2015-01-01", "2016-06-15"], [102.0, 100.0, 101.0]
    )
    _write_fake_modern_snapshot(tmp_path, "BTC", ["2018-01-02", "2018-01-01"], [103.0, 102.3])

    df = loaders._load_full_history("BTC", "1d", "2015-01-01", "2018-01-02")

    # PARTE B: sin fechas duplicadas, serie ordenada y monótona en el tiempo.
    assert not df.index.duplicated().any()
    assert df.index.is_monotonic_increasing
    assert len(df) == 5


def test_load_full_history_pre_cutoff_candles_are_close_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_coinmetrics_snapshot(tmp_path, "BTC", ["2015-01-01", "2017-12-31"], [100.0, 102.0])
    _write_fake_modern_snapshot(tmp_path, "BTC", ["2018-01-01"], [102.3])

    df = loaders._load_full_history("BTC", "1d", "2015-01-01", "2018-01-01")

    pre_cutoff = df.loc[df.index < pd.Timestamp(loaders.MERGE_CUTOFF_DATE, tz="UTC")]
    post_cutoff = df.loc[df.index >= pd.Timestamp(loaders.MERGE_CUTOFF_DATE, tz="UTC")]

    # PARTE B: las velas pre-2018 tienen open=high=low=close (son solo cierre).
    assert (pre_cutoff["open"] == pre_cutoff["close"]).all()
    assert (pre_cutoff["high"] == pre_cutoff["close"]).all()
    assert (pre_cutoff["low"] == pre_cutoff["close"]).all()
    # El tramo Binance/store, en cambio, NO debe aplanarse (OHLC real intacto).
    assert (post_cutoff["high"] > post_cutoff["close"]).all()
    assert (post_cutoff["low"] < post_cutoff["close"]).all()


def test_load_full_history_degrades_gracefully_without_coinmetrics_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sin '{asset}_coinmetrics_1d.parquet' (p. ej. SOL, sin cobertura
    # pre-2018) — debe devolver solo el tramo Binance/store, sin error.
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_modern_snapshot(tmp_path, "SOL", ["2018-01-01", "2018-01-02"], [10.0, 10.5])

    df = loaders._load_full_history("SOL", "1d", "2015-01-01", "2018-01-02")

    assert len(df) == 2
    assert df.index.min() == pd.Timestamp("2018-01-01", tz="UTC")


def test_load_full_history_raises_for_non_daily_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)

    with pytest.raises(ValueError, match="1d"):
        loaders._load_full_history("BTC", "1h", "2015-01-01", "2018-01-02")


def test_get_prices_source_full_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_coinmetrics_snapshot(tmp_path, "BTC", ["2015-01-01", "2017-12-31"], [100.0, 102.0])
    _write_fake_modern_snapshot(tmp_path, "BTC", ["2018-01-01", "2018-01-02"], [102.3, 103.0])

    out = loaders.get_prices(
        "BTC", source="full", interval="1d", start="2015-01-01", end="2018-01-02", use_cache=False
    )

    assert list(out.columns) == loaders.STANDARD_COLUMNS
    assert len(out) == 4
    assert out.index.tz is not None
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    assert out.index.min() == pd.Timestamp("2015-01-01", tz="UTC")


def test_get_prices_source_store_unaffected_by_coinmetrics_snapshot_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PARTE B: source='store' (Binance) sigue devolviendo lo mismo que
    antes de esta fase, incluso si el snapshot CoinMetrics pre-corte
    también existe en el mismo directorio — no debe filtrarse ni mezclarse.
    """
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_fake_coinmetrics_snapshot(tmp_path, "BTC", ["2015-01-01", "2017-12-31"], [100.0, 102.0])
    modern = _write_fake_modern_snapshot(tmp_path, "BTC", ["2018-01-01", "2018-01-02"], [102.3, 103.0])

    out = loaders.get_prices("BTC", source="store", start="2018-01-01", end="2018-01-02", use_cache=False)

    assert len(out) == 2
    assert out.index.min() == pd.Timestamp("2018-01-01", tz="UTC")
    pd.testing.assert_series_equal(out["close"], modern["close"], check_names=False)


# --------------------------------------------------------------------------
# Validación con datos REALES (Fase 31) — se salta si no se corrió
# `scripts/export_coinmetrics_history.py` localmente (data/snapshot/ está
# en .gitignore, no se versiona ni se comparte entre máquinas/CI).
# --------------------------------------------------------------------------


def test_real_btc_coinmetrics_history_starts_around_2010_2011() -> None:
    from config import SNAPSHOT_DIR as REAL_SNAPSHOT_DIR

    path = REAL_SNAPSHOT_DIR / "BTC_coinmetrics_1d.parquet"
    if not path.exists():
        pytest.skip(
            f"'{path}' no existe — corré 'python scripts/export_coinmetrics_history.py' para "
            "generarlo y validar esto con datos reales."
        )

    df = pd.read_parquet(path)
    assert pd.Timestamp("2010-01-01", tz="UTC") <= df.index.min() < pd.Timestamp("2012-01-01", tz="UTC")
