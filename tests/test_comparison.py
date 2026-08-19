"""Tests offline para analysis/comparison.py (Fase 12a)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.comparison import align_common_dates, compare_assets, normalize_to_base


# --------------------------------------------------------------------------
# normalize_to_base
# --------------------------------------------------------------------------


def test_normalize_to_base_starts_at_base_and_scales_proportionally() -> None:
    prices = pd.Series([50.0, 100.0, 25.0])

    out = normalize_to_base(prices, base=100.0)

    assert out.iloc[0] == pytest.approx(100.0)
    assert out.iloc[1] == pytest.approx(200.0)  # se duplicó
    assert out.iloc[2] == pytest.approx(50.0)  # cayó a la mitad del inicio


def test_normalize_to_base_respects_custom_base() -> None:
    prices = pd.Series([10.0, 20.0])
    out = normalize_to_base(prices, base=1.0)
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# align_common_dates
# --------------------------------------------------------------------------


def test_align_common_dates_starts_at_shortest_history() -> None:
    idx_a = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    idx_b = pd.date_range("2020-01-05", periods=10, freq="D", tz="UTC")  # arranca 4 días después
    series_a = pd.Series(np.arange(10, dtype=float), index=idx_a)
    series_b = pd.Series(np.arange(10, dtype=float), index=idx_b)

    aligned = align_common_dates({"A": series_a, "B": series_b})

    assert aligned.index.min() == idx_b.min()  # el más corto (B) define el inicio común
    assert list(aligned.columns) == ["A", "B"]
    assert not aligned.isna().any().any()


def test_align_common_dates_empty_when_no_overlap() -> None:
    idx_a = pd.date_range("2018-01-01", periods=5, freq="D", tz="UTC")
    idx_b = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
    series_a = pd.Series(np.arange(5, dtype=float), index=idx_a)
    series_b = pd.Series(np.arange(5, dtype=float), index=idx_b)

    aligned = align_common_dates({"A": series_a, "B": series_b})

    assert aligned.empty


# --------------------------------------------------------------------------
# compare_assets
# --------------------------------------------------------------------------


def test_compare_assets_all_series_start_at_100() -> None:
    idx = pd.date_range("2021-01-01", periods=30, freq="D", tz="UTC")
    prices = {
        "BTC": pd.Series(30000.0 + np.cumsum(np.random.default_rng(0).normal(0, 100, 30)), index=idx),
        "ETH": pd.Series(1000.0 + np.cumsum(np.random.default_rng(1).normal(0, 10, 30)), index=idx),
    }

    result = compare_assets(prices, limit=30)

    assert result["normalizado"]["BTC"].iloc[0] == pytest.approx(100.0)
    assert result["normalizado"]["ETH"].iloc[0] == pytest.approx(100.0)


def test_compare_assets_computes_correct_total_return() -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    # BTC se duplica (100% de retorno), ETH cae a la mitad (-50%).
    prices = {
        "BTC": pd.Series([100.0, 150.0, 200.0], index=idx),
        "ETH": pd.Series([100.0, 75.0, 50.0], index=idx),
    }

    result = compare_assets(prices, limit=3)

    assert result["rendimiento_total_pct"]["BTC"] == pytest.approx(100.0)
    assert result["rendimiento_total_pct"]["ETH"] == pytest.approx(-50.0)


def test_compare_assets_respects_limit_using_only_recent_window() -> None:
    idx = pd.date_range("2021-01-01", periods=100, freq="D", tz="UTC")
    # Precio sube constantemente los primeros 90 días, después se mantiene
    # plano los últimos 10 — con limit=10, el rendimiento del período debe
    # ser ~0%, no reflejar la suba de los primeros 90 días.
    values = np.concatenate([np.linspace(100.0, 1000.0, 90), np.full(10, 1000.0)])
    prices = {"BTC": pd.Series(values, index=idx), "ETH": pd.Series(values, index=idx)}

    result = compare_assets(prices, limit=10)

    assert len(result["fechas"]) == 10
    assert result["rendimiento_total_pct"]["BTC"] == pytest.approx(0.0, abs=1e-6)


def test_compare_assets_aligns_shorter_history_asset() -> None:
    idx_long = pd.date_range("2018-01-01", periods=20, freq="D", tz="UTC")
    idx_short = pd.date_range("2018-01-15", periods=20, freq="D", tz="UTC")  # arranca 14 días después
    prices = {
        "BTC": pd.Series(np.linspace(100.0, 200.0, 20), index=idx_long),
        "SOL": pd.Series(np.linspace(10.0, 20.0, 20), index=idx_short),
    }

    result = compare_assets(prices, limit=100)  # pide más de lo disponible: debe traer todo lo alineado

    assert result["fechas"].min() == idx_short.min()
    assert len(result["fechas"]) == 6  # solapan solo 2018-01-15..2018-01-20 (6 fechas)


def test_compare_assets_returns_empty_when_no_common_dates() -> None:
    idx_a = pd.date_range("2018-01-01", periods=5, freq="D", tz="UTC")
    idx_b = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
    prices = {
        "A": pd.Series(np.arange(5, dtype=float), index=idx_a),
        "B": pd.Series(np.arange(5, dtype=float), index=idx_b),
    }

    result = compare_assets(prices, limit=10)

    assert result["rendimiento_total_pct"] == {}
    assert len(result["fechas"]) == 0
