"""Tests offline para data/onchain.py (datos on-chain sintéticos, sin red).

Igual que en `tests/test_features.py`, la serie sintética representa un
activo con el set COMPLETO de columnas on-chain (el caso BTC/ETH, ver
docstring de `data/onchain.py`) para poder ejercitar todas las features de
`build_onchain_features` de una sola vez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.onchain import ZSCORE_WINDOW, build_onchain_features, load_onchain, merge_onchain


def _synthetic_onchain(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    flow_in = np.abs(rng.normal(5_000_000, 1_000_000, n))
    flow_out = np.abs(rng.normal(4_800_000, 1_000_000, n))
    active_addr = 500_000 + np.cumsum(rng.normal(50, 500, n))
    tx_count = 250_000 + np.cumsum(rng.normal(20, 300, n))
    mvrv = 1.0 + 0.5 * np.sin(np.linspace(0, 8, n)) + rng.normal(0, 0.05, n)
    hashrate = 1e17 * (1.0 + np.cumsum(rng.normal(0.001, 0.01, n)))
    fee = np.abs(rng.normal(50.0, 10.0, n))
    return pd.DataFrame(
        {
            "FlowInExUSD": flow_in,
            "FlowOutExUSD": flow_out,
            "AdrActCnt": active_addr,
            "TxCnt": tx_count,
            "CapMVRVCur": mvrv,
            "HashRate": hashrate,
            "FeeTotNtv": fee,
        },
        index=idx,
    )


# --------------------------------------------------------------------------
# build_onchain_features: columnas esperadas y estacionariedad
# --------------------------------------------------------------------------


def test_build_onchain_features_has_expected_columns() -> None:
    df = _synthetic_onchain(n=200)
    features = build_onchain_features(df)

    expected_cols = {
        "net_exchange_flow", "net_flow_zscore",
        "active_addr_growth", "active_addr_zscore",
        "tx_growth", "mvrv_level", "mvrv_zscore",
        "hashrate_growth", "fee_growth",
    }
    assert expected_cols.issubset(set(features.columns))
    assert features.index.equals(df.index)


def test_build_onchain_features_drops_raw_non_stationary_levels() -> None:
    """Los niveles crudos NO estacionarios (USD/conteos) no deben sobrevivir
    como feature — solo sus variaciones/z-scores. La única excepción
    documentada es CapMVRVCur ("mvrv_level"), que ya es un ratio.
    """
    df = _synthetic_onchain(n=200, seed=2)
    features = build_onchain_features(df)

    raw_level_columns = {"FlowInExUSD", "FlowOutExUSD", "AdrActCnt", "TxCnt", "HashRate", "FeeTotNtv"}
    assert raw_level_columns.isdisjoint(set(features.columns))


def test_build_onchain_features_only_uses_available_columns() -> None:
    """Cobertura parcial (p. ej. LTC/BNB, sin flujos de exchange): si faltan
    columnas de origen, las features derivadas de ellas simplemente no se
    agregan — no debe explotar ni inventar datos.
    """
    df = _synthetic_onchain(n=200, seed=3).drop(columns=["FlowInExUSD", "FlowOutExUSD", "HashRate"])
    features = build_onchain_features(df)

    assert "net_exchange_flow" not in features.columns
    assert "net_flow_zscore" not in features.columns
    assert "hashrate_growth" not in features.columns
    assert "mvrv_level" in features.columns
    assert "tx_growth" in features.columns


def test_build_onchain_features_empty_input_returns_empty_output() -> None:
    df = pd.DataFrame(index=pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"))
    features = build_onchain_features(df)
    assert features.empty
    assert features.index.equals(df.index)


# --------------------------------------------------------------------------
# Causalidad (OBLIGATORIO): método de truncación
# --------------------------------------------------------------------------


def test_build_onchain_features_has_no_lookahead_via_truncation() -> None:
    """Verificación de causalidad por TRUNCACIÓN (mismo método que
    `tests/test_features.py::test_build_feature_matrix_has_no_lookahead_via_truncation`,
    más estricto que perturbar un solo punto): para varias fechas de corte
    t, construir las features con la serie truncada en t (`df.loc[:t]`) y
    con la serie completa debe dar EXACTAMENTE la misma fila para t, en
    TODAS las columnas. Si alguna feature on-chain mirara el futuro (p. ej.
    un z-score calculado con una ventana centrada, o normalizado contra el
    máximo de TODA la serie), este test la delataría.
    """
    df = _synthetic_onchain(n=400, seed=1)
    features_full = build_onchain_features(df)

    for cutoff_pos in (100, 150, 250, 399):
        cutoff_date = df.index[cutoff_pos]
        features_truncated = build_onchain_features(df.loc[:cutoff_date])

        row_full = features_full.loc[cutoff_date]
        row_truncated = features_truncated.loc[cutoff_date]

        pd.testing.assert_series_equal(row_full, row_truncated, check_names=False, rtol=1e-6)


def test_build_onchain_features_has_no_lookahead_via_perturbation() -> None:
    """Segunda verificación de causalidad (perturbación de un solo punto):
    ninguna feature en el día t debería cambiar si se altera un dato
    POSTERIOR a t.
    """
    df = _synthetic_onchain(n=300, seed=4)
    features_original = build_onchain_features(df)

    df_perturbed = df.copy()
    df_perturbed.iloc[-1, df_perturbed.columns.get_loc("FlowInExUSD")] *= 3.0
    df_perturbed.iloc[-1, df_perturbed.columns.get_loc("CapMVRVCur")] *= 2.0

    features_perturbed = build_onchain_features(df_perturbed)

    pd.testing.assert_frame_equal(features_original.iloc[:-1], features_perturbed.iloc[:-1])


# --------------------------------------------------------------------------
# Manejo de NaN (warmup de los rolling)
# --------------------------------------------------------------------------


def test_build_onchain_features_zscore_warmup_is_nan_then_valid() -> None:
    df = _synthetic_onchain(n=200, seed=5)
    features = build_onchain_features(df)

    # Sin ventana completa (primeras ZSCORE_WINDOW - 1 filas) el z-score
    # rolling no puede calcularse: debe quedar NaN, no un valor inventado.
    assert features["net_flow_zscore"].iloc[: ZSCORE_WINDOW - 1].isna().all()
    assert features["mvrv_zscore"].iloc[: ZSCORE_WINDOW - 1].isna().all()

    # Una vez llena la ventana, no debería haber más NaN (la serie sintética
    # no tiene huecos).
    assert features["net_flow_zscore"].iloc[ZSCORE_WINDOW:].notna().all()
    assert features["mvrv_zscore"].iloc[ZSCORE_WINDOW:].notna().all()

    # pct_change() del primer día siempre es NaN (no hay t-1).
    assert pd.isna(features["active_addr_growth"].iloc[0])
    assert pd.isna(features["tx_growth"].iloc[0])


# --------------------------------------------------------------------------
# merge_onchain: alineación por fecha
# --------------------------------------------------------------------------


def test_merge_onchain_aligns_by_date() -> None:
    onchain_df = _synthetic_onchain(n=60, seed=6)
    features = build_onchain_features(onchain_df)

    price_idx = pd.date_range("2020-01-10", periods=30, freq="D", tz="UTC")
    price_df = pd.DataFrame({"close": np.linspace(100.0, 130.0, 30)}, index=price_idx)

    merged = merge_onchain(price_df, features)

    assert merged.index.equals(price_df.index)
    assert set(features.columns).issubset(set(merged.columns))
    assert "close" in merged.columns

    common_date = price_idx[10]
    pd.testing.assert_series_equal(
        merged.loc[common_date, list(features.columns)],
        features.loc[common_date],
        check_names=False,
        rtol=1e-6,
    )


def test_merge_onchain_leaves_nan_outside_onchain_coverage() -> None:
    onchain_df = _synthetic_onchain(n=20, seed=7)
    features = build_onchain_features(onchain_df)

    # El rango de precios excede, a ambos lados, la cobertura on-chain.
    price_idx = pd.date_range("2019-12-20", periods=60, freq="D", tz="UTC")
    price_df = pd.DataFrame({"close": np.linspace(100.0, 160.0, 60)}, index=price_idx)

    merged = merge_onchain(price_df, features)

    assert pd.isna(merged.loc[price_idx[0], "mvrv_level"])
    assert pd.isna(merged.loc[price_idx[-1], "mvrv_level"])


# --------------------------------------------------------------------------
# load_onchain: error claro si falta el snapshot
# --------------------------------------------------------------------------


def test_load_onchain_raises_clear_error_if_snapshot_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("data.onchain.SNAPSHOT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="on-chain"):
        load_onchain("SOL")
