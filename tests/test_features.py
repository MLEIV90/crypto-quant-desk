"""Tests offline para ml/features.py (OHLCV sintético, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import align_features_labels, build_feature_matrix
from ml.labeling import get_daily_volatility, triple_barrier_labels


def _synthetic_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}, index=idx
    )


# --------------------------------------------------------------------------
# build_feature_matrix
# --------------------------------------------------------------------------


def test_build_feature_matrix_has_expected_columns() -> None:
    df = _synthetic_ohlcv(n=250)
    features = build_feature_matrix(df)

    expected_new_cols = {
        "ret_1", "ret_5", "ret_10", "ret_20", "vol_realizada_20", "vol_ewma",
        "rsi_14", "macd", "macd_hist", "bb_pct_b", "bb_bandwidth", "atr_14",
    }
    assert expected_new_cols.issubset(set(features.columns))
    assert "vol_garch" not in features.columns  # removida por leakage, ver HOTFIX del módulo
    assert set(df.columns).issubset(set(features.columns))
    assert "ret_1" not in df.columns  # no mutó el df original


def test_build_feature_matrix_ret_columns_match_pct_change() -> None:
    df = _synthetic_ohlcv(n=100)
    features = build_feature_matrix(df)

    for lag in (1, 5, 10, 20):
        expected = df["close"].pct_change(periods=lag)
        pd.testing.assert_series_equal(features[f"ret_{lag}"], expected, check_names=False)


def test_build_feature_matrix_has_no_lookahead_via_perturbation() -> None:
    # Verificación de causalidad, método 1 (perturbación de un solo punto):
    # ninguna feature en el día t debería cambiar si se altera el precio de
    # un día POSTERIOR. Ya no hay excepciones documentadas (vol_garch se
    # removió por leakage, ver HOTFIX del módulo) — el test cubre TODAS las
    # columnas.
    df = _synthetic_ohlcv(n=250, seed=1)
    features_original = build_feature_matrix(df)

    df_perturbed = df.copy()
    close_col = df_perturbed.columns.get_loc("close")
    df_perturbed.iloc[-1, close_col] = df_perturbed.iloc[-1, close_col] * 1.5

    features_perturbed = build_feature_matrix(df_perturbed)

    pd.testing.assert_frame_equal(features_original.iloc[:-1], features_perturbed.iloc[:-1])


def test_build_feature_matrix_has_no_lookahead_via_truncation() -> None:
    """Verificación de causalidad, método 2 (TRUNCACIÓN — más estricto que
    perturbar un solo punto): para varias fechas de corte t, construir la
    matriz de features con la serie truncada en t (`df.loc[:t]`) y con la
    serie completa deben dar EXACTAMENTE la misma fila para t, en TODAS las
    columnas. Este test es el que hubiera detectado el leakage de la
    ex-feature `vol_garch` (que dependía de cuánta historia FUTURA hubiera
    disponible al momento del ajuste, no solo del pasado hasta t) — con
    perturbación de un solo punto quedaba enmascarado si el punto alterado
    estaba lejos del final de la ventana de ajuste.
    """
    df = _synthetic_ohlcv(n=250, seed=1)
    features_full = build_feature_matrix(df)

    for cutoff_pos in (100, 150, 200, 249):
        cutoff_date = df.index[cutoff_pos]
        features_truncated = build_feature_matrix(df.loc[:cutoff_date])

        row_full = features_full.loc[cutoff_date]
        row_truncated = features_truncated.loc[cutoff_date]

        pd.testing.assert_series_equal(row_full, row_truncated, check_names=False, rtol=1e-6)


# --------------------------------------------------------------------------
# align_features_labels
# --------------------------------------------------------------------------


def test_align_features_labels_returns_aligned_clean_data() -> None:
    df = _synthetic_ohlcv(n=300)
    X = build_feature_matrix(df)

    vol = get_daily_volatility(df["close"], span=50)
    labels_df = triple_barrier_labels(df["close"], vol, max_holding=10)

    X_clean, y, weights = align_features_labels(X, labels_df)

    assert len(X_clean) == len(y) == len(weights)
    assert len(X_clean) > 0
    assert X_clean.index.equals(y.index)
    assert X_clean.index.equals(weights.index)
    assert not X_clean.isna().any().any()
    assert not y.isna().any()
    assert not weights.isna().any()
    assert set(y.unique()).issubset({-1.0, 0.0, 1.0})


def test_align_features_labels_drops_warmup_and_tail() -> None:
    df = _synthetic_ohlcv(n=300)
    X = build_feature_matrix(df)
    vol = get_daily_volatility(df["close"], span=50)
    labels_df = triple_barrier_labels(df["close"], vol, max_holding=10)

    X_clean, y, _ = align_features_labels(X, labels_df)

    # Ni el primer día (warmup de indicadores) ni el último (sin etiqueta
    # completa) deberían sobrevivir a la alineación.
    assert df.index[0] not in X_clean.index
    assert df.index[-1] not in X_clean.index
