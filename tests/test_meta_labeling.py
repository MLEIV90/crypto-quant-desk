"""Tests offline para ml/meta_labeling.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.labeling import get_daily_volatility
from ml.meta_labeling import meta_labels, meta_positions, primary_side, train_meta_model
from signals.engine import compute_signal_components


def _trending_close(n: int = 60, drift: float = 0.02, noise_std: float = 0.002, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0.0, noise_std, n)
    close = 100.0 * np.cumprod(1.0 + returns)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(close, index=idx)


def _synthetic_ohlcv(n: int = 300, drift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    )


def _synthetic_meta_dataset(
    n: int = 600, horizon: int = 10, seed: int = 0, learnable: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset sintético (X, meta_y, t1) directo, sin pasar por triple-barrier
    real: `train_meta_model`/`meta_oos_proba` solo necesitan que `meta_y`
    sea binaria y `t1` tenga el formato de `ml.validation` — de dónde salió
    la etiqueta no les importa, así que alcanza con construirla a mano.

    Si `learnable=True`, la probabilidad de "trade ganador" depende de
    "signal" (una sigmoide: mayor signal, más probable ganar) — el caso
    donde el meta-modelo SÍ debería poder distinguir buenos de malos trades.
    Si `learnable=False`, el resultado es una moneda justa, independiente
    de todas las features.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n + horizon, freq="D")

    signal = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, (n, 4))
    X = pd.DataFrame(
        {
            "signal": signal,
            "noise_1": noise[:, 0],
            "noise_2": noise[:, 1],
            "noise_3": noise[:, 2],
            "noise_4": noise[:, 3],
        },
        index=idx[:n],
    )

    if learnable:
        prob_win = 1.0 / (1.0 + np.exp(-3.0 * signal))
        meta_y = pd.Series((rng.random(n) < prob_win).astype(float), index=idx[:n])
    else:
        meta_y = pd.Series(rng.integers(0, 2, n).astype(float), index=idx[:n])

    t1 = pd.Series(idx[horizon : horizon + n], index=idx[:n])
    return X, meta_y, t1


# --------------------------------------------------------------------------
# primary_side
# --------------------------------------------------------------------------


def test_primary_side_matches_trend_sign() -> None:
    df = _synthetic_ohlcv(n=250, drift=0.01, seed=0)
    side = primary_side(df)
    trend = compute_signal_components(df)["trend"]

    valid_idx = trend.dropna().index
    assert (side.loc[valid_idx][trend.loc[valid_idx] > 0.0] == 1.0).all()
    assert (side.loc[valid_idx][trend.loc[valid_idx] < 0.0] == -1.0).all()
    assert side.loc[trend.isna()].isna().all()


# --------------------------------------------------------------------------
# meta_labels: binaria correcta dado un lado conocido
# --------------------------------------------------------------------------


def test_meta_labels_long_wins_on_uptrend_and_loses_when_side_is_short() -> None:
    close = _trending_close(n=60, drift=0.02, noise_std=0.002)
    vol = get_daily_volatility(close, span=20)

    side_long = pd.Series(1.0, index=close.index)
    labels_long = meta_labels(close, vol, side_long, pt_mult=2.0, sl_mult=2.0, max_holding=10)
    valid_long = labels_long["label"].dropna()
    assert len(valid_long) > 0
    assert (valid_long == 1.0).all()  # largo en una tendencia alcista fuerte: siempre gana

    side_short = pd.Series(-1.0, index=close.index)
    labels_short = meta_labels(close, vol, side_short, pt_mult=2.0, sl_mult=2.0, max_holding=10)
    valid_short = labels_short["label"].dropna()
    assert len(valid_short) > 0
    assert (valid_short == 0.0).all()  # corto en esa misma tendencia: siempre pierde


def test_meta_labels_short_wins_on_downtrend() -> None:
    close = _trending_close(n=60, drift=-0.02, noise_std=0.002)
    vol = get_daily_volatility(close, span=20)
    side_short = pd.Series(-1.0, index=close.index)

    labels_short = meta_labels(close, vol, side_short, pt_mult=2.0, sl_mult=2.0, max_holding=10)
    valid_short = labels_short["label"].dropna()

    assert len(valid_short) > 0
    assert (valid_short == 1.0).all()


def test_meta_labels_side_zero_has_no_label() -> None:
    close = _trending_close(n=60, drift=0.01, noise_std=0.005)
    vol = get_daily_volatility(close, span=20)
    side = pd.Series(0.0, index=close.index)

    labels = meta_labels(close, vol, side, max_holding=10)

    assert labels["label"].isna().all()


def test_meta_labels_returns_same_columns_as_triple_barrier_labels() -> None:
    close = _trending_close(n=60, drift=0.0, noise_std=0.01)
    vol = get_daily_volatility(close, span=20)
    side = pd.Series(1.0, index=close.index)

    labels = meta_labels(close, vol, side, max_holding=10)

    assert list(labels.columns) == ["label", "barrera_tocada", "dias_hasta_evento", "retorno_realizado"]


# --------------------------------------------------------------------------
# train_meta_model: honestidad frente al baseline sin filtrar
# --------------------------------------------------------------------------


def test_train_meta_model_beats_baseline_when_wins_are_distinguishable() -> None:
    X, meta_y, t1 = _synthetic_meta_dataset(n=600, horizon=10, seed=0, learnable=True)

    result = train_meta_model(X, meta_y, t1, n_splits=5, embargo_pct=0.01)

    assert result["supera_baseline"] is True
    assert result["precision_media"] > result["baseline_precision_media"] + 0.1
    assert result["roc_auc_media"] > 0.6
    assert set(result.keys()) == {
        "precision_scores", "precision_media", "recall_scores", "recall_media",
        "f1_scores", "f1_media", "roc_auc_scores", "roc_auc_media",
        "baseline_precision_scores", "baseline_precision_media",
        "mejora_precision", "supera_baseline", "n_splits", "n_muestras",
    }


def test_train_meta_model_roc_auc_stays_near_half_when_wins_are_random() -> None:
    X, meta_y, t1 = _synthetic_meta_dataset(n=600, horizon=10, seed=1, learnable=False)

    result = train_meta_model(X, meta_y, t1, n_splits=5, embargo_pct=0.01)

    assert abs(result["roc_auc_media"] - 0.5) < 0.15


# --------------------------------------------------------------------------
# meta_positions: umbral + sizing
# --------------------------------------------------------------------------


def test_meta_positions_respects_threshold_and_proba_sizing() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    side = pd.Series([1.0, -1.0, 1.0, -1.0, 0.0], index=idx)
    proba = pd.Series([0.9, 0.8, 0.3, 0.6, 0.9], index=idx)

    positions_sized = meta_positions(side, proba, threshold=0.5, size_by_proba=True)
    expected_sized = pd.Series([0.9, -0.8, 0.0, -0.6, 0.0], index=idx, name="position")
    pd.testing.assert_series_equal(positions_sized, expected_sized)


def test_meta_positions_binary_filter_without_sizing() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    side = pd.Series([1.0, -1.0, 1.0, -1.0, 0.0], index=idx)
    proba = pd.Series([0.9, 0.8, 0.3, 0.6, 0.9], index=idx)

    positions_binary = meta_positions(side, proba, threshold=0.5, size_by_proba=False)
    expected_binary = pd.Series([1.0, -1.0, 0.0, -1.0, 0.0], index=idx, name="position")
    pd.testing.assert_series_equal(positions_binary, expected_binary)


def test_meta_positions_defaults_to_zero_when_side_or_proba_missing() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    side = pd.Series([1.0, np.nan, -1.0], index=idx)
    proba = pd.Series([0.8, 0.9, np.nan], index=idx)

    positions = meta_positions(side, proba, threshold=0.5)

    assert positions.iloc[0] == pytest.approx(0.8)
    assert positions.iloc[1] == 0.0
    assert positions.iloc[2] == 0.0
