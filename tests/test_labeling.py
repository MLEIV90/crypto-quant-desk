"""Tests offline para ml/labeling.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.labeling import get_daily_volatility, get_sample_weights, triple_barrier_labels


def _trending_close(n: int = 60, drift: float = 0.02, noise_std: float = 0.002, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0.0, noise_std, n)
    close = 100.0 * np.cumprod(1.0 + returns)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(close, index=idx)


def _flat_close(n: int = 60, noise_std: float = 0.0005, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, noise_std, n)
    close = 100.0 * np.cumprod(1.0 + returns)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(close, index=idx)


# --------------------------------------------------------------------------
# get_daily_volatility
# --------------------------------------------------------------------------


def test_get_daily_volatility_is_positive_and_trailing() -> None:
    close = _trending_close(n=200)
    vol = get_daily_volatility(close, span=50)

    valid = vol.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()

    # Trailing: recortar la cola no debería cambiar los valores ya calculados.
    vol_truncated = get_daily_volatility(close.iloc[:150], span=50)
    pd.testing.assert_series_equal(vol.iloc[:150], vol_truncated)


# --------------------------------------------------------------------------
# triple_barrier_labels: casos direccionales simples
# --------------------------------------------------------------------------


def test_triple_barrier_all_labels_positive_on_strong_uptrend() -> None:
    close = _trending_close(n=60, drift=0.02, noise_std=0.002)
    vol = get_daily_volatility(close, span=20)
    max_holding = 10

    labels_df = triple_barrier_labels(close, vol, pt_mult=2.0, sl_mult=2.0, max_holding=max_holding)

    valid_labels = labels_df["label"].dropna()
    assert len(valid_labels) > 0
    assert (valid_labels == 1.0).all()
    assert (labels_df.loc[valid_labels.index, "barrera_tocada"] == "take_profit").all()


def test_triple_barrier_all_labels_negative_on_strong_downtrend() -> None:
    close = _trending_close(n=60, drift=-0.02, noise_std=0.002)
    vol = get_daily_volatility(close, span=20)

    labels_df = triple_barrier_labels(close, vol, pt_mult=2.0, sl_mult=2.0, max_holding=10)

    valid_labels = labels_df["label"].dropna()
    assert len(valid_labels) > 0
    assert (valid_labels == -1.0).all()
    assert (labels_df.loc[valid_labels.index, "barrera_tocada"] == "stop_loss").all()


def test_triple_barrier_zero_label_on_flat_series() -> None:
    # Nota de calibración: con multiplicadores "normales" (p. ej. 2-3x),
    # el RANGO alcanzable por un camino aleatorio de 10 pasos (la ventana
    # de max_holding) ya es comparable a la barrera, aunque la serie sea
    # "plana" en el sentido de no tener tendencia — un random walk sin
    # drift igual se aleja de su punto de partida con el tiempo. Para que
    # el test sea determinísticamente "vence el tiempo" hace falta una
    # barrera bien ancha en relación a esa ventana, no alcanza con "poca
    # vol" sola.
    close = _flat_close(n=60, noise_std=0.0003)
    vol = get_daily_volatility(close, span=20)

    labels_df = triple_barrier_labels(close, vol, pt_mult=15.0, sl_mult=15.0, max_holding=10)

    valid_labels = labels_df["label"].dropna()
    assert len(valid_labels) > 0
    assert (valid_labels == 0.0).all()
    assert (labels_df.loc[valid_labels.index, "barrera_tocada"] == "vertical").all()
    assert (labels_df.loc[valid_labels.index, "dias_hasta_evento"] == 10).all()


# --------------------------------------------------------------------------
# triple_barrier_labels: respeta la barrera que se toca PRIMERO en el tiempo
# --------------------------------------------------------------------------


def test_triple_barrier_respects_first_barrier_in_time() -> None:
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    # entry=100; con vol=0.05 y pt_mult=sl_mult=2.0: TP=110, SL=90.
    # Camino: día1=100, día2=90 (toca SL), día3=100, día4=100, día5=115 (tocaría TP, pero DESPUÉS del SL).
    close = pd.Series([100.0, 100.0, 90.0, 100.0, 100.0, 115.0, 100.0, 100.0], index=idx)
    volatility = pd.Series(0.05, index=idx)

    labels_df = triple_barrier_labels(close, volatility, pt_mult=2.0, sl_mult=2.0, max_holding=5)

    assert labels_df["label"].iloc[0] == -1.0
    assert labels_df["barrera_tocada"].iloc[0] == "stop_loss"
    assert labels_df["dias_hasta_evento"].iloc[0] == 2


# --------------------------------------------------------------------------
# triple_barrier_labels: cola sin etiqueta
# --------------------------------------------------------------------------


def test_triple_barrier_last_max_holding_rows_are_nan() -> None:
    close = _trending_close(n=60, drift=0.005, noise_std=0.005)
    vol = get_daily_volatility(close, span=20)
    max_holding = 10

    labels_df = triple_barrier_labels(close, vol, max_holding=max_holding)

    assert labels_df["label"].iloc[-max_holding:].isna().all()
    # La fila justo ANTES de la cola sí debería tener etiqueta (siempre que
    # tenga volatilidad válida, ver warmup del EWMA).
    assert pd.notna(labels_df["label"].iloc[-max_holding - 1])


# --------------------------------------------------------------------------
# get_sample_weights
# --------------------------------------------------------------------------


def test_get_sample_weights_isolated_label_gets_weight_near_one() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    labels_df = pd.DataFrame(
        {
            "label": [1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan],
            "dias_hasta_evento": [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan],
        },
        index=idx,
    )

    weights = get_sample_weights(labels_df)

    # index 0,1,2 tienen horizontes [0,3],[1,4],[2,5]: se solapan fuertemente.
    # index 7 (horizonte [7,8]) no se solapa con nadie más -> peso 1.
    assert weights.iloc[7] == pytest.approx(1.0)
    assert weights.iloc[0] < weights.iloc[7]
    assert weights.iloc[3:7].isna().all()
    assert weights.iloc[8:].isna().all()


def test_get_sample_weights_are_between_zero_and_one() -> None:
    close = _trending_close(n=100, drift=0.001, noise_std=0.01, seed=1)
    vol = get_daily_volatility(close, span=20)
    labels_df = triple_barrier_labels(close, vol, max_holding=10)

    weights = get_sample_weights(labels_df)
    valid = weights.dropna()

    assert len(valid) > 0
    assert (valid > 0).all()
    assert (valid <= 1.0).all()
    # Coinciden exactamente las fechas con etiqueta válida y con peso válido.
    assert set(valid.index) == set(labels_df["label"].dropna().index)
