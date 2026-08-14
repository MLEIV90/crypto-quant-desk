"""Tests offline para ml/models.py (datasets sintéticos, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models import (
    evaluate_primary,
    feature_importances,
    fit_final,
    make_primary_model,
    model_to_positions,
    oos_predictions,
)


def _synthetic_dataset(
    n: int = 600, horizon: int = 10, seed: int = 0, learnable: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset sintético con 5 features (una "signal" y cuatro de ruido puro).

    Si `learnable=True`, `y` es una función determinística y simple de
    "signal" (fácil de aprender para un árbol) — el caso donde el modelo SÍ
    debería poder superar el azar. Si `learnable=False`, `y` es aleatoria e
    independiente de todas las features — el caso donde el modelo NO
    debería poder aprender nada real.

    `t1` sigue la convención de `ml.validation`: índice = inicio de cada
    muestra, valor = fecha de resolución (acá, un horizonte fijo simple,
    suficiente para testear el módulo sin depender del triple-barrier real).
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
        label = np.where(signal > 0.3, 1.0, np.where(signal < -0.3, -1.0, 0.0))
    else:
        label = rng.choice([-1.0, 0.0, 1.0], size=n)
    y = pd.Series(label, index=idx[:n])

    t1 = pd.Series(idx[horizon : horizon + n], index=idx[:n])
    return X, y, t1


# --------------------------------------------------------------------------
# make_primary_model
# --------------------------------------------------------------------------


def test_make_primary_model_uses_conservative_defaults() -> None:
    model = make_primary_model()
    assert model.max_depth == 3
    assert model.n_estimators == 200
    assert model.learning_rate == 0.05
    assert model.subsample == 0.8
    assert model.reg_lambda == 1.0


def test_make_primary_model_allows_overrides() -> None:
    model = make_primary_model(max_depth=5, n_estimators=50)
    assert model.max_depth == 5
    assert model.n_estimators == 50


# --------------------------------------------------------------------------
# evaluate_primary: honestidad frente a los baselines
# --------------------------------------------------------------------------


def test_evaluate_primary_beats_chance_on_learnable_signal() -> None:
    X, y, t1 = _synthetic_dataset(n=600, horizon=10, seed=0, learnable=True)

    result = evaluate_primary(X, y, t1, n_splits=5, embargo_pct=0.01)

    assert result["supera_azar"] is True
    assert result["accuracy_media"] > result["baseline_azar"] + 0.15
    assert result["n_muestras"] == 600
    assert set(result.keys()) == {
        "accuracy_scores", "accuracy_media", "f1_scores", "f1_media",
        "baseline_azar", "baseline_mayoritaria", "supera_azar", "supera_mayoritaria",
        "n_clases", "n_muestras",
    }


def test_evaluate_primary_stays_near_chance_on_random_labels() -> None:
    X, y, t1 = _synthetic_dataset(n=600, horizon=10, seed=1, learnable=False)

    result = evaluate_primary(X, y, t1, n_splits=5, embargo_pct=0.01)

    # Con etiquetas independientes de las features, no debería haber una
    # brecha grande sobre el azar como en el caso aprendible (puede haber
    # algo de ruido finito, pero no una señal real).
    assert result["accuracy_media"] < result["baseline_azar"] + 0.15


# --------------------------------------------------------------------------
# oos_predictions
# --------------------------------------------------------------------------


def test_oos_predictions_covers_every_sample_with_valid_classes() -> None:
    X, y, t1 = _synthetic_dataset(n=300, horizon=10, seed=2, learnable=True)

    preds = oos_predictions(X, y, t1, n_splits=5, embargo_pct=0.01)

    assert len(preds) == len(X)
    assert not preds.isna().any()
    assert set(preds.unique()).issubset({-1.0, 0.0, 1.0})
    assert preds.index.equals(X.index)


def test_oos_predictions_accepts_sample_weight() -> None:
    X, y, t1 = _synthetic_dataset(n=200, horizon=5, seed=3, learnable=True)
    weights = pd.Series(1.0, index=X.index)

    preds = oos_predictions(X, y, t1, sample_weight=weights, n_splits=4, embargo_pct=0.01)

    assert len(preds) == len(X)
    assert not preds.isna().any()


# --------------------------------------------------------------------------
# fit_final
# --------------------------------------------------------------------------


def test_fit_final_produces_a_model_that_predicts_valid_encoded_classes() -> None:
    X, y, _ = _synthetic_dataset(n=200, horizon=5, seed=4, learnable=True)
    model = fit_final(X, y)

    encoded_pred = model.predict(X)
    assert set(np.unique(encoded_pred)).issubset({0, 1, 2})


# --------------------------------------------------------------------------
# feature_importances
# --------------------------------------------------------------------------


def test_feature_importances_has_one_entry_per_feature_and_positive_sum() -> None:
    X, y, _ = _synthetic_dataset(n=300, horizon=10, seed=5, learnable=True)
    model = fit_final(X, y)

    importances = feature_importances(model, X.columns)

    assert len(importances) == X.shape[1]
    assert set(importances.index) == set(X.columns)
    assert importances.sum() > 0.0
    assert (importances >= 0.0).all()
    # ya viene ordenada de mayor a menor
    assert importances.is_monotonic_decreasing


# --------------------------------------------------------------------------
# model_to_positions
# --------------------------------------------------------------------------


def test_model_to_positions_maps_identity_for_valid_labels() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    preds = pd.Series([1.0, -1.0, 0.0, 1.0], index=idx)

    positions = model_to_positions(preds)

    pd.testing.assert_series_equal(positions, preds.rename("position"))


def test_model_to_positions_rejects_unexpected_values() -> None:
    preds = pd.Series([1.0, 2.0, 0.0])
    with pytest.raises(ValueError):
        model_to_positions(preds)
