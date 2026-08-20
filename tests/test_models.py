"""Tests offline para ml/models.py (datasets sintéticos, sin red).

`evaluate_with_without_onchain` es la excepción (Fase 5b): al ser una
función orquestadora `asset -> dict` que arma sus propias features/labels
desde cero, se testea contra el snapshot local real (`source="store"`),
igual que ya hace `tests/test_app_smoke.py` para `BacktestWorker` — no hay
red de por medio, solo lectura de los parquet ya exportados.

`BTC_onchain_1d.parquet` (`config.SNAPSHOT_DIR`) NO está versionado (ver
`.gitignore` y `scripts/export_onchain.py`) — a diferencia del snapshot de
precios "normal", que se comparte manualmente, el on-chain se genera aparte
y no siempre está presente (p. ej. en un clon limpio). El test que depende
de él (`test_evaluate_with_without_onchain_returns_expected_keys_on_real_btc_data`)
se OMITE (`pytest.skip`) en vez de fallar cuando ese archivo no existe — ver
Fase 14 / hallazgo F-01 de auditoría.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import SNAPSHOT_DIR
from ml.models import (
    evaluate_primary,
    evaluate_with_without_onchain,
    feature_importances,
    fit_final,
    latest_oos_prediction,
    make_primary_model,
    model_to_positions,
    oos_predictions,
    t1_from_labels,
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


# --------------------------------------------------------------------------
# t1_from_labels
# --------------------------------------------------------------------------


def test_t1_from_labels_uses_positional_offset() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    labels_df = pd.DataFrame(
        {"dias_hasta_evento": [2.0, 3.0, np.nan, 1.0, np.nan, 4.0, 2.0, np.nan, np.nan, np.nan]},
        index=idx,
    )
    valid_index = idx[[0, 1, 3, 5, 6]]

    t1 = t1_from_labels(labels_df, valid_index)

    assert t1.loc[idx[0]] == idx[0 + 2]
    assert t1.loc[idx[1]] == idx[1 + 3]
    assert t1.loc[idx[3]] == idx[3 + 1]
    assert t1.loc[idx[5]] == idx[5 + 4]
    assert t1.loc[idx[6]] == idx[6 + 2]
    assert t1.index.equals(valid_index)


# --------------------------------------------------------------------------
# latest_oos_prediction
# --------------------------------------------------------------------------


def test_latest_oos_prediction_returns_expected_structure() -> None:
    X, y, t1 = _synthetic_dataset(n=300, horizon=10, seed=7, learnable=True)

    result = latest_oos_prediction(X, y, t1, n_splits=5, embargo_pct=0.01)

    assert set(result.keys()) == {"fecha", "clase", "proba", "confianza"}
    assert result["fecha"] == X.index[-1]
    assert result["clase"] in {-1.0, 0.0, 1.0}
    assert set(result["proba"].keys()) == {-1.0, 0.0, 1.0}
    assert abs(sum(result["proba"].values()) - 1.0) < 1e-6
    assert result["confianza"] == max(result["proba"].values())
    assert result["proba"][result["clase"]] == result["confianza"]


# --------------------------------------------------------------------------
# evaluate_with_without_onchain (Fase 5b, datos reales vía source="store")
# --------------------------------------------------------------------------


def test_evaluate_with_without_onchain_returns_expected_keys_on_real_btc_data() -> None:
    # F-01 (auditoría, Fase 14): BTC_onchain_1d.parquet no está versionado
    # — sin él, este test se omite con un mensaje claro en vez de fallar
    # (p. ej. en un clon limpio que no corrió scripts/export_onchain.py).
    if not (SNAPSHOT_DIR / "BTC_onchain_1d.parquet").exists():
        pytest.skip("snapshot on-chain ausente (data/snapshot/BTC_onchain_1d.parquet); correr scripts/export_onchain.py")

    # n_splits chico a propósito: solo se busca validar la ESTRUCTURA del
    # resultado y que corre de punta a punta sobre datos reales, no medir
    # con precisión el desempeño (eso se corre aparte, con n_splits=5, para
    # la validación honesta que se reporta en el PR de esta fase).
    result = evaluate_with_without_onchain("BTC", n_splits=3, embargo_pct=0.01)

    assert result["asset"] == "BTC"
    assert result["n_muestras_periodo_comun"] > 0
    assert result["periodo_inicio"] < result["periodo_fin"]
    assert len(result["columnas_onchain_usadas"]) > 0
    assert "net_flow_zscore" in result["columnas_onchain_usadas"]

    expected_sub_keys = {
        "accuracy_scores", "accuracy_media", "f1_scores", "f1_media",
        "baseline_azar", "baseline_mayoritaria", "supera_azar", "supera_mayoritaria",
        "n_clases", "n_muestras", "roc_auc_scores", "roc_auc_media",
    }
    for key in ("solo_tecnicas", "con_onchain"):
        assert set(result[key].keys()) == expected_sub_keys
        assert len(result[key]["accuracy_scores"]) == 3
        assert len(result[key]["roc_auc_scores"]) == 3

    # Ambas configuraciones se evaluaron sobre EXACTAMENTE la misma cantidad
    # de muestras (mismo período común) — la comparación A/B es justa.
    assert result["solo_tecnicas"]["n_muestras"] == result["con_onchain"]["n_muestras"]

    assert isinstance(result["mejora_accuracy_media"], float)
    assert 0 <= result["folds_con_mejora_onchain"] <= 3
    assert isinstance(result["onchain_mejora_de_forma_consistente"], bool)


def test_evaluate_with_without_onchain_raises_for_asset_without_onchain_coverage() -> None:
    # SOL no tiene snapshot on-chain (ver data/onchain.py) — la comparación
    # A/B no tiene sentido sin una segunda configuración que comparar.
    with pytest.raises(ValueError, match="on-chain"):
        evaluate_with_without_onchain("SOL", n_splits=3)
