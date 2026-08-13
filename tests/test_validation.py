"""Tests offline para ml/validation.py (datasets sintéticos, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml.validation import PurgedKFold, get_train_times, purged_cv_score


def _overlapping_horizon_dataset(
    n: int, horizon: int, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Dataset sintético con etiquetas que son función del FUTURO: label_i
    depende de hacia dónde se mueve `state` (un random walk) entre i e
    i+horizon. `t1_i` es la fecha en la que se "resuelve" esa etiqueta
    (i+horizon) — exactamente como el t1 de un triple-barrier. Muestras
    cercanas en el tiempo comparten la mayor parte de su ventana futura
    (horizontes solapados): el ingrediente necesario para poder demostrar
    el leakage de un K-Fold estándar.
    """
    rng = np.random.default_rng(seed)
    state = np.cumsum(rng.normal(0.0, 1.0, n + horizon))
    idx = pd.date_range("2020-01-01", periods=n + horizon, freq="D")

    X = pd.DataFrame({"x": state[:n]}, index=idx[:n])
    y = pd.Series(np.where(state[horizon : horizon + n] > state[:n], 1, -1), index=idx[:n])
    t1 = pd.Series(idx[horizon : horizon + n], index=idx[:n])
    return X, y, t1


# --------------------------------------------------------------------------
# get_train_times: los 3 casos de solapamiento
# --------------------------------------------------------------------------


def test_get_train_times_purges_samples_starting_or_ending_during_test() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    t1 = pd.Series(idx + pd.Timedelta(days=3), index=idx)  # cada muestra dura 3 días
    test_times = pd.Series([idx[6]], index=[idx[4]])  # bloque de test: día4 a día6

    train = get_train_times(t1, test_times)

    # empiezan durante el test (día4: [4,7], día5: [5,8], día6: [6,9])
    for day in (4, 5, 6):
        assert idx[day] not in train.index
    # terminan durante el test (día1: [1,4], día2: [2,5], día3: [3,6])
    for day in (1, 2, 3):
        assert idx[day] not in train.index
    # ni empiezan ni terminan durante el test, ni lo envuelven -> sobreviven
    for day in (0, 7, 8, 9):
        assert idx[day] in train.index


def test_get_train_times_purges_samples_that_envelop_the_test_block() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    t1 = pd.Series(idx, index=idx)  # sin solapamiento por defecto (horizonte 0)
    t1.iloc[1] = idx[8]  # la muestra de día1 resuelve recién en día8: envuelve el test [4,6]
    test_times = pd.Series([idx[6]], index=[idx[4]])

    train = get_train_times(t1, test_times)

    assert idx[1] not in train.index  # envuelve el test -> purgada
    assert idx[0] in train.index
    assert idx[3] in train.index  # termina en día3, antes del test -> sobrevive


# --------------------------------------------------------------------------
# Contaminación controlada: el test clave
# --------------------------------------------------------------------------


def test_purged_kfold_gives_notably_lower_accuracy_than_standard_kfold_on_leaky_data() -> None:
    """Con etiquetas que son función del futuro y horizontes solapados,
    un K-Fold estándar (que baraja folds al azar) deja que muestras de
    train casi-duplicadas de una muestra de test terminen en train,
    inflando artificialmente el accuracy. PurgedKFold, al purgar ese
    solapamiento, no se deja "engañar" tanto — su accuracy debe ser
    notablemente MENOR.

    Se usa 1-NN a propósito: es el modelo que más se beneficia de una
    muestra casi-duplicada en train (la "memoriza" y acierta trivialmente
    si esa muestra queda en test).
    """
    X, y, t1 = _overlapping_horizon_dataset(n=580, horizon=20, seed=0)
    knn = KNeighborsClassifier(n_neighbors=1)

    standard_scores = []
    for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        knn.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = knn.predict(X.iloc[test_idx])
        standard_scores.append((pred == y.iloc[test_idx].to_numpy()).mean())
    standard_accuracy = float(np.mean(standard_scores))

    purged_result = purged_cv_score(knn, X, y, t1, n_splits=5, embargo_pct=0.01, scoring="accuracy")
    purged_accuracy = purged_result["score_medio"]

    assert standard_accuracy - purged_accuracy > 0.05


# --------------------------------------------------------------------------
# El purge no deja solapamiento en ningún fold
# --------------------------------------------------------------------------


def test_purged_kfold_train_never_overlaps_test_range_in_any_fold() -> None:
    X, _, t1 = _overlapping_horizon_dataset(n=300, horizon=15, seed=2)

    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
    for train_pos, test_pos in pkf.split(X):
        test_start = t1.index[test_pos[0]]
        test_end = t1.iloc[test_pos].max()

        train_starts = t1.index[train_pos]
        train_t1 = t1.iloc[train_pos]

        starts_during_test = (train_starts >= test_start) & (train_starts <= test_end)
        ends_during_test = (train_t1 >= test_start) & (train_t1 <= test_end)
        envelops_test = (train_starts <= test_start) & (train_t1 >= test_end)

        assert not (starts_during_test | ends_during_test | envelops_test).any()


# --------------------------------------------------------------------------
# El embargo elimina la franja posterior al test
# --------------------------------------------------------------------------


def test_embargo_removes_more_samples_than_no_embargo() -> None:
    # Horizonte CHICO (2 días) para que el purge por solapamiento deje
    # espacio: si el embargo (5% de 300 ~ 15 muestras) es mucho más ancho
    # que el horizonte, hay muestras que sobreviven al purge puro pero caen
    # dentro de la franja de embargo.
    X, _, t1 = _overlapping_horizon_dataset(n=300, horizon=2, seed=3)

    pkf_no_embargo = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
    pkf_with_embargo = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)

    folds_no_embargo = list(pkf_no_embargo.split(X))
    folds_with_embargo = list(pkf_with_embargo.split(X))

    # Un fold intermedio (no el último): tiene que haber muestras DESPUÉS
    # del test para que el embargo tenga algo que quitar.
    fold_idx = 1
    train_no_embargo = folds_no_embargo[fold_idx][0]
    train_with_embargo = folds_with_embargo[fold_idx][0]

    assert len(train_with_embargo) < len(train_no_embargo)


# --------------------------------------------------------------------------
# Interfaz de scikit-learn
# --------------------------------------------------------------------------


def test_purged_kfold_get_n_splits() -> None:
    t1 = pd.Series(pd.date_range("2020-01-01", periods=50), index=pd.date_range("2020-01-01", periods=50))
    pkf = PurgedKFold(n_splits=7, t1=t1)
    assert pkf.get_n_splits() == 7


def test_purged_kfold_is_compatible_with_sklearn_cross_val_score() -> None:
    X, y, t1 = _overlapping_horizon_dataset(n=200, horizon=5, seed=4)
    pkf = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)

    scores = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), X, y, cv=pkf)

    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


# --------------------------------------------------------------------------
# purged_cv_score
# --------------------------------------------------------------------------


def test_purged_cv_score_returns_expected_structure() -> None:
    X, y, t1 = _overlapping_horizon_dataset(n=200, horizon=10, seed=5)
    weights = pd.Series(1.0, index=X.index)

    result = purged_cv_score(
        DecisionTreeClassifier(max_depth=3, random_state=0),
        X, y, t1, n_splits=4, embargo_pct=0.01, sample_weight=weights, scoring="accuracy",
    )

    assert set(result.keys()) == {"scores", "score_medio", "score_std", "n_splits"}
    assert len(result["scores"]) == 4
    assert result["n_splits"] == 4
    assert 0.0 <= result["score_medio"] <= 1.0
    assert result["score_std"] >= 0.0


def test_purged_cv_score_rejects_invalid_scoring() -> None:
    X, y, t1 = _overlapping_horizon_dataset(n=100, horizon=5, seed=6)
    with pytest.raises(ValueError):
        purged_cv_score(DecisionTreeClassifier(), X, y, t1, scoring="not_a_scoring")


# --------------------------------------------------------------------------
# Validaciones
# --------------------------------------------------------------------------


def test_purged_kfold_requires_t1() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5, t1=None)


def test_purged_kfold_requires_at_least_two_splits() -> None:
    t1 = pd.Series(pd.date_range("2020-01-01", periods=10), index=pd.date_range("2020-01-01", periods=10))
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1, t1=t1)


def test_purged_kfold_split_rejects_length_mismatch() -> None:
    t1 = pd.Series(pd.date_range("2020-01-01", periods=10), index=pd.date_range("2020-01-01", periods=10))
    X = pd.DataFrame({"x": range(5)})
    pkf = PurgedKFold(n_splits=2, t1=t1)
    with pytest.raises(ValueError):
        list(pkf.split(X))
