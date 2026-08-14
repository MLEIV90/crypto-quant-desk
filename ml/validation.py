"""Validación sin fugas temporales: Purged K-Fold + embargo (López de Prado,
"Advances in Financial Machine Learning", cap. 7).

Fase 3b: SOLO el aparato de validación, testeado con estimadores simples de
scikit-learn sobre datos sintéticos. No se entrena ningún modelo real acá
(eso es Fase 3c).

CONTEXTO — por qué un K-Fold estándar filtra información: las etiquetas de
`ml.labeling.triple_barrier_labels` tienen un horizonte [fecha_entrada, t1]
que puede solaparse con el de otras muestras cercanas en el tiempo (dos
trades a pocos días de distancia comparten buena parte de su ventana
futura). Un K-Fold estándar arma folds al azar (o en bloques, pero sin
tener en cuenta ese solapamiento): si una muestra de train termina en el
mismo período que una muestra de test, el modelo puede "ver" en el train
información sobre el mismo tramo de precios que después se usa para
evaluarlo en el test — un test optimista, no una medida honesta de
generalización.

La solución tiene dos partes:
- PURGING: quitar del train toda muestra cuyo intervalo [inicio, t1] se
  solape temporalmente con el rango del bloque de test (ver
  `get_train_times`).
- EMBARGO: quitar además del train las muestras que empiezan en una franja
  corta INMEDIATAMENTE DESPUÉS del test — por autocorrelación serial, la
  información de esos días todavía puede estar "contaminada" por lo que
  pasó durante el test, aunque su horizonte no se solape técnicamente.

Convención del proyecto: `t1` es una `pd.Series` cuyo ÍNDICE es la fecha de
inicio de cada muestra (la fecha de entrada del triple-barrier) y cuyo
VALOR es la fecha en la que se resuelve su etiqueta (equivalente a
`fecha_entrada + dias_hasta_evento` de `ml.labeling.triple_barrier_labels`,
ya en formato de timestamp en vez de cantidad de días).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS: int = 5
DEFAULT_EMBARGO_PCT: float = 0.01

_SCORERS = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro", zero_division=0),
}

# "roc_auc" se maneja aparte de _SCORERS (no en ese dict): a diferencia de
# accuracy/f1/precision, necesita PROBABILIDADES (`predict_proba`), no la
# clase predicha (`predict`), y para multiclase one-vs-rest necesita además
# el set COMPLETO de clases del problema (`labels=`) — ver `purged_cv_score`.
_PROBA_SCORING_KEYS = {"roc_auc"}


def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """PURGE (AFML cap. 7.1): dado `t1` (candidatas de train: índice =
    inicio de la muestra, valor = fecha de resolución de su etiqueta) y
    `test_times` (índice = inicio de cada bloque de test, valor = fin de
    ese bloque — puede tener más de un bloque), devuelve el subconjunto de
    `t1` cuyas muestras NO se solapan temporalmente con NINGÚN bloque de
    test.

    Una muestra de train [inicio_i, t1_i] se purga si, para algún bloque de
    test [test_inicio, test_fin]:
    1. inicio_i cae DENTRO del bloque de test (la muestra empieza durante
       el test), o
    2. t1_i cae DENTRO del bloque de test (la muestra TERMINA/resuelve su
       etiqueta durante el test: usó información de precios que el test
       también usa para evaluarse — el caso central de leakage), o
    3. [inicio_i, t1_i] ENVUELVE por completo al bloque de test (la muestra
       empezó antes y termina después: usó información de todo el período
       de test para resolver su propia etiqueta).

    Devuelve un `pd.Series` (subconjunto de `t1`) con las muestras que
    sobreviven al purge — su ÍNDICE son las muestras de train válidas.
    """
    train = t1.copy()
    for test_start, test_end in test_times.items():
        starts_during_test = train[(train.index >= test_start) & (train.index <= test_end)].index
        ends_during_test = train[(train >= test_start) & (train <= test_end)].index
        envelops_test = train[(train.index <= test_start) & (train >= test_end)].index
        overlapping = starts_during_test.union(ends_during_test).union(envelops_test)
        train = train.drop(overlapping)
    return train


class PurgedKFold:
    """K-Fold con PURGING + EMBARGO, compatible con la interfaz `split()` de
    scikit-learn (se puede pasar como `cv=` a `cross_val_score`,
    `cross_validate`, etc.).

    A diferencia de `sklearn.model_selection.KFold`, el bloque de test de
    cada fold es siempre un tramo CONTIGUO en el tiempo (no aleatorio: barajar
    folds temporales es precisamente lo que causa el solapamiento que este
    validador existe para evitar), y el train de cada fold se purga +
    embarga contra ese bloque antes de devolverse.

    Parámetros
    ----------
    n_splits:
        Cantidad de folds (bloques de test contiguos, en orden temporal).
    t1:
        `pd.Series` obligatoria: índice = inicio de cada muestra, valor =
        fecha de resolución de su etiqueta (ver docstring del módulo). Debe
        tener la misma longitud, en el mismo orden, que `X` en `split()`.
    embargo_pct:
        Fracción del total de muestras (0 a 1) que se descarta del train
        justo DESPUÉS de cada bloque de test, además del purge. Con
        `embargo_pct=0` no hay embargo, solo purge.
    """

    def __init__(self, n_splits: int = DEFAULT_N_SPLITS, t1: pd.Series | None = None, embargo_pct: float = DEFAULT_EMBARGO_PCT) -> None:
        if t1 is None:
            raise ValueError("PurgedKFold: 't1' es obligatorio (fechas de resolución de cada etiqueta)")
        if n_splits < 2:
            raise ValueError(f"PurgedKFold: n_splits debe ser >= 2, recibido {n_splits}")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X: pd.DataFrame | None = None, y=None, groups=None) -> int:
        """Firma estándar de scikit-learn: cantidad de folds."""
        return self.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """Genera `(train_positions, test_positions)` por fold, como arrays
        de posiciones ENTERAS (0-based), igual que cualquier splitter de
        scikit-learn — no fechas ni etiquetas, para poder indexar `X`/`y`
        directamente con `.iloc`.
        """
        if len(X) != len(self.t1):
            raise ValueError(
                f"PurgedKFold.split: X tiene {len(X)} filas pero t1 tiene {len(self.t1)} — deben "
                "corresponder a las mismas muestras, en el mismo orden"
            )

        n = len(X)
        indices = np.arange(n)
        embargo_size = int(n * self.embargo_pct)

        # Bloques de test CONTIGUOS en el tiempo (no aleatorios).
        test_blocks = np.array_split(indices, self.n_splits)

        for test_positions in test_blocks:
            test_start_pos = int(test_positions[0])
            test_end_pos = int(test_positions[-1])
            test_start_time = self.t1.index[test_start_pos]
            # El fin del bloque de test, a efectos de purga, es el t1 MÁS
            # TARDÍO entre las muestras de test (no la fecha de inicio de la
            # última muestra: su etiqueta puede resolverse bastante después).
            test_end_time = self.t1.iloc[test_positions].max()

            test_times = pd.Series([test_end_time], index=[test_start_time])
            purged_t1 = get_train_times(self.t1, test_times)

            # EMBARGO: además del purge, se descartan las muestras cuyo
            # INICIO cae en la franja (fin del bloque de test posicional,
            # fin + embargo_size] — ver docstring del módulo.
            if embargo_size > 0:
                embargo_end_pos = min(test_end_pos + embargo_size, n - 1)
                embargo_start_time = self.t1.index[test_end_pos]
                embargo_end_time = self.t1.index[embargo_end_pos]
                in_embargo = (purged_t1.index > embargo_start_time) & (purged_t1.index <= embargo_end_time)
                purged_t1 = purged_t1[~in_embargo]

            train_positions = self.t1.index.get_indexer(purged_t1.index)
            yield train_positions, test_positions


def purged_cv_score(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_pct: float = DEFAULT_EMBARGO_PCT,
    sample_weight: pd.Series | None = None,
    scoring: str = "accuracy",
) -> dict:
    """Corre validación cruzada purgeada (`PurgedKFold`) con `estimator`
    (cualquier estimador scikit-learn: `.fit(X, y)` / `.predict(X)`, con
    `sample_weight` opcional en `.fit` si `sample_weight` no es None) y
    devuelve el score por fold más el resumen.

    Esta función solo ORQUESTA fit/predict/score sobre los folds que arma
    `PurgedKFold` — no reimplementa la purga ni el embargo. El estimador se
    clona (`sklearn.base.clone`) en cada fold para no arrastrar estado
    entre folds.

    `scoring`: "accuracy", "f1" (macro), "precision" (macro) — sobre la
    clase predicha (`estimator.predict`) — o "roc_auc" (one-vs-rest,
    promedio macro, sobre `estimator.predict_proba`; requiere que
    `estimator` implemente `predict_proba`, p. ej. cualquier clasificador
    de scikit-learn/XGBoost). Con "roc_auc", si algún fold de test termina
    con MENOS de 2 clases presentes (caso borde en folds chicos/desbalanceados
    — el ROC-AUC de esa clase queda indefinido), ese fold se omite del
    promedio y se loguea un warning explícito en vez de fallar todo el
    cómputo o inventar un valor.

    Devuelve un dict: "scores" (lista, un valor por fold — puede tener NaN
    en folds omitidos de "roc_auc"), "score_medio" (promedio SOLO de los
    folds válidos), "score_std" (0.0 si hay un solo fold válido, NaN si
    ninguno lo es), "n_splits".
    """
    if scoring not in _SCORERS and scoring not in _PROBA_SCORING_KEYS:
        valid = list(_SCORERS) + list(_PROBA_SCORING_KEYS)
        raise ValueError(f"purged_cv_score: scoring debe ser uno de {valid}, recibido '{scoring}'")

    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)
    all_labels = sorted(y.unique()) if scoring in _PROBA_SCORING_KEYS else None

    scores: list[float] = []
    for fold_i, (train_pos, test_pos) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]

        model = clone(estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.iloc[train_pos].to_numpy()
        model.fit(X_train, y_train, **fit_kwargs)

        if scoring in _PROBA_SCORING_KEYS:
            if y_test.nunique() < 2:
                logger.warning(
                    "purged_cv_score: fold %d/%d sin al menos 2 clases en el test, ROC-AUC indefinido — se omite (NaN)",
                    fold_i + 1, n_splits,
                )
                scores.append(float("nan"))
                continue
            y_proba = model.predict_proba(X_test)
            try:
                score = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=all_labels))
            except ValueError as exc:
                logger.warning(
                    "purged_cv_score: fold %d/%d ROC-AUC no se pudo calcular (%s) — se omite (NaN)",
                    fold_i + 1, n_splits, exc,
                )
                score = float("nan")
        else:
            y_pred = model.predict(X_test)
            score = float(_SCORERS[scoring](y_test, y_pred))

        scores.append(score)
        logger.info(
            "purged_cv_score: fold %d/%d, %s=%.4f (train=%d, test=%d)",
            fold_i + 1, n_splits, scoring, score, len(train_pos), len(test_pos),
        )

    valid_scores = [s for s in scores if not np.isnan(s)]
    return {
        "scores": scores,
        "score_medio": float(np.mean(valid_scores)) if valid_scores else float("nan"),
        "score_std": float(np.std(valid_scores, ddof=1)) if len(valid_scores) > 1 else 0.0,
        "n_splits": n_splits,
    }
