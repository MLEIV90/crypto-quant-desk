"""Modelo primario: gradient boosting (XGBoost) con validación purgeada.

Fase 3c: entrena y evalúa el modelo primario sobre el target triple-barrier
de Fase 3a (`ml.labeling`) y las features trailing de Fase 3a
(`ml.features`), usando la validación Purged K-Fold + embargo de Fase 3b
(`ml.validation`) — no se reimplementa nada de eso acá, solo se orquesta.

Convención de codificación de clases: las etiquetas del triple-barrier
viven en {-1, 0, +1} (ver `ml.labeling.triple_barrier_labels`), pero
XGBoost exige clases enteras 0..n_clases-1. Este módulo mapea
internamente {-1: 0, 0: 1, 1: 2} antes de cada `fit`/`predict` y decodifica
de vuelta a {-1, 0, +1} antes de devolver cualquier resultado — quien use
este módulo nunca ve ni tiene que pensar en la codificación interna.

HONESTIDAD ANTE TODO: `evaluate_primary` siempre compara el accuracy
purgeado contra el azar (1/n_clases) y contra la clase mayoritaria. Un
modelo que no le gana a esos dos baselines triviales no aporta nada, por
sofisticado que sea el algoritmo — y este módulo lo va a decir clarísimo en
vez de esconderlo detrás de una métrica que suene mejor.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.validation import PurgedKFold, purged_cv_score

logger = logging.getLogger(__name__)

# Triple-barrier -> clases 0-indexadas que exige XGBoost, y su inversa.
_LABEL_TO_CLASS: dict[float, int] = {-1.0: 0, 0.0: 1, 1.0: 2}
_CLASS_TO_LABEL: dict[int, float] = {v: k for k, v in _LABEL_TO_CLASS.items()}

_DEFAULT_PARAMS: dict = {
    "max_depth": 3,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}


def _encode_labels(y: pd.Series) -> pd.Series:
    encoded = y.astype(float).map(_LABEL_TO_CLASS)
    if encoded.isna().any():
        bad = set(y[encoded.isna()].unique())
        raise ValueError(f"_encode_labels: valores fuera de {{-1, 0, 1}} en 'y': {bad}")
    return encoded.astype(int)


def _decode_labels(y_encoded: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series([_CLASS_TO_LABEL[int(c)] for c in y_encoded], index=index, dtype=float)


def make_primary_model(**params) -> XGBClassifier:
    """Instancia el modelo primario: gradient boosting (XGBoost) con
    hiperparámetros CONSERVADORES por defecto, elegidos a propósito para
    MINIMIZAR sobreajuste sobre un dataset financiero (ruidoso, señal/ruido
    baja) en vez de maximizar el ajuste in-sample:

    - `max_depth=3`: árboles poco profundos. Cuanto más profundo un árbol,
      más fácil que memorice particularidades de la muestra de
      entrenamiento en vez de aprender algo que generalice.
    - `n_estimators=200` + `learning_rate=0.05`: muchos árboles, cada uno
      aportando poco (boosting "lento"). Más robusto que pocos árboles con
      learning_rate alto: una corrección chica por árbol evita que un solo
      árbol "raro" domine la predicción final.
    - `subsample=0.8`: cada árbol se entrena con el 80% de las filas
      (muestreadas al azar) — bagging dentro del boosting, reduce la
      varianza del ensamble.
    - `colsample_bytree=0.8`: cada árbol ve el 80% de las columnas — evita
      que el modelo dependa demasiado de una sola feature (o de un grupo
      chico de features correlacionadas) para todas sus decisiones.
    - `reg_lambda=1.0`: regularización L2 sobre los pesos de las hojas (el
      default de XGBoost, dejado explícito acá como elección deliberada,
      no un olvido).

    `**params` sobreescribe cualquiera de estos defaults, o agrega otros
    parámetros válidos de `xgboost.XGBClassifier`.
    """
    final_params = {**_DEFAULT_PARAMS, **params}
    return XGBClassifier(**final_params)


def evaluate_primary(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> dict:
    """Evalúa el modelo primario con validación Purged K-Fold
    (`ml.validation.purged_cv_score`, reutilizada tal cual — no
    reimplementada acá), comparado SIEMPRE contra dos baselines triviales:

    - "baseline_azar": 1 / n_clases presentes en `y` (un clasificador que
      tirara una moneda de n_clases lados).
    - "baseline_mayoritaria": la fracción de la clase más frecuente en `y`
      (un clasificador que SIEMPRE predijera esa clase, sin mirar ninguna
      feature).

    Devuelve un dict: "accuracy_scores" (por fold), "accuracy_media",
    "f1_scores" (por fold, F1 macro), "f1_media", "baseline_azar",
    "baseline_mayoritaria", "supera_azar" (bool), "supera_mayoritaria"
    (bool), "n_clases", "n_muestras". Si el modelo NO supera alguno de los
    dos baselines, se loguea un warning explícito — no queda escondido.
    """
    y_encoded = _encode_labels(y)

    accuracy_result = purged_cv_score(
        make_primary_model(), X, y_encoded, t1, n_splits=n_splits, embargo_pct=embargo_pct,
        sample_weight=sample_weight, scoring="accuracy",
    )
    f1_result = purged_cv_score(
        make_primary_model(), X, y_encoded, t1, n_splits=n_splits, embargo_pct=embargo_pct,
        sample_weight=sample_weight, scoring="f1",
    )

    class_freq = y.value_counts(normalize=True)
    n_classes = len(class_freq)
    baseline_azar = 1.0 / n_classes
    baseline_mayoritaria = float(class_freq.max())
    accuracy_media = accuracy_result["score_medio"]

    result = {
        "accuracy_scores": accuracy_result["scores"],
        "accuracy_media": accuracy_media,
        "f1_scores": f1_result["scores"],
        "f1_media": f1_result["score_medio"],
        "baseline_azar": baseline_azar,
        "baseline_mayoritaria": baseline_mayoritaria,
        "supera_azar": bool(accuracy_media > baseline_azar),
        "supera_mayoritaria": bool(accuracy_media > baseline_mayoritaria),
        "n_clases": n_classes,
        "n_muestras": len(y),
    }

    if not result["supera_mayoritaria"]:
        logger.warning(
            "evaluate_primary: accuracy purgeada (%.4f) NO supera el baseline de clase mayoritaria "
            "(%.4f) — el modelo no aporta frente a predecir siempre la clase más común.",
            accuracy_media, baseline_mayoritaria,
        )
    if not result["supera_azar"]:
        logger.warning(
            "evaluate_primary: accuracy purgeada (%.4f) NO supera el azar (%.4f = 1/%d clases)",
            accuracy_media, baseline_azar, n_classes,
        )
    if result["supera_azar"] and result["supera_mayoritaria"]:
        logger.info(
            "evaluate_primary: accuracy purgeada (%.4f) supera azar (%.4f) y mayoritaria (%.4f)",
            accuracy_media, baseline_azar, baseline_mayoritaria,
        )

    return result


def oos_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> pd.Series:
    """Predicciones OUT-OF-SAMPLE: cada muestra la predice un modelo
    entrenado SIN ella (y sin ninguna muestra purgada/embargada contra su
    fold), al estilo de `sklearn.model_selection.cross_val_predict` pero
    con `ml.validation.PurgedKFold` en vez de un K-Fold estándar.

    Son las ÚNICAS predicciones honestas para backtestear (ver
    `model_to_positions` + `backtest.engine.backtest_from_prices`): usar las
    predicciones in-sample de `fit_final` para backtestear sería mirar el
    futuro con el propio modelo que "memorizó" ese futuro al entrenar.

    Como `PurgedKFold` parte TODO el dataset en bloques de test contiguos
    que cubren el 100% de las muestras (cada muestra es test en exactamente
    un fold), la serie devuelta tiene una predicción por CADA fila de `X`,
    sin huecos.

    Devuelve una `pd.Series` en {-1, 0, +1} (ya decodificada a la escala
    original del triple-barrier), indexada igual que `X`.
    """
    y_encoded = _encode_labels(y)
    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    predictions = pd.Series(np.nan, index=X.index, dtype=float)
    for fold_i, (train_pos, test_pos) in enumerate(cv.split(X)):
        model = make_primary_model()
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.iloc[train_pos].to_numpy()
        model.fit(X.iloc[train_pos], y_encoded.iloc[train_pos], **fit_kwargs)

        pred_encoded = model.predict(X.iloc[test_pos])
        predictions.iloc[test_pos] = _decode_labels(pred_encoded, X.index[test_pos]).to_numpy()
        logger.info("oos_predictions: fold %d/%d, %d predicciones", fold_i + 1, n_splits, len(test_pos))

    return predictions


def fit_final(X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None, **params) -> XGBClassifier:
    """Ajuste FINAL sobre TODO el dataset — para la predicción de la foto
    ACTUAL (p. ej. "¿qué dice el modelo hoy?"), NUNCA para backtestear (ver
    `oos_predictions` para eso: este modelo vio todos los datos, así que
    predecir sobre una muestra que ya vio durante el entrenamiento sería
    in-sample, no una medida honesta de desempeño).

    `**params` se pasan a `make_primary_model` (sobreescriben los defaults
    conservadores si hace falta).
    """
    model = make_primary_model(**params)
    y_encoded = _encode_labels(y)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.to_numpy()
    model.fit(X, y_encoded, **fit_kwargs)
    return model


def feature_importances(model: XGBClassifier, feature_names: pd.Index | list[str]) -> pd.Series:
    """Importancias de features del modelo ya ajustado (`model.feature_importances_`
    de XGBoost, por defecto basadas en la ganancia ("gain") promedio que
    aporta cada feature en los splits del ensamble), ordenadas de mayor a
    menor.
    """
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def model_to_positions(oos_pred: pd.Series) -> pd.Series:
    """Mapea la clase predicha (-1/0/+1, ver `oos_predictions`) a una
    posición direccional lista para `backtest.engine.backtest_from_prices`:
    +1 (largo) si predijo +1 (toca take-profit primero), -1 (corto) si
    predijo -1 (toca stop-loss primero), 0 (flat) si predijo 0 (vence el
    tiempo sin tocar ninguna barrera — ni señal de compra ni de venta).

    No aplica ningún shift: igual que `signals.engine.generate_positions`,
    es responsabilidad de `backtest.engine.run_backtest` desplazar la
    posición un día antes de aplicarla a un retorno (ver la regla
    anti-lookahead documentada en `backtest/engine.py`).
    """
    valid_values = set(_LABEL_TO_CLASS.keys())
    observed = set(oos_pred.dropna().astype(float).unique())
    unexpected = observed - valid_values
    if unexpected:
        raise ValueError(f"model_to_positions: valores inesperados en 'oos_pred': {unexpected}")

    return oos_pred.astype(float).rename("position")
