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

Fase 5b agrega `evaluate_with_without_onchain`: la MISMA vara de
honestidad, ahora aplicada a la pregunta "¿agregar features on-chain
(`data/onchain.py`, vía `ml.features.build_feature_matrix(..., include_onchain=True)`)
mejora la predicción de forma consistente, o es solo ruido de un fold
suelto?" — compara ambas configuraciones sobre el MISMO período y las
MISMAS etiquetas, para que la diferencia (si la hay) sea atribuible a las
features, no a estar mirando distintos tramos de historia.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from data.loaders import get_prices
from ml.features import align_features_labels, build_feature_matrix
from ml.labeling import get_daily_volatility, triple_barrier_labels
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


def evaluate_primary_with_roc_auc(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> dict:
    """`evaluate_primary` + ROC-AUC purgeado (one-vs-rest, macro,
    `ml.validation.purged_cv_score(..., scoring="roc_auc")`, Fase 5b),
    agregado como "roc_auc_scores"/"roc_auc_media" al dict que ya arma
    `evaluate_primary`.

    Separada de `evaluate_primary` a propósito, en vez de agregarle las
    claves ahí directo: `evaluate_primary` tiene un contrato de salida ya
    testeado (ver `tests/test_models.py`) y cambiarle las claves rompería
    ese contrato para quien ya lo consume. Esta función es la que usan
    `evaluate_with_without_onchain` (Fase 5b) y el worker de predicción de
    la UI (`app.workers.PredictionWorker`) para no reimplementar el mismo
    cableado dos veces.
    """
    result = evaluate_primary(X, y, t1, sample_weight=sample_weight, n_splits=n_splits, embargo_pct=embargo_pct)
    roc_auc = purged_cv_score(
        make_primary_model(), X, _encode_labels(y), t1,
        n_splits=n_splits, embargo_pct=embargo_pct, sample_weight=sample_weight, scoring="roc_auc",
    )
    result["roc_auc_scores"] = roc_auc["scores"]
    result["roc_auc_media"] = roc_auc["score_medio"]
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


def t1_from_labels(labels_df: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Construye `t1` (convención de `ml.validation`: índice = inicio de la
    muestra, valor = fecha de resolución de su etiqueta) a partir de
    `labels_df` (`ml.labeling.triple_barrier_labels`), restringido a
    `index`.

    La fecha de resolución se ubica por POSICIÓN dentro del índice completo
    de `labels_df` (`dias_hasta_evento` cuenta FILAS futuras, no días de
    calendario — así resuelve el evento `triple_barrier_labels`
    internamente, indexando `closes[t + offset]`) en vez de sumar un
    `Timedelta` a la fecha: es exacto incluso si la serie de precios
    tuviera algún gap de calendario, donde "fecha + N días" y "N filas hacia
    adelante" dejarían de coincidir.
    """
    subset = labels_df.loc[index]
    positions = labels_df.index.get_indexer(index)
    offsets = subset["dias_hasta_evento"].astype(int).to_numpy()
    resolution_dates = labels_df.index[positions + offsets]
    return pd.Series(resolution_dates, index=index)


def evaluate_with_without_onchain(asset: str, n_splits: int = 5, embargo_pct: float = 0.01) -> dict:
    """Comparación A/B HONESTA (Fase 5b): ¿agregar features on-chain
    (`data/onchain.py`) mejora la predicción OOS purgeada del modelo
    primario, de forma CONSISTENTE entre folds, o es ruido?

    Arma DOS matrices de features para `asset` (`data.loaders.get_prices`,
    `source="store"`) con `ml.features.build_feature_matrix`: una solo con
    técnicas (`include_onchain=False`) y otra enriquecida
    (`include_onchain=True`) — MISMO target (`ml.labeling.triple_barrier_labels`
    sobre el mismo `close`), y ambas restringidas al MISMO período común:
    el que tiene cobertura on-chain (las fechas fuera de esa cobertura
    quedan NaN en la config enriquecida y `align_features_labels` ya las
    dropea; la config solo-técnica se recorta a esas mismas fechas a
    propósito, para que la comparación no esté confundida por evaluar sobre
    distinta cantidad/tramo de historia — un modelo entrenado sobre MÁS años
    de datos técnicos no es comparable, sin más, contra uno entrenado sobre
    solo los años con on-chain).

    Evalúa ambas configuraciones con `evaluate_primary_with_roc_auc`
    (accuracy y F1 purgeados, comparados contra los baselines de
    azar/mayoritaria, más ROC-AUC one-vs-rest purgeado — ver ese
    docstring).

    La pregunta central del A/B (`onchain_mejora_de_forma_consistente`) es
    ESTRICTA a propósito: exige que el accuracy purgeado con on-chain sea
    mayor que sin on-chain en TODOS los folds, no solo en el promedio — un
    promedio mejor puede esconder que la mejora vino de un solo fold
    favorable, lo que sería suerte de la partición temporal, no una señal
    robusta. Si no se cumple, se loguea un warning explícito — este módulo
    no maquilla un resultado nulo o mixto para que parezca mejor.

    Devuelve un dict:
    - "asset", "n_muestras_periodo_comun", "periodo_inicio"/"periodo_fin".
    - "columnas_onchain_usadas": las columnas on-chain que sí se pudieron
      agregar para este activo (puede ser un subconjunto, ver cobertura
      parcial en `data/onchain.py`).
    - "solo_tecnicas" / "con_onchain": cada uno, el dict completo de
      `evaluate_primary` más "roc_auc_scores"/"roc_auc_media".
    - "mejora_accuracy_media": accuracy_media(con_onchain) - accuracy_media(solo_tecnicas).
    - "folds_con_mejora_onchain": cantidad de folds (de `n_splits`) donde el
      accuracy con on-chain superó al accuracy solo-técnico en ESE fold.
    - "onchain_mejora_de_forma_consistente": bool, True solo si
      `folds_con_mejora_onchain == n_splits` Y `mejora_accuracy_media > 0`.
    """
    df = get_prices(asset, source="store")
    close = df["close"]

    X_technical_raw = build_feature_matrix(df, include_onchain=False)
    X_enriched_raw = build_feature_matrix(df, include_onchain=True, asset=asset)

    onchain_columns = [c for c in X_enriched_raw.columns if c not in X_technical_raw.columns]
    if not onchain_columns:
        raise ValueError(
            f"evaluate_with_without_onchain: '{asset}' no tiene ninguna columna on-chain disponible "
            "(ver data/onchain.py) — la comparación A/B no tiene sentido sin on-chain."
        )

    volatility = get_daily_volatility(close)
    labels_df = triple_barrier_labels(close, volatility)

    X_enriched, y_enriched, w_enriched = align_features_labels(X_enriched_raw, labels_df)
    X_technical_full, y_technical_full, w_technical_full = align_features_labels(X_technical_raw, labels_df)

    common_index = X_enriched.index.intersection(X_technical_full.index)
    if len(common_index) < n_splits * 10:
        raise ValueError(
            f"evaluate_with_without_onchain: '{asset}' tiene muy pocas muestras en el período común "
            f"con on-chain ({len(common_index)}) para {n_splits} folds — subí el rango de datos o bajá n_splits."
        )

    X_technical = X_technical_full.loc[common_index]
    y_technical = y_technical_full.loc[common_index]
    w_technical = w_technical_full.loc[common_index]

    X_enriched = X_enriched.loc[common_index]
    y_enriched = y_enriched.loc[common_index]
    w_enriched = w_enriched.loc[common_index]

    t1 = t1_from_labels(labels_df, common_index)

    logger.info(
        "evaluate_with_without_onchain: '%s' — %d muestras en el período común [%s, %s], on-chain: %s",
        asset, len(common_index), common_index.min().date(), common_index.max().date(), onchain_columns,
    )

    resultado_tecnico = evaluate_primary_with_roc_auc(
        X_technical, y_technical, t1, sample_weight=w_technical, n_splits=n_splits, embargo_pct=embargo_pct
    )
    resultado_enriquecido = evaluate_primary_with_roc_auc(
        X_enriched, y_enriched, t1, sample_weight=w_enriched, n_splits=n_splits, embargo_pct=embargo_pct
    )

    mejora_accuracy_media = resultado_enriquecido["accuracy_media"] - resultado_tecnico["accuracy_media"]
    folds_con_mejora = int(sum(
        e > t for e, t in zip(resultado_enriquecido["accuracy_scores"], resultado_tecnico["accuracy_scores"])
    ))
    mejora_consistente = bool(folds_con_mejora == n_splits and mejora_accuracy_media > 0.0)

    result = {
        "asset": asset,
        "n_muestras_periodo_comun": len(common_index),
        "periodo_inicio": common_index.min(),
        "periodo_fin": common_index.max(),
        "columnas_onchain_usadas": onchain_columns,
        "solo_tecnicas": resultado_tecnico,
        "con_onchain": resultado_enriquecido,
        "mejora_accuracy_media": mejora_accuracy_media,
        "folds_con_mejora_onchain": folds_con_mejora,
        "onchain_mejora_de_forma_consistente": mejora_consistente,
    }

    if mejora_consistente:
        logger.info(
            "evaluate_with_without_onchain: '%s' — on-chain mejora el accuracy purgeado en TODOS los folds "
            "(%d/%d), delta medio=%+.4f", asset, folds_con_mejora, n_splits, mejora_accuracy_media,
        )
    else:
        logger.warning(
            "evaluate_with_without_onchain: '%s' — on-chain NO mejora de forma consistente entre folds "
            "(mejoró en %d/%d, delta accuracy medio=%+.4f). No hay evidencia robusta de edge on-chain acá.",
            asset, folds_con_mejora, n_splits, mejora_accuracy_media,
        )

    return result


def latest_oos_prediction(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> dict:
    """Predicción OOS honesta (con probabilidades) para la ÚLTIMA fila de
    `X` — pensada para la pestaña "Predicción (ML)" de la UI
    (`app.workers.PredictionWorker`), que necesita una CONFIANZA real (la
    probabilidad que le asignó el modelo a la clase predicha), no solo la
    clase dura que devuelve `oos_predictions`.

    Entrena el modelo del ÚLTIMO fold de `ml.validation.PurgedKFold` (su
    bloque de test cubre, por construcción, el tramo más reciente de la
    serie — los bloques de `PurgedKFold.split` son contiguos y en orden
    temporal ascendente) y predice sobre la última fila con
    `predict_proba`. Es exactamente la misma predicción que produciría
    `oos_predictions` para esa fila (mismo fold, mismo split, un modelo que
    nunca vio esa muestra durante su entrenamiento) — esta función solo
    evita reconstruir TODOS los folds cuando lo único que hace falta es la
    fecha más reciente.

    Devuelve un dict: "fecha" (`X.index[-1]`), "clase" (-1.0/0.0/+1.0, ya
    decodificada), "proba" (dict `{clase: probabilidad}` para las 3
    clases), "confianza" (la probabilidad de la clase predicha, la más
    alta de las 3).
    """
    y_encoded = _encode_labels(y)
    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    last_train_pos, last_test_pos = None, None
    for train_pos, test_pos in cv.split(X):
        last_train_pos, last_test_pos = train_pos, test_pos

    model = make_primary_model()
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.iloc[last_train_pos].to_numpy()
    model.fit(X.iloc[last_train_pos], y_encoded.iloc[last_train_pos], **fit_kwargs)

    last_row_pos = last_test_pos[-1]
    proba_row = model.predict_proba(X.iloc[[last_row_pos]])[0]

    proba_by_label = {_CLASS_TO_LABEL[int(c)]: float(p) for c, p in zip(model.classes_, proba_row)}
    predicted_class = max(proba_by_label, key=proba_by_label.get)

    logger.info(
        "latest_oos_prediction: fecha=%s clase=%s confianza=%.4f",
        X.index[last_row_pos], predicted_class, proba_by_label[predicted_class],
    )

    return {
        "fecha": X.index[last_row_pos],
        "clase": predicted_class,
        "proba": proba_by_label,
        "confianza": proba_by_label[predicted_class],
    }
