"""Meta-labeling (López de Prado, "Advances in Financial Machine Learning",
cap. 3): un segundo modelo que decide CUÁNDO operar la señal primaria y
CUÁNTO apostar, no QUÉ dirección tomar.

ENCUADRE — léase antes de usar este módulo: el meta-labeling NO predice
dirección. Eso ya lo decide un modelo PRIMARIO simple y transparente (acá,
`primary_side`: el signo del componente de tendencia de
`signals.engine.compute_signal_components`, no una regla de ML). El
meta-modelo resuelve un problema distinto y más chico: dado que la primaria
ya dijo "comprar" (o "vender") en la fecha t, ¿ese trade específico hubiera
sido bueno o malo? Es una clasificación BINARIA (1 = trade ganador según
triple-barrier, 0 = trade perdedor), mucho más fácil de aprender que
predecir la dirección del mercado (Fase 3c mostró que ESE problema, con
estas mismas features, no le gana ni al azar). La probabilidad que devuelve
el meta-modelo alimenta el SIZING de la posición final (`meta_positions`):
apostar más cuando la confianza es alta, nada cuando es baja — un filtro y
un dimensionador sobre la señal primaria, no un reemplazo.

Reutiliza sin reimplementar: `signals.engine.compute_signal_components`
(para `primary_side`), `ml.labeling.triple_barrier_labels` (para
`meta_labels`, en ambos lados — ver su docstring), `ml.validation.PurgedKFold`
(para `train_meta_model`/`meta_oos_proba`). Los DataFrames que devuelve
`meta_labels` usan los mismos nombres de columna que
`ml.labeling.triple_barrier_labels` ("label", "dias_hasta_evento", etc.)
a propósito, para poder alimentarse directo a
`ml.features.align_features_labels` sin ningún adaptador.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from ml.labeling import triple_barrier_labels
from ml.validation import PurgedKFold
from signals.engine import compute_signal_components

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS: int = 5
DEFAULT_EMBARGO_PCT: float = 0.01
DEFAULT_THRESHOLD: float = 0.5

_DEFAULT_META_PARAMS: dict = {
    "max_depth": 3,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def primary_side(df: pd.DataFrame) -> pd.Series:
    """Lado {-1, 0, +1} de una regla PRIMARIA simple y transparente (no ML):
    el SIGNO del componente de TENDENCIA del engine de señales
    (`signals.engine.compute_signal_components`, columna "trend" — combina
    el gap EMA rápida/lenta y el histograma de MACD, ver ese módulo).

    Se eligió esta regla, en vez de armar una nueva desde cero (p. ej. un
    cruce de SMA a mano), porque ya está construida, testeada y documentada
    desde Fase 1e — reutilizarla evita tener dos definiciones de "tendencia"
    distintas conviviendo en el proyecto. side=+1 si trend>0 (tendencia
    alcista, señal de compra), side=-1 si trend<0 (señal de venta), side=0
    si trend es exactamente 0. Donde `trend` es NaN (warmup de los
    indicadores) `side` también queda en NaN — no se inventa una señal sin
    información suficiente.

    Esta es la señal PRIMARIA del esquema de meta-labeling: decide la
    DIRECCIÓN, no el timing ni el tamaño — eso lo deciden
    `train_meta_model`/`meta_positions`, que filtran y dimensionan esta
    señal, no la reemplazan.
    """
    trend = compute_signal_components(df)["trend"]

    side = pd.Series(np.nan, index=trend.index, dtype=float)
    side[trend > 0.0] = 1.0
    side[trend < 0.0] = -1.0
    side[trend == 0.0] = 0.0
    return side.rename("side")


def meta_labels(
    close: pd.Series,
    volatility: pd.Series,
    side: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 10,
) -> pd.DataFrame:
    """Etiqueta BINARIA de meta-labeling: para cada fecha con `side != 0`
    (un trade que la primaria SÍ habría tomado), aplica triple-barrier
    CONDICIONADO a ese lado y etiqueta 1 si el trade ganó, 0 si no. Fechas
    con `side == 0` (o NaN) no tienen trade que evaluar y quedan sin
    etiqueta (NaN) — igual que `ml.labeling.triple_barrier_labels` con su
    cola sin ventana futura suficiente.

    Truco de reutilización: `triple_barrier_labels` es long-only por
    diseño (TP arriba, SL abajo). Para el lado corto (`side=-1`) se la
    llama sobre el precio INVERTIDO (`1/close`) en vez de reimplementar la
    lógica de detección de barreras: si el precio invertido toca su
    "take-profit" (subió), es porque `close` CAYÓ lo suficiente — la
    condición de ganancia correcta para un short. Por la misma razón, el
    "retorno_realizado" de esa llamada (`(1/close)_evento / (1/close)_entrada - 1`,
    que se simplifica a `close_entrada / close_evento - 1`) YA es el P&L
    real de la posición corta, sin necesidad de volver a transformarlo.

    La MISMA `volatility` (computada sobre `close`, no sobre `1/close`) se
    usa para ambos lados: para retornos diarios chicos, el retorno de
    `1/close` es aproximadamente `-retorno de close` (por Taylor de primer
    orden de `1/(1+r) - 1 ≈ -r`), así que ambas series tienen
    aproximadamente el mismo desvío estándar. Es una aproximación, no una
    igualdad exacta, pero válida para las magnitudes de retorno diario
    típicas de cripto.

    Devuelve un `pd.DataFrame` (mismas columnas que `triple_barrier_labels`,
    a propósito, para poder pasarlo directo a
    `ml.features.align_features_labels` renombrando nada más que no haga
    falta): "label" (1/0/NaN — la etiqueta BINARIA de meta-labeling, no la
    ternaria de la primaria), "barrera_tocada", "dias_hasta_evento",
    "retorno_realizado" (P&L real de la posición, ya con el signo correcto
    para el lado tomado).
    """
    side = side.reindex(close.index)

    long_barriers = triple_barrier_labels(close, volatility, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding)
    short_barriers = triple_barrier_labels(
        1.0 / close, volatility, pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding
    )

    is_long = side == 1.0
    is_short = side == -1.0

    def _win_indicator(barrier_label: pd.Series) -> pd.Series:
        win = (barrier_label == 1.0).astype(float)
        win[barrier_label.isna()] = np.nan
        return win

    label = pd.Series(np.nan, index=close.index, dtype=float)
    barrera = pd.Series(None, index=close.index, dtype=object)
    dias = pd.Series(np.nan, index=close.index, dtype=float)
    retorno = pd.Series(np.nan, index=close.index, dtype=float)

    label[is_long] = _win_indicator(long_barriers.loc[is_long, "label"])
    barrera[is_long] = long_barriers.loc[is_long, "barrera_tocada"]
    dias[is_long] = long_barriers.loc[is_long, "dias_hasta_evento"]
    retorno[is_long] = long_barriers.loc[is_long, "retorno_realizado"]

    label[is_short] = _win_indicator(short_barriers.loc[is_short, "label"])
    barrera[is_short] = short_barriers.loc[is_short, "barrera_tocada"]
    dias[is_short] = short_barriers.loc[is_short, "dias_hasta_evento"]
    retorno[is_short] = short_barriers.loc[is_short, "retorno_realizado"]

    return pd.DataFrame(
        {"label": label, "barrera_tocada": barrera, "dias_hasta_evento": dias, "retorno_realizado": retorno},
        index=close.index,
    )


def train_meta_model(
    X: pd.DataFrame,
    meta_y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_pct: float = DEFAULT_EMBARGO_PCT,
    **params,
) -> dict:
    """Entrena y evalúa el meta-modelo (gradient boosting BINARIO) con
    validación Purged K-Fold (`ml.validation.PurgedKFold`, reutilizada
    directo en vez de `purged_cv_score`: ROC-AUC necesita las
    PROBABILIDADES del modelo, no solo la clase predicha, así que este loop
    pide `predict_proba` una sola vez por fold en vez de reajustar el
    modelo por métrica).

    Reporta precisión, recall, F1 y ROC-AUC por fold, y los compara SIEMPRE
    contra el baseline de "tomar todas las señales primarias sin filtrar"
    (aceptar cada trade que propuso la primaria, sin ningún filtro de
    meta-labeling): su "precisión" es simplemente la tasa de acierto de la
    primaria sola en ese mismo fold de test (`meta_y.mean()` del fold). Si
    el meta-modelo no mejora esa precisión, el filtro no está aportando
    nada — y este resultado se loguea explícito, no se esconde.

    Devuelve un dict: "precision_scores"/"precision_media",
    "recall_scores"/"recall_media", "f1_scores"/"f1_media",
    "roc_auc_scores"/"roc_auc_media" (NaN si algún fold no tuvo ambas
    clases en el test, caso borde con pocas muestras), "baseline_precision_scores"/
    "baseline_precision_media", "mejora_precision" (precision_media -
    baseline_precision_media), "supera_baseline" (bool), "n_splits",
    "n_muestras".
    """
    model_params = {**_DEFAULT_META_PARAMS, **params}
    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []
    roc_auc_scores: list[float] = []
    baseline_precision_scores: list[float] = []

    for fold_i, (train_pos, test_pos) in enumerate(cv.split(X)):
        model = XGBClassifier(**model_params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.iloc[train_pos].to_numpy()
        model.fit(X.iloc[train_pos], meta_y.iloc[train_pos], **fit_kwargs)

        y_test = meta_y.iloc[test_pos]
        y_pred = model.predict(X.iloc[test_pos])
        y_proba = model.predict_proba(X.iloc[test_pos])[:, 1]

        precision_scores.append(float(precision_score(y_test, y_pred, zero_division=0)))
        recall_scores.append(float(recall_score(y_test, y_pred, zero_division=0)))
        f1_scores.append(float(f1_score(y_test, y_pred, zero_division=0)))
        baseline_precision_scores.append(float(y_test.mean()))

        if y_test.nunique() > 1:
            roc_auc_scores.append(float(roc_auc_score(y_test, y_proba)))
        else:
            logger.warning("train_meta_model: fold %d sin ambas clases en test, se omite ROC-AUC de ese fold", fold_i + 1)

        logger.info(
            "train_meta_model: fold %d/%d, precision=%.4f recall=%.4f f1=%.4f (baseline=%.4f)",
            fold_i + 1, n_splits, precision_scores[-1], recall_scores[-1], f1_scores[-1], baseline_precision_scores[-1],
        )

    precision_media = float(np.mean(precision_scores))
    baseline_precision_media = float(np.mean(baseline_precision_scores))

    result = {
        "precision_scores": precision_scores,
        "precision_media": precision_media,
        "recall_scores": recall_scores,
        "recall_media": float(np.mean(recall_scores)),
        "f1_scores": f1_scores,
        "f1_media": float(np.mean(f1_scores)),
        "roc_auc_scores": roc_auc_scores,
        "roc_auc_media": float(np.mean(roc_auc_scores)) if roc_auc_scores else float("nan"),
        "baseline_precision_scores": baseline_precision_scores,
        "baseline_precision_media": baseline_precision_media,
        "mejora_precision": precision_media - baseline_precision_media,
        "supera_baseline": bool(precision_media > baseline_precision_media),
        "n_splits": n_splits,
        "n_muestras": len(meta_y),
    }

    if result["supera_baseline"]:
        logger.info(
            "train_meta_model: precisión purgeada (%.4f) SUPERA el baseline sin filtrar (%.4f)",
            precision_media, baseline_precision_media,
        )
    else:
        logger.warning(
            "train_meta_model: precisión purgeada (%.4f) NO supera el baseline de tomar todas las "
            "señales primarias sin filtrar (%.4f) — el meta-modelo no está aportando.",
            precision_media, baseline_precision_media,
        )

    return result


def meta_oos_proba(
    X: pd.DataFrame,
    meta_y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_pct: float = DEFAULT_EMBARGO_PCT,
    **params,
) -> pd.Series:
    """Probabilidades OUT-OF-SAMPLE del meta-modelo: cada muestra la predice
    un modelo entrenado SIN ella (vía `ml.validation.PurgedKFold`, mismo
    esquema que `ml.models.oos_predictions` para el modelo primario). Son
    las únicas probabilidades honestas para alimentar `meta_positions` de
    cara a un backtest — las de un modelo ajustado sobre todo el dataset
    estarían mirando el futuro con el propio modelo que lo memorizó.

    Devuelve una `pd.Series` de probabilidades (clase 1 = trade ganador) en
    [0, 1], indexada igual que `X`, una por fila (cada muestra es test en
    exactamente un fold de `PurgedKFold`).
    """
    model_params = {**_DEFAULT_META_PARAMS, **params}
    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    proba = pd.Series(np.nan, index=X.index, dtype=float)
    for fold_i, (train_pos, test_pos) in enumerate(cv.split(X)):
        model = XGBClassifier(**model_params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.iloc[train_pos].to_numpy()
        model.fit(X.iloc[train_pos], meta_y.iloc[train_pos], **fit_kwargs)

        proba.iloc[test_pos] = model.predict_proba(X.iloc[test_pos])[:, 1]
        logger.info("meta_oos_proba: fold %d/%d, %d probabilidades", fold_i + 1, n_splits, len(test_pos))

    return proba


def meta_positions(
    side: pd.Series, meta_proba: pd.Series, threshold: float = DEFAULT_THRESHOLD, size_by_proba: bool = True
) -> pd.Series:
    """Posición final tras aplicar el filtro + sizing de meta-labeling:

        posicion_t = side_t * (1 si meta_proba_t > threshold, si no 0) * tamaño_t

    con `tamaño_t = meta_proba_t` si `size_by_proba=True` (apostar más
    cuando la confianza del meta-modelo es más alta — el sizing propio del
    meta-labeling, AFML cap. 3), o `tamaño_t = 1.0` si `size_by_proba=False`
    (filtro binario puro: tamaño completo o nada, sin gradación por
    confianza).

    `side` y `meta_proba` se alinean por índice; donde falte alguno de los
    dos (p. ej. `side == 0`, sin trade primario que evaluar, o warmup del
    meta-modelo) la posición queda en 0 — nunca se opera sin ambas señales
    presentes.

    NO aplica ningún shift — igual que `signals.engine.generate_positions` y
    `ml.models.model_to_positions`, es responsabilidad de
    `backtest.engine.run_backtest` desplazar la posición un día antes de
    aplicarla a un retorno.
    """
    aligned = pd.concat([side.rename("side"), meta_proba.rename("proba")], axis=1)
    aligned["side"] = aligned["side"].fillna(0.0)
    aligned["proba"] = aligned["proba"].fillna(0.0)

    passes_filter = (aligned["proba"] > threshold).astype(float)
    size = aligned["proba"] if size_by_proba else 1.0

    positions = aligned["side"] * passes_filter * size
    return positions.rename("position")
