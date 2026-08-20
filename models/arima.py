"""ARIMA como baseline direccional honesto sobre retornos de cripto.

ENCUADRE (léase antes de usar este módulo): ARIMA ajustado sobre PRECIOS de
cripto captura muy poco — los precios son casi un random walk (ver el test
ADF de `eda/exploracion.ipynb`: el precio no es estacionario, el retorno sí).
Por eso acá se modela sobre RETORNOS, es decir un ARMA(p,q) disfrazado de
ARIMA con `d=0` (el parámetro de diferenciación de ARIMA), no un ARIMA sobre
precios.

Este módulo NO es un predictor principal del proyecto ni pretende serlo: es
un BASELINE. Su único criterio de éxito es el acierto direccional
walk-forward (`walk_forward_directional_accuracy`) frente al 50% de tirar
una moneda. Un accuracy cercano a 50% (con un test binomial que NO rechaza
H0: accuracy=0.5) es el resultado ESPERADO y HONESTO para un baseline lineal
sobre retornos de un activo ~eficiente — no un fracaso del módulo. Si algún
día un baseline de este tipo mostrara un accuracy consistentemente alto,
sería motivo de sospecha (data leakage, look-ahead bias) antes que de
festejo.

Convención del proyecto: `returns` es una serie de retornos en escala
DECIMAL (0.01 == 1%), frecuencia diaria, consistente con
`signals.returns.log_returns`.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

logger = logging.getLogger(__name__)

_ALLOWED_CRITERIA: tuple[str, ...] = ("aic", "bic")


def _fit_arima_order(returns: pd.Series, order: tuple[int, int, int]) -> ARIMAResultsWrapper:
    """Ajusta ARIMA(p,0,q) silenciando warnings de convergencia (se
    manejan como una excepción/descarte más arriba, no como ruido de log).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(returns, order=order).fit()


def select_arima_order(
    returns: pd.Series, max_p: int = 5, max_q: int = 5, criterion: str = "aic"
) -> tuple[int, int, int]:
    """Busca en la grilla (p, 0, q) para p en [0, max_p] y q en [0, max_q]
    (d=0 fijo: `returns` ya son estacionarios, no hace falta diferenciar) el
    order que minimiza `criterion` ("aic" o "bic").

    Los órdenes que no convergen (excepción de statsmodels durante el
    ajuste) se descartan silenciosamente de la comparación, no rompen la
    búsqueda completa.

    Devuelve el mejor order como tupla (p, 0, q).
    """
    if criterion not in _ALLOWED_CRITERIA:
        raise ValueError(f"criterion debe ser uno de {_ALLOWED_CRITERIA}, recibido '{criterion}'")

    clean_returns = returns.dropna()
    if clean_returns.empty:
        raise ValueError("select_arima_order: 'returns' no tiene observaciones válidas")

    candidates: list[dict] = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            order = (p, 0, q)
            try:
                result = _fit_arima_order(clean_returns, order)
            # F-03 (auditoría, Fase 14): acotado a los modos de falla
            # numérica documentados de la optimización MLE de statsmodels/
            # scipy (parámetros inválidos, matriz singular, no convergencia)
            # — no un `Exception` amplio que también trague bugs reales.
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                logger.debug("select_arima_order: order=%s no convergió: %s", order, exc)
                continue
            candidates.append({"order": order, "aic": float(result.aic), "bic": float(result.bic)})

    if not candidates:
        raise RuntimeError("select_arima_order: ningún order de la grilla convergió")

    best = min(candidates, key=lambda c: c[criterion])
    logger.info(
        "select_arima_order: mejor order = %s (%s=%.2f, de %d/%d órdenes convergidos)",
        best["order"], criterion, best[criterion], len(candidates), (max_p + 1) * (max_q + 1),
    )
    return best["order"]


def fit_arima(returns: pd.Series, order: tuple[int, int, int] | None = None) -> ARIMAResultsWrapper:
    """Ajusta un ARMA(p,q) (ARIMA con d=0) sobre `returns`.

    Si `order` es None, se determina automáticamente vía
    `select_arima_order` (grilla por AIC) antes de ajustar.
    """
    clean_returns = returns.dropna()
    if order is None:
        order = select_arima_order(clean_returns)
    return _fit_arima_order(clean_returns, order)


def forecast_returns(result: ARIMAResultsWrapper, horizon: int = 1, alpha: float = 0.05) -> pd.DataFrame:
    """Pronóstico de retorno (escala decimal) a `horizon` pasos, con
    intervalo de confianza al `1 - alpha` (95% por defecto).

    Devuelve un `pd.DataFrame` de `horizon` filas (índice "horizonte",
    1..horizon — pasos, no fechas de calendario) con columnas ["media",
    "lower", "upper"].
    """
    forecast_obj = result.get_forecast(steps=horizon)
    conf_int = forecast_obj.conf_int(alpha=alpha)

    return pd.DataFrame(
        {
            "media": forecast_obj.predicted_mean.to_numpy(),
            "lower": conf_int.iloc[:, 0].to_numpy(),
            "upper": conf_int.iloc[:, 1].to_numpy(),
        },
        index=pd.RangeIndex(1, horizon + 1, name="horizonte"),
    )


def directional_forecast(result: ARIMAResultsWrapper, horizon: int = 1) -> dict:
    """Signo esperado del retorno a `horizon` pasos, con una medida de
    convicción.

    La convicción es |media pronosticada| / desvío del pronóstico (el error
    estándar de la media pronosticada, `se_mean` de statsmodels): cuanto más
    grande, más se aleja el pronóstico puntual de 0 en unidades de su propia
    incertidumbre. NO es una probabilidad ni un p-valor — es una medida
    relativa, útil sobre todo para comparar la confianza entre pronósticos,
    no como umbral absoluto de "operar o no".

    Devuelve un dict: "horizonte", "signo" (+1 o -1), "media", "desvio",
    "conviccion".
    """
    forecast_obj = result.get_forecast(steps=horizon)
    mean = float(forecast_obj.predicted_mean.iloc[-1])
    se = float(forecast_obj.se_mean.iloc[-1])

    signo = 1 if mean >= 0 else -1
    conviccion = abs(mean) / se if se > 0 else float("inf")

    return {"horizonte": horizon, "signo": signo, "media": mean, "desvio": se, "conviccion": conviccion}


def walk_forward_directional_accuracy(
    returns: pd.Series,
    order: tuple[int, int, int] | None = None,
    train_window: int = 750,
    step: int = 1,
    refit_every: int = 30,
) -> dict:
    """Backtest walk-forward del acierto DIRECCIONAL de un ARMA(p,q): el
    entregable clave de este módulo (ver encuadre en el docstring del
    archivo).

    Para cada punto de evaluación i (con `train_window <= i < n`, avanzando
    de a `step`): se entrena/actualiza el modelo con la ventana DESLIZANTE
    `returns[i-train_window:i]` (tamaño fijo, no expandible — refleja un
    régimen reciente, no todo el histórico) y se pronostica el signo de
    `returns[i]` a 1 paso. Se compara contra el signo real observado.

    Por eficiencia, el modelo se REAJUSTA por completo (nueva optimización
    de parámetros) solo cada `refit_every` puntos de evaluación; en los
    pasos intermedios se reutilizan los parámetros ya estimados sobre la
    nueva ventana de datos vía `ARIMAResultsWrapper.apply(..., refit=False)`
    (mismo order, mismos parámetros, sin volver a optimizar) — el order en
    sí NO se vuelve a buscar en cada refit, solo se determina una vez al
    principio (con `select_arima_order` si `order` es None) para no pagar el
    costo de una grilla completa en cada reajuste.

    Devuelve un dict:
    - "accuracy": proporción de aciertos direccionales.
    - "n_predicciones": cantidad de pasos evaluados (pasos que no
      convergieron se descartan y no cuentan).
    - "accuracy_baseline": 0.5 (tirar una moneda).
    - "p_valor": test binomial de dos colas (H0: accuracy=0.5, ver
      `scipy.stats.binomtest`) — si NO rechaza H0 (p_valor alto), el
      accuracy observado es consistente con el azar, el resultado ESPERADO
      para este baseline (ver encuadre del módulo).
    - "rechaza_h0": bool, True si p_valor < 0.05.
    - "order": el order efectivamente usado.
    """
    clean_returns = returns.dropna()
    n = len(clean_returns)
    if n <= train_window:
        raise ValueError(
            f"walk_forward_directional_accuracy: se necesitan más de train_window={train_window} "
            f"observaciones válidas, hay {n}"
        )

    if order is None:
        order = select_arima_order(clean_returns.iloc[:train_window])
        logger.info("walk_forward_directional_accuracy: order seleccionado automáticamente = %s", order)

    correct = 0
    total = 0
    current_result: ARIMAResultsWrapper | None = None
    steps_since_refit = 0

    for i in range(train_window, n, step):
        window = clean_returns.iloc[i - train_window : i]
        need_refit = current_result is None or steps_since_refit >= refit_every

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if need_refit:
                    current_result = _fit_arima_order(window, order)
                    steps_since_refit = 0
                else:
                    current_result = current_result.apply(window, refit=False)
                steps_since_refit += 1

                predicted_mean = float(current_result.get_forecast(steps=1).predicted_mean.iloc[-1])
        # F-03 (auditoría, Fase 14): mismos modos de falla numérica acotados
        # que en `select_arima_order` — ver su comentario.
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            logger.warning("walk_forward_directional_accuracy: falló el paso i=%d: %s", i, exc)
            current_result = None  # fuerza un refit limpio en el próximo paso
            continue

        predicted_sign = 1 if predicted_mean >= 0 else -1
        actual_sign = 1 if clean_returns.iloc[i] >= 0 else -1

        total += 1
        if predicted_sign == actual_sign:
            correct += 1

    if total == 0:
        raise RuntimeError("walk_forward_directional_accuracy: ningún paso pudo evaluarse")

    accuracy = correct / total
    binom_result = scipy_stats.binomtest(correct, total, p=0.5, alternative="two-sided")
    p_value = float(binom_result.pvalue)

    logger.info(
        "walk_forward_directional_accuracy: accuracy=%.3f sobre %d predicciones (order=%s, p_valor=%.4f)",
        accuracy, total, order, p_value,
    )

    return {
        "accuracy": accuracy,
        "n_predicciones": total,
        "accuracy_baseline": 0.5,
        "p_valor": p_value,
        "rechaza_h0": bool(p_value < 0.05),
        "order": order,
    }
