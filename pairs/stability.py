"""Estabilidad ROLLING de la cointegración: ¿el par sigue cointegrado a lo
largo del tiempo, o fue cointegración de una sola foto?

`pairs.cointegration.find_cointegrated_pairs` testea cointegración sobre
TODA la historia común de una vez: un par puede dar un p-valor bajo
in-sample sin que eso garantice que la relación se sostenga en
sub-períodos — la cointegración pudo haber existido solo durante un tramo
puntual (p. ej. un régimen de mercado particular) y no ser una propiedad
estructural del par. Ese tipo de cointegración "de una sola ventana" no es
operable: no hay garantía de que siga sosteniéndose hacia adelante.

Este módulo re-testea Engle-Granger en ventanas deslizantes y resume qué
tan CONSISTENTE es la cointegración en el tiempo — un filtro de operabilidad
más honesto que el p-valor único. Reutiliza `pairs.cointegration.engle_granger`
tal cual, no reimplementa el test.
"""

from __future__ import annotations

import itertools
import logging

import pandas as pd

from config import UNIVERSE
from data.loaders import get_prices
from pairs.cointegration import DEFAULT_PVALUE_THRESHOLD, engle_granger

logger = logging.getLogger(__name__)

DEFAULT_STABLE_FRACTION_THRESHOLD: float = 0.6


def rolling_cointegration(
    y: pd.Series, x: pd.Series, window: int = 365, step: int = 30, pvalue: float = DEFAULT_PVALUE_THRESHOLD
) -> pd.DataFrame:
    """Corre `engle_granger` en ventanas deslizantes de `window`
    observaciones, avanzando de a `step` observaciones entre ventana y
    ventana (no se re-testea cada día: sería costoso y ventanas casi
    totalmente solapadas no agregan información nueva).

    `y`/`x` se alinean por índice ANTES de armar las ventanas, para que
    "ventana de `window` observaciones" sea la misma cantidad de días para
    ambas series.

    Devuelve un `pd.DataFrame`, una fila por ventana, columnas: "fecha_fin"
    (último timestamp de la ventana), "p_valor_adf", "beta", "es_cointegrado".
    """
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(aligned)
    if n < window:
        raise ValueError(
            f"rolling_cointegration: se necesitan al menos window={window} observaciones alineadas, hay {n}"
        )

    rows: list[dict] = []
    for start_idx in range(0, n - window + 1, step):
        window_slice = aligned.iloc[start_idx : start_idx + window]
        result = engle_granger(window_slice["y"], window_slice["x"], pvalue_threshold=pvalue)
        rows.append(
            {
                "fecha_fin": window_slice.index[-1],
                "p_valor_adf": result["p_valor_adf"],
                "beta": result["beta"],
                "es_cointegrado": result["es_cointegrado"],
            }
        )

    return pd.DataFrame(rows)


def stability_summary(
    rolling_df: pd.DataFrame, stable_fraction_threshold: float = DEFAULT_STABLE_FRACTION_THRESHOLD
) -> dict:
    """Resume la salida de `rolling_cointegration` en un veredicto de
    operabilidad.

    Devuelve un dict:
    - "n_ventanas": cantidad de ventanas testeadas.
    - "fraccion_cointegrada": proporción de ventanas con `es_cointegrado=True`
      (en [0, 1]) — el indicador central de este módulo.
    - "beta_medio", "beta_std": media y desvío del hedge ratio ESTÁTICO
      estimado en cada ventana. Un `beta_std` alto es una señal de que el
      hedge ratio salta de ventana a ventana, incluso si el par está
      cointegrado la mayor parte del tiempo — indicio de que conviene un
      hedge ratio DINÁMICO (ver `pairs.kalman_hedge`) en vez de asumirlo
      constante.
    - "estable": bool, True si `fraccion_cointegrada >= stable_fraction_threshold`.
      Un par cointegrado in-sample pero estable en pocas ventanas (fracción
      baja) NO es operable: no hay evidencia de que la relación se sostenga
      de forma consistente, más allá de lo que diga el test único sobre toda
      la muestra.
    """
    if rolling_df.empty:
        raise ValueError("stability_summary: 'rolling_df' está vacío")

    fraccion = float(rolling_df["es_cointegrado"].mean())
    beta_std = float(rolling_df["beta"].std(ddof=1)) if len(rolling_df) > 1 else 0.0

    return {
        "n_ventanas": len(rolling_df),
        "fraccion_cointegrada": fraccion,
        "beta_medio": float(rolling_df["beta"].mean()),
        "beta_std": beta_std,
        "estable": bool(fraccion >= stable_fraction_threshold),
    }


def screen_pairs_stability(
    assets: list[str] | None = None,
    source: str = "store",
    window: int = 365,
    step: int = 30,
    pvalue: float = DEFAULT_PVALUE_THRESHOLD,
    stable_fraction_threshold: float = DEFAULT_STABLE_FRACTION_THRESHOLD,
) -> pd.DataFrame:
    """Corre `rolling_cointegration` + `stability_summary` para todas las
    combinaciones de a pares de `assets` (default: `config.UNIVERSE`), en
    ambas direcciones (y~x y x~y), y se queda con la dirección de mayor
    "fraccion_cointegrada" para cada par.

    Pares sin historia suficiente para ni siquiera una ventana (`window`
    observaciones) se omiten con un warning, en vez de romper el screening
    completo.

    Devuelve un `pd.DataFrame` ordenado por "fraccion_cointegrada"
    DESCENDENTE — el ranking de operabilidad real (ver docstring del
    módulo), más honesto que ordenar por el p-valor in-sample de
    `pairs.cointegration.find_cointegrated_pairs`. Columnas: "par",
    "direccion", "n_ventanas", "fraccion_cointegrada", "beta_medio",
    "beta_std", "estable".
    """
    if assets is None:
        assets = list(UNIVERSE)
    if len(assets) < 2:
        raise ValueError("screen_pairs_stability: se necesitan al menos 2 activos")

    prices = {asset: get_prices(asset, source=source)["close"] for asset in assets}

    rows: list[dict] = []
    for asset_a, asset_b in itertools.combinations(assets, 2):
        try:
            rolling_ab = rolling_cointegration(prices[asset_a], prices[asset_b], window=window, step=step, pvalue=pvalue)
            rolling_ba = rolling_cointegration(prices[asset_b], prices[asset_a], window=window, step=step, pvalue=pvalue)
        except ValueError as exc:
            logger.warning("screen_pairs_stability: %s-%s omitido (historia insuficiente): %s", asset_a, asset_b, exc)
            continue

        summary_ab = stability_summary(rolling_ab, stable_fraction_threshold)
        summary_ba = stability_summary(rolling_ba, stable_fraction_threshold)

        if summary_ab["fraccion_cointegrada"] >= summary_ba["fraccion_cointegrada"]:
            best, direction = summary_ab, f"{asset_a}~{asset_b}"
        else:
            best, direction = summary_ba, f"{asset_b}~{asset_a}"

        rows.append({"par": f"{asset_a}-{asset_b}", "direccion": direction, **best})

    if not rows:
        raise RuntimeError("screen_pairs_stability: ningún par tuvo historia suficiente para evaluar estabilidad")

    table = pd.DataFrame(rows).sort_values("fraccion_cointegrada", ascending=False).reset_index(drop=True)
    logger.info(
        "screen_pairs_stability: %d/%d pares estables (fraccion_cointegrada>=%.2f)",
        int(table["estable"].sum()), len(table), stable_fraction_threshold,
    )
    return table
