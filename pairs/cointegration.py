"""Detección de pares cointegrados: base del stat-arb.

Convención de este módulo: a diferencia del resto del proyecto (que trabaja
sobre RETORNOS), la cointegración se testea sobre el NIVEL de PRECIO, en
escala LOGARÍTMICA (np.log del close) — la convención estándar en la
literatura de cointegración/stat-arb: si dos log-precios están cointegrados,
existe una combinación lineal de ellos (el spread) que es estacionaria,
aunque cada log-precio individualmente sea un random walk (I(1)). Las
funciones de este módulo reciben precios RAW (no log-transformados) y
aplican `np.log()` internamente, así que no hay que pre-procesar nada antes
de llamarlas.

Fase 2a: solo DETECCIÓN y caracterización de candidatos (Engle-Granger,
Johansen, hedge ratio estático, half-life). NO se implementa acá la
estrategia de trading sobre el spread ni su backtest — eso es Fase 2b (que
también agrega el hedge ratio DINÁMICO vía filtro de Kalman).

Convención del proyecto: `config.PERIODS_PER_YEAR` para todo lo que se
anualice, `data.loaders.get_prices(source="store")` como fuente de datos por
defecto (reproducible offline vía `scripts/export_snapshot.py`).
"""

from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import UNIVERSE
from data.loaders import get_prices

logger = logging.getLogger(__name__)

DEFAULT_PVALUE_THRESHOLD: float = 0.05


def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
    """Hedge ratio ESTÁTICO: beta de la regresión OLS de log(y) sobre log(x)
    con constante (log(y) = alpha + beta*log(x) + spread).

    Es el ratio de cobertura de un par estático: por cada unidad (en valor)
    larga de `y`, se short-ea `beta` unidades de `x`. El hedge ratio
    DINÁMICO (filtro de Kalman, que deja que `beta` varíe en el tiempo) es
    Fase 2b.
    """
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    log_y = np.log(aligned["y"])
    log_x = np.log(aligned["x"])
    model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    return float(model.params["x"])


def engle_granger(y: pd.Series, x: pd.Series, pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD) -> dict:
    """Test de cointegración de Engle-Granger de 2 pasos, sobre log-precios.

    Paso 1: regresión OLS log(y) = alpha + beta*log(x) + spread (con
    constante) — `beta` es el hedge ratio, `spread` son los residuos.
    Paso 2: test ADF (Augmented Dickey-Fuller) sobre `spread`. Si se
    rechaza la hipótesis nula de raíz unitaria (p-valor < `pvalue_threshold`),
    el spread es estacionario y el par se considera cointegrado.

    `y` y `x` se alinean por índice (inner join) antes de testear.

    Devuelve un dict: "beta" (hedge ratio), "alpha" (intercepto), "spread"
    (`pd.Series`, residuos de la regresión, indexado igual que la muestra
    alineada), "estadistico_adf", "p_valor_adf", "es_cointegrado" (bool).
    """
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    log_y = np.log(aligned["y"])
    log_x = np.log(aligned["x"])

    model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    beta = float(model.params["x"])
    alpha = float(model.params["const"])
    spread = model.resid.rename("spread")

    adf_stat, adf_pvalue, *_ = adfuller(spread, autolag="AIC")

    return {
        "beta": beta,
        "alpha": alpha,
        "spread": spread,
        "estadistico_adf": float(adf_stat),
        "p_valor_adf": float(adf_pvalue),
        "es_cointegrado": bool(adf_pvalue < pvalue_threshold),
    }


def johansen_test(df_prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Test de cointegración de Johansen (estadístico de traza) sobre 2+
    series de precios (columnas de `df_prices`, RAW — se loguean internamente).

    A diferencia de Engle-Granger (que testea UN par a la vez y depende de
    elegir cuál serie es "y" y cuál "x"), Johansen testea todas las
    combinaciones lineales posibles entre las columnas de `df_prices` a la
    vez y no depende de esa elección — más apropiado con 3+ activos.

    `det_order`: orden determinístico del VECM subyacente (0 = sin
    tendencia). `k_ar_diff`: rezagos en diferencias. Ver
    `statsmodels.tsa.vector_ar.vecm.coint_johansen`.

    El número de relaciones de cointegración se determina con la secuencia
    estándar de tests de traza: se parte de r=0 (H0: rango <= 0) y se van
    aceptando relaciones mientras el estadístico de traza supere su valor
    crítico al 95%, deteniéndose en el primer r donde NO se rechaza H0 (no
    alcanza con contar cuántos estadísticos superan su crítico de forma
    aislada, hay que parar en el primer no-rechazo).

    Devuelve un dict: "n_relaciones_cointegracion" (int), "estadisticos_traza"
    (array), "valores_criticos_95" (array, alineado con "estadisticos_traza"),
    "vectores_cointegracion" (matriz de autovectores; cada columna es una
    combinación lineal candidata, ordenadas por autovalor descendente — la
    primera columna es el vector de cointegración más fuerte).
    """
    if df_prices.shape[1] < 2:
        raise ValueError("johansen_test: 'df_prices' necesita al menos 2 columnas")

    log_prices = np.log(df_prices.dropna())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = coint_johansen(log_prices, det_order, k_ar_diff)

    trace_stats = result.lr1
    critical_values_95 = result.cvt[:, 1]  # columnas de cvt: [90%, 95%, 99%]

    n_relations = 0
    for stat, crit in zip(trace_stats, critical_values_95):
        if stat <= crit:
            break
        n_relations += 1

    return {
        "n_relaciones_cointegracion": n_relations,
        "estadisticos_traza": trace_stats,
        "valores_criticos_95": critical_values_95,
        "vectores_cointegracion": result.evec,
    }


def half_life(spread: pd.Series) -> float:
    """Vida media de reversión a la media del `spread` (p. ej. el residuo de
    `engle_granger`), asumiendo un proceso de Ornstein-Uhlenbeck
    discretizado como AR(1):

        Δspread_t = spread_t - spread_{t-1} = c + theta * spread_{t-1} + ruido

    `theta` se estima por OLS (con constante `c`, para no forzar al spread a
    tener media exactamente 0). Si `theta < 0` (reversión genuina):

        half_life = -ln(2) / theta

    Interpretación: la cantidad de PERÍODOS (en la frecuencia de `spread`,
    normalmente días) que tarda una desviación del spread respecto de su
    media en reducirse a la mitad bajo reversión a la media. Si `theta >= 0`
    (el spread no revierte: es un random walk o diverge), `half_life` no
    tiene una interpretación válida — se devuelve `inf` en vez de un número
    negativo sin sentido, y se loguea un warning.
    """
    spread = spread.dropna()
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    aligned = pd.concat([spread_diff.rename("diff"), spread_lag.rename("lag")], axis=1).dropna()

    model = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit()
    theta = float(model.params["lag"])

    if theta >= 0:
        logger.warning(
            "half_life: theta=%.4f >= 0, el spread no muestra reversión a la media (no converge)", theta
        )
        return float("inf")

    return -np.log(2.0) / theta


def find_cointegrated_pairs(
    assets: list[str] | None = None,
    source: str = "store",
    start: str | None = None,
    end: str | None = None,
    pvalue: float = DEFAULT_PVALUE_THRESHOLD,
) -> pd.DataFrame:
    """Corre Engle-Granger sobre todas las combinaciones de a pares de
    `assets` (default: todo `config.UNIVERSE`), en AMBAS direcciones (y~x y
    x~y — el hedge ratio y los residuos no son simétricos, así que hay que
    elegir una dirección) y se queda con la de menor p-valor ADF para cada
    par.

    Cada par se alinea por fechas comunes ANTES de testear (dentro de
    `engle_granger`): importante porque SOL tiene mucha menos historia que
    el resto del universo (arranca en 2020 vs. 2018 de BTC/ETH/BNB/LTC en el
    snapshot de `source="store"`), así que la muestra común de cualquier par
    que incluya SOL arranca ahí, no en 2018.

    Devuelve un `pd.DataFrame` ordenado por "p_valor_adf" ascendente (los
    candidatos más prometedores primero), con una fila por PAR (no por
    dirección: ya eligió la mejor), columnas: "par" ("A-B"), "direccion"
    ("A~B" o "B~A", donde la primera letra es la variable dependiente "y"),
    "beta", "p_valor_adf", "half_life", "es_cointegrado".
    """
    if assets is None:
        assets = list(UNIVERSE)
    if len(assets) < 2:
        raise ValueError("find_cointegrated_pairs: se necesitan al menos 2 activos")

    prices = {asset: get_prices(asset, source=source, start=start, end=end)["close"] for asset in assets}

    rows: list[dict] = []
    for asset_a, asset_b in itertools.combinations(assets, 2):
        result_ab = engle_granger(prices[asset_a], prices[asset_b], pvalue_threshold=pvalue)
        result_ba = engle_granger(prices[asset_b], prices[asset_a], pvalue_threshold=pvalue)

        if result_ab["p_valor_adf"] <= result_ba["p_valor_adf"]:
            best, direction = result_ab, f"{asset_a}~{asset_b}"
        else:
            best, direction = result_ba, f"{asset_b}~{asset_a}"

        rows.append(
            {
                "par": f"{asset_a}-{asset_b}",
                "direccion": direction,
                "beta": best["beta"],
                "p_valor_adf": best["p_valor_adf"],
                "half_life": half_life(best["spread"]),
                "es_cointegrado": best["es_cointegrado"],
            }
        )

    table = pd.DataFrame(rows).sort_values("p_valor_adf").reset_index(drop=True)
    logger.info(
        "find_cointegrated_pairs: %d/%d pares cointegrados (p<%.2f)",
        int(table["es_cointegrado"].sum()), len(table), pvalue,
    )
    return table
