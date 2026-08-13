"""Modelado de volatilidad condicional (GARCH/EGARCH/GJR) y régimen de riesgo.

Convención del proyecto: `returns` es siempre una serie de retornos en escala
DECIMAL (0.01 == 1%), frecuencia diaria, consistente con
`signals.returns.log_returns`. La anualización usa `config.PERIODS_PER_YEAR`
(252) salvo que se indique lo contrario.

Convención de estabilidad numérica (siguiendo el estilo del repo `momentum`
del autor, `modules/garch_volatility.py`, adaptado de mensual a diario):
`arch` tiene problemas de convergencia numérica cuando el optimizador recibe
valores muy chicos (retornos diarios en escala decimal, ~1e-2). Por eso
`fit_garch_variant` reescala los retornos a PORCENTAJE (x `RETURN_SCALE`=100)
antes de ajustar. El `ARCHModelResult` que devuelve queda, por lo tanto, en
esa escala porcentual internamente (`res.params`, `res.conditional_volatility`,
`res.forecast(...)` crudos NO están en escala decimal). El resto de las
funciones de este módulo (`conditional_volatility`, `forecast_volatility`,
`garch_var`) se encargan de reconvertir a decimal (y anualizar donde
corresponde) al reportar — no usar los atributos crudos del resultado fuera
de este módulo sin volver a escalar.

No se implementa selección de features ni backtesting acá: esto es
puramente el modelo de volatilidad y su validación de VaR (Kupiec). El
régimen de volatilidad (`volatility_regime`) es el insumo para vol targeting
(sizing por riesgo) en fases posteriores, no un modelo de trading en sí.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from scipy import stats as scipy_stats

from config import PERIODS_PER_YEAR

logger = logging.getLogger(__name__)

# Factor de reescalado retornos decimal -> porcentaje para el ajuste numérico
# de `arch` (ver docstring del módulo). Se divide por este mismo factor al
# reconvertir a decimal.
RETURN_SCALE: float = 100.0

# Nivel de significancia FIJO del test de Kupiec (no confundir con `alpha`,
# que es la probabilidad de cola del VaR que se está validando).
KUPIEC_SIGNIFICANCE: float = 0.05

_ALLOWED_DIST: tuple[str, ...] = ("t", "normal")

# Mapeo de la nomenclatura pública del proyecto a la de `arch`. GJR-GARCH no
# es un modelo de volatilidad distinto en `arch`: es GARCH con el término
# asimétrico `o=1` habilitado (capta el efecto apalancamiento: una caída
# dispara más volatilidad futura que una suba de igual magnitud).
_VOL_VARIANTS: dict[str, str] = {
    "Garch": "GARCH",
    "EGarch": "EGARCH",
    "GJR": "GARCH",
}

_GRID: list[tuple[str, str]] = [(vol, dist) for vol in _VOL_VARIANTS for dist in _ALLOWED_DIST]


def _resolve_vol_and_o(vol: str, o: int) -> tuple[str, int]:
    if vol not in _VOL_VARIANTS:
        raise ValueError(f"vol debe ser uno de {list(_VOL_VARIANTS)}, recibido '{vol}'")
    arch_vol = _VOL_VARIANTS[vol]
    if vol == "GJR" and o == 0:
        o = 1
    return arch_vol, o


def _validate_dist(dist: str) -> str:
    if dist not in _ALLOWED_DIST:
        raise ValueError(f"dist debe ser uno de {_ALLOWED_DIST}, recibido '{dist}'")
    return dist


# --------------------------------------------------------------------------
# Ajuste de modelos
# --------------------------------------------------------------------------


def fit_garch_variant(
    returns: pd.Series,
    vol: str = "Garch",
    dist: str = "t",
    p: int = 1,
    q: int = 1,
    o: int = 0,
) -> ARCHModelResult:
    """Ajusta un modelo de volatilidad condicional sobre `returns` (decimal).

    vol:
        "Garch" (GARCH simétrico), "EGarch" (asimetría vía log-varianza) o
        "GJR" (GARCH con término asimétrico; en `arch` es `vol="GARCH"` con
        `o=1`, ver `_VOL_VARIANTS`).
    dist:
        "t" (t de Student estandarizada — recomendada para cripto por las
        colas gordas encontradas en `eda/exploracion.ipynb`) o "normal".
    p, q:
        Órdenes ARCH/GARCH habituales (1,1 por defecto).
    o:
        Orden del término asimétrico. Si `vol="GJR"` y no se pasa `o`
        explícitamente, se fuerza a 1 (GJR sin término asimétrico no sería
        GJR). Para "Garch"/"EGarch" queda en 0 salvo que se pida lo contrario.

    Internamente reescala `returns` a porcentaje antes de ajustar (ver
    RETURN_SCALE en el docstring del módulo) y usa `mean="Constant"` (media
    condicional constante, no un modelo de retorno esperado).
    """
    dist = _validate_dist(dist)
    arch_vol, o = _resolve_vol_and_o(vol, o)

    clean_returns = returns.dropna()
    if clean_returns.empty:
        raise ValueError("fit_garch_variant: 'returns' no tiene observaciones válidas")

    scaled = clean_returns * RETURN_SCALE
    model = arch_model(scaled, mean="Constant", vol=arch_vol, p=p, o=o, q=q, dist=dist)
    return model.fit(disp="off")


def select_best_model(returns: pd.Series, criterion: str = "aic") -> dict:
    """Ajusta la grilla {Garch, EGarch, GJR} x {normal, t} (6 modelos) sobre
    `returns` y devuelve el mejor según `criterion` ("aic" o "bic", menor es
    mejor).

    Los modelos que no convergen (excepción de `arch` durante el ajuste) se
    loguean como warning y se excluyen de la comparación, en vez de romper
    toda la selección.

    Devuelve un dict: "vol", "dist", "aic", "bic", "params" (`pd.Series` de
    parámetros ajustados, en la escala porcentual interna — ver
    `fit_garch_variant`), "result" (el `ARCHModelResult` ajustado, para pasar
    directo a `conditional_volatility`/`forecast_volatility`/`garch_var`).
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion debe ser 'aic' o 'bic', recibido '{criterion}'")

    candidates: list[dict] = []
    for vol, dist in _GRID:
        try:
            result = fit_garch_variant(returns, vol=vol, dist=dist)
        except Exception as exc:  # noqa: BLE001 - un modelo que no converge no debe tumbar la selección
            logger.warning("select_best_model: '%s'/'%s' no convergió: %s", vol, dist, exc)
            continue
        candidates.append(
            {
                "vol": vol,
                "dist": dist,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "params": result.params,
                "result": result,
            }
        )

    if not candidates:
        raise RuntimeError("select_best_model: ningún modelo de la grilla convergió")

    best = min(candidates, key=lambda c: c[criterion])
    logger.info(
        "select_best_model: mejor modelo = %s/%s (%s=%.2f, de %d/%d modelos convergidos)",
        best["vol"], best["dist"], criterion, best[criterion], len(candidates), len(_GRID),
    )
    return best


# --------------------------------------------------------------------------
# Volatilidad condicional y pronóstico
# --------------------------------------------------------------------------


def conditional_volatility(res: ARCHModelResult, periods_per_year: int = PERIODS_PER_YEAR) -> pd.Series:
    """Volatilidad condicional diaria estimada por el modelo (`res`, salido
    de `fit_garch_variant`/`select_best_model`), reconvertida a escala
    DECIMAL y ANUALIZADA (x sqrt(periods_per_year)). Indexada por fecha,
    igual que `returns`.
    """
    daily_decimal = res.conditional_volatility / RETURN_SCALE
    return daily_decimal * np.sqrt(periods_per_year)


def forecast_volatility(
    res: ARCHModelResult,
    horizon: int = 1,
    periods_per_year: int = PERIODS_PER_YEAR,
    simulations: int = 10_000,
    random_state: int = 0,
) -> pd.Series:
    """Pronóstico de volatilidad ANUALIZADA en escala decimal a `horizon`
    días, usando el pronóstico de varianza de `arch` (`res.forecast`).

    Devuelve una `pd.Series` de longitud `horizon`, indexada 1..horizon (paso
    1 = mañana, paso 2 = pasado mañana, etc. — NO fechas de calendario, ya
    que `arch` no asume feriados/fines de semana en el horizonte pronosticado).

    Nota: EGARCH no tiene fórmula cerrada para pronósticos multi-paso (la
    recursión no es lineal en log-varianza), así que `arch` rechaza el
    método analítico por defecto cuando `horizon > 1` para esa variante. Acá
    se intenta primero el método analítico (determinístico, rápido) y, si
    falla, se recurre automáticamente al método de simulación de `arch`
    (`simulations` trayectorias Monte Carlo, con `random_state` fijo para
    que el resultado sea reproducible).
    """
    try:
        forecasts = res.forecast(horizon=horizon, reindex=False)
    except ValueError:
        logger.info(
            "forecast_volatility: pronóstico analítico no disponible para horizon=%d con este modelo "
            "(típico de EGARCH); usando método de simulación (%d trayectorias).",
            horizon, simulations,
        )
        forecasts = res.forecast(
            horizon=horizon,
            reindex=False,
            method="simulation",
            simulations=simulations,
            random_state=np.random.RandomState(random_state),
        )

    variance_pct2 = forecasts.variance.iloc[-1].to_numpy()
    daily_decimal_vol = np.sqrt(variance_pct2) / RETURN_SCALE
    annualized = daily_decimal_vol * np.sqrt(periods_per_year)
    return pd.Series(annualized, index=pd.RangeIndex(1, horizon + 1, name="horizonte"))


# --------------------------------------------------------------------------
# Régimen de volatilidad
# --------------------------------------------------------------------------


def volatility_regime(
    cond_vol: pd.Series, lookback: int = 252, low: float = 0.33, high: float = 0.66
) -> pd.Series:
    """Clasifica cada día de `cond_vol` (volatilidad condicional anualizada,
    ver `conditional_volatility`) en un régimen relativo a su propia
    historia reciente, según el percentil de la vol condicional dentro de
    una ventana móvil de `lookback` observaciones:
        - "calma": percentil <= `low`
        - "tension": percentil >= `high`
        - "normal": en el medio

    Los primeros `lookback - 1` días quedan en NaN por warmup (no hay
    suficiente historia para calcular el percentil).

    Este régimen es el insumo directo para vol targeting (sizing por riesgo)
    en fases posteriores: en "tension" el sizing debería reducir exposición,
    en "calma" puede tolerar mayor exposición, buscando mantener el riesgo
    (volatilidad) del portafolio aproximadamente constante en el tiempo.
    """
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"Se espera 0 <= low < high <= 1, recibido low={low}, high={high}")

    def _percentile_rank(window: np.ndarray) -> float:
        return float((window <= window[-1]).mean())

    pct_rank = cond_vol.rolling(window=lookback, min_periods=lookback).apply(_percentile_rank, raw=True)

    regime = pd.Series(np.nan, index=cond_vol.index, dtype=object)
    regime[pct_rank <= low] = "calma"
    regime[(pct_rank > low) & (pct_rank < high)] = "normal"
    regime[pct_rank >= high] = "tension"
    return regime


# --------------------------------------------------------------------------
# VaR paramétrico GARCH y validación (Kupiec)
# --------------------------------------------------------------------------


def garch_var(res: ARCHModelResult, alpha: float = 0.05) -> pd.Series:
    """VaR condicional paramétrico diario a partir de la volatilidad GARCH y
    la distribución ajustada del modelo (normal o t de Student estandarizada,
    con los grados de libertad estimados por `arch` si `dist="t"`).

    VaR_t = -(mu_t + sigma_t * q_alpha)

    con `q_alpha` el cuantil `alpha` de la distribución ESTANDARIZADA
    ajustada (media 0, varianza 1) y `mu_t`/`sigma_t` la media/volatilidad
    condicional del modelo en cada fecha `t`. Se reporta en escala DECIMAL,
    como PÉRDIDA POSITIVA (misma convención que
    `metrics.risk_measures.value_at_risk`): un VaR de 0.03 significa una
    pérdida esperada del 3% al nivel de confianza `1 - alpha`.
    """
    dist = res.model.distribution
    n_dist_params = dist.num_params
    dist_params = res.params.iloc[-n_dist_params:] if n_dist_params > 0 else []
    q_alpha = float(np.asarray(dist.ppf([alpha], dist_params)).ravel()[0])

    mu_pct = float(res.params.get("mu", 0.0))
    sigma_pct = res.conditional_volatility

    var_pct = -(mu_pct + sigma_pct * q_alpha)
    return var_pct / RETURN_SCALE


def kupiec_test(returns: pd.Series, var_series: pd.Series, alpha: float = 0.05) -> dict:
    """Test de cobertura incondicional de Kupiec (POF, Proportion of
    Failures) para validar una serie de VaR (p. ej. la de `garch_var`).

    Una EXCEPCIÓN (breach) ocurre en la fecha t si la pérdida realizada
    supera al VaR estimado ese día: `-returns_t > var_series_t` (ambos en
    convención de pérdida positiva). Bajo H0 (el VaR está bien calibrado),
    la tasa de excepciones esperada es `alpha` (p. ej. 0.05 para un VaR al
    95%) y el estadístico

        LR_pof = -2 * ln[(1-alpha)^(n-x) * alpha^x] + 2 * ln[(1-x/n)^(n-x) * (x/n)^x]

    se distribuye chi-cuadrado con 1 grado de libertad bajo H0. Se rechaza
    H0 (VaR mal calibrado) si el p-valor < `KUPIEC_SIGNIFICANCE` (0.05, fijo:
    es el nivel de significancia del TEST, no la `alpha` del VaR que se
    valida — son dos probabilidades conceptualmente distintas).

    Devuelve un dict: "n_obs", "n_excepciones", "tasa_esperada" (=alpha),
    "tasa_observada", "lr_estadistico", "p_valor", "rechaza_h0" (bool).
    """
    aligned = pd.concat([returns.rename("r"), var_series.rename("var")], axis=1).dropna()
    n = len(aligned)
    if n == 0:
        raise ValueError("kupiec_test: no hay observaciones alineadas entre 'returns' y 'var_series'")

    breaches = (-aligned["r"]) > aligned["var"]
    x = int(breaches.sum())
    observed_rate = x / n

    if x == 0:
        log_lik_null = n * np.log(1 - alpha)
        log_lik_alt = n * np.log(1 - observed_rate)
    elif x == n:
        log_lik_null = n * np.log(alpha)
        log_lik_alt = n * np.log(observed_rate)
    else:
        log_lik_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        log_lik_alt = (n - x) * np.log(1 - observed_rate) + x * np.log(observed_rate)

    lr_stat = -2.0 * (log_lik_null - log_lik_alt)
    p_value = float(1.0 - scipy_stats.chi2.cdf(lr_stat, df=1))

    return {
        "n_obs": n,
        "n_excepciones": x,
        "tasa_esperada": alpha,
        "tasa_observada": observed_rate,
        "lr_estadistico": float(lr_stat),
        "p_valor": p_value,
        "rechaza_h0": bool(p_value < KUPIEC_SIGNIFICANCE),
    }
