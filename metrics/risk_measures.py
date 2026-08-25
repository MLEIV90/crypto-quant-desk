"""Métricas de riesgo y performance para series de retornos.

Convención del proyecto: los retornos de entrada están en escala DECIMAL
(0.01 == 1%) y se asume frecuencia diaria por defecto
(`periods_per_year=config.PERIODS_PER_YEAR`, 365 para cripto — opera todos
los días del año, no solo días hábiles como las acciones). El estilo sigue
al repo `momentum` de MLEIV90, adaptado de mensual a diario.

Las medidas de riesgo de cola (VaR, Expected Shortfall) y el máximo drawdown
tienen convenciones de signo explícitas documentadas en cada función — no son
uniformes entre sí porque cada una sigue la convención estándar de su propia
literatura (VaR/ES como pérdida positiva, drawdown como caída negativa).

Casos degenerados (volatilidad o drawdown nulo) se definen explícitamente en
cada función en vez de dejar que devuelvan NaN/inf de forma implícita.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import PERIODS_PER_YEAR

# Tolerancia para tratar una volatilidad/downside-deviation como "cero".
# Retornos genuinamente constantes (p. ej. todos 0.001) pueden dar un
# std(ddof=1) no exactamente 0 por error de punto flotante en la resta de la
# media; sin esta tolerancia, sharpe/sortino devolverían un número enorme
# pero finito en vez del +-inf conceptualmente correcto.
_ZERO_VOL_TOLERANCE = 1e-12


def _sign_ratio_when_zero_denominator(mean_excess: float) -> float:
    """Convención compartida por sharpe/sortino/calmar cuando el denominador
    (volatilidad, downside deviation o drawdown) es exactamente 0: el ratio
    es +inf si el numerador es positivo, -inf si es negativo, y 0.0 si
    también es 0 (sin retorno y sin riesgo).
    """
    if mean_excess > 0:
        return np.inf
    if mean_excess < 0:
        return -np.inf
    return 0.0


# --------------------------------------------------------------------------
# Retorno y volatilidad anualizados
# --------------------------------------------------------------------------


def annualized_return(r: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Retorno anualizado geométrico (compuesto) a partir de retornos simples:

        annualized_return = (prod(1 + r))^(periods_per_year / n) - 1
    """
    r = r.dropna()
    n = len(r)
    if n == 0:
        return np.nan
    compounded_growth = (1.0 + r).prod()
    return compounded_growth ** (periods_per_year / n) - 1.0


def annualized_volatility(r: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Desvío estándar anualizado bajo el supuesto de retornos i.i.d.:
    std(r) * sqrt(periods_per_year).
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)


# --------------------------------------------------------------------------
# Ratios ajustados por riesgo
# --------------------------------------------------------------------------


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Sharpe ratio anualizado con tasa libre de riesgo anual `rf_annual`.

    La tasa libre de riesgo se convierte a la frecuencia de `r` de forma
    geométrica: rf_periodo = (1 + rf_annual)^(1/periods_per_year) - 1.

    Caso degenerado: si el exceso de retorno tiene volatilidad 0 (p. ej.
    retornos constantes), el ratio es +inf/0.0/-inf según el signo del
    retorno medio (ver `_sign_ratio_when_zero_denominator`), en vez de NaN.
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    vol = excess.std(ddof=1)
    if vol <= _ZERO_VOL_TOLERANCE:
        return _sign_ratio_when_zero_denominator(excess.mean())
    return (excess.mean() / vol) * np.sqrt(periods_per_year)


def sharpe_lo_adjusted(
    r: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    max_lags: int | None = None,
) -> float:
    """Sharpe ratio anualizado ajustado por autocorrelación serial (Lo, 2002).

    El Sharpe anualizado "naive" escala el Sharpe por período por
    sqrt(periods_per_year), lo cual asume retornos i.i.d. Si los retornos
    tienen autocorrelación serial (frecuente en series financieras por
    smoothing/iliquidez), ese escalado sobre- o subestima el Sharpe real.
    Lo (2002) corrige el factor de anualización:

        SR_q = SR_1 * sqrt(q / (1 + 2 * sum_{k=1}^{L} (1 - k/q) * rho_k))

    donde SR_1 es el Sharpe por período (sin anualizar), q = periods_per_year,
    rho_k es la autocorrelación muestral de orden k del exceso de retorno, y
    L es la cantidad de lags considerados (por defecto q - 1, como en Lo
    2002; se puede acotar con `max_lags` para series cortas o para evitar
    acumular ruido de autocorrelaciones de orden alto mal estimadas).

    Si el denominador del ajuste no es positivo (autocorrelación negativa
    extrema, fuera del rango teóricamente válido), se devuelve NaN en vez de
    un número sin sentido.
    """
    r = r.dropna()
    n = len(r)
    if n < 3:
        return np.nan

    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    vol = excess.std(ddof=1)
    if vol <= _ZERO_VOL_TOLERANCE:
        return _sign_ratio_when_zero_denominator(excess.mean())
    sr_period = excess.mean() / vol

    q = periods_per_year
    lags = min(max_lags if max_lags is not None else q - 1, n - 2)
    lags = max(lags, 0)
    if lags == 0:
        return sr_period * np.sqrt(q)

    autocorr_sum = 0.0
    for k in range(1, lags + 1):
        rho_k = excess.autocorr(lag=k)
        if pd.isna(rho_k):
            continue
        autocorr_sum += (1.0 - k / q) * rho_k

    denom = 1.0 + 2.0 * autocorr_sum
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return sr_period * np.sqrt(q / denom)


def sortino_ratio(r: pd.Series, rf_annual: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Sortino ratio anualizado: como el Sharpe, pero solo penaliza la
    volatilidad a la baja (downside deviation), no la dispersión total.

        downside_dev = sqrt(mean(min(excess, 0)^2))

    Caso degenerado: si no hay downside (downside_dev = 0), aplica la misma
    convención que `sharpe_ratio` para volatilidad nula.
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    downside = excess.clip(upper=0.0)
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev <= _ZERO_VOL_TOLERANCE:
        return _sign_ratio_when_zero_denominator(excess.mean())
    return (excess.mean() / downside_dev) * np.sqrt(periods_per_year)


# --------------------------------------------------------------------------
# Equity curve y drawdown
# --------------------------------------------------------------------------


def equity_curve(r: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Curva de capital acumulado a partir de retornos simples periódicos:

        equity_t = initial_value * prod_{i<=t} (1 + r_i)

    Los NaN (típicamente el primer valor de `signals.returns.simple_returns`)
    se tratan como 0 (sin variación en ese período) para no perder el primer
    punto de la curva.
    """
    return initial_value * (1.0 + r.fillna(0.0)).cumprod()


def drawdown_series(r: pd.Series) -> pd.Series:
    """Curva "underwater" completa: en cada fecha, cuánto por debajo del
    máximo histórico previo de la equity curve está el capital, como número
    NEGATIVO o cero (p. ej. -0.20 = -20% respecto del pico anterior; 0.0 en
    los picos nuevos). Es la serie completa detrás del escalar
    `max_drawdown` (Fase 21: graficarla es lo que deja VER por qué una
    estrategia sufre menos que otra, no solo su peor valor puntual).
    """
    r = r.dropna()
    if len(r) == 0:
        return r
    equity = equity_curve(r)
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(r: pd.Series) -> float:
    """Máxima caída porcentual de la equity curve desde un máximo histórico
    previo, reportada como número NEGATIVO o cero (p. ej. -0.20 = caída del
    20%; 0.0 si la equity curve nunca cae por debajo de un máximo previo).
    """
    drawdown = drawdown_series(r)
    if len(drawdown) == 0:
        return np.nan
    return drawdown.min()


def calmar_ratio(r: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Calmar ratio: retorno anualizado / |máximo drawdown|.

    Caso degenerado: si el máximo drawdown es 0 (la equity curve nunca cae),
    aplica la misma convención que `sharpe_ratio` para denominador nulo,
    usando el signo del retorno anualizado.
    """
    ann_ret = annualized_return(r, periods_per_year=periods_per_year)
    mdd = max_drawdown(r)
    if pd.isna(ann_ret) or pd.isna(mdd):
        return np.nan
    if mdd == 0:
        return _sign_ratio_when_zero_denominator(ann_ret)
    return ann_ret / abs(mdd)


# --------------------------------------------------------------------------
# Riesgo de cola: VaR y Expected Shortfall
# --------------------------------------------------------------------------


def value_at_risk(r: pd.Series, level: float = 0.95) -> float:
    """VaR histórico al nivel de confianza `level` (p. ej. 0.95 = 95%),
    reportado como PÉRDIDA POSITIVA: VaR95% = 0.03 significa que, con 95% de
    confianza, la pérdida en un período no supera el 3%.

    Se calcula como el percentil (1 - level) de la distribución empírica de
    retornos (método histórico, sin supuesto de normalidad).
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    quantile = 1.0 - level
    return -float(np.quantile(r, quantile))


def expected_shortfall(r: pd.Series, level: float = 0.95) -> float:
    """Expected Shortfall / CVaR histórico al nivel `level`: pérdida promedio
    en el peor (1 - level) de los casos, reportada como PÉRDIDA POSITIVA.

    Por construcción, ES >= VaR al mismo nivel de confianza (es el promedio
    de la cola que define el VaR, que es al menos tan extrema como el propio
    percentil).
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    quantile = 1.0 - level
    threshold = np.quantile(r, quantile)
    tail = r[r <= threshold]
    if len(tail) == 0:
        tail = r.nsmallest(1)
    return -float(tail.mean())


# --------------------------------------------------------------------------
# Percentil histórico y VaR/ES en ventana móvil (Fase 20a)
# --------------------------------------------------------------------------


def historical_percentile(series: pd.Series, value: float | None = None) -> float:
    """Percentil (0-100) de `value` dentro de la distribución histórica de
    `series` — "¿qué fracción de las observaciones históricas tuvo un valor
    MENOR O IGUAL al de hoy?". Si `value` es None (default), se usa el
    ÚLTIMO valor no-NaN de `series` — la pregunta natural "¿el valor de HOY
    es alto o bajo comparado con su propia historia?".

    Es una foto DESCRIPTIVA de dónde cae un valor respecto de su propio
    pasado — no una probabilidad de nada futuro ni una predicción: un
    percentil alto de volatilidad hoy no dice nada sobre la volatilidad de
    mañana, solo que hoy es inusual respecto de lo que ya se vio.

    Devuelve NaN si `series` no tiene ninguna observación válida.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan
    if value is None:
        value = float(clean.iloc[-1])
    return float((clean <= value).mean() * 100.0)


def rolling_value_at_risk(r: pd.Series, window: int, level: float = 0.95) -> pd.Series:
    """`value_at_risk` (reutilizada tal cual, sin reimplementar el cálculo)
    recalculada en una ventana móvil de `window` observaciones — para poder
    ubicar el VaR de HOY en el contexto de cómo fue variando en el tiempo
    (ver `historical_percentile`), en vez de un único número sobre TODA la
    historia. Los primeros `window - 1` valores quedan en NaN (warmup, sin
    suficientes observaciones todavía para esa ventana).

    USO EN LA API (Fase 20c): esta es la vía "VaR actual" para
    `GET /api/risk-summary`, que a propósito NO ajusta un modelo GARCH por
    activo (ver ese endpoint) — para `GET /api/risk`, que SÍ tiene un GARCH
    ya ajustado, el VaR "actual" que refleja el régimen de hoy es
    `models.garch.garch_var` (paramétrico, genuinamente del momento), no
    esta versión por ventana móvil.
    """
    return r.rolling(window).apply(lambda window_values: value_at_risk(pd.Series(window_values), level=level), raw=True)
