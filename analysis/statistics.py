"""Análisis estadístico de estacionalidad, autocorrelación y ciclos
(Fase 11) — insumo de la vista "Estadística" del frontend (`GET /api/stats`).

HONESTIDAD, mismo criterio que el resto del proyecto (ver
`signals/studies.py`, `eda/eda_report.py`): NADA de lo que hay acá predice
el precio. La estacionalidad histórica en cripto es débil e inestable
(muestras de pocos años, sin razón estructural fuerte como calendario
fiscal o climático); la ausencia de un ciclo dominante en el periodograma
es el resultado TÍPICO, no un fallo del análisis. Se documenta así en cada
función, para que el frontend pueda mostrarlo con el mismo tono.

Reutiliza `eda.eda_report.adf_test` para la estacionariedad (no
reimplementado acá, ver `api/main.py::get_stats`) y `statsmodels`/`scipy`
para ACF y el periodograma — este módulo no reimplementa ningún cálculo
estadístico de bajo nivel, solo arma las agregaciones sobre retornos que
todavía no existían en el proyecto.

Convención: todas las funciones esperan `returns` en escala DECIMAL
(0.01 == 1%), con índice `DatetimeIndex` UTC tz-aware (el contrato de
`data.loaders.get_prices`) — igual que el resto del proyecto.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import welch
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)

DEFAULT_ACF_LAGS: int = 30
DEFAULT_PERIODOGRAM_TOP_N: int = 3

# Fechas de halving de Bitcoin conocidas (reducción a la mitad de la
# recompensa por bloque, cada ~4 años) — hechos públicos, no un ajuste
# estadístico. Se usan para marcar el gráfico de ciclos cuando asset="BTC".
BTC_HALVING_DATES: list[str] = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]


# --------------------------------------------------------------------------
# Estacionalidad
# --------------------------------------------------------------------------


def monthly_seasonality(returns: pd.Series) -> pd.DataFrame:
    """Retorno por MES calendario (1=enero .. 12=diciembre), agregando todos
    los años disponibles en la serie.

    HONESTIDAD: con pocos años de historia (cripto rara vez supera 6-8 años
    por activo), cada mes tiene pocas observaciones INDEPENDIENTES (una
    franja de ~30 días por año, no 30 observaciones independientes) — un
    patrón mensual acá es, como mucho, sugestivo, nunca una regla operable.

    Devuelve un DataFrame indexado 1-12 (columna `mes`, meses sin ninguna
    observación no aparecen), columnas ["retorno_medio", "mediana",
    "desvio", "n"]. `desvio` puede ser NaN si un mes tiene una sola
    observación (desvío muestral con 1 dato no está definido).
    """
    r = returns.dropna()
    months = r.index.month
    grouped = r.groupby(months)
    out = pd.DataFrame(
        {
            "retorno_medio": grouped.mean(),
            "mediana": grouped.median(),
            "desvio": grouped.std(ddof=1),
            "n": grouped.count(),
        }
    )
    out.index.name = "mes"
    return out


def weekday_seasonality(returns: pd.Series) -> pd.DataFrame:
    """Retorno por día de la semana (convención `DatetimeIndex.dayofweek`:
    0=lunes ... 6=domingo).

    A diferencia de las acciones (que no operan fin de semana), cripto
    opera los 7 días — no hay un "hueco" de fin de semana sin datos.

    Devuelve un DataFrame indexado 0-6, columnas ["retorno_medio", "n"].
    """
    r = returns.dropna()
    weekdays = r.index.dayofweek
    grouped = r.groupby(weekdays)
    out = pd.DataFrame({"retorno_medio": grouped.mean(), "n": grouped.count()})
    out.index.name = "dia_semana"
    return out


def hourly_seasonality(returns: pd.Series) -> pd.DataFrame:
    """Retorno por hora del día (UTC, 0-23) — SOLO tiene sentido con
    retornos HORARIOS (`interval="1h"`): con retornos diarios todas las
    velas caen a la misma hora (medianoche UTC) y el resultado sería un
    único bucket sin información. Quien llama es responsable de pasar
    retornos horarios (ver `api/main.py::get_stats`, que solo la corre si
    `interval == "1h"`).

    Devuelve un DataFrame indexado 0-23, columnas ["retorno_medio", "n"].
    """
    r = returns.dropna()
    hours = r.index.hour
    grouped = r.groupby(hours)
    out = pd.DataFrame({"retorno_medio": grouped.mean(), "n": grouped.count()})
    out.index.name = "hora"
    return out


# --------------------------------------------------------------------------
# Autocorrelación
# --------------------------------------------------------------------------


def autocorrelation(returns: pd.Series, lags: int = DEFAULT_ACF_LAGS) -> pd.DataFrame:
    """ACF (función de autocorrelación) de los retornos y de los retornos
    AL CUADRADO, rezago a rezago — reutiliza `statsmodels.tsa.stattools.acf`
    (no reimplementa el cálculo).

    Dos lecturas distintas, a propósito:
    - ACF de los retornos: autocorrelación lineal del NIVEL de retorno. En
      un mercado razonablemente eficiente debería rondar 0 en todos los
      rezagos (el retorno de ayer no debería predecir el de hoy). Valores
      cerca de 0 son la evidencia ESPERADA, no un fallo del análisis.
    - ACF de los retornos al cuadrado: evidencia de CLUSTERING DE
      VOLATILIDAD (a un día volátil le sigue otro día volátil, aunque el
      SIGNO del retorno no se pueda predecir). Típicamente positiva y
      persistente en cripto en los primeros rezagos — la motivación
      empírica de modelar la volatilidad con GARCH (ver `models/garch.py`).
      Es la misma idea que testea `eda.eda_report.ljung_box_squared` (un
      único test de hipótesis); acá se devuelve el COEFICIENTE completo
      rezago a rezago, pensado para graficarlo de una.

    `lags` se recorta a `len(r) - 1` si la serie es corta (mismo criterio
    que `eda.eda_report.plot_acf_returns`/`ljung_box_squared`).

    Devuelve un DataFrame indexado 0..lags_efectivos (columna `lag`),
    columnas ["acf_retornos", "acf_retornos2"]. El rezago 0 es siempre 1.0
    (la correlación de la serie consigo misma).
    """
    r = returns.dropna()
    effective_lags = max(min(lags, len(r) - 1), 0)
    if effective_lags < lags:
        logger.warning(
            "autocorrelation: se pidieron %d lags pero solo hay %d observaciones válidas; se usan %d lags",
            lags, len(r), effective_lags,
        )

    acf_returns = acf(r, nlags=effective_lags, fft=True)
    acf_returns_sq = acf(r**2, nlags=effective_lags, fft=True)

    out = pd.DataFrame(
        {"acf_retornos": acf_returns, "acf_retornos2": acf_returns_sq},
        index=pd.RangeIndex(effective_lags + 1),
    )
    out.index.name = "lag"
    return out


# --------------------------------------------------------------------------
# Ciclos (periodograma espectral)
# --------------------------------------------------------------------------


def spectral_periodogram(
    returns: pd.Series, periods_per_day: float = 1.0, top_n: int = DEFAULT_PERIODOGRAM_TOP_N
) -> dict:
    """DEPRECADA (Fase 15a): ya NO se expone en `GET /api/stats` ni en la
    vista "Ciclos y Estadística" del frontend. Sobre retornos DIARIOS, el
    período dominante que devolvía eran 2-3 días — ruido de alta
    frecuencia, sin ningún significado de mercado, no un hallazgo útil.
    Se reemplazó por `analysis/cycles.py` (drawdowns, fases bull/bear,
    ciclos de halving), que sí mide ciclos con significado reconocible.
    Esta función se deja acá tal cual (sigue funcionando, sigue testeada)
    por si hiciera falta en el futuro, pero no la use código nuevo.

    Periodograma de Welch de los retornos (`scipy.signal.welch`, no
    reimplementa la FFT ni el promediado de segmentos) — busca ciclos
    periódicos en la serie.

    HONESTIDAD: en series financieras, lo TÍPICO es que el periodograma NO
    muestre un pico dominante claro — los retornos se parecen mucho a
    ruido, sin periodicidad fuerte. La AUSENCIA de un pico no es un fallo
    de este análisis, es la evidencia de que no hay un ciclo operable. Un
    pico aislado y fuerte SÍ sería evidencia de periodicidad, pero conviene
    sospechar primero de artefactos conocidos (efectos de calendario, de
    fin de semana/mes, o simplemente ruido de muestra chica) antes de
    leerlo como una señal real explotable.

    `periods_per_day`: cuántas observaciones hay por día en `returns` (1.0
    para retornos diarios, 24.0 para horarios) — necesario para expresar
    los períodos dominantes en DÍAS sin importar el intervalo de entrada.

    Devuelve un dict:
    - "frecuencias": frecuencias en ciclos/día (lista de float).
    - "potencia": densidad espectral de potencia en cada frecuencia (mismo
      orden que "frecuencias").
    - "top_periodos_dias": hasta `top_n` tuplas (periodo_en_dias, potencia),
      ordenadas por potencia DESCENDENTE, excluyendo la frecuencia 0 (el
      nivel medio de la serie, que no es un ciclo).
    """
    r = returns.dropna().to_numpy()
    nperseg = min(len(r), 256)
    if nperseg < 2:
        return {"frecuencias": [], "potencia": [], "top_periodos_dias": []}

    freqs, power = welch(r, fs=periods_per_day, nperseg=nperseg)

    nonzero = freqs > 0
    freqs_nonzero = freqs[nonzero]
    power_nonzero = power[nonzero]

    order = np.argsort(power_nonzero)[::-1][:top_n]
    top_periods = [(float(1.0 / freqs_nonzero[i]), float(power_nonzero[i])) for i in order]

    return {
        "frecuencias": freqs.tolist(),
        "potencia": power.tolist(),
        "top_periodos_dias": top_periods,
    }
