"""Funciones de análisis exploratorio (EDA) para retornos de crypto-quant-desk.

Este módulo contiene solo funciones importables y testeables; el notebook
`eda/exploracion.ipynb` únicamente las llama y muestra sus resultados, no
reimplementa ninguna lógica acá.

Convención del proyecto: los retornos de entrada están en escala DECIMAL
(0.01 == 1%), frecuencia diaria por defecto (`PERIODS_PER_YEAR`, ver
config.py).

Las funciones de ploteo (`plot_*`) SIEMPRE guardan la figura a
`eda/figures/` y nunca la muestran (backend matplotlib no interactivo), para
poder correr el notebook headless (`jupyter nbconvert --execute`) sin
depender de un display.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend no interactivo: estas funciones solo guardan a disco.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from config import PERIODS_PER_YEAR

logger = logging.getLogger(__name__)

FIGURES_DIR: Path = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cuantiles reportados por `descriptive_stats`.
DESCRIPTIVE_QUANTILES: list[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


# --------------------------------------------------------------------------
# Estadística descriptiva y tests
# --------------------------------------------------------------------------


def descriptive_stats(returns: pd.Series, name: str | None = None) -> pd.DataFrame:
    """Estadística descriptiva de una serie de retornos (escala decimal).

    Devuelve un DataFrame de UNA fila (índice = `name`, o el `.name` de la
    serie, o "returns" si ninguno está definido) con columnas: n, media, std,
    skew, kurtosis_exceso, min, max, y los cuantiles de
    `DESCRIPTIVE_QUANTILES` (q01, q05, q25, q50, q75, q95, q99).

    La kurtosis es EXCESO (kurtosis normal = 0, convención de pandas/scipy),
    el diagnóstico estándar para detectar colas más gordas que una normal.

    Al devolver una sola fila, es fácil construir una tabla comparativa
    entre varios activos con `pd.concat([descriptive_stats(r, a) for ...])`.
    """
    r = returns.dropna()
    label = name or (returns.name if returns.name else "returns")

    stats: dict[str, float] = {
        "n": len(r),
        "media": r.mean(),
        "std": r.std(ddof=1),
        "skew": r.skew(),
        "kurtosis_exceso": r.kurt(),
        "min": r.min(),
        "max": r.max(),
    }
    for q in DESCRIPTIVE_QUANTILES:
        stats[f"q{int(round(q * 100)):02d}"] = r.quantile(q)

    return pd.DataFrame([stats], index=[label])


def adf_test(series: pd.Series, *, significance: float = 0.05) -> dict:
    """Test Augmented Dickey-Fuller de raíz unitaria (`statsmodels.adfuller`).

    H0: la serie tiene raíz unitaria (no es estacionaria). Se rechaza H0 (se
    concluye estacionariedad) si el p-valor < `significance`.

    Se usa tanto sobre el nivel de precio (típicamente NO estacionario, un
    "random walk" con tendencia) como sobre los retornos (típicamente SÍ
    estacionarios) para ilustrar por qué se modela sobre retornos y no sobre
    precio.

    Devuelve un dict: estadistico, p_valor, n_lags (rezagos elegidos por
    AIC), n_obs, valores_criticos (dict al 1%/5%/10%), es_estacionaria (bool).
    """
    r = series.dropna()
    stat, p_value, n_lags, n_obs, crit_values, _ = adfuller(r, autolag="AIC")
    return {
        "estadistico": float(stat),
        "p_valor": float(p_value),
        "n_lags": int(n_lags),
        "n_obs": int(n_obs),
        "valores_criticos": {k: float(v) for k, v in crit_values.items()},
        "es_estacionaria": bool(p_value < significance),
    }


def ljung_box_squared(returns: pd.Series, lags: int = 20, *, significance: float = 0.05) -> dict:
    """Test de Ljung-Box sobre los RETORNOS AL CUADRADO: evidencia indirecta
    de clustering de volatilidad (efecto ARCH). Si los retornos al cuadrado
    están autocorrelacionados, la varianza condicional no es constante en el
    tiempo, lo que justifica modelar con GARCH en fases posteriores.

    H0: no hay autocorrelación en los retornos al cuadrado hasta `lags`
    rezagos. Se rechaza H0 (hay evidencia de clustering de volatilidad) si el
    p-valor al último rezago < `significance`.

    Devuelve un dict: lags, estadistico y p_valor (al último rezago), tabla
    (DataFrame completo de statsmodels con todos los rezagos), y
    hay_clustering_volatilidad (bool).

    Si la serie tiene pocas observaciones, `lags` se recorta automáticamente
    a como máximo `len(r2) - 1` (statsmodels no puede calcular la ACF más
    allá de eso) y se loguea un warning; con series demasiado cortas (menos
    de 2 observaciones válidas, p. ej. un activo con muy poco histórico en
    una fuente de fallback) no se puede correr el test y se devuelve un
    resultado con NaN en vez de propagar la excepción de statsmodels.
    """
    r2 = returns.dropna() ** 2
    effective_lags = min(lags, len(r2) - 1)

    if effective_lags < 1:
        logger.warning(
            "ljung_box_squared: solo %d observaciones válidas, no se puede correr el test (se requieren >= 2)",
            len(r2),
        )
        return {
            "lags": 0,
            "estadistico": float("nan"),
            "p_valor": float("nan"),
            "tabla": pd.DataFrame(),
            "hay_clustering_volatilidad": False,
        }
    if effective_lags < lags:
        logger.warning(
            "ljung_box_squared: se pidieron %d lags pero solo hay %d observaciones válidas; se usan %d lags",
            lags, len(r2), effective_lags,
        )

    tabla = acorr_ljungbox(r2, lags=effective_lags, return_df=True)
    ultimo = tabla.iloc[-1]
    return {
        "lags": effective_lags,
        "estadistico": float(ultimo["lb_stat"]),
        "p_valor": float(ultimo["lb_pvalue"]),
        "tabla": tabla,
        "hay_clustering_volatilidad": bool(ultimo["lb_pvalue"] < significance),
    }


def correlation_matrix(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Matriz de correlación de retornos entre activos.

    Alinea las series por fecha con un inner join implícito (`dropna`):
    solo se usan fechas donde TODOS los activos tienen dato, para no sesgar
    la correlación con periodos donde falta alguno.

    Insumo para la selección de candidatos a pares en la Fase 2 (stat-arb):
    activos muy correlacionados son los primeros candidatos a testear por
    cointegración.
    """
    df = pd.DataFrame(returns_dict).dropna(how="any")
    return df.corr()


# --------------------------------------------------------------------------
# Ploteo (guardan a eda/figures/, nunca muestran)
# --------------------------------------------------------------------------


def _savefig(fig: plt.Figure, filename: str) -> Path:
    out_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Figura guardada en '%s'", out_path)
    return out_path


def plot_price_log(close: pd.Series, asset: str) -> Path:
    """Serie de precio de cierre en escala logarítmica: hace comparable la
    magnitud relativa de las variaciones a lo largo de años de crecimiento
    exponencial típico de cripto (en escala lineal, los primeros años
    quedarían aplastados contra el eje). Guarda en
    `eda/figures/{asset}_price_log.png`.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(close.index, close.to_numpy())
    ax.set_yscale("log")
    ax.set_title(f"{asset}: precio de cierre (escala log)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD, log)")
    return _savefig(fig, f"{asset}_price_log.png")


def plot_returns_histogram(returns: pd.Series, asset: str) -> Path:
    """Histograma + KDE de retornos diarios, contrastado con una normal de
    igual media/std, para inspeccionar visualmente colas gordas y asimetría.
    Guarda en `eda/figures/{asset}_returns_hist.png`.
    """
    r = returns.dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r, bins=100, density=True, alpha=0.6, label="empírico")

    x = np.linspace(r.min(), r.max(), 500)
    kde = scipy_stats.gaussian_kde(r)
    ax.plot(x, kde(x), label="KDE")
    normal = scipy_stats.norm(loc=r.mean(), scale=r.std(ddof=1))
    ax.plot(x, normal.pdf(x), linestyle="--", label="normal (misma media/std)")

    ax.set_title(f"{asset}: distribución de retornos diarios")
    ax.set_xlabel("Retorno (decimal)")
    ax.legend()
    return _savefig(fig, f"{asset}_returns_hist.png")


def plot_qq(returns: pd.Series, asset: str) -> Path:
    """QQ-plot de los retornos contra una normal teórica: desviaciones de la
    diagonal en los extremos son evidencia visual de colas más gordas que
    una normal. Guarda en `eda/figures/{asset}_qq.png`.
    """
    r = returns.dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    scipy_stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(f"{asset}: QQ-plot vs. normal")
    return _savefig(fig, f"{asset}_qq.png")


def plot_acf_returns(returns: pd.Series, asset: str, lags: int = 20) -> Path:
    """ACF de los retornos (evidencia de autocorrelación lineal en el nivel,
    típicamente débil en cripto) y de los retornos al cuadrado (evidencia de
    clustering de volatilidad), lado a lado. Guarda en
    `eda/figures/{asset}_acf.png`.

    Si la serie tiene pocas observaciones (p. ej. un activo con muy poco
    histórico en una fuente de fallback), `lags` se recorta automáticamente
    a como máximo `len(r) - 1`: `statsmodels.plot_acf` no puede graficar más
    rezagos que observaciones disponibles.
    """
    r = returns.dropna()
    effective_lags = max(min(lags, len(r) - 1), 0)
    if effective_lags < lags:
        logger.warning(
            "plot_acf_returns: se pidieron %d lags pero solo hay %d observaciones válidas; se usan %d lags",
            lags, len(r), effective_lags,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(r, lags=effective_lags, ax=axes[0])
    axes[0].set_title(f"{asset}: ACF retornos")
    plot_acf(r**2, lags=effective_lags, ax=axes[1])
    axes[1].set_title(f"{asset}: ACF retornos²")
    return _savefig(fig, f"{asset}_acf.png")


def plot_rolling_volatility(
    returns: pd.Series, asset: str, window: int = 30, periods_per_year: int = PERIODS_PER_YEAR
) -> Path:
    """Volatilidad rolling anualizada: desvío estándar móvil de `window` días
    escalado por sqrt(periods_per_year), para visualizar regímenes de
    volatilidad cambiantes en el tiempo (motivación directa para GARCH en
    fases posteriores). Guarda en `eda/figures/{asset}_rolling_vol.png`.
    """
    r = returns.dropna()
    rolling_vol = r.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling_vol.index, rolling_vol.to_numpy())
    ax.set_title(f"{asset}: volatilidad rolling anualizada ({window}d)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Volatilidad anualizada")
    return _savefig(fig, f"{asset}_rolling_vol.png")
