"""Comparación de rendimiento normalizado entre activos (Fase 12a) — insumo
de la pestaña "Comparación" del frontend (`GET /api/compare`).

No reimplementa la carga de precios (eso lo hace `data.loaders.get_prices`,
llamado por `api/main.py`, no por acá): estas funciones reciben las series
de precio YA CARGADAS y solo alinean/normalizan/rankean. Mismo criterio de
alineación por fechas comunes que ya usaba `eda.eda_report.correlation_matrix`
(`pd.DataFrame(dict).dropna(how="any")`, inner join implícito) — no se
reimplementa esa idea, se reutiliza el mismo patrón.

Fase 27: además del rendimiento normalizado, `compare_assets` calcula
métricas de RIESGO por activo sobre la misma ventana comparada —
reutilizando `signals.returns.simple_returns` y `metrics.risk_measures`
(`annualized_volatility`/`max_drawdown`/`sharpe_ratio`) tal cual, sin
reimplementar ningún cálculo de riesgo acá.

HONESTIDAD: esto es una comparación de desempeño HISTÓRICO, no una
predicción — el desempeño pasado no garantiza el futuro (ver el texto que
muestra el frontend). Rendimiento alto suele venir acompañado de riesgo
alto: "quién subió más" y "quién rindió mejor ajustado por riesgo" no son
la misma pregunta, por eso se exponen las dos.
"""

from __future__ import annotations

import pandas as pd

from metrics.risk_measures import annualized_volatility, max_drawdown, sharpe_ratio
from signals.returns import simple_returns

DEFAULT_NORMALIZATION_BASE: float = 100.0


def normalize_to_base(prices: pd.Series, base: float = DEFAULT_NORMALIZATION_BASE) -> pd.Series:
    """Normaliza una serie de precios a un valor `base` común en su PRIMER
    punto: `normalizado_t = precio_t / precio_0 * base`.

    Tras normalizar, todas las series comparadas empiezan en el mismo valor
    (`base`) el mismo día, así que la diferencia entre ellas en cualquier
    punto posterior refleja directamente la diferencia de RENDIMIENTO
    acumulado desde el inicio de la ventana, no la diferencia de escala de
    precio (comparar $60.000 de BTC contra $2.000 de ETH directamente no
    dice nada sobre cuál "rindió mejor").
    """
    return prices / prices.iloc[0] * base


def align_common_dates(price_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Alinea varias series de precio por FECHAS COMUNES (inner join
    implícito vía `dropna`, mismo patrón que
    `eda.eda_report.correlation_matrix`): solo conserva las fechas donde
    TODOS los activos pedidos tienen dato.

    Importante para activos con historia de distinta longitud (p. ej. SOL
    arranca en 2020, mucho después que BTC/ETH) — la fecha de inicio de la
    comparación queda determinada por el activo con la historia MÁS CORTA
    dentro del grupo pedido, no por el de historia más larga.
    """
    return pd.DataFrame(price_series).dropna(how="any")


def compare_assets(price_series: dict[str, pd.Series], limit: int) -> dict:
    """Compara el rendimiento Y EL RIESGO de varios activos: alinea por
    fechas comunes (`align_common_dates`), recorta a las últimas `limit`
    fechas comunes, y normaliza cada activo a base 100 DENTRO de esa
    ventana recortada — así "quién rindió mejor" se mide sobre el PERÍODO
    elegido (últimas `limit` fechas), no sobre toda la historia común
    disponible.

    Devuelve un dict:
    - "fechas": `pd.DatetimeIndex` de las fechas comunes usadas (recortadas)
      — `fechas[0]` es la fecha BASE de la comparación: la primera fecha
      donde TODOS los activos pedidos tienen dato dentro de la ventana.
    - "normalizado": `pd.DataFrame`, una columna por activo, cada una
      arrancando en 100.0 en la primera fecha de la ventana.
    - "rendimiento_total_pct": dict `{activo: rendimiento total del período,
      en puntos porcentuales}` — equivalente a `normalizado[activo].iloc[-1]
      - 100`, para armar un ranking.
    - "riesgo" (Fase 27): dict `{activo: {"vol_anualizada", "max_drawdown",
      "sharpe"}}` — reutiliza `signals.returns.simple_returns` y
      `metrics.risk_measures.annualized_volatility`/`max_drawdown`/
      `sharpe_ratio` TAL CUAL, calculados sobre retornos simples de la
      MISMA ventana recortada que `rendimiento_total_pct` (no toda la
      historia del activo) — para que "quién subió más" se pueda comparar
      al lado de "con cuánto riesgo lo hizo" sobre el mismo período exacto.

    Si `align_common_dates` da un DataFrame vacío (sin fechas comunes en
    absoluto, p. ej. activos sin ningún solapamiento de historia), devuelve
    todo vacío en vez de lanzar una excepción — decisión de quien llama qué
    hacer con eso (ver `api/main.py::get_compare`).
    """
    aligned = align_common_dates(price_series)
    recent = aligned.tail(limit)

    if recent.empty:
        return {"fechas": recent.index, "normalizado": recent, "rendimiento_total_pct": {}, "riesgo": {}}

    normalized = recent.apply(normalize_to_base, axis=0)
    rendimiento_total_pct = {asset: float(normalized[asset].iloc[-1] - DEFAULT_NORMALIZATION_BASE) for asset in normalized.columns}

    riesgo = {}
    for asset in recent.columns:
        asset_returns = simple_returns(recent[asset]).dropna()
        riesgo[asset] = {
            "vol_anualizada": annualized_volatility(asset_returns),
            "max_drawdown": max_drawdown(asset_returns),
            "sharpe": sharpe_ratio(asset_returns),
        }

    return {
        "fechas": recent.index,
        "normalizado": normalized,
        "rendimiento_total_pct": rendimiento_total_pct,
        "riesgo": riesgo,
    }
