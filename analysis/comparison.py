"""Comparación de rendimiento normalizado entre activos (Fase 12a) — insumo
de la pestaña "Comparación" del frontend (`GET /api/compare`).

No reimplementa la carga de precios (eso lo hace `data.loaders.get_prices`,
llamado por `api/main.py`, no por acá): estas funciones reciben las series
de precio YA CARGADAS y solo alinean/normalizan/rankean. Mismo criterio de
alineación por fechas comunes que ya usaba `eda.eda_report.correlation_matrix`
(`pd.DataFrame(dict).dropna(how="any")`, inner join implícito) — no se
reimplementa esa idea, se reutiliza el mismo patrón.

HONESTIDAD: esto es una comparación de desempeño HISTÓRICO, no una
predicción — el desempeño pasado no garantiza el futuro (ver el texto que
muestra el frontend).
"""

from __future__ import annotations

import pandas as pd

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
    """Compara el rendimiento de varios activos: alinea por fechas comunes
    (`align_common_dates`), recorta a las últimas `limit` fechas comunes, y
    normaliza cada activo a base 100 DENTRO de esa ventana recortada — así
    "quién rindió mejor" se mide sobre el PERÍODO elegido (últimas `limit`
    fechas), no sobre toda la historia común disponible.

    Devuelve un dict:
    - "fechas": `pd.DatetimeIndex` de las fechas comunes usadas (recortadas).
    - "normalizado": `pd.DataFrame`, una columna por activo, cada una
      arrancando en 100.0 en la primera fecha de la ventana.
    - "rendimiento_total_pct": dict `{activo: rendimiento total del período,
      en puntos porcentuales}` — equivalente a `normalizado[activo].iloc[-1]
      - 100`, para armar un ranking.

    Si `align_common_dates` da un DataFrame vacío (sin fechas comunes en
    absoluto, p. ej. activos sin ningún solapamiento de historia), devuelve
    todo vacío en vez de lanzar una excepción — decisión de quien llama qué
    hacer con eso (ver `api/main.py::get_compare`).
    """
    aligned = align_common_dates(price_series)
    recent = aligned.tail(limit)

    if recent.empty:
        return {"fechas": recent.index, "normalizado": recent, "rendimiento_total_pct": {}}

    normalized = recent.apply(normalize_to_base, axis=0)
    rendimiento_total_pct = {asset: float(normalized[asset].iloc[-1] - DEFAULT_NORMALIZATION_BASE) for asset in normalized.columns}

    return {
        "fechas": recent.index,
        "normalizado": normalized,
        "rendimiento_total_pct": rendimiento_total_pct,
    }
