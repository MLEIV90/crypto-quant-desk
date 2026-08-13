"""Etiquetado triple-barrier (López de Prado, "Advances in Financial Machine
Learning", caps. 2-3) para el pipeline de ML de crypto-quant-desk.

Fase 3a: SOLO etiquetado y pesos por muestra. No se entrena ningún modelo
acá (eso es Fase 3c) ni se implementa la validación walk-forward (Fase 3b).

Convención del proyecto: `close` en escala de precio (no retornos), los
retornos internos se calculan en escala DECIMAL/SIMPLE
(`signals.returns.simple_returns`), consistente con el resto del repo.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from signals.returns import simple_returns

logger = logging.getLogger(__name__)

DEFAULT_VOLATILITY_SPAN: int = 100
DEFAULT_PT_MULT: float = 2.0
DEFAULT_SL_MULT: float = 2.0
DEFAULT_MAX_HOLDING: int = 10


def get_daily_volatility(close: pd.Series, span: int = DEFAULT_VOLATILITY_SPAN) -> pd.Series:
    """Volatilidad diaria estimada con EWMA de retornos simples (span=100
    por defecto, convención habitual en AFML cap. 3).

    Se usa para escalar las barreras de `triple_barrier_labels` por la
    volatilidad LOCAL de cada fecha, en vez de un umbral fijo que no se
    adapta a los cambios de régimen del activo (un movimiento del 5% es
    "normal" en un período de alta volatilidad y "enorme" en uno de calma).

    `Series.ewm(span=span).std()` es trailing por construcción (solo usa
    retornos hasta la fecha t para el valor en t) — sin lookahead. El primer
    valor queda en NaN (con una sola observación, el desvío no está
    definido).
    """
    returns = simple_returns(close)
    return returns.ewm(span=span).std()


def triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    pt_mult: float = DEFAULT_PT_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    max_holding: int = DEFAULT_MAX_HOLDING,
) -> pd.DataFrame:
    """Etiquetado triple-barrier: para cada fecha t (posible entrada LARGA —
    este etiquetado asume una posición compradora, no evalúa el lado corto),
    se definen tres barreras:

        take-profit = close_t * (1 + pt_mult * volatility_t)
        stop-loss   = close_t * (1 - sl_mult * volatility_t)
        vertical    = t + max_holding días

    Se recorre el futuro de t (SOLO dentro de la ventana `max_holding`) día
    a día y se etiqueta según cuál barrera se toca PRIMERO (comparando
    contra el CLOSE de cada día; con datos diarios no hay forma de saber el
    orden intradía si dos barreras se cruzan el mismo día — en ese caso
    empate se resuelve a favor de take-profit, ver el código).

    IMPORTANTE — ESTO MIRA EL FUTURO POR CONSTRUCCIÓN: es la etiqueta (la
    "y" del problema supervisado), nunca debe usarse como feature/input de
    un modelo — para eso está `ml.features`, que es estrictamente trailing.
    Por la misma razón, las últimas `max_holding` filas de la serie no
    tienen ventana futura completa para evaluar la barrera vertical y
    quedan con `label` en NaN — no se inventa una etiqueta con información
    incompleta.

    Devuelve un `pd.DataFrame` indexado igual que `close`, columnas:
    - "label": +1 (take-profit primero), -1 (stop-loss primero), 0 (vence
      el tiempo sin tocar ninguna barrera de precio), NaN (sin ventana
      futura suficiente, o sin volatilidad válida ese día).
    - "barrera_tocada": "take_profit" / "stop_loss" / "vertical" / None.
    - "dias_hasta_evento": cantidad de días desde t hasta que se resuelve
      el evento (entre 1 y `max_holding`).
    - "retorno_realizado": retorno simple desde close_t hasta el precio del
      día en que se resuelve el evento (el retorno que habría dado la
      operación).
    """
    close = close.astype(float)
    vol = volatility.reindex(close.index)

    n = len(close)
    closes = close.to_numpy()
    vols = vol.to_numpy()

    labels = np.full(n, np.nan)
    barriers: list[str | None] = [None] * n
    days_to_event = np.full(n, np.nan)
    realized_returns = np.full(n, np.nan)

    for t in range(n):
        if t + max_holding >= n:
            continue  # sin ventana futura completa: no se etiqueta (queda NaN)
        if np.isnan(vols[t]) or vols[t] <= 0:
            continue  # sin volatilidad válida (warmup de get_daily_volatility)

        entry_price = closes[t]
        take_profit_price = entry_price * (1.0 + pt_mult * vols[t])
        stop_loss_price = entry_price * (1.0 - sl_mult * vols[t])

        label = 0.0
        barrier = "vertical"
        event_offset = max_holding

        for offset in range(1, max_holding + 1):
            future_price = closes[t + offset]
            if future_price >= take_profit_price:
                label, barrier, event_offset = 1.0, "take_profit", offset
                break
            if future_price <= stop_loss_price:
                label, barrier, event_offset = -1.0, "stop_loss", offset
                break

        labels[t] = label
        barriers[t] = barrier
        days_to_event[t] = event_offset
        realized_returns[t] = closes[t + event_offset] / entry_price - 1.0

    return pd.DataFrame(
        {
            "label": labels,
            "barrera_tocada": barriers,
            "dias_hasta_evento": days_to_event,
            "retorno_realizado": realized_returns,
        },
        index=close.index,
    )


def get_sample_weights(labels_df: pd.DataFrame) -> pd.Series:
    """Pesos por SOLAPAMIENTO (concurrencia), siguiendo la idea de "unicidad
    promedio" de López de Prado (AFML cap. 4): el horizonte de una etiqueta
    en t es el rango [t, t + dias_hasta_evento_t]. Dos etiquetas cuyos
    horizontes se superponen comparten los mismos movimientos de precio del
    activo durante el solapamiento — no son observaciones independientes.
    Tratarlas como si lo fueran sobre-representa los períodos con muchos
    trades solapados (el modelo vería, en efecto, "más" de esos períodos
    que de los tramos con trades más espaciados) y subestima cuánta
    información realmente nueva aporta cada muestra.

    Para cada fecha t con etiqueta válida, se cuenta cuántos horizontes
    (incluido el propio) cubren cada punto del tiempo dentro de [t, t+dias]
    (la "concurrencia" c). El peso de la etiqueta en t es el promedio de
    1/c a lo largo de TODO su propio horizonte (su unicidad promedio): un
    trade cuyo horizonte se solapa mucho con otros pesa cerca de 0; uno
    aislado (nadie más lo cubre) pesa 1.

    Los pesos NO se normalizan a sumar 1 acá (queda para quien los consuma
    en Fase 3c, según lo que necesite el modelo/librería de turno).

    Devuelve una `pd.Series` indexada igual que `labels_df`, con NaN donde
    no hay etiqueta válida (`label` o `dias_hasta_evento` en NaN).
    """
    valid = labels_df.dropna(subset=["label", "dias_hasta_evento"])
    n = len(labels_df)
    position_of = {idx: pos for pos, idx in enumerate(labels_df.index)}

    concurrency = np.zeros(n)
    spans: list[tuple[int, int]] = []
    for idx, dias in valid["dias_hasta_evento"].items():
        start = position_of[idx]
        end = min(start + int(dias), n - 1)
        spans.append((start, end))
        concurrency[start : end + 1] += 1.0

    weights = pd.Series(np.nan, index=labels_df.index, dtype=float)
    for (start, end), idx in zip(spans, valid.index):
        weights[idx] = float(np.mean(1.0 / concurrency[start : end + 1]))

    return weights
