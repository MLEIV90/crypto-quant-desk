"""Análisis de ciclos de mercado REALES (Fase 15a) — reemplaza al
periodograma espectral (`analysis.statistics.spectral_periodogram`) como
insumo de "ciclos" de la vista "Ciclos y Estadística" del frontend.

POR QUÉ EL REEMPLAZO: el periodograma sobre retornos DIARIOS daba como
"ciclo dominante" períodos de 2-3 días — ruido de alta frecuencia sin
ningún significado de mercado, no un hallazgo útil. Este módulo mide ciclos
que SÍ tienen un significado reconocible para quien opera cripto: caídas
históricas (drawdowns), fases alcistas/bajistas delimitadas por una regla
clara, y los ciclos de ~4 años alrededor de los halvings de Bitcoin.

HONESTIDAD (mismo criterio que `analysis/statistics.py`, `signals/studies.py`):
nada de esto predice el próximo drawdown, la próxima fase o el próximo
ciclo — son estadísticas DESCRIPTIVAS de lo que ya pasó. En particular
`halving_cycles` opera sobre, como mucho, 4 halvings conocidos en toda la
historia de Bitcoin — un n=4 es estadísticamente insuficiente para
cualquier generalización, más allá de lo popular que sea la narrativa del
"ciclo de 4 años"; se documenta ese caveat explícitamente en el resultado.

Convención del proyecto: `close`/`returns` con índice `DatetimeIndex` UTC
tz-aware (el contrato de `data.loaders.get_prices`), `returns` en escala
DECIMAL (0.01 == 1%). Reutiliza `analysis.statistics.BTC_HALVING_DATES` (no
lo redefine) para `halving_cycles`.
"""

from __future__ import annotations

import logging

import pandas as pd

from analysis.statistics import BTC_HALVING_DATES

logger = logging.getLogger(__name__)

DEFAULT_DRAWDOWN_TOP_N: int = 10
DEFAULT_PHASE_THRESHOLD: float = 0.20
DEFAULT_MIN_PHASE_DURATION_DAYS: int = 30


# --------------------------------------------------------------------------
# Drawdowns históricos
# --------------------------------------------------------------------------


def drawdown_analysis(close: pd.Series, top_n: int = DEFAULT_DRAWDOWN_TOP_N) -> list[dict]:
    """Los `top_n` peores drawdowns (caídas pico-a-valle) históricos de `close`.

    Un "episodio" de drawdown es un tramo CONTIGUO donde el precio está por
    debajo de su máximo histórico previo ("underwater"). Se identifican
    todos los episodios de la serie completa (vectorizado vía
    `cummax`/agrupamiento por transición, sin loop día a día) y se
    devuelven los `top_n` más profundos.

    Para cada episodio:
    - "fecha_pico": el último día en que el precio marcó un máximo histórico
      antes de empezar a caer (el precio ahí es el 100% de referencia).
    - "fecha_fondo": el día del mínimo dentro de ese episodio.
    - "profundidad_pct": `(precio_fondo / precio_pico - 1) * 100` (negativo).
    - "fecha_recuperacion": el primer día, después del fondo, en que el
      precio vuelve a igualar o superar el precio del pico — `None` si
      todavía no recuperó (el episodio sigue "underwater" al final de la
      serie que se pasó).
    - "dias_caida": días corridos entre pico y fondo.
    - "dias_recuperacion": días corridos entre fondo y recuperación, o
      `None` si no recuperó.

    Ordenados por `profundidad_pct` ASCENDENTE (el más negativo primero —
    el peor drawdown de la lista).
    """
    c = close.dropna()
    if c.empty:
        return []

    running_max = c.cummax()
    drawdown = c / running_max - 1.0
    underwater = drawdown < 0.0

    # Cada vez que `underwater` cambia de valor arranca un grupo nuevo —
    # separa la serie en tramos "underwater" y tramos "en máximo" (drawdown
    # == 0) sin iterar día a día.
    episode_id = (underwater != underwater.shift(fill_value=False)).cumsum()

    episodes: list[dict] = []
    n = len(c)
    for _, mask in underwater.groupby(episode_id):
        if not bool(mask.iloc[0]):
            continue  # tramo en máximo (drawdown == 0), no es un episodio de caída

        idx = mask.index
        start_pos = c.index.get_loc(idx[0])
        end_pos = c.index.get_loc(idx[-1])

        peak_pos = start_pos - 1 if start_pos > 0 else start_pos
        peak_date = c.index[peak_pos]
        peak_price = float(c.iloc[peak_pos])

        segment = drawdown.loc[idx]
        trough_date = segment.idxmin()
        depth = float(segment.loc[trough_date])

        recovery_pos = end_pos + 1
        if recovery_pos < n:
            recovery_date = c.index[recovery_pos]
            dias_recuperacion = (recovery_date - trough_date).days
        else:
            recovery_date = None
            dias_recuperacion = None

        episodes.append(
            {
                "fecha_pico": peak_date,
                "fecha_fondo": trough_date,
                "profundidad_pct": depth * 100.0,
                "fecha_recuperacion": recovery_date,
                "dias_caida": (trough_date - peak_date).days,
                "dias_recuperacion": dias_recuperacion,
            }
        )

    episodes.sort(key=lambda e: e["profundidad_pct"])
    return episodes[:top_n]


# --------------------------------------------------------------------------
# Fases de mercado (bull/bear)
# --------------------------------------------------------------------------


def _merge_two_adjacent(a: dict, b: dict, tipo: str) -> dict:
    """Fusiona dos fases ADYACENTES y CONSECUTIVAS (`a` termina donde
    empieza `b`) en una sola, con `tipo` explícito (lo decide quien llama —
    ver `_merge_short_phases`). Recalcula "duracion_dias"/"retorno_pct" con
    los precios reales de los extremos combinados (`precio_inicio`/
    `precio_fin`, campos INTERNOS que no viajan en el dict final — ver
    `market_phases`), no solo concatena los números de `a`/`b`.
    """
    return {
        "tipo": tipo,
        "fecha_inicio": a["fecha_inicio"],
        "fecha_fin": b["fecha_fin"],
        "precio_inicio": a["precio_inicio"],
        "precio_fin": b["precio_fin"],
        "duracion_dias": (b["fecha_fin"] - a["fecha_inicio"]).days,
        "retorno_pct": (b["precio_fin"] / a["precio_inicio"] - 1.0) * 100.0 if a["precio_inicio"] != 0 else 0.0,
        # El extremo más reciente domina: si `b` todavía está "en curso",
        # la fase combinada también lo está.
        "confirmada": b["confirmada"],
    }


def _merge_short_phases(phases: list[dict], min_duration_days: int) -> list[dict]:
    """Fusiona/descarta micro-fases más cortas que `min_duration_days`
    (Fase 16a) — con el umbral de 20% de `market_phases` es común que
    salgan varias fases de 5-7 días (whipsaws) que no representan un
    movimiento de mercado significativo, solo ruido alrededor del umbral.

    Algoritmo (voraz, determinístico): mientras quede más de una fase y la
    MÁS CORTA de todas no llegue a `min_duration_days`, se la fusiona con
    su(s) vecina(s):
    - Si es la primera o la última fase: se fusiona con su único vecino,
      que ABSORBE el tramo corto (el tipo del resultado es el del vecino,
      solo se extiende su fecha de inicio o fin para cubrir también el
      tramo descartado).
    - Si es una fase intermedia: sus DOS vecinas son necesariamente del
      MISMO tipo (bull/bear alternan por construcción), así que fusionar
      las tres en una sola de ese tipo compartido es la única fusión
      posible sin inventar un tipo nuevo.

    Cada fusión recalcula duración/retorno con los precios reales de los
    extremos combinados (`_merge_two_adjacent`), no promedia ni concatena
    los números de las fases originales. Termina cuando queda una sola
    fase o cuando la más corta restante ya cumple `min_duration_days` —
    si TODA la historia es más corta que `min_duration_days`, devuelve esa
    única fase igual (no hay con qué fusionarla).
    """
    if min_duration_days <= 0:
        return phases

    merged = list(phases)
    while len(merged) > 1:
        idx = min(range(len(merged)), key=lambda i: merged[i]["duracion_dias"])
        if merged[idx]["duracion_dias"] >= min_duration_days:
            break

        if idx == 0:
            merged[1] = _merge_two_adjacent(merged[0], merged[1], tipo=merged[1]["tipo"])
            del merged[0]
        elif idx == len(merged) - 1:
            merged[idx - 1] = _merge_two_adjacent(merged[idx - 1], merged[idx], tipo=merged[idx - 1]["tipo"])
            del merged[idx]
        else:
            shared_tipo = merged[idx - 1]["tipo"]
            combined = _merge_two_adjacent(merged[idx - 1], merged[idx], tipo=shared_tipo)
            combined = _merge_two_adjacent(combined, merged[idx + 1], tipo=shared_tipo)
            merged[idx - 1] = combined
            del merged[idx : idx + 2]

    return merged


def market_phases(
    close: pd.Series,
    threshold: float = DEFAULT_PHASE_THRESHOLD,
    min_duration_days: int = DEFAULT_MIN_PHASE_DURATION_DAYS,
) -> list[dict]:
    """Fases bull/bear delimitadas por una regla clara: un mercado entra en
    BEAR cuando cae `threshold` (20% por defecto) o más desde un máximo, y
    entra en BULL cuando sube `threshold` o más desde un mínimo — la misma
    convención que usan medios financieros tradicionales para declarar un
    "bear market", aplicada acá de forma mecánica y reproducible.

    Algoritmo (una pasada, sin lookahead): se rastrea el pico/valle
    corriente; en cuanto el precio se mueve `threshold` desde ese extremo,
    se CONFIRMA la fase opuesta completa, con:
    - "fecha_inicio"/"fecha_fin": el extremo anterior (pico o valle) y el
      extremo REAL alcanzado en el movimiento que se acaba de confirmar
      (no el día en que se cruzó el 20%, sino el máximo/mínimo efectivo de
      ese tramo — un bear puede seguir cayendo bien más allá de -20% antes
      de que empiece la recuperación, y acá se refleja ese mínimo real).
    - "duracion_dias", "retorno_pct" (`(precio_fin/precio_inicio - 1)*100`).
    - "confirmada": `True` para las fases ya cerradas (el movimiento
      opuesto de `threshold` ya se confirmó).

    La ÚLTIMA fase queda "en curso" (`confirmada=False`): desde el último
    extremo confirmado hasta el último dato disponible, aunque todavía no
    haya cruzado el umbral — para no esconder el tramo más reciente. Si en
    TODA la serie nunca se cruzó `threshold` ni una vez, devuelve `[]`.

    `min_duration_days` (Fase 16a, default 30): con `threshold=0.20` es
    común que salgan varias fases de 5-7 días — un rebote o una caída que
    apenas cruzó el 20% antes de revertirse de nuevo, un "whipsaw" sin
    significado de mercado real, no un cambio de régimen. Por debajo de
    esta duración, la fase se FUSIONA con sus vecinas en vez de mostrarse
    suelta (ver `_merge_short_phases` para el algoritmo exacto) — el
    resultado final son fases más largas y más representativas, a costa de
    perder el detalle de los whipsaws que se fusionaron. `min_duration_days
    <= 0` desactiva el filtro (se devuelven las fases crudas del umbral).
    """
    c = close.dropna()
    n = len(c)
    if n < 2:
        return []

    dates = c.index
    values = c.to_numpy()

    def _phase(kind: str, start_pos: int, end_pos: int, confirmada: bool) -> dict:
        start_price, end_price = float(values[start_pos]), float(values[end_pos])
        return {
            "tipo": kind,
            "fecha_inicio": dates[start_pos],
            "fecha_fin": dates[end_pos],
            "precio_inicio": start_price,
            "precio_fin": end_price,
            "duracion_dias": (dates[end_pos] - dates[start_pos]).days,
            "retorno_pct": (end_price / start_price - 1.0) * 100.0 if start_price != 0 else 0.0,
            "confirmada": confirmada,
        }

    phases: list[dict] = []
    peak_pos = 0
    trough_pos = 0
    state: str | None = None  # None hasta el primer movimiento confirmado

    for i in range(1, n):
        if state is None:
            if values[i] > values[peak_pos]:
                peak_pos = i
            if values[i] < values[trough_pos]:
                trough_pos = i
            # Acá NO se reporta ninguna fase todavía (solo se fija la
            # dirección): `trough_pos`/`peak_pos` quedan como el ancla del
            # PRIMER tramo real, y ese tramo completo (desde esa ancla hasta
            # el próximo extremo) se reporta recién cuando el tramo
            # siguiente confirma el primer cruce de `threshold` en las
            # ramas "up"/"down" de abajo — no se pierde, solo se emite un
            # paso más tarde, con el pico/valle real ya conocido.
            if values[peak_pos] > 0 and (values[i] / values[peak_pos] - 1.0) <= -threshold:
                state = "down"
                trough_pos = i
            elif values[trough_pos] > 0 and (values[i] / values[trough_pos] - 1.0) >= threshold:
                state = "up"
                peak_pos = i
        elif state == "down":
            if values[i] < values[trough_pos]:
                trough_pos = i
            elif (values[i] / values[trough_pos] - 1.0) >= threshold:
                phases.append(_phase("bear", peak_pos, trough_pos, confirmada=True))
                state = "up"
                peak_pos = i
        elif state == "up":
            if values[i] > values[peak_pos]:
                peak_pos = i
            elif (values[i] / values[peak_pos] - 1.0) <= -threshold:
                phases.append(_phase("bull", trough_pos, peak_pos, confirmada=True))
                state = "down"
                trough_pos = i

    if state == "down":
        phases.append(_phase("bear", peak_pos, n - 1, confirmada=False))
    elif state == "up":
        phases.append(_phase("bull", trough_pos, n - 1, confirmada=False))

    phases = _merge_short_phases(phases, min_duration_days)
    for phase in phases:
        del phase["precio_inicio"]
        del phase["precio_fin"]
    return phases


# --------------------------------------------------------------------------
# Ciclos de halving (Bitcoin)
# --------------------------------------------------------------------------


def halving_cycles(close: pd.Series, halving_dates: list[str] | None = None) -> dict:
    """Segmenta `close` (se espera BTC) en los ciclos delimitados por los
    halvings conocidos (`analysis.statistics.BTC_HALVING_DATES`, reutilizado
    tal cual — hechos públicos, no un ajuste estadístico).

    CAVEAT EXPLÍCITO: Bitcoin solo tuvo 4 halvings en TODA su historia — acá
    se usan los que caen DENTRO del rango de `close` (puede ser menos de 4,
    si el histórico de precios no llega hasta los halvings más viejos). Con
    ese n tan chico, cualquier lectura de "el ciclo de 4 años se repite" es,
    en el mejor de los casos, sugestiva — no hay muestra suficiente para
    tratarlo como una regla.

    Para cada ciclo (entre un halving y el siguiente, o entre el último
    halving disponible y el final de `close` para el ciclo EN CURSO):
    - "fecha_inicio"/"fecha_fin", "en_curso" (bool, `True` para el tramo
      desde el último halving hasta hoy, todavía sin cerrar).
    - "duracion_dias", "retorno_pct" (precio fin vs. precio inicio).
    - "drawdown_maximo_pct": la peor caída pico-a-valle DENTRO de ese ciclo
      (no reutiliza `drawdown_analysis` — acá alcanza con el mínimo de
      `precio / cummax_del_tramo - 1`, un solo número por ciclo).

    Devuelve un dict: "ciclos" (lista, en orden cronológico),
    "n_halvings_totales" (4, la cantidad real conocida),
    "n_halvings_con_datos" (cuántos de esos caen dentro de `close`).
    """
    if halving_dates is None:
        halving_dates = BTC_HALVING_DATES

    c = close.dropna().sort_index()
    n_halvings_totales = len(halving_dates)

    if c.empty:
        return {"ciclos": [], "n_halvings_totales": n_halvings_totales, "n_halvings_con_datos": 0}

    halving_ts = [pd.Timestamp(d, tz=c.index.tz) for d in halving_dates]
    halvings_in_range = sorted(ts for ts in halving_ts if c.index.min() <= ts <= c.index.max())

    ciclos: list[dict] = []
    for i, start_date in enumerate(halvings_in_range):
        end_date = halvings_in_range[i + 1] if i + 1 < len(halvings_in_range) else c.index.max()
        en_curso = i + 1 >= len(halvings_in_range)

        segment = c.loc[start_date:end_date]
        if len(segment) < 2:
            continue

        start_price, end_price = float(segment.iloc[0]), float(segment.iloc[-1])
        segment_drawdown = segment / segment.cummax() - 1.0

        ciclos.append(
            {
                "fecha_inicio": segment.index[0],
                "fecha_fin": segment.index[-1],
                "en_curso": en_curso,
                "duracion_dias": (segment.index[-1] - segment.index[0]).days,
                "retorno_pct": (end_price / start_price - 1.0) * 100.0 if start_price != 0 else 0.0,
                "drawdown_maximo_pct": float(segment_drawdown.min()) * 100.0,
            }
        )

    return {
        "ciclos": ciclos,
        "n_halvings_totales": n_halvings_totales,
        "n_halvings_con_datos": len(halvings_in_range),
    }


# --------------------------------------------------------------------------
# Heatmap mes x año
# --------------------------------------------------------------------------


def monthly_yearly_heatmap(returns: pd.Series) -> dict:
    """Retorno COMPUESTO de cada mes de cada año (matriz mes x año), para
    ver la estacionalidad real año por año — a diferencia de
    `analysis.statistics.monthly_seasonality` (un único promedio agregado
    de todos los años juntos), acá cada celda es un mes-año concreto, sin
    promediar entre años.

    El retorno de un mes es el COMPUESTO de los retornos diarios/horarios
    de ese mes (`(1 + r).prod() - 1`), no la suma simple — la convención
    estándar para acumular retornos porcentuales en un período.

    Devuelve un dict:
    - "anios": lista de años (ascendente) presentes en `returns`.
    - "matriz": lista de 12 listas (una por mes, índice 0 = enero .. 11 =
      diciembre), cada una con un valor por año en "anios" (en el mismo
      orden) — `retorno_pct` (ya multiplicado por 100) o `None` si ese
      mes-año no tiene ninguna observación (años parciales, historia que
      no llega hasta ahí, etc.).
    """
    r = returns.dropna()
    if r.empty:
        return {"anios": [], "matriz": [[] for _ in range(12)]}

    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    years = sorted({idx[0] for idx in monthly.index})

    matriz: list[list[float | None]] = []
    for month in range(1, 13):
        row: list[float | None] = []
        for year in years:
            key = (year, month)
            row.append(float(monthly.loc[key]) * 100.0 if key in monthly.index else None)
        matriz.append(row)

    return {"anios": years, "matriz": matriz}
