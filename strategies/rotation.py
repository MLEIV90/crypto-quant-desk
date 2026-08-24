"""Backtest de ROTACIÓN por MOMENTUM RELATIVO entre dos monedas (Fase 19a).

IDEA: rotar entre dos monedas según cuál viene rindiendo mejor
recientemente (momentum) — estar 100% en la que tuvo mayor retorno en los
últimos `lookback_days`, reconsiderando solo cada `rebalance_days`. Es
MOMENTUM, no arbitraje/reversión: apuesta a que la tendencia RELATIVA entre
las dos monedas continúa, lo opuesto de apostar a que un spread revierte a
su media (ver `pairs/` para esa otra familia de estrategias, y notar que
ambas conviven en este proyecto sin contradecirse: son apuestas distintas
sobre fenómenos distintos).

ESTE ES EL TEST PREVIO AL ML (ver `scripts/run_rotation_experiment.py`): si
esta regla simple no le gana de forma robusta a quedarse quieto, un modelo
de Machine Learning más complejo sobre la misma idea NO la va a "rescatar"
— mismo criterio anti-hype que ya aplican `ml/models.py`,
`signals/suggester.py` y el experimento de RL de `rl/`. El resultado
esperado, honestamente, es que la rotación NO le gane de forma consistente
a los baselines después de costos — momentum cruzado entre dos activos que
ya están altamente correlacionados (ver la vista "Correlación") tiene poco
margen para generar una señal que sobreviva al costo de rotar.

Reutiliza `data.loaders.get_prices`, `analysis.comparison.align_common_dates`,
`signals.returns.simple_returns` y `metrics.risk_measures` (Sharpe, retorno
anualizado, máximo drawdown, equity curve) tal cual — lo único nuevo acá es
la lógica de DECISIÓN (qué moneda mantener) y la combinación de retornos de
dos series según esa decisión, que no existía en una forma reutilizable en
ningún otro módulo del proyecto (`backtest.engine` está pensado para UNA
posición sobre UN activo, no para elegir entre dos).

RESTRICCIÓN CLAVE — LONG-ONLY, SIN DEUDA: la estrategia SIEMPRE está
comprada en UNA sola moneda (o en efectivo, durante el warmup) con capital
propio. NUNCA vende en corto, NUNCA usa apalancamiento, NUNCA pide
prestado — "cambiar de moneda" acá significa literalmente vender el 100%
de lo que se tiene y comprar la otra, no abrir una posición corta en la que
se vende. El único riesgo de la estrategia es el mismo que el de comprar y
mantener: que la moneda que se tiene en cartera baje de precio. En términos
de pesos de cartera (`holdings_to_weights`), esto se traduce en una
invariante ESTRICTA que se verifica en los tests (`tests/test_rotation.py`):
todo peso es `>= 0` (nunca corto) y la suma de pesos en cada fecha es
`<= 1` (nunca apalancado — de hecho, siempre exactamente 1 o 0 acá, porque
no hay una posición "parcial" entre las dos monedas, solo 100% en una o
100% en la otra).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.comparison import align_common_dates
from config import TRANSACTION_COST_BPS, UNIVERSE
from data.loaders import get_prices
from metrics.risk_measures import annualized_return, equity_curve, max_drawdown, sharpe_ratio
from signals.returns import simple_returns

logger = logging.getLogger(__name__)

# Grilla de parámetros sugerida por la Fase 19a: lookback en días de
# calendario (~1 mes, ~2 meses, ~3 meses de momentum) x rebalanceo semanal
# o mensual. `run_rotation_experiment` (rl/... perdón, strategies/) las usa
# como default, pero cualquier combinación puede pasarse a mano.
DEFAULT_LOOKBACK_GRID: tuple[int, ...] = (20, 50, 100)
DEFAULT_REBALANCE_GRID: tuple[int, ...] = (7, 30)


def _forward_returns(close: pd.Series) -> pd.Series:
    """Retorno simple realizado ENTRE el cierre de t y el cierre de t+1 (la
    fila t queda con "lo que pasa DESPUÉS de t", no con lo que ya se conocía
    al decidir en t) — misma convención anti-lookahead que
    `rl.features.build_portfolio_dataset`.
    """
    return simple_returns(close).shift(-1)


def _decide_holdings(
    mom_a: np.ndarray, mom_b: np.ndarray, rebalance_days: int, label_a: str, label_b: str
) -> np.ndarray:
    """Decide qué activo mantener en cada fecha (array de `label_a`/`label_b`/`None`).

    En las fechas de REBALANCEO (cada `rebalance_days` pasos, contados desde
    la primera fecha con momentum ya disponible) se compara `mom_a` vs
    `mom_b` y se fija esa elección — empate exacto (prácticamente imposible
    con precios reales, pero definido de todos modos) gana `label_b`, una
    convención determinística sin intención de favorecer ningún activo. En
    el resto de las fechas se MANTIENE la elección del último rebalanceo:
    la esencia de "estar en la que rinde mejor HASTA el próximo
    rebalanceo", no perseguir el momentum día a día.

    Antes de que haya suficiente historia para calcular momentum (NaN en
    `mom_a`/`mom_b`, el warmup de `lookback_days`), la elección queda en
    `None`.
    """
    n = len(mom_a)
    choice: np.ndarray = np.full(n, None, dtype=object)
    first_valid: int | None = None
    for i in range(n):
        if np.isnan(mom_a[i]) or np.isnan(mom_b[i]):
            continue
        if first_valid is None:
            first_valid = i
        if (i - first_valid) % rebalance_days == 0:
            choice[i] = label_a if mom_a[i] > mom_b[i] else label_b
        else:
            choice[i] = choice[i - 1]
    return choice


def holdings_to_weights(holdings: pd.Series, asset_a: str, asset_b: str) -> pd.DataFrame:
    """Convierte la serie categórica `holdings` (valores `asset_a`/`asset_b`,
    o `None` durante el warmup) en pesos de cartera EXPLÍCITOS — una
    columna por activo, con efectivo implícito como `1 - suma` de esas dos.

    Existe para poder VERIFICAR la restricción long-only/sin-deuda del
    módulo (ver el docstring del módulo) en un formato directamente
    chequeable: cada fila tiene, por construcción, un peso en `{0.0, 1.0}`
    por columna, nunca negativo, y la suma de las dos columnas nunca supera
    1 (es exactamente 1 fuera del warmup, 0 durante el warmup — 100%
    efectivo, ninguna moneda todavía). No se usa para el cálculo de
    retorno/costos de `momentum_rotation_backtest` (que opera directo sobre
    la serie categórica, más simple para ese propósito) — es una vista
    alternativa para tests/inspección.
    """
    weight_a = (holdings == asset_a).astype(float)
    weight_b = (holdings == asset_b).astype(float)
    return pd.DataFrame({asset_a: weight_a, asset_b: weight_b}, index=holdings.index)


def _metrics_from_returns(returns_series: pd.Series) -> dict:
    equity = equity_curve(returns_series)
    return {
        "cagr": annualized_return(returns_series),
        "sharpe": sharpe_ratio(returns_series),
        "max_drawdown": max_drawdown(returns_series),
        "retorno_total": float(equity.iloc[-1] - 1.0),
    }


@dataclass
class RotationResult:
    asset_a: str
    asset_b: str
    lookback_days: int
    rebalance_days: int
    cost_bps: float
    equity_curve: pd.Series
    returns: pd.Series
    holdings: pd.Series
    metrics: dict


def momentum_rotation_backtest(
    asset_a: str,
    asset_b: str,
    lookback_days: int,
    rebalance_days: int,
    cost_bps: float | None = None,
    source: str = "store",
    interval: str = "1d",
) -> RotationResult:
    """Backtestea la rotación por momentum relativo entre `asset_a` y
    `asset_b`.

    En cada fecha de rebalanceo, calcula el momentum de cada moneda como el
    retorno de los últimos `lookback_days` (`close.pct_change(lookback_days)`,
    TRAILING/causal: usa solo `close` hasta esa fecha) y mantiene el 100%
    de la cartera en la que tuvo mayor momentum hasta el PRÓXIMO
    rebalanceo, sin reconsiderar en el medio (`_decide_holdings`).

    ANTI-LOOKAHEAD: la decisión de la fecha t usa `close` hasta t; el
    retorno que la puntúa es el REALIZADO entre t y t+1 (`_forward_returns`).
    Los costos de transacción se cobran completos (`cost_bps/1e4` sobre un
    turnover de 1.0 "one-way" — mover el 100% de la cartera de una moneda a
    otra, ver `config.TURNOVER_CONVENTION`) en cada fecha en que la moneda
    elegida CAMBIA respecto del día anterior, incluida la primera fecha
    utilizable (se entra desde 100% efectivo, sin posición heredada de
    fuera de la muestra — misma convención que `backtest.engine.run_backtest`).

    Devuelve un `RotationResult` con la curva de equity, los retornos netos
    diarios, la serie de qué activo se mantenía cada día (`holdings`), y un
    dict de métricas: "cagr", "sharpe", "max_drawdown", "retorno_total",
    "n_rotaciones" (cuenta la entrada inicial), "pct_tiempo_a"/"pct_tiempo_b"
    (fracción de días post-warmup en cada activo).

    Lanza `ValueError` si no hay suficiente historia común para al menos un
    período de warmup (`lookback_days`) más un día de retorno.
    """
    if cost_bps is None:
        cost_bps = TRANSACTION_COST_BPS

    close_a = get_prices(asset_a, source=source, interval=interval, use_cache=False)["close"]
    close_b = get_prices(asset_b, source=source, interval=interval, use_cache=False)["close"]
    aligned = align_common_dates({asset_a: close_a, asset_b: close_b})
    if len(aligned) <= lookback_days + 1:
        raise ValueError(
            f"momentum_rotation_backtest: solo {len(aligned)} fechas comunes entre '{asset_a}'/'{asset_b}', "
            f"no alcanza para lookback_days={lookback_days}"
        )

    mom_a = aligned[asset_a].pct_change(periods=lookback_days).to_numpy()
    mom_b = aligned[asset_b].pct_change(periods=lookback_days).to_numpy()
    holdings_raw = _decide_holdings(mom_a, mom_b, rebalance_days, asset_a, asset_b)

    forward_a = _forward_returns(aligned[asset_a]).to_numpy()
    forward_b = _forward_returns(aligned[asset_b]).to_numpy()

    df = pd.DataFrame(
        {"holding": holdings_raw, "forward_a": forward_a, "forward_b": forward_b}, index=aligned.index
    )
    df = df[df["holding"].notna()]  # descarta el warmup (sin momentum todavía)
    df = df.dropna(subset=["forward_a", "forward_b"])  # descarta el último día (sin retorno siguiente)

    holding = df["holding"].to_numpy()
    gross_return = np.where(holding == asset_a, df["forward_a"].to_numpy(), df["forward_b"].to_numpy())

    prev_holding = np.empty(len(holding), dtype=object)
    prev_holding[0] = None  # sin posición previa: 100% efectivo antes del primer día utilizable
    prev_holding[1:] = holding[:-1]
    is_rotation = holding != prev_holding

    net_return = gross_return - is_rotation.astype(float) * (cost_bps / 1e4)
    returns_series = pd.Series(net_return, index=df.index, name="rotation_return")

    metrics = _metrics_from_returns(returns_series)
    metrics["n_rotaciones"] = int(is_rotation.sum())
    metrics["pct_tiempo_a"] = float((holding == asset_a).mean())
    metrics["pct_tiempo_b"] = float((holding == asset_b).mean())

    logger.info(
        "momentum_rotation_backtest(%s/%s, lookback=%d, rebalance=%d): sharpe=%.2f, cagr=%.1f%%, "
        "n_rotaciones=%d, %%tiempo_%s=%.0f%%",
        asset_a, asset_b, lookback_days, rebalance_days, metrics["sharpe"], metrics["cagr"] * 100,
        metrics["n_rotaciones"], asset_a, metrics["pct_tiempo_a"] * 100,
    )

    return RotationResult(
        asset_a=asset_a,
        asset_b=asset_b,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        cost_bps=cost_bps,
        equity_curve=equity_curve(returns_series),
        returns=returns_series,
        holdings=df["holding"],
        metrics=metrics,
    )


def _fifty_fifty_returns(forward_a: np.ndarray, forward_b: np.ndarray, cost_bps: float) -> np.ndarray:
    """Retorno neto de un portafolio 50/50 REBALANCEADO A DIARIO entre dos
    activos: cada período arranca en pesos exactos 0.5/0.5 (recién
    rebalanceados), se realiza el retorno del período, y el DRIFT resultante
    (los pesos ya no están exactos a 0.5/0.5 porque los dos activos
    rindieron distinto ese día) se corrige con una operación de rebalanceo
    ANTES del período siguiente — ESE rebalanceo es lo que genera turnover
    real cada día, a diferencia de simplemente mantener un peso objetivo
    constante sin ningún drift intermedio que corregir (lo que daría
    turnover cero después de la entrada inicial, y subestimaría el costo
    real de "rebalancear todos los días").
    """
    n = len(forward_a)
    net = np.empty(n, dtype=np.float64)
    weight_a_drifted = 0.5

    for t in range(n):
        turnover = 1.0 if t == 0 else abs(weight_a_drifted - 0.5)
        portfolio_return = 0.5 * forward_a[t] + 0.5 * forward_b[t]
        net[t] = portfolio_return - turnover * (cost_bps / 1e4)

        value_a = 0.5 * (1.0 + forward_a[t])
        value_b = 0.5 * (1.0 + forward_b[t])
        total = value_a + value_b
        weight_a_drifted = value_a / total if total > 0 else 0.5

    return net


def baseline_backtests(
    asset_a: str,
    asset_b: str,
    dates: pd.DatetimeIndex,
    cost_bps: float | None = None,
    source: str = "store",
    interval: str = "1d",
) -> dict[str, dict]:
    """Los 3 baselines de comparación (100% `asset_a` siempre, 100% `asset_b`
    siempre, 50/50 rebalanceado a diario), evaluados sobre el MISMO rango
    `dates` que una corrida de `momentum_rotation_backtest` (típicamente su
    `returns.index`, el tramo post-warmup) y el MISMO `cost_bps` — para que
    la comparación contra la rotación sea de verdad "mismo período, mismos
    costos".

    Devuelve `{"buy_hold_<asset_a>": {...}, "buy_hold_<asset_b>": {...},
    "50_50_rebalanceado": {...}}`, cada uno con las claves "cagr", "sharpe",
    "max_drawdown", "retorno_total" (mismas que `momentum_rotation_backtest`,
    salvo "n_rotaciones"/"pct_tiempo_*", que no aplican: estos baselines no
    "rotan" entre monedas).
    """
    if cost_bps is None:
        cost_bps = TRANSACTION_COST_BPS

    close_a = get_prices(asset_a, source=source, interval=interval, use_cache=False)["close"]
    close_b = get_prices(asset_b, source=source, interval=interval, use_cache=False)["close"]
    aligned = align_common_dates({asset_a: close_a, asset_b: close_b})

    forward_a = _forward_returns(aligned[asset_a]).reindex(dates).to_numpy()
    forward_b = _forward_returns(aligned[asset_b]).reindex(dates).to_numpy()

    is_first = np.zeros(len(dates), dtype=bool)
    is_first[0] = True
    entry_cost = is_first.astype(float) * (cost_bps / 1e4)

    results: dict[str, dict] = {
        f"buy_hold_{asset_a}": _metrics_from_returns(pd.Series(forward_a - entry_cost, index=dates)),
        f"buy_hold_{asset_b}": _metrics_from_returns(pd.Series(forward_b - entry_cost, index=dates)),
        "50_50_rebalanceado": _metrics_from_returns(
            pd.Series(_fifty_fifty_returns(forward_a, forward_b, cost_bps), index=dates)
        ),
    }
    return results


# --------------------------------------------------------------------------
# Validación HONESTA (Fase 19a): correr TODA la grilla de parámetros, para
# TODOS los pares, y reportarla ENTERA — no cherry-pickear la combinación
# que mejor le fue. Ver `scripts/run_rotation_experiment.py` para el punto
# de entrada de línea de comandos.
# --------------------------------------------------------------------------

# Todos los pares posibles del universo del proyecto (10 = C(5,2)) — "para
# ver si es robusto o casualidad" (ver el pedido de la Fase 19a): probar
# más allá de ETH-BTC es lo que permite distinguir un patrón genuino de una
# combinación que funcionó por azar en un solo par.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = tuple(itertools.combinations(UNIVERSE, 2))


@dataclass
class PairGridResult:
    """Resultado de correr TODA la grilla de (lookback, rebalance) para un
    par. `rows` tiene una fila por combinación (nunca se descarta ninguna).
    `robusto` es `True` solo si la rotación superó al MEJOR de los 3
    baselines (por Sharpe) en TODAS las combinaciones — un criterio
    deliberadamente estricto: si ganó en algunas y perdió en otras, no es
    una estrategia robusta, es una combinación con suerte (ver el docstring
    del módulo).
    """

    asset_a: str
    asset_b: str
    rows: list[dict] = field(default_factory=list)
    robusto: bool = False


def run_pair_grid(
    asset_a: str,
    asset_b: str,
    lookback_grid: tuple[int, ...] = DEFAULT_LOOKBACK_GRID,
    rebalance_grid: tuple[int, ...] = DEFAULT_REBALANCE_GRID,
    cost_bps: float | None = None,
) -> PairGridResult:
    """Corre TODAS las combinaciones de `lookback_grid` x `rebalance_grid`
    para el par (`asset_a`, `asset_b`) contra sus 3 baselines, y arma una
    fila de resultado por combinación.

    Los baselines se recalculan una sola vez POR VALOR DE `lookback`
    (determina el rango de fechas post-warmup; `rebalance_days` no cambia
    ese rango, solo cada cuánto se reconsidera la posición) y se reutilizan
    entre los distintos `rebalance_days` de ese mismo lookback — no vuelve
    a llamar a `baseline_backtests` innecesariamente.
    """
    rows: list[dict] = []
    baselines_by_lookback: dict[int, dict] = {}

    for lookback in lookback_grid:
        for rebalance in rebalance_grid:
            rotation = momentum_rotation_backtest(asset_a, asset_b, lookback, rebalance, cost_bps=cost_bps)

            if lookback not in baselines_by_lookback:
                baselines_by_lookback[lookback] = baseline_backtests(
                    asset_a, asset_b, rotation.returns.index, cost_bps=cost_bps
                )
            baselines = baselines_by_lookback[lookback]
            best_baseline_name = max(baselines, key=lambda name: baselines[name]["sharpe"])
            best_baseline_sharpe = baselines[best_baseline_name]["sharpe"]

            rows.append(
                {
                    "par": f"{asset_a}-{asset_b}",
                    "lookback_days": lookback,
                    "rebalance_days": rebalance,
                    "sharpe_rotacion": rotation.metrics["sharpe"],
                    "cagr_rotacion": rotation.metrics["cagr"],
                    "retorno_total_rotacion": rotation.metrics["retorno_total"],
                    "max_drawdown_rotacion": rotation.metrics["max_drawdown"],
                    "n_rotaciones": rotation.metrics["n_rotaciones"],
                    "mejor_baseline": best_baseline_name,
                    "sharpe_mejor_baseline": best_baseline_sharpe,
                    "gana_al_mejor_baseline": rotation.metrics["sharpe"] > best_baseline_sharpe,
                    **{f"sharpe_{name}": metrics["sharpe"] for name, metrics in baselines.items()},
                }
            )

    robusto = all(row["gana_al_mejor_baseline"] for row in rows)
    return PairGridResult(asset_a=asset_a, asset_b=asset_b, rows=rows, robusto=robusto)


def run_full_experiment(
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
    lookback_grid: tuple[int, ...] = DEFAULT_LOOKBACK_GRID,
    rebalance_grid: tuple[int, ...] = DEFAULT_REBALANCE_GRID,
    cost_bps: float | None = None,
) -> list[PairGridResult]:
    """Corre `run_pair_grid` para cada par de `pairs` (default: los 10 pares
    posibles del universo del proyecto). Devuelve la lista de resultados,
    uno por par — `summarize_experiment`/`rotation_beats_baselines_robustly`
    la convierten en una tabla y una conclusión, respectivamente.
    """
    results: list[PairGridResult] = []
    for asset_a, asset_b in pairs:
        logger.info("run_full_experiment: par %s-%s", asset_a, asset_b)
        results.append(run_pair_grid(asset_a, asset_b, lookback_grid, rebalance_grid, cost_bps=cost_bps))
    return results


def summarize_experiment(pair_results: list[PairGridResult]) -> pd.DataFrame:
    """Aplana todos los `PairGridResult.rows` de todos los pares en una
    única tabla — TODAS las combinaciones de TODOS los pares, sin
    cherry-pickear ninguna fila (ver el docstring del módulo).
    """
    rows = [row for result in pair_results for row in result.rows]
    return pd.DataFrame(rows)


def rotation_beats_baselines_robustly(pair_results: list[PairGridResult], primary_pair: tuple[str, str]) -> dict:
    """Conclusión automática y HONESTA: ¿la rotación por momentum supera a
    quedarse quieto, después de costos, de forma ROBUSTA entre parámetros?

    Se distingue el par PRINCIPAL (`primary_pair`, típicamente ETH-BTC, el
    pedido original de la Fase 19a) del resto del universo (evidencia
    adicional de si el efecto es genuino o una casualidad de un solo par):

    - "robusto_par_principal": `True` solo si TODAS las combinaciones de
      parámetros del par principal ganaron a su mejor baseline.
    - "fraccion_pares_robustos": de TODOS los pares corridos (incluido el
      principal), qué fracción fue robusta en ese mismo sentido.
    - "veredicto_global": `True` solo si el par principal es robusto Y al
      menos la MITAD del resto de los pares también lo es — exigir esto
      además del par principal es lo que distingue "ETH-BTC específicamente
      tiene esta propiedad" de "el momentum cruzado funciona de verdad
      entre cripto en general". Un umbral estricto a propósito: con un
      universo de solo 10 pares y una expectativa de base de que la
      estrategia NO funcione, cualquier cosa por debajo de la mitad es
      indistinguible de azar.
    """
    # Comparación por conjunto, no por tupla ordenada: `DEFAULT_PAIRS` genera
    # cada par en el orden de `config.UNIVERSE` (p. ej. ("BTC", "ETH")), que
    # puede no coincidir con el orden en que se pide `primary_pair` (p. ej.
    # ("ETH", "BTC")) — la rotación es simétrica en qué activo se llama "a"
    # o "b", así que el orden no debería importar para identificar el par.
    primary_set = frozenset(primary_pair)
    primary_result = next(
        (r for r in pair_results if frozenset((r.asset_a, r.asset_b)) == primary_set), None
    )
    if primary_result is None:
        raise ValueError(
            f"rotation_beats_baselines_robustly: no se corrió el par principal {primary_pair[0]}-{primary_pair[1]}"
        )
    primary_key = f"{primary_result.asset_a}-{primary_result.asset_b}"

    n_robust = sum(1 for r in pair_results if r.robusto)
    fraccion_robustos = n_robust / len(pair_results)

    return {
        "robusto_par_principal": primary_result.robusto,
        "par_principal": primary_key,
        "fraccion_pares_robustos": fraccion_robustos,
        "pares_robustos": [f"{r.asset_a}-{r.asset_b}" for r in pair_results if r.robusto],
        "veredicto_global": primary_result.robusto and fraccion_robustos >= 0.5,
    }
