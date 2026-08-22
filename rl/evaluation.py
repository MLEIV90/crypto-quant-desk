"""Evaluación de pesos de cartera + validación walk-forward + orquestador
del experimento completo de Fase 18 (`rl/`).

Ver `rl/__init__.py` para el encuadre honesto del experimento.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import TRANSACTION_COST_BPS
from data.loaders import get_prices
from metrics.risk_measures import annualized_return, max_drawdown, sharpe_ratio
from rl.baselines import (
    buy_and_hold_btc_positions,
    equal_weight_positions,
    random_positions,
    vol_targeting_positions,
)
from rl.features import DEFAULT_ASSETS, PortfolioDataset, build_portfolio_dataset

logger = logging.getLogger(__name__)

METRIC_KEYS: tuple[str, ...] = (
    "sharpe", "retorno_anualizado", "retorno_total", "max_drawdown", "turnover_total", "turnover_medio_diario",
)


# --------------------------------------------------------------------------
# Evaluación de una secuencia de pesos (compartida por el agente RL y los 4
# baselines — la MISMA fórmula para todos, ver el docstring de abajo).
# --------------------------------------------------------------------------


def evaluate_weights(weights: np.ndarray, asset_returns: np.ndarray, cost_bps: float | None = None) -> dict:
    """Evalúa una secuencia de pesos de cartera `(T, N_ASSETS)` contra los
    retornos realizados `(T, N_ASSETS)` YA ALINEADOS fila a fila: la fila t
    de `weights` es la decisión tomada con información hasta t; la fila t
    de `asset_returns` es el retorno YA REALIZADO de t a t+1 (misma
    convención que `rl.features.build_portfolio_dataset`/`rl.env.PortfolioEnv`)
    — así el agente RL y los 4 baselines se evalúan con la IDÉNTICA
    fórmula, sin ninguna diferencia sutil entre "cómo entrena el agente" y
    "cómo se lo compara".

        turnover_t      = 0.5 * sum(|pesos_t - pesos_{t-1}|)   (one-way, pesos_{-1}
                                                                 = 100% efectivo,
                                                                 misma convención
                                                                 que backtest.engine.run_backtest)
        retorno_neto_t   = pesos_t . retornos_t - (cost_bps/1e4) * turnover_t

    El factor 0.5 es necesario porque acá el efectivo es una columna MÁS
    del vector de pesos (a diferencia de `backtest.engine.run_backtest`,
    donde una posición de un solo activo en [-1,1] deja el efectivo
    IMPLÍCITO): mover 100% de cartera de efectivo a un activo cambia DOS
    columnas a la vez (efectivo -1, activo +1), así que sumar los valores
    absolutos de TODOS los cambios de peso cuenta ambas piernas del mismo
    movimiento — la convención "one_way" de `config.TURNOVER_CONVENTION`
    (contar un solo lado de cada rebalanceo) exige dividir esa suma por 2.

    Reutiliza `metrics.risk_measures` (Sharpe, retorno anualizado, máximo
    drawdown) sobre la serie de retornos netos resultante — no reimplementa
    ninguna métrica de riesgo/performance, solo la combinación de retorno de
    cartera MULTI-activo (que no existe en forma reutilizable en
    `backtest.engine`, pensado para un solo activo con una posición
    escalar, no un vector de pesos que suma 1).

    Devuelve un dict con las claves de `METRIC_KEYS` más `"retornos_netos"`
    (la propia serie, sin índice de fecha todavía — quien llama se lo
    asigna, para poder concatenar tramos OOS con sus fechas reales).
    """
    if cost_bps is None:
        cost_bps = TRANSACTION_COST_BPS

    weights = np.asarray(weights, dtype=np.float64)
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    if weights.shape != asset_returns.shape:
        raise ValueError(
            f"evaluate_weights: 'weights' {weights.shape} y 'asset_returns' {asset_returns.shape} "
            "deben tener la misma forma"
        )
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("evaluate_weights: cada fila de 'weights' debe sumar 1")

    n_assets = weights.shape[1]
    initial_weights = np.zeros(n_assets, dtype=np.float64)
    initial_weights[-1] = 1.0  # 100% efectivo antes del primer paso

    prev = np.vstack([initial_weights, weights[:-1]])
    turnover = 0.5 * np.abs(weights - prev).sum(axis=1)
    gross_returns = np.einsum("ij,ij->i", weights, asset_returns)
    net_returns = gross_returns - (cost_bps / 1e4) * turnover

    returns_series = pd.Series(net_returns)

    return {
        "sharpe": sharpe_ratio(returns_series),
        "retorno_anualizado": annualized_return(returns_series),
        "retorno_total": float((1.0 + returns_series).prod() - 1.0),
        "max_drawdown": max_drawdown(returns_series),
        "turnover_total": float(turnover.sum()),
        "turnover_medio_diario": float(turnover.mean()),
        "retornos_netos": returns_series,
    }


# --------------------------------------------------------------------------
# Partición walk-forward
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardBlock:
    """Un bloque walk-forward: entrenar en `[train_start, train_end)`,
    evaluar (OOS) en `[test_start, test_end)`. `train_start` es siempre 0
    (ventana EXPANSIVA, ver `make_walkforward_blocks`).
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def make_walkforward_blocks(n_total: int, min_train: int, n_blocks: int) -> list[WalkForwardBlock]:
    """Partición walk-forward de ventana EXPANSIVA: el bloque k entrena con
    TODA la historia disponible hasta el inicio de su tramo de test
    (`train_start` siempre 0) y evalúa en un tramo de `test_window` pasos
    NUNCA visto en entrenamiento, rodando hacia adelante. Los tramos de
    test son CONSECUTIVOS y no se solapan entre sí — concatenarlos cubre
    exactamente `[min_train, n_total)`, la curva de equity OOS completa.

    `test_window = (n_total - min_train) // n_blocks` (el último bloque se
    queda con el resto de días, si la división no es exacta).

    Lanza `ValueError` si no hay suficientes datos para `n_blocks` bloques
    con al menos 1 paso de test cada uno.
    """
    if min_train >= n_total:
        raise ValueError("make_walkforward_blocks: 'min_train' debe ser menor que 'n_total'")
    if n_blocks < 1:
        raise ValueError("make_walkforward_blocks: 'n_blocks' debe ser >= 1")

    test_window = (n_total - min_train) // n_blocks
    if test_window < 1:
        raise ValueError(
            f"make_walkforward_blocks: no hay suficientes datos ({n_total - min_train} días libres) "
            f"para {n_blocks} bloques"
        )

    blocks: list[WalkForwardBlock] = []
    test_start = min_train
    for i in range(n_blocks):
        test_end = n_total if i == n_blocks - 1 else test_start + test_window
        blocks.append(WalkForwardBlock(train_start=0, train_end=test_start, test_start=test_start, test_end=test_end))
        test_start = test_end
    return blocks


# --------------------------------------------------------------------------
# Entrenamiento + evaluación OOS del agente RL, walk-forward, multi-semilla.
# --------------------------------------------------------------------------


_DEFAULT_PPO_KWARGS: dict = {"n_steps": 256, "batch_size": 64, "n_epochs": 10, "verbose": 0}


def train_and_evaluate_rl(
    dataset: PortfolioDataset,
    blocks: list[WalkForwardBlock],
    seeds: list[int],
    total_timesteps: int,
    cost_bps: float | None = None,
    ppo_kwargs: dict | None = None,
) -> dict[int, dict]:
    """Entrena un agente PPO (`stable_baselines3`) POR BLOQUE Y POR SEMILLA
    — walk-forward: cada bloque entrena en `[0, test_start)` y evalúa
    (política DETERMINÍSTICA, sin exploración) en `[test_start, test_end)`,
    nunca visto durante ESE entrenamiento — y devuelve, por semilla, la
    evaluación de la curva OOS COMPLETA (todos los bloques concatenados en
    orden cronológico, ver `evaluate_weights`).

    Concatenar los PESOS (no solo los retornos) de cada bloque antes de
    evaluar, en vez de evaluar bloque por bloque y promediar las métricas,
    es deliberado: el turnover en el empalme entre el último día de un
    bloque y el primer día del siguiente (donde una policy recién
    reentrenada puede pedir una asignación muy distinta a la que traía el
    bloque anterior) es un costo REAL de este esquema de "reentrenar
    periódicamente" — omitirlo subestimaría el costo de transacción
    genuino de desplegar esto en producción.

    Devuelve `{seed: {**METRIC_KEYS, "retornos_netos": pd.Series con índice de fecha}}`.
    """
    from stable_baselines3 import PPO

    from rl.env import PortfolioEnv

    kwargs = dict(_DEFAULT_PPO_KWARGS)
    if ppo_kwargs:
        kwargs.update(ppo_kwargs)

    results: dict[int, dict] = {}

    for seed in seeds:
        all_weights: list[np.ndarray] = []
        all_returns: list[np.ndarray] = []
        all_dates: list[pd.DatetimeIndex] = []

        for block_idx, block in enumerate(blocks):
            train_env = PortfolioEnv(
                dataset.obs_features[block.train_start:block.train_end],
                dataset.asset_returns[block.train_start:block.train_end],
                cost_bps=cost_bps,
            )
            model = PPO("MlpPolicy", train_env, seed=seed, **kwargs)
            model.learn(total_timesteps=total_timesteps)

            test_obs_arr = dataset.obs_features[block.test_start:block.test_end]
            test_ret_arr = dataset.asset_returns[block.test_start:block.test_end]
            test_env = PortfolioEnv(test_obs_arr, test_ret_arr, cost_bps=cost_bps)

            weights_block = np.zeros_like(test_ret_arr, dtype=np.float64)
            obs, _info = test_env.reset()
            for t in range(len(test_ret_arr)):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = test_env.step(action)
                weights_block[t] = info["weights"]
                if terminated or truncated:
                    break

            all_weights.append(weights_block)
            all_returns.append(test_ret_arr)
            all_dates.append(dataset.dates[block.test_start:block.test_end])
            logger.info(
                "RL seed=%d bloque=%d/%d: train=%d pasos, test=%d pasos",
                seed, block_idx + 1, len(blocks), block.train_end - block.train_start, len(test_ret_arr),
            )

        weights_full = np.concatenate(all_weights, axis=0)
        returns_full = np.concatenate(all_returns, axis=0)
        dates_full = all_dates[0].append(all_dates[1:]) if len(all_dates) > 1 else all_dates[0]

        evaluation = evaluate_weights(weights_full, returns_full, cost_bps=cost_bps)
        evaluation["retornos_netos"].index = dates_full
        results[seed] = evaluation
        logger.info(
            "RL seed=%d: sharpe OOS=%.2f, retorno_total=%.1f%%, turnover_total=%.1f",
            seed, evaluation["sharpe"], evaluation["retorno_total"] * 100, evaluation["turnover_total"],
        )

    return results


# --------------------------------------------------------------------------
# Baselines sobre el mismo tramo OOS completo (concatenación de bloques).
# --------------------------------------------------------------------------


def evaluate_baselines(
    dataset: PortfolioDataset,
    blocks: list[WalkForwardBlock],
    ohlcv_by_asset: dict[str, pd.DataFrame],
    seeds: list[int],
    cost_bps: float | None = None,
) -> dict:
    """Evalúa los 4 baselines sobre el MISMO tramo OOS que el agente RL
    (`[blocks[0].test_start, blocks[-1].test_end)`, la unión de todos los
    tramos de test). Los 3 determinísticos (equiponderado, buy&hold BTC,
    vol targeting) se evalúan una sola vez; el asignador aleatorio, una vez
    POR SEMILLA (misma lista `seeds` que el agente RL), para poder reportar
    su propia media ± desvío — el punto de comparación "de verdad" para
    decidir si el RL aprendió algo (ver `rl/__init__.py`).

    Devuelve `{"buy_hold_equiponderado": {...}, "buy_hold_btc": {...},
    "vol_targeting": {...}, "asignador_aleatorio": {seed: {...}}}`.
    """
    oos_start = blocks[0].test_start
    oos_end = blocks[-1].test_end
    n_steps = oos_end - oos_start
    n_assets = dataset.asset_returns.shape[1]
    dates = dataset.dates[oos_start:oos_end]
    returns_slice = dataset.asset_returns[oos_start:oos_end]

    results: dict = {}

    equal = equal_weight_positions(n_steps, n_assets)
    ev = evaluate_weights(equal, returns_slice, cost_bps=cost_bps)
    ev["retornos_netos"].index = dates
    results["buy_hold_equiponderado"] = ev

    btc_index = dataset.asset_names.index("BTC")
    btc_only = buy_and_hold_btc_positions(n_steps, n_assets, btc_index=btc_index)
    ev = evaluate_weights(btc_only, returns_slice, cost_bps=cost_bps)
    ev["retornos_netos"].index = dates
    results["buy_hold_btc"] = ev

    vol_target = vol_targeting_positions(ohlcv_by_asset, dates, asset_order=dataset.asset_names[:-1])
    ev = evaluate_weights(vol_target, returns_slice, cost_bps=cost_bps)
    ev["retornos_netos"].index = dates
    results["vol_targeting"] = ev

    random_per_seed: dict[int, dict] = {}
    for seed in seeds:
        random_w = random_positions(n_steps, n_assets, seed=seed)
        ev = evaluate_weights(random_w, returns_slice, cost_bps=cost_bps)
        ev["retornos_netos"].index = dates
        random_per_seed[seed] = ev
    results["asignador_aleatorio"] = random_per_seed

    return results


# --------------------------------------------------------------------------
# Orquestador del experimento completo + resumen + conclusión honesta.
# --------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    rl_per_seed: dict[int, dict]
    random_per_seed: dict[int, dict]
    deterministic_baselines: dict[str, dict]
    blocks: list[WalkForwardBlock] = field(default_factory=list)
    dates: pd.DatetimeIndex | None = None
    params: dict = field(default_factory=dict)


def run_walkforward_experiment(
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    min_train_days: int = 730,
    n_blocks: int = 4,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    total_timesteps: int = 20_000,
    cost_bps: float | None = None,
    ppo_kwargs: dict | None = None,
) -> ExperimentResult:
    """Corre el experimento completo: arma el dataset, parte en bloques
    walk-forward, entrena+evalúa el agente RL (por bloque y por semilla) y
    evalúa los 4 baselines sobre el mismo tramo OOS. Ver `rl/__init__.py`
    para el encuadre honesto y `scripts/run_rl_experiment.py` para el punto
    de entrada de línea de comandos que arma la tabla final y la guarda.
    """
    dataset = build_portfolio_dataset(assets)
    n_total = len(dataset.dates)
    blocks = make_walkforward_blocks(n_total, min_train=min_train_days, n_blocks=n_blocks)

    ohlcv_by_asset = {
        asset: get_prices(asset, source="store", interval="1d", use_cache=False) for asset in assets
    }

    rl_results = train_and_evaluate_rl(
        dataset, blocks, list(seeds), total_timesteps, cost_bps=cost_bps, ppo_kwargs=ppo_kwargs
    )
    baseline_results = evaluate_baselines(dataset, blocks, ohlcv_by_asset, list(seeds), cost_bps=cost_bps)
    random_per_seed = baseline_results.pop("asignador_aleatorio")

    return ExperimentResult(
        rl_per_seed=rl_results,
        random_per_seed=random_per_seed,
        deterministic_baselines=baseline_results,
        blocks=blocks,
        dates=dataset.dates,
        params={
            "assets": list(assets),
            "min_train_days": min_train_days,
            "n_blocks": n_blocks,
            "seeds": list(seeds),
            "total_timesteps": total_timesteps,
            "cost_bps": TRANSACTION_COST_BPS if cost_bps is None else cost_bps,
        },
    )


def _aggregate_seeds(per_seed: dict[int, dict], metric_keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """(media, desvío estándar) de cada métrica entre semillas — desvío 0.0
    si solo hay 1 semilla (ddof=1 no está definido con n=1).
    """
    out: dict[str, tuple[float, float]] = {}
    for key in metric_keys:
        values = np.array([per_seed[seed][key] for seed in per_seed], dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[key] = (float(values.mean()), std)
    return out


def summarize_experiment(result: ExperimentResult) -> pd.DataFrame:
    """Tabla final: una fila por estrategia. El agente RL y el asignador
    aleatorio (los dos con componente aleatoria) llevan media Y desvío
    estándar entre semillas; los 3 baselines determinísticos llevan un solo
    valor (desvío 0 por definición, no hay semilla que variar).
    """
    rows: list[dict] = []

    rl_agg = _aggregate_seeds(result.rl_per_seed, METRIC_KEYS)
    rows.append(
        {
            "estrategia": "RL (PPO)",
            **{f"{k}_media": v[0] for k, v in rl_agg.items()},
            **{f"{k}_std": v[1] for k, v in rl_agg.items()},
        }
    )

    random_agg = _aggregate_seeds(result.random_per_seed, METRIC_KEYS)
    rows.append(
        {
            "estrategia": "Asignador aleatorio",
            **{f"{k}_media": v[0] for k, v in random_agg.items()},
            **{f"{k}_std": v[1] for k, v in random_agg.items()},
        }
    )

    for name, ev in result.deterministic_baselines.items():
        rows.append(
            {
                "estrategia": name,
                **{f"{k}_media": ev[k] for k in METRIC_KEYS},
                **{f"{k}_std": 0.0 for k in METRIC_KEYS},
            }
        )

    return pd.DataFrame(rows)


def rl_beats_all_baselines(result: ExperimentResult) -> dict:
    """Conclusión automática y HONESTA: ¿el RL supera a TODOS los baselines
    en Sharpe OOS, de forma consistente entre semillas?

    Criterio deliberadamente estricto: se exige que la PEOR semilla del RL
    (no su media) le gane a cada baseline determinístico y a la MEDIA del
    asignador aleatorio. Una sola semilla mala ya alcanza para responder
    "No" — el objetivo es descartar que un resultado positivo sea puro
    ruido de inicialización de UNA corrida con suerte (ver `rl/__init__.py`:
    RL es de alta varianza, "le ganó en la media de 5 corridas, pero 2 de
    esas 5 fueron peores que un mono" NO es una conclusión de "aprendió algo").
    """
    rl_sharpes = [result.rl_per_seed[seed]["sharpe"] for seed in result.rl_per_seed]
    worst_rl_sharpe = min(rl_sharpes)
    best_rl_sharpe = max(rl_sharpes)

    comparisons: dict[str, float] = {name: ev["sharpe"] for name, ev in result.deterministic_baselines.items()}
    random_sharpes = [result.random_per_seed[seed]["sharpe"] for seed in result.random_per_seed]
    comparisons["asignador_aleatorio (media)"] = float(np.mean(random_sharpes))

    supera_todos = all(worst_rl_sharpe > sharpe for sharpe in comparisons.values())

    return {
        "supera_a_todos_los_baselines_consistentemente": supera_todos,
        "sharpe_rl_peor_semilla": worst_rl_sharpe,
        "sharpe_rl_mejor_semilla": best_rl_sharpe,
        "sharpe_baselines": comparisons,
    }
