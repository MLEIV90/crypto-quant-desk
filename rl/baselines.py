"""Baselines de comparación para el experimento de Fase 18 (`rl/`).

Cada función devuelve una matriz de pesos `(T, N_ASSETS + 1)` sobre el MISMO
rango de pasos que se le pida, en el mismo orden de columnas que
`rl.features.PortfolioDataset.asset_names` (termina en "CASH") — lista para
`rl.evaluation.evaluate_weights`, la MISMA función que evalúa al agente RL,
para que la comparación sea sobre una fórmula de costos/turnover idéntica
para todos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.engine import generate_positions


def equal_weight_positions(n_steps: int, n_assets: int) -> np.ndarray:
    """Buy&hold equiponderado (1/N), REBALANCEADO cada paso: cada activo
    cripto (todas las columnas salvo la última, "CASH") recibe `1/n_crypto`
    de peso en TODOS los pasos. El rebalanceo diario de vuelta a pesos
    iguales (a medida que los precios se separan) es justamente lo que
    genera turnover — y por lo tanto costos — en esta estrategia, a
    diferencia de un buy&hold puro sin rebalancear.
    """
    n_crypto = n_assets - 1
    weights = np.zeros((n_steps, n_assets), dtype=np.float64)
    weights[:, :n_crypto] = 1.0 / n_crypto
    return weights


def buy_and_hold_btc_positions(n_steps: int, n_assets: int, btc_index: int = 0) -> np.ndarray:
    """100% BTC en todos los pasos (`btc_index` es la posición de BTC en el
    orden de columnas — 0 por convención de `config.UNIVERSE`/
    `rl.features.DEFAULT_ASSETS`, donde BTC es siempre el primer activo).
    """
    weights = np.zeros((n_steps, n_assets), dtype=np.float64)
    weights[:, btc_index] = 1.0
    return weights


def random_positions(n_steps: int, n_assets: int, seed: int) -> np.ndarray:
    """Asignador ALEATORIO — el baseline CRÍTICO del experimento (ver
    `rl/__init__.py`): pesos softmax de logits normales i.i.d. en CADA
    paso, sin ninguna señal de mercado. Si el agente RL no le gana a esto
    de forma consistente entre semillas, no aprendió ninguna estructura
    real de los datos — le estaría ganando (o perdiendo) a un mono tirando
    dardos, no a una estrategia informada.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n_steps, n_assets))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def vol_targeting_positions(
    ohlcv_by_asset: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    asset_order: tuple[str, ...],
) -> np.ndarray:
    """Extiende la estrategia de vol targeting YA EXISTENTE
    (`signals.engine.generate_positions`, reutilizada TAL CUAL, sin
    reimplementar ningún cálculo) de un solo activo a una CARTERA de varios:
    corre `generate_positions(..., allow_short=False)` por separado para
    cada activo de `asset_order` (posición en `[0, config.MAX_LEVERAGE]`
    por activo, según su propio score de tendencia/momentum/reversión y su
    propio sizing por volatilidad realizada) y normaliza el vector
    resultante en cada fecha:

    - Si la SUMA de posiciones de esa fecha es <= 1: los pesos son esas
      posiciones tal cual y el resto queda en EFECTIVO (`1 - suma`) — la
      cartera solo invierte tanto como los scores individuales lo
      justifican, ni más.
    - Si la suma excede 1 (varios activos con score/tamaño alto a la vez):
      se escalan todos proporcionalmente para que sumen exactamente 1, sin
      apalancamiento ni efectivo — la única normalización que no existía
      todavía en el proyecto (`generate_positions` es de un solo activo).

    Devuelve una matriz `(len(dates), len(asset_order) + 1)`, con la
    columna de efectivo al final (mismo orden que
    `rl.features.PortfolioDataset.asset_names`).
    """
    per_asset_positions = pd.DataFrame(
        {asset: generate_positions(ohlcv_by_asset[asset], allow_short=False) for asset in asset_order}
    ).reindex(dates).fillna(0.0)

    raw = per_asset_positions.to_numpy(dtype=np.float64)
    totals = raw.sum(axis=1, keepdims=True)
    scale = np.where(totals > 1.0, totals, 1.0)
    scaled = raw / scale
    cash = np.clip(1.0 - scaled.sum(axis=1, keepdims=True), 0.0, None)
    return np.concatenate([scaled, cash], axis=1)
