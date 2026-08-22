"""`PortfolioEnv`: entorno `gymnasium.Env` de asignación de cartera (Fase 18).

Ver `rl/__init__.py` para el encuadre honesto del experimento completo y
`rl.features.build_portfolio_dataset` para cómo se arma la observación/los
retornos que este entorno consume (ya trailing/causales por construcción).
"""

from __future__ import annotations

import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import TRANSACTION_COST_BPS

logger = logging.getLogger(__name__)

# Amplifica los logits de la acción ANTES del softmax (ver `softmax_weights`).
ACTION_SCALE: float = 5.0


def softmax_weights(action: np.ndarray, scale: float = ACTION_SCALE) -> np.ndarray:
    """Convierte un vector de acción continua (logits sin restricción de
    signo ni de suma) en pesos de cartera NO NEGATIVOS que SUMAN 1 (softmax
    numéricamente estable: se resta el máximo antes de exponenciar).

    `scale` amplifica los logits antes del softmax: el espacio de acción de
    `PortfolioEnv` es `Box([-1, 1])` (rango simétrico y acotado, la
    convención estándar para la exploración gaussiana de una policy PPO) —
    sin este factor, un softmax sobre logits ya comprimidos en [-1, 1]
    produce pesos casi uniformes incluso con una policy ya convergida,
    poniéndole un techo artificial a cuánta convicción puede expresar el
    agente. `scale=5` dejar que el softmax sature lo suficiente como para
    expresar una posición casi 100% concentrada en un solo activo cuando la
    policy empuja la acción a un extremo de la Box.
    """
    logits = np.asarray(action, dtype=np.float64) * scale
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class PortfolioEnv(gym.Env):
    """Entorno de asignación de cartera sobre el universo de
    `rl.features.DEFAULT_ASSETS` + efectivo.

    OBSERVACIÓN: `obs_features[t]` (estacionaria/causal, ver
    `rl.features.build_portfolio_dataset`) concatenada con los pesos
    ACTUALES de la cartera (los que el propio agente fijó en el paso
    anterior, o 100% efectivo al `reset`) — el agente necesita saber "dónde
    está parado" para poder razonar sobre el COSTO de moverse a una nueva
    asignación, no solo sobre el estado del mercado.

    ACCIÓN: vector continuo de `n_assets` dimensiones en `Box([-1, 1])`, que
    `step` convierte en pesos de cartera vía `softmax_weights` — el softmax
    garantiza, POR CONSTRUCCIÓN, que los pesos resultantes sean no negativos
    y sumen exactamente 1: no hace falta validar ni normalizar la acción del
    agente por fuera, cualquier vector de logits produce una distribución
    válida.

    RECOMPENSA: retorno LOGARÍTMICO de la cartera del paso, menos el costo
    de transacción (turnover * cost_bps):

        turnover_t      = 0.5 * sum(|pesos_t - pesos_{t-1}|)  (one-way, ver
                                                                config.TURNOVER_CONVENTION
                                                                — el factor 0.5 es necesario
                                                                porque acá el efectivo es una
                                                                columna más del vector de
                                                                pesos, así que un movimiento
                                                                cambia DOS columnas a la vez;
                                                                ver rl.evaluation.evaluate_weights)
        retorno_bruto_t = pesos_t . retornos_activos_t
        reward_t        = log(1 + retorno_bruto_t) - (cost_bps / 1e4) * turnover_t

    Se eligió retorno LOG (no un differential Sharpe u otra métrica de
    riesgo-ajustado por paso) a propósito, como piso honesto del
    experimento: es ADITIVO en el tiempo — maximizar la SUMA de reward a lo
    largo del episodio es exactamente maximizar el retorno compuesto de la
    cartera, sin ninguna sorpresa de agregación — y es mucho más difícil de
    "gamear" que una recompensa de Sharpe por paso, que puede premiar a un
    agente por reducir la VARIANZA reportada de un puñado de pasos sin que
    eso refleje ninguna habilidad real (especialmente temprano en el
    entrenamiento, con una muestra de retornos todavía chica). Una variante
    con differential Sharpe queda como extensión futura documentada, no la
    recompensa por defecto de este experimento.

    SIN LOOKAHEAD: en el paso t, la observación es `obs_features[t]`
    (calculada, por construcción, solo con datos hasta el cierre de t) y la
    recompensa usa `asset_returns[t]`, que YA es el retorno realizado entre
    t y t+1 (la fila t de `asset_returns` es "lo que pasó DESPUÉS de t", ver
    `rl.features.build_portfolio_dataset`) — la separación entre "lo que el
    agente vio" y "lo que se usa para puntuarlo" vive en el DATASET, no acá:
    este entorno no necesita ningún shift adicional porque `asset_returns[t]`
    ya es el retorno futuro correcto para la acción tomada con `obs_features[t]`.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        obs_features: np.ndarray,
        asset_returns: np.ndarray,
        cost_bps: float | None = None,
        action_scale: float = ACTION_SCALE,
    ) -> None:
        super().__init__()
        obs_features = np.asarray(obs_features, dtype=np.float32)
        asset_returns = np.asarray(asset_returns, dtype=np.float32)
        if len(obs_features) != len(asset_returns):
            raise ValueError("PortfolioEnv: 'obs_features' y 'asset_returns' deben tener la misma longitud")
        if len(obs_features) < 2:
            raise ValueError("PortfolioEnv: se necesitan al menos 2 pasos de datos")

        self._obs_features = obs_features
        self._asset_returns = asset_returns
        self._cost_bps = TRANSACTION_COST_BPS if cost_bps is None else cost_bps
        self._action_scale = action_scale
        self._n_assets = asset_returns.shape[1]
        self._n_steps = len(obs_features)

        obs_dim = obs_features.shape[1] + self._n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self._n_assets,), dtype=np.float32)

        self._t = 0
        self._weights = self._initial_weights()

    def _initial_weights(self) -> np.ndarray:
        # Arranca 100% en efectivo -- ninguna posición "heredada" de fuera
        # de la muestra (misma convención que backtest.engine.run_backtest:
        # la posición efectiva del primer día siempre es 0). Por convención
        # de rl.features.build_portfolio_dataset, la ÚLTIMA columna de
        # asset_returns es siempre "CASH".
        weights = np.zeros(self._n_assets, dtype=np.float32)
        weights[-1] = 1.0
        return weights

    def _get_obs(self) -> np.ndarray:
        idx = min(self._t, self._n_steps - 1)
        return np.concatenate([self._obs_features[idx], self._weights]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        self._weights = self._initial_weights()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self._t >= self._n_steps:
            raise RuntimeError("PortfolioEnv.step: llamado después de que el episodio terminó (llamá a reset())")

        new_weights = softmax_weights(action, scale=self._action_scale).astype(np.float32)
        turnover = 0.5 * float(np.abs(new_weights - self._weights).sum())
        gross_return = float(np.dot(new_weights, self._asset_returns[self._t]))
        cost = (self._cost_bps / 1e4) * turnover
        reward = float(np.log1p(gross_return) - cost)

        self._weights = new_weights
        self._t += 1

        terminated = False
        truncated = self._t >= self._n_steps
        obs = self._get_obs()
        info = {"weights": new_weights.copy(), "turnover": turnover, "gross_return": gross_return}
        return obs, reward, terminated, truncated, info
