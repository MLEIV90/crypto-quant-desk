"""Tests para rl/env.py, rl/evaluation.py y rl/baselines.py (Fase 18) —
sintéticos, sin red y sin entrenar ningún modelo (eso es lento por diseño,
ver `rl/__init__.py`; el entrenamiento real se ejerce en
`scripts/run_rl_experiment.py`, no en la suite de pytest). Lo que este
archivo verifica es el CONTRATO del entorno: cumple la API de gymnasium,
los pesos siempre suman 1, la recompensa descuenta costos, y no hay
lookahead (la observación en t no depende de t+1 en adelante).
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl.env import PortfolioEnv, softmax_weights
from rl.evaluation import evaluate_weights, make_walkforward_blocks
from rl.baselines import buy_and_hold_btc_positions, equal_weight_positions, random_positions


def _make_env(n_steps: int = 40, n_features: int = 6, n_assets: int = 4, seed: int = 0) -> PortfolioEnv:
    rng = np.random.default_rng(seed)
    obs_features = rng.normal(size=(n_steps, n_features)).astype(np.float32)
    asset_returns = (rng.normal(scale=0.02, size=(n_steps, n_assets))).astype(np.float32)
    asset_returns[:, -1] = 0.0  # CASH
    return PortfolioEnv(obs_features, asset_returns)


# --------------------------------------------------------------------------
# softmax_weights
# --------------------------------------------------------------------------


def test_softmax_weights_always_sums_to_one_and_nonnegative() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        action = rng.uniform(-1.0, 1.0, size=6)
        weights = softmax_weights(action)
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (weights >= 0.0).all()


def test_softmax_weights_extreme_action_concentrates_on_one_asset() -> None:
    action = np.array([1.0, -1.0, -1.0, -1.0])
    weights = softmax_weights(action, scale=5.0)
    assert weights[0] > 0.9  # el resto casi no recibe peso


# --------------------------------------------------------------------------
# PortfolioEnv: API de gymnasium
# --------------------------------------------------------------------------


def test_portfolio_env_passes_gymnasium_check_env() -> None:
    env = _make_env()
    check_env(env, skip_render_check=True)


def test_portfolio_env_requires_matching_lengths() -> None:
    with pytest.raises(ValueError):
        PortfolioEnv(np.zeros((10, 3), dtype=np.float32), np.zeros((9, 4), dtype=np.float32))


def test_portfolio_env_reset_returns_valid_observation() -> None:
    env = _make_env(n_steps=10, n_features=5, n_assets=3)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert info == {}


def test_portfolio_env_runs_full_episode_and_truncates_at_the_end() -> None:
    env = _make_env(n_steps=10, n_features=5, n_assets=3)
    env.reset()
    n_transitions = 0
    truncated = False
    while not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated  # este entorno no tiene condición de "terminated", solo time-limit
        n_transitions += 1
    assert n_transitions == 10  # se usan las 10 filas de datos, ni una de más ni de menos


# --------------------------------------------------------------------------
# Los pesos que devuelve el entorno SIEMPRE suman 1
# --------------------------------------------------------------------------


def test_portfolio_env_weights_always_sum_to_one_across_random_rollout() -> None:
    env = _make_env(n_steps=30, n_features=6, n_assets=5, seed=7)
    env.reset()
    truncated = False
    while not truncated:
        action = env.action_space.sample()
        _obs, _reward, _terminated, truncated, info = env.step(action)
        assert info["weights"].sum() == pytest.approx(1.0, abs=1e-5)
        assert (info["weights"] >= -1e-8).all()


# --------------------------------------------------------------------------
# La recompensa descuenta costos de transacción (turnover * cost_bps)
# --------------------------------------------------------------------------


def test_portfolio_env_reward_matches_hand_computed_value_with_costs() -> None:
    # Un solo activo cripto + cash, retorno conocido, acción conocida ->
    # se puede calcular a mano el resultado exacto del softmax/turnover/reward.
    obs_features = np.zeros((3, 1), dtype=np.float32)
    asset_returns = np.array([[0.10, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    cost_bps = 10.0
    env = PortfolioEnv(obs_features, asset_returns, cost_bps=cost_bps)
    env.reset()

    action = np.array([1.0, -1.0], dtype=np.float32)  # empuja fuerte hacia el activo 0
    weights = softmax_weights(action, scale=env._action_scale)  # noqa: SLF001 (test blanco, acceso intencional)

    obs, reward, terminated, truncated, info = env.step(action)

    initial_weights = np.array([0.0, 1.0])  # 100% efectivo antes del primer paso
    expected_turnover = 0.5 * np.abs(weights - initial_weights).sum()  # convención "one_way", ver evaluate_weights
    expected_gross_return = float(np.dot(weights, asset_returns[0]))
    expected_reward = np.log1p(expected_gross_return) - (cost_bps / 1e4) * expected_turnover

    assert info["turnover"] == pytest.approx(expected_turnover, abs=1e-6)
    assert reward == pytest.approx(expected_reward, abs=1e-6)


def test_portfolio_env_higher_cost_bps_strictly_lowers_reward_for_same_action() -> None:
    obs_features = np.zeros((3, 1), dtype=np.float32)
    asset_returns = np.array([[0.05, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    action = np.array([1.0, -1.0], dtype=np.float32)

    env_cheap = PortfolioEnv(obs_features.copy(), asset_returns.copy(), cost_bps=1.0)
    env_expensive = PortfolioEnv(obs_features.copy(), asset_returns.copy(), cost_bps=500.0)
    env_cheap.reset()
    env_expensive.reset()

    _obs1, reward_cheap, *_ = env_cheap.step(action)
    _obs2, reward_expensive, *_ = env_expensive.step(action)

    assert reward_expensive < reward_cheap


# --------------------------------------------------------------------------
# Sin lookahead: la observación en t (y la recompensa en t) no dependen de
# ningún dato posterior a t+1.
# --------------------------------------------------------------------------


def test_portfolio_env_reset_and_first_steps_do_not_depend_on_future_data() -> None:
    """Dos entornos IDÉNTICOS hasta el paso k+1, pero con datos DISTINTOS
    a partir de ahí, deben producir exactamente las mismas observaciones y
    recompensas en los pasos 0..k — si no, algo en el entorno estaría
    mirando datos "futuros" respecto del paso actual.
    """
    rng = np.random.default_rng(42)
    n_features, n_assets = 6, 4
    shared_len = 5
    obs_shared = rng.normal(size=(shared_len, n_features)).astype(np.float32)
    ret_shared = (rng.normal(scale=0.02, size=(shared_len, n_assets))).astype(np.float32)
    ret_shared[:, -1] = 0.0

    # Cola A y cola B: mismos primeros `shared_len` pasos, colas DISTINTAS.
    tail_a = rng.normal(size=(5, n_features)).astype(np.float32)
    tail_b = rng.normal(size=(5, n_features)).astype(np.float32) * 100.0  # bien distinto
    ret_tail_a = (rng.normal(scale=0.02, size=(5, n_assets))).astype(np.float32)
    ret_tail_b = (rng.normal(scale=0.02, size=(5, n_assets))).astype(np.float32) * 50.0
    ret_tail_a[:, -1] = 0.0
    ret_tail_b[:, -1] = 0.0

    obs_a = np.concatenate([obs_shared, tail_a])
    obs_b = np.concatenate([obs_shared, tail_b])
    ret_a = np.concatenate([ret_shared, ret_tail_a])
    ret_b = np.concatenate([ret_shared, ret_tail_b])

    env_a = PortfolioEnv(obs_a, ret_a)
    env_b = PortfolioEnv(obs_b, ret_b)

    obs0_a, _ = env_a.reset()
    obs0_b, _ = env_b.reset()
    np.testing.assert_allclose(obs0_a, obs0_b)

    rng_actions = np.random.default_rng(99)
    for step_idx in range(shared_len):
        action = rng_actions.uniform(-1.0, 1.0, size=n_assets).astype(np.float32)
        obs_a_next, reward_a, *_ = env_a.step(action.copy())
        obs_b_next, reward_b, *_ = env_b.step(action.copy())

        assert reward_a == pytest.approx(reward_b, abs=1e-6), f"reward diverge en el paso {step_idx}"
        if step_idx < shared_len - 1:
            # La OBSERVACIÓN del próximo paso todavía cae dentro del tramo
            # compartido -> debe ser idéntica entre A y B.
            np.testing.assert_allclose(obs_a_next, obs_b_next)


def test_evaluate_weights_uses_only_same_row_return_no_shift_leakage() -> None:
    """`evaluate_weights` empareja la fila t de pesos con la fila t de
    retornos tal cual (sin desplazar) — cambiar el retorno de una fila
    FUTURA no puede cambiar el resultado de las filas anteriores.
    """
    weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    returns_original = np.array([[0.10, 0.0], [0.05, 0.0], [0.02, 0.0]])
    returns_changed_future = returns_original.copy()
    returns_changed_future[2, 0] = 9.99  # cambia SOLO el retorno futuro (última fila)

    result_original = evaluate_weights(weights, returns_original, cost_bps=10.0)
    result_changed = evaluate_weights(weights, returns_changed_future, cost_bps=10.0)

    # Los dos primeros retornos netos (filas 0 y 1) no pueden haber cambiado.
    np.testing.assert_allclose(
        result_original["retornos_netos"].to_numpy()[:2],
        result_changed["retornos_netos"].to_numpy()[:2],
    )


# --------------------------------------------------------------------------
# evaluate_weights: validaciones y fórmula de turnover/costos
# --------------------------------------------------------------------------


def test_evaluate_weights_rejects_rows_that_do_not_sum_to_one() -> None:
    weights = np.array([[0.5, 0.4], [1.0, 0.0]])  # primera fila suma 0.9
    returns = np.zeros((2, 2))
    with pytest.raises(ValueError, match="sumar 1"):
        evaluate_weights(weights, returns, cost_bps=10.0)


def test_evaluate_weights_first_step_turnover_counts_move_from_all_cash() -> None:
    # Pesos constantes 100% en el activo 0 desde el primer paso: el
    # turnover del primer paso es 1.0 (mover TODO desde 100% efectivo),
    # y 0.0 en adelante (no cambia nada).
    weights = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    returns = np.array([[0.01, 0.0], [0.01, 0.0], [0.01, 0.0]])
    result = evaluate_weights(weights, returns, cost_bps=100.0)
    assert result["turnover_total"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Baselines: forma y propiedades básicas
# --------------------------------------------------------------------------


def test_equal_weight_positions_sum_to_one_and_split_evenly() -> None:
    weights = equal_weight_positions(n_steps=5, n_assets=4)
    assert weights.shape == (5, 4)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[:, :3], 1.0 / 3)
    np.testing.assert_allclose(weights[:, 3], 0.0)  # CASH sin peso


def test_buy_and_hold_btc_positions_is_fully_concentrated() -> None:
    weights = buy_and_hold_btc_positions(n_steps=5, n_assets=4, btc_index=0)
    np.testing.assert_allclose(weights[:, 0], 1.0)
    np.testing.assert_allclose(weights[:, 1:], 0.0)


def test_random_positions_sum_to_one_and_are_reproducible_by_seed() -> None:
    a = random_positions(n_steps=10, n_assets=5, seed=123)
    b = random_positions(n_steps=10, n_assets=5, seed=123)
    c = random_positions(n_steps=10, n_assets=5, seed=456)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


# --------------------------------------------------------------------------
# make_walkforward_blocks
# --------------------------------------------------------------------------


def test_make_walkforward_blocks_covers_test_range_without_overlap() -> None:
    blocks = make_walkforward_blocks(n_total=100, min_train=40, n_blocks=3)
    assert len(blocks) == 3
    assert blocks[0].train_start == 0
    assert blocks[0].test_start == 40
    # Cada bloque entrena con TODA la historia hasta el inicio de su test (expansivo).
    for block in blocks:
        assert block.train_start == 0
        assert block.train_end == block.test_start
    # Los tramos de test son consecutivos, sin solaparse, y cubren [40, 100).
    assert blocks[0].test_start == 40
    assert blocks[-1].test_end == 100
    for a, b in zip(blocks, blocks[1:]):
        assert a.test_end == b.test_start


def test_make_walkforward_blocks_rejects_insufficient_data() -> None:
    with pytest.raises(ValueError):
        make_walkforward_blocks(n_total=10, min_train=9, n_blocks=5)
