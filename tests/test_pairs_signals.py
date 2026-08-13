"""Tests offline para pairs/signals.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.signals import generate_pair_signals, zscore


def _spike_spread(n_burn: int = 60, spike: float = 5.0, n_revert: int = 15, seed: int = 6) -> pd.Series:
    """Spread plano (ruido chico) + un salto grande y aislado + reversión
    cerca de la media original: un caso simple y controlado para verificar
    entradas/salidas.

    seed=6 está elegido a propósito: con ruido N(0, 0.1) puro, es estadística
    normal que el z-score expansivo cruce +-2 por azar de vez en cuando
    (P(|Z|>2) ~ 4.6% por observación bajo H0) — no es un bug del módulo,
    es la naturaleza de un umbral de 2 desvíos sobre ruido gaussiano. Esta
    semilla da un burn-in "limpio" (sin cruces espurios) para poder aislar
    la señal del salto sin ese ruido de fondo estadístico en el test.
    """
    rng = np.random.default_rng(seed)
    burn = rng.normal(0.0, 0.1, n_burn)
    revert = np.full(n_revert, 0.05)
    values = np.concatenate([burn, [spike], revert])
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


# --------------------------------------------------------------------------
# zscore
# --------------------------------------------------------------------------


def test_zscore_expanding_uses_only_past_and_present() -> None:
    spread = _spike_spread()
    z_full = zscore(spread)
    cutoff = 65
    z_truncated = zscore(spread.iloc[:cutoff])

    pd.testing.assert_series_equal(z_full.iloc[:cutoff], z_truncated)


def test_zscore_rolling_window_matches_manual_rolling_stats() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    spread = pd.Series(np.arange(50, dtype=float), index=idx)

    z = zscore(spread, window=10)

    expected_mean = spread.rolling(10).mean()
    expected_std = spread.rolling(10).std(ddof=1)
    expected = (spread - expected_mean) / expected_std
    pd.testing.assert_series_equal(z, expected, check_names=False)


# --------------------------------------------------------------------------
# generate_pair_signals: entradas/salidas
# --------------------------------------------------------------------------


def test_generate_pair_signals_opens_short_on_spike_and_closes_on_reversion() -> None:
    spread = _spike_spread(n_burn=60, spike=5.0, n_revert=15)
    signals = generate_pair_signals(spread, entry=2.0, exit=0.5, stop=10.0)

    spike_idx = 60  # posición del salto (después de los 60 días de burn-in)

    # Antes del salto: nunca hay posición abierta (el ruido de burn-in es chico).
    assert (signals["posicion_spread"].iloc[:spike_idx] == 0.0).all()

    # En el salto: z debe superar 'entry' y abrir CORTO-spread.
    assert signals["z"].iloc[spike_idx] > 2.0
    assert signals["posicion_spread"].iloc[spike_idx] == -1.0
    assert signals["evento"].iloc[spike_idx] == "entrada_corta"

    # Al volver cerca de la media (revert), debería cerrar.
    assert (signals["evento"] == "cierre").any()
    post_spike = signals.iloc[spike_idx + 1 :]
    close_positions = post_spike.index[post_spike["evento"] == "cierre"]
    assert len(close_positions) >= 1
    first_close = close_positions[0]
    assert signals.loc[first_close, "posicion_spread"] == 0.0


def test_generate_pair_signals_opens_long_on_negative_spike() -> None:
    spread = _spike_spread(n_burn=60, spike=-5.0, n_revert=15)
    signals = generate_pair_signals(spread, entry=2.0, exit=0.5, stop=10.0)

    spike_idx = 60
    assert signals["z"].iloc[spike_idx] < -2.0
    assert signals["posicion_spread"].iloc[spike_idx] == 1.0
    assert signals["evento"].iloc[spike_idx] == "entrada_larga"


def test_generate_pair_signals_stop_loss_triggers_when_spread_keeps_diverging() -> None:
    # Después de abrir, el spread se aleja AÚN MÁS en vez de revertir. Un
    # burn-in largo (250 obs) hace que la media/desvío expansivos ya estén
    # bien establecidos antes de la divergencia (con un burn-in corto, la
    # divergencia se "absorbe" muy rápido en la media/desvío expansivos —
    # todavía basados en pocos puntos — y el z-score no llega a sostenerse
    # por encima de `stop`).
    rng = np.random.default_rng(1)
    n_burn = 250
    burn = rng.normal(0.0, 0.1, n_burn)
    diverging = np.linspace(5.0, 30.0, 30)  # sigue subiendo tras el salto inicial
    values = np.concatenate([burn, diverging])
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    spread = pd.Series(values, index=idx)

    signals = generate_pair_signals(spread, entry=2.0, exit=0.5, stop=6.0)

    assert (signals["evento"] == "stop_loss").any()
    stop_idx = signals.index[signals["evento"] == "stop_loss"][0]
    assert signals.loc[stop_idx, "posicion_spread"] == 0.0


def test_generate_pair_signals_rejects_invalid_threshold_ordering() -> None:
    spread = _spike_spread()
    with pytest.raises(ValueError):
        generate_pair_signals(spread, entry=0.5, exit=2.0, stop=4.0)  # exit > entry, inválido
    with pytest.raises(ValueError):
        generate_pair_signals(spread, entry=2.0, exit=0.5, stop=1.0)  # stop < entry, inválido


# --------------------------------------------------------------------------
# Sin lookahead
# --------------------------------------------------------------------------


def test_generate_pair_signals_has_no_lookahead() -> None:
    spread = _spike_spread(n_burn=100, spike=5.0, n_revert=30, seed=2)
    full_signals = generate_pair_signals(spread)

    cutoff = len(spread) - 20
    truncated_signals = generate_pair_signals(spread.iloc[:cutoff])

    pd.testing.assert_frame_equal(full_signals.iloc[:cutoff], truncated_signals)
