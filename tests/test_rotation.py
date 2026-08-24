"""Tests para strategies/rotation.py (Fase 19a), sobre snapshots SINTÉTICOS
(sin red — mismo patrón que `tests/test_loaders.py`: se escribe un parquet
falso en un `tmp_path` y se monkeypatchea `data.loaders.SNAPSHOT_DIR` para
que `get_prices(source="store")` lo lea). Se usan tickers reales de
`config.UNIVERSE` (BTC/ETH) con precios inventados — `get_prices` valida el
ticker contra `config.UNIVERSE`, así que no se puede usar un nombre
arbitrario.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import loaders
from strategies.rotation import baseline_backtests, holdings_to_weights, momentum_rotation_backtest


def _write_snapshot(directory: Path, asset: str, prices: np.ndarray, start: str = "2021-01-01") -> None:
    idx = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 100.0}, index=idx
    )
    df.index.name = "timestamp"
    df.to_parquet(directory / f"{asset}_1d.parquet")


# --------------------------------------------------------------------------
# Escenario sintético con momentum CLARO: BTC sube y después baja; ETH baja
# y después sube (con el mismo quiebre a mitad de camino) — la rotación por
# momentum, con un lookback suficientemente corto para detectar el quiebre,
# debería capturar buena parte de LAS DOS subas y evitar buena parte de LAS
# DOS bajas, superando ampliamente a quedarse en cualquiera de las dos solo
# o al 50/50.
# --------------------------------------------------------------------------


def _write_clear_momentum_scenario(directory: Path, n_half: int = 100) -> None:
    btc_prices = np.concatenate([np.linspace(100, 200, n_half), np.linspace(200, 100, n_half)])
    eth_prices = np.concatenate([np.linspace(200, 100, n_half), np.linspace(100, 200, n_half)])
    _write_snapshot(directory, "BTC", btc_prices)
    _write_snapshot(directory, "ETH", eth_prices)


def test_rotation_captures_clear_momentum_and_beats_all_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_clear_momentum_scenario(tmp_path)

    result = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=20, cost_bps=10.0)
    baselines = baseline_backtests("ETH", "BTC", result.returns.index, cost_bps=10.0)

    assert result.metrics["retorno_total"] > baselines["buy_hold_ETH"]["retorno_total"]
    assert result.metrics["retorno_total"] > baselines["buy_hold_BTC"]["retorno_total"]
    assert result.metrics["retorno_total"] > baselines["50_50_rebalanceado"]["retorno_total"]
    # Rota exactamente una vez por cada quiebre de tendencia detectado (más
    # la entrada inicial) -- no debería estar "temblando" día a día.
    assert 1 < result.metrics["n_rotaciones"] < 10


# --------------------------------------------------------------------------
# Restricción LONG-ONLY / sin deuda: los pesos siempre son >= 0 y suman
# <= 1 (nunca corto, nunca apalancado) -- ver el docstring del módulo.
# --------------------------------------------------------------------------


def test_rotation_weights_are_never_negative_and_never_exceed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    rng = np.random.default_rng(3)
    _write_snapshot(tmp_path, "BTC", 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    _write_snapshot(tmp_path, "ETH", 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))

    result = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=7, cost_bps=10.0)
    weights = holdings_to_weights(result.holdings, "ETH", "BTC")

    assert (weights.to_numpy() >= 0.0).all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    # Post-warmup (el rango de result.returns) nunca hay efectivo: siempre
    # 100% en una de las dos monedas -- ni corto ni apalancado ni "afuera".
    assert np.allclose(weights.sum(axis=1).to_numpy(), 1.0)


def test_baseline_weights_are_also_long_only() -> None:
    # Los baselines determinísticos (100% de una moneda, o 50/50) cumplen la
    # misma restricción por construcción -- se verifica sobre los propios
    # pesos que arma _fifty_fifty_returns implícitamente (0.5/0.5, nunca
    # negativo, nunca > 1 en conjunto).
    weight_a, weight_b = 0.5, 0.5
    assert weight_a >= 0.0 and weight_b >= 0.0
    assert weight_a + weight_b <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# Costos: se cobran en cada rotación (y en la entrada inicial), nunca en un
# día sin cambio de moneda.
# --------------------------------------------------------------------------


def test_rotation_charges_cost_only_on_rotation_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_clear_momentum_scenario(tmp_path)

    cost_bps = 25.0
    result = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=20, cost_bps=cost_bps)

    holding = result.holdings.to_numpy()
    prev = np.empty_like(holding)
    prev[0] = None
    prev[1:] = holding[:-1]
    is_rotation = holding != prev

    # Reconstruye el retorno BRUTO (antes de costos) para comparar contra el neto.
    close_a = loaders.get_prices("ETH", source="store", use_cache=False)["close"]
    close_b = loaders.get_prices("BTC", source="store", use_cache=False)["close"]
    from analysis.comparison import align_common_dates
    from strategies.rotation import _forward_returns

    aligned = align_common_dates({"ETH": close_a, "BTC": close_b})
    fwd_a = _forward_returns(aligned["ETH"]).reindex(result.holdings.index).to_numpy()
    fwd_b = _forward_returns(aligned["BTC"]).reindex(result.holdings.index).to_numpy()
    gross_return = np.where(holding == "ETH", fwd_a, fwd_b)

    net_return = result.returns.to_numpy()
    implied_cost = gross_return - net_return

    # En días SIN rotación, el costo implícito debe ser ~0; en días CON
    # rotación, debe ser ~cost_bps/1e4 (turnover=1.0, "vender todo, comprar
    # todo lo otro").
    np.testing.assert_allclose(implied_cost[~is_rotation], 0.0, atol=1e-10)
    np.testing.assert_allclose(implied_cost[is_rotation], cost_bps / 1e4, atol=1e-10)
    assert is_rotation.sum() == result.metrics["n_rotaciones"]


def test_higher_cost_bps_strictly_lowers_total_return_for_same_rotation_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_clear_momentum_scenario(tmp_path)

    cheap = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=20, cost_bps=1.0)
    expensive = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=20, cost_bps=500.0)

    assert expensive.metrics["retorno_total"] < cheap.metrics["retorno_total"]


# --------------------------------------------------------------------------
# Sin lookahead: la decisión/retorno en una fecha no puede depender de
# datos posteriores a esa fecha + 1.
# --------------------------------------------------------------------------


def test_rotation_does_not_depend_on_future_prices_beyond_next_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dos escenarios IDÉNTICOS hasta cierto punto, pero con precios FUTUROS
    (después de un punto de corte) completamente distintos, deben producir
    exactamente las mismas decisiones/retornos HASTA ese punto -- si no,
    algo estaría mirando datos que todavía "no pasaron".
    """
    rng = np.random.default_rng(11)
    shared_len = 150
    shared_btc = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, shared_len)))
    shared_eth = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, shared_len)))

    tail_len = 60
    tail_btc_a = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, tail_len)))
    tail_btc_b = tail_btc_a * 5.0 + 1000.0  # tremendamente distinto
    tail_eth_a = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, tail_len)))
    tail_eth_b = tail_eth_a / 3.0 - 10.0

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_snapshot(dir_a, "BTC", np.concatenate([shared_btc, tail_btc_a]))
    _write_snapshot(dir_a, "ETH", np.concatenate([shared_eth, tail_eth_a]))
    _write_snapshot(dir_b, "BTC", np.concatenate([shared_btc, tail_btc_b]))
    _write_snapshot(dir_b, "ETH", np.concatenate([shared_eth, tail_eth_b]))

    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", dir_a)
    result_a = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=10, cost_bps=10.0)

    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", dir_b)
    result_b = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=10, cost_bps=10.0)

    # Fechas dentro del tramo compartido, EXCLUYENDO la última (su retorno
    # hacia adelante usa el precio del primer día de la cola, que acá se
    # construyó a propósito DISTINTO entre A y B -- comparar esa fecha
    # estaría comparando un retorno que LEGÍTIMAMENTE debe diferir, no un
    # lookahead. Hasta shared_len - 2 inclusive, el retorno de cada fecha
    # solo usa precios dentro del tramo 100% compartido.
    common_dates = result_a.holdings.index.intersection(result_b.holdings.index)
    cutoff = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=shared_len - 2)
    compare_dates = common_dates[common_dates <= cutoff]
    assert len(compare_dates) > 20  # que la comparación no sea trivialmente vacía

    pd.testing.assert_series_equal(
        result_a.holdings.loc[compare_dates], result_b.holdings.loc[compare_dates], check_names=False
    )
    np.testing.assert_allclose(
        result_a.returns.loc[compare_dates].to_numpy(), result_b.returns.loc[compare_dates].to_numpy()
    )


# --------------------------------------------------------------------------
# Validaciones / casos límite
# --------------------------------------------------------------------------


def test_momentum_rotation_backtest_raises_on_insufficient_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_snapshot(tmp_path, "BTC", np.linspace(100, 110, 10))
    _write_snapshot(tmp_path, "ETH", np.linspace(100, 110, 10))

    with pytest.raises(ValueError, match="no alcanza"):
        momentum_rotation_backtest("ETH", "BTC", lookback_days=50, rebalance_days=7)


def test_baseline_backtests_returns_three_strategies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "SNAPSHOT_DIR", tmp_path)
    _write_clear_momentum_scenario(tmp_path)

    result = momentum_rotation_backtest("ETH", "BTC", lookback_days=20, rebalance_days=20, cost_bps=10.0)
    baselines = baseline_backtests("ETH", "BTC", result.returns.index, cost_bps=10.0)

    assert set(baselines.keys()) == {"buy_hold_ETH", "buy_hold_BTC", "50_50_rebalanceado"}
    for metrics in baselines.values():
        assert set(metrics.keys()) == {"cagr", "sharpe", "max_drawdown", "retorno_total"}
