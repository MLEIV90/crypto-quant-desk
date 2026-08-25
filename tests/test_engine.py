"""Tests offline para signals/engine.py (OHLCV sintético, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import SCORE_THRESHOLD_LONG, SCORE_THRESHOLD_SHORT
from signals.engine import (
    compute_signal_components,
    composite_score,
    generate_positions,
    generate_positions_engine_signal,
    generate_positions_vol_targeting,
    latest_recommendation,
    size_from_volatility,
)


def _synthetic_ohlcv(n: int = 400, drift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """OHLCV sintético con OHLC degenerado (open=high=low=close), suficiente
    para testear el motor de señales: ninguno de sus componentes usa
    high/low de forma distinta al close.
    """
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    )


# --------------------------------------------------------------------------
# Rango de los componentes y del score
# --------------------------------------------------------------------------


def test_components_and_composite_score_are_bounded_in_unit_interval() -> None:
    df = _synthetic_ohlcv(n=400, drift=0.0)
    components = compute_signal_components(df)

    for col in ("trend", "momentum", "mean_reversion"):
        valid = components[col].dropna()
        assert len(valid) > 0
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    score = composite_score(components)
    valid_score = score.dropna()
    assert (valid_score >= -1.0).all() and (valid_score <= 1.0).all()


def test_composite_score_warns_on_weights_not_summing_to_one(caplog: pytest.LogCaptureFixture) -> None:
    df = _synthetic_ohlcv(n=100)
    components = compute_signal_components(df)

    with caplog.at_level("WARNING", logger="signals.engine"):
        composite_score(components, weights={"trend": 0.5, "momentum": 0.5, "mean_reversion": 0.5})

    assert any("suman" in record.message for record in caplog.records)


def test_composite_score_raises_on_missing_columns() -> None:
    components = pd.DataFrame({"trend": [0.1, 0.2]})
    with pytest.raises(ValueError):
        composite_score(components)


# --------------------------------------------------------------------------
# Tendencia fuerte => trend > 0 y posiciones mayormente largas
# --------------------------------------------------------------------------


def test_strong_uptrend_gives_positive_trend_and_mostly_long_positions() -> None:
    # drift=0.02/día es una tendencia deliberadamente extrema (no realista
    # como calibración de mercado, sí como caso de test inequívoco): con una
    # tendencia más moderada, el componente mean_reversion (que por diseño
    # se OPONE a movimientos sostenidos, ver compute_signal_components)
    # empuja el score compuesto justo al borde de la zona muerta y la
    # mayoría deja de ser clara — acá se busca una tendencia tan fuerte que
    # trend+momentum la dominen con margen.
    df = _synthetic_ohlcv(n=400, drift=0.02)

    components = compute_signal_components(df)
    trend_after_warmup = components["trend"].iloc[100:]
    assert (trend_after_warmup > 0).mean() > 0.9  # casi siempre positivo

    positions = generate_positions(df)
    positions_after_warmup = positions.iloc[100:].dropna()
    assert (positions_after_warmup > 0).mean() > 0.85


def test_strong_downtrend_gives_negative_trend() -> None:
    df = _synthetic_ohlcv(n=400, drift=-0.01)
    components = compute_signal_components(df)
    trend_after_warmup = components["trend"].iloc[100:]
    assert (trend_after_warmup < 0).mean() > 0.9


# --------------------------------------------------------------------------
# ALLOW_SHORT=False
# --------------------------------------------------------------------------


def test_allow_short_false_never_gives_negative_positions() -> None:
    df = _synthetic_ohlcv(n=400, drift=-0.01)  # tendencia bajista: sin el flag, daría cortos

    positions_with_short = generate_positions(df, allow_short=True)
    positions_without_short = generate_positions(df, allow_short=False)

    assert (positions_with_short.dropna() < 0).any()  # confirma que el escenario SÍ generaría cortos
    assert (positions_without_short.dropna() >= 0).all()


# --------------------------------------------------------------------------
# size_from_volatility
# --------------------------------------------------------------------------


def test_size_from_volatility_is_inversely_related_to_vol() -> None:
    low_vol = pd.Series([0.10] * 5)
    high_vol = pd.Series([0.80] * 5)

    size_low = size_from_volatility(low_vol, target_vol=0.5, max_leverage=100.0)
    size_high = size_from_volatility(high_vol, target_vol=0.5, max_leverage=100.0)

    assert (size_low > size_high).all()
    assert size_low.iloc[0] == pytest.approx(0.5 / 0.10)
    assert size_high.iloc[0] == pytest.approx(0.5 / 0.80)


def test_size_from_volatility_respects_max_leverage() -> None:
    tiny_vol = pd.Series([0.001] * 5)  # target/vol sería enorme sin el cap
    size = size_from_volatility(tiny_vol, target_vol=0.5, max_leverage=1.0)
    assert np.allclose(size.to_numpy(), 1.0)


def test_size_from_volatility_zero_vol_gives_zero_size() -> None:
    zero_vol = pd.Series([0.0, 0.0])
    size = size_from_volatility(zero_vol, target_vol=0.5, max_leverage=1.0)
    assert (size == 0.0).all()


# --------------------------------------------------------------------------
# latest_recommendation
# --------------------------------------------------------------------------


def test_latest_recommendation_has_expected_keys_without_garch() -> None:
    df = _synthetic_ohlcv(n=200, drift=0.01)
    rec = latest_recommendation(df, garch_regime=False)

    assert set(rec.keys()) == {"accion", "score", "confianza", "tamaño_sugerido", "desglose", "vol_realizada"}
    assert rec["accion"] in ("LONG", "FLAT", "SHORT")
    assert set(rec["desglose"].keys()) == {"trend", "momentum", "mean_reversion"}
    assert rec["confianza"] == pytest.approx(abs(rec["score"]))


def test_latest_recommendation_accion_matches_score_thresholds() -> None:
    df = _synthetic_ohlcv(n=300, drift=0.01)
    rec = latest_recommendation(df, garch_regime=False)

    if rec["score"] > SCORE_THRESHOLD_LONG:
        assert rec["accion"] == "LONG"
    elif rec["score"] < SCORE_THRESHOLD_SHORT:
        assert rec["accion"] == "SHORT"
    else:
        assert rec["accion"] == "FLAT"


def test_latest_recommendation_with_garch_regime_adds_expected_keys() -> None:
    df = _synthetic_ohlcv(n=500, drift=0.0005, seed=3)
    rec = latest_recommendation(df, garch_regime=True)

    assert "vol_garch" in rec and "regimen" in rec
    assert rec["vol_garch"] > 0
    assert rec["regimen"] in ("calma", "normal", "tension", None)


# --------------------------------------------------------------------------
# Fase 21: estrategias "puras" para el selector de la pestaña Backtest
# --------------------------------------------------------------------------


def test_generate_positions_vol_targeting_is_always_long_only() -> None:
    # Tendencia bajista fuerte: si esto mirara la dirección, daría posiciones
    # negativas (ver test_strong_downtrend_gives_negative_trend) — vol
    # targeting puro NO debe, por diseño, adivinar la dirección.
    df = _synthetic_ohlcv(n=400, drift=-0.02)
    positions = generate_positions_vol_targeting(df)
    assert (positions.dropna() >= 0.0).all()


def test_generate_positions_vol_targeting_shrinks_size_when_vol_rises() -> None:
    # Dos tramos: calmo primero, agitado después (mismo patrón que el test
    # de coherencia de riesgo de Fase 20c) — el tamaño de posición promedio
    # del tramo agitado debe ser menor.
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0, 0.005, 300)
    stressed = rng.normal(0.0, 0.05, 300)
    close = 100.0 * np.exp(np.cumsum(np.concatenate([calm, stressed])))
    idx = pd.date_range("2020-01-01", periods=600, freq="D", tz="UTC")
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx)

    positions = generate_positions_vol_targeting(df)
    calm_size = positions.iloc[100:300].mean()
    stressed_size = positions.iloc[400:600].mean()
    assert stressed_size < calm_size


def test_generate_positions_vol_targeting_respects_max_leverage_default() -> None:
    df = _synthetic_ohlcv(n=300, drift=0.0)
    positions = generate_positions_vol_targeting(df, target_vol=0.3)
    assert positions.max() <= 1.0 + 1e-9  # MAX_LEVERAGE por defecto


def test_generate_positions_engine_signal_has_no_vol_sizing() -> None:
    # A diferencia de generate_positions (combo), la magnitud acá es
    # directamente el score saturado: nunca puede superar 1.0 en valor
    # absoluto, sea cual sea la volatilidad del activo.
    df = _synthetic_ohlcv(n=400, drift=0.02)
    positions = generate_positions_engine_signal(df)
    assert (positions.dropna().abs() <= 1.0 + 1e-9).all()


def test_generate_positions_engine_signal_matches_dead_zone_direction() -> None:
    # La dirección de la señal pura del engine debe coincidir en signo con
    # generate_positions (combo) salvo por el sizing — mismo score
    # subyacente, ver compute_signal_components/composite_score.
    df = _synthetic_ohlcv(n=400, drift=0.02)
    engine_only = generate_positions_engine_signal(df).dropna()
    combo = generate_positions(df).dropna()

    both_nonzero = (engine_only != 0.0) & (combo != 0.0)
    assert (np.sign(engine_only[both_nonzero]) == np.sign(combo[both_nonzero])).all()


def test_generate_positions_engine_signal_allow_short_false_never_negative() -> None:
    df = _synthetic_ohlcv(n=400, drift=-0.02)
    positions_with_short = generate_positions_engine_signal(df, allow_short=True)
    positions_without_short = generate_positions_engine_signal(df, allow_short=False)

    assert (positions_with_short.dropna() < 0.0).any()
    assert (positions_without_short.dropna() >= 0.0).all()
