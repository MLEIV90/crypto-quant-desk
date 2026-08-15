"""Tests offline para signals/suggester.py (series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.suggester import suggest


def _synthetic_ohlcv(
    n_flat: int, n_trend: int, drift: float, flat_noise: float = 0.004, trend_noise: float = 0.002, seed: int = 5
) -> pd.DataFrame:
    """OHLCV sintético: un tramo lateral/sin tendencia (`n_flat` velas, para
    que las medias/RSI/ADX etc. tengan warmup de sobra sin arrastrar
    tendencia previa) seguido de un tramo con `drift` diario sostenido
    (`n_trend` velas) — el "movimiento fuerte reciente" que debería
    inclinar el consenso de `suggest()`.
    """
    rng = np.random.default_rng(seed)
    flat_returns = rng.normal(0.0, flat_noise, n_flat)
    trend_returns = drift + rng.normal(0.0, trend_noise, n_trend)
    returns = np.concatenate([flat_returns, trend_returns])
    close = 100.0 * np.cumprod(1.0 + returns)
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.003, "low": close * 0.997, "close": close, "volume": 1000.0}, index=idx
    )


# --------------------------------------------------------------------------
# suggest: estructura del resultado
# --------------------------------------------------------------------------


def test_suggest_returns_expected_keys() -> None:
    df = _synthetic_ohlcv(n_flat=60, n_trend=15, drift=0.004)
    result = suggest(df)

    assert set(result.keys()) == {
        "sugerencia", "votos_alcistas", "votos_bajistas", "votos_neutrales",
        "confianza", "detalle", "desempeno_historico",
    }
    assert result["sugerencia"] in {"COMPRAR", "VENDER", "ESPERAR"}
    assert result["votos_alcistas"] + result["votos_bajistas"] + result["votos_neutrales"] == len(result["detalle"])
    assert 0.0 <= result["confianza"] <= 1.0
    assert set(result["detalle"].keys()) == {"rsi", "medias", "macd", "estocastico", "bollinger", "pivotes"}
    assert set(result["detalle"].values()).issubset({"alcista", "bajista", "neutral"})

    desempeno = result["desempeno_historico"]
    assert set(desempeno.keys()) == {
        "cagr_sugeridor", "sharpe_sugeridor", "max_drawdown_sugeridor", "n_trades_sugeridor",
        "cagr_buy_and_hold", "sharpe_buy_and_hold", "max_drawdown_buy_and_hold",
    }


# --------------------------------------------------------------------------
# suggest: votos consistentes con una serie direccional clara
# --------------------------------------------------------------------------


def test_suggest_strong_uptrend_gives_more_bullish_votes() -> None:
    df = _synthetic_ohlcv(n_flat=60, n_trend=15, drift=0.004, seed=5)
    result = suggest(df)

    assert result["votos_alcistas"] > result["votos_bajistas"]
    assert result["sugerencia"] == "COMPRAR"
    assert result["confianza"] > 0.0


def test_suggest_strong_downtrend_gives_more_bearish_votes() -> None:
    df = _synthetic_ohlcv(n_flat=60, n_trend=15, drift=-0.004, seed=5)
    result = suggest(df)

    assert result["votos_bajistas"] > result["votos_alcistas"]
    assert result["sugerencia"] == "VENDER"
    assert result["confianza"] > 0.0


def test_suggest_waits_on_a_tie() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=idx
    )

    result = suggest(df)

    # Serie plana y corta (sin warmup de ningún indicador): todos los
    # estudios quedan neutrales -> empate 0 a 0 -> ESPERAR, confianza 0.
    assert result["sugerencia"] == "ESPERAR"
    assert result["votos_alcistas"] == result["votos_bajistas"]
    assert result["confianza"] == pytest.approx(0.0)
