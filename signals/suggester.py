"""Sugeridor combinado de crypto-quant-desk (Fase 7a): "consenso de
indicadores" sobre los estudios de `signals.studies`/`signals.indicators`.

HONESTIDAD ANTE TODO — léase antes de usar este módulo: `suggest()` NO es
un sistema con edge direccional demostrado. Es un conteo de votos de reglas
técnicas clásicas (RSI, medias, MACD, estocástico, Bollinger, pivotes),
exactamente el tipo de indicador que este proyecto ya evaluó formalmente
(`ml/models.py`) y encontró sin ventaja consistente sobre un baseline
trivial. Por eso `suggest()` SIEMPRE devuelve, junto con la sugerencia, un
`desempeno_historico`: un backtest real (`backtest.engine.backtest_from_prices`,
reutilizado tal cual) de esta MISMA regla de consenso contra buy & hold,
con costos de transacción incluidos — para que la sugerencia nunca se
muestre sin su propio historial de desempeño al lado. Si ese desempeño es
malo, se reporta tal cual: este módulo no filtra ni maquilla el resultado.

Reutiliza `signals.indicators.add_all_indicators` y `signals.studies.stochastic`
tal cual — no reimplementa ningún cálculo de indicador.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtest.engine import backtest_from_prices, compare_to_buy_and_hold
from signals.indicators import add_all_indicators
from signals.returns import simple_returns
from signals.studies import stochastic

logger = logging.getLogger(__name__)

# Umbrales de sobrecompra/sobreventa — convención de mercado estándar, no
# calibrados estadísticamente (mismo espíritu honesto que el resto del
# módulo: son las reglas de libro de texto, no un resultado de research).
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0
STOCH_OVERBOUGHT: float = 80.0
STOCH_OVERSOLD: float = 20.0

_VOTE_LABELS: dict[int, str] = {1: "alcista", -1: "bajista", 0: "neutral"}


def _rsi_votes(rsi: pd.Series) -> pd.Series:
    """+1 (alcista) si RSI < sobrevendido (posible rebote), -1 (bajista) si
    RSI > sobrecomprado (posible corrección), 0 en el resto — incluido el
    warmup (RSI en NaN no dispara ningún voto).
    """
    votes = pd.Series(0, index=rsi.index, dtype=int)
    votes[rsi > RSI_OVERBOUGHT] = -1
    votes[rsi < RSI_OVERSOLD] = 1
    return votes


def _moving_average_votes(close: pd.Series, sma_20: pd.Series, sma_50: pd.Series) -> pd.Series:
    """+1 si el precio está por ENCIMA de ambas medias (tendencia alcista
    de corto y mediano plazo alineadas), -1 si está por DEBAJO de ambas,
    0 si están mezcladas (una a favor, otra en contra — sin consenso).
    """
    votes = pd.Series(0, index=close.index, dtype=int)
    votes[(close > sma_20) & (close > sma_50)] = 1
    votes[(close < sma_20) & (close < sma_50)] = -1
    return votes


def _macd_votes(macd_hist: pd.Series) -> pd.Series:
    """+1 si el histograma MACD es positivo (línea MACD sobre la señal,
    momentum alcista), -1 si es negativo, 0 si es exactamente 0 o NaN.
    """
    return np.sign(macd_hist).fillna(0.0).astype(int)


def _stochastic_votes(stoch_k: pd.Series) -> pd.Series:
    """+1 (alcista) si %K < sobrevendido, -1 (bajista) si %K > sobrecomprado,
    0 en el resto (incluido NaN por rango cero, ver `signals.studies.stochastic`).
    """
    votes = pd.Series(0, index=stoch_k.index, dtype=int)
    votes[stoch_k > STOCH_OVERBOUGHT] = -1
    votes[stoch_k < STOCH_OVERSOLD] = 1
    return votes


def _bollinger_votes(bb_pct_b: pd.Series) -> pd.Series:
    """+1 si el precio rompió la banda INFERIOR (`bb_pct_b < 0`, posible
    sobreventa/rebote), -1 si rompió la banda SUPERIOR (`bb_pct_b > 1`,
    posible sobrecompra/corrección), 0 si está dentro de las bandas.
    """
    votes = pd.Series(0, index=bb_pct_b.index, dtype=int)
    votes[bb_pct_b > 1.0] = -1
    votes[bb_pct_b < 0.0] = 1
    return votes


def _rolling_pivot(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Versión TRAILING (serie completa, una por vela) del punto pivote
    clásico de `signals.studies.pivot_points` (misma fórmula, solo el
    nivel "P" — es el único que necesita el voto de consenso): calculado
    con el high/low/close de la vela ANTERIOR (`shift(1)`) para proyectar
    el pivote vigente en la vela actual — la misma convención de "período
    ya cerrado proyecta el siguiente" que usa `pivot_points`, pero aplicada
    vela a vela en vez de una sola vez. Trailing por construcción: el
    pivote de la vela t usa datos hasta t-1, sin lookahead.
    """
    return (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0


def _pivot_votes(close: pd.Series, pivot: pd.Series) -> pd.Series:
    """+1 si el precio cerró por ENCIMA del pivote (sesgo alcista para el
    período), -1 si cerró por debajo, 0 si son iguales o el pivote es NaN
    (warmup: la primera vela no tiene vela anterior).
    """
    return np.sign(close - pivot).fillna(0.0).astype(int)


def _compute_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Arma la matriz de votos completa (una fila por vela, una columna por
    estudio), en {-1, 0, +1} — la ÚNICA fuente de verdad tanto para la
    sugerencia de la última vela como para el backtest histórico de
    `_historical_performance`, para que ambos sean SIEMPRE consistentes
    entre sí (nunca dos implementaciones separadas de la misma regla).
    """
    high, low, close = df["high"], df["low"], df["close"]
    indicators = add_all_indicators(df)
    stoch_df = stochastic(high, low, close)
    pivot = _rolling_pivot(high, low, close)

    return pd.DataFrame(
        {
            "rsi": _rsi_votes(indicators["rsi_14"]),
            "medias": _moving_average_votes(close, indicators["sma_20"], indicators["sma_50"]),
            "macd": _macd_votes(indicators["macd_hist"]),
            "estocastico": _stochastic_votes(stoch_df["stoch_k"]),
            "bollinger": _bollinger_votes(indicators["bb_pct_b"]),
            "pivotes": _pivot_votes(close, pivot),
        },
        index=df.index,
    )


def _historical_performance(close: pd.Series, votes_df: pd.DataFrame) -> dict:
    """Backtest rápido y honesto de la regla de consenso: convierte el
    voto neto de cada vela (`suma_de_votos / cantidad_de_estudios`, en
    [-1, 1]) en una posición y la corre por
    `backtest.engine.backtest_from_prices` (con costos de transacción,
    reutilizado tal cual — no se reimplementa el backtester acá), comparada
    contra buy & hold vía `compare_to_buy_and_hold`.

    NO aplica ningún `shift` acá: la posición de la vela t usa información
    disponible hasta el cierre de t (los votos ya son trailing, ver cada
    `_*_votes`), y es responsabilidad exclusiva de `backtest.engine.run_backtest`
    desplazarla un día antes de aplicarla a un retorno — misma regla que
    `signals.engine.generate_positions` en el resto del proyecto.
    """
    n_estudios = votes_df.shape[1]
    positions = (votes_df.sum(axis=1) / n_estudios).clip(-1.0, 1.0)
    positions.name = "position"

    result = backtest_from_prices(close, positions)
    asset_returns = simple_returns(close)
    comparison = compare_to_buy_and_hold(asset_returns, result)

    estrategia = comparison["estrategia"]
    buy_and_hold = comparison["buy_and_hold"]

    return {
        "cagr_sugeridor": estrategia["cagr"],
        "sharpe_sugeridor": estrategia["sharpe"],
        "max_drawdown_sugeridor": estrategia["max_drawdown"],
        "n_trades_sugeridor": estrategia["n_trades"],
        "cagr_buy_and_hold": buy_and_hold["cagr"],
        "sharpe_buy_and_hold": buy_and_hold["sharpe"],
        "max_drawdown_buy_and_hold": buy_and_hold["max_drawdown"],
    }


def suggest(df: pd.DataFrame) -> dict:
    """Sugerencia de "consenso de indicadores" para la ÚLTIMA vela de `df`
    (un OHLCV estandarizado, el de `data.loaders.get_prices`).

    Cuenta los votos de 6 estudios clásicos (`_compute_votes`: RSI,
    posición del precio vs. medias móviles, histograma MACD, estocástico,
    posición vs. bandas de Bollinger, posición vs. pivote) y arma la
    sugerencia por MAYORÍA simple de votos:
    - "COMPRAR" si hay más votos alcistas que bajistas.
    - "VENDER" si hay más votos bajistas que alcistas.
    - "ESPERAR" si están empatados (incluido 0 a 0, todo neutral).

    Ver el HONESTIDAD ANTE TODO del docstring del módulo: esto es un conteo
    de reglas técnicas de libro de texto, no un sistema con edge probado —
    por eso SIEMPRE viaja junto con `desempeno_historico`.

    Devuelve un dict:
    - "sugerencia": "COMPRAR" / "VENDER" / "ESPERAR".
    - "votos_alcistas" / "votos_bajistas" / "votos_neutrales": conteos (int).
    - "confianza": `|votos_alcistas - votos_bajistas| / total_estudios`, en
      [0, 1] — 0 = empate total, 1 = todos los estudios coinciden.
    - "detalle": dict `{nombre_estudio: "alcista"/"bajista"/"neutral"}`
      para la última vela.
    - "desempeno_historico": ver `_historical_performance` — SIEMPRE
      presente, nunca opcional.
    """
    close = df["close"]
    votes_df = _compute_votes(df)

    last_votes = votes_df.iloc[-1]
    votos_alcistas = int((last_votes == 1).sum())
    votos_bajistas = int((last_votes == -1).sum())
    votos_neutrales = int((last_votes == 0).sum())
    total_estudios = len(last_votes)

    if votos_alcistas > votos_bajistas:
        sugerencia = "COMPRAR"
    elif votos_bajistas > votos_alcistas:
        sugerencia = "VENDER"
    else:
        sugerencia = "ESPERAR"

    confianza = abs(votos_alcistas - votos_bajistas) / total_estudios

    detalle = {columna: _VOTE_LABELS[int(last_votes[columna])] for columna in votes_df.columns}

    desempeno_historico = _historical_performance(close, votes_df)

    logger.info(
        "suggest: sugerencia=%s (alcistas=%d bajistas=%d neutrales=%d confianza=%.2f) "
        "sharpe_sugeridor=%.2f vs sharpe_buy_and_hold=%.2f",
        sugerencia, votos_alcistas, votos_bajistas, votos_neutrales, confianza,
        desempeno_historico["sharpe_sugeridor"], desempeno_historico["sharpe_buy_and_hold"],
    )

    return {
        "sugerencia": sugerencia,
        "votos_alcistas": votos_alcistas,
        "votos_bajistas": votos_bajistas,
        "votos_neutrales": votos_neutrales,
        "confianza": confianza,
        "detalle": detalle,
        "desempeno_historico": desempeno_historico,
    }
