"""Engine integrador de decisión: la "calculadora" de crypto-quant-desk.

Combina indicadores técnicos en un score compuesto, lo traduce en una
posición dimensionada por volatilidad, y arma la recomendación final para
el usuario sobre la última vela disponible.

REGLA ANTI-LOOKAHEAD (léase antes de usar este módulo): toda función de este
módulo usa, para la fecha t, solo información disponible hasta el CIERRE de
t. Los indicadores de `signals.indicators` ya son trailing (rolling/ewm
hacia atrás) y la vol realizada usada para el sizing también es una ventana
rolling hacia atrás — nada mira el futuro. `generate_positions` NO aplica
ningún `shift`: la posición que devuelve para la fecha t es la decisión
tomada CON el cierre de t, y es responsabilidad exclusiva de
`backtest.engine.run_backtest` desplazarla un día antes de aplicarla a un
retorno. Aplicar el shift acá TAMBIÉN duplicaría el desfase y invalidaría el
backtest (dos shifts es peor que ninguno, no es "más seguro").

Este módulo NO reimplementa nada de lo ya construido: reutiliza
`signals.indicators.add_all_indicators`, `signals.returns` (simple y log),
`models.garch` (para el régimen de volatilidad de `latest_recommendation`)
y `backtest.engine.backtest_from_prices` (para backtestear las posiciones
generadas acá, ver el bloque de validación en el README/notebook de la
fase). Solo agrega la lógica de scoring y sizing.

Convención del proyecto: retornos en escala DECIMAL, frecuencia diaria,
`config.PERIODS_PER_YEAR` (365 para cripto). Los tres componentes del score
(`trend`, `momentum`, `mean_reversion`) y el score compuesto viven siempre
en [-1, 1].
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    ALLOW_SHORT,
    MAX_LEVERAGE,
    PERIODS_PER_YEAR,
    SCORE_THRESHOLD_LONG,
    SCORE_THRESHOLD_SHORT,
    SIGNAL_WEIGHTS,
    TARGET_VOL,
    VOL_WINDOW,
)
from models.garch import conditional_volatility, select_best_model, volatility_regime
from signals.indicators import add_all_indicators
from signals.returns import log_returns, simple_returns

logger = logging.getLogger(__name__)

# Columnas de indicadores que necesitan los componentes del score; si el df
# de entrada no las tiene todas, se calculan con add_all_indicators.
_REQUIRED_INDICATOR_COLUMNS: set[str] = {"ema_12", "ema_26", "macd_hist", "rsi_14", "bb_zscore"}

# Factor de escala usado para llevar el gap EMA/MACD (una fracción del
# precio, típicamente chica) a un rango donde tanh satura de forma suave
# dentro de movimientos de tendencia razonables para cripto.
_TREND_TANH_SCALE = 50.0
# Ídem para el mapeo de RSI: RSI ya da (rsi-50)/50 en [-1,1] linealmente;
# este factor solo suaviza la saturación cerca de los extremos (RSI ~0/100).
_MOMENTUM_TANH_SCALE = 1.5


def compute_signal_components(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los tres componentes del score, cada uno en [-1, 1].

    `df` es un OHLCV estandarizado (el de `data.loaders.get_prices`); si no
    tiene ya las columnas de indicadores necesarias, se calculan acá mismo
    con `signals.indicators.add_all_indicators` (no se reimplementa nada,
    solo se llama).

    Componentes:
    - "trend": combina (a) el gap relativo EMA rápida (12) vs. lenta (26)
      respecto del precio, y (b) el histograma de MACD relativo al precio,
      cada uno pasado por `tanh` para saturar suavemente en vez de crecer
      sin límite. EMA rápida por encima de la lenta y MACD positivo =>
      tendencia alcista => trend > 0.
    - "momentum": RSI mapeado linealmente a [-1,1] vía (rsi-50)/50, con una
      capa adicional de `tanh` para suavizar la saturación cerca de los
      extremos (RSI muy sobrecomprado/sobrevendido).
    - "mean_reversion": -tanh(zscore de Bollinger). Precio muy por encima de
      su media móvil (zscore alto) => señal de reversión a la baja (venta);
      precio muy por debajo => señal de reversión al alza (compra).
    """
    if not _REQUIRED_INDICATOR_COLUMNS.issubset(df.columns):
        df = add_all_indicators(df)

    ema_gap = (df["ema_12"] - df["ema_26"]) / df["close"]
    macd_hist_norm = df["macd_hist"] / df["close"]
    trend = 0.5 * np.tanh(ema_gap * _TREND_TANH_SCALE) + 0.5 * np.tanh(macd_hist_norm * _TREND_TANH_SCALE)
    trend = trend.clip(-1.0, 1.0)

    momentum = np.tanh(((df["rsi_14"] - 50.0) / 50.0) * _MOMENTUM_TANH_SCALE)

    mean_reversion = -np.tanh(df["bb_zscore"])

    return pd.DataFrame(
        {"trend": trend, "momentum": momentum, "mean_reversion": mean_reversion}, index=df.index
    )


def composite_score(components: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Suma ponderada de los componentes del score, recortada a [-1, 1].

    `weights` (default `config.SIGNAL_WEIGHTS`) debe sumar 1 — si no, se
    loguea un warning (no se aborta: los pesos se usan tal cual, la suma
    ponderada simplemente queda en una escala distinta antes del recorte).
    """
    if weights is None:
        weights = SIGNAL_WEIGHTS

    missing = set(weights) - set(components.columns)
    if missing:
        raise ValueError(f"composite_score: faltan columnas {missing} en 'components'")

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        logger.warning("composite_score: los pesos suman %.4f, no 1.0 (%s)", total_weight, weights)

    score = pd.Series(0.0, index=components.index)
    for name, weight in weights.items():
        score = score + components[name] * weight

    return score.clip(-1.0, 1.0)


def size_from_volatility(
    realized_vol: pd.Series, target_vol: float | None = None, max_leverage: float | None = None
) -> pd.Series:
    """Vol targeting: tamaño de posición inversamente proporcional a la
    volatilidad realizada, acotado a [0, max_leverage].

        tamaño_t = clip(target_vol / realized_vol_t, 0, max_leverage)

    A mayor volatilidad realizada, MENOR el tamaño — la lógica estándar de
    vol targeting: tomar menos riesgo de posición cuando el mercado está más
    agitado, para que el riesgo del PORTAFOLIO (no el del activo, que
    cambia con el tiempo) se mantenga aproximadamente constante. Si
    `realized_vol` es 0 o NaN (p. ej. warmup de la ventana rolling), el
    tamaño resultante es 0 en vez de +inf o NaN — no se puede escalar de
    forma sensata por una volatilidad nula o desconocida.
    """
    if target_vol is None:
        target_vol = TARGET_VOL
    if max_leverage is None:
        max_leverage = MAX_LEVERAGE

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_size = target_vol / realized_vol
    raw_size = raw_size.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return raw_size.clip(lower=0.0, upper=max_leverage)


def _apply_dead_zone(score: pd.Series) -> pd.Series:
    """Aplica la zona muerta de config (SCORE_THRESHOLD_LONG/SHORT): dentro
    de la zona muerta, la dirección es 0 (FLAT); fuera, se conserva el valor
    continuo del score (no solo su signo), para que la convicción del score
    siga influyendo en el tamaño final de la posición.
    """
    direction = score.copy()
    dead_zone = (score > SCORE_THRESHOLD_SHORT) & (score < SCORE_THRESHOLD_LONG)
    direction[dead_zone] = 0.0
    return direction


def _classify_action(score: float) -> str:
    """Traduce un score puntual a la etiqueta LONG/FLAT/SHORT (mismos
    umbrales que `_apply_dead_zone`, para que la acción mostrada al usuario
    sea siempre consistente con la posición generada).
    """
    if score > SCORE_THRESHOLD_LONG:
        return "LONG"
    if score < SCORE_THRESHOLD_SHORT:
        return "SHORT"
    return "FLAT"


def generate_positions(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
    allow_short: bool | None = None,
    target_vol: float | None = None,
    vol_window: int | None = None,
) -> pd.Series:
    """Pipeline completo: precios -> componentes -> score compuesto -> zona
    muerta -> ALLOW_SHORT -> sizing por vol realizada -> posición final en
    [-max_leverage, max_leverage].

    Para el sizing se usa volatilidad REALIZADA rolling (`asset_returns`
    simples, ventana `vol_window`, anualizada con `config.PERIODS_PER_YEAR`)
    — rápida y trailing, NO un GARCH ajustado día a día (eso sería
    prohibitivamente lento en un backtest y no es lo que pide este
    pipeline; para el régimen de volatilidad vía GARCH ver
    `latest_recommendation`, que sí ajusta uno, pero solo para la última foto).

    `prices` es un OHLCV estandarizado (el de `data.loaders.get_prices`).
    Ver la REGLA ANTI-LOOKAHEAD del docstring del módulo: esta función NO
    desplaza la posición resultante; el desfase lo aplica
    `backtest.engine.run_backtest`.
    """
    if allow_short is None:
        allow_short = ALLOW_SHORT
    if vol_window is None:
        vol_window = VOL_WINDOW

    components = compute_signal_components(prices)
    score = composite_score(components, weights=weights)
    direction = _apply_dead_zone(score)

    if not allow_short:
        direction = direction.clip(lower=0.0)

    asset_returns = simple_returns(prices["close"])
    realized_vol = asset_returns.rolling(window=vol_window).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    size = size_from_volatility(realized_vol, target_vol=target_vol)

    positions = (direction * size).clip(-MAX_LEVERAGE, MAX_LEVERAGE)
    positions.name = "position"
    return positions


def generate_positions_vol_targeting(
    prices: pd.DataFrame,
    target_vol: float | None = None,
    vol_window: int | None = None,
) -> pd.Series:
    """Estrategia "vol targeting puro" (Fase 21, pestaña Backtest): SIEMPRE
    largo (direction=1) — a diferencia de `generate_positions`, esta función
    NO usa el score direccional del engine, no intenta adivinar si conviene
    estar largo o corto. Solo ajusta el TAMAÑO de la posición según la
    volatilidad realizada reciente, vía `size_from_volatility` (la misma
    pieza de sizing que usa `generate_positions`, reutilizada tal cual).
    """
    if vol_window is None:
        vol_window = VOL_WINDOW

    asset_returns = simple_returns(prices["close"])
    realized_vol = asset_returns.rolling(window=vol_window).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    positions = size_from_volatility(realized_vol, target_vol=target_vol).clip(0.0, MAX_LEVERAGE)
    positions.name = "position"
    return positions


def generate_positions_engine_signal(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
    allow_short: bool | None = None,
) -> pd.Series:
    """Estrategia "señal del engine de consenso" (Fase 21, pestaña Backtest):
    la dirección técnica compuesta (trend/momentum/mean-reversion -> score
    -> zona muerta, ver `compute_signal_components`/`composite_score`/
    `_apply_dead_zone`), SIN vol targeting — a diferencia de
    `generate_positions`, el tamaño de la posición es el score saturado tal
    cual (en [-1, 1], o [0, 1] si `allow_short=False`), no ajustado por
    volatilidad.
    """
    if allow_short is None:
        allow_short = ALLOW_SHORT

    components = compute_signal_components(prices)
    score = composite_score(components, weights=weights)
    direction = _apply_dead_zone(score)

    if not allow_short:
        direction = direction.clip(lower=0.0)

    direction.name = "position"
    return direction


def latest_recommendation(prices: pd.DataFrame, garch_regime: bool = True) -> dict:
    """Recomendación de decisión para la ÚLTIMA vela de `prices`: la salida
    pensada para el usuario final (no para el backtester).

    Devuelve un dict:
    - "accion": "LONG" / "FLAT" / "SHORT" (ver `_classify_action`).
    - "score": score compuesto en [-1, 1] de la última vela.
    - "confianza": |score| (0 = sin convicción, 1 = convicción máxima).
    - "tamaño_sugerido": la posición que devolvería `generate_positions`
      para esta última fecha (ya con zona muerta, ALLOW_SHORT y vol
      targeting aplicados).
    - "desglose": dict con los tres componentes (trend/momentum/mean_reversion)
      de la última vela, para poder explicar el score.
    - "vol_realizada": volatilidad realizada anualizada (rolling, la misma
      que usa el sizing).

    Si `garch_regime=True` (default), ajusta UN modelo GARCH
    (`models.garch.select_best_model`, sobre log-retornos, su convención
    documentada) solo para esta foto — no se usa para el sizing del
    backtest, que es deliberadamente más liviano (ver `generate_positions`)
    — y agrega:
    - "vol_garch": volatilidad condicional GARCH anualizada de la última vela.
    - "regimen": "calma" / "normal" / "tension" (o None si no hay suficiente
      historia para calcularlo, ver `models.garch.volatility_regime`).
    """
    components = compute_signal_components(prices)
    score = composite_score(components)
    positions = generate_positions(prices)

    last_score = float(score.iloc[-1])
    asset_returns = simple_returns(prices["close"])
    realized_vol = float(
        (asset_returns.rolling(window=VOL_WINDOW).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)).iloc[-1]
    )

    result: dict = {
        "accion": _classify_action(last_score),
        "score": last_score,
        "confianza": abs(last_score),
        "tamaño_sugerido": float(positions.iloc[-1]),
        "desglose": {
            "trend": float(components["trend"].iloc[-1]),
            "momentum": float(components["momentum"].iloc[-1]),
            "mean_reversion": float(components["mean_reversion"].iloc[-1]),
        },
        "vol_realizada": realized_vol,
    }

    if garch_regime:
        garch_returns = log_returns(prices["close"]).dropna()
        best = select_best_model(garch_returns, criterion="aic")
        cond_vol = conditional_volatility(best["result"])
        regime = volatility_regime(cond_vol)
        last_regime = regime.iloc[-1]

        result["vol_garch"] = float(cond_vol.iloc[-1])
        result["regimen"] = last_regime if pd.notna(last_regime) else None
        logger.info(
            "latest_recommendation: modelo GARCH %s/%s, vol_garch=%.2f%%, regimen=%s",
            best["vol"], best["dist"], result["vol_garch"] * 100, result["regimen"],
        )

    logger.info(
        "latest_recommendation: accion=%s score=%.3f tamaño_sugerido=%.3f",
        result["accion"], last_score, result["tamaño_sugerido"],
    )
    return result
