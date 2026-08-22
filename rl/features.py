"""Dataset de observaciones + retornos para `rl.env.PortfolioEnv` (Fase 18).

Reutiliza `ml.features.build_feature_matrix` (Fase 3a/6a) para las features
por activo — TODAS estacionarias/trailing por construcción, ver ese módulo
— y solo agrega lo que no existía: la covarianza rolling ENTRE activos (útil
para un asignador de cartera, no para un modelo de un solo activo) y el
ensamblado final en un array numérico lista para un `gymnasium.Env`.

REGLA ANTI-LOOKAHEAD: la fila t de `obs_features` usa exclusivamente datos
hasta el cierre de la fecha t (misma regla que `ml/features.py`); la fila t
de `asset_returns` es el retorno YA REALIZADO entre t y t+1 — deliberadamente
"adelantado" un paso porque ese es su rol (la recompensa que corresponde a
la acción tomada con `obs_features[t]`), no una feature de observación. Ver
el docstring de `build_portfolio_dataset` para el detalle exacto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.comparison import align_common_dates
from config import UNIVERSE
from data.loaders import get_prices
from ml.features import build_feature_matrix
from signals.returns import simple_returns

logger = logging.getLogger(__name__)

DEFAULT_ASSETS: tuple[str, ...] = tuple(UNIVERSE)  # ("BTC", "ETH", "SOL", "BNB", "LTC")
CASH_LABEL: str = "CASH"

# Mismo horizonte que `ml.features._REALIZED_VOL_WINDOW`, por consistencia
# con el resto del proyecto (una sola "ventana de volatilidad reciente"
# canónica, no dos convenciones distintas conviviendo).
COVARIANCE_WINDOW: int = 20

# Subconjunto de `ml.features.build_feature_matrix` usado como observación
# por activo (Fase 18 pide explícitamente: "retornos recientes", "volatilidad
# realizada", "RSI y MACD normalizados"). El resto de las ~19 columnas de
# `build_feature_matrix` (Bollinger, distancia a medias, ATR, volumen
# relativo, etc.) queda disponible para una iteración futura, pero no es
# parte del pedido explícito de esta fase — mantener la observación acotada
# también ayuda a que la policy (una red chica) tenga menos ruido que filtrar.
PER_ASSET_FEATURE_COLUMNS: tuple[str, ...] = ("ret_1", "ret_5", "vol_realizada_20", "rsi_14_norm", "macd_hist_norm")


@dataclass
class PortfolioDataset:
    """Dataset ya alineado y listo para partir en bloques walk-forward
    (`rl.evaluation.make_walkforward_blocks`) — cada bloque es simplemente
    un rango de ÍNDICES sobre los arrays de acá, sin recalcular nada.

    Atributos
    ---------
    dates:
        Índice de fechas (UTC), longitud T.
    obs_features:
        ndarray (T, F) — TODAS las columnas de observación de MERCADO
        (por activo + covarianza). NO incluye los pesos actuales de la
        cartera: esos dependen de la trayectoria del agente, no son un dato
        de mercado, y los agrega dinámicamente `rl.env.PortfolioEnv` en
        cada `step`.
    asset_returns:
        ndarray (T, N_ASSETS + 1) — columna por activo (mismo orden que
        `asset_names`) más una columna final "CASH" siempre 0. La fila t es
        el retorno SIMPLE realizado entre el cierre de t y el cierre de
        t+1 — la recompensa que le corresponde a la acción tomada con
        `obs_features[t]`, ver el docstring de `build_portfolio_dataset`.
    asset_names:
        Nombres de columna de `asset_returns`, en orden (termina en "CASH").
    feature_names:
        Nombres de columna de `obs_features`, en orden.
    """

    dates: pd.DatetimeIndex
    obs_features: np.ndarray
    asset_returns: np.ndarray
    asset_names: tuple[str, ...]
    feature_names: tuple[str, ...]


def _rolling_covariance_features(returns_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Covarianza rolling TRAILING (ventana de `window` días terminando en
    cada fecha) entre cada par de activos de `returns_df`, incluida la
    varianza propia (par consigo mismo) — el triángulo superior de la
    matriz de covarianza, sin duplicar el triángulo inferior (simétrico por
    definición).

    `Series.rolling(window).cov(other)` es trailing por construcción (usa
    únicamente las `window` observaciones HASTA la fecha de cada fila) — se
    reutiliza el cálculo de covarianza de pandas, no se reimplementa.
    """
    assets = list(returns_df.columns)
    cols: dict[str, pd.Series] = {}
    for i, asset_a in enumerate(assets):
        for asset_b in assets[i:]:
            cols[f"cov_{asset_a}_{asset_b}"] = returns_df[asset_a].rolling(window).cov(returns_df[asset_b])
    return pd.DataFrame(cols, index=returns_df.index)


def _per_asset_observation_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y renombra el subconjunto estacionario de
    `ml.features.build_feature_matrix` que usa `PortfolioEnv` como
    observación por activo (ver `PER_ASSET_FEATURE_COLUMNS`), normalizando
    además el RSI a [-1, 1] vía (rsi-50)/50 — misma convención que
    `signals.engine.compute_signal_components` — para que quede en una
    escala comparable con el resto de las features (todas ya en un rango
    chico alrededor de 0), en vez de en su escala nativa [0, 100].
    """
    feats = build_feature_matrix(ohlcv)
    feats = feats.assign(rsi_14_norm=(feats["rsi_14"] - 50.0) / 50.0)
    return feats[list(PER_ASSET_FEATURE_COLUMNS)]


def build_portfolio_dataset(
    assets: tuple[str, ...] = DEFAULT_ASSETS,
    interval: str = "1d",
    cov_window: int = COVARIANCE_WINDOW,
) -> PortfolioDataset:
    """Arma el dataset completo (features de observación + retornos de
    recompensa/evaluación) para `PortfolioEnv`, sobre TODA la historia común
    disponible de `assets` (`source="store"`, sin red).

    Cada feature de acá es trailing/causal por construcción (`ml.features.build_feature_matrix`,
    la covarianza rolling de `_rolling_covariance_features`) — partir el
    resultado en bloques walk-forward DESPUÉS de calcularlo no introduce
    lookahead: el valor de la fila t es idéntico si se calcula sobre toda la
    historia o sobre cualquier sub-serie que incluya hasta la fila t (mismo
    argumento que ya documenta `ml/features.py`).

    OBSERVACIÓN, por activo (ver `PER_ASSET_FEATURE_COLUMNS`) más, para el
    conjunto, la covarianza rolling de `cov_window` días entre cada par de
    activos (triángulo superior). Los PESOS ACTUALES de la cartera NO viven
    acá — los agrega `PortfolioEnv` en cada `step` (son estado de la
    trayectoria del agente, no un dato de mercado observable de antemano).

    RETORNOS para la recompensa/evaluación (`asset_returns`): fila t =
    retorno simple realizado ENTRE el cierre de t y el cierre de t+1
    (`signals.returns.simple_returns(close).shift(-1)`, por activo) más una
    columna "CASH" siempre 0 — así la fila t de `obs_features` (causal,
    hasta el cierre de t) se empareja exactamente con la fila t de
    `asset_returns` (el retorno que se REALIZA después de actuar con esa
    observación), sin necesitar ningún shift adicional en `PortfolioEnv` ni
    en `rl.evaluation`. La última fecha de la historia común no tiene "día
    siguiente" -> se descarta (no se puede calcular su recompensa).

    Devuelve un `PortfolioDataset` sin NaN (se eliminan las filas con
    warmup de alguna ventana rolling, en cualquier activo, o sin retorno
    siguiente disponible) — la primera fecha utilizable es la primera en la
    que TODOS los activos y TODAS las features ya tienen dato.
    """
    ohlcv_by_asset: dict[str, pd.DataFrame] = {
        asset: get_prices(asset, source="store", interval=interval, use_cache=False) for asset in assets
    }
    closes = {asset: df["close"] for asset, df in ohlcv_by_asset.items()}
    aligned_close = align_common_dates(closes)

    # Las features por activo se calculan sobre la historia COMPLETA de
    # cada uno (antes de alinear) y se recortan a las fechas comunes
    # DESPUÉS — mismo razonamiento que arriba: son trailing, así que
    # calcularlas sobre más historia de la que se usa no cambia ningún
    # valor en las fechas comunes, y evita cortar el warmup de cada
    # indicador justo en el inicio de la ventana común en vez de en el
    # propio inicio de vida de cada activo.
    obs_parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    for asset in assets:
        asset_feats = _per_asset_observation_features(ohlcv_by_asset[asset]).reindex(aligned_close.index)
        renamed = asset_feats.rename(columns={col: f"{asset}_{col}" for col in asset_feats.columns})
        obs_parts.append(renamed)
        feature_names.extend(renamed.columns)

    returns_df = pd.DataFrame({asset: simple_returns(aligned_close[asset]) for asset in assets})
    cov_features = _rolling_covariance_features(returns_df, window=cov_window)
    obs_parts.append(cov_features)
    feature_names.extend(cov_features.columns)

    obs_df = pd.concat(obs_parts, axis=1)

    forward_returns = returns_df.shift(-1)
    forward_returns[CASH_LABEL] = 0.0
    asset_names = tuple(assets) + (CASH_LABEL,)
    forward_returns = forward_returns[list(asset_names)].add_prefix("__ret__")

    combined = pd.concat([obs_df, forward_returns], axis=1).dropna()

    dates = combined.index
    obs_features = combined[feature_names].to_numpy(dtype=np.float32)
    asset_returns = combined[[f"__ret__{name}" for name in asset_names]].to_numpy(dtype=np.float32)

    logger.info(
        "build_portfolio_dataset: %d activos + efectivo, %d fechas utilizables (%s a %s), %d features de observación",
        len(assets), len(dates), dates.min(), dates.max(), obs_features.shape[1],
    )

    return PortfolioDataset(
        dates=dates,
        obs_features=obs_features,
        asset_returns=asset_returns,
        asset_names=asset_names,
        feature_names=tuple(feature_names),
    )
