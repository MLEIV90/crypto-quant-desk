"""Pipeline de datos ON-CHAIN de crypto-quant-desk (Fase 5a).

CONTEXTO: hasta la Fase 3d, el ML del proyecto usó exclusivamente features
técnicas derivadas del precio (ver `ml/features.py`) y no mostró edge
predictivo — resultado nulo, consistente con la literatura (los
indicadores técnicos clásicos no tienen poder predictivo robusto
demostrado en cripto). La literatura sí documenta algo de señal en datos
ON-CHAIN (flujos hacia/desde exchanges, direcciones activas, MVRV); este
módulo transforma esos datos crudos (ya descargados por
`scripts/export_onchain.py`) en features utilizables SIN fugas. Fase 5a NO
entrena nada — eso es Fase 5b — el foco acá es exclusivamente traer los
datos y transformarlos de forma causal.

Cobertura confirmada en CoinMetrics Community (ver
`scripts/export_onchain.py` y su resumen de export): BTC y ETH tienen el
set completo, incluidos los flujos de exchange (FlowInExUSD/FlowOutExUSD);
LTC y BNB tienen cobertura parcial (sin flujos de exchange); SOL no tiene
datos on-chain usables en CoinMetrics Community (solo reporta precio) —
`load_onchain("SOL")` lanza `FileNotFoundError`.

REGLA ANTI-LOOKAHEAD (léase antes de usar este módulo): igual que en
`signals/engine.py` y `ml/features.py`, todas las features de
`build_onchain_features` son TRAILING — para la fecha t usan solo
observaciones on-chain <= t (`pct_change`, `rolling` hacia atrás — nada
mira el futuro). Los NIVELES crudos (FlowInExUSD, FlowOutExUSD, AdrActCnt,
TxCnt, HashRate, FeeTotNtv) NO son estacionarios (tienen tendencia de largo
plazo por el crecimiento/adopción de la red) y por eso NUNCA se usan
directamente como feature — solo sus variaciones (`pct_change`) o z-scores
rolling. La única excepción es CapMVRVCur: ya es un RATIO (capitalización
de mercado / capitalización realizada) y por lo tanto razonablemente
estacionario por construcción, así que se conserva como feature además de
su z-score. `merge_onchain` solo ALINEA por fecha; el desfase de un día
para evitar mirar el futuro al entrenar (igual que en
`backtest.engine.run_backtest` o `signals.engine.generate_positions`) es
responsabilidad de quien consuma esto para modelar (Fase 5b), no de este
módulo — acá no se pre-desplaza nada.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import SNAPSHOT_DIR

logger = logging.getLogger(__name__)

# Ventana (en días) de los z-scores rolling — trailing, [t-window+1, t].
ZSCORE_WINDOW = 30


def load_onchain(asset: str) -> pd.DataFrame:
    """Lee el parquet on-chain de `asset` exportado por
    `scripts/export_onchain.py` (`config.SNAPSHOT_DIR/{asset}_onchain_1d.parquet`).

    Devuelve un DataFrame con índice `DatetimeIndex` diario en UTC y solo
    las columnas on-chain crudas que ese activo tenía disponibles (ver
    resumen impreso por el script de export).

    Lanza `FileNotFoundError` con un mensaje claro si el activo no tiene
    on-chain exportado (p. ej. SOL, ver docstring del módulo) o si todavía
    no se corrió el script de export.
    """
    path = SNAPSHOT_DIR / f"{asset}_onchain_1d.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el snapshot on-chain '{path}'. Corré "
            "'python scripts/export_onchain.py' para generarlo, o confirmá que "
            f"'{asset}' tiene cobertura on-chain en CoinMetrics Community (ver "
            "docstring de data/onchain.py — SOL, por ejemplo, no tiene)."
        )

    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    logger.info("load_onchain: '%s' leído desde '%s' (%d filas, columnas %s)", asset, path, len(df), list(df.columns))
    return df


def _rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score rolling TRAILING: `(x_t - media_móvil_t) / desvío_móvil_t`,
    ambos calculados sobre la ventana `[t-window+1, t]` — nunca con datos
    futuros. Sin datos suficientes para llenar la ventana (warmup) o con
    desvío 0, el resultado es NaN (no se rellena artificialmente).
    """
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std(ddof=1)
    return (series - mean) / std


def build_onchain_features(onchain_df: pd.DataFrame) -> pd.DataFrame:
    """Transforma niveles on-chain crudos (`load_onchain`) en features
    CAUSALES y estacionarias. No modifica `onchain_df` in-place.

    Cada feature se agrega solo si sus columnas de origen están presentes
    en `onchain_df` (el activo puede tener cobertura parcial, ver docstring
    del módulo):

    - "net_exchange_flow" = FlowInExUSD - FlowOutExUSD: flujo neto NETO
      hacia exchanges en USD del día. Positivo = más entra a exchanges de lo
      que sale = presión potencialmente vendedora (el usuario suele mover
      cripto A un exchange para vender, no para holdear). Es una diferencia
      de dos niveles del MISMO día — no mezcla información futura, pero
      sigue siendo una magnitud en USD que escala con el tamaño del
      mercado, así que no es estacionaria por sí sola: se usa junto con...
    - "net_flow_zscore": z-score rolling (`ZSCORE_WINDOW`=30 días) de
      `net_exchange_flow` — SÍ estacionario, mide cuán inusual es el flujo
      neto de hoy respecto de su propia historia reciente, sin importar la
      escala absoluta del mercado en ese momento.
    - "active_addr_growth" = variación % día a día de AdrActCnt
      (`pct_change()`, trailing: usa el valor de t y t-1) — la CANTIDAD de
      direcciones activas crece con la adopción de la red (no estacionaria),
      pero su variación porcentual sí es una medida de actividad relativa
      razonablemente estacionaria.
    - "active_addr_zscore": z-score rolling de `active_addr_growth`, para
      detectar picos/caídas de actividad inusuales respecto de la historia
      reciente de esa misma variación.
    - "tx_growth" = variación % día a día de TxCnt (mismo razonamiento que
      `active_addr_growth`, para el conteo de transacciones on-chain).
    - "mvrv_level" = CapMVRVCur tal cual: a diferencia de los conteos/USD de
      arriba, MVRV ya es un RATIO (cap. de mercado / cap. realizada) —
      razonablemente estacionario por construcción, así que se conserva
      como nivel (ver la excepción documentada en el docstring del módulo).
    - "mvrv_zscore": z-score rolling de `mvrv_level`, para ubicar el MVRV de
      hoy respecto de su propio rango reciente (en vez de un umbral fijo
      tipo ">3.5 = sobrevaluado", que no se adapta a distintos regímenes).
    - "hashrate_growth" = variación % día a día de HashRate (si la columna
      está disponible — no todos los activos la reportan).
    - "fee_growth" = variación % día a día de FeeTotNtv (fees totales, en
      unidades nativas del activo).

    Nota: `SplyCur` (supply circulante) se descarga por
    `scripts/export_onchain.py` pero todavía NO se usa acá — queda
    disponible en el parquet crudo para una futura normalización (p. ej.
    flujos relativos al supply), fuera del alcance de esta fase.

    Devuelve un DataFrame con el mismo índice que `onchain_df` y solo las
    columnas de feature (nunca los niveles crudos no-estacionarios de
    origen).
    """
    out = pd.DataFrame(index=onchain_df.index)
    cols = set(onchain_df.columns)

    if {"FlowInExUSD", "FlowOutExUSD"}.issubset(cols):
        net_flow = onchain_df["FlowInExUSD"] - onchain_df["FlowOutExUSD"]
        out["net_exchange_flow"] = net_flow
        out["net_flow_zscore"] = _rolling_zscore(net_flow)

    if "AdrActCnt" in cols:
        addr_growth = onchain_df["AdrActCnt"].pct_change()
        out["active_addr_growth"] = addr_growth
        out["active_addr_zscore"] = _rolling_zscore(addr_growth)

    if "TxCnt" in cols:
        out["tx_growth"] = onchain_df["TxCnt"].pct_change()

    if "CapMVRVCur" in cols:
        out["mvrv_level"] = onchain_df["CapMVRVCur"]
        out["mvrv_zscore"] = _rolling_zscore(onchain_df["CapMVRVCur"])

    if "HashRate" in cols:
        out["hashrate_growth"] = onchain_df["HashRate"].pct_change()

    if "FeeTotNtv" in cols:
        out["fee_growth"] = onchain_df["FeeTotNtv"].pct_change()

    if out.empty:
        logger.warning("build_onchain_features: 'onchain_df' no tiene ninguna columna reconocida, devolviendo vacío")

    return out


def merge_onchain(price_df: pd.DataFrame, onchain_features: pd.DataFrame) -> pd.DataFrame:
    """Alinea `onchain_features` (ver `build_onchain_features`) al índice de
    `price_df` (el de `data.loaders.get_prices`), por fecha UTC (join left
    sobre `price_df`, normalizando ambos índices a medianoche para tolerar
    cualquier diferencia de horario intradiario).

    Convención de uso (responsabilidad de quien llama, no de esta función):
    el valor on-chain de la fecha t que queda en la fila t del resultado es
    información TRAILING disponible al cierre de t, y por lo tanto solo
    puede usarse para predecir el retorno de t+1 — igual que las posiciones
    de `signals.engine.generate_positions`, esta función NO aplica ningún
    `shift`; el desfase es responsabilidad exclusiva de quien entrene un
    modelo con esto (Fase 5b), en el mismo punto único de desfase que ya
    usa `backtest.engine.run_backtest`.

    Fechas de `price_df` sin cobertura on-chain (antes del inicio de la
    serie on-chain, o el activo simplemente no la tiene para esas columnas)
    quedan en NaN — no se rellenan hacia adelante ni hacia atrás acá.
    """
    price_aligned = price_df.copy()
    price_aligned.index = price_aligned.index.normalize()

    onchain_aligned = onchain_features.copy()
    onchain_aligned.index = onchain_aligned.index.normalize()
    onchain_aligned = onchain_aligned[~onchain_aligned.index.duplicated(keep="last")]

    merged = price_aligned.join(onchain_aligned, how="left")
    merged.index.name = price_df.index.name
    return merged
