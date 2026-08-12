"""Validación y saneamiento de calidad de datos OHLCV.

Este módulo NO reimplementa la ingesta: consume el contrato de salida de
`data.loaders.get_prices` (índice DatetimeIndex UTC tz-aware, columnas
["open", "high", "low", "close", "volume"] en float, sin duplicados; en
fuentes de solo-cierre como CoinMetrics/CoinGecko, OHLC = close y volume
puede venir en NaN).

Dos funciones, dos responsabilidades separadas a propósito:
    - `validate_ohlcv`: solo LEE y reporta. Nunca modifica el DataFrame de
      entrada ni corrige nada por su cuenta.
    - `clean_ohlcv`: aplica una política de saneamiento EXPLÍCITA, elegida
      por quien llama, y documenta en un log de cambios qué se hizo. Nada se
      altera en silencio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

# Umbral por defecto para outliers de retorno, en múltiplos de la mediana de
# desviación absoluta (MAD) de los log-retornos (ver `_detect_return_outliers`).
DEFAULT_OUTLIER_MAD_MULTIPLIER: float = 8.0

# Cantidad mínima de observaciones consecutivas con el mismo `close` para
# reportarlas como un tramo plano (posible dato stale).
DEFAULT_FLAT_RUN_MIN_DAYS: int = 5

GAP_POLICIES: tuple[str, ...] = ("report_only", "ffill", "drop")
OUTLIER_POLICIES: tuple[str, ...] = ("report_only", "winsorize", "nan")


@dataclass
class DataQualityReport:
    """Snapshot de calidad de un OHLCV, producido por `validate_ohlcv`.

    Es un reporte de solo lectura: no implica ninguna corrección sobre los
    datos (para eso está `clean_ohlcv`).

    Atributos
    ---------
    activo:
        Nombre opcional del activo (solo para identificar el reporte, no
        afecta el cálculo).
    n_filas:
        Cantidad de observaciones en el DataFrame validado.
    fecha_inicio, fecha_fin:
        Primer y último timestamp observado (None si `n_filas == 0`).
    n_duplicados_indice:
        Cantidad de timestamps duplicados en el índice original.
    gaps:
        Fechas diarias faltantes contra un `date_range` diario completo entre
        `fecha_inicio` y `fecha_fin`.
    n_precios_no_positivos:
        Cantidad de FILAS con al menos una columna OHLC <= 0 (no es un
        conteo de celdas individuales).
    n_volumen_negativo:
        Cantidad de filas con `volume` < 0. Los NaN de volumen (fuentes de
        solo-cierre) no cuentan como negativos: son ausencia de dato, no un
        valor inválido.
    outliers_retorno:
        Lista de (fecha, log_retorno) cuyo log-retorno se aleja de la
        mediana más de `outlier_mad_multiplier` veces el MAD. Se REPORTAN,
        no se eliminan.
    tramos_planos:
        Lista de (fecha_inicio, fecha_fin, n_dias) de corridas de `close`
        idéntico de al menos `flat_run_min_days` observaciones.
    warnings:
        Mensajes agregados legibles por humanos con cualquier otro hallazgo
        (índice desordenado/sin timezone, filas con high<low, close fuera de
        [low, high], filas con OHLC degenerado, etc.).
    """

    activo: str | None
    n_filas: int
    fecha_inicio: pd.Timestamp | None
    fecha_fin: pd.Timestamp | None
    n_duplicados_indice: int
    gaps: list[pd.Timestamp]
    n_precios_no_positivos: int
    n_volumen_negativo: int
    outliers_retorno: list[tuple[pd.Timestamp, float]]
    tramos_planos: list[tuple[pd.Timestamp, pd.Timestamp, int]]
    warnings: list[str]

    @property
    def n_gaps(self) -> int:
        return len(self.gaps)

    @property
    def n_outliers(self) -> int:
        return len(self.outliers_retorno)

    def resumen(self) -> str:
        """Línea de resumen legible, usada para logging y para mostrar en el notebook de EDA."""
        prefijo = f"[{self.activo}] " if self.activo else ""
        return (
            f"{prefijo}{self.n_filas} filas ({self.fecha_inicio} a {self.fecha_fin}) | "
            f"{self.n_duplicados_indice} duplicados | {self.n_gaps} gaps | "
            f"{self.n_precios_no_positivos} filas con precio<=0 | "
            f"{self.n_volumen_negativo} filas con volumen<0 | "
            f"{self.n_outliers} outliers de retorno | {len(self.tramos_planos)} tramos planos | "
            f"{len(self.warnings)} warnings"
        )


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    activo: str | None = None,
    outlier_mad_multiplier: float = DEFAULT_OUTLIER_MAD_MULTIPLIER,
    flat_run_min_days: int = DEFAULT_FLAT_RUN_MIN_DAYS,
) -> DataQualityReport:
    """Valida un OHLCV estandarizado y devuelve un `DataQualityReport`.

    Es un chequeo de SOLO LECTURA: `df` nunca se modifica. Solo lanza
    excepción si `df` está mal formado (no es un DataFrame con DatetimeIndex
    y las columnas OHLCV esperadas); cualquier otro hallazgo de calidad se
    reporta en el `DataQualityReport` devuelto, nunca interrumpe la ejecución.

    Chequeos realizados:
    - Índice UTC tz-aware, ordenado y sin duplicados: si no se cumple, se
      agrega un warning (los cálculos internos de gaps/retornos igual se
      hacen sobre una copia ordenada y deduplicada, solo para que el resto
      de los chequeos sean coherentes; `df` en sí no se toca).
    - Precios OHLC > 0 (ver `n_precios_no_positivos`).
    - high >= low y low <= close <= high, PERO solo en filas donde el OHLC
      no sea degenerado (open == high == low == close, como en fuentes de
      solo-cierre tipo CoinMetrics/CoinGecko vía data.loaders): en esas
      filas el chequeo no aplica por construcción y se omite, dejando
      constancia en `warnings`.
    - volume >= 0 (los NaN de fuentes de solo-cierre no cuentan como
      inválidos).
    - Gaps: fechas faltantes contra un `date_range` diario completo entre el
      primer y el último timestamp observado.
    - Outliers de retorno: ver `_detect_return_outliers` (método de MAD).
    - Tramos planos: corridas de `close` idéntico de al menos
      `flat_run_min_days` observaciones consecutivas (posible dato stale).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate_ohlcv espera un pandas.DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("validate_ohlcv espera un DataFrame con DatetimeIndex")
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas {missing_cols} en el DataFrame a validar")

    if len(df) == 0:
        logger.info("validate_ohlcv: DataFrame vacío%s", f" [{activo}]" if activo else "")
        return DataQualityReport(
            activo=activo,
            n_filas=0,
            fecha_inicio=None,
            fecha_fin=None,
            n_duplicados_indice=0,
            gaps=[],
            n_precios_no_positivos=0,
            n_volumen_negativo=0,
            outliers_retorno=[],
            tramos_planos=[],
            warnings=["DataFrame vacío"],
        )

    warnings: list[str] = []

    if df.index.tz is None:
        warnings.append("Índice sin timezone (se esperaba UTC tz-aware)")
    elif str(df.index.tz) != "UTC":
        warnings.append(f"Índice con timezone '{df.index.tz}' (se esperaba UTC)")

    if not df.index.is_monotonic_increasing:
        warnings.append("Índice no está ordenado ascendentemente")

    n_duplicados_indice = int(df.index.duplicated().sum())
    if n_duplicados_indice > 0:
        warnings.append(f"{n_duplicados_indice} timestamps duplicados en el índice")

    # De acá en más se trabaja sobre una copia ordenada y deduplicada (se
    # conserva la última observación de cada timestamp repetido) solo para
    # que gaps/outliers/tramos planos sean coherentes. `df` no se modifica.
    df_sorted = df[~df.index.duplicated(keep="last")].sort_index()

    fecha_inicio = df_sorted.index.min()
    fecha_fin = df_sorted.index.max()

    ohlc = df_sorted[["open", "high", "low", "close"]]
    n_precios_no_positivos = int((ohlc <= 0).any(axis=1).sum())
    if n_precios_no_positivos > 0:
        warnings.append(f"{n_precios_no_positivos} filas con algún precio OHLC <= 0")

    n_volumen_negativo = int((df_sorted["volume"] < 0).sum())
    if n_volumen_negativo > 0:
        warnings.append(f"{n_volumen_negativo} filas con volume < 0")

    is_degenerate = (
        (df_sorted["open"] == df_sorted["high"])
        & (df_sorted["high"] == df_sorted["low"])
        & (df_sorted["low"] == df_sorted["close"])
    )
    n_degenerate = int(is_degenerate.sum())
    real_ohlc = df_sorted.loc[~is_degenerate]
    n_high_lt_low = int((real_ohlc["high"] < real_ohlc["low"]).sum())
    n_close_out_of_range = int(
        ((real_ohlc["close"] < real_ohlc["low"]) | (real_ohlc["close"] > real_ohlc["high"])).sum()
    )
    if n_degenerate > 0:
        warnings.append(
            f"{n_degenerate} filas con OHLC degenerado (open=high=low=close, típico de fuentes de "
            "solo-cierre); se omitió el chequeo high>=low / close-en-rango en esas filas"
        )
    if n_high_lt_low > 0:
        warnings.append(f"{n_high_lt_low} filas con high < low")
    if n_close_out_of_range > 0:
        warnings.append(f"{n_close_out_of_range} filas con close fuera de [low, high]")

    full_range = pd.date_range(fecha_inicio, fecha_fin, freq="D", tz=df_sorted.index.tz)
    gaps = full_range.difference(df_sorted.index).tolist()
    if gaps:
        warnings.append(f"{len(gaps)} fechas faltantes (gaps) entre {fecha_inicio.date()} y {fecha_fin.date()}")

    outliers_retorno, threshold = _detect_return_outliers(
        df_sorted["close"], mad_multiplier=outlier_mad_multiplier
    )
    if threshold is None and len(df_sorted) > 1:
        warnings.append(
            "MAD de los log-retornos es 0 (demasiados retornos idénticos); no se pudo calcular "
            "un umbral de outliers significativo"
        )
    elif outliers_retorno:
        warnings.append(
            f"{len(outliers_retorno)} outliers de retorno (|log_retorno - mediana| > "
            f"{outlier_mad_multiplier}*MAD)"
        )

    tramos_planos = _find_flat_runs(df_sorted["close"], min_days=flat_run_min_days)
    if tramos_planos:
        warnings.append(f"{len(tramos_planos)} tramos de precio idéntico de al menos {flat_run_min_days} días")

    report = DataQualityReport(
        activo=activo,
        n_filas=len(df),
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        n_duplicados_indice=n_duplicados_indice,
        gaps=gaps,
        n_precios_no_positivos=n_precios_no_positivos,
        n_volumen_negativo=n_volumen_negativo,
        outliers_retorno=outliers_retorno,
        tramos_planos=tramos_planos,
        warnings=warnings,
    )
    logger.info(report.resumen())
    return report


def clean_ohlcv(
    df: pd.DataFrame,
    gap_policy: str = "report_only",
    outlier_policy: str = "report_only",
    *,
    gap_fill_limit: int | None = None,
    outlier_mad_multiplier: float = DEFAULT_OUTLIER_MAD_MULTIPLIER,
) -> tuple[pd.DataFrame, dict]:
    """Aplica una política de saneamiento EXPLÍCITA sobre un OHLCV estandarizado.

    Nada se altera en silencio: cada política está documentada acá abajo, y
    el resultado incluye un log de cambios (`dict`) con lo que efectivamente
    se hizo. `df` original nunca se modifica in-place (se devuelve siempre
    una copia nueva). Con las políticas por defecto ("report_only" para
    ambas), el DataFrame de salida es una copia sin cambios de `df`.

    gap_policy — tratamiento de fechas faltantes en el índice diario:
        - "report_only" (default): no modifica el índice ni las filas.
        - "ffill": reindexa a un `date_range` diario completo (requiere
          índice sin duplicados, por eso este caso sí deduplica/ordena antes
          de reindexar) y hace forward-fill de open/high/low/close con el
          último precio observado. TRADE-OFF IMPORTANTE: cada día rellenado
          tiene retorno exactamente 0 por construcción, lo que ARTIFICIALMENTE
          reduce la volatilidad estimada (annualized_volatility, un futuro
          GARCH, etc. subestimarán el riesgo real si hay muchos gaps).
          `volume` en los días rellenados se deja en NaN (no se
          forward-fillea) para no fingir actividad de trading que no ocurrió.
          `gap_fill_limit` acota cuántos días consecutivos se rellenan como
          máximo (se pasa a `Series.ffill(limit=...)`); gaps más largos que
          el límite quedan en NaN en vez de rellenarse indefinidamente.
        - "drop": no agrega filas para cubrir los gaps (los gaps YA son
          ausencia de filas: no hay nada que "tirar"). Documenta la elección
          deliberada de trabajar solo con los días efectivamente observados.

    outlier_policy — tratamiento de outliers de retorno (ver
    `_detect_return_outliers`), evaluados sobre el DataFrame resultante de
    aplicar `gap_policy` (no sobre `df` original), para que las fechas
    coincidan con el índice que se está modificando:
        - "report_only" (default): no modifica ningún precio.
        - "winsorize": para cada fecha marcada como outlier, recorta el
          log-retorno de ese día al umbral (±outlier_mad_multiplier*MAD) y
          reconstruye `close` en esa fecha como
          close[t-1] * exp(log_retorno_recortado). TRADE-OFF: solo se ajusta
          `close` (no open/high/low), así que la fila deja de ser
          internamente consistente como vela OHLC; y como el retorno del día
          SIGUIENTE se calcula contra este `close` ya ajustado, el recorte
          también absorbe parte del retorno del día siguiente (el precio no
          "salta de vuelta" de golpe al nivel original). Es un suavizado
          deliberado para no romper visualizaciones ni el ajuste de modelos
          de volatilidad con un solo día extremo, NO una corrección de un
          dato erróneo confirmado.
        - "nan": pone en NaN las 4 columnas OHLC de la fila marcada como
          outlier (deja `volume` intacto). No inventa un precio de
          reemplazo: deja el hueco explícito para que una etapa posterior
          decida cómo imputarlo (o simplemente lo excluya).
    """
    if gap_policy not in GAP_POLICIES:
        raise ValueError(f"gap_policy debe ser uno de {GAP_POLICIES}, recibido '{gap_policy}'")
    if outlier_policy not in OUTLIER_POLICIES:
        raise ValueError(f"outlier_policy debe ser uno de {OUTLIER_POLICIES}, recibido '{outlier_policy}'")

    report = validate_ohlcv(df, outlier_mad_multiplier=outlier_mad_multiplier)

    out = df.copy()
    log_changes: dict = {
        "gap_policy": gap_policy,
        "outlier_policy": outlier_policy,
        "n_filas_entrada": len(df),
        "n_gaps_detectados": report.n_gaps,
        "n_outliers_detectados": report.n_outliers,
        "n_gaps_rellenados": 0,
        "n_outliers_modificados": 0,
        "fechas_outliers_modificados": [],
    }

    if gap_policy == "ffill" and report.gaps:
        out = out[~out.index.duplicated(keep="last")].sort_index()
        full_range = pd.date_range(out.index.min(), out.index.max(), freq="D", tz=out.index.tz)
        price_cols = ["open", "high", "low", "close"]
        out = out.reindex(full_range)
        out[price_cols] = out[price_cols].ffill(limit=gap_fill_limit)
        n_rellenados = int(len(out.index.difference(df.index)))
        log_changes["n_gaps_rellenados"] = n_rellenados
        logger.info("clean_ohlcv: gap_policy='ffill' rellenó %d fechas (limit=%s)", n_rellenados, gap_fill_limit)
    elif gap_policy == "drop":
        logger.info("clean_ohlcv: gap_policy='drop', se mantienen solo los días observados (%d filas)", len(out))

    if outlier_policy != "report_only":
        outliers, threshold = _detect_return_outliers(out["close"], mad_multiplier=outlier_mad_multiplier)
        for fecha, r in outliers:
            pos = out.index.get_loc(fecha)
            if not isinstance(pos, (int, np.integer)) or pos == 0:
                # Fecha con índice ambiguo (duplicado) o sin observación previa: se omite.
                continue

            if outlier_policy == "winsorize":
                prev_close = out["close"].iloc[pos - 1]
                r_clipped = float(np.clip(r, -threshold, threshold))
                out.iloc[pos, out.columns.get_loc("close")] = prev_close * np.exp(r_clipped)
            elif outlier_policy == "nan":
                for col in ("open", "high", "low", "close"):
                    out.iloc[pos, out.columns.get_loc(col)] = np.nan

            log_changes["n_outliers_modificados"] += 1
            log_changes["fechas_outliers_modificados"].append(fecha)

        logger.info(
            "clean_ohlcv: outlier_policy='%s' modificó %d fechas outlier",
            outlier_policy, log_changes["n_outliers_modificados"],
        )

    log_changes["n_filas_salida"] = len(out)
    return out, log_changes


# --------------------------------------------------------------------------
# Helpers internos
# --------------------------------------------------------------------------


def _detect_return_outliers(
    close: pd.Series, *, mad_multiplier: float
) -> tuple[list[tuple[pd.Timestamp, float]], float | None]:
    """Detecta outliers de log-retorno por el método de la mediana de
    desviación absoluta (MAD), robusto a los propios outliers (a diferencia
    de usar media/std, que los outliers distorsionan).

    mediana_r = mediana(log_retorno)
    MAD       = mediana(|log_retorno - mediana_r|)
    outlier   = |log_retorno - mediana_r| > mad_multiplier * MAD

    Devuelve (outliers, umbral). `umbral` es None si el MAD da 0 (más de la
    mitad de los retornos son idénticos, p. ej. tras un ffill agresivo): en
    ese caso no se puede definir un umbral significativo y no se reportan
    outliers, en vez de marcar como outlier cualquier retorno no-cero.
    """
    log_ret = np.log(close / close.shift(1)).dropna()
    if log_ret.empty:
        return [], None

    median_ret = log_ret.median()
    mad = (log_ret - median_ret).abs().median()
    if mad == 0:
        return [], None

    threshold = mad_multiplier * mad
    flagged = log_ret[(log_ret - median_ret).abs() > threshold]
    outliers = list(zip(flagged.index.tolist(), flagged.to_numpy().tolist()))
    return outliers, threshold


def _find_flat_runs(close: pd.Series, *, min_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Detecta corridas de `close` idéntico de al menos `min_days` observaciones
    consecutivas (posible dato stale / fuente sin actualizar).
    """
    if close.empty:
        return []

    run_id = close.ne(close.shift(1)).cumsum()
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for _, group in close.groupby(run_id):
        if len(group) >= min_days:
            runs.append((group.index[0], group.index[-1], len(group)))
    return runs
