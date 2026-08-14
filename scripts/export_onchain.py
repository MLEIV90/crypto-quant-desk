"""Script de exportación de datos ON-CHAIN desde CoinMetrics Community (Fase 5a).

Uso puntual, de un solo uso (no forma parte del pipeline regular del
proyecto, igual que `scripts/export_snapshot.py`): para cada activo de
`config.UNIVERSE`, descarga el CSV completo de CoinMetrics Community (el
mismo endpoint que usa `data.loaders._load_coinmetrics` para precio, pero
acá se leen TODAS las columnas del CSV, no solo la de precio) y guarda las
columnas on-chain disponibles a `data/snapshot/{ASSET}_onchain_1d.parquet`.

No reimplementa el request HTTP: reutiliza `data.loaders._http_get_with_retry`
(reintentos con backoff exponencial ante fallos de red/HTTP) y
`data.loaders.COINMETRICS_CSV_URL` tal cual.

De la lista fija de columnas candidatas (`ONCHAIN_CANDIDATE_COLUMNS`), se
guarda cada una que (a) exista en el CSV del activo y (b) tenga más de
`MIN_OBSERVATIONS` observaciones no nulas — no hardcodea qué activos tienen
qué columnas: si un activo no tiene NINGUNA columna que pase ese filtro
(el caso de SOL, que en CoinMetrics Community solo reporta precio), se
loguea "sin on-chain" y se sigue con el resto del universo, sin abortar.

Uso:
    python scripts/export_onchain.py
"""

from __future__ import annotations

import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SNAPSHOT_DIR, UNIVERSE  # noqa: E402
from data.loaders import COINMETRICS_CSV_URL, _http_get_with_retry  # noqa: E402

logger = logging.getLogger(__name__)

# Columnas on-chain candidatas (nombres tal cual los publica CoinMetrics
# Community). No todos los activos las tienen todas — ver docstring del
# módulo y `data/onchain.py` para el detalle de qué representa cada una y
# en qué feature se transforma.
ONCHAIN_CANDIDATE_COLUMNS: list[str] = [
    "FlowInExUSD", "FlowOutExUSD", "AdrActCnt", "TxCnt",
    "CapMVRVCur", "HashRate", "FeeTotNtv", "SplyCur",
]

# Una columna con pocas observaciones no alcanza para construir features
# rolling razonables (ver ZSCORE_WINDOW en data.onchain) — se descarta.
MIN_OBSERVATIONS = 500


def _download_onchain_csv(symbol: str) -> pd.DataFrame:
    """Descarga el CSV completo de CoinMetrics Community para `symbol`
    (identificador de `config.UNIVERSE[asset]["coinmetrics"]`, p. ej. "btc")
    y lo devuelve indexado por fecha UTC, SIN filtrar columnas.
    """
    url = COINMETRICS_CSV_URL.format(symbol=symbol)
    response = _http_get_with_retry(url, params={})
    df = pd.read_csv(StringIO(response.text))
    if "time" not in df.columns:
        raise ValueError(f"CSV de CoinMetrics para '{symbol}' no tiene la columna 'time' esperada")
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("timestamp").sort_index()


def _extract_onchain_columns(raw: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Filtra `raw` a las columnas de `ONCHAIN_CANDIDATE_COLUMNS` que existan
    y tengan más de `MIN_OBSERVATIONS` valores no nulos. Devuelve un
    DataFrame vacío (0 columnas) si ninguna califica — el caller interpreta
    eso como "el activo no tiene on-chain usable", no como un error.

    Las filas donde TODAS las columnas guardadas son NaN se eliminan (no
    aportan nada); NaN parciales (p. ej. una columna con historia más corta
    que otra) se conservan tal cual — `data.onchain.build_onchain_features`
    los maneja en su warmup de ventanas rolling.
    """
    kept: list[str] = []
    for col in ONCHAIN_CANDIDATE_COLUMNS:
        if col not in raw.columns:
            continue
        n_valid = int(raw[col].notna().sum())
        if n_valid <= MIN_OBSERVATIONS:
            logger.info(
                "%s: columna '%s' descartada (%d observaciones no nulas, <= %d)",
                asset, col, n_valid, MIN_OBSERVATIONS,
            )
            continue
        kept.append(col)

    if not kept:
        return pd.DataFrame(index=raw.index[:0])

    out = raw[kept].astype(float)
    return out.dropna(how="all")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    summary: dict[str, tuple[list[str], pd.Timestamp | None, pd.Timestamp | None]] = {}

    for asset, sources in UNIVERSE.items():
        symbol = sources["coinmetrics"]
        logger.info("=== %s (%s) ===", asset, symbol)
        try:
            raw = _download_onchain_csv(symbol)
        except Exception as exc:  # noqa: BLE001 - loguear y seguir con el resto del universo
            logger.error("%s: falló la descarga del CSV de CoinMetrics: %s", asset, exc)
            summary[asset] = ([], None, None)
            continue

        onchain = _extract_onchain_columns(raw, asset)
        if onchain.empty:
            logger.info(
                "%s: sin on-chain usable en CoinMetrics Community (0 columnas con >%d observaciones)",
                asset, MIN_OBSERVATIONS,
            )
            summary[asset] = ([], None, None)
            continue

        path = SNAPSHOT_DIR / f"{asset}_onchain_1d.parquet"
        onchain.to_parquet(path)
        logger.info(
            "%s: guardado '%s' (%d filas, columnas %s)", asset, path, len(onchain), list(onchain.columns)
        )
        summary[asset] = (list(onchain.columns), onchain.index.min(), onchain.index.max())

    print("\nResumen del export on-chain:")
    for asset, (cols, start_ts, end_ts) in summary.items():
        if not cols:
            print(f"  {asset}: sin on-chain")
        else:
            print(f"  {asset}: {cols} ({start_ts.date()} a {end_ts.date()})")


if __name__ == "__main__":
    main()
