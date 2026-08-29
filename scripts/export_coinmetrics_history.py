"""Script de exportación del tramo PRE-2018 de CoinMetrics para la serie de
precio larga fusionada (Fase 31, `data.loaders._load_full_history`).

Uso puntual, de un solo uso (no forma parte del pipeline regular del
proyecto, igual que `scripts/export_snapshot.py`): para cada activo de
`config.UNIVERSE`, descarga el histórico de CoinMetrics Community (`GET
/api/pairs`... en realidad el CSV público de GitHub, vía
`data.loaders.get_prices(source="coinmetrics")`, reutilizado tal cual) y
guarda SOLO la porción anterior a `data.loaders.MERGE_CUTOFF_DATE`
(2018-01-01) a Parquet en `data/snapshot/{ASSET}_coinmetrics_1d.parquet`.

Por qué solo el tramo pre-corte: desde esa fecha en adelante, la fuente de
verdad es Binance (`scripts/export_snapshot.py`, `source="store"`) — no
tiene sentido guardar dos copias del mismo período de dos fuentes
distintas, y `_load_full_history` ya sabe leer únicamente este archivo
para el tramo antiguo y `_load_store` para el resto.

DEGRADACIÓN CON GRACIA: algunas monedas del universo (p. ej. SOL, cuya red
recién arrancó en 2020) no tienen ningún dato de CoinMetrics anterior al
corte — para esas, este script loguea el motivo y sigue con el resto, sin
escribir ningún archivo (`_load_full_history` ya maneja la ausencia del
snapshot devolviendo solo el tramo Binance/store, sin error).

Uso:
    python scripts/export_coinmetrics_history.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SNAPSHOT_DIR, UNIVERSE  # noqa: E402
from data.loaders import FULL_HISTORY_START_DATE, MERGE_CUTOFF_DATE, get_prices  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    day_before_cutoff = (pd.Timestamp(MERGE_CUTOFF_DATE, tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(
        "Exportando tramo pre-%s de CoinMetrics para %s -> %s",
        MERGE_CUTOFF_DATE, list(UNIVERSE), SNAPSHOT_DIR,
    )

    summary: dict[str, tuple[int, pd.Timestamp | None, pd.Timestamp | None]] = {}

    for asset in UNIVERSE:
        logger.info("=== %s ===", asset)
        try:
            df = get_prices(
                asset, source="coinmetrics", interval="1d",
                start=FULL_HISTORY_START_DATE, end=day_before_cutoff, use_cache=False,
            )
        except Exception as exc:  # noqa: BLE001 - loguear y seguir con el resto del universo
            logger.warning(
                "%s: sin histórico pre-%s en CoinMetrics (%s) — probablemente la red no existía "
                "todavía en esa fecha. Se omite, get_prices(source='full') degradará con gracia.",
                asset, MERGE_CUTOFF_DATE, exc,
            )
            summary[asset] = (0, None, None)
            continue

        if df.empty:
            logger.warning(
                "%s: CoinMetrics no devolvió ninguna fila antes de %s — se omite.",
                asset, MERGE_CUTOFF_DATE,
            )
            summary[asset] = (0, None, None)
            continue

        parquet_path = SNAPSHOT_DIR / f"{asset}_coinmetrics_1d.parquet"
        df.to_parquet(parquet_path)
        logger.info("%s: guardado '%s' (%d filas, %s a %s)", asset, parquet_path, len(df), df.index.min().date(), df.index.max().date())
        summary[asset] = (len(df), df.index.min(), df.index.max())

    print(f"\nResumen del histórico pre-{MERGE_CUTOFF_DATE} de CoinMetrics:")
    for asset, (n_rows, start_ts, end_ts) in summary.items():
        if n_rows == 0:
            print(f"  {asset}: SIN COBERTURA pre-{MERGE_CUTOFF_DATE} (get_prices(source='full') usará solo Binance/store)")
        else:
            print(f"  {asset}: {n_rows} filas ({start_ts.date()} a {end_ts.date()})")


if __name__ == "__main__":
    main()
