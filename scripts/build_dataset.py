"""CLI para descargar y cachear el histórico de precios de `config.UNIVERSE`.

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --source binance --start 2020-01-01

Además de poblar la caché parquet (`data/cache/*.parquet`, ignorada en git),
exporta por cada activo un CSV de verificación desde CoinMetrics
(`data/cache/{ASSET}_coinmetrics_verification.csv`). Estos CSV SÍ se
commitean al repo para poder verificar el pipeline de datos sin necesidad de
acceso a Binance (ver .gitignore).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CACHE_DIR, DEFAULT_INTERVAL, DEFAULT_SOURCE, RAW_START_DATE, UNIVERSE  # noqa: E402
from data.loaders import get_prices  # noqa: E402

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Descarga y cachea el histórico de precios de config.UNIVERSE.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"Fuente primaria (default: {DEFAULT_SOURCE})")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help=f"Intervalo de velas (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--start", default=RAW_START_DATE, help=f"Fecha de inicio ISO (default: {RAW_START_DATE})")
    parser.add_argument("--end", default=None, help="Fecha de fin ISO (default: hoy UTC)")
    parser.add_argument("--no-cache", action="store_true", help="Ignora la caché existente y fuerza la descarga")
    parser.add_argument(
        "--skip-verification-csv",
        action="store_true",
        help="No exporta los CSV de verificación de CoinMetrics a data/cache/",
    )
    return parser


def _export_verification_csv(asset: str, start: str, end: str | None) -> None:
    """Exporta un CSV de CoinMetrics por activo a data/cache/ para verificar
    el pipeline de datos sin depender de la disponibilidad de Binance.
    """
    df = get_prices(asset, source="coinmetrics", start=start, end=end, use_cache=False)
    out_path = CACHE_DIR / f"{asset}_coinmetrics_verification.csv"
    df.to_csv(out_path)
    logger.info("CSV de verificación exportado: %s (%d filas)", out_path, len(df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    use_cache = not args.no_cache

    for asset in UNIVERSE:
        try:
            df = get_prices(
                asset,
                source=args.source,
                interval=args.interval,
                start=args.start,
                end=args.end,
                use_cache=use_cache,
            )
            logger.info("%s: %d filas descargadas/cacheadas desde '%s'", asset, len(df), args.source)
        except Exception:  # noqa: BLE001 - falla de red/fuente de un activo no debe tumbar el resto del universo
            logger.exception("Falló la descarga de %s", asset)

        if not args.skip_verification_csv:
            try:
                _export_verification_csv(asset, args.start, args.end)
            except Exception:  # noqa: BLE001 - idem: loguear y seguir con el resto del universo
                logger.exception("Falló la exportación del CSV de verificación de %s", asset)


if __name__ == "__main__":
    main()
