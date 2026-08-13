"""Script de exportación de un snapshot de datos OHLCV diario desde Binance.

Uso puntual, de un solo uso (no forma parte del pipeline regular del
proyecto): vuelca el histórico diario de todo `config.UNIVERSE` a Parquet
en `data/snapshot/`, para compartir el dataset o verificar el pipeline
offline. No reimplementa la descarga: usa `data.loaders.get_prices` (que ya
pagina las klines de Binance) y `data.quality.validate_ohlcv` para loguear
un reporte de calidad antes de guardar.

Uso:
    python scripts/export_snapshot.py
    python scripts/export_snapshot.py --start 2020-01-01 --out-dir data/mi_snapshot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import UNIVERSE  # noqa: E402
from data.loaders import get_prices  # noqa: E402
from data.quality import validate_ohlcv  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_START = "2018-01-01"
DEFAULT_OUT_DIR = "data/snapshot"

# Códigos HTTP típicos del geobloqueo de Binance en ciertas regiones
# (451 = Unavailable For Legal Reasons, 403 = Forbidden).
GEOBLOCK_STATUS_CODES = (451, 403)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta un snapshot OHLCV diario de Binance (config.UNIVERSE) a Parquet."
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"Fecha de inicio ISO (default: {DEFAULT_START})")
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, help=f"Directorio de salida (default: {DEFAULT_OUT_DIR})"
    )
    return parser


def _yesterday_utc() -> str:
    """'Ayer' en UTC, en formato ISO. Se usa como fecha de fin para excluir
    la vela diaria de hoy, que todavía no cerró.
    """
    return (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _is_geoblocked_error(exc: BaseException) -> bool:
    """Recorre la cadena de excepciones (`__cause__`) buscando un status HTTP
    451 o 403: la señal típica de geobloqueo de Binance en ciertas regiones.
    `get_prices` ya reintenta con CoinMetrics/CoinGecko antes de propagar
    cualquier excepción, así que esto solo dispara si TODAS las fuentes
    fallaron y alguna de ellas devolvió ese status.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(getattr(current, "response", None), "status_code", None)
        if status_code in GEOBLOCK_STATUS_CODES:
            return True
        current = current.__cause__
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end = _yesterday_utc()
    logger.info("Exportando snapshot de Binance: %s a %s -> %s", args.start, end, out_dir)

    summary: dict[str, tuple[int, pd.Timestamp | None, pd.Timestamp | None]] = {}

    for asset in UNIVERSE:
        logger.info("=== %s ===", asset)
        try:
            # use_cache=False deliberado: este script busca un volcado
            # fresco hasta "ayer", y la clave de caché de data/cache/ no
            # encodea start/end (solo asset/source/interval) — reutilizar
            # una caché vieja de otra corrida podría devolver un rango
            # desactualizado sin pegarle a la red.
            df = get_prices(asset, source="binance", start=args.start, end=end, use_cache=False)
        except Exception as exc:  # noqa: BLE001 - loguear y seguir con el resto del universo
            if _is_geoblocked_error(exc):
                logger.error(
                    "%s: la descarga de Binance falló con HTTP 403/451 — posible geobloqueo de "
                    "Binance en tu región. Se omite este activo y se continúa con el resto.",
                    asset,
                )
            else:
                logger.error("%s: falló la descarga (todas las fuentes agotadas): %s", asset, exc)
            summary[asset] = (0, None, None)
            continue

        report = validate_ohlcv(df, activo=asset)
        logger.info(report.resumen())
        if any("degenerado" in w for w in report.warnings):
            logger.warning(
                "%s: el OHLC parece degenerado (open=high=low=close) — probablemente get_prices() "
                "cayó a un fallback de solo-cierre (CoinMetrics/CoinGecko) en vez de usar Binance. "
                "Si no se esperaba esto, puede ser una señal indirecta de geobloqueo de Binance.",
                asset,
            )
        for w in report.warnings:
            logger.info("%s: %s", asset, w)

        parquet_path = out_dir / f"{asset}_1d.parquet"
        df.to_parquet(parquet_path)
        logger.info("%s: guardado '%s' (%d filas)", asset, parquet_path, len(df))

        summary[asset] = (len(df), df.index.min(), df.index.max())

    print("\nResumen del snapshot:")
    for asset, (n_rows, start_ts, end_ts) in summary.items():
        if n_rows == 0:
            print(f"  {asset}: SIN DATOS (descarga falló)")
        else:
            print(f"  {asset}: {n_rows} filas ({start_ts.date()} a {end_ts.date()})")


if __name__ == "__main__":
    main()
