"""Script del experimento de Fase 19a: backtest honesto de rotación por
momentum relativo entre monedas, contra 3 baselines, sobre TODA una grilla
de parámetros (lookback x rebalanceo) y TODOS los pares del universo — el
TEST PREVIO al ML (ver `strategies/rotation.py`).

Reporta TODAS las combinaciones probadas, nunca solo la mejor — reportar
solo el ganador sería, literalmente, el sobreajuste que este script existe
para descartar.

Uso:
    python scripts/run_rotation_experiment.py
    python scripts/run_rotation_experiment.py --pairs BTC-ETH,BTC-SOL --lookbacks 20,50,100 --rebalances 7,30
    python scripts/run_rotation_experiment.py --primary-pair ETH-BTC --out-dir strategies/results

Guarda dos archivos en `--out-dir` (default `strategies/results/`):
    - `rotation_experiment_<timestamp>.csv`: la tabla completa (una fila
      por combinación de par x lookback x rebalanceo).
    - `rotation_experiment_<timestamp>.json`: la misma tabla + la
      conclusión automática + los parámetros exactos de la corrida.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import UNIVERSE  # noqa: E402
from strategies.rotation import (  # noqa: E402
    DEFAULT_LOOKBACK_GRID,
    DEFAULT_PAIRS,
    DEFAULT_REBALANCE_GRID,
    rotation_beats_baselines_robustly,
    run_full_experiment,
    summarize_experiment,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "strategies/results"
DEFAULT_PRIMARY_PAIR = "ETH-BTC"


def _parse_pairs(raw: str | None) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return DEFAULT_PAIRS
    pairs = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        asset_a, _, asset_b = token.partition("-")
        if asset_a not in UNIVERSE or asset_b not in UNIVERSE:
            raise ValueError(f"--pairs: '{token}' no son dos activos válidos de config.UNIVERSE ({list(UNIVERSE)})")
        pairs.append((asset_a, asset_b))
    return tuple(pairs)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in raw.split(",") if v.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest honesto de rotación por momentum relativo vs. baselines, sobre toda una grilla "
            "de parámetros y todos los pares del universo (Fase 19a — ver strategies/rotation.py)."
        )
    )
    parser.add_argument(
        "--pairs", default=None,
        help="Pares separados por coma, formato 'A-B' (ej. 'BTC-ETH,BTC-SOL'). Default: los 10 pares de config.UNIVERSE",
    )
    parser.add_argument(
        "--primary-pair", default=DEFAULT_PRIMARY_PAIR,
        help=f"Par principal para la conclusión (default: {DEFAULT_PRIMARY_PAIR})",
    )
    parser.add_argument(
        "--lookbacks", default=",".join(str(v) for v in DEFAULT_LOOKBACK_GRID),
        help=f"Lookbacks en días, separados por coma (default: {DEFAULT_LOOKBACK_GRID})",
    )
    parser.add_argument(
        "--rebalances", default=",".join(str(v) for v in DEFAULT_REBALANCE_GRID),
        help=f"Frecuencias de rebalanceo en días, separadas por coma (default: {DEFAULT_REBALANCE_GRID})",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=None,
        help="Costo de transacción en basis points (default: config.TRANSACTION_COST_BPS)",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Directorio de salida (default: {DEFAULT_OUT_DIR})")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()

    pairs = _parse_pairs(args.pairs)
    lookback_grid = _parse_int_list(args.lookbacks)
    rebalance_grid = _parse_int_list(args.rebalances)
    primary_asset_a, _, primary_asset_b = args.primary_pair.partition("-")
    primary_pair = (primary_asset_a, primary_asset_b)

    n_combos = len(pairs) * len(lookback_grid) * len(rebalance_grid)
    print(
        f"Corriendo experimento Fase 19a (test previo al ML — ver strategies/rotation.py): "
        f"{len(pairs)} pares x {len(lookback_grid)} lookbacks x {len(rebalance_grid)} rebalanceos "
        f"= {n_combos} backtests. Par principal: {args.primary_pair}."
    )

    t0 = time.time()
    results = run_full_experiment(
        pairs=pairs, lookback_grid=lookback_grid, rebalance_grid=rebalance_grid, cost_bps=args.cost_bps
    )
    elapsed_seconds = time.time() - t0

    summary_df = summarize_experiment(results)
    conclusion = rotation_beats_baselines_robustly(results, primary_pair=primary_pair)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"rotation_experiment_{timestamp}.csv"
    json_path = out_dir / f"rotation_experiment_{timestamp}.json"

    summary_df.to_csv(csv_path, index=False)

    payload = {
        "params": {
            "pairs": [f"{a}-{b}" for a, b in pairs],
            "lookback_grid": list(lookback_grid),
            "rebalance_grid": list(rebalance_grid),
            "cost_bps": args.cost_bps,
            "primary_pair": args.primary_pair,
        },
        "elapsed_seconds": elapsed_seconds,
        "n_combos": n_combos,
        "summary_table": summary_df.to_dict(orient="records"),
        "per_pair_robusto": {f"{r.asset_a}-{r.asset_b}": r.robusto for r in results},
        "conclusion": conclusion,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float, ensure_ascii=False)

    print(f"\nTerminado en {elapsed_seconds:.1f}s.")
    print(f"\nTabla completa ({len(summary_df)} filas, TODAS las combinaciones, ninguna descartada):\n")
    display_cols = [
        "par", "lookback_days", "rebalance_days", "sharpe_rotacion", "cagr_rotacion",
        "max_drawdown_rotacion", "n_rotaciones", "mejor_baseline", "sharpe_mejor_baseline", "gana_al_mejor_baseline",
    ]
    print(summary_df[display_cols].to_string(index=False))

    print("\nRobustez por par (¿ganó en TODAS las combinaciones de parámetros?):")
    for pair_key, robusto in payload["per_pair_robusto"].items():
        print(f"  {pair_key}: {'SÍ' if robusto else 'no'}")

    print("\nConclusión automática:")
    veredicto = "SÍ" if conclusion["veredicto_global"] else "NO"
    print(
        f"  ¿La rotación por momentum supera a quedarse quieto, después de costos, de forma ROBUSTA? -> {veredicto}"
    )
    print(f"  Par principal ({conclusion['par_principal']}) robusto en todas sus combinaciones: "
          f"{'SÍ' if conclusion['robusto_par_principal'] else 'no'}")
    print(
        f"  Fracción de TODOS los pares corridos que fueron robustos: "
        f"{conclusion['fraccion_pares_robustos']:.0%} ({conclusion['pares_robustos']})"
    )

    print(f"\nResultados guardados en:\n  {csv_path}\n  {json_path}")

    if not conclusion["veredicto_global"]:
        print(
            "\nRecordatorio honesto (ver strategies/rotation.py): este es el resultado ESPERADO — si la regla "
            "simple de rotación no le gana de forma robusta a quedarse quieto, un modelo de ML sobre la misma "
            "idea tampoco lo haría. No hace falta intentar la versión con ML de esto."
        )
    else:
        print(
            "\nADVERTENCIA: la rotación superó a los baselines de forma robusta. Esto es INUSUAL para una regla "
            "de momentum simple entre criptos altamente correlacionadas — antes de considerar una versión con "
            "ML, revisar si el resultado se sostiene con otros costos/rangos de fecha, no asumir que ya está probado."
        )


if __name__ == "__main__":
    main()
