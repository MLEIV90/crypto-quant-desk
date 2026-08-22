"""Script del experimento de Fase 18: Deep RL (PPO) para asignación de
cartera, validado walk-forward y comparado contra 4 baselines.

RESEARCH, no una promesa de rentabilidad — ver `rl/__init__.py` antes de
interpretar cualquier número que imprima este script. El resultado más
probable, y el HONESTO, es que el agente NO le gane de forma consistente a
los baselines fuera de muestra.

Uso:
    python scripts/run_rl_experiment.py
    python scripts/run_rl_experiment.py --seeds 0,1,2,3,4 --timesteps 30000 --blocks 4
    python scripts/run_rl_experiment.py --min-train-days 900 --blocks 5 --out-dir rl/results

Guarda dos archivos en `--out-dir` (default `rl/results/`):
    - `rl_experiment_<timestamp>.csv`: la tabla resumen (una fila por estrategia).
    - `rl_experiment_<timestamp>.json`: la tabla resumen + la conclusión
      automática + los parámetros exactos de la corrida (para poder
      reproducirla o auditarla después).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.evaluation import rl_beats_all_baselines, run_walkforward_experiment, summarize_experiment  # noqa: E402
from rl.features import DEFAULT_ASSETS  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "rl/results"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimento walk-forward de Deep RL (PPO) para asignación de cartera vs. 4 baselines "
            "(Fase 18, research — ver rl/__init__.py)."
        )
    )
    parser.add_argument(
        "--min-train-days", type=int, default=730,
        help="Días mínimos de historia antes del primer tramo de test OOS (default: 730, ~2 años)",
    )
    parser.add_argument("--blocks", type=int, default=4, help="Cantidad de bloques walk-forward (default: 4)")
    parser.add_argument(
        "--seeds", default="0,1,2,3,4",
        help="Semillas separadas por coma para las corridas del agente RL y del asignador aleatorio (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=30_000,
        help="total_timesteps de PPO por bloque y por semilla (default: 30000)",
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
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    n_ppo_runs = args.blocks * len(seeds)
    print(
        f"Corriendo experimento Fase 18 (RESEARCH, ver rl/__init__.py): activos={list(DEFAULT_ASSETS)}, "
        f"min_train_days={args.min_train_days}, bloques={args.blocks}, semillas={list(seeds)}, "
        f"timesteps/corrida={args.timesteps} -> {n_ppo_runs} entrenamientos PPO en total."
    )

    t0 = time.time()
    result = run_walkforward_experiment(
        min_train_days=args.min_train_days,
        n_blocks=args.blocks,
        seeds=seeds,
        total_timesteps=args.timesteps,
        cost_bps=args.cost_bps,
    )
    elapsed_seconds = time.time() - t0

    summary_df = summarize_experiment(result)
    conclusion = rl_beats_all_baselines(result)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"rl_experiment_{timestamp}.csv"
    json_path = out_dir / f"rl_experiment_{timestamp}.json"

    summary_df.to_csv(csv_path, index=False)

    payload = {
        "params": result.params,
        "elapsed_seconds": elapsed_seconds,
        "n_ppo_runs": n_ppo_runs,
        "oos_date_range": [str(result.dates[result.blocks[0].test_start]), str(result.dates[-1])],
        "blocks": [
            {"train_start": b.train_start, "train_end": b.train_end, "test_start": b.test_start, "test_end": b.test_end}
            for b in result.blocks
        ],
        "summary_table": summary_df.to_dict(orient="records"),
        "conclusion": conclusion,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float, ensure_ascii=False)

    pd_display = summary_df.drop(
        columns=[c for c in summary_df.columns if c.endswith("_std") and (summary_df[c] == 0).all()]
    )

    print(f"\nTerminado en {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.1f} min).")
    print(f"Rango OOS evaluado: {payload['oos_date_range'][0]} a {payload['oos_date_range'][1]}")
    print(f"\nTabla resumen (Sharpe/retorno/drawdown/turnover OOS, con costos):\n")
    print(pd_display.to_string(index=False))

    print("\nConclusión automática:")
    veredicto = "SÍ" if conclusion["supera_a_todos_los_baselines_consistentemente"] else "NO"
    print(
        f"  ¿El RL supera a TODOS los baselines OOS, de forma consistente entre semillas? -> {veredicto}"
    )
    print(
        f"  Sharpe RL: peor semilla={conclusion['sharpe_rl_peor_semilla']:.3f}, "
        f"mejor semilla={conclusion['sharpe_rl_mejor_semilla']:.3f}"
    )
    for name, sharpe in conclusion["sharpe_baselines"].items():
        print(f"  Sharpe {name}: {sharpe:.3f}")

    print(f"\nResultados guardados en:\n  {csv_path}\n  {json_path}")

    if not conclusion["supera_a_todos_los_baselines_consistentemente"]:
        print(
            "\nRecordatorio honesto (ver rl/__init__.py): este es el resultado ESPERADO — no encontrar "
            "un edge consistente no es una falla del experimento, es la conclusión de investigación."
        )
    else:
        print(
            "\nADVERTENCIA (ver rl/__init__.py): el RL superó a TODOS los baselines de forma consistente. "
            "Esto es INUSUAL — antes de confiar en el resultado, revisar primero leakage/overfitting al "
            "período de test (¿el mismo resultado se sostiene con otros bloques/semillas/costos?), no "
            "asumir que 'el RL encontró algo real'."
        )


if __name__ == "__main__":
    main()
