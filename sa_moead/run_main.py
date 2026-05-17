"""
run_main.py — 30-run main experiment.

Each run samples one visitor profile (without replacement), fixes a random seed,
and records Pareto-front quality metrics.

Output: results/tables/main_results.csv
"""

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.moead import SA_MOEAD
from src.problem import TourGuideProblem

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "tables"

CSV_FIELDS = [
    "run_id",
    "profile_id",
    "seed",
    "hv",
    "pareto_size",
    "f1_interest_mean",   # higher is better
    "cog_gap_mean",       # abs(cl_mean - cl_opt), lower is better
    "f3_coherence_mean",  # higher is better
    "feasible_ratio",
    "runtime_sec",
]


def load_data() -> tuple[TourGuideProblem, list[dict]]:
    problem = TourGuideProblem.from_files(
        DATA_DIR / "segments.json",
        DATA_DIR / "embeddings.npy",
        T0=300,
        eps=5,
    )
    with open(DATA_DIR / "profiles.json", encoding="utf-8") as f:
        profiles = json.load(f)
    return problem, profiles


def run_single(
    problem: TourGuideProblem,
    profile: dict,
    seed: int,
    pop_size: int,
    t_neighbor: int,
    max_gen: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    algo = SA_MOEAD(
        problem,
        profile,
        pop_size=pop_size,
        T_neighbor=t_neighbor,
        max_gen=max_gen,
    )
    result = algo.run()

    pf = result["pareto_front"]   # shape (M, 3), stored as negated minimisation values
    solutions = result["solutions"]

    # Recover maximisation values
    f1_mean = float(-pf[:, 0].mean())         # interest alignment ↑
    cog_gap_mean = float(pf[:, 1].mean())      # abs cognitive gap ↓ (pf[:,1] = -f2 = abs diff)
    f3_mean = float(-pf[:, 2].mean())          # narrative coherence ↑

    hv = result["history_hv"][-1][1] if result["history_hv"] else 0.0
    n_feasible = sum(1 for s in solutions if problem.is_feasible(s))

    return {
        "hv": hv,
        "pareto_size": len(pf),
        "f1_interest_mean": f1_mean,
        "cog_gap_mean": cog_gap_mean,
        "f3_coherence_mean": f3_mean,
        "feasible_ratio": n_feasible / max(len(solutions), 1),
        "runtime_sec": result["runtime_sec"],
    }


def _fmt(v: Any) -> Any:
    return f"{v:.6f}" if isinstance(v, float) else v


def print_summary(csv_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    hv_vals = [float(r["hv"]) for r in rows]
    f1_vals = [float(r["f1_interest_mean"]) for r in rows]
    cg_vals = [float(r["cog_gap_mean"]) for r in rows]
    f3_vals = [float(r["f3_coherence_mean"]) for r in rows]
    rt_vals = [float(r["runtime_sec"]) for r in rows]

    std = statistics.stdev if n > 1 else lambda _: float("nan")

    print(f"\n{'─'*55}")
    print(f"  Summary over {n} runs")
    print(f"{'─'*55}")
    print(f"  HV              : {statistics.mean(hv_vals):.4f} ± {std(hv_vals):.4f}")
    print(f"  f1 interest     : {statistics.mean(f1_vals):.4f} ± {std(f1_vals):.4f}")
    print(f"  cog gap (↓)     : {statistics.mean(cg_vals):.4f} ± {std(cg_vals):.4f}")
    print(f"  f3 coherence    : {statistics.mean(f3_vals):.4f} ± {std(f3_vals):.4f}")
    print(f"  runtime (s)     : {statistics.mean(rt_vals):.1f} ± {std(rt_vals):.1f}")
    print(f"{'─'*55}")
    print(f"  Saved → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SA-MOEAD main experiment (30 runs)")
    parser.add_argument("--n-runs", type=int, default=30, help="number of independent runs")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--t-neighbor", type=int, default=20)
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=42,
                        help="base seed; run k uses seed (seed_base + k)")
    args = parser.parse_args()

    problem, profiles = load_data()
    print(f"Loaded {problem.N} segments, {len(profiles)} profiles")
    print(f"Running {args.n_runs} experiments "
          f"(pop={args.pop_size}, T={args.t_neighbor}, gen={args.max_gen})\n")

    rng = random.Random(args.seed_base)
    sampled_profiles = rng.sample(profiles, min(args.n_runs, len(profiles)))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "main_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for run_id, profile in enumerate(sampled_profiles, start=1):
            seed = args.seed_base + run_id
            print(f"[{run_id:02d}/{args.n_runs}] profile={profile['id']}  seed={seed}",
                  end="  ", flush=True)

            metrics = run_single(
                problem, profile, seed,
                pop_size=args.pop_size,
                t_neighbor=args.t_neighbor,
                max_gen=args.max_gen,
            )
            print(f"HV={metrics['hv']:.4f}  "
                  f"pareto={metrics['pareto_size']}  "
                  f"t={metrics['runtime_sec']:.1f}s")

            writer.writerow({
                "run_id": run_id,
                "profile_id": profile["id"],
                "seed": seed,
                **{k: _fmt(v) for k, v in metrics.items()},
            })
            f.flush()

    print_summary(csv_path)


if __name__ == "__main__":
    main()
