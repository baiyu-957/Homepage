"""
run_main.py — SA-MOEAD vs baselines: 30-run experiment per (method × profile_type × T0).

Methods  : SA_MOEAD, Random, Greedy, NSGA2_std, MOEAD_std
Profiles : child, scholar, tourist, senior
T0 values: 300 s, 600 s

Output: results/tables/main_results.csv
"""

import argparse
import csv
import random
import statistics
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.moead import SA_MOEAD
from src.operators import ecr_repair
from src.problem import TourGuideProblem

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "tables"

# Slightly looser eps for T0=600 to keep feasibility achievable
T0_EPS: dict[int, int] = {300: 5, 600: 10}

PROFILE_TEMPLATES: dict[str, dict] = {
    "child": {
        "id": "child", "age": 0.0, "edu": 0.25, "cog": 0.3, "purpose": 0.0,
        "interest": [0.1, 0.1, 0.2, 0.1, 0.5],
    },
    "scholar": {
        "id": "scholar", "age": 0.67, "edu": 1.0, "cog": 0.9, "purpose": 0.5,
        "interest": [0.6, 0.1, 0.1, 0.1, 0.1],
    },
    "tourist": {
        "id": "tourist", "age": 0.67, "edu": 0.5, "cog": 0.5, "purpose": 0.5,
        "interest": [0.2, 0.2, 0.2, 0.2, 0.2],
    },
    "senior": {
        "id": "senior", "age": 1.0, "edu": 0.5, "cog": 0.4, "purpose": 0.0,
        "interest": [0.1, 0.1, 0.1, 0.6, 0.1],
    },
}

ALL_METHODS = ["SA_MOEAD", "Random", "Greedy", "NSGA2_std", "MOEAD_std"]
ALL_T0 = [300, 600]

CSV_FIELDS = [
    "method", "profile_type", "T0", "run_id", "seed",
    "hv", "pareto_size",
    "f1_interest_mean", "cog_gap_mean", "f3_coherence_mean",
    "feasible_ratio", "runtime_sec",
]


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_problem(T0: int) -> TourGuideProblem:
    return TourGuideProblem.from_files(
        DATA_DIR / "segments.json",
        DATA_DIR / "embeddings.npy",
        T0=T0,
        eps=T0_EPS[T0],
    )


# -----------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------

def _decode_perm(perm: np.ndarray, problem: TourGuideProblem) -> list[int]:
    """Decode a permutation to a feasible segment sequence (greedy prefix)."""
    chosen: list[int] = []
    total = 0
    for raw in perm:
        idx = int(raw)
        dur = problem.segs[idx]["duration_sec"]
        if total + dur <= problem.T0 + problem.eps * 3:
            chosen.append(idx)
            total += dur
        if total >= problem.T0 - problem.eps:
            break
    return chosen


def _ox_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Order crossover (OX) — preserves relative order from both parents."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=int)
    child[a : b + 1] = p1[a : b + 1]
    segment = set(p1[a : b + 1].tolist())
    fill = [x for x in p2 if x not in segment]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return child


def _swap_mutation(perm: np.ndarray) -> np.ndarray:
    """Swap two randomly chosen positions."""
    result = perm.copy()
    i, j = random.sample(range(len(perm)), 2)
    result[i], result[j] = result[j], result[i]
    return result


def _das_dennis(n_obj: int, n_partitions: int) -> np.ndarray:
    def _rec(n: int, left: int, prefix: list[int]):
        if n == 1:
            yield prefix + [left]
            return
        for i in range(left + 1):
            yield from _rec(n - 1, left - i, prefix + [i])

    return np.array(
        [[c / n_partitions for c in combo] for combo in _rec(n_obj, n_partitions, [])],
        dtype=np.float64,
    )


def _tchebycheff(obj: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
    return float(np.max(w * np.abs(obj - z)))


def _non_dominated(obj_pop: np.ndarray) -> list[int]:
    n = len(obj_pop)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if np.all(obj_pop[j] <= obj_pop[i]) and np.any(obj_pop[j] < obj_pop[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def _compute_hv(pf: np.ndarray) -> float:
    if len(pf) == 0:
        return 0.0
    ref = pf.max(axis=0) + np.array([0.05, 0.1, 0.05])
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=ref)(pf))
    except Exception:
        return float(np.sum(np.prod(ref - pf, axis=1).clip(0)))


def _extract_metrics(
    problem: TourGuideProblem,
    pf: np.ndarray,
    solutions: list[list[int]],
    runtime: float,
) -> dict[str, Any]:
    hv = _compute_hv(pf)
    n_feasible = sum(1 for s in solutions if problem.is_feasible(s))
    return {
        "hv": hv,
        "pareto_size": len(pf),
        "f1_interest_mean": float(-pf[:, 0].mean()),
        "cog_gap_mean": float(pf[:, 1].mean()),    # abs(cl_mean - cl_opt) ↓
        "f3_coherence_mean": float(-pf[:, 2].mean()),
        "feasible_ratio": n_feasible / max(len(solutions), 1),
        "runtime_sec": runtime,
    }


def _init_weights(pop_size: int, n_obj: int) -> np.ndarray:
    """Generate pop_size weight vectors via Das-Dennis; subsample or pad as needed."""
    W = _das_dennis(n_obj, pop_size - 1)
    if len(W) > pop_size:
        W = W[np.random.choice(len(W), pop_size, replace=False)]
    elif len(W) < pop_size:
        W = np.vstack([W, np.random.dirichlet(np.ones(n_obj), pop_size - len(W))])
    return W


# -----------------------------------------------------------------------
# Method implementations
# -----------------------------------------------------------------------

def run_random(
    problem: TourGuideProblem, profile: dict, seed: int, **_: Any
) -> dict[str, Any]:
    """10-trial random search; keep the solution with highest f1."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    random.seed(seed)

    best_seq: list[int] = []
    best_f1 = -np.inf

    for _ in range(10):
        candidate = _decode_perm(rng.permutation(problem.N), problem)
        candidate = ecr_repair(candidate, problem)
        f1 = problem.f1_interest(candidate, profile)
        if f1 > best_f1:
            best_f1, best_seq = f1, candidate

    if not best_seq:
        best_seq = ecr_repair(list(range(problem.N)), problem)

    obj = problem.evaluate(best_seq, profile).reshape(1, 3)
    return _extract_metrics(problem, obj, [best_seq], time.time() - t0)


def run_greedy(
    problem: TourGuideProblem, profile: dict, seed: int, **_: Any
) -> dict[str, Any]:
    """Greedy: sort segments by f1(s, profile) descending, fill up to T0, ECR repair."""
    t0 = time.time()

    order = sorted(range(problem.N),
                   key=lambda i: problem.f1_interest([i], profile),
                   reverse=True)

    chosen: list[int] = []
    total = 0
    for idx in order:
        dur = problem.segs[idx]["duration_sec"]
        if total + dur <= problem.T0 + problem.eps * 3:
            chosen.append(idx)
            total += dur
        if total >= problem.T0 - problem.eps:
            break

    seq = ecr_repair(chosen, problem)
    obj = problem.evaluate(seq, profile).reshape(1, 3)
    return _extract_metrics(problem, obj, [seq], time.time() - t0)


def run_nsga2_std(
    problem: TourGuideProblem, profile: dict, seed: int,
    pop_size: int = 100, max_gen: int = 200, **_: Any
) -> dict[str, Any]:
    """NSGA-II with random-key encoding (SBX + polynomial mutation), no SAX/SNM."""
    t0 = time.time()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    np.random.seed(seed)
    random.seed(seed)

    class _Prob(ElementwiseProblem):
        def __init__(self_):
            super().__init__(n_var=problem.N, n_obj=3, xl=0.0, xu=1.0)

        def _evaluate(self_, x: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
            perm = np.argsort(x)
            seq = _decode_perm(perm, problem)
            out["F"] = problem.evaluate(seq, profile)

    res = minimize(
        _Prob(),
        NSGA2(pop_size=pop_size),
        ("n_gen", max_gen),
        seed=seed,
        verbose=False,
    )

    if res.F is not None and len(res.F) > 0:
        pf = res.F
        solutions = [_decode_perm(np.argsort(x), problem) for x in res.X]
    else:
        obj0 = problem.evaluate([], profile)
        pf, solutions = obj0.reshape(1, 3), [[]]

    return _extract_metrics(problem, pf, solutions, time.time() - t0)


def run_moead_std(
    problem: TourGuideProblem, profile: dict, seed: int,
    pop_size: int = 100, t_neighbor: int = 20, max_gen: int = 200, **_: Any
) -> dict[str, Any]:
    """MOEA/D with OX crossover + swap mutation on permutation encoding, no SAX/SNM/ECR."""
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)

    n_obj, N = 3, problem.N
    W = _init_weights(pop_size, n_obj)

    dists = np.sum((W[:, None] - W[None]) ** 2, axis=2)
    neighbors = [list(np.argsort(dists[i])[:t_neighbor]) for i in range(pop_size)]

    pop_perms = [np.random.permutation(N) for _ in range(pop_size)]
    pop_seqs = [_decode_perm(p, problem) for p in pop_perms]
    obj_pop = np.array([problem.evaluate(seq, profile) for seq in pop_seqs])
    z_star = obj_pop.min(axis=0).copy()

    for _ in range(max_gen):
        for i in range(pop_size):
            nb = neighbors[i]
            p1_i, p2_i = random.sample(nb, 2)

            child_perm = _ox_crossover(pop_perms[p1_i], pop_perms[p2_i])
            child_perm = _swap_mutation(child_perm)
            child_seq = _decode_perm(child_perm, problem)
            child_obj = problem.evaluate(child_seq, profile)

            z_star = np.minimum(z_star, child_obj)

            for j in nb:
                if (_tchebycheff(child_obj, W[j], z_star) <=
                        _tchebycheff(obj_pop[j], W[j], z_star)):
                    pop_perms[j] = child_perm
                    pop_seqs[j] = child_seq
                    obj_pop[j] = child_obj

    pareto_idx = _non_dominated(obj_pop)
    pf = obj_pop[pareto_idx]
    solutions = [pop_seqs[i] for i in pareto_idx]
    return _extract_metrics(problem, pf, solutions, time.time() - t0)


def run_sa_moead(
    problem: TourGuideProblem, profile: dict, seed: int,
    pop_size: int = 100, t_neighbor: int = 20, max_gen: int = 200, **_: Any
) -> dict[str, Any]:
    """SA-MOEAD with SAX crossover, SNM mutation, and ECR repair."""
    random.seed(seed)
    np.random.seed(seed)

    algo = SA_MOEAD(problem, profile,
                    pop_size=pop_size, T_neighbor=t_neighbor, max_gen=max_gen)
    result = algo.run()

    pf = result["pareto_front"]
    solutions = result["solutions"]
    hv = result["history_hv"][-1][1] if result["history_hv"] else _compute_hv(pf)

    n_feasible = sum(1 for s in solutions if problem.is_feasible(s))
    return {
        "hv": hv,
        "pareto_size": len(pf),
        "f1_interest_mean": float(-pf[:, 0].mean()),
        "cog_gap_mean": float(pf[:, 1].mean()),
        "f3_coherence_mean": float(-pf[:, 2].mean()),
        "feasible_ratio": n_feasible / max(len(solutions), 1),
        "runtime_sec": result["runtime_sec"],
    }


_RUNNERS: dict[str, Any] = {
    "SA_MOEAD":  run_sa_moead,
    "Random":    run_random,
    "Greedy":    run_greedy,
    "NSGA2_std": run_nsga2_std,
    "MOEAD_std": run_moead_std,
}


def _run(method: str, problem: TourGuideProblem, profile: dict, seed: int,
         pop_size: int, t_neighbor: int, max_gen: int) -> dict[str, Any]:
    return _RUNNERS[method](
        problem, profile, seed,
        pop_size=pop_size, t_neighbor=t_neighbor, max_gen=max_gen,
    )


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

def _fmt(v: Any) -> Any:
    return f"{v:.6f}" if isinstance(v, float) else v


def print_summary(csv_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r["method"], r["profile_type"], r["T0"])
        groups.setdefault(key, []).append(r)

    sf = statistics.stdev

    print(f"\n{'─'*84}")
    print(f"{'Method':12s} {'Profile':8s} {'T0':>5s} | "
          f"{'HV':>14s}  {'f1':>12s}  {'f3':>12s}  {'t(s)':>10s}")
    print(f"{'─'*84}")

    for (method, pt, t0), grp in sorted(groups.items()):
        n = len(grp)
        hv = [float(r["hv"]) for r in grp]
        f1 = [float(r["f1_interest_mean"]) for r in grp]
        f3 = [float(r["f3_coherence_mean"]) for r in grp]
        rt = [float(r["runtime_sec"]) for r in grp]
        _s = sf if n > 1 else lambda _: float("nan")
        print(
            f"{method:12s} {pt:8s} {t0:>5s} | "
            f"{statistics.mean(hv):.4f}±{_s(hv):.4f}  "
            f"{statistics.mean(f1):.4f}±{_s(f1):.4f}  "
            f"{statistics.mean(f3):.4f}±{_s(f3):.4f}  "
            f"{statistics.mean(rt):6.1f}±{_s(rt):.1f}"
        )

    print(f"{'─'*84}")
    print(f"  Saved → {csv_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SA-MOEAD vs baselines, 30 runs × profile × T0"
    )
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--t-neighbor", type=int, default=20)
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, metavar="METHOD")
    parser.add_argument("--t0-values", nargs="+", type=int, default=ALL_T0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "main_results.csv"

    problems: dict[int, TourGuideProblem] = {
        T0: load_problem(T0) for T0 in args.t0_values
    }
    n_segs = next(iter(problems.values())).N
    combos = list(product(args.methods, PROFILE_TEMPLATES.keys(), args.t0_values))
    total = len(combos) * args.n_runs

    print(f"Segments: {n_segs}  |  "
          f"{len(combos)} combinations × {args.n_runs} runs = {total} total")
    print(f"pop={args.pop_size}  T={args.t_neighbor}  gen={args.max_gen}\n")

    done = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for method, profile_type, T0 in combos:
            problem = problems[T0]
            profile = PROFILE_TEMPLATES[profile_type]

            for run_id in range(1, args.n_runs + 1):
                seed = args.seed_base + run_id
                done += 1
                print(
                    f"[{done:4d}/{total}] {method:12s} {profile_type:8s} "
                    f"T0={T0:3d}  run={run_id:02d}",
                    end="  ", flush=True,
                )

                metrics = _run(
                    method, problem, profile, seed,
                    pop_size=args.pop_size,
                    t_neighbor=args.t_neighbor,
                    max_gen=args.max_gen,
                )
                print(
                    f"HV={metrics['hv']:.4f}  "
                    f"pareto={metrics['pareto_size']:3d}  "
                    f"t={metrics['runtime_sec']:.1f}s"
                )

                writer.writerow({
                    "method": method,
                    "profile_type": profile_type,
                    "T0": T0,
                    "run_id": run_id,
                    "seed": seed,
                    **{k: _fmt(v) for k, v in metrics.items()},
                })
                f.flush()

    print_summary(csv_path)


if __name__ == "__main__":
    main()
