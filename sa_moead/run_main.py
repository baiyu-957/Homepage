"""
run_main.py — SA-MOEAD vs baselines: fair 30-run experiment.

Architecture (two-phase for unified HV):
  Phase 1: run all experiments, collect raw (pf, solutions, runtime) per run.
  Phase 2: compute UNIFIED_REF = global nadir across all feasible objs + 10% margin,
           then recompute hv_unified and all metrics uniformly.

Fixes applied:
  FIX-1  unified dynamic reference point (two-phase; nadir + 0.1 * span, not nadir * 1.1)
  FIX-2  NSGA2_std sequence-diversity diagnosis printed once per combo
  FIX-3  best_f1 / mean_f1 / best_cog_gap / best_f3 columns
  NEW-1  qualitative_analysis.json export for 4 visitor types

Output: results/tables/main_results_fixed.csv
        results/tables/qualitative_analysis.json
"""

import argparse
import csv
import json
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

# Richer profiles for qualitative case study (NEW-1)
QUALITATIVE_PROFILES: dict[str, dict] = {
    "child":   {"id": "child",   "age": 0.0,  "edu": 0.25, "cog": 0.2, "purpose": 0.0,
                "interest": [0.05, 0.05, 0.15, 0.25, 0.50]},
    "scholar": {"id": "scholar", "age": 0.67, "edu": 1.0,  "cog": 1.0, "purpose": 1.0,
                "interest": [0.50, 0.25, 0.10, 0.10, 0.05]},
    "senior":  {"id": "senior",  "age": 1.0,  "edu": 0.5,  "cog": 0.3, "purpose": 0.5,
                "interest": [0.15, 0.35, 0.25, 0.20, 0.05]},
    "tourist": {"id": "tourist", "age": 0.67, "edu": 0.5,  "cog": 0.5, "purpose": 0.0,
                "interest": [0.20, 0.20, 0.20, 0.20, 0.20]},
}

ALL_METHODS = ["SA_MOEAD", "Random", "Greedy", "NSGA2_std", "MOEAD_std"]
ALL_T0 = [300, 600]

# FIX-3: extended CSV fields
CSV_FIELDS = [
    "method", "profile_type", "T0", "run_id", "seed",
    "hv_unified", "ref_f1", "ref_f2", "ref_f3",
    "pareto_size", "pareto_feas",
    "best_f1", "mean_f1", "best_cog_gap", "best_f3",
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


def _init_weights(pop_size: int, n_obj: int) -> np.ndarray:
    W = _das_dennis(n_obj, pop_size - 1)
    if len(W) > pop_size:
        W = W[np.random.choice(len(W), pop_size, replace=False)]
    elif len(W) < pop_size:
        W = np.vstack([W, np.random.dirichlet(np.ones(n_obj), pop_size - len(W))])
    return W


def _compute_hv(pf: np.ndarray, ref: np.ndarray) -> float:
    if len(pf) == 0:
        return 0.0
    valid = np.all(pf < ref, axis=1)
    pf_valid = pf[valid]
    if len(pf_valid) == 0:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=ref)(pf_valid))
    except Exception:
        return float(np.sum(np.prod(ref - pf_valid, axis=1).clip(0)))


# -----------------------------------------------------------------------
# FIX-1: unified reference point (two-phase)
# -----------------------------------------------------------------------

def compute_unified_ref(
    raw_results: list[dict], problems: dict[int, TourGuideProblem]
) -> np.ndarray:
    """Global nadir + 10% of span. Handles negative objectives correctly."""
    all_objs: list[list[float]] = []
    for r in raw_results:
        problem = problems[r["T0"]]
        for obj, sol in zip(r["pf"], r["solutions"]):
            if problem.is_feasible(sol):
                all_objs.append(obj.tolist())

    if not all_objs:
        return np.array([0.05, 1.0, 0.05])

    arr = np.array(all_objs)
    nadir = arr.max(axis=0)
    ideal = arr.min(axis=0)
    span = np.abs(nadir - ideal)
    # Add 10% of span so all feasible points are strictly dominated by ref
    return nadir + 0.1 * np.where(span > 1e-9, span, 0.1)


# -----------------------------------------------------------------------
# FIX-1 + FIX-3: unified metrics computation
# -----------------------------------------------------------------------

def _compute_metrics(
    problem: TourGuideProblem,
    pf: np.ndarray,
    solutions: list[list[int]],
    runtime: float,
    unified_ref: np.ndarray,
) -> dict[str, Any]:
    """All metrics computed from feasible solutions only, using unified_ref."""
    feas_mask = np.array([problem.is_feasible(s) for s in solutions], dtype=bool)
    n_feas = int(feas_mask.sum())
    pf_feas = pf[feas_mask] if n_feas > 0 else np.empty((0, 3))

    hv_unified = _compute_hv(pf_feas, unified_ref)

    if n_feas > 0:
        # FIX-3: best/mean metrics across feasible Pareto solutions
        best_f1 = float(-pf_feas[:, 0].min())     # max interest score
        mean_f1 = float(-pf_feas[:, 0].mean())
        best_cog_gap = float(pf_feas[:, 1].min()) # min gap = best cognitive match
        best_f3 = float(-pf_feas[:, 2].min())     # max coherence score
        f1_mean = mean_f1
        cg_mean = float(pf_feas[:, 1].mean())
        f3_mean = float(-pf_feas[:, 2].mean())
    else:
        best_f1 = mean_f1 = best_cog_gap = best_f3 = 0.0
        f1_mean = cg_mean = f3_mean = 0.0

    return {
        "hv_unified": hv_unified,
        "ref_f1": float(unified_ref[0]),
        "ref_f2": float(unified_ref[1]),
        "ref_f3": float(unified_ref[2]),
        "pareto_size": len(pf),
        "pareto_feas": n_feas,
        "best_f1": best_f1,
        "mean_f1": mean_f1,
        "best_cog_gap": best_cog_gap,
        "best_f3": best_f3,
        "f1_interest_mean": f1_mean,
        "cog_gap_mean": cg_mean,
        "f3_coherence_mean": f3_mean,
        "feasible_ratio": n_feas / max(len(solutions), 1),
        "runtime_sec": runtime,
    }


# -----------------------------------------------------------------------
# Phase-1 raw runners: return (pf, solutions, runtime_sec)
# -----------------------------------------------------------------------

def _raw_random(
    problem: TourGuideProblem, profile: dict, seed: int, **_: Any
) -> tuple[np.ndarray, list[list[int]], float]:
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

    pf = problem.evaluate(best_seq, profile).reshape(1, 3)
    return pf, [best_seq], time.time() - t0


def _raw_greedy(
    problem: TourGuideProblem, profile: dict, seed: int, **_: Any
) -> tuple[np.ndarray, list[list[int]], float]:
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
    pf = problem.evaluate(seq, profile).reshape(1, 3)
    return pf, [seq], time.time() - t0


def _raw_nsga2_std(
    problem: TourGuideProblem, profile: dict, seed: int,
    pop_size: int = 100, max_gen: int = 200,
    diagnose: bool = False, **_: Any
) -> tuple[np.ndarray, list[list[int]], float]:
    """FIX-2: ECR repair inside _evaluate; optional sequence-diversity diagnosis."""
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
            seq = ecr_repair(seq, problem)
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
        solutions = [ecr_repair(_decode_perm(np.argsort(x), problem), problem)
                     for x in res.X]
    else:
        obj0 = problem.evaluate([], profile)
        pf, solutions = obj0.reshape(1, 3), [[]]

    # FIX-2: print sequence-diversity diagnosis once per combo when requested
    if diagnose:
        feas_mask = [problem.is_feasible(s) for s in solutions]
        feas_sols = [s for s, f in zip(solutions, feas_mask) if f]
        unique = len({tuple(sorted(s)) for s in feas_sols}) if feas_sols else 0
        f1_vals = [-pf[i, 0] for i, f in enumerate(feas_mask) if f]
        f1_range = (f"[{min(f1_vals):.3f}, {max(f1_vals):.3f}]"
                    if f1_vals else "n/a")
        print(f"    [NSGA2_std FIX-2 diag] "
              f"feasible={sum(feas_mask)}/{len(solutions)} "
              f"unique_seqs={unique}/{len(feas_sols)} "
              f"f1_range={f1_range}")

    return pf, solutions, time.time() - t0


def _raw_moead_std(
    problem: TourGuideProblem, profile: dict, seed: int,
    pop_size: int = 100, t_neighbor: int = 20, max_gen: int = 200, **_: Any
) -> tuple[np.ndarray, list[list[int]], float]:
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
    return pf, solutions, time.time() - t0


def _raw_sa_moead(
    problem: TourGuideProblem, profile: dict, seed: int,
    pop_size: int = 100, t_neighbor: int = 20, max_gen: int = 200, **_: Any
) -> tuple[np.ndarray, list[list[int]], float]:
    random.seed(seed)
    np.random.seed(seed)
    algo = SA_MOEAD(problem, profile,
                    pop_size=pop_size, T_neighbor=t_neighbor, max_gen=max_gen)
    result = algo.run()
    return result["pareto_front"], result["solutions"], result["runtime_sec"]


_RAW_RUNNERS: dict[str, Any] = {
    "SA_MOEAD":  _raw_sa_moead,
    "Random":    _raw_random,
    "Greedy":    _raw_greedy,
    "NSGA2_std": _raw_nsga2_std,
    "MOEAD_std": _raw_moead_std,
}

# Track which (method, profile_type, T0) combos have had FIX-2 diagnosis printed
_nsga2_diagnosed: set[tuple] = set()


def _run_raw(
    method: str, problem: TourGuideProblem, profile: dict, seed: int,
    pop_size: int, t_neighbor: int, max_gen: int,
    profile_type: str, T0: int,
) -> tuple[np.ndarray, list[list[int]], float]:
    combo_key = (profile_type, T0)
    diagnose = (method == "NSGA2_std" and combo_key not in _nsga2_diagnosed)
    if diagnose:
        _nsga2_diagnosed.add(combo_key)

    return _RAW_RUNNERS[method](
        problem, profile, seed,
        pop_size=pop_size, t_neighbor=t_neighbor, max_gen=max_gen,
        diagnose=diagnose,
    )


# -----------------------------------------------------------------------
# NEW-1: qualitative analysis export
# -----------------------------------------------------------------------

def export_qualitative(
    problem: TourGuideProblem,
    out_path: Path,
    pop_size: int = 100,
    max_gen: int = 300,
) -> None:
    """Run SA_MOEAD on QUALITATIVE_PROFILES, export best-f1 solution to JSON."""
    results: dict[str, Any] = {}

    for ptype, profile in QUALITATIVE_PROFILES.items():
        random.seed(42)
        np.random.seed(42)

        algo = SA_MOEAD(problem, profile,
                        pop_size=pop_size, T_neighbor=20, max_gen=max_gen)
        raw = algo.run()

        pf = raw["pareto_front"]
        solutions = raw["solutions"]

        feas_mask = np.array([problem.is_feasible(s) for s in solutions], dtype=bool)
        if not feas_mask.any():
            results[ptype] = {"error": "no_feasible_solution"}
            continue

        pf_feas = pf[feas_mask]
        feas_sols = [s for s, f in zip(solutions, feas_mask.tolist()) if f]

        # Best-f1 solution
        best_idx = int(np.argmin(pf_feas[:, 0]))   # most negative = highest f1
        best_seq = feas_sols[best_idx]
        best_obj = pf_feas[best_idx]

        seg_details = []
        for seg_idx in best_seq:
            seg = problem.segs[seg_idx]
            seg_details.append({
                "seg_id": int(seg_idx),
                "name": seg.get("name", f"seg_{seg_idx}"),
                "category": seg.get("category", ""),
                "duration_sec": int(seg.get("duration_sec", 0)),
                "f1_contribution": float(problem.f1_interest([seg_idx], profile)),
            })

        results[ptype] = {
            "profile": profile,
            "best_f1": float(-best_obj[0]),
            "best_cog_gap": float(best_obj[1]),
            "best_f3": float(-best_obj[2]),
            "total_duration_sec": int(sum(s["duration_sec"] for s in seg_details)),
            "pareto_size": int(len(pf_feas)),
            "segments": seg_details,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [NEW-1] qualitative_analysis.json → {out_path}")


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

    print(f"\n{'─'*96}")
    print(f"{'Method':12s} {'Profile':8s} {'T0':>5s} | "
          f"{'HV_unified':>14s}  {'best_f1':>10s}  {'best_f3':>10s}  {'t(s)':>10s}")
    print(f"{'─'*96}")

    for (method, pt, t0), grp in sorted(groups.items()):
        n = len(grp)
        hv = [float(r["hv_unified"]) for r in grp]
        bf1 = [float(r["best_f1"]) for r in grp]
        bf3 = [float(r["best_f3"]) for r in grp]
        rt = [float(r["runtime_sec"]) for r in grp]
        _s = sf if n > 1 else lambda _: float("nan")
        print(
            f"{method:12s} {pt:8s} {t0:>5s} | "
            f"{statistics.mean(hv):.4f}±{_s(hv):.4f}  "
            f"{statistics.mean(bf1):.4f}±{_s(bf1):.4f}  "
            f"{statistics.mean(bf3):.4f}±{_s(bf3):.4f}  "
            f"{statistics.mean(rt):6.1f}±{_s(rt):.1f}"
        )

    print(f"{'─'*96}")
    print(f"  Saved → {csv_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SA-MOEAD vs baselines, 30 runs × profile × T0 (two-phase unified HV)"
    )
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--t-neighbor", type=int, default=20)
    parser.add_argument("--max-gen", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, metavar="METHOD")
    parser.add_argument("--t0-values", nargs="+", type=int, default=ALL_T0)
    parser.add_argument("--skip-qualitative", action="store_true",
                        help="Skip qualitative_analysis.json export")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "main_results_fixed.csv"
    qual_path = RESULTS_DIR / "qualitative_analysis.json"

    problems: dict[int, TourGuideProblem] = {
        T0: load_problem(T0) for T0 in args.t0_values
    }
    n_segs = next(iter(problems.values())).N
    combos = list(product(args.methods, PROFILE_TEMPLATES.keys(), args.t0_values))
    total = len(combos) * args.n_runs

    print(f"Segments: {n_segs}  |  "
          f"{len(combos)} combinations × {args.n_runs} runs = {total} total")
    print(f"pop={args.pop_size}  T={args.t_neighbor}  gen={args.max_gen}")
    print("Architecture: two-phase (Phase-1 collect raw → Phase-2 unified ref → metrics)\n")

    # ── Phase 1: collect raw results ──────────────────────────────────────
    print("=== Phase 1: running experiments ===")
    raw_results: list[dict] = []
    done = 0

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

            pf, solutions, runtime = _run_raw(
                method, problem, profile, seed,
                pop_size=args.pop_size,
                t_neighbor=args.t_neighbor,
                max_gen=args.max_gen,
                profile_type=profile_type,
                T0=T0,
            )
            feas_count = sum(1 for s in solutions if problem.is_feasible(s))
            print(
                f"pareto={len(pf):3d}(feas={feas_count:3d})  "
                f"t={runtime:.1f}s"
            )

            raw_results.append({
                "method": method,
                "profile_type": profile_type,
                "T0": T0,
                "run_id": run_id,
                "seed": seed,
                "pf": pf,
                "solutions": solutions,
                "runtime_sec": runtime,
            })

    # ── Phase 2: compute unified reference point ──────────────────────────
    print("\n=== Phase 2: computing unified reference point ===")
    unified_ref = compute_unified_ref(raw_results, problems)
    print(f"UNIFIED_REF = [{unified_ref[0]:.4f}, {unified_ref[1]:.4f}, {unified_ref[2]:.4f}]")

    # ── Phase 2: compute metrics and write CSV ────────────────────────────
    print("\n=== Phase 2: writing metrics to CSV ===")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for r in raw_results:
            problem = problems[r["T0"]]
            metrics = _compute_metrics(
                problem, r["pf"], r["solutions"], r["runtime_sec"], unified_ref
            )
            writer.writerow({
                "method": r["method"],
                "profile_type": r["profile_type"],
                "T0": r["T0"],
                "run_id": r["run_id"],
                "seed": r["seed"],
                **{k: _fmt(v) for k, v in metrics.items()},
            })

    # ── NEW-1: qualitative analysis ────────────────────────────────────────
    if not args.skip_qualitative:
        print("\n=== NEW-1: qualitative analysis ===")
        export_qualitative(problems[300], qual_path,
                           pop_size=args.pop_size, max_gen=args.max_gen)

    print_summary(csv_path)


if __name__ == "__main__":
    main()
