"""
Minimal end-to-end smoke test: one visitor, T0=300s, 20 generations.
Verifies data loading, problem definition, and SA-MOEAD can complete.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from src.problem import TourGuideProblem
from src.moead import SA_MOEAD

BASE = os.path.join(os.path.dirname(__file__), "..")

def make_profile(kind: str) -> dict:
    templates = {
        "child":   {"id": "child",   "age": 0.0,  "edu": 0.25, "cog": 0.3, "purpose": 0.0,
                    "interest": [0.1, 0.1, 0.2, 0.1, 0.5]},
        "scholar": {"id": "scholar", "age": 0.67, "edu": 1.0,  "cog": 0.9, "purpose": 0.5,
                    "interest": [0.6, 0.1, 0.1, 0.1, 0.1]},
        "tourist": {"id": "tourist", "age": 0.67, "edu": 0.5,  "cog": 0.5, "purpose": 0.5,
                    "interest": [0.2, 0.2, 0.2, 0.2, 0.2]},
        "senior":  {"id": "senior",  "age": 1.0,  "edu": 0.5,  "cog": 0.4, "purpose": 0.0,
                    "interest": [0.1, 0.1, 0.1, 0.6, 0.1]},
    }
    return templates[kind]


def run_smoke():
    problem = TourGuideProblem.from_files(
        os.path.join(BASE, "data/segments.json"),
        os.path.join(BASE, "data/embeddings.npy"),
        T0=300, eps=5
    )
    print(f"Loaded {problem.N} segments, emb shape {problem.embs.shape}")

    profile = make_profile("tourist")

    algo = SA_MOEAD(problem, profile, pop_size=20, T_neighbor=5, max_gen=20)
    result = algo.run()

    pf = result["pareto_front"]
    sols = result["solutions"]
    print(f"\nSmoke test complete in {result['runtime_sec']:.1f}s")
    print(f"Pareto front size: {len(pf)}")
    print(f"Obj (neg): mean f1={-pf[:,0].mean():.3f}  f2={-pf[:,1].mean():.3f}  "
          f"f3={-pf[:,2].mean():.3f}")

    # Feasibility check
    n_feasible = sum(1 for s in sols if problem.is_feasible(s))
    print(f"Feasible solutions: {n_feasible}/{len(sols)}")

    # Show best solution
    best_idx = np.argmin(pf[:, 0])   # best f1
    best_seq = sols[best_idx]
    dur = problem.duration(best_seq)
    print(f"\nBest f1 solution: {len(best_seq)} segments, duration={dur}s "
          f"(target {problem.T0}±{problem.eps}s)")
    for idx in best_seq:
        seg = problem.segs[idx]
        print(f"  [{seg['id']}] {seg['category']:4s} depth={seg['depth_level']} "
              f"{seg['duration_sec']}s  {seg['text'][:30]}...")


if __name__ == "__main__":
    run_smoke()
