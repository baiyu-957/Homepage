import numpy as np
import random
import time
from typing import List, Dict, Any

from .operators import sax_crossover, snm_mutation, ecr_repair


# -----------------------------------------------------------------------
# Weight vector generation (Das-Dennis simplex lattice)
# -----------------------------------------------------------------------

def _das_dennis(n_obj: int, n_partitions: int) -> np.ndarray:
    """Generate uniformly distributed weight vectors on the unit simplex."""
    from itertools import combinations_with_replacement

    def _recursive(n, left, prefix):
        if n == 1:
            yield prefix + [left]
            return
        for i in range(left + 1):
            yield from _recursive(n - 1, left - i, prefix + [i])

    weights = []
    for combo in _recursive(n_obj, n_partitions, []):
        weights.append([c / n_partitions for c in combo])
    return np.array(weights, dtype=np.float64)


def _tchebycheff(obj_vec: np.ndarray, weight: np.ndarray, z_star: np.ndarray) -> float:
    """Tchebycheff scalarisation (minimisation)."""
    return float(np.max(weight * np.abs(obj_vec - z_star)))


# -----------------------------------------------------------------------
# SA-MOEAD
# -----------------------------------------------------------------------

class SA_MOEAD:
    """
    Semantics-Aware MOEA/D.
    Integrates SAX crossover, SNM mutation, and ECR repair into MOEA/D.
    """

    def __init__(self, problem, profile: Dict,
                 pop_size: int = 100,
                 T_neighbor: int = 20,
                 max_gen: int = 200,
                 tau_sem: float = 0.7,
                 mut_rate: float = 0.2):
        self.problem = problem
        self.profile = profile
        self.pop_size = pop_size
        self.T_neighbor = T_neighbor
        self.max_gen = max_gen
        self.tau_sem = tau_sem
        self.mut_rate = mut_rate
        self.n_obj = 3

    # ------------------------------------------------------------------
    def _init_weights(self) -> np.ndarray:
        H = self.pop_size - 1
        W = _das_dennis(self.n_obj, H)
        # If exact size differs, subsample or pad
        if len(W) > self.pop_size:
            idx = np.random.choice(len(W), self.pop_size, replace=False)
            W = W[idx]
        elif len(W) < self.pop_size:
            extra = self.pop_size - len(W)
            extra_w = np.random.dirichlet(np.ones(self.n_obj), size=extra)
            W = np.vstack([W, extra_w])
        return W

    def _profile_to_bias(self) -> np.ndarray:
        """
        Return a segment-score array biased by visitor profile.
        Used to bias initial population construction.
        """
        scores = np.array([
            self.problem.seg_score(i, self.profile)
            for i in range(self.problem.N)
        ], dtype=np.float64)
        return scores

    def _init_population(self, bias_scores: np.ndarray) -> List[List[int]]:
        """
        Construct initial population.  Each individual is a random sequence
        built by sampling segments weighted by bias_scores, then ECR-repaired.
        """
        pop = []
        probs = np.clip(bias_scores, 0, None) + 1e-3
        probs /= probs.sum()
        for _ in range(self.pop_size):
            # Sample without replacement until close to T0
            chosen = []
            total = 0
            perm = np.random.choice(self.problem.N, size=self.problem.N,
                                    replace=False, p=probs)
            for idx in perm:
                dur = self.problem.segs[int(idx)]["duration_sec"]
                if total + dur <= self.problem.T0 + self.problem.eps * 3:
                    chosen.append(int(idx))
                    total += dur
                if total >= self.problem.T0 - self.problem.eps:
                    break
            seq = ecr_repair(chosen, self.problem)
            pop.append(seq)
        return pop

    def _compute_neighbors(self, W: np.ndarray) -> List[List[int]]:
        """T_neighbor nearest weight vectors for each sub-problem."""
        dists = np.sum((W[:, None, :] - W[None, :, :]) ** 2, axis=2)
        neighbors = [
            list(np.argsort(dists[i])[:self.T_neighbor])
            for i in range(self.pop_size)
        ]
        return neighbors

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        t_start = time.time()
        W = self._init_weights()
        neighbors = self._compute_neighbors(W)
        bias = self._profile_to_bias()
        pop = self._init_population(bias)

        # Evaluate initial population
        obj_pop = np.array([
            self.problem.evaluate(seq, self.profile) for seq in pop
        ])                                          # (pop_size, 3)

        # Reference point z* (ideal point, minimisation so take min)
        z_star = obj_pop.min(axis=0).copy()

        history_hv: List[float] = []

        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                nb = neighbors[i]
                # Select two parents from neighbourhood
                p1_idx, p2_idx = random.sample(nb, 2)
                p1_seq = pop[p1_idx]
                p2_seq = pop[p2_idx]

                # SAX crossover
                child_seq = sax_crossover(p1_seq, p2_seq, self.profile,
                                          self.problem)
                # SNM mutation
                child_seq = snm_mutation(child_seq, self.profile, self.problem,
                                         tau_sem=self.tau_sem,
                                         mut_rate=self.mut_rate)

                child_obj = self.problem.evaluate(child_seq, self.profile)

                # Update reference point
                z_star = np.minimum(z_star, child_obj)

                # Update neighbourhood solutions
                for j in nb:
                    if (_tchebycheff(child_obj, W[j], z_star) <=
                            _tchebycheff(obj_pop[j], W[j], z_star)):
                        pop[j] = child_seq
                        obj_pop[j] = child_obj

            # Record approximate HV every 10 gens
            if gen % 10 == 0:
                hv = self._approx_hv(obj_pop)
                history_hv.append((gen, hv))

        # Collect Pareto-non-dominated solutions
        pareto_idx = _non_dominated(obj_pop)
        pareto_front = obj_pop[pareto_idx]
        solutions = [pop[i] for i in pareto_idx]

        return {
            "pareto_front": pareto_front,   # (M, 3) negated objectives
            "solutions": solutions,
            "history_hv": history_hv,
            "runtime_sec": time.time() - t_start,
        }

    def _approx_hv(self, obj_pop: np.ndarray) -> float:
        """Approximate hypervolume relative to nadir + margin reference point.

        f2 objective = abs(cl_mean - cl_opt) is in [0, ~0.8], not <= 0,
        so a fixed [0.1, 0.1, 0.1] ref incorrectly filters most solutions.
        """
        ref = obj_pop.max(axis=0) + np.array([0.05, 0.1, 0.05])
        try:
            from pymoo.indicators.hv import HV
            ind = HV(ref_point=ref)
            return float(ind(obj_pop))
        except Exception:
            return float(np.sum(np.prod(ref - obj_pop, axis=1).clip(0)))


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _non_dominated(obj_pop: np.ndarray) -> List[int]:
    """Return indices of non-dominated solutions (minimisation)."""
    n = len(obj_pop)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # j dominates i if obj[j] <= obj[i] in all and < in at least one
            if np.all(obj_pop[j] <= obj_pop[i]) and np.any(obj_pop[j] < obj_pop[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]
