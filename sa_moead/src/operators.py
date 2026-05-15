import numpy as np
import random
from typing import List


# -----------------------------------------------------------------------
# ECR — Equality Constraint Repair
# -----------------------------------------------------------------------

def ecr_repair(seq: List[int], problem, max_iter: int = 50) -> List[int]:
    """
    Greedy repair so |sum(duration[seq]) - T0| <= eps.

    Returns a (possibly modified) list of segment indices.
    """
    seq = list(seq)
    T0, eps = problem.T0, problem.eps

    for _ in range(max_iter):
        delta = problem.duration(seq) - T0
        if abs(delta) <= eps:
            break

        if delta > 0:
            # Over-time: try to swap a segment for a shorter one, or remove
            best_swap = None
            best_val = abs(delta)
            for pos, seg_id in enumerate(seq):
                t_i = problem.segs[seg_id]["duration_sec"]
                # look for replacement with smaller duration
                for k in range(problem.N):
                    if k in seq:
                        continue
                    t_k = problem.segs[k]["duration_sec"]
                    new_delta = abs(delta - (t_i - t_k))
                    if new_delta < best_val:
                        best_val = new_delta
                        best_swap = (pos, k)
                        if best_val <= eps:
                            break
                if best_swap and best_val <= eps:
                    break

            if best_swap:
                pos, k = best_swap
                seq[pos] = k
            else:
                # Remove segment closest to |delta|
                diffs = [abs(problem.segs[seq[j]]["duration_sec"] - delta)
                         for j in range(len(seq))]
                remove_pos = int(np.argmin(diffs))
                seq.pop(remove_pos)

        else:
            # Under-time: insert a segment whose duration is close to |delta|
            need = -delta
            candidates = [k for k in range(problem.N) if k not in seq]
            if not candidates:
                break
            diffs = [abs(problem.segs[k]["duration_sec"] - need)
                     for k in candidates]
            best_k = candidates[int(np.argmin(diffs))]
            # Insert at position that maximises f3 coherence
            best_pos = _best_insert_pos(seq, best_k, problem)
            seq.insert(best_pos, best_k)

    return seq


def _best_insert_pos(seq: List[int], k: int, problem) -> int:
    """Return insertion position for segment k that maximises f3 coherence."""
    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        sim_before = float(np.dot(problem.embs[k], problem.embs[seq[0]]))
        sim_after = float(np.dot(problem.embs[seq[0]], problem.embs[k]))
        return 0 if sim_before >= sim_after else 1

    best_score, best_pos = -np.inf, 0
    ek = problem.embs[k]
    for pos in range(len(seq) + 1):
        score = 0.0
        if pos > 0:
            score += float(np.dot(problem.embs[seq[pos - 1]], ek))
        if pos < len(seq):
            score += float(np.dot(ek, problem.embs[seq[pos]]))
        if score > best_score:
            best_score, best_pos = score, pos
    return best_pos


# -----------------------------------------------------------------------
# SAX — Semantics-Aware Crossover
# -----------------------------------------------------------------------

def sax_crossover(parent1_seq: List[int], parent2_seq: List[int],
                  profile, problem,
                  high_prob: float = 0.8, low_prob: float = 0.3) -> List[int]:
    """
    Merge both parents, score each segment by f1+f2, and greedily build a
    child up to duration T0, then repair with ECR.
    """
    all_segs = list(set(parent1_seq) | set(parent2_seq))

    # Score each candidate
    scores = {}
    for idx in all_segs:
        s1 = problem.seg_score(idx, profile)
        cl_opt = 0.3 + 0.5 * profile["edu"]
        s2 = -abs(problem.segs[idx]["cognitive_load"] - cl_opt)
        scores[idx] = s1 + s2

    # Sort by score descending
    all_segs.sort(key=lambda x: scores[x], reverse=True)

    child = []
    total_dur = 0
    used = set()
    for idx in all_segs:
        prob = high_prob if scores[idx] >= 0 else low_prob
        if random.random() < prob and idx not in used:
            dur = problem.segs[idx]["duration_sec"]
            if total_dur + dur <= problem.T0 + problem.eps * 3:
                child.append(idx)
                used.add(idx)
                total_dur += dur
            if total_dur >= problem.T0 - problem.eps:
                break

    if len(child) == 0:
        child = list(parent1_seq)

    return ecr_repair(child, problem)


# -----------------------------------------------------------------------
# SNM — Semantic Neighbourhood Mutation
# -----------------------------------------------------------------------

def snm_mutation(seq: List[int], profile, problem,
                 tau_sem: float = 0.7, mut_rate: float = 0.2) -> List[int]:
    """
    For each position (with probability mut_rate) find a semantically similar
    segment with higher f1 and swap.  Repair afterwards.
    """
    seq = list(seq)
    for pos in range(len(seq)):
        if random.random() > mut_rate:
            continue

        seg_id = seq[pos]
        e_j = problem.embs[seg_id]
        f1_j = problem.seg_score(seg_id, profile)

        # Build semantic neighbourhood: cos-sim >= tau_sem AND f1 > f1_j
        candidates = []
        for k in range(problem.N):
            if k in seq:
                continue
            sim = float(np.dot(problem.embs[k], e_j))
            if sim >= tau_sem:
                f1_k = problem.seg_score(k, profile)
                if f1_k > f1_j:
                    candidates.append((k, f1_k))

        if not candidates:
            continue

        # Weighted random sample by f1 score
        ks, weights = zip(*candidates)
        weights = np.array(weights, dtype=np.float64)
        weights = weights - weights.min() + 1e-9
        weights /= weights.sum()
        chosen = int(np.random.choice(list(ks), p=weights))
        seq[pos] = chosen

    return ecr_repair(seq, problem)
