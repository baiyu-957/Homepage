import numpy as np
import json


# Map segment interest_tags to the 5-dim profile interest vector indices
TAG_TO_IDX = {"历史": 0, "艺术": 1, "建筑": 2, "文化": 3, "趣闻": 4}


class TourGuideProblem:
    """
    Decision variable X = (seq, theta)
      seq   : list[int], segment index sequence
      theta : ndarray shape(4,), style params [depth, complexity, metaphor, humor]
    Constraint : |sum(t[seq]) - T0| <= eps
    Objectives : maximize [f1, f2, f3]  (stored as negated minimisation values)
    """

    def __init__(self, segments, embeddings, T0=300, eps=5):
        self.segs = segments
        self.embs = embeddings          # ndarray (N, 768)
        self.T0 = T0
        self.eps = eps
        self.N = len(segments)
        self._build_tag_matrix()

    def _build_tag_matrix(self):
        """Pre-compute (N, 5) interest-tag binary matrix."""
        mat = np.zeros((self.N, 5), dtype=np.float32)
        for i, seg in enumerate(self.segs):
            for tag in seg.get("interest_tags", []):
                j = TAG_TO_IDX.get(tag)
                if j is not None:
                    mat[i, j] = 1.0
        # row-normalise so each segment sums to 1 (or stays 0 if no tags)
        norms = mat.sum(axis=1, keepdims=True).clip(min=1e-9)
        self.tag_mat = mat / norms   # shape (N, 5)

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------
    def is_feasible(self, seq) -> bool:
        if len(seq) == 0:
            return False
        total = sum(self.segs[i]["duration_sec"] for i in seq)
        return abs(total - self.T0) <= self.eps

    def duration(self, seq) -> int:
        return sum(self.segs[i]["duration_sec"] for i in seq)

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------
    def f1_interest(self, seq, profile) -> float:
        """Mean cosine similarity between segment tag vectors and visitor interest."""
        if not seq:
            return 0.0
        interest = np.array(profile["interest"], dtype=np.float32)   # (5,)
        norm = np.linalg.norm(interest)
        if norm < 1e-9:
            return 0.0
        interest_n = interest / norm
        scores = self.tag_mat[list(seq)] @ interest_n   # (L,)
        return float(scores.mean())

    def f2_cognitive(self, seq, profile) -> float:
        """Negative absolute difference between mean cognitive load and optimal load."""
        if not seq:
            return -1.0
        cl_mean = float(np.mean([self.segs[i]["cognitive_load"] for i in seq]))
        cl_opt = 0.3 + 0.5 * profile["edu"]
        return -abs(cl_mean - cl_opt)

    def f3_coherence(self, seq) -> float:
        """Mean cosine similarity of adjacent segment embeddings."""
        if len(seq) < 2:
            return 0.0
        sims = [
            float(np.dot(self.embs[seq[j]], self.embs[seq[j + 1]]))
            for j in range(len(seq) - 1)
        ]
        return float(np.mean(sims))

    def evaluate(self, seq, profile) -> np.ndarray:
        """Return shape-(3,) objective vector (all negative → MOEA/D minimises)."""
        return np.array([
            -self.f1_interest(seq, profile),
            -self.f2_cognitive(seq, profile),
            -self.f3_coherence(seq),
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def seg_score(self, idx: int, profile) -> float:
        """Quick single-segment relevance score used by operators."""
        interest = np.array(profile["interest"], dtype=np.float32)
        norm = np.linalg.norm(interest)
        if norm < 1e-9:
            return 0.0
        return float(self.tag_mat[idx] @ (interest / norm))

    @classmethod
    def from_files(cls, seg_path, emb_path, T0=300, eps=5):
        with open(seg_path, encoding="utf-8") as f:
            segs = json.load(f)
        embs = np.load(emb_path).astype(np.float32)
        return cls(segs, embs, T0=T0, eps=eps)
