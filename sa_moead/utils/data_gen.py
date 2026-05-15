import numpy as np
import json
import os


def gen_profiles(n=500, seed=42):
    np.random.seed(seed)
    profiles = []
    for i in range(n):
        a = float(np.random.choice([0.0, 0.33, 0.67, 1.0], p=[0.15, 0.25, 0.45, 0.15]))
        e = float(np.random.choice([0.25, 0.5, 0.75, 1.0], p=[0.10, 0.30, 0.40, 0.20]))
        c = float(np.random.uniform(0, 1))
        u = float(np.random.choice([0.0, 0.5, 1.0], p=[0.50, 0.30, 0.20]))
        v = np.random.dirichlet(np.ones(5))
        profiles.append({
            "id": f"p{i:04d}",
            "age": a,
            "edu": e,
            "cog": c,
            "purpose": u,
            "interest": v.tolist()
        })
    return profiles


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "../data/profiles.json")
    profiles = gen_profiles(500)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(profiles)} profiles -> {out_path}")
