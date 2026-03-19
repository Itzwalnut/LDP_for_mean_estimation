import math
import os
import numpy as np
import pandas as pd


# =========================
# Configuration (edit here)
# =========================
FILE_PATH = r".\Dataset\amazon_sales_dataset_5000dataset.xlsx"
SHEET = 0
COL = "rating (1-5)"


# ---------------------------
# Piecewise Mechanism (PM)
# Wang et al., ICDE'19 (Alg. 2).  # Ref: https://arxiv.org/pdf/1907.00782
# ---------------------------

def pm_vectorized(v: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply the 1-D Piecewise Mechanism to all normalized inputs v in [-1, 1].
    Returns v' in [-C, C] with E[v'|v] = v.
    """
    e_half = math.exp(eps / 2.0)
    C = (e_half + 1.0) / (e_half - 1.0)
    prob_center = e_half / (e_half + 1.0)

    # Center interval [ℓ(v), r(v)] with length (C - 1)
    l = ((C + 1.0) / 2.0) * v - (C - 1.0) / 2.0
    r = l + (C - 1.0)

    # Center vs outer selection
    center_mask = rng.random(size=v.shape) < prob_center
    vp = np.empty_like(v, dtype=float)

    # Center draws
    if center_mask.any():
        vp[center_mask] = rng.uniform(l[center_mask], r[center_mask])

    # Outer draws
    outer_mask = ~center_mask
    if outer_mask.any():
        l_out = l[outer_mask]
        r_out = r[outer_mask]
        C_arr = np.full(l_out.shape, C)
        len_left = (l_out + C_arr)
        len_right = (C_arr - r_out)
        total = len_left + len_right

        # Degenerate guard: if total<=0, fall back to center draw
        deg_mask = (total <= 0)
        out_idx = np.where(outer_mask)[0]

        if deg_mask.any():
            vp[out_idx[deg_mask]] = rng.uniform(l_out[deg_mask], r_out[deg_mask])
            # Remove degenerate from the set
            keep = ~deg_mask
            l_out, r_out = l_out[keep], r_out[keep]
            len_left, len_right = len_left[keep], len_right[keep]
            C_arr = C_arr[keep]
            out_idx = out_idx[keep]

        if out_idx.size > 0:
            prob_left = len_left / (len_left + len_right)
            flip = rng.random(size=out_idx.size) < prob_left
            if flip.any():
                vp[out_idx[flip]] = rng.uniform(-C, l_out[flip])
            if (~flip).any():
                vp[out_idx[~flip]] = rng.uniform(r_out[~flip], C_arr[~flip])

    return vp


# ---------------------------
# Data loading and pipeline
# ---------------------------

def load_ratings_xlsx(path: str, sheet=0, col_name="rating (1-5)") -> np.ndarray:
    """Read the ratings column, keep numeric values in [1,5], return as float array."""
    df = pd.read_excel(path, sheet_name=sheet, usecols=[col_name], engine="openpyxl")
    s = pd.to_numeric(df[col_name], errors="coerce").dropna()
    s = s[s.between(1, 5)]
    return s.to_numpy(dtype=float)


def run_pm_trials(ratings: np.ndarray, eps: float, n_runs: int) -> dict:
    """
    Normalize ratings in [1,5] to v in [-1,1], run PM n_runs times with fresh seeds,
    de-normalize outputs, and compute privatized mean each run.
    """
    if ratings.size == 0:
        raise ValueError("No ratings found after cleaning.")

    # Normalize to [-1,1]
    L, U = 1.0, 5.0
    m, r = (L + U) / 2.0, (U - L) / 2.0  # m=3, r=2
    v = (ratings - m) / r

    est_means = []
    seeds = []

    for _ in range(n_runs):
        seed = int.from_bytes(os.urandom(8), "little", signed=False)
        rng = np.random.default_rng(seed)
        v_priv = pm_vectorized(v, eps, rng)
        x_priv = m + r * v_priv
        est_means.append(float(np.mean(x_priv)))
        seeds.append(seed)

    est_means = np.array(est_means, dtype=float)
    return {
        "N": int(ratings.size),
        "epsilon": float(eps),
        "runs": int(n_runs),
        "true_mean": float(ratings.mean()),
        "est_means": est_means,
        "est_mean_avg": float(est_means.mean()),
        "est_mean_std": float(est_means.std(ddof=1)) if n_runs > 1 else 0.0,
        "seeds": seeds,
    }


def prompt_positive_float(msg: str) -> float:
    while True:
        try:
            val = float(input(msg).strip())
            if val <= 0:
                print("Please enter a number > 0.")
                continue
            return val
        except ValueError:
            print("Invalid number, try again.")


def prompt_int_ge1(msg: str) -> int:
    while True:
        try:
            val = int(input(msg).strip())
            if val < 1:
                print("Please enter an integer >= 1.")
                continue
            return val
        except ValueError:
            print("Invalid integer, try again.")


def main():
    print("=== Piecewise Mechanism (PM) for Amazon Ratings [1–5] ===")
    print(f"Reading ratings from: {FILE_PATH}\nColumn: {COL}  |  Sheet: {SHEET}")
    ratings = load_ratings_xlsx(FILE_PATH, sheet=SHEET, col_name=COL)

    eps = prompt_positive_float("Enter epsilon (>0): ")
    runs = prompt_int_ge1("Enter number of runs (>=1): ")

    summary = run_pm_trials(ratings, eps=eps, n_runs=runs)

    print(f"\nPM on ratings (N={summary['N']}), epsilon={summary['epsilon']}, runs={summary['runs']}")
    print(f"True mean (no privacy): {summary['true_mean']:.6f}")
    print(f"Average privatized mean over runs: {summary['est_mean_avg']:.6f}")
    if summary["runs"] > 1:
        print(f"Std. dev. across runs: {summary['est_mean_std']:.6f}")



if __name__ == "__main__":
    main()
