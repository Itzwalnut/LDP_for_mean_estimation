import os
import math
import numpy as np
import pandas as pd


# =========================
# Configuration (edit here)
# =========================
FILE_PATH = r".\Dataset\amazon_sales_dataset_5000dataset.xlsx"
SHEET = 0
COL = "rating (1-5)"

# HM switching threshold epsilon* (≈ 0.61 as in your report)
EPS_STAR = 0.61  # change if you refine the threshold in your paper


# ---------------------------
# Utilities: prompts
# ---------------------------

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


# ---------------------------
# Data loading
# ---------------------------

def load_ratings_xlsx(path: str, sheet=0, col_name="rating (1-5)") -> np.ndarray:
    """
    Read the ratings column, keep numeric values in [1,5], return as float array.
    """
    df = pd.read_excel(path, sheet_name=sheet, usecols=[col_name], engine="openpyxl")
    s = pd.to_numeric(df[col_name], errors="coerce").dropna()
    return s.to_numpy(dtype=float)


# ---------------------------
# Piecewise Mechanism (PM)
# Wang et al., ICDE'19, Algorithm 2
# ---------------------------

def pm_vectorized(v: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """
    Piecewise Mechanism on normalized inputs v in [-1,1].
    Returns v' with E[v'|v] = v.
    """
    e_half = math.exp(eps / 2.0)
    C = (e_half + 1.0) / (e_half - 1.0)
    prob_center = e_half / (e_half + 1.0)

    # Center interval endpoints: ℓ(v), r(v) with length (C - 1)
    l = ((C + 1.0) / 2.0) * v - (C - 1.0) / 2.0
    r = l + (C - 1.0)

    # Decide center vs outer interval per entry
    center_mask = rng.random(size=v.shape) < prob_center
    v_priv = np.empty_like(v, dtype=float)

    # Center draws
    if center_mask.any():
        v_priv[center_mask] = rng.uniform(l[center_mask], r[center_mask])

    # Outer draws
    outer_mask = ~center_mask
    if outer_mask.any():
        l_out = l[outer_mask]
        r_out = r[outer_mask]
        C_arr = np.full(l_out.shape, C)
        len_left = (l_out + C_arr)
        len_right = (C_arr - r_out)
        total = len_left + len_right

        # Guard degenerate cases (total <= 0): fall back to center draw
        deg_mask = (total <= 0)
        out_idx = np.where(outer_mask)[0]

        if deg_mask.any():
            v_priv[out_idx[deg_mask]] = rng.uniform(l_out[deg_mask], r_out[deg_mask])
            # Keep only non-degenerate for usual outer processing
            keep = ~deg_mask
            l_out, r_out = l_out[keep], r_out[keep]
            len_left, len_right = len_left[keep], len_right[keep]
            C_arr = C_arr[keep]
            out_idx = out_idx[keep]

        if out_idx.size > 0:
            prob_left = len_left / (len_left + len_right)
            flip_left = rng.random(size=out_idx.size) < prob_left
            if flip_left.any():
                v_priv[out_idx[flip_left]] = rng.uniform(-C, l_out[flip_left])
            if (~flip_left).any():
                v_priv[out_idx[~flip_left]] = rng.uniform(r_out[~flip_left], C_arr[~flip_left])

    return v_priv


# ---------------------------
# Duchi et al.'s (1-D) mechanism
# Reproduced in Wang et al., Sec. III-A
# ---------------------------

def duchi_vectorized(v: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """
    Duchi et al.'s 1-D LDP mechanism on v in [-1,1].
    Outputs ±K with E[v'|v] = v.
    """
    e = math.exp(eps)
    K = (e + 1.0) / (e - 1.0)
    p_pos = ((e - 1.0) / (e + 1.0)) * v + 0.5
    u = rng.random(size=v.shape)
    return np.where(u < p_pos, K, -K)


# ---------------------------
# Hybrid Mechanism (HM)
# If eps <= EPS_STAR, use Duchi only;
# else choose PM with prob beta = 1 - exp(-eps/2), Duchi otherwise.
# ---------------------------

def hm_vectorized(v: np.ndarray, eps: float, rng: np.random.Generator, eps_star: float = EPS_STAR) -> np.ndarray:
    if eps <= eps_star:
        return duchi_vectorized(v, eps, rng)

    beta = 1.0 - math.exp(-eps / 2.0)  # probability to use PM
    choose_pm = (rng.random(size=v.shape) < beta)

    out = np.empty_like(v, dtype=float)
    if choose_pm.any():
        out[choose_pm] = pm_vectorized(v[choose_pm], eps, rng)
    if (~choose_pm).any():
        out[~choose_pm] = duchi_vectorized(v[~choose_pm], eps, rng)
    return out


# ---------------------------
# Trials: normalize, privatize, mean
# ---------------------------

def run_hm_trials(ratings: np.ndarray, eps: float, n_runs: int, eps_star: float = EPS_STAR) -> dict:
    """
    Normalize ratings [1,5] -> [-1,1], run HM n_runs times with fresh seeds,
    map back to [1,5], and compute privatized mean each run.
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

        v_priv = hm_vectorized(v, eps, rng, eps_star=eps_star)
        x_priv = m + r * v_priv  # back to [1,5]

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
        "eps_star": float(eps_star),
        "beta_if_large_eps": float(1.0 - math.exp(-eps / 2.0)) if eps > eps_star else 0.0,
    }


# ---------------------------
# Main (interactive prompts)
# ---------------------------

def main():
    print("=== Hybrid Mechanism (HM) for Amazon Ratings [1–5] ===")
    print(f"Reading ratings from: {FILE_PATH}\nColumn: {COL}  |  Sheet: {SHEET}")

    ratings = load_ratings_xlsx(FILE_PATH, sheet=SHEET, col_name=COL)

    eps = prompt_positive_float("Enter epsilon (>0): ")
    runs = prompt_int_ge1("Enter number of runs (>=1): ")

    summary = run_hm_trials(ratings, eps=eps, n_runs=runs, eps_star=EPS_STAR)

    print(f"\nHM on ratings (N={summary['N']}), epsilon={summary['epsilon']}, runs={summary['runs']}")
    print(f"  Using eps* = {summary['eps_star']:.4f}.  "
          f"Beta (if eps>eps*): {summary['beta_if_large_eps']:.6f}")
    print(f"True mean (no privacy): {summary['true_mean']:.6f}")
    print(f"Average privatized mean over runs: {summary['est_mean_avg']:.6f}")
    if summary["runs"] > 1:
        print(f"Std. dev. across runs: {summary['est_mean_std']:.6f}")



if __name__ == "__main__":
    main()
