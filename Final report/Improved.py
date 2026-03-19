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

# Public bounds for star ratings
L, U = 1.0, 5.0


# ---------------------------
# Prompt helpers
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


def prompt_float_in_range(msg: str, low: float, high: float) -> float:
    while True:
        try:
            val = float(input(msg).strip())
            if not (low < val < high):
                print(f"Please enter a number in ({low}, {high}).")
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
    arr = s.to_numpy(dtype=float)
    if arr.size == 0:
        raise ValueError("No valid ratings found in [1,5].")
    return arr


# ---------------------------
# Improved Laplace (vectorized)
# ---------------------------

def improved_laplace_vectorized(
    x: np.ndarray,
    eps: float,
    k: float,
    p_rel: float,
    sigma: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Improved Laplace on original scale [L,U].
    x: original ratings in [L,U]
    eps: privacy budget (>0)
    k: balance parameter in (0,1)
    p_rel: relative caps p in (0,1)
    sigma: absolute std cap (>0)
    rng: numpy Generator
    Returns: privatized x' (same shape as x)
    """
    N = x.size
    R = U - L
    m = (L + U) / 2.0

    # 1) Targets and c
    n_relative = 2.0 * k * p_rel / eps
    n_absolute = 2.0 * k * R / (eps * sigma)
    n = max(n_relative, n_absolute)
    # Guard against extremely small n (numerical)
    if n <= 0:
        n = 1e-12
    c = (k * R) / n
    b = c / eps  # Laplace scale parameter

    # 2) Surrogate sensitivity S = |x - m| / (N - 1)
    denomN = max(1, N - 1)  # avoid divide-by-zero if N=1
    S = np.abs(x - m) / denomN

    # 3) Personalized sampling probability p[u=1]
    #    For S < c: 1; else: 1 - (1 - e^{-eps}) / (1 - e^{-eps*S/c})
    p1 = np.ones_like(S)
    mask = (S >= c)
    if mask.any():
        denom = 1.0 - np.exp(-eps * (S[mask] / c))
        # Guard denom -> 0 (when S ~ 0); use eps to avoid division by zero
        denom = np.where(denom <= 1e-18, 1e-18, denom)
        p1[mask] = 1.0 - (1.0 - math.exp(-eps)) / denom

        # Clip tiny negatives due to floating error, and cap at [0,1]
        p1 = np.clip(p1, 0.0, 1.0)

    # 4) Bernoulli draws and Horvitz–Thompson correction
    u = rng.random(size=N) < p1

    # Laplace noise ~ Lap(0, b)
    noise = rng.laplace(loc=0.0, scale=b, size=N)

    x_priv = np.empty_like(x, dtype=float)

    # When selected (u=1): HT correction + noise
    if u.any():
        pu = p1[u]
        # Guard p very small: (though u=1 improbable when p is tiny)
        pu = np.clip(pu, 1e-18, 1.0)
        xu = x[u]
        x_priv[u] = (xu / pu) - ((1.0 - pu) / pu) * m + noise[u]

    # When not selected (u=0): midpoint + noise
    if (~u).any():
        x_priv[~u] = m + noise[~u]

    return x_priv, {
        "R": R, "m": m, "n_relative": n_relative, "n_absolute": n_absolute,
        "n": n, "c": c, "b": b
    }


# ---------------------------
# Trials wrapper
# ---------------------------

def run_improved_laplace_trials(
    ratings: np.ndarray,
    eps: float,
    k: float,
    p_rel: float,
    sigma: float,
    n_runs: int
) -> dict:
    """
    Runs the Improved Laplace mechanism n_runs times, each with a fresh OS-random seed.
    Returns summary dict with per-run privatized means and parameters used.
    """
    if ratings.size == 0:
        raise ValueError("No ratings available.")

    est_means = []
    seeds = []
    params_ref = None

    for _ in range(n_runs):
        seed = int.from_bytes(os.urandom(8), "little", signed=False)
        rng = np.random.default_rng(seed)

        x_priv, params = improved_laplace_vectorized(
            ratings, eps=eps, k=k, p_rel=p_rel, sigma=sigma, rng=rng
        )
        if params_ref is None:
            params_ref = params
        est_means.append(float(np.mean(x_priv)))
        seeds.append(seed)

    est_means = np.array(est_means, dtype=float)
    return {
        "N": int(ratings.size),
        "epsilon": float(eps),
        "k": float(k),
        "p_rel": float(p_rel),
        "sigma": float(sigma),
        "runs": int(n_runs),
        "true_mean": float(ratings.mean()),
        "est_means": est_means,
        "est_mean_avg": float(est_means.mean()),
        "est_mean_std": float(est_means.std(ddof=1)) if n_runs > 1 else 0.0,
        "seeds": seeds,
        "params": params_ref  # shows R, m, n_relative, n_absolute, n, c, b
    }


# ---------------------------
# Main
# ---------------------------

def main():
    print("=== Improved Laplace for Amazon Ratings [1–5] ===")
    print(f"Reading ratings from: {FILE_PATH}\nColumn: {COL}  |  Sheet: {SHEET}")
    ratings = load_ratings_xlsx(FILE_PATH, sheet=SHEET, col_name=COL)

    eps = prompt_positive_float("Enter epsilon (>0): ")
    k = prompt_float_in_range("Enter k in (0,1): ", 0.0, 1.0)
    p_rel = prompt_float_in_range("Enter relative caps p in (0,1) (e.g., 0.05): ", 0.0, 1.0)
    sigma = prompt_positive_float("Enter sigma (>0) for absolute noise cap (e.g., 5): ")
    runs = prompt_int_ge1("Enter number of runs (>=1): ")

    summary = run_improved_laplace_trials(
        ratings, eps=eps, k=k, p_rel=p_rel, sigma=sigma, n_runs=runs
    )

    P = summary["params"]
    print("\n--- Parameters used ---")
    print(f" N={summary['N']}, epsilon={summary['epsilon']}, k={summary['k']}, p={summary['p_rel']}, sigma={summary['sigma']}")
    print(f" R={P['R']}, m={P['m']}, n_rel={P['n_relative']:.6g}, n_abs={P['n_absolute']:.6g}, n={P['n']:.6g}, c={P['c']:.6g}, b={P['b']:.6g}")

    print("\n--- Results ---")
    print(f" True mean (no privacy): {summary['true_mean']:.6f}")
    print(f" Average privatized mean over runs: {summary['est_mean_avg']:.6f}")
    if summary["runs"] > 1:
        print(f" Std. dev. across runs: {summary['est_mean_std']:.6f}")




if __name__ == "__main__":
    main()
