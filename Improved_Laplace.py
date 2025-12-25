from __future__ import annotations
import json
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------- Core Mechanism ----------------------------- #

def pslm_variant_b(
    x: np.ndarray,
    L: float,
    U: float,
    epsilon: float = 0.8,
    kappa: float = 0.5,
    seed: Optional[int] = 42,
) -> dict:
    """
    Run PSLM Variant B on a 1D numeric array x.

    Parameters
    ----------
    x : np.ndarray   1D array of numeric values (target column).
    L, U : float     Known bounds of the data domain (L < U).
    epsilon : float  Privacy budget (>0).
    kappa : float    Balance parameter scale in (0,1); c = kappa * (U - L) / n.
    seed : Optional[int] RNG seed for reproducibility.

    Returns
    -------
    dict Summary including true_mean, private_mean_hat, c, b, selection_rate, etc.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < kappa < 1):
        raise ValueError("kappa must be in (0, 1)")
    if not (np.isfinite(L) and np.isfinite(U)):
        raise ValueError("L and U must be finite numbers.")
    if U <= L:
        raise ValueError("U must be greater than L.")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D numeric array")
    n = x.shape[0]
    if n < 2:
        raise ValueError("n must be >= 2 for RS_tilde to be defined")

    # Midpoint, balance parameter, surrogate sensitivity
    m = (L + U) / 2.0
    c = kappa * (U - L) / n
    RS_tilde = np.abs(x - m) / (n - 1)

    # Personalized sampling probability
    # pi_i = 1 if RS_tilde_i < c else (1 - e^{-epsilon}) / (1 - e^{-epsilon * RS_tilde_i / c})
    pi = np.where(
        RS_tilde < c,
        1.0,
        (1.0 - np.exp(-epsilon)) / (1.0 - np.exp(-epsilon * RS_tilde / c))
    )

    # Bernoulli selection and Laplace noise
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, pi)
    b = c / epsilon
    noise = rng.laplace(0.0, b, size=n)
    y = np.where(s == 1, x + noise, noise)  # every user sends a message

    mu_hat = float(np.mean(y))
    mu_true = float(np.mean(x))

    return {
        "n": int(n),
        "L": float(L),
        "U": float(U),
        "midpoint_m": float(m),
        "epsilon": float(epsilon),
        "kappa": float(kappa),
        "c": float(c),
        "laplace_scale_b": float(b),
        "true_mean": mu_true,
        "private_mean_hat": mu_hat,
        "avg_pi": float(np.mean(pi)),
        "selection_rate": float(np.mean(s)),
        "min_pi": float(np.min(pi)),
        "max_pi": float(np.max(pi)),
    }

# ----------------------------- I/O Utilities ----------------------------- #

def load_pH_column_from_wine_csv(csv_path: Path) -> np.ndarray:
    """
    Load semicolon-separated CSV and extract column I ('pH').

    Parameters
    ----------
    csv_path : Path  Path to 'dataset/winequality-red.csv'

    Returns
    -------
    np.ndarray of pH values.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Semicolon-separated
    df = pd.read_csv(csv_path)

    # Column I is the 9th column (1-based), header 'pH'
    if 'pH' not in df.columns:
        # fallback by position if needed
        if df.shape[1] < 9:
            raise ValueError("CSV does not have at least 9 columns; cannot locate column I 'pH'.")
        series = df.iloc[:, 8]  # zero-based index 8
        series = pd.to_numeric(series, errors='coerce')
        col_name = df.columns[8]
        if series.isna().all():
            raise ValueError(f"Column I ('{col_name}') could not be parsed as numeric.")
    else:
        series = pd.to_numeric(df['pH'], errors='coerce')

    series = series.dropna().astype(float)
    if series.empty:
        raise ValueError("Column 'pH' has no numeric values after dropping NaNs.")
    return series.values

def prompt_float(label: str, default: Optional[float] = None) -> float:
    """Interactive float prompt with optional default."""
    while True:
        s = input(f"Enter {label}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not s and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print(f"Invalid number for {label}. Please try again.")

# ----------------------------- Main (Interactive) ----------------------------- #

def main():
    # Fixed file location and column per your requirement
    csv_path = Path("dataset/winequality-red.csv")
    x = load_pH_column_from_wine_csv(csv_path)

    print(f"Loaded {len(x)} values from column I ('pH') in {csv_path}")

    # Prompt user for L and U (you can also change defaults here)
    L = prompt_float("Lower bound L", default=None)
    U = prompt_float("Upper bound U", default=None)

    # Optional: allow quick overrides for epsilon/kappa via prompts
    epsilon = prompt_float("Privacy budget epsilon (>0) [default 0.8]", default=0.8)
    kappa = prompt_float("Balance parameter kappa in (0,1) [default 0.5]", default=0.5)

    # Optional reproducibility seed (press Enter to skip)
    s = input("Enter RNG seed for reproducibility (press Enter for default 42): ").strip()
    seed = 42 if s == "" else int(s)

    # Run PSLM Variant B
    summary = pslm_variant_b(x, L, U, epsilon=epsilon, kappa=kappa, seed=seed)

    # Pretty print results
    print("\nPSLM Variant B (Improved Laplace Mechanism) Summary")
    print("-------------------------------------------------")
    print(f"n:                 {summary['n']}")
    print(f"Bounds L,U:        {summary['L']}, {summary['U']} (midpoint m={summary['midpoint_m']:.6f})")
    print(f"epsilon, kappa:    {summary['epsilon']}, {summary['kappa']}")
    print(f"c (balance):       {summary['c']:.12f}")
    print(f"Laplace scale b:   {summary['laplace_scale_b']:.12f}")
    print(f"True mean:         {summary['true_mean']:.6f}")
    print(f"Private mean hat:  {summary['private_mean_hat']:.6f}")
    print(f"avg(pi):           {summary['avg_pi']:.6f}")
    print(f"selection rate:    {summary['selection_rate']:.6f}")
    print(f"min(pi), max(pi):  {summary['min_pi']:.6f}, {summary['max_pi']:.6f}")

    # Optional: write JSON report
    save_json = input("\nSave summary JSON? (y/N): ").strip().lower()
    if save_json == "y":
        out_path = Path("pslm_summary.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary JSON to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
