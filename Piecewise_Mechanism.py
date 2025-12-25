
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# ---------------- PM core ---------------- #

def _piecewise_C(epsilon: float) -> float:
    e_half = np.exp(epsilon / 2.0)
    return (e_half + 1.0) / (e_half - 1.0)

def _pm_interval_bounds(t: float, C: float) -> tuple[float, float]:
    # Center interval [ℓ(t), r(t)] consistent with Fig. 2
    l = ((C + 1.0) / 2.0) * t - (C - 1.0) / 2.0
    r = ((C + 1.0) / 2.0) * t + (C - 1.0) / 2.0
    l = float(np.clip(l, -C, C))
    r = float(np.clip(r, -C, C))
    if l > r:
        l, r = r, l
    return l, r

def pm_scalar(x: float, epsilon: float, L: float, U: float, rng: Optional[np.random.Generator] = None) -> float:
    """Apply PM to one value x ∈ [L,U]."""
    if rng is None:
        rng = np.random.default_rng()
    if U <= L:
        raise ValueError("U must be greater than L")
    m = (L + U) / 2.0
    r = (U - L) / 2.0
    if r == 0:
        return float(m)
    # normalize to [-1,1]
    t = float(np.clip((x - m) / r, -1.0, 1.0))
    C = _piecewise_C(epsilon)
    e_half = np.exp(epsilon / 2.0)
    p_center = e_half / (e_half + 1.0)
    l, rr = _pm_interval_bounds(t, C)
    u = rng.random()
    if u < p_center:
        noisy_t = rng.uniform(l, rr)
    else:
        left_len = (l - (-C))
        right_len = (C - rr)
        total_len = left_len + right_len
        if total_len <= 1e-12:
            noisy_t = rng.uniform(l, rr)
        else:
            u2 = rng.random() * total_len
            if u2 <= left_len:
                noisy_t = rng.uniform(-C, l)
            else:
                noisy_t = rng.uniform(rr, C)
    # map back to [L,U]
    return float(m + r * noisy_t)

# -------------- I/O + CLI -------------- #

def load_column(csv_path: Path, column: str = 'pH') -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
    s = pd.to_numeric(df[column], errors='coerce').dropna().astype(float)
    if s.empty:
        raise ValueError(f"Column '{column}' has no numeric values after dropping NaNs.")
    return s.values


def prompt_float(label: str, default: Optional[float] = None) -> float:
    while True:
        s = input(f"Enter {label}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print(f"Invalid number for {label}. Try again.")


def run(values: np.ndarray, epsilon: float, L: float, U: float, seed: Optional[int]) -> dict:
    rng = np.random.default_rng(seed)
    true_mean = float(np.mean(values))
    noisy_vals = np.array([pm_scalar(x, epsilon, L, U, rng=rng) for x in values], dtype=float)
    private_mean_hat = float(np.mean(noisy_vals))
    return {
        "n": int(values.size),
        "epsilon": float(epsilon),
        "bounds": {"L": float(L), "U": float(U)},
        "true_mean": true_mean,
        "private_mean_hat": private_mean_hat,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Piecewise Mechanism (PM) under LDP for a single CSV column.")
    p.add_argument('--csv', type=str, default='dataset/winequality-red.csv', help='Path to CSV (default dataset/winequality-red.csv).')
    p.add_argument('--column', type=str, default='pH', help="Column name (default 'pH').")
    p.add_argument('--sep', type=str, default=';', help="CSV separator (default ';').")
    p.add_argument('--L', type=float, default=None, help='Lower bound L (prompted if missing).')
    p.add_argument('--U', type=float, default=None, help='Upper bound U (prompted if missing).')
    p.add_argument('-e', '--epsilon', type=float, default=None, help='Privacy budget epsilon (>0). Prompted if missing.')
    p.add_argument('--seed', type=int, default=42, help='RNG seed (default 42).')
    p.add_argument('--json-out', type=str, default=None, help='Optional path to save summary JSON.')
    return p


def main():
    args = build_argparser().parse_args()
    values = load_column(Path(args.csv), column=args.column)
    print(f"Loaded {values.size} values from {args.csv} column '{args.column}'")
    L = args.L if args.L is not None else prompt_float('Lower bound L')
    U = args.U if args.U is not None else prompt_float('Upper bound U')
    epsilon = args.epsilon if args.epsilon is not None else prompt_float('Privacy budget epsilon (>0)', default=0.8)
    if epsilon <= 0:
        raise ValueError('epsilon must be > 0')
    seed = args.seed

    summary = run(values, epsilon, L, U, seed)
    print("\nPM Summary")
    print("----------")
    print(f"n:                 {summary['n']}")
    print(f"Bounds L,U:        {summary['bounds']['L']}, {summary['bounds']['U']}")
    print(f"epsilon:           {summary['epsilon']}")
    print(f"True mean:         {summary['true_mean']:.6f}")
    print(f"Private mean hat:  {summary['private_mean_hat']:.6f}")

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary JSON to: {args.json_out}")

if __name__ == '__main__':
    main()
