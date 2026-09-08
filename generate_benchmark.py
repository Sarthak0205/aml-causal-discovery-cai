"""
Canonical Benchmark Generator for Additive Noise Model (ANM) Causal Discovery.

This module is the SINGLE SOURCE OF TRUTH for the 5-variable nonlinear Additive
Noise Model synthetic benchmark dataset used in the AML causal discovery project.

Benchmark Specification:
========================
- Variables: X1, X2, X3, X4, X5
- Sample Size: 300 observations
- Ground-Truth Causal Structure (Directed Chain):
    X1 → X2 → X3 → X4 → X5
- Direct Ground-Truth Edges (4):
    {("X1", "X2"), ("X2", "X3"), ("X3", "X4"), ("X4", "X5")}

Structural Causal Model (SCM):
------------------------------
    X1 = U1
    X2 = tanh(X1) + U2
    X3 = 0.5 * (X2 ** 2) + U3
    X4 = sin(X3) + 0.3 * X3 + U4
    X5 = 0.7 * X4 - 0.2 * (X4 ** 2) + U5

Exogenous Noise Distributions & Explicit Random Seeds:
------------------------------------------------------
    U1 ~ Uniform(-2.0, 2.0),  Seed: 42 (via local np.random.RandomState - MT19937)
    U2 ~ Laplace(scale=0.4),  Seed: 1  (via scipy.stats.laplace.rvs)
    U3 ~ Laplace(scale=0.4),  Seed: 2  (via scipy.stats.laplace.rvs)
    U4 ~ Laplace(scale=0.4),  Seed: 3  (via scipy.stats.laplace.rvs)
    U5 ~ Laplace(scale=0.4),  Seed: 4  (via scipy.stats.laplace.rvs)

NOTE ON RANDOM NUMBER GENERATION:
---------------------------------
To preserve exact continuity with existing baseline experiments without mutating
global NumPy random state, U1 is generated using an isolated local instance of
np.random.RandomState(BENCHMARK_UNIFORM_SEED). This guarantees that neither global
np.random state nor unrelated notebook cells are affected.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import laplace as sp_laplace

# ==============================================================================
# Explicit Benchmark Constants
# ==============================================================================
BENCHMARK_SAMPLE_SIZE: int = 300
BENCHMARK_FEATURE_NAMES: List[str] = ["X1", "X2", "X3", "X4", "X5"]

# Exogenous Uniform Noise Configuration for Root Cause X1
BENCHMARK_UNIFORM_SEED: int = 42
BENCHMARK_UNIFORM_LOW: float = -2.0
BENCHMARK_UNIFORM_HIGH: float = 2.0

# Exogenous Laplace Noise Configuration for Variables X2 through X5
BENCHMARK_LAPLACE_SCALE: float = 0.4
BENCHMARK_LAPLACE_SEEDS: Dict[str, int] = {
    "U2": 1,
    "U3": 2,
    "U4": 3,
    "U5": 4,
}

# Ground-Truth Causal Edges
BENCHMARK_GROUND_TRUTH_EDGES: Set[Tuple[str, str]] = {
    ("X1", "X2"),
    ("X2", "X3"),
    ("X3", "X4"),
    ("X4", "X5"),
}


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Immutable metadata describing the canonical synthetic benchmark."""
    sample_size: int
    feature_names: List[str]
    uniform_seed: int
    laplace_seeds: Dict[str, int]
    ground_truth_edges: Set[Tuple[str, str]]
    description: str


def generate_synthetic_anm_benchmark(
    n_samples: int = BENCHMARK_SAMPLE_SIZE,
    uniform_seed: int = BENCHMARK_UNIFORM_SEED,
    laplace_seeds: Dict[str, int] = None,
) -> Tuple[pd.DataFrame, BenchmarkMetadata]:
    """
    Generates the canonical 5-variable nonlinear ANM benchmark dataset.

    Parameters:
        n_samples: Number of observations to generate (default: 300).
        uniform_seed: Random seed for U1 (default: 42).
        laplace_seeds: Dictionary of random seeds for U2..U5 (default: {"U2":1, ...}).

    Returns:
        (df, metadata): Pandas DataFrame of shape (n_samples, 5) and associated metadata.
    """
    if laplace_seeds is None:
        laplace_seeds = BENCHMARK_LAPLACE_SEEDS

    # 1. Generate Root Exogenous Noise U1 using an isolated local RandomState
    #    (Preserves exact MT19937 sequence without mutating global np.random state)
    local_rng = np.random.RandomState(uniform_seed)
    U1 = local_rng.uniform(BENCHMARK_UNIFORM_LOW, BENCHMARK_UNIFORM_HIGH, size=n_samples)

    # 2. Generate Exogenous Laplace Noise U2 through U5 using explicit scipy seeds
    U2 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=laplace_seeds["U2"])
    U3 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=laplace_seeds["U3"])
    U4 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=laplace_seeds["U4"])
    U5 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=laplace_seeds["U5"])

    # 3. Apply Structural Causal Equations
    X1 = U1
    X2 = np.tanh(X1) + U2
    X3 = 0.5 * (X2 ** 2) + U3
    X4 = np.sin(X3) + 0.3 * X3 + U4
    X5 = 0.7 * X4 - 0.2 * (X4 ** 2) + U5

    df = pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "X4": X4,
        "X5": X5,
    })

    metadata = BenchmarkMetadata(
        sample_size=n_samples,
        feature_names=BENCHMARK_FEATURE_NAMES.copy(),
        uniform_seed=uniform_seed,
        laplace_seeds=laplace_seeds.copy(),
        ground_truth_edges=BENCHMARK_GROUND_TRUTH_EDGES.copy(),
        description="Canonical 5-variable nonlinear Additive Noise Model (ANM) benchmark",
    )

    return df, metadata


def export_benchmark_csv(
    filepath: str = "sample_causal_data.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Exports the canonical synthetic benchmark dataset to a CSV file.
    This file is intended ONLY for testing CSV ingestion pipelines.
    """
    df, _ = generate_synthetic_anm_benchmark()
    df.to_csv(filepath, index=False)
    if verbose:
        print(f"✅ Canonical benchmark exported to: '{filepath}' ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


def verify_benchmark_integrity(csv_filepath: str = "sample_causal_data.csv") -> bool:
    """
    Verifies that:
    1. The generator is strictly deterministic across repeated calls.
    2. The generated data matches the existing sample_causal_data.csv to machine precision.
    """
    df1, _ = generate_synthetic_anm_benchmark()
    df2, _ = generate_synthetic_anm_benchmark()

    if not df1.equals(df2):
        raise AssertionError("Non-deterministic generation: consecutive calls yielded different DataFrames!")

    if os.path.exists(csv_filepath):
        df_csv = pd.read_csv(csv_filepath)
        max_diff = float((df1 - df_csv).abs().max().max())
        if not np.allclose(df1.values, df_csv.values, atol=1e-14):
            raise AssertionError(f"Divergence detected between generator and '{csv_filepath}' (max diff: {max_diff})")
        print(f"✅ Integrity verified: Generator matches '{csv_filepath}' (max diff: {max_diff:.2e} <= machine precision).")
    else:
        print(f"ℹ️ '{csv_filepath}' not found; verified generator determinism only.")

    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        target = sys.argv[2] if len(sys.argv) > 2 else "sample_causal_data.csv"
        export_benchmark_csv(target)
    else:
        verify_benchmark_integrity()
