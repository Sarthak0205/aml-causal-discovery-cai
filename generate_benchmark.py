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
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple
import os
import sys
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import laplace as sp_laplace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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


@dataclass(frozen=True)
class CausalEvaluationResult:
    """Comprehensive benchmark evaluation metrics comparing discovered vs ground-truth DAG."""
    has_ground_truth: bool
    tp: int
    fp: int
    fn: int
    tp_edges: Set[Tuple[str, str]]
    fp_edges: Set[Tuple[str, str]]
    fn_edges: Set[Tuple[str, str]]
    reversed_edges: Set[Tuple[str, str]]
    transitive_fp: Set[Tuple[str, str]]
    spurious_fp: Set[Tuple[str, str]]
    precision: float
    recall: float
    f1: float
    direction_accuracy: Optional[float]


def evaluate_discovered_graph(
    discovered_edges: Set[Tuple[str, str]],
    ground_truth_edges: Optional[Set[Tuple[str, str]]] = None,
) -> CausalEvaluationResult:
    """
    Evaluates discovered causal edges against ground-truth direct edges.

    Classifies false positives into:
    - Transitive / Indirect False Positives: Directed path exists in ground-truth DAG.
    - Structurally Spurious False Positives: No directed path exists in ground-truth DAG.

    Calculates:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1 Score: 2 * Precision * Recall / (Precision + Recall)
    - Direction Accuracy: TP / (TP + Reversed) among recovered true causal relationships.
    """
    if not ground_truth_edges:
        return CausalEvaluationResult(
            has_ground_truth=False,
            tp=0,
            fp=len(discovered_edges),
            fn=0,
            tp_edges=set(),
            fp_edges=set(discovered_edges),
            fn_edges=set(),
            reversed_edges=set(),
            transitive_fp=set(),
            spurious_fp=set(discovered_edges),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            direction_accuracy=None,
        )

    disc = set(discovered_edges)
    gt = set(ground_truth_edges)

    tp_edges = disc.intersection(gt)
    fp_edges = disc.difference(gt)
    fn_edges = gt.difference(disc)

    # Orientation analysis: Reversed ground-truth edges
    reversed_gt = {(v, u) for u, v in gt}
    reversed_edges = disc.intersection(reversed_gt)

    # Classify false positives: Transitive vs Spurious
    GT = nx.DiGraph()
    GT.add_edges_from(gt)
    transitive_fp = {
        (u, v) for u, v in fp_edges
        if u in GT and v in GT and nx.has_path(GT, u, v)
    }
    spurious_fp = fp_edges.difference(transitive_fp)

    tp = len(tp_edges)
    fp = len(fp_edges)
    fn = len(fn_edges)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    oriented_true_pairs = tp + len(reversed_edges)
    direction_accuracy = float(tp / oriented_true_pairs) if oriented_true_pairs > 0 else None

    return CausalEvaluationResult(
        has_ground_truth=True,
        tp=tp,
        fp=fp,
        fn=fn,
        tp_edges=tp_edges,
        fp_edges=fp_edges,
        fn_edges=fn_edges,
        reversed_edges=reversed_edges,
        transitive_fp=transitive_fp,
        spurious_fp=spurious_fp,
        precision=precision,
        recall=recall,
        f1=f1,
        direction_accuracy=direction_accuracy,
    )


def generate_synthetic_anm_benchmark(
    n_samples: int = BENCHMARK_SAMPLE_SIZE,
    random_state: int = BENCHMARK_UNIFORM_SEED,
    uniform_seed: Optional[int] = None,
    laplace_seeds: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, BenchmarkMetadata]:
    """
    Generates the canonical 5-variable nonlinear ANM benchmark dataset.

    Parameters:
        n_samples: Number of observations to generate (default: 300).
        random_state: Master random seed (default: 42 reproduces canonical benchmark).
        uniform_seed: Optional explicit seed for U1 (overrides random_state for backward compatibility).
        laplace_seeds: Optional explicit seeds for U2..U5 (overrides random_state).

    Returns:
        (df, metadata): Pandas DataFrame of shape (n_samples, 5) and associated metadata.
    """
    if random_state == BENCHMARK_UNIFORM_SEED and uniform_seed is None and laplace_seeds is None:
        effective_uniform_seed = BENCHMARK_UNIFORM_SEED
        effective_laplace_seeds = BENCHMARK_LAPLACE_SEEDS.copy()
    else:
        if uniform_seed is not None and laplace_seeds is not None:
            effective_uniform_seed = uniform_seed
            effective_laplace_seeds = laplace_seeds.copy()
        else:
            ss = np.random.SeedSequence(random_state)
            child_seeds = ss.generate_state(5)
            effective_uniform_seed = uniform_seed if uniform_seed is not None else int(child_seeds[0])
            effective_laplace_seeds = laplace_seeds if laplace_seeds is not None else {
                "U2": int(child_seeds[1]),
                "U3": int(child_seeds[2]),
                "U4": int(child_seeds[3]),
                "U5": int(child_seeds[4]),
            }

    # 1. Generate Root Exogenous Noise U1 using an isolated local RandomState
    local_rng = np.random.RandomState(effective_uniform_seed)
    U1 = local_rng.uniform(BENCHMARK_UNIFORM_LOW, BENCHMARK_UNIFORM_HIGH, size=n_samples)

    # 2. Generate Exogenous Laplace Noise U2 through U5 using explicit scipy seeds
    U2 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=effective_laplace_seeds["U2"])
    U3 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=effective_laplace_seeds["U3"])
    U4 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=effective_laplace_seeds["U4"])
    U5 = sp_laplace.rvs(scale=BENCHMARK_LAPLACE_SCALE, size=n_samples, random_state=effective_laplace_seeds["U5"])

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
        uniform_seed=effective_uniform_seed,
        laplace_seeds=effective_laplace_seeds.copy(),
        ground_truth_edges=BENCHMARK_GROUND_TRUTH_EDGES.copy(),
        description="Canonical 5-variable nonlinear Additive Noise Model (ANM) benchmark",
    )

    return df, metadata


def compute_bandwidth(X: np.ndarray, max_subsample: int = 300) -> float:
    """Computes median Euclidean distance heuristic bandwidth for RBF kernel."""
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    n_pts = len(X)
    idx = np.random.choice(n_pts, min(n_pts, max_subsample), replace=False)
    d = np.sqrt(((X[idx, None] - X[None, idx, :]) ** 2).sum(-1))
    m = np.median(d[d > 0])
    return float(m) if m > 0 else 1.0


def compute_rbf_kernel(X: np.ndarray, bw: float) -> np.ndarray:
    """Computes RBF Gram matrix using specified bandwidth."""
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    return np.exp(-((X[:, None] - X[None, :]) ** 2).sum(-1) / (2 * (bw ** 2)))


def compute_hsic(X: np.ndarray, Y: np.ndarray, max_subsample: int = 300) -> float:
    """Computes empirical Hilbert-Schmidt Independence Criterion (HSIC)."""
    bx = compute_bandwidth(X, max_subsample=max_subsample)
    by = compute_bandwidth(Y, max_subsample=max_subsample)
    n_pts = len(X)
    H = np.eye(n_pts) - (1.0 / n_pts)
    Kc = H @ compute_rbf_kernel(X, bx) @ H
    Lc = H @ compute_rbf_kernel(Y, by) @ H
    return float(np.trace(Kc @ Lc)) / (n_pts ** 2)


def compute_residuals_gp(
    cause: np.ndarray,
    effect: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Computes out-of-fold residuals of effect on cause using GP regression with KFold cross-fitting.
    Supports both 1D and multidimensional cause (conditioning set).
    """
    cause = np.asarray(cause, dtype=float)
    effect = np.asarray(effect, dtype=float)
    cause_mat = cause.reshape(-1, 1) if cause.ndim == 1 else cause
    n_pts = len(cause)
    residuals = np.empty(n_pts)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(cause_mat):
        gp = GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(),
            alpha=1e-3,
            normalize_y=True,
            n_restarts_optimizer=0,
        )
        gp.fit(cause_mat[train_idx], effect[train_idx])
        pred = gp.predict(cause_mat[test_idx])
        residuals[test_idx] = effect[test_idx] - pred
    return residuals


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Structured container for discovery pipeline execution outputs."""
    dataset_seed: Optional[int]
    pipeline_seed: int
    n_samples: int
    feature_names: List[str]
    threshold: float
    pairwise_records: List[Dict]
    edge_candidates: List[Tuple[int, int, float]]
    hsic_diffs: List[float]
    adj_initial: np.ndarray
    adj_final: np.ndarray
    retained_edges: Set[Tuple[str, str]]
    cycles_removed: int
    removed_edges_log: List[Dict]
    is_dag: bool
    evaluation: Optional[CausalEvaluationResult]


def run_discovery_pipeline(
    df: Optional[pd.DataFrame] = None,
    dataset_seed: Optional[int] = None,
    pipeline_seed: int = 42,
    n_splits: int = 5,
    percentile: float = 75.0,
    subsample_size: int = 500,
    bw_subsample_size: int = 300,
    ground_truth_edges: Optional[Set[Tuple[str, str]]] = BENCHMARK_GROUND_TRUTH_EDGES,
) -> PipelineExecutionResult:
    """
    Executes the complete ANM+HSIC causal discovery pipeline on a dataset.
    Preserves exact algorithmic parity with ANM_DAG_Discovery.ipynb.
    """
    if df is None:
        seed = dataset_seed if dataset_seed is not None else BENCHMARK_UNIFORM_SEED
        df, _ = generate_synthetic_anm_benchmark(n_samples=BENCHMARK_SAMPLE_SIZE, random_state=seed)

    # 1. Feature Preprocessing
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    data = scaler.fit_transform(df[numeric_cols].values)
    feature_names = list(numeric_cols)
    n, p = data.shape

    # 2. Algorithmic Helpers (delegating to shared implementations)
    def _bw(X, max_subsample=bw_subsample_size):
        return compute_bandwidth(X, max_subsample=max_subsample)

    def _K(X, bw):
        return compute_rbf_kernel(X, bw)

    def _hsic(X, Y):
        return compute_hsic(X, Y, max_subsample=bw_subsample_size)

    def _residuals(cause, effect):
        return compute_residuals_gp(cause, effect, n_splits=n_splits, random_state=pipeline_seed)

    # 3. Pairwise Direction Discovery
    np.random.seed(pipeline_seed)
    hsic_diffs = []
    edge_candidates = []
    pairwise_records = []

    for pair_idx, (i, j) in enumerate(combinations(range(p), 2), 1):
        idx = np.random.choice(n, min(n, subsample_size), replace=False)
        Rf = _residuals(data[idx, i], data[idx, j])
        hf = _hsic(Rf, data[idx, i])
        Rr = _residuals(data[idx, j], data[idx, i])
        hr = _hsic(Rr, data[idx, j])
        diff = abs(hf - hr)
        hsic_diffs.append(diff)

        if hf < hr:
            u_cause, v_effect = i, j
        else:
            u_cause, v_effect = j, i

        edge_candidates.append((u_cause, v_effect, diff))
        pairwise_records.append({
            "Pair_Index": pair_idx,
            "Variable_1": feature_names[i],
            "Variable_2": feature_names[j],
            "Forward_HSIC": hf,
            "Reverse_HSIC": hr,
            "Confidence_Score": diff,
            "Cause_Idx": u_cause,
            "Effect_Idx": v_effect,
        })

    # 4. Dynamic Thresholding
    threshold = float(np.percentile(hsic_diffs, percentile))
    adj_initial = np.zeros((p, p), dtype=int)
    for u, v, conf in edge_candidates:
        if conf >= threshold:
            adj_initial[u, v] = 1

    # 5. Iterative Minimum-Confidence Cycle Elimination
    edge_conf_map = {(u, v): conf for u, v, conf in edge_candidates}
    adj_final = adj_initial.copy()
    cycles_removed = 0
    removed_edges_log = []

    while True:
        G_temp = nx.DiGraph(np.argwhere(adj_final).tolist())
        try:
            cycle = nx.find_cycle(G_temp)
            cycles_removed += 1
            cyc_edges_with_conf = [((u, v), edge_conf_map.get((u, v), 0.0)) for u, v in cycle]
            min_edge, min_conf = min(cyc_edges_with_conf, key=lambda item: (item[1], item[0]))
            adj_final[min_edge[0], min_edge[1]] = 0
            removed_edges_log.append({
                "Iteration": cycles_removed,
                "Cycle": cycle,
                "Edge_Removed": min_edge,
                "Confidence": min_conf,
            })
        except nx.NetworkXNoCycle:
            break

    # 6. Graph Verification & Evaluation
    G_final = nx.DiGraph()
    for node_idx in range(p):
        G_final.add_node(node_idx)
    for u in range(p):
        for v in range(p):
            if adj_final[u, v]:
                G_final.add_edge(u, v)

    is_dag = nx.is_directed_acyclic_graph(G_final)
    retained_edges = {
        (feature_names[u], feature_names[v])
        for u in range(p) for v in range(p) if adj_final[u, v]
    }

    eval_result = evaluate_discovered_graph(retained_edges, ground_truth_edges) if ground_truth_edges else None

    return PipelineExecutionResult(
        dataset_seed=dataset_seed,
        pipeline_seed=pipeline_seed,
        n_samples=n,
        feature_names=feature_names,
        threshold=threshold,
        pairwise_records=pairwise_records,
        edge_candidates=edge_candidates,
        hsic_diffs=hsic_diffs,
        adj_initial=adj_initial,
        adj_final=adj_final,
        retained_edges=retained_edges,
        cycles_removed=cycles_removed,
        removed_edges_log=removed_edges_log,
        is_dag=is_dag,
        evaluation=eval_result,
    )


def run_percentile_sensitivity_matrix(
    seeds: Optional[List[int]] = None,
    percentiles: Optional[List[float]] = None,
    pipeline_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate ANM causal discovery performance across multiple
    dataset seeds and percentile thresholds.

    Parameters:
        seeds: List of benchmark dataset seeds (default: range(10)).
        percentiles: List of percentile thresholds to evaluate (default: [50, 60, 70, 75, 80, 90]).
        pipeline_seed: Random seed for pipeline cross-fitting and sampling (default: 42).

    Returns:
        per_run_df:
            One row per (dataset_seed, percentile) experiment.
        summary_df:
            Aggregate performance statistics grouped by percentile.
    """
    if seeds is None:
        seeds = list(range(10))
    if percentiles is None:
        percentiles = [50.0, 60.0, 70.0, 75.0, 80.0, 90.0]

    per_run_records: List[Dict] = []

    for s in seeds:
        # Pre-generate dataset once per seed to guarantee identical data across percentiles
        df, meta = generate_synthetic_anm_benchmark(
            n_samples=BENCHMARK_SAMPLE_SIZE,
            random_state=s,
        )
        gt_edges = meta.ground_truth_edges

        for pct in percentiles:
            # Execute pipeline with percentile as the only changing variable
            res = run_discovery_pipeline(
                df=df,
                dataset_seed=s,
                pipeline_seed=pipeline_seed,
                percentile=float(pct),
                ground_truth_edges=gt_edges,
            )
            ev = res.evaluation

            per_run_records.append({
                "dataset_seed": s,
                "percentile": float(pct),
                "threshold_value": res.threshold,
                "precision": ev.precision if ev else 0.0,
                "recall": ev.recall if ev else 0.0,
                "f1": ev.f1 if ev else 0.0,
                "direction_accuracy": ev.direction_accuracy if (ev and ev.direction_accuracy is not None) else 0.0,
                "true_positives": ev.tp if ev else 0,
                "false_positives": ev.fp if ev else 0,
                "false_negatives": ev.fn if ev else 0,
                "transitive_false_positives": len(ev.transitive_fp) if ev else 0,
                "spurious_false_positives": len(ev.spurious_fp) if ev else 0,
                "reversed_edges": len(ev.reversed_edges) if ev else 0,
                "retained_edges": len(res.retained_edges),
                "cycles_removed": res.cycles_removed,
                "is_dag": res.is_dag,
            })

    per_run_df = pd.DataFrame(per_run_records)

    # Compute aggregate performance statistics grouped by percentile
    summary_rows: List[Dict] = []
    for pct in percentiles:
        sub = per_run_df[per_run_df["percentile"] == float(pct)]
        summary_rows.append({
            "percentile": float(pct),
            "precision_mean": float(sub["precision"].mean()),
            "precision_std": float(sub["precision"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "recall_mean": float(sub["recall"].mean()),
            "recall_std": float(sub["recall"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "f1_mean": float(sub["f1"].mean()),
            "f1_std": float(sub["f1"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "direction_accuracy_mean": float(sub["direction_accuracy"].mean()),
            "retained_edges_mean": float(sub["retained_edges"].mean()),
            "retained_edges_std": float(sub["retained_edges"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "transitive_false_positives_mean": float(sub["transitive_false_positives"].mean()),
            "spurious_false_positives_mean": float(sub["spurious_false_positives"].mean()),
            "cycles_removed_mean": float(sub["cycles_removed"].mean()),
            "dag_validity_rate": float(sub["is_dag"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)

    return per_run_df, summary_df


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
    elif len(sys.argv) > 1 and sys.argv[1] == "--sensitivity":
        print("=" * 80)
        print("AML MINI PROJECT: MULTI-SEED PERCENTILE SENSITIVITY EXPERIMENT")
        print("Evaluating 10 seeds × 6 percentiles = 60 pipeline runs...")
        print("=" * 80)
        per_run, summary = run_percentile_sensitivity_matrix()
        print("\n--- AGGREGATE SUMMARY BY PERCENTILE ---")
        display_cols = [
            "percentile", "precision_mean", "precision_std",
            "recall_mean", "recall_std", "f1_mean", "f1_std",
            "direction_accuracy_mean", "retained_edges_mean",
            "transitive_false_positives_mean", "spurious_false_positives_mean",
            "dag_validity_rate"
        ]
        # Format floating points for clean terminal printing
        pd.set_option("display.max_columns", 15)
        pd.set_option("display.width", 120)
        print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n✅ Sensitivity evaluation completed successfully.")
    else:
        verify_benchmark_integrity()
