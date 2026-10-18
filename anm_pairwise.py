"""
Standalone Pairwise Additive Noise Model (ANM) Direction Evaluator.

This module implements a reusable, scientifically rigorous, and 100% deterministic
pairwise ANM causal direction evaluator. It evaluates bivariate causal direction
between two 1D variables X and Y without access to metadata, causal labels, or ground truth.

Scientific Logic:
-----------------
Given observational vectors X and Y:
1. Standardize X and Y independently to zero mean and unit variance.
2. Forward Model: Fit Y = f(X) + R_f using 5-Fold cross-fitted Gaussian Process Regression.
   Compute dependence: h_f = HSIC(X, R_f).
3. Reverse Model: Fit X = g(Y) + R_r using 5-Fold cross-fitted Gaussian Process Regression.
   Compute dependence: h_r = HSIC(Y, R_r).
4. Decision Rule:
   - If h_f < h_r: Predict "x->y" (lower residual-input dependence).
   - If h_r <= h_f: Predict "y->x".
   - Confidence margin: Delta = |h_f - h_r|.

Determinism:
------------
All stochastic operations (controlled subsampling and bandwidth heuristic selection)
use an isolated local numpy.random.RandomState instance derived from `random_state`.
The global NumPy random generator is never mutated.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PairwiseANMResult:
    """
    Structured outcome of a pairwise ANM direction evaluation.

    Attributes:
        predicted_direction: Inferred causal arrow ("x->y" or "y->x").
        hsic_forward: Hilbert-Schmidt Independence Criterion HSIC(X, R_f).
        hsic_reverse: Hilbert-Schmidt Independence Criterion HSIC(Y, R_r).
        margin: Confidence score Delta = |hsic_forward - hsic_reverse|.
        n_samples: Effective sample size used for evaluation.
    """
    predicted_direction: str
    hsic_forward: float
    hsic_reverse: float
    margin: float
    n_samples: int


def compute_bandwidth_deterministic(
    X: np.ndarray,
    max_subsample: int = 300,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Computes median Euclidean distance heuristic bandwidth for RBF kernel deterministically.

    If n_pts <= max_subsample, computes pairwise distances across all points without subsampling.
    If n_pts > max_subsample, draws exactly max_subsample points using an isolated RandomState.
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    n_pts = len(X)
    if n_pts <= max_subsample:
        idx = np.arange(n_pts)
    else:
        if rng is None:
            rng = np.random.RandomState(42)
        idx = rng.choice(n_pts, max_subsample, replace=False)
    d = np.sqrt(((X[idx, None] - X[None, idx, :]) ** 2).sum(-1))
    m = np.median(d[d > 0])
    return float(m) if m > 0 else 1.0


def compute_rbf_kernel(X: np.ndarray, bw: float) -> np.ndarray:
    """Computes RBF Gram matrix using specified bandwidth."""
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    return np.exp(-((X[:, None] - X[None, :]) ** 2).sum(-1) / (2 * (bw ** 2)))


def compute_hsic_deterministic(
    X: np.ndarray,
    Y: np.ndarray,
    max_subsample: int = 300,
    random_state: int = 42,
) -> float:
    """
    Computes empirical Hilbert-Schmidt Independence Criterion (HSIC) using isolated RNG.
    """
    rng = np.random.RandomState(random_state)
    bx = compute_bandwidth_deterministic(X, max_subsample=max_subsample, rng=rng)
    by = compute_bandwidth_deterministic(Y, max_subsample=max_subsample, rng=rng)
    n_pts = len(X)
    H = np.eye(n_pts) - (1.0 / n_pts)
    Kc = H @ compute_rbf_kernel(X, bx) @ H
    Lc = H @ compute_rbf_kernel(Y, by) @ H
    return float(np.trace(Kc @ Lc)) / (n_pts ** 2)


def compute_residuals_gp_local(
    cause: np.ndarray,
    effect: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Computes out-of-fold residuals of effect on cause using 5-Fold cross-fitting GP regression.
    Matches the exact specification from the project benchmark.
    """
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


def evaluate_anm_direction(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: Optional[int] = None,
    random_state: int = 42,
    n_splits: int = 5,
    max_hsic_samples: int = 300,
) -> PairwiseANMResult:
    """
    Evaluates causal direction between anonymous scalar variables x and y using ANM + HSIC.

    Parameters:
        x: 1D array representing first variable.
        y: 1D array representing second variable.
        max_samples: If specified, draws exactly max_samples observations using local RNG.
                     If max_samples > len(x), raises ValueError (pairs with fewer samples must be skipped).
        random_state: Seed for local RandomState (GP cross-fitting, subsampling, bandwidth heuristic).
        n_splits: Number of cross-validation folds for GP residual cross-fitting (default 5).
        max_hsic_samples: Maximum subsample size for median heuristic bandwidth estimation (default 300).

    Returns:
        PairwiseANMResult containing predicted direction ("x->y" or "y->x"),
        forward and reverse HSIC values, confidence margin, and sample count.

    Raises:
        ValueError: If input is non-numeric, multidimensional, contains NaN/Inf, has length mismatch,
                    has zero variance, has fewer samples than n_splits, or if max_samples > len(x).
    """
    # 1. Anonymous input validation & numeric conversion
    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Inputs x and y must be convertible to numeric arrays: {err}") from err

    if x_arr.ndim != 1:
        raise ValueError(f"Expected 1D scalar array for x, got {x_arr.ndim}D shape {x_arr.shape}")
    if y_arr.ndim != 1:
        raise ValueError(f"Expected 1D scalar array for y, got {y_arr.ndim}D shape {y_arr.shape}")

    if len(x_arr) != len(y_arr):
        raise ValueError(f"Length mismatch: len(x)={len(x_arr)} != len(y)={len(y_arr)}")

    if not np.all(np.isfinite(x_arr)):
        raise ValueError("Input x contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("Input y contains non-finite values (NaN or Inf)")

    if len(x_arr) < n_splits:
        raise ValueError(f"Insufficient samples (N={len(x_arr)}) for n_splits={n_splits}")

    if np.isclose(np.std(x_arr), 0.0):
        raise ValueError("Input x has zero variance (constant array)")
    if np.isclose(np.std(y_arr), 0.0):
        raise ValueError("Input y has zero variance (constant array)")

    # 2. Controlled subsampling if max_samples is requested
    if max_samples is not None:
        if max_samples > len(x_arr):
            raise ValueError(
                f"Requested max_samples={max_samples} exceeds available sample count N={len(x_arr)}"
            )
        if max_samples < n_splits:
            raise ValueError(
                f"Requested max_samples={max_samples} is less than n_splits={n_splits}"
            )
        if max_samples < len(x_arr):
            rng = np.random.RandomState(random_state)
            sub_idx = rng.choice(len(x_arr), size=max_samples, replace=False)
            x_arr = x_arr[sub_idx]
            y_arr = y_arr[sub_idx]

    # 3. Standardization (independent per variable)
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_arr.reshape(-1, 1)).ravel()
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_arr.reshape(-1, 1)).ravel()

    assert np.isclose(np.mean(x_scaled), 0.0, atol=1e-5), "Standardization check failed: x mean is not ~0"
    assert np.isclose(np.std(x_scaled), 1.0, atol=1e-5), "Standardization check failed: x std is not ~1"
    assert np.isclose(np.mean(y_scaled), 0.0, atol=1e-5), "Standardization check failed: y mean is not ~0"
    assert np.isclose(np.std(y_scaled), 1.0, atol=1e-5), "Standardization check failed: y std is not ~1"

    # 4. Forward model: y = f(x) + R_f, evaluate HSIC(x, R_f)
    residual_forward = compute_residuals_gp_local(
        cause=x_scaled,
        effect=y_scaled,
        n_splits=n_splits,
        random_state=random_state,
    )
    forward_hsic = compute_hsic_deterministic(
        x_scaled,
        residual_forward,
        max_subsample=max_hsic_samples,
        random_state=random_state,
    )

    # 5. Reverse model: x = g(y) + R_r, evaluate HSIC(y, R_r)
    residual_reverse = compute_residuals_gp_local(
        cause=y_scaled,
        effect=x_scaled,
        n_splits=n_splits,
        random_state=random_state,
    )
    reverse_hsic = compute_hsic_deterministic(
        y_scaled,
        residual_reverse,
        max_subsample=max_hsic_samples,
        random_state=random_state,
    )

    # 6. Directional decision and confidence margin
    margin = abs(forward_hsic - reverse_hsic)
    predicted_direction = "x->y" if forward_hsic < reverse_hsic else "y->x"

    return PairwiseANMResult(
        predicted_direction=predicted_direction,
        hsic_forward=forward_hsic,
        hsic_reverse=reverse_hsic,
        margin=margin,
        n_samples=len(x_scaled),
    )
