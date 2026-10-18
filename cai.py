"""
Causal Feature Attribution (CAI) Core Module.

This module implements structural equation modeling and interventional causal feature
attribution for machine learning models based on learned Additive Noise Models (ANM):

    X_j = g_j(Pa(j)) + U_j

Components:
-----------
- _StructEq: Gaussian Process structural equation for a single variable.
- CausalANM: Structural causal model fitted on a discovered Directed Acyclic Graph (DAG).
- CAIResult: Structured container holding causal feature attributions and statistics.
- CausalFeatureAttributor: Computes interventional attributions:
      CAI_j = E[f(X) | do(X_j = x_j)] - E[f(X)]
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.stats import laplace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


class _StructEq:
    """
    Gaussian Process structural equation for a single causal variable:

        X_j = g_j(Pa(j)) + U_j

    Parameters
    ----------
    j : int
        Index of the target causal variable.
    parents : Sequence[int]
        Indices of the parent variables in the causal DAG.
    random_state : int, default=42
        Random seed for GP training subsampling and optimizer.
    """

    def __init__(
        self,
        j: int,
        parents: Sequence[int],
        random_state: int = 42,
    ) -> None:
        self.j = int(j)
        self.parents = list(parents)
        self.is_root = len(self.parents) == 0
        self.gp: Optional[GaussianProcessRegressor] = None
        self.sc = StandardScaler()
        self.random_state = int(random_state)

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the structural equation on observational data.

        Returns
        -------
        residuals : np.ndarray
            Residual noise vector U_j = X_j - g_j(Pa(j)) (or X_j directly if root).
        """
        Xj = data[:, self.j]

        if self.is_root:
            return Xj

        Pa = data[:, self.parents]
        Ps = self.sc.fit_transform(Pa)

        gp = GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(),
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=self.random_state,
        )

        n = len(Ps)
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(
            n,
            min(n, 400),
            replace=False,
        )

        gp.fit(Ps[idx], Xj[idx])
        self.gp = gp

        return Xj - gp.predict(Ps)

    def predict(self, pa: np.ndarray) -> np.ndarray:
        """
        Predict conditional mean g_j(Pa(j)).
        """
        if self.is_root:
            return np.zeros(
                pa.shape[0] if pa.ndim > 1 else 1,
                dtype=float,
            )

        pa = pa.reshape(1, -1) if pa.ndim == 1 else pa
        return self.gp.predict(self.sc.transform(pa))


class CausalANM:
    """
    Additive Noise Model (ANM) structural model fitted on a discovered DAG.

    Each variable is modeled as:
        X_j = g_j(Pa(j)) + U_j

    where g_j is fitted using Gaussian Process Regression and U_j is modeled
    using a fitted Laplace noise distribution.

    Parameters
    ----------
    adj : np.ndarray
        (p, p) binary adjacency matrix where adj[u, v] != 0 denotes directed edge u -> v.
    order : Sequence[int]
        Topological ordering of nodes (length p).
    random_state : int, default=42
        Default random seed for reproducibility.
    """

    def __init__(
        self,
        adj: np.ndarray,
        order: Sequence[int],
        random_state: int = 42,
    ) -> None:
        self.adj = np.asarray(adj, dtype=int)
        self.order = [int(x) for x in order]
        self.p = self.adj.shape[0]
        self.random_state = int(random_state)

        self.eqs: List[Optional[_StructEq]] = []
        self.noise: List[Optional[Tuple[float, float]]] = []

    def fit(self, data: np.ndarray) -> "CausalANM":
        """
        Fit structural equations and Laplace noise distributions for all variables.
        """
        self.eqs = [None] * self.p
        self.noise = [None] * self.p

        for j in self.order:
            parents = np.where(self.adj[:, j] != 0)[0].tolist()

            eq = _StructEq(j, parents, random_state=self.random_state)
            residuals = eq.fit(data)

            self.eqs[j] = eq

            if eq.is_root:
                loc, scale = laplace.fit(data[:, j])
            else:
                loc, scale = laplace.fit(residuals)

            self.noise[j] = (float(loc), float(scale))

        return self

    def do_sample(
        self,
        fixed: Dict[int, float],
        K: int = 100,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate samples from the fitted structural model under interventions do(X_j = value).

        Parameters
        ----------
        fixed : dict
            Mapping {variable_index: fixed_value} specifying interventions.
        K : int, default=100
            Number of Monte Carlo samples to draw.
        random_state : int, optional
            Random state seed for the Laplace sampling. If None, uses self.random_state.

        Returns
        -------
        samples : np.ndarray
            (K, p) array of simulated interventional samples.
        """
        fixed_clean = {int(j): float(value) for j, value in fixed.items()}
        intervention_nodes = set(fixed_clean.keys())

        samples = np.zeros((K, self.p), dtype=float)

        seed = self.random_state if random_state is None else int(random_state)
        rng = np.random.RandomState(seed)

        for j in self.order:
            if j in intervention_nodes:
                samples[:, j] = fixed_clean[j]
                continue

            eq = self.eqs[j]
            loc, scale = self.noise[j]

            if eq.is_root:
                mean_prediction = np.zeros(K, dtype=float)
            else:
                mean_prediction = eq.predict(samples[:, eq.parents])

            samples[:, j] = mean_prediction + laplace.rvs(
                loc=loc,
                scale=scale,
                size=K,
                random_state=rng,
            )

        return samples


@dataclass
class CAIResult:
    """
    Stores causal feature attribution results.

    Attributes
    ----------
    x_obs : np.ndarray
        Observed feature vector of shape (p,).
    cai : np.ndarray
        Raw causal attributions (f_do - baseline) of shape (p,).
    cai_norm : np.ndarray
        L1-normalized causal attributions of shape (p,).
    baseline : float
        Expected prediction under observational distribution E[f(X)].
    f_do : np.ndarray
        Expected prediction under each intervention E[f(X) | do(X_j = x_j)].
    feature_names : list
        List of feature names corresponding to variables 0..p-1.
    standard_errors : np.ndarray
        Monte Carlo standard error for each CAI estimate.
    confidence_intervals : np.ndarray
        95% confidence intervals of shape (p, 2) [lower, upper].
    intervention_values : list
        List of Monte Carlo prediction arrays under each intervention.
    n_samples : int
        Number of Monte Carlo samples used.
    """

    x_obs: np.ndarray
    cai: np.ndarray
    cai_norm: np.ndarray
    baseline: float
    f_do: np.ndarray
    feature_names: list
    standard_errors: np.ndarray
    confidence_intervals: np.ndarray
    intervention_values: list
    n_samples: int


class CausalFeatureAttributor:
    """
    Causal feature attribution using interventions on a fitted ANM structural model.

        CAI_j = E[f(X) | do(X_j = x_j)] - E[f(X)]

    where f is the predictive model function being explained.

    Parameters
    ----------
    anm : CausalANM
        Fitted CausalANM instance.
    prediction_function : Callable[[np.ndarray], np.ndarray]
        Callable mapping (N, p) samples to a 1D array of N scalar predictions.
    K_cai : int, default=100
        Number of Monte Carlo samples per intervention.
    """

    def __init__(
        self,
        anm: CausalANM,
        prediction_function: Callable[[np.ndarray], np.ndarray],
        K_cai: int = 100,
    ) -> None:
        self.anm = anm
        self.prediction_function = prediction_function
        self.K_cai = int(K_cai)

    def attribute(
        self,
        x_obs: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> CAIResult:
        """
        Compute causal feature attributions for a given observation x_obs.

        Parameters
        ----------
        x_obs : np.ndarray
            1D array of observed feature values with shape (p,).
        feature_names : Sequence[str], optional
            Names of the features. Defaults to ["Feature_0", ...].
        verbose : bool, default=True
            Whether to print progress.

        Returns
        -------
        result : CAIResult
            Structured attribution result.
        """
        x_obs = np.asarray(x_obs, dtype=float)

        if x_obs.shape != (self.anm.p,):
            raise ValueError(
                f"x_obs must have shape ({self.anm.p},), got {x_obs.shape}."
            )

        if feature_names is None:
            feature_names = [f"Feature_{j}" for j in range(self.anm.p)]

        if len(feature_names) != self.anm.p:
            raise ValueError(
                "feature_names length must match the number of causal variables."
            )

        def predict_samples(samples: np.ndarray) -> np.ndarray:
            predictions = self.prediction_function(samples)
            predictions = np.asarray(predictions, dtype=float)

            if predictions.ndim != 1:
                raise ValueError(
                    "prediction_function must return one scalar prediction per sample."
                )

            if len(predictions) != len(samples):
                raise ValueError(
                    "prediction_function returned an unexpected number of predictions."
                )

            return predictions

        # ----------------------------------------------------
        # Baseline: observational model distribution
        # ----------------------------------------------------
        baseline_samples = self.anm.do_sample(
            {},
            K=self.K_cai,
        )

        baseline_predictions = predict_samples(baseline_samples)
        baseline = float(np.mean(baseline_predictions))

        # ----------------------------------------------------
        # Interventional attribution
        # ----------------------------------------------------
        f_do = np.zeros(self.anm.p, dtype=float)
        standard_errors = np.zeros(self.anm.p, dtype=float)
        confidence_intervals = np.zeros((self.anm.p, 2), dtype=float)
        intervention_values = []

        for j in range(self.anm.p):
            intervention_samples = self.anm.do_sample(
                {j: x_obs[j]},
                K=self.K_cai,
            )

            intervention_predictions = predict_samples(intervention_samples)
            intervention_mean = float(np.mean(intervention_predictions))

            f_do[j] = intervention_mean
            cai_j = intervention_mean - baseline

            # Monte-Carlo standard error
            se_j = np.sqrt(
                np.var(intervention_predictions, ddof=1) / self.K_cai
                + np.var(baseline_predictions, ddof=1) / self.K_cai
            )

            standard_errors[j] = float(se_j)
            confidence_intervals[j] = (
                cai_j - 1.96 * se_j,
                cai_j + 1.96 * se_j,
            )

            intervention_values.append(intervention_predictions)

            if verbose:
                print(
                    f"  [{j + 1:2d}/{self.anm.p:2d}] "
                    f"{feature_names[j]:<35} "
                    f"CAI={cai_j:+.6f}"
                )

        # ----------------------------------------------------
        # Final CAI scores
        # ----------------------------------------------------
        cai = f_do - baseline
        cai_norm = cai / (np.abs(cai).sum() + 1e-8)

        return CAIResult(
            x_obs=x_obs.copy(),
            cai=cai,
            cai_norm=cai_norm,
            baseline=baseline,
            f_do=f_do,
            feature_names=list(feature_names),
            standard_errors=standard_errors,
            confidence_intervals=confidence_intervals,
            intervention_values=intervention_values,
            n_samples=self.K_cai,
        )
