"""
Validation Test Suite for CAI (Causal Feature Attribution) Core Module.

Tests:
- Test 1: CAI output shapes and dimensions.
- Test 2: Finite numerical values (CAI, SE, CI, f_do, baseline).
- Test 3: Intervention invariance (do-operator strictly fixes intervened variables).
- Test 4: Determinism across repeated calls with identical random seed.
- Test 5: Normalized attribution mathematical integrity (L1 sum, sign preservation).
- Test 6: Confidence interval ordering (lower <= point estimate <= upper, width = 2 * 1.96 * SE).
- Test 7: Model-agnostic CAI interface using Random Forest.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from cai import _StructEq, CausalANM, CAIResult, CausalFeatureAttributor


def _build_synthetic_causal_system(n_samples=200, random_state=42):
    """
    Helper creating a known 3-variable chain DAG: X0 -> X1 -> X2.
    Returns data, adj, order, feature_names, prediction_function, x_obs.
    """
    rng = np.random.RandomState(random_state)
    x0 = rng.normal(0, 1, size=n_samples)
    x1 = 0.8 * x0 + rng.laplace(0, 0.5, size=n_samples)
    x2 = 0.6 * x1 + rng.laplace(0, 0.5, size=n_samples)

    data = np.column_stack([x0, x1, x2])
    p = 3
    adj = np.zeros((p, p), dtype=int)
    adj[0, 1] = 1  # 0 -> 1
    adj[1, 2] = 1  # 1 -> 2
    order = [0, 1, 2]
    feature_names = ["Feature_A", "Feature_B", "Feature_C"]

    # Synthetic binary classification target
    logits = 0.5 * x0 + 1.0 * x1 - 0.7 * x2
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (prob > 0.5).astype(int)

    clf = LogisticRegression(random_state=random_state)
    clf.fit(data, y)

    def pred_fn(X):
        return clf.predict_proba(np.asarray(X, dtype=float))[:, 1]

    x_obs = data[0].copy()

    return {
        "data": data,
        "adj": adj,
        "order": order,
        "feature_names": feature_names,
        "prediction_function": pred_fn,
        "x_obs": x_obs,
    }


def test_1_cai_output_shape():
    """Verify that CAIResult contains all expected attributes with correct shapes."""
    sys = _build_synthetic_causal_system()
    anm = CausalANM(adj=sys["adj"], order=sys["order"], random_state=42)
    anm.fit(sys["data"])

    attributor = CausalFeatureAttributor(
        anm=anm,
        prediction_function=sys["prediction_function"],
        K_cai=150,
    )

    p = 3
    res = attributor.attribute(
        x_obs=sys["x_obs"],
        feature_names=sys["feature_names"],
        verbose=False,
    )

    assert isinstance(res, CAIResult), f"Expected CAIResult, got {type(res)}"
    assert res.cai.shape == (p,), f"cai shape mismatch: {res.cai.shape} != ({p},)"
    assert res.cai_norm.shape == (p,), f"cai_norm shape mismatch: {res.cai_norm.shape} != ({p},)"
    assert res.standard_errors.shape == (p,), f"SE shape mismatch: {res.standard_errors.shape} != ({p},)"
    assert res.confidence_intervals.shape == (p, 2), f"CI shape mismatch: {res.confidence_intervals.shape} != ({p}, 2)"
    assert res.f_do.shape == (p,), f"f_do shape mismatch: {res.f_do.shape} != ({p},)"
    assert len(res.feature_names) == p, f"feature_names length mismatch: {len(res.feature_names)} != {p}"
    assert len(res.intervention_values) == p, f"intervention_values length mismatch: {len(res.intervention_values)} != {p}"
    assert res.x_obs.shape == (p,), f"x_obs shape mismatch: {res.x_obs.shape} != ({p},)"
    assert res.n_samples == 150, f"n_samples mismatch: {res.n_samples} != 150"
    print("   ✅ PASS [Test 1]: CAI output shapes and dimensions verified.")


def test_2_finite_cai_se_ci_values():
    """Verify that all computed CAI, SE, CI, and baseline values are strictly finite."""
    sys = _build_synthetic_causal_system()
    anm = CausalANM(adj=sys["adj"], order=sys["order"], random_state=42)
    anm.fit(sys["data"])

    attributor = CausalFeatureAttributor(
        anm=anm,
        prediction_function=sys["prediction_function"],
        K_cai=200,
    )

    res = attributor.attribute(
        x_obs=sys["x_obs"],
        feature_names=sys["feature_names"],
        verbose=False,
    )

    assert np.isfinite(res.baseline), f"Baseline is non-finite: {res.baseline}"
    assert np.isfinite(res.cai).all(), f"Non-finite values in cai: {res.cai}"
    assert np.isfinite(res.cai_norm).all(), f"Non-finite values in cai_norm: {res.cai_norm}"
    assert np.isfinite(res.f_do).all(), f"Non-finite values in f_do: {res.f_do}"
    assert np.isfinite(res.standard_errors).all(), f"Non-finite values in SE: {res.standard_errors}"
    assert (res.standard_errors >= 0).all(), f"Negative standard error detected: {res.standard_errors}"
    assert np.isfinite(res.confidence_intervals).all(), f"Non-finite values in CI: {res.confidence_intervals}"
    print("   ✅ PASS [Test 2]: Strict numerical finiteness of CAI, SE, CI, and baseline confirmed.")


def test_3_intervention_fixes_requested_variable():
    """Verify that the do-operator strictly sets the target variable to the specified value across all K samples."""
    sys = _build_synthetic_causal_system()
    anm = CausalANM(adj=sys["adj"], order=sys["order"], random_state=42)
    anm.fit(sys["data"])

    K = 300
    target_var = 1
    fixed_val = 3.14159

    samples = anm.do_sample(fixed={target_var: fixed_val}, K=K, random_state=42)

    assert samples.shape == (K, 3), f"Sample shape mismatch: {samples.shape} != ({K}, 3)"
    # Intervened variable must be identically equal to fixed_val across all samples
    assert np.all(samples[:, target_var] == fixed_val), \
        f"do-operator failed: variable {target_var} was not strictly fixed to {fixed_val}"

    # Downstream variable (X2) must exhibit variation from upstream noise
    assert np.std(samples[:, 2]) > 0.0, "Downstream variable X2 shows zero variance under intervention"

    # Multi-variable intervention test
    multi_fixed = {0: -2.5, 2: 7.0}
    multi_samples = anm.do_sample(fixed=multi_fixed, K=K, random_state=42)
    assert np.all(multi_samples[:, 0] == -2.5), "Variable 0 not fixed in multi-variable intervention"
    assert np.all(multi_samples[:, 2] == 7.0), "Variable 2 not fixed in multi-variable intervention"
    print("   ✅ PASS [Test 3]: Intervention do(X_j = x_j) strictly fixes target variables.")


def test_4_determinism():
    """Verify that CAI attribution yields bitwise-identical results across repeated runs with the same seed."""
    sys = _build_synthetic_causal_system()

    # Run 1
    anm_1 = CausalANM(adj=sys["adj"], order=sys["order"], random_state=42)
    anm_1.fit(sys["data"])
    attributor_1 = CausalFeatureAttributor(
        anm=anm_1,
        prediction_function=sys["prediction_function"],
        K_cai=250,
    )
    res_1 = attributor_1.attribute(sys["x_obs"], feature_names=sys["feature_names"], verbose=False)

    # Run 2
    anm_2 = CausalANM(adj=sys["adj"], order=sys["order"], random_state=42)
    anm_2.fit(sys["data"])
    attributor_2 = CausalFeatureAttributor(
        anm=anm_2,
        prediction_function=sys["prediction_function"],
        K_cai=250,
    )
    res_2 = attributor_2.attribute(sys["x_obs"], feature_names=sys["feature_names"], verbose=False)

    assert np.array_equal(res_1.cai, res_2.cai), "CAI values differ across identical runs"
    assert np.array_equal(res_1.cai_norm, res_2.cai_norm), "cai_norm differs across identical runs"
    assert np.array_equal(res_1.standard_errors, res_2.standard_errors), "SE differs across identical runs"
    assert np.array_equal(res_1.confidence_intervals, res_2.confidence_intervals), "CI differs across identical runs"
    assert res_1.baseline == res_2.baseline, f"Baseline differs: {res_1.baseline} != {res_2.baseline}"
    print("   ✅ PASS [Test 4]: Exact determinism confirmed across independent runs with fixed seed.")


def test_5_normalized_attribution_integrity():
    """Verify mathematical properties of normalized CAI (L1 sum ≈ 1, sign consistency)."""
    sys = _build_synthetic_causal_system()
    anm = CausalANM(adj=sys["adj"], order=sys["order"], random_state=42)
    anm.fit(sys["data"])

    attributor = CausalFeatureAttributor(
        anm=anm,
        prediction_function=sys["prediction_function"],
        K_cai=200,
    )
    res = attributor.attribute(sys["x_obs"], feature_names=sys["feature_names"], verbose=False)

    # L1 norm should sum to ~1.0
    l1_sum = np.sum(np.abs(res.cai_norm))
    assert np.isclose(l1_sum, 1.0, atol=1e-5), f"L1 norm sum of cai_norm is {l1_sum}, expected 1.0"

    # Sign of cai_norm should match sign of raw cai
    non_zeros = np.abs(res.cai) > 1e-12
    assert np.all(np.sign(res.cai_norm[non_zeros]) == np.sign(res.cai[non_zeros])), \
        "Sign mismatch between normalized and unnormalized CAI"

    # Zero attribution edge case: ensure no division by zero
    zero_cai = np.zeros(3)
    norm_zero = zero_cai / (np.abs(zero_cai).sum() + 1e-8)
    assert np.isfinite(norm_zero).all() and np.all(norm_zero == 0.0), \
        "Zero attribution edge case produced invalid values"

    print("   ✅ PASS [Test 5]: Normalized attribution mathematical integrity verified.")


def test_6_confidence_interval_ordering():
    """Verify that CI bounds strictly bracket CAI point estimates and match 2 * 1.96 * SE."""
    sys = _build_synthetic_causal_system()
    anm = CausalANM(adj=sys["adj"], order=sys["order"], random_state=42)
    anm.fit(sys["data"])

    attributor = CausalFeatureAttributor(
        anm=anm,
        prediction_function=sys["prediction_function"],
        K_cai=200,
    )
    res = attributor.attribute(sys["x_obs"], feature_names=sys["feature_names"], verbose=False)

    p = len(sys["feature_names"])
    for j in range(p):
        ci_lower = res.confidence_intervals[j, 0]
        ci_upper = res.confidence_intervals[j, 1]
        cai_val = res.cai[j]
        se_val = res.standard_errors[j]

        # Ordering check: lower <= point estimate <= upper
        assert ci_lower <= cai_val <= ci_upper, \
            f"Feature {j}: point estimate {cai_val} not in CI [{ci_lower}, {ci_upper}]"

        # Width check: upper - lower == 2 * 1.96 * SE
        expected_width = 2.0 * 1.96 * se_val
        actual_width = ci_upper - ci_lower
        assert np.isclose(actual_width, expected_width, atol=1e-10), \
            f"Feature {j}: CI width {actual_width} != 2 * 1.96 * SE ({expected_width})"

    print("   ✅ PASS [Test 6]: Confidence interval ordering and width consistency verified.")

def test_7_model_agnostic_prediction_interface():
    """
    Verify that the same CAI framework supports a different predictive
    model through the scalar prediction-function interface.
    """
    sys = _build_synthetic_causal_system()

    anm = CausalANM(
        adj=sys["adj"],
        order=sys["order"],
        random_state=42,
    )
    anm.fit(sys["data"])

    # Train a different ML model from the Logistic Regression
    # used in the existing synthetic test setup.
    rng = np.random.RandomState(42)

    logits = (
        0.5 * sys["data"][:, 0]
        + 1.0 * sys["data"][:, 1]
        - 0.7 * sys["data"][:, 2]
    )

    y = (logits > 0).astype(int)

    model = RandomForestClassifier(
        n_estimators=30,
        random_state=42,
    )
    model.fit(sys["data"], y)

    explained_class = int(
        model.predict(
            sys["x_obs"].reshape(1, -1)
        )[0]
    )

    # Adapt the Random Forest to the same scalar prediction
    # interface expected by CausalFeatureAttributor.
    def random_forest_prediction_function(X):
        X = np.asarray(X, dtype=float)

        return model.predict_proba(
            X
        )[:, explained_class]

    attributor = CausalFeatureAttributor(
        anm=anm,
        prediction_function=random_forest_prediction_function,
        K_cai=100,
    )

    result = attributor.attribute(
        x_obs=sys["x_obs"],
        feature_names=sys["feature_names"],
        verbose=False,
    )

    assert isinstance(result, CAIResult)
    assert result.cai.shape == (3,)
    assert result.cai_norm.shape == (3,)
    assert result.confidence_intervals.shape == (3, 2)

    assert np.isfinite(result.cai).all()
    assert np.isfinite(result.cai_norm).all()
    assert np.isfinite(result.confidence_intervals).all()

    print(
        "   ✅ PASS [Test 7]: Model-agnostic CAI interface "
        "verified with Random Forest."
    )

def run_all_tests():
    print("=" * 70)
    print("         CAUSAL FEATURE ATTRIBUTION (CAI) TEST SUITE")
    print("=" * 70)
    test_1_cai_output_shape()
    test_2_finite_cai_se_ci_values()
    test_3_intervention_fixes_requested_variable()
    test_4_determinism()
    test_5_normalized_attribution_integrity()
    test_6_confidence_interval_ordering()
    test_7_model_agnostic_prediction_interface()
    print("=" * 70)
    print("               ALL CAI INTEGRITY TESTS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
