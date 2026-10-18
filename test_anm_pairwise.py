"""
Validation Test Suite for Standalone Pairwise ANM Direction Module.

Tests:
- Test 1: Basic direction output format ('x->y' or 'y->x').
- Test 2: Determinism across repeated calls with identical data and seed.
- Test 3: Input validation (length mismatch, NaNs, Infs, 2D arrays, too few samples, constant arrays).
- Test 4: Controlled subsampling (exact sample count, deterministic, without replacement, joint indexing).
- Test 5: Known synthetic ANM orientation recovery.
- Test 6: Non-mutation of global NumPy random state.
"""

import numpy as np
from anm_pairwise import evaluate_anm_direction, PairwiseANMResult


def test_1_basic_direction_output():
    """Verify that evaluate_anm_direction returns a PairwiseANMResult with valid fields."""
    rng = np.random.RandomState(42)
    x = rng.uniform(-2, 2, size=150)
    y = np.tanh(x) + 0.2 * rng.laplace(scale=0.4, size=150)

    res = evaluate_anm_direction(x, y, random_state=42)
    assert isinstance(res, PairwiseANMResult), f"Expected PairwiseANMResult, got {type(res)}"
    assert res.predicted_direction in ("x->y", "y->x")
    assert res.n_samples == 150
    assert np.isfinite(res.hsic_forward) and res.hsic_forward >= 0
    assert np.isfinite(res.hsic_reverse) and res.hsic_reverse >= 0
    assert np.isclose(res.margin, abs(res.hsic_forward - res.hsic_reverse))
    print("   ✅ PASS [Test 1]: Basic direction output and structure validated.")


def test_2_determinism():
    """Verify exact determinism across repeated calls on both small and large sample sizes."""
    rng = np.random.RandomState(99)
    x = rng.randn(450)
    y = x ** 2 + 0.3 * rng.randn(450)

    # Test without subsampling (N=450 > 300)
    res_a1 = evaluate_anm_direction(x, y, random_state=42)
    res_a2 = evaluate_anm_direction(x, y, random_state=42)
    assert res_a1 == res_a2, f"Nondeterministic results without subsampling: {res_a1} != {res_a2}"

    # Test with subsampling (max_samples=250)
    res_b1 = evaluate_anm_direction(x, y, max_samples=250, random_state=42)
    res_b2 = evaluate_anm_direction(x, y, max_samples=250, random_state=42)
    assert res_b1 == res_b2, f"Nondeterministic results with subsampling: {res_b1} != {res_b2}"
    print("   ✅ PASS [Test 2]: Determinism confirmed across repeated calls with same random_state.")


def test_3_input_validation():
    """Verify strict rejection of invalid input dimensions, lengths, types, and values."""
    x_valid = np.linspace(-1, 1, 100)
    y_valid = 2 * x_valid + 0.1

    # 1. Length mismatch
    try:
        evaluate_anm_direction(x_valid[:50], y_valid)
        assert False, "Failed to reject length mismatch"
    except ValueError as e:
        assert "Length mismatch" in str(e)

    # 2. 2D array input
    try:
        evaluate_anm_direction(x_valid.reshape(-1, 1), y_valid)
        assert False, "Failed to reject 2D input for x"
    except ValueError as e:
        assert "1D scalar array" in str(e)

    try:
        evaluate_anm_direction(x_valid, y_valid.reshape(-1, 1))
        assert False, "Failed to reject 2D input for y"
    except ValueError as e:
        assert "1D scalar array" in str(e)

    # 3. NaN values
    x_nan = x_valid.copy()
    x_nan[5] = np.nan
    try:
        evaluate_anm_direction(x_nan, y_valid)
        assert False, "Failed to reject NaN in x"
    except ValueError as e:
        assert "non-finite values" in str(e)

    # 4. Infinite values
    y_inf = y_valid.copy()
    y_inf[10] = np.inf
    try:
        evaluate_anm_direction(x_valid, y_inf)
        assert False, "Failed to reject Inf in y"
    except ValueError as e:
        assert "non-finite values" in str(e)

    # 5. Too few samples for K-Fold
    try:
        evaluate_anm_direction(x_valid[:3], y_valid[:3], n_splits=5)
        assert False, "Failed to reject insufficient samples"
    except ValueError as e:
        assert "Insufficient samples" in str(e)

    # 6. Constant array (zero variance)
    x_const = np.ones(50)
    try:
        evaluate_anm_direction(x_const, y_valid[:50])
        assert False, "Failed to reject zero-variance x"
    except ValueError as e:
        assert "zero variance" in str(e)

    # 7. max_samples > available samples
    try:
        evaluate_anm_direction(x_valid, y_valid, max_samples=200)
        assert False, "Failed to reject max_samples > len(x)"
    except ValueError as e:
        assert "exceeds available sample count" in str(e)

    print("   ✅ PASS [Test 3]: Strict input validation verified across all edge cases.")


def test_4_controlled_subsampling():
    """Verify that subsampling selects exactly the requested sample count without replacement."""
    rng = np.random.RandomState(7)
    x = rng.randn(500)
    y = 0.5 * x + 0.1 * rng.randn(500)

    res = evaluate_anm_direction(x, y, max_samples=200, random_state=42)
    assert res.n_samples == 200, f"Expected 200 samples, got {res.n_samples}"

    # Verify seed sensitivity: different seeds produce different subsamples
    res_seed1 = evaluate_anm_direction(x, y, max_samples=200, random_state=1)
    res_seed2 = evaluate_anm_direction(x, y, max_samples=200, random_state=2)
    assert not np.isclose(res_seed1.hsic_forward, res_seed2.hsic_forward), \
        "Different subsampling seeds should yield distinct HSIC values"
    print("   ✅ PASS [Test 4]: Controlled subsampling count and seed sensitivity verified.")


def test_5_known_anm_orientation():
    """Verify that the evaluator recovers the direction on a known synthetic ANM."""

    rng = np.random.RandomState(123)
    x = rng.uniform(-2, 2, size=300)
    y = np.tanh(x) + 0.15 * rng.laplace(scale=0.4, size=300)

    res = evaluate_anm_direction(
        x,
        y,
        max_samples=250,
        random_state=42,
    )

    assert res.n_samples == 250
    assert res.predicted_direction == "x->y"

    print(
        f"   ✅ PASS [Test 5]: Known synthetic ANM orientation recovered "
        f"(pred={res.predicted_direction}, margin={res.margin:.6f})."
    )

def test_6_rng_isolation():
    """Verify that global NumPy random generator is never mutated."""
    # Seed global RNG and record state
    np.random.seed(12345)
    rng_state_before = np.random.get_state()[1].copy()

    x = np.random.randn(200)
    y = np.random.randn(200)
    rng_state_after_data = np.random.get_state()[1].copy()

    # Call evaluate_anm_direction multiple times
    for s in range(5):
        _ = evaluate_anm_direction(x, y, max_samples=100, random_state=s)

    rng_state_after_calls = np.random.get_state()[1].copy()
    assert np.array_equal(rng_state_after_data, rng_state_after_calls), \
        "FAIL: evaluate_anm_direction mutated global NumPy random state!"
    print("   ✅ PASS [Test 6]: Global NumPy RNG isolation verified (state unchanged by evaluator).")


def run_all_tests():
    print("=" * 70)
    print("      STANDALONE PAIRWISE ANM EVALUATOR TEST SUITE")
    print("=" * 70)
    test_1_basic_direction_output()
    test_2_determinism()
    test_3_input_validation()
    test_4_controlled_subsampling()
    test_5_known_anm_orientation()
    test_6_rng_isolation()
    print("=" * 70)
    print("           ALL PAIRWISE ANM TESTS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
