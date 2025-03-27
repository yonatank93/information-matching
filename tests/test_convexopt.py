import pytest
import numpy as np
import cvxpy as cp

from information_matching import ConvexOpt

np.random.seed(1)


# Define the FIMs to try
scale = np.random.uniform(0, 10)
# Target
fim_target = np.eye(3) * scale
# Candidates
fim_candidate_1 = np.eye(3)
fim_candidate_2 = np.diag(np.zeros(3))
fim_candidate_3 = np.diag([1, 2, 3])
fim_candidates = {"1": fim_candidate_1, "2": fim_candidate_2}

# Solve
cvxopt = ConvexOpt(fim_target, fim_candidates)
cvxopt.solve()


def test_default_scale():
    """Test if the default scaling factor is set correctly. The values should all be 1.0."""
    assert cvxopt.scale_qoi == 1.0, "Default scale_qoi fail"
    assert all(cvxopt.scale_conf == np.ones(2)), "Default scale_conf fail"
    assert all(cvxopt.scale_weights == np.ones(2)), "Default scale_weights fail"


def test_scaling():
    """Test if FIM scaling is implemented correctly."""
    fim_target_weighted = {"fim": fim_target, "fim_scale": 1 / 3}
    fim_candidates_weighted = {
        "1": {"fim": fim_candidate_1, "fim_scale": 3},
        "2": fim_candidate_2,
    }
    cvxopt = ConvexOpt(fim_target_weighted, fim_candidates_weighted)
    # FIM target scaling
    assert cvxopt.scale_qoi == 1 / 3, "Retrieving target FIM scaling fail"
    assert np.allclose(
        cvxopt.fim_qoi_vec, fim_target.flatten() / 3
    ), "Target FIM scaling fail"
    # FIM candidates scaling
    assert all(cvxopt.scale_conf == [3, 1]), "Retrieving candidate FIM scaling fail"
    assert np.allclose(
        cvxopt.fim_configs_vec[0], fim_candidate_1.flatten() * 3
    ), "Candidate FIM scaling fail"


def test_result_keys():
    """Check the keys of the .result property."""
    for key in ["status", "wm", "dual_wm", "value", "rel_error", "violation"]:
        assert key in cvxopt.result, f"{key} not in result"


def test_optimal_result():
    """Check the values of the optimal weights.

    Since the target FIM is just a scalar multiple of one of the candidate FIM, then the
    optimal weight should just be the same scaling factor.
    """
    opt_candidate = cvxopt.get_config_weights(1e-6, 1e-6)
    # For this simple case, we know which candidate is optimal and its optimal weight
    assert list(opt_candidate)[0] == "1", "Wrong optimal candidate"
    assert np.isclose(
        opt_candidate["1"], scale, rtol=1e-4, atol=1e-4
    ), "Wrong optimal weight"


def test_weight_upper_bound():
    """Test if the additional constraint of weight upper bound is implemented correctly.

    If it is implemented correctly, then the optimal weights should NOT exceed the upper
    bound.
    """
    fim_candidates = {"1": fim_candidate_1, "2": fim_candidate_2, "3": fim_candidate_3}
    w_ub = 3  # Upper bound for the optimal weights
    cvxopt = ConvexOpt(fim_target, fim_candidates, weight_upper_bound=w_ub)
    cvxopt.solve()
    opt_candidate = cvxopt.get_config_weights(1e-6, 1e-6)
    # This optimization should be successful, and the weights should be lower than the
    # upper bound
    for val in opt_candidate.values():
        assert val <= w_ub, "Optimal weight exceeds upper bound"


def test_different_objective_fn():
    """Test if the option to specify objective function is implemented correctly.

    In theory, if we use l2-norm objective function, then if we have a redundant data,
    the optimal weights of the redundant data should be the same. In contrast, the
    weights of the redundant data if we use l1-norm objective function don't need to be
    the same.
    """
    # Add a redundant data
    fim_candidates = {
        "1": fim_candidate_1,
        "2": fim_candidate_2,
        "3": fim_candidate_3,
        "4": fim_candidate_1,
    }
    # Test if an exception is raised if the objective function is not convex
    with pytest.raises(ValueError):
        cvxopt = ConvexOpt(fim_target, fim_candidates, obj_fn=cp.sqrt)
    # Test the optimization result
    cvxopt = ConvexOpt(fim_target, fim_candidates, obj_fn=cp.norm2)
    cvxopt.solve()
    # First, the optimal result should NOT select candidate 2, which has zero matrix
    # as the FIM
    assert "2" not in cvxopt.get_config_weights(), "Failed: Candidate 2 selected"
    # Second, the optimal weights of the redundant data (first and last data) should be
    # the same
    weights = list(cvxopt.get_config_weights().values())
    assert np.isclose(
        weights[0], weights[-1], atol=1e-6, rtol=1e-6
    ), "Failed: Redundant data weights should have the same weights"


if __name__ == "__main__":
    test_default_scale()
    test_scaling()
    test_result_keys()
    test_optimal_result()
    test_weight_upper_bound()
    test_different_objective_fn()
