import numpy as np

from information_matching.convex_optimization import ConvexOpt

np.random.seed(1)


# Define the FIMs to try
scale = np.random.uniform(0, 10)
# Target
fim_target = np.eye(3) * scale
fim_target_weighted = {"fim": fim_target, "fim_scale": 1 / 3}
# Configurations
fim_config_1 = np.eye(3)
fim_config_2 = np.diag(np.zeros(3))
fim_config_3 = np.diag([1, 2, 3])
fim_configs = {"1": fim_config_1, "2": fim_config_2}
fim_configs_weighted = {"1": {"fim": fim_config_1, "fim_scale": 3}, "2": fim_config_2}
fim_configs_extended = {"1": fim_config_1, "2": fim_config_2, "3": fim_config_3}

# Solve
cvxopt = ConvexOpt(fim_target, fim_configs)
cvxopt.solve()
# Alternative cvxopt to test the FIM scaling
cvxopt_alt = ConvexOpt(fim_target_weighted, fim_configs_weighted)


def test_default_scale():
    assert cvxopt.scale_qoi == 1.0, "Default scale_qoi fail"
    assert all(cvxopt.scale_conf == np.ones(2)), "Default scale_conf fail"
    assert all(cvxopt.scale_weights == np.ones(2)), "Default scale_weights fail"


def test_scaling():
    # FIM target scaling
    assert cvxopt_alt.scale_qoi == 1 / 3, "Retrieving target FIM scaling fail"
    assert np.allclose(
        cvxopt_alt.fim_qoi_vec, fim_target.flatten() / 3
    ), "Target FIM scaling fail"
    # FIM configs scaling
    assert all(cvxopt_alt.scale_conf == [3, 1]), "Retrieving config FIM scaling fail"
    assert np.allclose(
        cvxopt_alt.fim_configs_vec[0], fim_config_1.flatten() * 3
    ), "Config FIM scaling fail"


def test_result_keys():
    for key in ["status", "wm", "dual_wm", "value", "rel_error", "violation"]:
        assert key in cvxopt.result, f"{key} not in result"


def test_optimal_result():
    opt_config = cvxopt.get_config_weights(1e-6, 1e-6)
    # For this simple case, we know which configuration is optimal and its optimal weight
    assert list(opt_config)[0] == "1", "Wrong optimal config"
    assert np.isclose(
        opt_config["1"], scale, rtol=1e-4, atol=1e-4
    ), "Wrong optimal weight"


def test_weight_upper_bound():
    w_ub = 3  # Upper bound for the optimal weights
    cvxopt = ConvexOpt(fim_target, fim_configs_extended, weight_upper_bound=w_ub)
    cvxopt.solve()
    opt_config = cvxopt.get_config_weights(1e-6, 1e-6)
    # This optimization should be successful, and the weights should be lower than the
    # upper bound
    for val in opt_config.values():
        assert val <= w_ub, "Optimal weight exceeds upper bound"


if __name__ == "__main__":
    test_default_scale()
    test_scaling()
    test_result_keys()
    test_optimal_result()
    test_weight_upper_bound()
