import numpy as np

from information_matching.convex_optimization import ConvexOpt

np.random.seed(1)


# Define the FIMs to try
scale = np.random.uniform(0, 10)
# Target
fim_target = np.eye(3) * scale
# Configurations
fim_config_1 = np.eye(3)
fim_config_2 = np.diag(np.zeros(3))
fim_configs_tensor = np.array([fim_config_1, fim_config_2])
# Config IDs
config_ids = ["1", "2"]

# Solve
cvxopt = ConvexOpt(fim_target, fim_configs_tensor, config_ids)
cvxopt.solve()


def test_default_norm():
    assert cvxopt.norm_qoi == 1.0, "Default norm_qoi fail"
    assert all(cvxopt.norm_conf == np.ones(2)), "Default norm_conf fail"
    assert all(cvxopt.norm_weights == np.ones(2)), "Default norm_weights fail"


def test_result_keys():
    for key in ["status", "wm", "dual_wm", "value", "rel_error", "violation"]:
        assert key in cvxopt.result, f"{key} not in result"


def test_optimal_result():
    opt_config = cvxopt.get_config_weights(1e-6)
    # For this simple case, we know which configuration is optimal and its optimal weight
    assert list(opt_config)[0] == config_ids[0], "Wrong optimal config"
    assert np.isclose(
        opt_config[config_ids[0]], scale, rtol=1e-4, atol=1e-4
    ), "Wrong optimal weight"


if __name__ == "__main__":
    test_default_norm()
    test_result_keys()
    test_optimal_result()