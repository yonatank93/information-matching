import numpy as np
import scipy.optimize as scopt

from information_matching import leastsq

np.random.seed(1)


# Define the test model
y = np.random.randn(10)
J = np.random.randn(10, 3)


def residual(x):
    return y - J @ x


def cost(x):
    return 0.5 * np.linalg.norm(residual(x)) ** 2


# Optimizations using scipy
opt_leastsq = scopt.least_squares(residual, np.ones(3))
opt_minimize = scopt.minimize(cost, np.ones(3))
# Convert
leastsq_dict = leastsq.convert_leastsq_result_format(opt_leastsq)
minimize_dict = leastsq.convert_minimize_result_format(opt_minimize)
# Toy result
opt_toy = np.linalg.lstsq(J, y, rcond=-1)
toy_dict = (
    opt_toy[0],
    {
        "converged": 1,
        "iters": [1, 1, 1, 1],
        "msg": "Linear least squares",
        "fvec": 0.0,  # We set this to zero, so the comparison function should pick this
        "fjac": np.zeros(3),
    },
)


def test_format_conversion():
    # Check the keys
    expected_keys = ["converged", "iters", "msg", "fvec", "fjac"]
    assert all([key in expected_keys for key in leastsq_dict[1]])
    assert all([key in expected_keys for key in minimize_dict[1]])
    # Also check the other direction to see if the keys are exactly the same
    assert all([key in leastsq_dict[1] for key in expected_keys])
    assert all([key in minimize_dict[1] for key in expected_keys])


def test_comparison_function():
    opt_results = {0: leastsq_dict, 1: minimize_dict, 2: toy_dict}
    best_result = leastsq.compare_opt_results(opt_results)
    # Compare
    assert all(best_result[0] == toy_dict[0])  # Best parameters
    assert best_result[2][1] == toy_dict[1]  # Information dictionary


if __name__ == "__main__":
    test_format_conversion()
    test_comparison_function()
