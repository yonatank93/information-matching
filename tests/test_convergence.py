import numpy as np

from information_matching.termination import check_convergence

np.random.seed(1)

# Dummy weights
weight_1 = {"1": abs(np.random.randn()), "2": abs(np.random.randn())}
weight_2 = weight_1.copy()
weight_3 = {"1": abs(np.random.randn()), "2": abs(np.random.randn())}


def test_convergence():
    assert check_convergence(weight_1, weight_2), "Weights 1 and 2 are the same"
    assert not check_convergence(weight_1, weight_3), "Weights 1 and 3 are different"


if __name__ == "__main__":
    test_convergence()
