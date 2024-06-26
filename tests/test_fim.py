import numpy as np

from information_matching.fim import FIM_fd
from information_matching.fim.fim_fd import CD
from information_matching.fim import FIM_nd


xlist = np.linspace(0, 1)


def fn(x):
    return np.exp(x)


# The derivative of this function is just np.exp(x
jac_truth = np.diag(np.exp(xlist))
fim_truth = jac_truth.T @ jac_truth


def test_fd():
    fim_fd = FIM_fd(fn, CD)
    jac_fd = fim_fd.compute_jacobian(xlist, h=0.001)
    # Compare the jacobian
    assert np.all(np.abs(jac_truth - jac_fd) < 1e-6)
    # Compare the FIM
    fim_fd = fim_fd.compute_FIM(xlist, h=0.001)
    assert np.all(np.abs(fim_truth - fim_fd) < 1e-4)


def test_nd():
    fim_nd = FIM_nd(fn)
    jac_nd = fim_nd.compute_jacobian(xlist)
    # Compare the jacobian
    assert np.all(np.abs(jac_truth - jac_nd) < 1e-8)
    # Compare the FIM
    fim_nd = fim_nd.compute_FIM(xlist)
    assert np.all(np.abs(fim_truth - fim_nd) < 1e-6)


if __name__ == "__main__":
    test_fd()
    test_nd()
