import pytest
import numpy as np

from information_matching.fim import FIM_fd
from information_matching.fim.fim_fd import CD
from information_matching.fim import FIM_nd


xlist = np.random.uniform(0, 1, 10)
tlist = np.random.uniform(0, 1, 10)


def fn(x):
    return np.exp(x)


def fn_kwargs(x, t):
    return np.sum([np.exp(xi * t) for xi in x], axis=0)


# The derivative of this function is just np.exp(x)
jac_truth = np.diag(np.exp(xlist))
fim_truth = jac_truth.T @ jac_truth

# The following is the analytic derivative for the function with kwargs
jac_kwargs_truth = np.array([tlist * np.exp(x * tlist) for x in xlist]).T
fim_kwargs_truth = jac_kwargs_truth.T @ jac_kwargs_truth


def test_evaluation_fd():
    fim_fn = FIM_fd(fn, CD)
    jac_fd = fim_fn.compute_jacobian(xlist, h=0.001)
    # Compare the jacobian
    assert np.all(np.abs(jac_truth - jac_fd) < 1e-6)
    # Compare the FIM
    fim_fd = fim_fn.compute_FIM(xlist, h=0.001)
    assert np.all(np.abs(fim_truth - fim_fd) < 1e-4)


def test_evaluation_nd():
    fim_fn = FIM_nd(fn)
    jac_nd = fim_fn.compute_jacobian(xlist)
    # Compare the jacobian
    assert np.all(np.abs(jac_truth - jac_nd) < 1e-8)
    # Compare the FIM
    fim_nd = fim_fn.compute_FIM(xlist)
    assert np.all(np.abs(fim_truth - fim_nd) < 1e-6)


def test_kwargs_nd():
    step = 0.01  # Keyword argument for nd.Jacobian
    fim_fn = FIM_nd(fn_kwargs, step=step)
    # Test for the nd.Jacobian kwargs
    assert fim_fn.jac_kwargs == {"step": step}
    # Test for exception if function kwargs is not given
    with pytest.raises(TypeError):
        _ = fim_fn.compute_jacobian(xlist)
    # Test the Jacobian value
    jac_nd = fim_fn.compute_jacobian(xlist, tlist)
    assert np.allclose(jac_nd, jac_kwargs_truth, atol=1e-4, rtol=1e-4)
    # Test the FIM value
    fim_nd = fim_fn(xlist, tlist)
    assert np.allclose(fim_nd, fim_kwargs_truth, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_evaluation_fd()
    test_evaluation_nd()
    test_kwargs_nd()
