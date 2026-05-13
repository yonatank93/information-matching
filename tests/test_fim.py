import pytest
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from information_matching.fim.finitediff import FiniteDifference
from information_matching.fim import FIM_fd, FIM_nd, FIM_linear
from information_matching.transform import AffineTransform

# try:
#     from information_matching.fim.fim_jl import FIM_jl

#     julia_avail = True
# except ImportError:
#     print("Skipping testing FIM module that uses Julia")
#     julia_avail = False


# Define functions/models to test the FIM methods
np.random.seed(1)
xlist = np.random.uniform(0, 1, 5)  # Parameters
tlist = np.random.uniform(0, 1, 10)  # Inputs

# The following is the analytic derivative
jac_truth = np.array([tlist * np.exp(x * tlist) for x in xlist]).T
fim_truth = jac_truth.T @ jac_truth
nparams = len(xlist)
design_matrix = np.array([tlist**ii for ii in range(nparams)]).T
idx = np.random.choice(range(nparams), size=nparams - 2, replace=False)


def fn(x, t):
    """General test function."""
    return np.sum([np.exp(xi * t) for xi in x], axis=0)


def fn_linear(x):
    """Linear test function."""
    return design_matrix @ x


def test_finitediff():
    # Derivative setup
    FD = FiniteDifference(xlist)
    params_set = FD.generate_params_set()
    # Generate predictions set
    predictions_set = {}
    for param_key, values in params_set.items():
        predictions_set.update({param_key: fn(values, tlist)})
    # Estimate derivative
    jac = FD.estimate_derivative(predictions_set)
    # Compare with analytical derivative
    assert np.all(np.diag(jac - jac_truth) / np.diag(jac_truth) < 0.1)


def test_fim_fd():
    h = 0.01
    with ThreadPoolExecutor(2) as pool:
        fim_fn = FIM_fd(fn, h=h, pool=pool)
        # Test the Jacobian value
        # The first derivative error should be of order h^2, since we are using CD
        jac_fd = fim_fn.Jacobian(xlist, tlist)
        assert np.all(np.abs((jac_fd - jac_truth) / jac_truth) < h**2)
        # Test the FIM value
        # The error should be of order h
        fim_fd = fim_fn(xlist, tlist)
        assert np.all(np.abs((fim_fd - fim_truth) / fim_truth) < h)


def test_fim_nd():
    kwargs = dict(step=0.01)  # Keyword argument for nd.Jacobian
    fim_fn = FIM_nd(fn, **kwargs)
    # Test the Jacobian value
    jac_nd = fim_fn.Jacobian(xlist, tlist)
    assert np.allclose(jac_nd, jac_truth, atol=1e-4, rtol=1e-4)
    # Test the FIM value
    fim_nd = fim_fn(xlist, tlist)
    assert np.allclose(fim_nd, fim_truth, atol=1e-4, rtol=1e-4)


def test_fim_linear():
    with ThreadPoolExecutor(2) as pool:
        fim_fn = FIM_linear(fn_linear, idx_list=idx, pool=pool)
        jac_linear = fim_fn.Jacobian(xlist)
        # Test the Jacobian - The Jacobian should be the same as the design matrix
        assert np.allclose(jac_linear, design_matrix[:, idx])
        # Test the FIM value
        fim_nd = fim_fn(xlist)
        D = design_matrix[:, idx]
        assert np.allclose(fim_nd, D.T @ D)


def test_fim_with_transformation():
    # Test that the FIM methods can handle a transformation of the parameters
    b = np.random.uniform(0, 1, len(xlist))
    # Finite difference
    fim_fd = FIM_fd(fn_linear)(xlist)
    fim_fd_transform = FIM_fd(fn_linear, transform=AffineTransform(b=b))(xlist)
    assert np.allclose(fim_fd, fim_fd_transform, atol=1e-4, rtol=1e-4)
    # Numdifftools
    fim_nd = FIM_nd(fn_linear)(xlist)
    fim_nd_transform = FIM_nd(fn_linear, transform=AffineTransform(b=b))(xlist)
    assert np.allclose(fim_nd, fim_nd_transform, atol=1e-4, rtol=1e-4)
    # # Julia
    # fim_jl = FIM_jl(fn_linear)(xlist)
    # fim_jl_transform = FIM_jl(fn_linear, transform=AffineTransform(b=b))(xlist)
    # assert np.allclose(fim_jl, fim_jl_transform, atol=1e-4, rtol=1e-4)


# def test_fim_jl():
#     if julia_avail:
#         kwargs = dict(h=0.01)  # Keyword argument for Julia's Numdifftools.Jacobian
#         fim_fn = FIM_jl(fn, **kwargs)
#         # Test for exception if function kwargs is not given
#         # Test the Jacobian value
#         jac_jl = fim_fn.Jacobian(xlist, tlist)
#         assert np.allclose(jac_jl, jac_truth, atol=1e-4, rtol=1e-4)
#         # Test the FIM value
#         fim_jl = fim_fn(xlist, tlist)
#         assert np.allclose(fim_jl, fim_truth, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_finitediff()
    test_fim_fd()
    test_fim_nd()
    test_fim_linear()
    # test_fim_jl()
    test_fim_with_transformation()
