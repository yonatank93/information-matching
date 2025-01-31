import pytest
import numpy as np

from information_matching.fim.finitediff import FiniteDifference
from information_matching.fim import FIM_fd, FIM_nd

# try:
#     from information_matching.fim.fim_jl import FIM_jl

#     julia_avail = True
# except ImportError:
#     print("Skipping testing FIM module that uses Julia")
#     julia_avail = False


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


def test_finitediff():
    # Derivative setup
    FD = FiniteDifference(xlist)
    params_set = FD.generate_params_set()
    # Generate predictions set
    predictions_set = {}
    for param_key, values in params_set.items():
        predictions_set.update({param_key: fn(values)})
    # Estimate derivative
    jac = FD.estimate_derivative(predictions_set)
    # Compare with analytical derivative
    assert np.all(np.diag(jac - jac_truth) / np.diag(jac_truth) < 0.1)


def test_evaluation_fd():
    # The comparison tolerance should be set pretty high, because this method is not that
    # accurate
    fim_fn = FIM_fd(fn)
    jac_fd = fim_fn.Jacobian(xlist)
    # Compare the Jacobian
    # The first derivative error should be of order h (default h=0.1)
    assert np.all(np.diag(jac_fd - jac_truth) / np.diag(jac_truth) < 0.1)
    # Compare the FIM
    # The error should be of order sqrt(h)
    fim_fd = fim_fn.FIM(xlist)
    assert np.all(np.diag(fim_fd - fim_truth) / np.diag(fim_truth) < np.sqrt(0.1))


def test_evaluation_nd():
    fim_fn = FIM_nd(fn)
    jac_nd = fim_fn.Jacobian(xlist)
    # Compare the Jacobian
    assert np.all(np.diag(jac_nd - jac_truth) / np.diag(jac_truth) < 1e-8)
    # Compare the FIM
    fim_nd = fim_fn.FIM(xlist)
    assert np.all(np.diag(fim_nd - fim_truth) / np.diag(fim_truth) < 1e-8)


# def test_evaluation_jl():
#     if julia_avail:
#         fim_fn = FIM_jl(fn)
#         jac_jl = fim_fn.Jacobian(xlist)
#         # Compare the Jacobian
#         assert np.all(np.diag(jac_jl - jac_truth) / np.diag(jac_truth) < 1e-8)
#         # Compare the FIM
#         fim_jl = fim_fn.FIM(xlist)
#         assert np.all(np.diag(fim_jl - fim_truth) / np.diag(fim_truth) < 1e-8)


def test_kwargs_fd():
    h = 0.01
    fim_fn = FIM_fd(fn_kwargs, method="CD", h=h, nprocs=2)
    # Test for exception if function kwargs is not given
    with pytest.raises(TypeError):
        _ = fim_fn.Jacobian(xlist)
    # Test the Jacobian value
    # The first derivative error should be of order h^2, since we are using CD
    jac_fd = fim_fn.Jacobian(xlist, tlist)
    assert np.all(np.abs((jac_fd - jac_kwargs_truth) / jac_kwargs_truth) < h**2)
    # Test the FIM value
    # The error should be of order h
    fim_fd = fim_fn(xlist, tlist)
    assert np.all(np.abs((fim_fd - fim_kwargs_truth) / fim_kwargs_truth) < h)


def test_kwargs_nd():
    kwargs = dict(step=0.01)  # Keyword argument for nd.Jacobian
    fim_fn = FIM_nd(fn_kwargs, **kwargs)
    # Test for exception if function kwargs is not given
    with pytest.raises(TypeError):
        _ = fim_fn.Jacobian(xlist)
    # Test the Jacobian value
    jac_nd = fim_fn.Jacobian(xlist, tlist)
    assert np.allclose(jac_nd, jac_kwargs_truth, atol=1e-4, rtol=1e-4)
    # Test the FIM value
    fim_nd = fim_fn(xlist, tlist)
    assert np.allclose(fim_nd, fim_kwargs_truth, atol=1e-4, rtol=1e-4)


# def test_kwargs_jl():
#     if julia_avail:
#         kwargs = dict(h=0.01)  # Keyword argument for Julia's Numdifftools.Jacobian
#         fim_fn = FIM_jl(fn_kwargs, **kwargs)
#         # Test for exception if function kwargs is not given
#         with pytest.raises(TypeError):
#             _ = fim_fn.Jacobian(xlist)
#         # Test the Jacobian value
#         jac_jl = fim_fn.Jacobian(xlist, tlist)
#         assert np.allclose(jac_jl, jac_kwargs_truth, atol=1e-4, rtol=1e-4)
#         # Test the FIM value
#         fim_jl = fim_fn(xlist, tlist)
#         assert np.allclose(fim_jl, fim_kwargs_truth, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_finitediff()
    test_evaluation_fd()
    test_evaluation_nd()
    # test_evaluation_jl()
    test_kwargs_fd()
    test_kwargs_nd()
    # test_kwargs_jl()
