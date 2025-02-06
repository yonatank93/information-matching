"""Unit tests for the Transform module for parameter transformation."""

import numpy as np
from information_matching.transform import (
    func_wrapper,
    AffineTransform,
    LogTransform,
    CombinedTransform,
)

np.random.seed(1)

# Test parameter value
x = np.random.randn(3)

# Instantiate the Affine parameter transform
A = np.random.randn(3, 3)
x0 = np.random.randn(3)
b = np.random.randn(3)
transform_affine = AffineTransform(A, x0, b)

# Instantiate the Log parameter transform
sign = np.sign(x)
transform_log = LogTransform(sign)

# Instantiate the Combined parameter transform
transforms = [transform_affine, transform_log]
xlong = np.append(x, x)
param_idx = [range(len(x)), range(len(x), 2 * len(x))]
transform_combined = CombinedTransform(transforms, param_idx)


def test_transform():
    """Test forward transformation method."""
    # Test affine transformation by manual calculation
    assert np.allclose(transform_affine(x), A @ (x - x0) + b)
    # Test log transformation by manual calculation
    assert np.allclose(transform_log(x), np.log(np.abs(x)))
    # Test combined transformation by manual calculation
    assert np.allclose(
        transform_combined(xlong), np.append(A @ (x - x0) + b, np.log(np.abs(x)))
    )


def test_inverse_transform():
    """Test inverse transformation method."""
    # Test if affine inverse transformation gives back the original parameter values
    assert np.allclose(transform_affine.inverse_transform(transform_affine(x)), x)
    # Test if log inverse transformation gives back the original parameter values
    assert np.allclose(transform_log.inverse_transform(transform_log(x)), x)
    # Test if combined inverse transformation gives back the original parameter values
    assert np.allclose(
        transform_combined.inverse_transform(transform_combined(xlong)), xlong
    )


def test_func_wrapper():
    """Test func_wrapper function."""

    def test_func(x):
        return x + 1

    # Test if the wrapped function returns the expected result. If so, then test_func
    # evaluated at x is the same as func_wrapper(test_func, transform) evaluated at
    # transform(x).
    assert np.allclose(
        func_wrapper(test_func, transform_log)(transform_log(x)), test_func(x)
    )


if __name__ == "__main__":
    test_affine_transform()
    test_inverse_transform()
    test_func_wrapper()
