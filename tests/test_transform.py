"""Unit tests for the Transform module for parameter transformation."""

import numpy as np
from information_matching.transform import (
    func_wrapper,
    AffineTransform,
    LogTransform,
    SplitTransform,
    ComposedTransform,
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

# Instantiate the Split parameter transform
transforms_list = [transform_log, transform_affine]
xlong = np.append(x, x)
param_idx = [range(len(x)), range(len(x), 2 * len(x))]
transform_split = SplitTransform(transforms_list, param_idx)

# Instantiate the Composed parameter transform
transform_composed = ComposedTransform(transforms_list)


def test_transform():
    """Test forward transformation method."""
    # Test affine transformation by manual calculation
    assert np.allclose(transform_affine(x), A @ (x - x0) + b)
    # Test log transformation by manual calculation
    assert np.allclose(transform_log(x), np.log(np.abs(x)))
    # Test split transformation by manual calculation
    assert np.allclose(
        transform_split(xlong), np.append(np.log(np.abs(x)), A @ (x - x0) + b)
    )


def test_inverse_transform():
    """Test inverse transformation method."""
    # Test if affine inverse transformation gives back the original parameter values
    assert np.allclose(transform_affine.inverse_transform(transform_affine(x)), x)
    # Test if log inverse transformation gives back the original parameter values
    assert np.allclose(transform_log.inverse_transform(transform_log(x)), x)
    # Test if split inverse transformation gives back the original parameter values
    assert np.allclose(transform_split.inverse_transform(transform_split(xlong)), xlong)


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
    test_transform()
    test_inverse_transform()
    test_func_wrapper()
