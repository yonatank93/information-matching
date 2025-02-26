"""Unit tests for the Transform module for parameter transformation."""

import numpy as np
from information_matching.transform import (
    func_wrapper,
    AffineTransform,
    LogTransform,
    AtanhTransform,
    SplitTransform,
    ComposedTransform,
    transform_builder,
)

np.random.seed(1)
tol = np.finfo(float).eps ** 0.5

# Test parameter value
nparams = np.random.randint(1, 10)
x = np.random.randn(nparams)

# Instantiate the Affine parameter transform
A = np.random.randn(nparams, nparams)
x0 = np.random.randn(nparams)
b = np.random.randn(nparams)
transform_affine = AffineTransform(A, x0, b)

# Instantiate the Log parameter transform
sign = np.sign(x)
transform_log = LogTransform(sign)

# Instantiate the Archtanh parameter transform
midpoint = x + 0.1 * np.random.randn(nparams)
half_range = np.random.randint(1, 5, nparams)
low = midpoint - half_range
high = midpoint + half_range
transform_atanh = AtanhTransform(low, high)

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
    # Test arctanh transformation by manual calculation
    assert np.allclose(transform_atanh(x), np.arctanh((x - midpoint) / half_range))
    # Test split transformation by manual calculation
    assert np.allclose(
        transform_split(xlong), np.append(np.log(np.abs(x)), A @ (x - x0) + b)
    )
    # Test composed transformation by manual calculation
    assert np.allclose(transform_composed(x), A @ (np.log(np.abs(x)) - x0) + b)


def test_transform2():
    """These tests are to check if transfomations that are motivated by certain
    constraints really satisfy those constraints. For example, the log transformation
    should return positive values for positive input values and negative values for
    negative input values.
    """
    # Generate random parameter values for testing --- Don't make the range too large
    # so that the test won't be affected by numerical precision.
    test_values = np.random.uniform(-1e2, 1e2, (100, nparams))

    # Test if log transformation preserves the sign of the input values
    for val in test_values:
        test_val = transform_log.inverse_transform(val)
        assert np.all(np.sign(test_val) == sign)
    # Test if arctanh transformation really works within a range
    for val in test_values:
        # Invert transformation to see if all numbers are mapped within a range
        test_val = transform_atanh.inverse_transform(val)
        # We need to give some tolerance to the range because of the numerical precision
        assert np.all(test_val >= low - tol) and np.all(test_val <= high + tol)


def test_inverse_transform():
    """Test inverse transformation method."""
    # Test if affine inverse transformation gives back the original parameter values
    assert np.allclose(transform_affine.inverse_transform(transform_affine(x)), x)
    # Test if log inverse transformation gives back the original parameter values
    assert np.allclose(transform_log.inverse_transform(transform_log(x)), x)
    # Test if arctanh inverse transformation gives back the original parameter values
    assert np.allclose(transform_atanh.inverse_transform(transform_atanh(x)), x)
    # Test if split inverse transformation gives back the original parameter values
    assert np.allclose(transform_split.inverse_transform(transform_split(xlong)), xlong)
    # Test if composed inverse transformation gives back the original parameter values
    assert np.allclose(transform_composed.inverse_transform(transform_composed(x)), x)


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


def test_transform_builder():
    """Test transform_builder function."""
    # Dictionaries for the transformation classes
    transform_affine_dict = {
        "transform_type": "AffineTransform",
        "transform_args": {"a": A, "x0": x0, "b": b},
    }
    transform_log_dict = {
        "transform_type": "LogTransform",
        "transform_args": {"sign": sign},
    }
    transform_split_dict = {
        "transform_type": "SplitTransform",
        "transform_args": {
            "transform_list": [transform_log_dict, transform_affine_dict],
            "param_idx": param_idx,
        },
    }
    transform_composed_dict = {
        "transform_type": "ComposedTransform",
        "transform_args": {"transform_list": [transform_log_dict, transform_affine_dict]},
    }

    # Test if the transform_builder function returns the correct type of transformation
    # class
    assert isinstance(transform_builder(**transform_affine_dict), AffineTransform)
    assert isinstance(transform_builder(**transform_log_dict), LogTransform)
    assert isinstance(transform_builder(**transform_split_dict), SplitTransform)
    assert isinstance(transform_builder(**transform_composed_dict), ComposedTransform)

    # Test if the transform_builder function create the correct transformation class by
    # comparing the forward transformation
    assert np.allclose(transform_builder(**transform_affine_dict)(x), transform_affine(x))
    assert np.allclose(transform_builder(**transform_log_dict)(x), transform_log(x))
    assert np.allclose(
        transform_builder(**transform_split_dict)(xlong), transform_split(xlong)
    )
    assert np.allclose(
        transform_builder(**transform_composed_dict)(x), transform_composed(x)
    )


if __name__ == "__main__":
    test_transform()
    test_transform2()
    test_inverse_transform()
    test_func_wrapper()
    test_transform_builder()
