"""Unit tests for the Transform module for parameter transformation."""

import numpy as np
from information_matching.transform import (
    AffineTransform,
    LogTransform,
    CombinedTransform,
)

np.random.seed(1)

# Test parameter value
x = np.random.randn(3)

# Instantiate the Affine parameter transform
A = np.random.randn(3, 3)
b = np.random.randn(3)
transform_affine = AffineTransform(A, b)

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
    assert np.allclose(transform_affine(x), A @ x + b)
    # Test log transformation by manual calculation
    assert np.allclose(transform_log(x), np.log(np.abs(x)))
    # Test combined transformation by manual calculation
    assert np.allclose(transform_combined(xlong), np.append(A @ x + b, np.log(np.abs(x))))


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


if __name__ == "__main__":
    test_affine_transform()
    test_inverse_transform()
