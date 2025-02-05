import pytest
import numpy as np

from information_matching.precondition import preconditioner

np.random.seed(1)


# Dimensionality
nparams = np.random.randint(1, 10)
npreds = np.random.randint(nparams, 10)


def test_array():
    """Test the preconditioner function if an array is given."""
    J = np.random.randn(npreds, nparams)
    I = J.T @ J
    # Reference value
    ref = {"fim": I, "fim_scale": 1 / np.linalg.norm(I)}
    # Use preconditioner
    test_val = preconditioner(I, "frobenius")

    assert ref == test_val


def test_dict():
    """Test the preconditioner function if a dictionary is given."""
    J1 = np.random.randn(npreds, nparams)
    I1 = J1.T @ J1
    J2 = np.random.randn(npreds, nparams)
    I2 = J2.T @ J2
    norm1 = np.linalg.norm(I1)
    norm2 = np.linalg.norm(I2)

    # If the dictionary value is an array
    dict1 = {0: I1, 1: I2}
    # Reference value
    ref1 = {
        0: {"fim": I1, "fim_scale": 1 / norm1},
        1: {"fim": I2, "fim_scale": 1 / norm2},
    }
    # Use preconditioner
    test_val = preconditioner(dict1, "frobenius")
    assert ref1 == test_val

    # If the dictionary value is a dictionary
    dict2 = {0: {"fim": I1}, 1: {"fim": I2}}
    # Reference value should be the same
    # Use preconditioner
    test_val = preconditioner(dict2, "frobenius")
    assert ref1 == test_val

    # Test using "max_frobenius" scaling type
    test_val = preconditioner(dict1, "max_frobenius")
    test_scale = np.array([val["fim_scale"] for val in test_val.values()])
    assert np.allclose(test_scale, 1 / max([norm1, norm2]))


def test_exception():
    """Test if exception is raised when the input is invalid."""
    J1 = np.random.randn(npreds, nparams)
    I1 = J1.T @ J1
    J2 = np.random.randn(npreds, nparams)
    I2 = J2.T @ J2
    norm1 = np.linalg.norm(I1)
    norm2 = np.linalg.norm(I2)
    # Test if exception is raised when fim_scale exists
    with pytest.raises(ValueError):
        dict2 = {"fim": I1, "fim_scale": 1 / norm1}
        preconditioner(dict2, "frobenius")
    with pytest.raises(ValueError):
        dict3 = {0: {"fim": I1}, 1: {"fim": I2, "fim_scale": 1 / norm2}}
        preconditioner(dict2, "frobenius")


if __name__ == "__main__":
    test_array()
    test_dict()
    test_exception()
