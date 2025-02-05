"""Collection of parameter transformations."""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class TransformBase(ABC):
    """
    Abstract base class for parameter transformations
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform parameters from the original parameterization to the
        transformed parameterization to work with.

        :param x: parameter values in the original parameterization
        :type x: np.ndarray

        :returns: parameter values in the transformed parameterization
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform parameters from the transformed parameterization to the
        original parameterization to work with.

        :param x: parameter values in the transformed parameterization
        :type x: np.ndarray

        :returns: parameter values in the original parameterization
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @property
    def jsonable_kwargs(self) -> dict:
        """
        Convert the keyword arguments used when initializing the class to a
        JSON serializable dictionary
        """
        kwargs = self._kwargs.copy()
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                kwargs[key] = val.tolist()
        return kwargs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)


class AffineTransform(TransformBase):
    """
    An affine transformation class :math:`y = A.x + b`.

    :param A: 2d array or a float
    :type A: float or np.ndarray

    :param b: 1d array or a float
    :type : float or np.ndarray

    :param Ainv: Inverse of A. If None is given, inverse will be computed by
        taking reciprocal of a if a is a float, otherwise using
        `np.linalg.pinv(a)`.
    :type: None or float or np.ndarray
    """

    def __init__(
        self,
        a: Optional[Union[float, np.ndarray]] = 1.0,
        b: Optional[Union[float, np.ndarray]] = 0.0,
        ainv: Optional[Union[None, float, np.ndarray]] = None,
    ):
        super().__init__(a=a, b=b, ainv=ainv)
        self.a = a
        if isinstance(self.a, (float, int)):
            # Use * multiplication operator
            self._mult_fn = self._scalar_mult
        elif isinstance(self.a, np.ndarray):
            # Use @ multiplication operator
            self._mult_fn = self._matrix_mult

        self.b = b

        if ainv is None:
            if isinstance(a, (float, int)):
                self.ainv = 1 / float(a)
            elif isinstance(a, np.ndarray):
                self.ainv = np.linalg.pinv(a)
        else:
            self.ainv = ainv

        super().__init__(a=self.a, b=self.b, ainv=self.ainv)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Perform parameter transformation to the transformed space.

        :param x: parameter values to transform
        :type x: np.ndarray

        :returns: parameter values in affine transformed space.
        :rtype: np.ndarray
        """
        return self._mult_fn(self.a, x) + self.b

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Invert the transformation back to the original space.

        :param x: parameter values to inverse transform
        :type x: np.ndarray

        :returns: parameter values in the original parameterization
        :rtype: np.ndarray
        """
        return self._mult_fn(self.ainv, (x - self.b))

    @staticmethod
    def _scalar_mult(a: float, b: float) -> float:
        """
        Multiplication function when the arguments are scalar.
        """
        return a * b

    @staticmethod
    def _matrix_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiplication function when the arguments are vectors or matrices.
        """
        return a @ b


class LogTransform(TransformBase):
    """
    An logarithmic transformation base e class :math:`y = log(|x|)`.
    This transformation preserves the sign, and the sign needs to be specified
    as an input argument.

    :param sign: sign of the parameter values, which is used to invert the
        transformation. Make sure that the element values are either 1 or -1
        only.
    :type sign: np.ndarray
    """

    def __init__(self, sign: np.ndarray):
        self.sign = sign
        super().__init__(sign=self.sign)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Perform parameter transformation to the transformed space.

        :param x: parameter values to transform
        :type x: np.ndarray

        :returns: parameter values in logarithmically transformed space
        :rtype: np.ndarray
        """
        return np.log(np.abs(x))

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Invert the transformation back to the original space.

        :param x: parameter values to inverse transform
        :type x: np.ndarray

        :returns: parameter values in the original parameterization
        :rtype: np.ndarray
        """
        return self.sign * np.exp(x)


avail_transform = {"affine": AffineTransform, "log": LogTransform}
