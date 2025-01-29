import re
from typing import Optional, Union
import numpy as np

# Dictionary containing information used for finite difference calculation.
# The "keys" key set a uniform naming scheme for the generated parameters and
# predictions sets dictionary. The "scale" key sets the multiplicative scaling
# factor for each term and it is closely related to the "keys". However, the
# first element of the "scale" list correspond to the predictions at the
# unperturbed parameters. The key "denominator" sets the multiplicative factor
# for h on the denominator of the finite difference formula.
finitediff_info = {
    "FD": {
        "keys": ["plus1h"],
        "scale": [-1.0, 1.0],
        "denominator": 1.0,
    },
    "FD2": {
        "keys": ["plus1h", "plus2h"],
        "scale": [-3.0, 4.0, -1.0],
        "denominator": 2.0,
    },
    "FD3": {
        "keys": ["plus1h", "plus2h", "plus4h"],
        "scale": [-21.0, 32.0, -12.0, 1.0],
        "denominator": 12.0,
    },
    "FD4": {
        "keys": ["plus1h", "plus2h", "plus4h", "plus8h"],
        "scale": [-315.0, 512.0, -224.0, 28.0, 1.0],
        "denominator": 168.0,
    },
    "CD": {
        "keys": ["minus1h", "plus1h"],
        "scale": [0.0, -1.0, 1.0],
        "denominator": 2.0,
    },
    "CD4": {
        "keys": ["minus2h", "minus1h", "plus1h", "plus2h"],
        "scale": [0.0, 1.0, -8.0, 8.0, -1.0],
        "denominator": 12.0,
    },
}
avail_method = list(finitediff_info)


class FiniteDifference:
    """
    Estimate the derivative of a model with respect to the parameters via
    finite difference numerical derivative.

    What this module does is 1) generate a set of perturbed parameters that
    will need to be evaluated and 2) use the model outputs evaluated at this
    set of parameters and estimate the derivative of the model.
    Note that this means the model needs to be evaluated outside of this class,
    and this class will use the predictions information to estimate the
    derivative.

    Currently, the available methods are:
    * "FD" - forward difference
    * "FD2" - 2-point forward difference
    * "FD3" - 3-point forward difference
    * "FD4" - 4-point forward difference
    * "CD" - central difference
    * "CD$" - 4-point central difference

    When generating the parameter set, the parameter values will be given as a
    dictionary with the following format (assuming we have a model with 2
    parameters and use central difference (CD) method):

    params_set = {
        "original": <original_parameter_values>,
        "param0-plus1h": <param>,
        "param0-minus1h": <param>,
        "param1-plus1h": <param>,
        "param1-minus1h": <param>,
    }

    Then, when estimating the derivative from the predictions set, the
    predictions should be given as a dictionary in the following format (also
    assuming a model with 2 parameters and using CD method):

    predictions_set = {
        "original": <predictions_at_original_param>,
        "param0-plus1h": <preds>,
        "param0-minus1h": <preds>,
        "param1-plus1h": <preds>,
        "param0-minus1h": <preds>,
    }

    Notes: The central difference formulae should not need predictions
    computation at the unperturbed parameters. However, we still require that
    key to make the interface uniform across different methods. One can just
    set the this predictions to a random array, but with the same length as the
    length of the predictions vector.

    :param params: Parameter values in which we want to estimate the
        derivative at.
    :type params: np.ndarray

    :param h: Finite difference step size. Can be a single number for all
        parameters or a list of numbers, each for different parameters.
    :type h: float or list

    :param method: A string that identifies the finite difference method.
        Needs to be one of the available methods.
    :type method: str
    """

    def __init__(
        self,
        params: np.ndarray,
        h: Optional[Union[float, list, np.ndarray]] = 0.1,
        method: Optional[str] = "FD",
    ):
        """
        Instantiate class to estimate derivative via finite difference.

        :param params: Parameter values at which we want to estimate the
            derivative.
        :type params: np.ndarray

        :param h: Finite difference step size. Can be a single number for all
            parameters or a list of numbers, each for different parameters.
        :type h: float or list

        :param method: A string that identifies the finite difference method.
            Needs to be one of the available methods.
        :type method: str
        """
        self.params = params
        self.nparams = len(self.params)

        self.h = h
        if isinstance(self.h, (float, int)):
            self.h = [self.h] * self.nparams
        elif isinstance(self.h, (list, np.ndarray)):
            assert len(self.h) == self.nparams, f"Please provide {self.nparams} h values"
        else:
            raise ValueError("Unknown format, input a float or an array")

        self.method = method.upper()
        assert self.method in finitediff_info, f"Available method: {avail_method}"

        # Directions of perturbations, which is just an identity matrix. Each
        # row dives the direction of perturbation for each parameter.
        self._V = np.eye(self.nparams)

        # Stored information to do finite difference
        self._fd_info = finitediff_info[self.method]

    def generate_params_set(self) -> dict:
        """
        Generate a set of parameter values that will be use to estimate the
        derivative.

        :returns: Parameters set
        :rtype: dict
        """

        params_set = {"original": self.params}
        for ii in range(self.nparams):
            param_key = f"param{ii}"  # Key to store the set for each parameter
            for direc_key in self._fd_info["keys"]:
                direction = int(re.findall("[\d]", direc_key)[0]) * self._V[ii]
                if "minus" in direc_key:
                    direction *= -1
                perturbation = self.h[ii] * direction
                params_set.update(
                    {f"{param_key}-{direc_key}": self.params + perturbation}
                )
        return params_set

    def estimate_derivative(self, preds_set: dict) -> np.ndarray:
        """
        Estimate the derivative given the set of predictions evaluated at the
        generated set of parameters.

        The format of ``preds_set`` needs to look like:
        predictions_set = {
            "original": <predictions_at_original_param>,
            "param0-plus1h": <preds>,
            "param0-minus1h": <preds>,
            "param1-plus1h": <preds>,
            "param0-minus1h": <preds>,
        }


        :param preds_set: A set of predictions
        :type preds: dict

        :returns: array (npreds, nparams,)
        :rtype: np.ndarray
        """
        # Retrieve predictions at the unperturbed parameters
        center = self._fd_info["scale"][0] * preds_set["original"]
        # Prepare a matrix to store the Jacobian
        npreds = len(center)
        jac = np.empty((npreds, self.nparams))

        for ii in range(self.nparams):
            param_key = f"param{ii}"
            # Numerator of the finite difference formula
            deriv_values = center.copy()
            for direc_key, scale in zip(
                self._fd_info["keys"], self._fd_info["scale"][1:]
            ):
                deriv_values += scale * preds_set[f"{param_key}-{direc_key}"]
            deriv_values /= self._fd_info["denominator"] * self.h[ii]
            jac[:, ii] = deriv_values
        return jac
