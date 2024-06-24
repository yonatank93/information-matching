"""This script contains models for SW that compute the target QoI. The target properties
included here are:
* Energy vs lattice parameter curve (ModelEnergyCurve)
* Equilibrium lattice and elastic constants (ModelLatticeElastics)
* Phonon dispersion curve (ModelPhononDispersion)
"""

import numpy as np

from kimmodel.EnergyLatconstDiamondSi import EnergyVsLatticeConstant
from kimmodel.ElasticConstantsCubic import ElasticConstantsCubic
from kimmodel.PhononDispersionCurveCubic import PhononDispersionCurveCubic

potential = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
param_names = ["A", "B", "sigma", "gamma", "lambda"]


class BaseModel:
    @staticmethod
    def transform(x):
        """Transform the parameters from the original parameterization to the working
        parameterization. In this case, the working parameterization is the logarithm of
        the original parameterization.
        """
        return np.log(x)

    @staticmethod
    def inverse_transform(x):
        """Transform the parameters from the working parameterization to the original
        parameterization.
        """
        return np.exp(x)

    def _convert_to_params_dict(self, x):
        """Convert the parameters from an array format to a dictionary format."""
        params_dict = {name: [[0], [value]] for name, value in zip(self.param_names, x)}
        return params_dict

    def _convert_to_params_array(self, x):
        """Convert the parameters from a dictionary format to an array format."""
        return np.array([x[name][1][0] for name in self.param_names])

    def residuals(self, x):
        """Compute the residuals of the model. Like the ``predictions`` method,
        The value corresponding to :math:`\Delta a = 0` will not be returned.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Model residual vector.
        """
        preds = self.predictions(x)
        return (self.data - preds) / self.std


class ModelEnergyCurve(BaseModel):
    """This class is used to encode the model that computes the energy vs.
    lattice constant curve.

    Parameters
    ----------
    data: np.ndarray
        Reference data.
    error: np.ndarray
        Error bars of the data.
    da_list: np.ndarray
        List of the change in lattice constants that are used to compute the predictions.
    """

    def __init__(self, data, error, da_list=np.arange(-0.5, 0.51, 0.05)):
        self._model = EnergyVsLatticeConstant(potential, da_list)

        # Index to remove point in the middle with zero value in the predictions
        npred_guess = len(da_list)
        idx_delete = np.where(np.isclose(da_list, 0.0))[0][0]
        self._idx = np.delete(np.arange(npred_guess), idx_delete)
        # Data and error bars
        self.data = data[self._idx]
        self.std = error[self._idx]
        # Parameters
        self.param_names = param_names
        self.best_fit = self.transform(
            self._convert_to_params_array(self._model.default_params)
        )

        # Dimensionality of the model
        self.npred = len(self.data)
        self.nparams = len(self.best_fit)

    def predictions(self, x):
        """Compute the predicted energy vs. lattice constant curve. The value
        at :math:`\Delta a = 0` will not be returned, since it will always be
        zero.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        params = self._convert_to_params_dict(self.inverse_transform(x))
        preds_dict = self._model.compute(**params)
        return self._convert_preds_dict(preds_dict)

    def _convert_preds_dict(self, preds_dict):
        """Convert the predictions dictionar to an array format."""
        e0 = preds_dict["ecoh"]["value"]
        elist = preds_dict["elist"]["value"]
        return (elist - e0)[self._idx]


class ModelLatticeElastics(BaseModel):
    """This class is used to encode the model that computes the lattice and elastic
    constants.

    Parameters
    ----------
    data: np.ndarray
        Reference data.
    error: np.ndarray
        Error bars of the data.
    """

    def __init__(self, data, error):
        self.data = data
        self.std = error

        self._model = ElasticConstantsCubic("Si", "diamond", potential)
        self.param_names = param_names
        self.best_fit = self.transform(
            self._convert_to_params_array(self._model.default_params)
        )
        self.nparams = len(self.best_fit)
        self.npred = len(data)

    def predictions(self, x):
        """Compute the predicted energy vs. lattice constant curve. The value
        at :math:`\Delta a = 0` will not be returned, since it will always be
        zero.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        params = self._convert_to_params_dict(self.inverse_transform(x))
        preds_dict = self._model.compute(**params)
        return self._convert_preds_dict(preds_dict)

    @staticmethod
    def _convert_preds_dict(preds_dict):
        """Convert the predictions dictionary to an array format."""
        preds_array = np.array(
            [preds_dict[key] for key in ["a", "ecoh", "c11", "c12", "c44"]]
        )
        return preds_array


class ModelPhononDispersion(BaseModel):
    """This class is used to encode the model that computes the phonon
    dispersion curve for diamond silicon.

    Parameters
    ----------
    data_dict: dict
        A dictionary containing the reference data, target error bars, and label
        information that is used to plot the phonon dispersion curve.
    a: float
        Lattice constant used to compute the phonon dispersion curve. The default value is
        obtained from Materials Project database.
    bandpath: str
        String of special point names defining the band path.
    npoints: int
        Number of points calculated per branch.
    branch_idx: list of int
        Indices that point to which branch to return. 0 correspond to the lowest branch.
    """

    element = "Si"
    lattice = "diamond"
    # a = 5.47  # Lattice constant in angstrom, from materials project
    # bandpath = "GXULGK"  # "GXWKGLUWLUX"

    def __init__(self, data_dict, a=5.47, bandpath="GXULGK", npoints=100, branch_idx=[0]):
        self._model = PhononDispersionCurveCubic(
            self.element, self.lattice, potential, a, bandpath, npoints
        )

        # Load stuffs corresponding to the model
        self.branch_idx = branch_idx
        self.data_with_zeros = data_dict["data"][:, :, branch_idx].flatten()
        self.std_with_zeros = data_dict["error"][:, :, branch_idx].flatten()

        # Some values are always zero (or near zero) in the calculation. So, we
        # need to exclude these quantities.
        self._idx_zero = np.where(self.data_with_zeros <= 1e-8)[0]
        self._idx_nonzero = np.where(self.data_with_zeros > 1e-8)[0]
        self.data = self.data_with_zeros[self._idx_nonzero]
        self.std = self.std_with_zeros[self._idx_nonzero]
        self.labels = data_dict["labels"]
        self.param_names = param_names
        self.best_fit = self.transform(
            self._convert_to_params_array(self._model.default_params)
        )

        # Dimensionality of the model
        self.npred = len(self.data)
        self.npred_with_zeros = len(self.data_with_zeros)
        self.nparams = len(self.best_fit)

    def predictions_with_zeros(self, x):
        """Compute the predicted energies in the phonon dispersion curve. Values
        with zero energies are still included in the output. Returns the energy
        array

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Model predictions with zeros included.
        """
        params = self._convert_to_params_dict(self.inverse_transform(x))
        preds_dict = self._model.compute(**params)
        return preds_dict["energies"][:, :, self.branch_idx].flatten()

    def predictions(self, x):
        """Compute the predicted energies in the phonon dispersion curve. Values
        with zero energies are omitted. Returns the energy array

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        preds = self.predictions_with_zeros(x)
        return preds[self._idx_nonzero]

    def residuals_with_zeros(self, x):
        """Compute the residuals of the model. Like ``predictions_with_zeros``
        method, values with zero energies are still included.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Residual vector.
        """
        preds = self.predictions_with_zeros(x)
        res = (self.data_with_zeros - preds) / self.std_with_zeros
        # Fix values that has zero error bars
        res[self._idx_zero] = 0.0
        return res
