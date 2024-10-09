"""This script contains models for SW MoS2 that are used to train the potential and
compute the target properties. The training quantity is the atomic forces and the target
property is energy change as a function of lattice compression/tension.
"""

from pathlib import Path
import glob

import numpy as np

# Use KLIFF for the scenario 1 - training
from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.models.parameter_transform import LogParameterTransform

# Import model for scenario 2 - target quantities
from .EnergyLatconstMoS2 import EnergyVsLatticeConstant

SWMoS2_DIR = Path(__file__).absolute().parents[1]
potential = "SW_MX2_WenShirodkarPlechac_2017_MoS__MO_201919462778_001"
# potential = "SW_MX2_KurniawanPetrieWilliams_2021_MoS__MO_677328661525_000"
param_names = ["A", "B", "p", "sigma", "lambda", "gamma"]
eps = np.finfo(float).eps


class Configs:
    """Collection of variables related to the configurations, including paths and lists of
    configuration files.
    """

    dataset_path = SWMoS2_DIR / "sw_mos2_training_dataset"
    dataset_files = sorted(glob.glob(str(dataset_path / "*")))
    ids = [Path(path).name for path in dataset_files]
    nconfigs = len(dataset_files)


class CustomParameterTransform(LogParameterTransform):
    """Modified log parameter transformation class for the specific model case used here."""

    def __init__(self, parnames):
        super().__init__(parnames)

    def transform(self, model_params):
        for name in self.param_names:
            p = model_params[name]
            p_val = np.asarray(p.value)

            if name == "lambda":
                if np.isnan(p_val[2]):
                    p_val[2] = eps
                else:
                    p_val[2] = np.abs(p_val[2])

            if name == "gamma":
                p_val[0] = p_val[1]
                p_val[2] = p_val[1]

            assert np.asarray(p_val).min() >= 0, (
                f"Cannot log transform parameter `{name}`, "
                f"got negative values: {p_val}"
            )
            p.value = np.log(p_val)

        return model_params

    def inverse_transform(self, model_params):
        for name in self.param_names:
            p = model_params[name]
            p_val = np.asarray(p.value)
            if name == "gamma":
                p_val[0] = p_val[1]
            p.value = np.exp(p_val)

        return model_params

    def __call__(self, model_params):
        return self.transform(model_params)


class ModelTrainingBase:
    """Base model to compute training quantities, e.g., energy and forces, using KLIFF.
    In some sense, this class is just a wrapper for KLIFF functions.

    Parameters
    ----------
    config_path: str or Path
        Path to the configuration files.
    gamma: float (optional)
        This controls the regularization term.
    weight:
        Instance of weights of the dataset.
    nprocs: int (optional)
        Number of parallel processes.
    qoi: list of "energy", "forces", "stress"
        Quantities to compute.
    """

    def __init__(
        self,
        config_path,
        gamma=0.0,
        weight=None,
        nprocs=1,
        qoi=["energy", "forces"],
    ):
        self.gamma = gamma

        # Parameter transformation
        self.param_names = param_names
        params_transform = CustomParameterTransform(self.param_names)

        # Instantiate the model and set the potential
        self.kimmodel = KIMModel(model_name=potential, params_transform=params_transform)
        # Set the tunable parameters and the initial guess
        opt_params = {
            name: [["default"], ["default"], ["default"]] for name in self.param_names
        }
        # Fix some parameters
        opt_params["lambda"][2] = [-7.4097433601858154e2, "fix"]
        opt_params["gamma"][0] = [3.0500530749075999e-1, "fix"]
        opt_params["gamma"][2] = [3.0500530749075999e-1, "fix"]
        self.kimmodel.set_opt_params(**opt_params)

        # Read the dataset and set the weight
        configs, self.nconfigs = self._read_dataset(config_path, weight)

        # Create calculator
        self.calculator, self.cas = self._create_calculator(configs, qoi)
        self.best_fit = self.calculator.get_opt_params()
        self.nparams = len(self.best_fit)
        # Reference data
        self.data = self._get_reference_data()
        self.npred = len(self.data)

        # Instantiate loss function
        self.loss = self._create_loss(nprocs)

    def _read_dataset(self, config_path, weight):
        """Read the dataset from the configuration files."""
        tset = Dataset(config_path, weight=weight)
        configs = tset.get_configs()
        nconfigs = len(configs)
        return configs, nconfigs

    def _create_calculator(self, configs, qoi):
        """Create a calculator."""
        kwargs = {"use_energy": False, "use_forces": False, "use_stress": False}
        if "energy" in qoi:
            kwargs["use_energy"] = True
        if "forces" in qoi:
            kwargs["use_forces"] = True
        if "stress" in qoi:
            kwargs["use_stress"] = True

        calc = Calculator(self.kimmodel)  # Calculator
        cas = calc.create(configs, **kwargs)  # Compute arguments
        return calc, cas

    def _get_reference_data(self):
        """Retrieve the reference data."""
        ref = []
        for ca in self.cas:
            current_ref = self.calculator.get_reference(ca)
            ref = np.concatenate((ref, current_ref))
        return ref

    def _create_loss(self, req_nprocs):
        """Create an instance of the loss function."""
        # To prevent from populating the entire machine, set a maximum allowed
        # number of parallel processes.
        max_nprocs = 20
        nprocs = min([req_nprocs, self.nconfigs, max_nprocs])
        if nprocs < req_nprocs:
            print("Change the number of parallel processes to", nprocs)

        loss = Loss(
            self.calculator, residual_data={"normalize_by_natoms": False}, nprocs=nprocs
        )
        return loss

    def predictions(self, x):
        """Compute the predictions at parameter x. Note that multiprocessing is
        NOT applied in this calculation.
        """
        # Update model parameters
        self.calculator.update_model_params(x)
        preds = []
        for ca in self.cas:
            self.calculator.compute(ca)
            preds = np.append(preds, self.calculator.get_prediction(ca))
        if self.npred == 1:
            return preds[0]
        else:
            return preds

    def penalty(self, x):
        """Compute the penalty term or the regularization of the cost function. The
        regularization term is related to the l2-norm of the parameters.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            The penalty in the cost function is related to the l2-norm of this array.
        """
        scale = np.sqrt((self.npred - self.nparams) / self.nparams)
        penalty = self.best_fit - x
        return scale * penalty * np.sqrt(2 * self.gamma)

    def residuals(self, x):
        """Compute the residuals of the model. Multiprocessing is used here.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Residual array.
        """
        res = self.loss._get_residual(x)
        if self.gamma > 0:
            penalty = self.penalty(x)
            return np.append(res, penalty)
        else:
            return res

    def cost(self, x):
        """Compute the cost value. Multiprocessing is used here."""
        c = self.loss._get_loss(x)
        penalty = self.gamma * np.linalg.norm(x) ** 2
        return c + penalty


class ModelTraining:
    """This model class will handle combination of configurations for forces
    and configurationa for energy.

    Parameters:
    -----------
    config_path_energy: Path (Optional)
        Path of the directory containing the unperturbed configurations.
    config_path_forces: Path (Optional)
        Path of the directory containing the perturbed configurations.
    weight_energy: Weight (Optional)
        Weight instance for energy configurations.
    weight_forces: Weight (Optional)
        Weight instance for forces configurations.
    gamma: float (Optional)
        Scale for the regularization term.
    nprocs: int (Optional)
        Number of parallel process to use for each type of QoI.
    """

    def __init__(
        self,
        config_path_energy=None,
        config_path_forces=None,
        weight_energy=None,
        weight_forces=None,
        gamma=0.0,
        nprocs=1,
    ):
        self.gamma = gamma
        self.use_energy = False
        self.use_forces = False
        self._models_list = []

        # Instantiate the model for each QoI
        if config_path_energy is not None:
            self._models_list.append(
                ModelTrainingBase(
                    config_path_energy, 0.0, weight_energy, nprocs, ["energy"]
                )
            )
            self.use_energy = True
        if config_path_forces is not None:
            self._models_list.append(
                ModelTrainingBase(
                    config_path_forces, 0.0, weight_forces, nprocs, ["forces"]
                )
            )
            self.use_forces = True

        # Other information
        self.param_names = self._models_list[0].param_names
        self.best_fit = self._models_list[0].best_fit

        # Dimensionality
        self.nparams = len(self.best_fit)
        self.npred = int(np.sum([model.npred for model in self._models_list]))

    def predictions(self, x):
        """Compute the predictions of the model.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        return np.concatenate([model.predictions(x) for model in self._models_list])

    def penalty(self, x):
        """Compute the penalty term or the regularization of the cost function. The
        regularization term is related to the l2-norm of the parameters.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            The penalty in the cost function is related to the l2-norm of this array.
        """
        scale = np.sqrt((self.npred - self.nparams) / self.nparams)
        penalty = self.best_fit - x
        return scale * penalty * np.sqrt(2 * self.gamma)

    def residuals(self, x):
        """Compute the residuals of the model. Multiprocessing is used here.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Residual array.
        """
        res = np.concatenate([model.residuals(x) for model in self._models_list])
        if self.gamma > 0:
            penalty = self.penalty(x)
            return np.append(res, penalty)
        else:
            return res

    def cost(self, x):
        """Compute the cost value. Multiprocessing is used here.

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        float
            Cost value.
        """
        c = np.sum([model.loss._get_loss(x) for model in self._models_list])
        if self.gamma == 0:
            return c
        else:
            penalty = np.linalg.norm(self.penalty(x)) ** 2 / 2
            return c + penalty


class ModelEnergyCurve:
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
        idx = np.arange(3)
        params_dict = {}
        # A
        params_dict.update({"A": [idx, x[0:3]]})
        # B
        params_dict.update({"B": [idx, x[3:6]]})
        # p
        params_dict.update({"p": [idx, x[6:9]]})
        # sigma
        params_dict.update({"sigma": [idx, x[9:12]]})
        # lambda
        params_dict.update({"lambda": [idx, np.array([x[12], x[13], 1.6e-322])]})
        # gamma
        params_dict.update({"gamma": [idx, np.array([x[14], x[14], x[14]])]})
        return params_dict

    def _convert_to_params_array(self, x):
        """Convert the parameters from a dictionary format to an array format."""
        params_array = []
        # A
        params_array = np.append(params_array, x["A"][1])
        # B
        params_array = np.append(params_array, x["B"][1])
        # p
        params_array = np.append(params_array, x["p"][1])
        # sigma
        params_array = np.append(params_array, x["sigma"][1])
        # lambda
        params_array = np.append(params_array, np.array(x["lambda"][1])[[0, 1]])
        # gamma
        params_array = np.append(params_array, x["gamma"][1][1])
        return params_array

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
        print(x)
        params = self._convert_to_params_dict(self.inverse_transform(x))
        preds_dict = self._model.compute(**params)
        return self._convert_preds_dict(preds_dict)

    def _convert_preds_dict(self, preds_dict):
        """Convert the predictions dictionar to an array format."""
        e0 = preds_dict["ecell"]["value"]
        elist = preds_dict["elist"]["value"]
        return (elist - e0)[self._idx]

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


class WeightIndicatorConfiguration(Weight):
    """Weight class that set the weights of the configurations from a dictionary. See
    below for the format of the dictionary

    Parameters
    ----------
    energy_weights_info, forces_weight_info: dict
        A dictionary containing the weights for the reduced configurations. The
        format is shown below ::

        weights_info = {
            pointer1: weight1,
            pointer2: weight2,
        }

        The key pointer would be the file name of the configuration.
    """

    def __init__(self, energy_weights_info=None, forces_weights_info=None):
        super().__init__()
        self.energy_weights_info = energy_weights_info
        self.forces_weights_info = forces_weights_info

    def compute_weight(self, config):
        identifier = Path(config.identifier).name
        if self.energy_weights_info is not None:
            self._energy_weight = np.sqrt(self.energy_weights_info[identifier])
        if self.forces_weights_info is not None:
            self._forces_weight = np.sqrt(self.forces_weights_info[identifier])


def convert_cvxopt_weights_to_model_weights(cvx_weights):
    """This function converts the weight dictionary from the output of
    ``ConvexOpt.get_config_weights`` to the format that is used by
    ``WeightIndicatorConfiguration``.

    Parameters
    ----------
    cvx_weight: dict
        Weight dictionary from the output of cvxopt.

    Returns
    -------
    model_weights: dict
        Weight dictionary that is compatible with ``WeightIndicatorConfiguration``.
    """
    model_weights = {"forces": cvx_weights}
    return model_weights
