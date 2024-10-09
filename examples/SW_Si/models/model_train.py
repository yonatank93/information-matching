"""This script contains model for SW that is used to train the potential. In other words,
the model here computes the configuration energy and atomic forces.
"""

from pathlib import Path
import glob
import copy

import numpy as np

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.models.parameter_transform import LogParameterTransform

SWSI_DIR = Path(__file__).absolute().parents[1]
potential = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
param_names = ["A", "B", "sigma", "gamma", "lambda"]


class Configs:
    """Collection of variables related to the configurations, including paths and lists of
    configuration files.
    """

    alat_datapath = SWSI_DIR / "sw_si_training_dataset"
    path_energy = alat_datapath / "unperturbed"
    path_forces = alat_datapath / "perturbed"
    files_energy = glob.glob(str(path_energy / "*"))
    files_forces = glob.glob(str(path_forces / "*"))
    ids_energy = [Path(path).name for path in files_energy]
    ids_forces = [Path(path).name for path in files_forces]
    ids = np.append(ids_energy, ids_forces)
    nconfigs_energy = len(files_energy)
    nconfigs_forces = len(files_forces)


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
        qoi=["forces"],
    ):
        self.gamma = gamma

        # Parameter transformation
        self.param_names = param_names
        params_transform = LogParameterTransform(self.param_names)

        # Instantiate the model and set the potential
        self.kimmodel = KIMModel(model_name=potential, params_transform=params_transform)
        # Set the tunable parameters and the initial guess
        opt_params = {name: [["default"]] for name in self.param_names}
        self.kimmodel.set_opt_params(**opt_params)

        # Read the dataset and set the weight
        configs, self.nconfigs = self._read_dataset(config_path, weight)

        # Create calculator
        self.calculator, self.cas = self._create_calculator(configs, qoi)
        self.best_fit = copy.copy(self.calculator.get_opt_params())
        self.nparams = self.calculator.get_num_opt_params()
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

        Parameters
        ----------
        x: np.ndarray
            Parameters to evaluate.

        Returns
        -------
        np.ndarray
            Model predictions.
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
        c = self.loss._get_loss(x)
        penalty = np.linalg.norm(self.penalty(x)) ** 2 / 2
        return c + penalty


class ModelTraining:
    """This model class will handle combination of configurations for forces
    and configurations for energy.

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
        self.nparams = len(self.param_names)
        self.npred = int(np.sum([model.npred for model in self._models_list]))

        # Reference data
        self.data = np.concatenate([model.data for model in self._models_list])

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
        penalty = np.linalg.norm(self.penalty(x)) ** 2 / 2
        return c + penalty


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

        The key pointer would be the **file name** of the configuration.
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
    ``WeightIndicatorConfiguration``. Specifically, we want to have a nested dictionary,
    where one sub-dictionary contains the weights corresponding to configurations with
    the energy data and the other sub-dictionary with the force data.

    Parameters
    ----------
    cvx_weight: dict
        Weight dictionary from the output of cvxopt.

    Returns
    -------
    model_weights: dict
        Weight dictionary that is compatible with ``WeightIndicatorConfiguration``.
    """
    model_weights = {"energy": {}, "forces": {}}
    for key, val in cvx_weights.items():
        if key in Configs.ids_energy:
            model_weights["energy"].update({key: val})
        elif key in Configs.ids_forces:
            model_weights["forces"].update({key: val})
    return model_weights
