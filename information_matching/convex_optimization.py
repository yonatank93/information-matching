"""This module contains the functions that we need to perform convex optimization and the
corresponding post-processings, such as get the non-zero weights from the convex
optimization output.
"""

import numpy as np
import cvxpy as cp

from .utils import eps


default_solver = dict(solver=cp.SDPA)


class ConvexOpt:
    """This class is for formulating and solving the convex optimization problem to find
    the indicator configuration. For the notation below, :math:`N` denotes the number of
    parameters and :math:`L` denotes the number of configurations.

    Parameters
    ----------
    fim_qoi: np.ndarray (N, N) or dict {"fim": np.ndarray(N, N), "scale": float}
        The FIM of the target quantities of interest. The keyword scale in the dictionary
        format specifies the muultiplicative factor for the FIM as a preconditioning step.
        In the dictionary format, this keyword is optional, with a default value of 1.0.
    fim_configs: dict
        Information about the FIM of the candidate configurations, written as a dictionary
        in one of the following format::

            fim_configs = {config_id_1: value_1, config_id_2: value_2, ...}

        where ``value_ii`` can be a ``np.ndarray(N, N)`` or a dictionary with the
        following format::

            value = {"fim": np.ndarray(N, N), "fim_scale": float, "weight_scale": float}

        The keywords ``fim_scale`` and ``weight_scale`` specify the multiplicative factor
        for the FIM and weight, respectively, to precondition the FIM and the weights to
        help with the optimization. These numbers are optional, with default value of 1.0.
    l1norm_obj: bool
        An option to explicitly use l1-norm for the objective function instead of just a
        sum. In theory, these are the same, given the non-negative constraint on the
        weights. However, the algorithm might explore the infeasible set, and the choice
        of objective function can affect the answer. Also note that using sum is faster.

    Notes
    -----
    The indicator configuration calculation done here will flatten the FIMs into vectors.
    Computationally, this is still the same as the original formulation.
    """

    def __init__(self, fim_qoi, fim_configs, l1norm_obj=False):
        # Obtain the target FIM of the QoI
        (
            self.fim_qoi_vec,
            self.scale_qoi,
            self._fim_shape,
        ) = self._get_target_fim_and_scale(fim_qoi)

        # Obtain the configuration FIMs
        self.config_ids = self._get_config_ids(fim_configs)
        self.nconfigs = len(self.config_ids)
        self.fim_configs_vec, self.scale_conf = self._get_config_fims_and_scale(
            fim_configs
        )
        # Get the scaling factor for the weights
        self.scale_weights = self._get_weight_scale(fim_configs)

        # Other settings
        self._l1norm_obj = l1norm_obj

        # Construct the problem
        self._construct_problem()
        self._result = None  # To initialize the result property

    def solve(self, solver=default_solver):
        """Solve the convex optimization problem.

        Parameters
        ----------
        solver: dict
            The dictionary containing the information of the solver that will
            be passed into ``cp.Problem().solve()``. The dictionary should
            contain the name of the solver and other keyword arguments for the
            solver. Default: ``{"solver"=cp.OSQP}``
        """
        self.problem.solve(**solver)

        # Store the result
        status = self.problem.status
        opt_wm = (self.wm.value).flatten()
        dual_wm = (self.constraints[0].dual_value).flatten()
        opt_val = self.problem.value
        fim_diff = self._difference_matrix(self.wm).value
        rel_error = np.linalg.norm(fim_diff) / np.linalg.norm(self.fim_qoi_vec)
        violation_psd = self.constraints[1].violation()

        result_dict = {
            "status": status,
            "wm": opt_wm,
            "dual_wm": dual_wm,
            "value": opt_val,
            "rel_error": rel_error,
            "violation": violation_psd,
        }
        self._result = result_dict

    @property
    def result(self):
        """Retrieve the result of the convex optimization as a dictionary.

        Returns
        -------
        dict
            The information we need for indicator configuration from the convex
            optimization. The keys of the dictionary are:

            * ``status``: status of the convex optimization.
            * ``wm``: the weights :math:`\bm w`.
            * ``dual_wm``: dual values of :math:`\bm w`, used to infer whether
              the weights are essentially ``zero''.
            * ``value``: the optimal value of the objective function.
            * ``rel_error``: relative error, calculated by taking ratio of the
              optimal value of the objective function to
              :math:\mathcal{I}^{EC}`.
        """
        return self._result

    @result.setter
    def result(self, value):
        self._result = value
        self.wm.value = self._result["wm"].reshape((-1, 1))

    def get_config_weights(self, zero_tol):
        """Get the non-zero weights from the current convex optimization result.

        Parameters
        ----------
        zero_tol: float
            Tolerance for the zero weights. This value will be compared to the
            dual values of the weights. This is typically set to be the same
            as the tolerance in the convex optimization solver.

        Returns
        -------
        configs_weights: dict
            A dictionary containing the information of the weights of each
            configuration. The keys are the configurations' identifiers.
        """
        # Interpret the convex optimization result and get index of non-zero
        # weights.
        idx_nonzero_wm = self._get_idx_nonzero_wm(zero_tol)
        return self._get_config_weights_from_idx(idx_nonzero_wm)

    @staticmethod
    def _get_target_fim_and_scale(fim_qoi):
        """From the FIM target input dictionary, extact the FIM matrix and the scale.
        Then, scale the FIM right away. Additionally, we will use the flatten matrix in
        the optimization, so we will do that too here.
        """
        if isinstance(fim_qoi, np.ndarray):
            # The argument is given as an array, so we don't need to do anything to get
            # the matrix. But, we need to set the scale
            fim_qoi_array = fim_qoi
            scale = 1.0
        elif isinstance(fim_qoi, dict):
            fim_qoi_array = fim_qoi["fim"]
            if "scale" not in fim_qoi:
                scale = 1.0
            else:
                scale = fim_qoi["scale"]
        else:
            raise ValueError("Unknown format, input dict(fim=..., scale=...)")

        # First, we will actually use the flatten FIM
        shape = fim_qoi_array.shape  # Just store this value for future use
        fim_qoi_vec = fim_qoi_array.flatten()
        # Apply the scaling preconditioning to the target FIM
        fim_qoi_array *= scale
        return fim_qoi_vec, scale, shape

    @staticmethod
    def _get_config_ids(fim_configs):
        """Retrieve the configuration identifiers, which are the keys to the fim_configs
        dictionary.
        """
        return list(fim_configs)

    def _get_config_fims_and_scale(self, fim_configs):
        """From the FIM configs input dictionary, retrieve the FIM values and the scaling
        factor. Set any default values as necessary. And scale the FIM right away.
        """
        fim_configs_vec = np.empty((self.nconfigs, np.prod(self._fim_shape)))
        scale_conf = np.empty(self.nconfigs)
        for ii, identifier in enumerate(self.config_ids):
            value = fim_configs[identifier]
            if isinstance(value, np.ndarray):
                # It is an array, so it must be the FIM matrix
                # Flatten and assign
                fim_configs_vec[ii] = value.flatten()
                # There is no scale information, so set to the default value
                scale_conf[ii] = 1.0
            elif isinstance(value, dict):
                # First, retrieve the scaling factor
                if "fim_scale" in value:
                    scale_conf[ii] = value["fim_scale"]
                else:
                    scale_conf[ii] = 1.0
                # Retrieve the FIM, flatten, scale, and assign
                fim_configs_vec[ii] = scale_conf[ii] * value["fim"].flatten()
            else:
                raise ValueError(
                    "Unknown input format, input dict(fim=..., fim_scale=...) "
                    + "for each configuration"
                )
        return fim_configs_vec, scale_conf

    def _get_weight_scale(self, fim_configs):
        """From the FIM_configs input dictionary, retrieve the weight scaling for each
        configuration.
        """
        scale_weights = np.empty(self.nconfigs)
        for ii, identifier in enumerate(self.config_ids):
            value = fim_configs[identifier]
            if isinstance(value, np.ndarray):
                # Use default value
                scale_weights[ii] = 1.0
            elif isinstance(value, dict):
                if "weight_scale" in value:
                    scale_weights[ii] = value["weight_scale"]
                else:
                    scale_weights[ii] = 1.0
            else:
                raise ValueError(
                    "Provide weight as 'weight_scale' key in fim_configs dictionary "
                    + "for each configuration"
                )
        return scale_weights

    def _construct_problem(self):
        """Formulate the convex optimization problem."""
        # Variable to optimize
        self.wm = cp.Variable((self.nconfigs, 1))
        # Objective - equivalent to minimizing the l1 norm due to constraint (1)
        self.objective = cp.Minimize(self._objective_fn())
        # Constraints: (1) the weights are positive and (2) the information from
        # the configurations needs to be at least as much as the information
        # needed by the target, imposing positive definiteness on the difference
        # between the 2 FIMs.
        self.constraints = [self.wm >= 0.0, self._difference_matrix(self.wm) >> 0]
        # Problem
        self.problem = cp.Problem(self.objective, self.constraints)

    def _objective_fn(self):
        """Objective function of the convex optimization."""
        # Apply weight scaling factor
        scaled_weights = cp.multiply(self.wm, self.scale_weights.reshape((-1, 1)))
        if self._l1norm_obj:
            return cp.norm(scaled_weights, p=1)
        else:
            return cp.sum(scaled_weights)

    def _difference_matrix(self, weights):
        """This function compute the matrix of difference between the weighted sum of the
        configuration FIMs and the target FIM.
        """
        # Weighted sum of configuration FIMs
        fim_configs = cp.sum(cp.multiply(self.fim_configs_vec, weights), axis=0)
        # Difference
        diff = fim_configs - self.fim_qoi_vec
        return cp.reshape(diff, self._fim_shape, order="C")

    def _get_unscaled_weights(self, weights):
        """Revert the scalling for the weights and get the weights that we will get if we
        have used unscaled FIMs.

        Parameters
        ----------
        weights: np.ndarray
            The scaled weight values.

        Returns
        -------
        dict
            Unscaled weight values. The weights corresponding to the energy and forces
            are stored in separate keys.
        """
        unscaled_wm = weights * self.scale_conf / self.scale_qoi
        return unscaled_wm

    def _get_config_weights_from_idx(self, idx):
        """Get configuration weights from the index of weights.

        Parameters
        ----------
        idx: array-like of int
            A list containing index to retrieve from the unscaled weight
            dictionary.

        Returns
        -------
        configs_weights: dict
            A dictionary containing the information of the weights of each
            configuration. The keys are the configurations' identifiers.
        """
        wm = self.result["wm"]
        unscaled_wm = self._get_unscaled_weights(wm)
        nonzero_weights = unscaled_wm[idx]
        nonzero_config_ids = np.array(self.config_ids)[idx]

        # Put the weight results into a dictionary
        weight_dict = {
            conf: val for conf, val in zip(nonzero_config_ids, nonzero_weights)
        }

        return weight_dict

    def _get_idx_nonzero_wm(self, zero_tol):
        """Get index of non-zero weights from the output of convex optimization.

        Parameters
        ----------
        zero_tol: float
            Tolerance for the zero weights. This value will be compared to the
            dual values of the weights. This is typically set to be the same
            as the tolerance in the convex optimization solver.

        Returns
        -------
        idx_nonzero_wm: list(int)
            List of index that points to the resulting weights from the convex
            optimization that points to the effectively non-zero weights.
        """
        # Interpret the convex optimization result
        weights = self.result["wm"]
        unscaled_weights = self._get_unscaled_weights(weights)
        dual_weights = self.result["dual_wm"]
        # Get non-zero weights
        idx_nonzero_from_val = np.where(unscaled_weights > eps**2)[0]
        idx_nonzero_from_dual = np.where(dual_weights < zero_tol)[0]
        idx_nonzero_wm = list(
            set(idx_nonzero_from_val).intersection(idx_nonzero_from_dual)
        )
        return idx_nonzero_wm
