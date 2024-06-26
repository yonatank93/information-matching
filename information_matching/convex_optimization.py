"""This module contains the functions that we need to perform convex optimization and the
corresponding post-processings, such as get the non-zero weights from the convex
optimization output.
"""

from copy import copy

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
    fim_qoi: np.ndarray (N, N)
        The FIM of the target quantities of interest.
    fim_conf: np.ndarray (L, N, N)
        A 3d-array containing the FIMs of the training configurations.
    config_ids: list (L,)
        A list of strings that are used as identifiers of the configurations. The order of
        this list needs to be the same as the order of ``fim_conf``.
    norm: dict ``{"qoi": ..., "configs": ..., "weights": ...}``
        Additional normalization constant to use in the calculation, applied by dividing
        the corresponding quantities with this constant. User can provide the
        normalization constant for the target FIM (qoi), configuration FIMs (configs), or
        the weights being optimized. The orders of the normalization arrays need to be the
        same as that of ``config_ids``. Default: 1.0 for all quantities.
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

    def __init__(self, fim_qoi, fim_conf, config_ids, norm=None, l1norm_obj=False):
        self.config_ids = config_ids
        self.nconfigs = len(self.config_ids)
        self._l1norm_obj = l1norm_obj

        # Deal with the FIM arguments. We store the flatten the FIMs, which speeds up the
        # calculation.
        self._fim_shape = fim_qoi.shape  # Store the shape to reshape the FIM back
        self.fim_qoi_vec = copy(fim_qoi).flatten()
        self.fim_configs_vec = np.array([fconf.flatten() for fconf in fim_conf])

        # Normalize the FIM and get the normalization constants
        # Normalizing the FIM can make the optimization more stable
        # The weight norm is used to scale the weights during the optimization
        self.norm_qoi, self.norm_conf, self.norm_weights = self._apply_norm(norm)

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

    def _apply_norm(self, norm):
        """Apply the normalization constants. This action includes checking the values
        and scaling the FIMs. The default value is 1.0 for all quantities.
        """
        if norm is None:
            norm = {}

        # Target FIM
        if "qoi" in norm:
            norm_qoi = norm.pop("qoi")
        else:
            norm_qoi = 1.0

        # Configuration FIMs
        if "configs" in norm:
            norm_conf = norm.pop("configs")
        else:
            norm_conf = 1.0
        if isinstance(norm_conf, float):
            norm_conf = np.ones(self.nconfigs) * norm_conf
        else:
            msg = f"Please provide {self.nconfigs} values for the configs norm"
            assert len(norm_conf) == self.nconfigs, msg
            norm_conf = np.array(norm_conf)

        # Configuration FIMs
        if "weights" in norm:
            norm_wm = norm.pop("weights")
        else:
            norm_wm = 1.0
        if isinstance(norm_wm, float):
            norm_wm = np.ones(self.nconfigs) * norm_wm
        else:
            msg = f"Please provide {self.nconfigs} values for the weights norm"
            assert len(norm_wm) == self.nconfigs, msg
            norm_wm = np.array(norm_wm)

        # Finally, we need to scale the FIMs
        self.fim_qoi_vec /= norm_qoi
        self.fim_configs_vec /= norm_conf.reshape((-1, 1))

        return norm_qoi, norm_conf, norm_wm

    def _objective_fn(self):
        """Objective function of the convex optimization."""
        if self._l1norm_obj:
            return cp.norm(self.wm / self.norm_weights.reshape((-1, 1)), p=1)
        else:
            return cp.sum(self.wm / self.norm_weights.reshape((-1, 1)))

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
        unscaled_wm = weights * self.norm_qoi / self.norm_conf
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
