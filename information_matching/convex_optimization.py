"""This module contains the functions that we need to perform convex optimization in the
information-matching schema and the corresponding post-processings, such as get the
non-zero weights from the convex optimization output.
"""

from copy import deepcopy
import warnings
import numpy as np
import cvxpy as cp

default_kwargs = dict(solver=cp.SDPA)
eps = np.finfo(float).eps


def default_obj_fn(x):
    return cp.sum(x)


def obj_fn_l1_norm(x):
    return cp.norm(x, p=1)


class ConvexOpt:
    """This class is for formulating and solving the convex optimization problem to find
    the information-matching. For the notation below, :math:`N` denotes the number of
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
    weight_upper_bound: float or np.ndarray
        The allowed upper bound for the optimal weights. This can be a single value,
        which will be broadcasted to all configurations, or a list of values for each
        configuration.
    l1norm_obj: bool (Deprecated)
        An option to explicitly use l1-norm for the objective function instead of just a
        sum. In theory, these are the same, given the non-negative constraint on the
        weights. However, the algorithm might explore the infeasible set, and the choice
        of objective function can affect the answer. Also note that using sum is faster.
    obj_fn: callable obj_fn(x, **obj_fn_kwargs)
        The objective function for weights minimization. The function should be convex
        and derived from CVXPY. The default is the sum of the weights, which is the same
        as l1-norm when weights are non-negative.
    obj_fn_kwargs: dict
        Additional keyword arguments to be passed to the objective function.

    Notes
    -----
    - The information-matching calculation done here flattens the FIMs into vectors.
      Computationally, this is still the same as the original formulation.
    - The weight_scale keyword in the fim_configs argument can be used to enforce a
      sparser solution. When the weights are scaled down, the solver tends to set them to
      zero, effectively approximating an :math:`\ell_0`-norm optimization.
    - A motivation for setting upper bounds on the weights is if there is a limitation in
      the feasible accuracy of reference data collection for a given configuration.
    - The choice of the objective function can affect the solution. For example, the
      default objective function, the l1-norm minimization, encourages sparsity in the
      solution. An alternative is to use l2-norm objective function, which doesn't
      encourage sparsity.
    """

    def __init__(
        self,
        fim_qoi,
        fim_configs,
        weight_upper_bound=None,
        l1norm_obj=False,
        obj_fn=None,
        obj_fn_kwargs=None,
    ):
        # Obtain the target FIM of the QoI
        self._fim_shape = self._get_fim_shape(fim_qoi)
        self.fim_qoi_vec, self.scale_qoi = self._get_target_fim_and_scale(fim_qoi)

        # Obtain the configuration FIMs
        self.config_ids = self._get_config_ids(fim_configs)
        self.nconfigs = len(self.config_ids)
        self.fim_configs_vec, self.scale_conf = self._get_config_fims_and_scale(
            fim_configs
        )
        # Get the scaling factor for the weights
        self.scale_weights = self._get_weight_scale(fim_configs)
        # Scale the upper bound for the weights
        self._weight_upper_bound = self._compute_scaled_weight_upper_bound(
            weight_upper_bound
        )

        # Objective function
        self._l1norm_obj = l1norm_obj
        if self._l1norm_obj:
            # Depracated
            warnings.warn(
                "The argument '_l1norm_obj' is deprecated "
                "and will be removed in a future release."
                "Please use the argument `obj_fn`.",
                DeprecationWarning,
                stacklevel=2,  # Ensures the warning points to the caller
            )
        if obj_fn is None:
            if self._l1norm_obj:
                self._obj_fn = obj_fn_l1_norm
            else:
                self._obj_fn = default_obj_fn
        else:
            self._obj_fn = obj_fn
        if obj_fn_kwargs is None:
            self._obj_fn_kwargs = {}
        else:
            self._obj_fn_kwargs = obj_fn_kwargs
        # Check convexity of the objective function
        if not self._check_obj_fn():
            raise ValueError("Objective function doesn't seem to be convex.")

        # Construct the problem
        self._construct_problem()
        self._result = None  # To initialize the result property

    def solve(self, **kwargs):
        """Solve the convex optimization problem.

        Parameters
        ----------
        kwargs: dict
            Additional keyword arguments to be passed into ``cp.Problem().solve()``.
            Default: ``{"solver"=cp.SDPA}``
        """
        if kwargs == {} or kwargs is None:
            kwargs = default_kwargs
        self.problem.solve(**kwargs)

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
            The information we need for information-matching from the convex
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

    def get_config_weights(self, zero_tol=1e-4, zero_tol_dual=1e-4):
        """Get the non-zero weights from the current convex optimization result.

        Parameters
        ----------
        zero_tol: float
            Tolerance for the zero weights. The weight value below this tolerance is
            treated as zero. This is typically set to be the same as the tolerance in
            the convex optimization solver.

        zero_tol: float
            Tolerance for the zero weights using the dual values of the weights for the
            positivity constraint. If the dual value is smaller than this tolerance,
            the weight is treated as zero. This is typically set to be the same as the
            tolerance in the convex optimization solver.

        Returns
        -------
        configs_weights: dict
            A dictionary containing the information of the weights of each
            configuration. The keys are the configurations' identifiers.
        """
        # Interpret the convex optimization result and get index of non-zero
        # weights.
        idx_nonzero_wm = self._get_idx_nonzero_wm(zero_tol, zero_tol_dual)
        return self._get_config_weights_from_idx(idx_nonzero_wm)

    @staticmethod
    def _get_fim_shape(fim_qoi):
        """Get the shape of the FIM."""
        if isinstance(fim_qoi, np.ndarray):
            # The argument is given as an array, so we don't need to do anything to get
            # the matrix. But, we need to set the scale
            fim_qoi_array = fim_qoi
        elif isinstance(fim_qoi, dict):
            fim_qoi_array = fim_qoi["fim"]
        else:
            raise ValueError("Unknown format, input dict(fim=..., scale=...)")
        return fim_qoi_array.shape

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
            if "fim_scale" not in fim_qoi:
                scale = 1.0
            else:
                scale = fim_qoi["fim_scale"]
        else:
            raise ValueError("Unknown format, input dict(fim=..., scale=...)")

        # First, we will actually use the flatten FIM
        fim_qoi_vec = fim_qoi_array.flatten()
        # Apply the scaling preconditioning to the target FIM
        fim_qoi_vec *= scale
        return fim_qoi_vec, scale

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

    def _compute_scaled_weight_upper_bound(self, weight_upper_bound):
        """Compute the scaled weight upper bound, so that it won't be affected with the
        choice of FIM scaling factors.
        """
        if weight_upper_bound in [None, np.inf, "inf"]:
            return None
        else:
            if np.isscalar(weight_upper_bound):
                # If a scalar is given, broadcast it to all configurations
                weight_upper_bound = weight_upper_bound * np.ones(self.nconfigs)
            else:
                # If a list is given, check if the length is correct
                if len(weight_upper_bound) != self.nconfigs:
                    raise ValueError(
                        "The length of weight_upper_bound should be the same as the "
                        "number of configurations"
                    )
            # If the length is correct, convert it to an array
            weight_upper_bound = np.array(weight_upper_bound)

            # Scale the upper bounds
            scaled_weight_upper_bound = (
                weight_upper_bound * self.scale_qoi / self.scale_conf
            )
            return scaled_weight_upper_bound.reshape((-1, 1))

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
        # Add an upper bound on the weights
        if self._weight_upper_bound is not None:
            self.constraints.append(self.wm <= self._weight_upper_bound)
        # Problem
        self.problem = cp.Problem(self.objective, self.constraints)

    def _check_obj_fn(self):
        """Check if the objective function is convex, which is a requirement for this
        calculation.
        """
        x = cp.Variable()
        test = self._obj_fn(x, **self._obj_fn_kwargs)
        return test.is_convex()

    def _objective_fn(self):
        """Objective function of the convex optimization."""
        # Apply weight scaling factor
        scaled_weights = cp.multiply(self.wm, self.scale_weights.reshape((-1, 1)))
        return self._obj_fn(scaled_weights, **self._obj_fn_kwargs)

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

    def _get_idx_nonzero_wm(self, zero_tol, zero_tol_dual):
        """Get index of non-zero weights from the output of convex optimization.

        Parameters
        ----------
        zero_tol: float
            Tolerance for the zero weights. The weight value below this tolerance is
            treated as zero. This is typically set to be the same as the tolerance in
            the convex optimization solver.

        zero_tol: float
            Tolerance for the zero weights using the dual values of the weights for the
            positivity constraint. If the dual value is smaller than this tolerance,
            the weight is treated as zero. This is typically set to be the same as the
            tolerance in the convex optimization solver.

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
        idx_nonzero_from_val = np.where(unscaled_weights > zero_tol)[0]
        idx_nonzero_from_dual = np.where(dual_weights < zero_tol_dual)[0]
        idx_nonzero_wm = list(
            set(idx_nonzero_from_val).intersection(idx_nonzero_from_dual)
        )
        return idx_nonzero_wm


def compare_weights(old_weights, current_weights):
    """Compare the weights from two different convex optimization iterations and return
    the new optimal weights. For each configuration, the new optimal weight is the larger
    of the two weights.

    Parameters
    ----------
    old_weights: dict
        The weights from the previous convex optimization iteration.
    current_weights: dict

    Returns
    -------
    new_weights: dict
        The new optimal weights.
    """
    new_weights = deepcopy(old_weights)
    for name, weight in current_weights.items():
        if name in new_weights:
            # If the configuration is already in the dictionary, update the weight to the
            # larger of the two
            new_weights[name] = max([weight, new_weights[name]])
        else:
            # If the configuration is not in the dictionary, add the new weight
            new_weights.update({name: weight})
    return new_weights
