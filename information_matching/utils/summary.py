import json

import numpy as np

from .misc import set_file


class Summary:
    """A class to handle exporting information during the calculation as a JSON
    file.

    In some sense, this class is a backend that export the results of the
    calculation. However, it will only export important, readable information.
    Other information can be retrieved from the files that are exported during
    the calculation.

    Parameters
    ----------
    filename: str or pathlib.Path
        File's name to which the information dictionary is exported/saved.
    """

    def __init__(self, filename):
        self.filename = set_file(filename)
        self.information = {}

        self.export()

    def export(self):
        """Export the information dictionary to a JSON file."""
        with open(self.filename, "w") as f:
            json.dump(self.information, f, indent=4)

    def update(self, info_value, info_type, **kwargs):
        """Update the information dictionary.

        Parameters
        ----------
        info_value: np.ndarray or list or dict
            Information from the output of some calculation.
        info_type: str
            What kind of calculation correspond to the information passed in.
            It can only be "convex optimization", "reduced configurations",
            "model training", or "new predictions".
        """
        if info_type == "convex optimization":
            self._update_cvxopt(info_value, info_type, **kwargs)
        elif info_type == "reduced configurations weights":
            self._update_configs(info_value, info_type, **kwargs)
        elif info_type == "model training":
            self._update_training(info_value, info_type, **kwargs)
        elif info_type == "new predictions":
            self._update_preds(info_value, info_type, **kwargs)

        self.export()

    def _update_cvxopt(self, info_value, info_type, **kwargs):
        """Update information dictionary from convex optimization result."""
        status = info_value["status"]
        value = info_value["value"]
        rel_error = info_value["rel_error"]
        violation = info_value["violation"]
        self.information.update(
            {
                info_type: {
                    "status": status,
                    "optimal value": value,
                    "relative error": rel_error,
                    "violation": violation,
                }
            }
        )

    def _update_configs(self, info_value, info_type, **kwargs):
        """Update information dictionary with the reduced configurations."""
        self.information.update({info_type: info_value})

    def _update_training(self, info_value, info_type, **kwargs):
        """Update information dictionary from result of model training."""
        gamma = kwargs["gamma"]
        opt_params = list(info_value[0])
        nparams = len(opt_params)
        converged = info_value[1]["msg"]
        residuals = info_value[1]["fvec"]
        opt_cost = np.sum(residuals**2) / 2
        npred = len(residuals) - nparams if gamma else len(residuals)
        penalty = gamma * np.sum(residuals[-nparams:] ** 2) / 2
        if gamma != 0.0:
            cost_without_penalty = np.sum(residuals[:-nparams] ** 2) / 2
        else:
            cost_without_penalty = np.sum(residuals**2) / 2
        cost_per_ndata = cost_without_penalty / npred

        self.information.update(
            {
                info_type: {
                    "gamma": gamma,
                    "bestfit": opt_params,
                    "converged": converged,
                    "cost": opt_cost,
                    "penalty": penalty,
                    "cost without penalty": cost_without_penalty,
                    "cost per residual": cost_per_ndata,
                }
            }
        )

    def _update_preds(self, info_value, info_type, **kwargs):
        """Update information dictionary with predictions evaluated at the new
        best fit.
        """
        self.information.update({info_type: list(info_value)})
