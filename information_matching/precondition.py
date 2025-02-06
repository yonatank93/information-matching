"""The purpose of this module is to provide several built-in options to precondition the
FIMs prior to running information-matching calculation.

What are some preconditioning options:
* Frobenius norm scaling - Scale each FIM by their Frobenius norm
* Maximum Frobenius norm scaling - Scale all FIMs uniformly by the largest Frobenius norm
"""

import numpy as np


avail_scale_type = ["frobenius", "max_frobenius"]


def preconditioner(fim, scale_type, pad=0.0):
    """Precondition the FIMs prior to running information-matching calculation.

    Parameters
    ----------
    fim : np.ndarray or dict
        The Fisher Information Matrix (FIM) for the target property or a dictionary of
        FIMs for different configurations.
    scale_type : str
        The type of scaling to apply to the FIMs. Available options are:
        * 'frobenius' - Scale each FIM by their Frobenius norm
        * 'max_frobenius' - Scale all FIMs uniformly by the largest Frobenius norm
    pad : float, optional
        The padding value to add to the Frobenius nor to avoid division by zero.
        Default is 0.0.

    Notes
    -----
    Scale type "max_frobenius" scales all FIMs uniformly, resulting in the same convex
    optimization result as the unscaled FIMs. However, scaling down the FIMs helps
    stabilize the optimization process by balancing their magnitudes.

    Scale type "frobenius" scales each FIM by its Frobenius norm, resulting in a
    non-uniform scaling. Unlike "max_frobenius", this affects the optimization result.
    Users should consider this option if they want to match the **information type** of
    each configuration with the target predictions.

    Returns
    -------
    dict
        The dictionary containing the FIMs and their scaling factors.
    """
    # Check the scale_type argument
    if scale_type.lower() not in avail_scale_type:
        raise ValueError(
            f"Unknown scaling type: {scale_type}, available type: {avail_scale_type}"
        )

    if isinstance(fim, np.ndarray):
        # This implies that the FIM is for target property.
        if scale_type.lower() in ["frobenius", "max_frobenius"]:
            scale = 1 / np.linalg.norm(fim)
        return {"fim": fim, "fim_scale": scale}
    elif isinstance(fim, dict):
        if "fim" in fim:
            # This also implies that the FIM is for target property.
            if "fim_scale" in fim:
                raise ValueError(
                    "The dictionary already contain scaling factor ('fim_scale' key)"
                )
            scale = 1 / np.linalg.norm(fim["fim"])
            return {"fim": fim["fim"], "fim_scale": scale}
        else:
            config_ids = list(fim)
            # Each item in the dictionary can still be an array or a dictionary. Let's
            # standardize the format to a dictionary
            for config_id in config_ids:
                if isinstance(fim[config_id], np.ndarray):
                    fim[config_id] = {"fim": fim[config_id]}
                elif isinstance(fim[config_id], dict):
                    # Let's also check if the dictionary contains scaling factor already.
                    # If so, raise an error since we don't want to overwrite the scaling
                    # factor.
                    if "fim_scale" in fim:
                        raise ValueError(
                            "The dictionary already contain scaling factor "
                            "('fim_scale' key)"
                        )
            # Compute the scaling factor
            if scale_type == "frobenius":
                # Each FIM has their own unique scaling factor, which is their inverse
                # Frobenius norm.
                scales = [1 / (np.linalg.norm(val["fim"]) + pad) for val in fim.values()]
            elif scale_type == "max_frobenius":
                max_norm = max([np.linalg.norm(val["fim"]) + pad for val in fim.values()])
                scales = [1 / max_norm] * len(config_ids)

            # Add the scaling values into the dictionary
            for ii, config_id in enumerate(config_ids):
                fim[config_id]["fim_scale"] = scales[ii]
            return fim
