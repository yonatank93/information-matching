from pathlib import Path
from tqdm import tqdm

import numpy as np

import julia
from julia import Base, NumDiffTools
import models.model_IEEE14 as model

WORK_DIR = Path(__file__).absolute().parent

case = 14  # 14-bus case
FIM_DIR = WORK_DIR / "FIMs"
if not FIM_DIR.exists():
    FIM_DIR.mkdir(parents=True)


# Compute FIMs of the configuration
nparams = 2 * case
fim_configs_tensor = np.empty((case, nparams, nparams))
for idx in tqdm(range(case)):  # Iterate over bus index
    idx += 1  # Julia start index at 1

    def model_wrapper(x):
        """Wrapper model to compute predictions corresponding to bus ``idx``."""
        return model.h(x, PMU_idx=idx)

    # Compute Jacobian
    jac_file = FIM_DIR / f"jacobian_bus{idx}.npy"
    fim_file = FIM_DIR / f"fim_bus{idx}.csv"
    if jac_file.exists() and fim_file.exists():
        jac = np.load(jac_file)
        fim = np.loadtxt(fim_file, delimiter=",")
    else:
        jac = NumDiffTools.jacobian(
            model_wrapper, model.x0, abstol=1e-8, reltol=1e-8, h=0.1, t=2.0001, maxiters=8
        )
        fim = jac.T @ jac

        # Export
        np.save(jac_file, jac)
        np.savetxt(fim_file, fim, delimiter=",")

    fim_configs_tensor[idx - 1] = fim


# # Compute the FIM of combined configurations
# def model_wrapper(x):
#     return model.h(x, PMU_idx=np.arange(case) + 1)


# npreds = len(model_wrapper(model.x0))
# jmodel = Models.Model(npreds, model.nparams, model_wrapper, Base.Val(False))
# jac = jmodel.jacobian(model.x0)
# fim_comb = jac.T @ jac
