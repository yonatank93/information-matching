from pathlib import Path
from tqdm import tqdm

import numpy as np

from information_matching.fim.fim_jl import FIM_jl
from information_matching.utils import set_directory
import models.model_IEEE39 as model

WORK_DIR = Path(__file__).absolute().parent

case = 39  # 39-bus case
FIM_DIR = set_directory(WORK_DIR / "FIMs")

# Compute FIMs of the configuration
nparams = 2 * case
fim_configs_tensor = np.empty((case, nparams, nparams))
# Instantiate class to compute the FIM
fim_fn = FIM_jl(model.h, abstol=1e-8, reltol=1e-8, h=0.1, t=2.0001, maxiters=8)

for idx in tqdm(range(case)):  # Iterate over bus index
    idx += 1  # Julia start index at 1
    # Compute Jacobian
    jac_file = FIM_DIR / f"jacobian_bus{idx}.npy"
    fim_file = FIM_DIR / f"fim_bus{idx}.csv"
    if jac_file.exists() and fim_file.exists():
        jac = np.load(jac_file)
        fim = np.loadtxt(fim_file, delimiter=",")
    else:
        jac = fim_fn.compute_jacobian(model.x0, PMU_idx=idx)
        fim = jac.T @ jac

        # Export
        np.save(jac_file, jac)
        np.savetxt(fim_file, fim, delimiter=",")

    fim_configs_tensor[idx - 1] = fim


# # Compute the FIM of combined configurations
# jac = fim_fn.compute_jacobian(model.x0, PMU_idx=np.arange(case) + 1)
# fim = jac.T @ jac
