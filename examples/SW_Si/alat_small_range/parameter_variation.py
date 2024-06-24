"""In this script, I want to look at how the parameters vary over the iterations of
indicator configuration calculation. We take the parameters found at the end of each
iteration.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from information_matching.mcmc_utils import mser

# This directory, i.e., working directory
WORK_DIR = Path(__file__).absolute().parent
sys.path.append(str(WORK_DIR.parent / "models"))
from model_target import ModelEnergyCurve


argv = sys.argv
dirname = argv[1]
print(argv)

RES_DIR = WORK_DIR / "results" / dirname

# Initial parameter values
energycurve_data = np.load("energycurve_data.npz")
model_target = ModelEnergyCurve(
    data=energycurve_data["data"], error=energycurve_data["error"]
)
params_list = model_target.best_fit

# Parameters over each iteration
step = 0
while True:
    params_file = RES_DIR / str(step) / "bestfit.txt"
    if params_file.exists():
        params = np.loadtxt(params_file)
        params_list = np.row_stack((params_list, params))
        step += 1
    else:
        print(step, " steps are completed")
        break

# Plot the parameter variation
# Embedding function
plt.figure()
plt.title("Variation of parameters")
for ii, par_list in enumerate(params_list.T):
    plt.plot(np.arange(-1, step), par_list - params_list[0, ii])
plt.xlabel("Step (step -1 is for before training)")
plt.ylabel("Relative difference compared to initial guess")

# After we run for several iterations, we can estimate the burn-in period.
n = len(params_list)
burnin_list = np.array(
    [mser(par_list, dmin=1, dstep=1, dmax=n - 1) - 1 for par_list in params_list.T]
)
print(f"Burn-in for {max(burnin_list)} steps")

# FInalize
plt.show()
