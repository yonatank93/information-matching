"""In this script, I want to look at how the parameters vary over the iterations of
indicator configuration calculation. We take the parameters found at the end of each
iteration.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from information_matching.sampling_utils import mser


argv = sys.argv
dirname = argv[1]
print(argv)

FILE_DIR = Path(__file__).absolute().parent
RES_DIR = FILE_DIR / "results" / dirname

# Parameters over each iteration
step = 0
while True:
    params_file = RES_DIR / str(step) / "bestfit.txt"
    if params_file.exists():
        params = np.loadtxt(params_file)
        if step == 0:  # Initialize the array to store the parameters
            params_list = params
        else:
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
    plt.plot(np.arange(step), par_list - params_list[0, ii])
plt.xlabel("Step")
plt.ylabel("Relative difference compared to initial guess")

# After we run for several iterations, we can estimate the burn-in period.
n = len(params_list)
burnin_list = np.array(
    [mser(par_list, dmin=1, dstep=1, dmax=n - 1) - 1 for par_list in params_list.T]
)
print(f"Burn-in for {max(burnin_list)} steps")

# FInalize
plt.show()
