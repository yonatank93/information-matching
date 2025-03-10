"""What's next? After the indicator configuration calculation has converged, we will
propagate the uncertainty from the training data, to the parameters, then to the target
QoIs.

Instead of running a long MCMC simulation to propagate the uncertainty from the data to
the parameters, we can do a quick check by using the FIM to estimate the uncertainty of
the parameters. This actually assumes that the distribution of the parameters is a
Gaussian, which is a good local approximation.

Additionally, we can do Monte Carlo sampling to propagate the uncertainty from the
parameters to the target QoIs. However, instead of running the full mapping, we can use
a linear approximation of this mapping to save time. This is done by using the Jacobian
of this mapping. This might be relevant since we also did similar thing, i.e., using the
FIM, in the indicator configuration calculation.
"""

from pathlib import Path
import pickle
import glob
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from corner import corner

from information_matching.fim import FIM_nd
from information_matching.utils import set_directory

# This directory, i.e., working directory
WORK_DIR = Path(__file__).absolute().parent
sys.path.append(str(WORK_DIR.parent / "models"))
# Import model to train
from model_train import (
    ModelTraining,
    WeightIndicatorConfiguration,
)
from model_target import ModelEnergyCurve


mpl.use("qt5agg")
np.random.seed(2022)

# Several flags to turn on/off some calculations/commands
plot_eigenvalues_eigenvectors = False
plot_parameter_samples = False
# plot_predictions_full = False
plot_predictions_lin = True
plot_fim_compare = True

# Setup some directories
# Directories that contain the results
error_scale = 1.0
RES_DIR = WORK_DIR / "results" / f"scale{int(error_scale)}"
# RES_DIR = WORK_DIR / "results" / "uniform"
result_files = glob.glob(str(RES_DIR / "[0-9]"))
final_step = len(result_files) - 1
STEP_DIR = RES_DIR / str(final_step)
# Directories to store the results of the calculations done here
LIN_DIR = set_directory(STEP_DIR / "linearized_uq")
PLOT_DIR = set_directory(STEP_DIR / "plots")

##########################################################################################
# Define the model
print("Defining the model")
# Defining the weight instance
config_dir = STEP_DIR / "reduced_configs"
weight_dict = pickle.load(open(STEP_DIR / "configs_weights.pkl", "rb"))
# See which QoI is needed
config_dir_E = None
config_dir_F = None
weight_E = None
weight_F = None
if len(glob.glob(str(config_dir) + "/energy/*.xyz")) > 0:
    config_dir_E = config_dir / "energy"
    weight_E = WeightIndicatorConfiguration(energy_weights_info=weight_dict["energy"])
if len(glob.glob(str(config_dir) + "/forces/*.xyz")) > 0:
    config_dir_F = config_dir / "forces"
    weight_F = WeightIndicatorConfiguration(forces_weights_info=weight_dict["forces"])
# Load the best fit
bestfit = np.loadtxt(STEP_DIR / "bestfit.txt")
# Parameter names
param_names = [
    r"$\log(A)$",
    r"$\log(B)$",
    r"$\log(\sigma)$",
    r"$\log(\gamma)$",
    r"$\log(\lambda)$",
]
# Instantiate the training model
model = ModelTraining(config_dir_E, config_dir_F, weight_E, weight_F)


##########################################################################################
# Check local sloppiness
print("Assess local sloppiness")
jac_train_file = LIN_DIR / "jacobian_train.npy"
if jac_train_file.exists():
    J = np.load(jac_train_file)
else:
    # Compute Jacobian matrix
    fim_inst = FIM_nd(model.residuals, step=0.1 * bestfit + 1e-4)
    J = fim_inst.Jacobian(bestfit)
    np.save(jac_train_file, J)
# FIM
fim = J.T @ J
# Eigenvalue decomposition of the FIM
lambdas, vecs = np.linalg.eigh(fim)

# Plot eigenvalues and eigenvectors
if plot_eigenvalues_eigenvectors:
    try:
        # Plot eigenvalues
        plt.figure()
        for lam in lambdas:
            plt.axhline(lam)
        plt.yscale("log")
        plt.ylabel(r"$\lambda$")
        plt.savefig(PLOT_DIR / "parameter_eigenvalues.png")

        # Plot eigenvectors
        plt.figure()
        plt.imshow(vecs[:, ::-1] ** 2, vmin=0.0, vmax=1.0)
        plt.yticks(range(model.nparams), param_names)
        plt.xticks([0, model.nparams - 1], ["stiff", "sloppy"])
        clb = plt.colorbar()
        clb.set_label("Participation factor")
        plt.savefig(PLOT_DIR / "parameter_eigenvectors.png")

        plt.show()
        plt.close("all")
    except Exception as e:
        plt.close("all")
        raise e


##########################################################################################
# Uncertainty propagation to the parameter space
# Generate normally distributed samples
print("Generate Gaussian samples")
T = 1.0
mean = bestfit
cov = np.linalg.pinv(fim / T)
nsamples = 4000
samples_file = LIN_DIR / f"samples_T{T:0.0e}.npy"
if samples_file.exists():
    samples = np.load(samples_file)
else:
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=nsamples)
    np.save(samples_file, samples)

# Plot the samples
if plot_parameter_samples:
    print("Red -> bestfit")
    try:
        bins = [len(np.histogram(ss, bins="auto")[0]) for ss in samples.T]

        fig = corner(
            samples,
            labels=param_names,
            range=np.tile([-8, 8], (model.nparams, 1)),
            bins=bins,
            hist_kwargs=dict(color="k", histtype="stepfilled"),
            **dict(
                color="k",
                plot_contours=False,
                plot_density=False,
                data_kwargs=dict(alpha=0.5),
            ),
        )
        axes = np.array(fig.axes).reshape((model.nparams, model.nparams))
        for row in range(model.nparams):
            axes[row, row].axvline(bestfit[row], ls="--", c="r")
            if row > 0:
                for col in range(row):
                    ax = axes[row, col]
                    ax.plot(bestfit[col], bestfit[row], ".", c="r")
        plt.savefig(PLOT_DIR / "linearized_parameter_samples.png")

        plt.show()
        plt.close()
    except Exception as e:
        plt.close()
        raise e


##########################################################################################
# Uncertainty propagation to target QoI
print("Uncertainty of energy vs. lattice constant")
# Define the model to compute the target QoI
dalist = np.arange(-0.5, 0.51, 0.05)
energycurve_data = np.load("energycurve_data.npz")
# Target precission
target_error_scale = error_scale
target_error = target_error_scale * energycurve_data["error"]
# target_error = np.ones_like(energycurve_data["error"]) * np.max(energycurve_data["error"])
# Instantiate the model to compute target QoI
model_target = ModelEnergyCurve(data=energycurve_data["data"], error=target_error)
# Data and target error bars
elist = np.insert(model_target.data, 10, 0.0)
eng_error = np.insert(model_target.std, 10, 0.0)

# # Uncomment the following to use the full mapping from the parameters to the target QoIs.
# # The mapping will take some time.
# # Map the parameter ensemble to the target QoIs.
# print("Get the ensemble of target QoI")
# samples_target_full_file = LIN_DIR / f"samples_target_full_T{T:0.0e}.npy"
# if samples_target_file.exists():
#     samples_target_full = np.load(samples_target_file)
# else:
#     samples_target_full = np.zeros((nsamples, model_target.npred))
#     for ns, sample in enumerate(samples):
#         if np.all(samples_target_full[ns] == 0.0):
#             try:
#                 target_preds = model_target.predictions(sample)
#                 samples_target_full[ns] = target_preds
#             except Exception:
#                 pass
#             np.save(samples_target_full_file, samples_target_full)

# # Plot uncertainty of the target qoi
# mean_target_full = np.insert(np.mean(samples_target_full, axis=0), 10, 0.0)
# error_target_full = np.insert(np.std(samples_target_full, axis=0), 10, 0.0)

# if plot_predictions_full:
#     try:
#         plt.figure()
#         plt.errorbar(dalist, elist, eng_error, c="k", lw=3, label="data")
#         plt.errorbar(dalist, mean_target_full, error_target_full, zorder=10, label="MCMC")
#         plt.xlabel(r"$a-a_0$")
#         plt.ylabel(r"$E-E_c$")
#         plt.legend()
#         plt.savefig(PLOT_DIR / "uncertainty_prediction_full.png")
#         plt.show()
#         plt.close()
#     except Exception as e:
#         plt.close()
#         raise e


# Using linear approximation of the target predictions
print("Linearized mapping to the target predictions")
jac_target_file = LIN_DIR / "jacobian_target.npy"
if jac_target_file.exists():
    J_target = np.load(jac_target_file)
else:
    # Compute the Jacobian for linear mapping
    # We want to take the Jacobian of the prediction mapping, without the error bar.
    # However, the module that we have used the residuals. We can just modify the result
    # taking the negative value and multiplying it with the error bars.
    Jac_target_inst = FIM_nd(model_target.residuals, step=0.1 * bestfit + 1e-4)
    J_target = -Jac_target_inst.Jacobian(bestfit)
    J_target *= model_target.std.reshape((-1, 1))
    np.save(jac_target_file, J_target)

# Perforn linear mapping of the samples
f0 = model_target.predictions(bestfit)
# Note: We can use sampling method, i.e., map the parameter ensemble to target qoi.
# Alternatively, we can use a closed-form formula to get the uncertainty of the target qoi.
# # Sampling approach
# samples_target_lin = np.array(
#     [f0 + J_target @ (sample - bestfit) for sample in samples]
# )
# mean_target_lin = np.insert(np.mean(samples_target_lin, axis=0), 10, 0.0)
# error_target_lin = np.insert(np.std(samples_target_lin, axis=0), 10, 0.0)
# Closed-form expression
mean_target_lin = np.insert(f0, 10, 0.0)
cov_target_lin = J_target @ cov @ J_target.T
error_target_lin = np.insert(np.sqrt(np.diag(cov_target_lin)), 10, 0.0)

# Plot uncertainty of the target qoi
if plot_predictions_lin:
    try:
        plt.figure()
        plt.fill_between(
            dalist,
            mean_target_lin - eng_error,
            mean_target_lin + eng_error,
            alpha=0.5,
            label="target",
        )
        plt.errorbar(
            dalist,
            mean_target_lin,
            error_target_lin,
            c="k",
            lw=3,
            capsize=6,
            capthick=3,
            label="linear",
        )
        plt.xlabel(r"$a-a_0$")
        plt.ylabel(r"$E-E_c$")
        plt.legend()
        plt.savefig(PLOT_DIR / "uncertainty_predictions_linear.png")
        plt.show()
        plt.close()
    except Exception as e:
        plt.close()
        raise e


##########################################################################################
# Compare the FIMs
J_target_std = J_target / model_target.std.reshape((-1, 1))
fim_target = J_target_std.T @ J_target_std
lam_conf = np.linalg.eigvalsh(fim / T)
lam_target = np.linalg.eigvalsh(fim_target)
# Plot FIMs comparison
if plot_fim_compare:
    try:
        plt.figure()
        plt.title("Configurations")
        plt.imshow(fim / T)
        plt.colorbar()
        plt.savefig(PLOT_DIR / "fim_configurations.png")

        plt.figure()
        plt.title("Target QoI")
        plt.imshow(fim_target)
        plt.colorbar()
        plt.savefig(PLOT_DIR / "fim_target.png")

        plt.figure()
        for l1, l2 in zip(lam_target, lam_conf):
            plt.plot([0, 1], [l1, l1], c="tab:blue", lw=3)
            plt.plot([1, 2], [l1, l2], ":", c="gray")
            plt.plot([2, 3], [l2, l2], "k", lw=3)
        # plt.axhline(1.0, ls="--")
        plt.yscale("log")
        plt.xticks([0.5, 2.5], ["Target QoI", "Configurations"])
        plt.ylabel("Eigenvalues")
        plt.savefig(PLOT_DIR / "compare_eigenvalues.png")

        plt.show()
        plt.close()
    except Exception as e:
        plt.close()
        raise e
