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
from model_target import ModelPhononDispersion


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
RES_DIR = WORK_DIR / "results" / "measurement"
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
# with open("phonondispersion_data_uniform.pkl", "rb") as f:
with open("phonondispersion_data.pkl", "rb") as f:
    phonondispersion_data = pickle.load(f)
model_target = ModelPhononDispersion(data_dict=phonondispersion_data)
nbranch = len(model_target.branch_idx)
target_data = model_target.data_with_zeros.reshape((-1, nbranch))
target_error = model_target.std_with_zeros.reshape((-1, nbranch))
# Other information for plotting the predictions
labels = model_target.labels
xcoords = labels[0]
labels_xcoords = labels[1]
bandpaths = labels[2]


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
#                 target_preds = model_target.predictions_with_zeros(sample)
#                 samples_target_full[ns] = target_preds
#             except Exception:
#                 pass
#             np.save(samples_target_full_file, samples_target_full)

# # Plot uncertainty of the target qoi
# mean_target_full = np.mean(samples_target_full, axis=0).reshape((-1, nbranch))
# error_target_full = np.std(samples_target_full, axis=0).reshape((-1, nbranch))

# if plot_predictions_full:
#     try:
#         plt.figure()
#         for jj, eng in enumerate(target_data.T):
#             mean_energy = mean_target_full[:, jj]
#             error_energy = error_target_full[:, jj]

#             # Plot the target error bars
#             err = target_error[:, jj]
#             plt.fill_between(
#                 xcoords,
#                 mean_energy + err,
#                 mean_energy - err,
#                 alpha=0.3,
#                 color="tab:blue",
#             )
#             plt.plot(xcoords, eng, "r", zorder=10)

#             # Plot the error bars from propagating the uncertainty
#             plt.errorbar(xcoords, mean_energy, error_energy, color="k", capsize=2)
#             plt.plot(xcoords, mean_energy, "k", zorder=10)
#         for x in labels_xcoords:
#             plt.axvline(x, ls="--")
#         plt.xlim(labels_xcoords[[0, -1]])
#         plt.xticks(labels_xcoords, bandpaths)

#         plt.ylabel("Energy (eV)")
#         plt.ylim(bottom=0.0)
#         plt.savefig(PLOT_DIR / "uncertainty_predictions_full.png")
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
    Jac_target_inst = FIM_nd(model_target.residuals_with_zeros, step=0.1 * bestfit + 1e-4)
    J_target = -Jac_target_inst.Jacobian(bestfit)
    J_target *= model_target.std_with_zeros.reshape((-1, 1))
    np.save(jac_target_file, J_target)

# Perforn linear mapping of the samples
f0 = model_target.predictions_with_zeros(bestfit)
# Note: We can use sampling method, i.e., map the parameter ensemble to target qoi.
# Alternatively, we can use a closed-form formula to get the uncertainty of the target qoi.
# # Sampling approach
# samples_target_lin = np.array([f0 + J_target @ (sample - bestfit) for sample in samples])
# mean_target_lin = np.mean(samples_target_lin, axis=0).reshape((-1, nbranch))
# error_target_lin = np.std(samples_target_lin, axis=0).reshape((-1, nbranch))
# Closed-form expression
mean_target_lin = f0.reshape((-1, nbranch))
cov_target_lin = J_target @ cov @ J_target.T
error_target_lin = np.sqrt(np.diag(cov_target_lin)).reshape((-1, nbranch))

if plot_predictions_lin:
    try:
        plt.figure()
        for jj, eng in enumerate(target_data.T):
            mean_energy = mean_target_lin[:, jj]
            error_energy = error_target_lin[:, jj]

            # Plot the target error bars
            err = target_error[:, jj]
            plt.fill_between(
                xcoords,
                mean_energy + err,
                mean_energy - err,
                alpha=0.3,
                color="tab:blue",
            )
            # plt.plot(xcoords, eng, "r", zorder=10)

            # Plot the error bars from linear approximation
            plt.errorbar(xcoords, mean_energy, error_energy, color="k", capsize=2)
            plt.plot(xcoords, mean_energy, "k", zorder=10)
        for x in labels_xcoords:
            plt.axvline(x, ls="--")
        plt.xlim(labels_xcoords[[0, -1]])
        plt.xticks(labels_xcoords, bandpaths)

        plt.ylabel("Energy (eV)")
        plt.ylim(bottom=0.0)
        plt.savefig(PLOT_DIR / "uncertainty_predictions_linear.png")
        plt.show()
        plt.close()
    except Exception as e:
        plt.close()
        raise e


##########################################################################################
# Compare the FIMs
J_target_std = J_target / model_target.std_with_zeros.reshape((-1, 1))
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
