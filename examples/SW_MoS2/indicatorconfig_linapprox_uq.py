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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from corner import corner

from information_matching.fim.fim_jl import FIM_jl
from information_matching.utils import set_directory, tol

from models.models import (
    ModelTraining,
    ModelEnergyCurve,
    WeightIndicatorConfiguration,
)


mpl.use("qt5agg")
np.random.seed(2022)

# This directory, i.e., working directory
WORK_DIR = Path(__file__).absolute().parent
DATA_DIR = WORK_DIR / "data"

# Several flags to turn on/off some calculations/commands
plot_eigenvalues_eigenvectors = False
plot_parameter_samples = False
# plot_predictions_full = False
plot_predictions_lin = True
plot_fim_compare = True

# Setup some directories
error_str = "mingjian"  # Indicate which target error to use
start_str = "mingjian"  # Indicate which starting parameters to use
suffix = f"error_{error_str}_start_{start_str}"
RES_DIR = WORK_DIR / "results" / suffix

# Notes for the error_str variable:
# * If "mingjian", use data predicted by Mingjian's model and the target error is 10% of
#   the predictions.
# * If "yonatan", use data predicted by Yonatan's model and target error is 10% of the
#   predictions.
# * If "ips", use data predicted by Mingjian's model and target error is the standard
#   deviation obtained using different IPs, as presented in Mingjian's paper.
# Notes for start_str: To change the starting point, edit models/models.py file.
# Directories that contain the results

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
weight_F = WeightIndicatorConfiguration(forces_weights_info=weight_dict["forces"])
# Load the best fit
bestfit = np.loadtxt(STEP_DIR / "bestfit.txt")
# Parameter names
param_names = [
    r"$\log(A_{Mo-Mo})$",
    r"$\log(A_{Mo-S})$",
    r"$\log(A_{S-S})$",
    r"$\log(B_{Mo-Mo})$",
    r"$\log(B_{Mo-S})$",
    r"$\log(B_{S-S})$",
    r"$\log(p_{Mo-Mo})$",
    r"$\log(p_{Mo-S})$",
    r"$\log(p_{S-S})$",
    r"$\log(\sigma_{Mo-Mo})$",
    r"$\log(\sigma_{Mo-S})$",
    r"$\log(\sigma_{S-S})$",
    r"$\log(\lambda_{S-Mo-S})$",
    r"$\log(\lambda_{Mo-S-Mo})$",
    r"$\log(\gamma)$",
]
# Instantiate the training model
model = ModelTraining(None, config_dir, None, weight_F)


##########################################################################################
# Check local sloppiness
print("Assess local sloppiness")
jac_train_file = LIN_DIR / "jacobian_train.npy"
if jac_train_file.exists():
    J = np.load(jac_train_file)
else:
    # Compute Jacobian matrix
    fim_inst = FIM_jl(model.residuals)
    J = fim_inst.compute_jacobian(
        bestfit, h=0.1, t=2.0001, maxiters=100, abstol=tol, reltol=tol
    )
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
# Covarianc matrix
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
# Load data and error bars
energycurve_data = np.load(DATA_DIR / f"energycurve_data_{error_str}.npz")
# Target precission of the target QoI
target_error = energycurve_data["error"]
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
    Jac_target_inst = FIM_jl(model_target.residuals)
    J_target = -Jac_target_inst.compute_jacobian(
        bestfit, h=0.1, t=2.0001, maxiters=100, abstol=tol, reltol=tol
    )
    # bestfit, h=0.1, t=2.0, maxiters=100, abstol=1e-8, reltol=1e-8
    J_target *= model_target.std.reshape((-1, 1))
    np.save(jac_target_file, J_target)

# Perforn linear mapping of the samples
f0 = model_target.predictions(bestfit)
# Note: We can use sampling method, i.e., map the parameter ensemble to target qoi.
# Alternatively, we can use a closed-form formula to get the uncertainty of the target qoi.
# # Sampling approach
# samples_target_lin = np.array([f0 + J_target @ (sample - bestfit) for sample in samples])
# mean_target_lin = np.insert(np.mean(samples_target_lin, axis=0), 10, 0.0)
# error_target_lin = np.insert(np.std(samples_target_lin, axis=0), 10, 0.0)
# Closed-form expression
mean_target_lin = np.insert(f0, 10, 0.0)
cov_target_lin = J_target @ np.linalg.lstsq(fim, J_target.T, rcond=-1)[0]
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
