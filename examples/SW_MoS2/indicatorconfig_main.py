"""This is the main Python script to run indicator configuration calculation
* Model/potential: SW for MoS_2 system
* Candidate configurations: The training dataset used in the paper about this potential
  by Mingjian Wen.
* Target QoI: energy change vs lattice compression, with change in lattice dimension
  varies from -0.5 to 0.5 Angstrom.
* Target error bars: proportional to 10% of the predicted values made using KIM
  parameters (Mingjian's or Yonatan's version). See below for more information.
"""

from pathlib import Path
from tqdm import tqdm
import pickle
import copy

import numpy as np

from information_matching.fim.fim_jl import FIM_jl
from information_matching.convex_optimization import ConvexOpt, compare_weights
from information_matching.preconditioning import preconditioning
from information_matching.leastsq import leastsq, compare_opt_results
from information_matching.utils.misc import (
    set_directory,
    set_file,
    copy_configurations,
    check_convergence,
    Summary,
)

from models.models import (
    Configs,
    ModelTrainingBase,
    ModelTraining,
    ModelEnergyCurve,
    WeightIndicatorConfiguration,
    convert_cvxopt_weights_to_model_weights,
)

np.random.seed(2022)

# This directory, i.e., working directory
WORK_DIR = Path(__file__).absolute().parent
DATA_DIR = WORK_DIR / "data"

error_str = "mingjian"  # Indicate which target error to use
start_str = "mingjian"  # Indicate which starting parameters to use
suffix = f"error_{error_str}_start_{start_str}"
# Notes for the error_str variable:
# * If "mingjian", use data predicted by Mingjian's model and the target error is 10% of
#   the predictions.
# * If "yonatan", use data predicted by Yonatan's model and target error is 10% of the
#   predictions.
# * If "ips", use data predicted by Mingjian's model and target error is the standard
#   deviation obtained using different IPs, as presented in Mingjian's paper.
# Notes for start_str: To change the starting point, edit models/model.py file.

print("Starting the indicator configuration calculation")


# Define the model to compute the target QoIs
# Load data and error bars
energycurve_data = np.load(DATA_DIR / f"energycurve_data_{error_str}.npz")
# Target precission of the target QoI
target_error = energycurve_data["error"]
# Instantiate model class to compute the target QoI
model_target = ModelEnergyCurve(data=energycurve_data["data"], error=target_error)
kim_best_fit = model_target.best_fit
nparams = model_target.nparams

# Path to the folder to store the results of each iteration
RES_DIR = set_directory(WORK_DIR / "results" / suffix)

# Tolerances
eps = np.finfo(float).eps
tol = eps**0.5
cvx_tol = tol**0.5  # Convex optimization
lstsq_tol = eps**0.75  # Least-squares training
converge_tol = cvx_tol**0.5  # Tolerance on the weights to converge
# Optimizer settings
geodesiclm_kwargs = dict(
    atol=1e-4,
    xtol=lstsq_tol,
    xrtol=lstsq_tol,
    ftol=lstsq_tol,
    frtol=lstsq_tol,
    gtol=lstsq_tol,
    Cgoal=lstsq_tol,
    maxiters=1000000,
    avmax=0.5,
    factor_accept=5,
    factor_reject=2,
    h1=1e-6,
    h2=1e-1,
    imethod=1,
    iaccel=0,
    ibold=0,
    ibroyden=1,
    print_level=3,
)
trf_kwargs = dict(
    ftol=lstsq_tol, xtol=lstsq_tol, gtol=lstsq_tol, max_nfev=1000000, verbose=2
)


##########################################################################################
# Run indicator configuration calculation
maxsteps = 10  # Maximum number of steps
warmup = 1  # How many steps are used as a warm-up or burn-in

# Initial state
step = 0
params = copy.deepcopy(kim_best_fit)  # Initial parameter values
opt_weights = {"energy": {}, "forces": {}}

while step < maxsteps:
    print("Calculation for step", step)
    print("Parameters:", params)
    # Set path to the folder to contain results for this iteration.
    STEP_DIR = set_directory(RES_DIR / str(step))

    # Start a summary file
    summary = Summary(STEP_DIR / "summary.json")

    ######################################################################################
    # Compute the FIMs
    FIM_DIR = set_directory(STEP_DIR / "FIM")

    # FIM of energy vs lattice constant curve, i.e., the target FIM
    print("Compute the FIM for energy curve")
    fim_target_file = FIM_DIR / "energycurve.npy"
    if fim_target_file.exists():
        fim_target = np.load(fim_target_file)
    else:
        fim_target_model = FIM_jl(
            model_target.residuals, h=0.1, t=2.0001, maxiters=100, abstol=tol, reltol=tol
        )
        fim_target = fim_target_model.FIM(params)
        # params, h=0.1, t=2.0, maxiters=100, abstol=1e-8, reltol=1e-8
        np.save(fim_target_file, fim_target)

    # FIMs of energy and forces
    print("Compute the FIMs for each candidate configuration")
    # Folder to contain FIMs with training quantities
    FIM_F_DIR = set_directory(FIM_DIR / "forces")  # Forces

    # Try to use multiprocessing to parallelize FIM calculation. The derivative
    # is calculated using numdifftools.

    # Define functions to compute energy and forces FIMs (separately) for one
    # configuration. These functions will be used in parallelization.
    def FIM_forces_1config(test_id_item):
        ii, cpath = test_id_item
        # Path to the configuration file
        identifier = ".".join((Path(cpath).name).split(".")[:-1])
        # File containing calculated forces FIM
        fim_F_file = FIM_F_DIR / f"{identifier}.npy"

        # Forces
        if fim_F_file.exists():
            # FIM calculation has been done, we can just load the value
            fim_F = np.load(fim_F_file)
        else:
            # Compute forces FIM
            fim_F_model = FIM_jl(
                ModelTrainingBase(cpath, qoi=["forces"]).predictions,
                h=0.1,
                t=2.0001,
                maxiters=100,
                abstol=tol,
                reltol=tol,
            )
            fim_F = fim_F_model.FIM(params)
            np.save(fim_F_file, fim_F)

        return fim_F

    # Run FIM calculations for the training quantities
    # Forces
    fim_configs_tensor = np.array(
        list(
            tqdm(
                map(FIM_forces_1config, enumerate(Configs.dataset_files)),
                total=Configs.nconfigs,
            )
        )
    )

    ######################################################################################
    # Solve the convex optimization problem
    print("Solve convex optimization problem")
    cvx_file = set_file(STEP_DIR / "cvx_result.pkl")

    # Construct the input FIMs
    # FIM target
    fim_target = preconditioning(fim_target, "frobenius")
    # FIM configs
    fim_configs = preconditioning(
        {identifier: fim_configs_tensor[ii] for ii, identifier in enumerate(Configs.ids)},
        "frobenius",
    )

    # Instantiate convex optimization class
    cvxopt = ConvexOpt(fim_target, fim_configs)

    # Solve
    solver = dict(
        verbose=True,
        solver="SCS",
        max_iters=1_000_000,
        eps=cvx_tol,
        acceleration_lookback=0,
        normalize=True,
        scale=cvx_tol**0.5,
    )

    if cvx_file.exists():
        with open(cvx_file, "rb") as f:
            cvxopt.result = pickle.load(f)
    else:
        cvxopt.solve(**solver)
        if "optimal" in cvxopt.result["status"]:
            with open(cvx_file, "wb+") as f:
                pickle.dump(cvxopt.result, f, protocol=4)
        else:
            raise ValueError("Convex optimization doesn't converge")

    # Add convex optimization result to the summary file
    summary.update(cvxopt.result, "convex optimization")

    ######################################################################################
    # Interpret the convex optimization result, i.e., get the nonzero weights and update
    # the optimal weights
    print("Interpret the convex optimization result")
    # Get the nonzero weights from convex optimization result
    current_weights = cvxopt.get_config_weights(cvx_tol, cvx_tol)

    # Update the optimal weights
    # We will also store the weights from previous iteration to use for convergence test
    if step >= warmup:
        old_opt_weights = copy.deepcopy(opt_weights)
        # Compare the old optimal weights to the current result
        opt_weights = compare_weights(old_opt_weights, current_weights)
    else:
        old_opt_weights = {}
        opt_weights = copy.deepcopy(current_weights)
    # Convert the format of the weights to be compatible for the training model
    configs_weights = convert_cvxopt_weights_to_model_weights(opt_weights)
    configs_weights_file = set_file(STEP_DIR / "configs_weights.pkl")
    with open(configs_weights_file, "wb+") as f:
        pickle.dump(configs_weights, f)
    # Copy the indicator configurations
    reduced_configs_F_dir = set_directory(STEP_DIR / "reduced_configs")
    copy_configurations(
        configs_weights["forces"], Configs.dataset_path, reduced_configs_F_dir
    )

    # Add the indicator configuration information to the summary file
    summary.update(configs_weights, "reduced configurations weights")

    ######################################################################################
    # Train the model with the reduced, indicator configurations and the optimal weights
    print("Train the model with the reduced configurations")
    # Path to the file that contain the result of model training
    opt_results_file = set_file(STEP_DIR / "opt_results.pkl")

    # Instantiate the training model class
    # Instantiate weight class and set the path to the folders that contain the indicator
    # configurations
    weight_F = WeightIndicatorConfiguration(forces_weights_info=configs_weights["forces"])
    config_path_F = reduced_configs_F_dir

    model_training = ModelTraining(None, config_path_F, None, weight_F, nprocs=20)
    gamma = 0.0  # Lagrange multiplier for the regularization
    model_training.gamma = gamma

    # Training
    if opt_results_file.exists():
        # We did this calculation already
        with open(opt_results_file, "rb") as f:
            opt_results = pickle.load(f)
    else:
        # To have more confident in finding the "true" best fit, we will try
        # several different starting points
        starting_points = copy.deepcopy(kim_best_fit).reshape((1, -1))
        # Previous step's best fit
        if step:
            prev_bestfit_file = RES_DIR / str(step - 1) / "bestfit.txt"
            prev_best_fit = np.loadtxt(prev_bestfit_file)
            starting_points = np.row_stack((starting_points, prev_best_fit))

        # Optimization with different starting points
        if model_training.npred >= model_training.nparams:
            # Use geodesic LM
            method = "geodesiclm"
            leastsq_kwargs = geodesiclm_kwargs
        else:
            # Use TRF if we have fewer data than parameters. The notes in
            # scipy.optimize.least_squares mentions that this algorithm tries to avoid
            # the boundaries.
            method = "trf"
            leastsq_kwargs = trf_kwargs

        opt_results = {}
        for ii, point in enumerate(starting_points):
            print("Optimization with starting point:", point)
            tmp_opt_result = leastsq(
                model_training.residuals, point, method, **leastsq_kwargs
            )
            opt_results.update({ii: tmp_opt_result})
            # Export all optimization results
            with open(opt_results_file, "wb+") as f:
                pickle.dump(opt_results, f)

    # Comparing the cost and get the point with the lowest cost
    new_bestfit, opt_cost, opt_result = compare_opt_results(opt_results)
    params = new_bestfit  # Update the parameter values for the next iteration
    # Export the optimal parameters
    np.savetxt(STEP_DIR / "bestfit.txt", new_bestfit)

    print(opt_result[1]["converged"], opt_result[1]["msg"])
    print("New best fit:", new_bestfit)
    print("Optimal cost:", opt_cost)

    # Add the model training result to the summary file
    summary.update(opt_result, "model training", gamma=gamma)

    ######################################################################################
    # Calculate the new energy vs lattice constant curve using the new optimal parameters
    # Path to the file that contain the new energy vs lattice constant predictions
    preds_file = STEP_DIR / "new_predictions.txt"
    if preds_file.exists():
        # We did this calculation already
        new_preds = np.loadtxt(preds_file)
    else:
        # Compute the new energy vs. lattice constant predictions
        new_preds = model_target.predictions(new_bestfit)
        np.savetxt(preds_file, new_preds)

    print("New predictions:", new_preds)

    # Add the new target prediction values to the summary file
    summary.update(new_preds, "new predictions")

    ######################################################################################
    # Check termination condition
    terminate = False
    if step < warmup:
        # We are not going to check the termination condition if we are still in the
        # warm-up period
        print("Still in the warm-up rounds")
    else:
        # The calculation is converged if the weights on the current iteration is the same
        # as the optimal weights from the previous iteration (within some tolerance)
        terminate = check_convergence(old_opt_weights, opt_weights, tol=converge_tol)
        print("Converged:", terminate)

    if terminate:
        break
    else:
        step += 1

##########################################################################################
print("Indicator configuration calculation terminated due to:")
if terminate:
    print(f"The calculation has converged in {step} steps")
else:
    print("The maximum number of steps is exceeded.")
