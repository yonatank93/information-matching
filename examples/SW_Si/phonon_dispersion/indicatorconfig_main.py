"""This is the main Python script to run indicator configuration calculation
* Model/potential: SW for Si system
* Candidate configurations: The example dataset in KLIFF for Si with varying lattice
  constants.
* Target QoI: phonon dispersion curve for silicon diamon.
* Target error bars: proportional to 10% of the predicted values made using KIM
  parameters or a uniform error bars.
"""

from pathlib import Path
import sys
from tqdm import tqdm
import pickle
import copy

import numpy as np
import cvxpy as cp

from information_matching.fim import FIM_nd
from information_matching.parallel import NonDaemonicPool as Pool
from information_matching.convex_optimization import ConvexOpt
from information_matching.leastsq import leastsq, compare_opt_results
from information_matching.summary import Summary
from information_matching.utils import (
    eps,
    tol,
    set_directory,
    set_file,
    copy_configurations,
)
from information_matching.termination import check_convergence

# This directory, i.e., working directory
WORK_DIR = Path(__file__).absolute().parent
sys.path.append(str(WORK_DIR.parent / "models"))
# Import model to train
from model_train import (
    Configs,
    ModelTrainingBase,
    ModelTraining,
    WeightIndicatorConfiguration,
    convert_cvxopt_weights_to_model_weights,
)
from model_target import ModelPhononDispersion


np.random.seed(2022)
prefix = "measurement"

print("Starting the indicator configuration calculation")


# Define the model to compute the target QoIs
# Load data and target error
# with open("phonondispersion_data_uniform.pkl", "rb") as f:
with open("phonondispersion_data.pkl", "rb") as f:
    phonondispersion_data = pickle.load(f)
# Instantiate model class to compute the target QoI
model_target = ModelPhononDispersion(data_dict=phonondispersion_data)
kim_best_fit = model_target.best_fit
nparams = model_target.nparams

# Path to the folder to store the results of each iteration
RES_DIR = set_directory(WORK_DIR / "results" / prefix)

# Tolerances
cvx_tol = tol**0.75  # Convex optimization
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
warmup = 3  # How many steps are used as a warm-up or burn-in

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
    fim_target_file = FIM_DIR / "phonondispersion.npy"
    if fim_target_file.exists():
        fim_target = np.load(fim_target_file)
    else:
        fim_target_model = FIM_nd(model_target.residuals, step=0.1 * params + 1e-4)
        fim_target = fim_target_model.compute_FIM(params)
        np.save(fim_target_file, fim_target)

    # FIMs of energy and forces
    print("Compute the FIMs for each candidate configuration")
    # Folder to contain FIMs with training quantities
    FIM_E_DIR = set_directory(FIM_DIR / "energy")  # Energy
    FIM_F_DIR = set_directory(FIM_DIR / "forces")  # Forces

    # Try to use multiprocessing to parallelize FIM calculation. The derivative
    # is calculated using numdifftools.

    # Define functions to compute energy and forces FIMs (separately) for one
    # configuration. These functions will be used in parallelization.
    def compute_FIM_energy_1config(test_id_item):
        ii, cpath = test_id_item
        # Path to the configuration file
        identifier = ".".join((Path(cpath).name).split(".")[:-1])
        # File containing calculated energy FIM
        fim_E_file = FIM_E_DIR / f"{identifier}.npy"

        # Energy
        if fim_E_file.exists():
            # FIM calculation has been done, we can just load the value
            fim_E = np.load(fim_E_file)
        else:
            # Compute energy FIM
            fim_E_model = FIM_nd(
                ModelTrainingBase(cpath, qoi=["energy"]).predictions,
                step=0.1 * params + 1e-4,
            )
            fim_E = fim_E_model.compute_FIM(params)
            np.save(fim_E_file, fim_E)

        return fim_E

    def compute_FIM_forces_1config(test_id_item):
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
            fim_F_model = FIM_nd(
                ModelTrainingBase(cpath, qoi=["forces"]).predictions,
                step=0.1 * params + 1e-4,
            )
            fim_F = fim_F_model.compute_FIM(params)
            np.save(fim_F_file, fim_F)

        return fim_F

    # Run FIM calculations for the training quantities
    nprocs = 10  # Number of parallel processes
    # Energy
    with Pool(nprocs) as p:
        fim_E_tensor = np.array(
            list(
                tqdm(
                    p.imap(compute_FIM_energy_1config, enumerate(Configs.files_energy)),
                    total=Configs.nconfigs_energy,
                )
            )
        )
    # Forces
    with Pool(nprocs) as p:
        fim_F_tensor = np.array(
            list(
                tqdm(
                    p.imap(compute_FIM_forces_1config, enumerate(Configs.files_forces)),
                    total=Configs.nconfigs_forces,
                )
            )
        )
    # Append the energy and forces FIMs
    fim_configs_tensor = np.concatenate((fim_E_tensor, fim_F_tensor), axis=0)

    ######################################################################################
    # Solve the convex optimization problem
    print("Solve convex optimization problem")
    cvx_file = set_file(STEP_DIR / "cvx_result.pkl")

    # Construct the input FIMs
    # FIM target
    norm = np.linalg.norm(fim_target)
    fim_target = {"fim": fim_target, "scale": 1 / norm}
    # FIM configs
    fim_configs = {}
    for ii, identifier in enumerate(Configs.ids):
        norm = np.linalg.norm(fim_configs_tensor[ii])
        fim_configs.update(
            {identifier: {"fim": fim_configs_tensor[ii], "fim_scale": 1 / norm}}
        )

    # Instantiate convex optimization class
    cvxopt = ConvexOpt(fim_target, fim_configs)

    # Solve
    solver = dict(
        verbose=True,
        solver=cp.SDPA,
        maxIteration=100_000,
        epsilonStar=cvx_tol,
        numThreads=nprocs,
    )

    if cvx_file.exists():
        with open(cvx_file, "rb") as f:
            cvxopt.result = pickle.load(f)
    else:
        cvxopt.solve(solver=solver)
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
    current_weights = cvxopt.get_config_weights(cvx_tol)

    # Update the optimal weights
    # We will also store the weights from previous iteration to use for convergence test
    if step >= warmup:
        old_opt_weights = copy.deepcopy(opt_weights)
        # Compare the old optimal weights to the current result
        for conf in current_weights:
            if conf in opt_weights:
                # Get the maximum weight between value stored in the current optimal
                # weights and the convex optimization result
                opt_weights[conf] = max([current_weights[conf], opt_weights[conf]])
            else:
                # Insert this weight to the optimal weight
                opt_weights.update({conf: current_weights[conf]})
    else:
        old_opt_weights = {}
        opt_weights = copy.deepcopy(current_weights)
    # Convert the format of the weights to be compatible for the training model
    configs_weights = convert_cvxopt_weights_to_model_weights(opt_weights)
    configs_weights_file = set_file(STEP_DIR / "configs_weights.pkl")
    with open(configs_weights_file, "wb+") as f:
        pickle.dump(configs_weights, f)
    # Copy the indicator configurations
    reduced_configs_E_dir = set_directory(STEP_DIR / "reduced_configs/energy")
    reduced_configs_F_dir = set_directory(STEP_DIR / "reduced_configs/forces")
    copy_configurations(
        configs_weights["energy"], Configs.path_energy, reduced_configs_E_dir
    )
    copy_configurations(
        configs_weights["forces"], Configs.path_forces, reduced_configs_F_dir
    )

    # Add the indicator configuration information to the summary file
    summary.update(configs_weights, "reduced configurations weights")

    ######################################################################################
    # Train the model with the reduced, indicator configurations and the optimal weights
    print("Train the model with the reduced configurations")
    # Path to the file that contain the result of model training
    opt_results_file = set_file(STEP_DIR / "opt_results.pkl")

    # Instantiate the training model class
    # Check if we use energy or forces quantities in the training
    use_energy = False
    use_forces = False
    if len(configs_weights["energy"]) > 0:
        use_energy = True
    if len(configs_weights["forces"]) > 0:
        use_forces = True
    # Instantiate weight class and set the path to the folders that contain the indicator
    # configurations
    weight_E = None
    weight_F = None
    config_path_E = None
    config_path_F = None
    if use_energy:
        weight_E = WeightIndicatorConfiguration(
            energy_weights_info=configs_weights["energy"]
        )
        config_path_E = reduced_configs_E_dir
    if use_forces:
        weight_F = WeightIndicatorConfiguration(
            forces_weights_info=configs_weights["forces"]
        )
        config_path_F = reduced_configs_F_dir

    model_training = ModelTraining(
        config_path_E, config_path_F, weight_E, weight_F, nprocs=20
    )
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


# """This module is to test the convex optimization and, by extension, the
# weights later.
# """

# from pathlib import Path
# from tqdm import tqdm
# import pickle

# import numpy as np
# import cvxpy as cp
# from cvxpy.error import SolverError
# from byukim.utils import NonDaemonicPool as Pool

# # from multiprocessing import Pool

# from models import (
#     ModelAlatBase,
#     ModelAlat,
#     ModelPhononDispersion,
#     config_alat_datapath,
#     config_paths_energy,
#     config_paths_forces,
# )
# from fim import FIM_jl, FIM_nd

# from convex_optimization import ConvexOpt
# from leastsq import leastsq, compare_opt_results, WeightIndicatorConfiguration
# from summary import Summary
# from utils import WORK_DIR, eps, tol, set_directory, set_file
# from termination import check_convergence, get_results_array, check_periodicity


# np.random.seed(2022)

# # Define the models
# tmp_model = ModelAlatBase(config_alat_datapath, nprocs=20)
# orig_best_fit = tmp_model.best_fit
# default_best_fit = tmp_model._default_bestfit
# nparams = tmp_model.nparams
# model_PD = ModelPhononDispersion()
# warmup = 0  # How many steps are used as a warm-up


# ################################################################################


# step = 0

# # Find how many steps have been completed
# while True:
#     STEP_DIR = WORK_DIR / "steps" / str(step)
#     if STEP_DIR.exists():
#         step += 1
#     else:
#         laststep_finalfile = STEP_DIR / "new_predictions.txt"
#         if step > 0:
#             if laststep_finalfile.exists():
#                 # The previous step is completely done, we can go to the next
#                 # step.
#                 step += 1
#             else:
#                 # Subtract 1 before breaking because the last step is not
#                 # completely done.
#                 step -= 1
#         print(step, "steps have been done.")
#         break

# while step < 100:
#     print("Calculation for step", step)
#     STEP_DIR = set_directory(WORK_DIR / "steps" / str(step))

#     # Add summary
#     summary = Summary(STEP_DIR / "summary.json")

#     if step:
#         PREV_STEP_DIR = WORK_DIR / "steps" / str(step - 1)
#         prev_bestfit_file = PREV_STEP_DIR / "bestfit.txt"
#         params = np.loadtxt(prev_bestfit_file)
#     else:
#         params = default_best_fit  # np.zeros(nparams)
#     print("Parameters:", params)

#     ############################################################################

#     # Compute the FIM
#     FIM_DIR = set_directory(STEP_DIR / "FIM")
#     FIM_E_DIR = set_directory(FIM_DIR / "energy")
#     FIM_F_DIR = set_directory(FIM_DIR / "forces")

#     # FIM of energy vs lattice constant curve
#     print("Compute the FIM for energy curve")
#     fim_EC_file = FIM_DIR / "phonondispersion.npy"
#     if fim_EC_file.exists():
#         fim_EC = np.load(fim_EC_file)
#     else:
#         fim_EC_model = FIM_jl(model_PD)
#         fim_EC = fim_EC_model.compute_FIM(params)
#         np.save(fim_EC_file, fim_EC)

#     # FIMs of energy and forces
#     print("Compute the FIMs for forces for each configuration")
#     nconfig_E = len(config_paths_energy)
#     nconfig_F = len(config_paths_forces)

#     # Try to use multiprocessing to parallelize FIM calculation. The derivative
#     # is calculated using numdifftools.
#     test_id_items_energy = [[ii, cpath] for ii, cpath in enumerate(config_paths_energy)]
#     test_id_items_forces = [[ii, cpath] for ii, cpath in enumerate(config_paths_forces)]

#     # Parallel run starts here
#     # Function to use in multiprocessing to compute FIM
#     def compute_FIM_energy_1config(test_id_item):
#         ii, cpath = test_id_item
#         identifier = ".".join((Path(cpath).name).split(".")[:-1])
#         fim_E_file = FIM_E_DIR / f"{identifier}.npy"

#         # Energy
#         if fim_E_file.exists():
#             fim_E = np.load(fim_E_file)
#         else:
#             fim_E_model = FIM_nd(ModelAlatBase(cpath, qoi=["energy"]))
#             fim_E = fim_E_model.compute_FIM(params)
#             np.save(fim_E_file, fim_E)

#         return fim_E

#     def compute_FIM_forces_1config(test_id_item):
#         ii, cpath = test_id_item
#         identifier = ".".join((Path(cpath).name).split(".")[:-1])
#         fim_F_file = FIM_F_DIR / f"{identifier}.npy"

#         # Forces
#         if fim_F_file.exists():
#             fim_F = np.load(fim_F_file)
#         else:
#             fim_F_model = FIM_nd(ModelAlatBase(cpath, qoi=["forces"]))
#             fim_F = fim_F_model.compute_FIM(params)
#             np.save(fim_F_file, fim_F)

#         return fim_F

#     # Define function to run parallel calculation for FIM over single
#     # configurations
#     def run_FIM_mp(test_id_items, FIM_fn, nprocs):
#         nconfigs = len(test_id_items)
#         fim_tensor = np.empty((nconfigs, nparams, nparams))

#         pbar = tqdm(total=nconfigs)
#         # Split the calculation into batches
#         nbatch = int(np.ceil(nconfigs / nprocs))
#         cpath_batch = []
#         for n in range(nbatch):
#             batch = test_id_items[n * nprocs : (n + 1) * nprocs]
#             cpath_batch.append(batch)
#         # Run parallel calculation for each batch
#         ndone = 0  # Number of calculations done
#         for batch in cpath_batch:
#             nitems = len(batch)
#             nprocs = min([nprocs, nitems])  # Don't waste compute resource
#             pool = Pool(nprocs)
#             fim_tensor[ndone : ndone + nitems] = np.array(pool.map(FIM_fn, batch))
#             ndone += nitems
#             pbar.update(n=nitems)
#             pool.close()
#         pbar.close()

#         return fim_tensor

#     # Run FIM calculation
#     nprocs = 20
#     fim_E_tensor = run_FIM_mp(test_id_items_energy, compute_FIM_energy_1config, nprocs)
#     fim_F_tensor = run_FIM_mp(test_id_items_forces, compute_FIM_forces_1config, nprocs)

#     ############################################################################
#     # Solve the convex optimization problem
#     print("Solve convex optimization problem")
#     cvx_file = set_file(STEP_DIR / "cvx_result.pkl")
#     cvx_tol = tol ** 0.75
#     # Instantiate convex optimization class
#     cvxopt = ConvexOpt(fim_EC, fim_E_tensor, fim_F_tensor, normalize=True)

#     # Solve
#     # while cvx_tol < 1e-1:
#     #     try:
#     # I set the tolerance this way so that I can update the tolerance
#     # if the calculation fails.
#     solver = dict(
#         solver=cp.SCS,
#         max_iters=100_000_000,
#         eps=cvx_tol,
#         acceleration_lookback=0,
#         normalize=True,
#         scale=cvx_tol ** 0.5,
#     )

#     if cvx_file.exists():
#         cvxopt.result = pickle.load(open(cvx_file, "rb"))
#     else:
#         cvxopt.solve(solver=solver)
#         pickle.dump(cvxopt.result, open(cvx_file, "wb+"), protocol=4)
#     #     break
#     # except SolverError:
#     #     # Soften the termination condition
#     #     cvx_tol *= 10
#     #     print("Increase the tolerance to", cvx_tol)

#     summary.update(cvxopt.result, "convex optimization")

#     ############################################################################

#     # Interpret the result of the convex optimization problem and get the
#     # indicator configurations
#     print("Get the reduced configurations")
#     configs_weights_file = set_file(STEP_DIR / "configs_weights.pkl")
#     reduced_configs_E_dir = set_directory(STEP_DIR / "reduced_configs/energy")
#     reduced_configs_F_dir = set_directory(STEP_DIR / "reduced_configs/forces")

#     if configs_weights_file.exists():
#         configs_weights = pickle.load(open(configs_weights_file, "rb"))
#     else:
#         # Get the weights of the reduced configurations
#         ss = step if step > warmup else 0
#         configs_weights = cvxopt.get_configs_weights(ss, cvx_tol)
#         pickle.dump(configs_weights, open(configs_weights_file, "wb+"))
#         # Copy the configurations
#         cvxopt.copy_configurations(configs_weights, reduced_configs_E_dir, "energy")
#         cvxopt.copy_configurations(configs_weights, reduced_configs_F_dir, "forces")

#     use_energy = False
#     use_forces = False
#     if len(configs_weights["energy"]) > 0:
#         use_energy = True
#     if len(configs_weights["forces"]) > 0:
#         use_forces = True
#     summary.update(configs_weights, "reduced configurations weights")

#     ############################################################################

#     # Train the model with the reduced configurations
#     print("Train the model with the reduced configurations")
#     opt_results_file = set_file(STEP_DIR / "opt_results.pkl")
#     # Instantiate the model
#     weight_E = None
#     weight_F = None
#     config_path_E = None
#     config_path_F = None
#     if use_energy:
#         weight_E = WeightIndicatorConfiguration(
#             energy_weights_info=configs_weights["energy"]
#         )
#         config_path_E = reduced_configs_E_dir
#     if use_forces:
#         weight_F = WeightIndicatorConfiguration(
#             forces_weights_info=configs_weights["forces"]
#         )
#         config_path_F = reduced_configs_F_dir

#     model_EF = ModelAlat(config_path_E, config_path_F, weight_E, weight_F, nprocs=20)
#     # Set the Lagrange multiplier to be the number of predictions
#     gamma = 0.0
#     model_EF.gamma = gamma

#     if opt_results_file.exists():
#         opt_results = pickle.load(open(opt_results_file, "rb"))
#     else:
#         # To have more confident in finding the "true" best fit, we will try
#         # several different starting points
#         starting_points = np.row_stack((orig_best_fit, default_best_fit))
#         # Previous step's best fit
#         if step:
#             prev_bestfit_file = PREV_STEP_DIR / "bestfit.txt"
#             prev_best_fit = np.loadtxt(prev_bestfit_file)
#             starting_points = np.row_stack((starting_points, prev_best_fit))
#         # # Random points
#         # # We will get a different set of random points in every iteration.
#         # nrand_points = 5
#         # rand_points = orig_best_fit + np.random.randn(nrand_points, nparams)
#         # starting_points = np.row_stack((starting_points, rand_points))

#         # Optimizer setting
#         # If there are more parameters than data, then we shoot for atol
#         # termination condition.
#         tol = eps ** 0.75
#         if model_EF.npred >= model_EF.nparams:
#             # Use geodesic LM
#             method = "geodesiclm"
#             leastsq_kwargs = dict(
#                 atol=1e-4,
#                 xtol=tol,
#                 xrtol=tol,
#                 ftol=tol,
#                 frtol=tol,
#                 gtol=tol,
#                 Cgoal=tol,
#                 maxiters=1000000,
#                 avmax=0.5,
#                 factor_accept=5,
#                 factor_reject=2,
#                 h1=1e-6,
#                 h2=1e-1,
#                 imethod=1,
#                 iaccel=0,
#                 ibold=0,
#                 ibroyden=1,
#                 print_level=3,
#             )
#         else:
#             # Use TRF. The Notes in scipy.optimize.least_squares mentions that
#             # this algorithm tries to avoid the boundaries.
#             method = "trf"
#             leastsq_kwargs = {
#                 "ftol": tol,
#                 "xtol": tol,
#                 "gtol": tol,
#                 "max_nfev": 1000000,
#                 "verbose": 2,
#             }

#         # Optimization with different starting points
#         opt_results = {}
#         for ii, point in enumerate(starting_points):
#             print("Optimization with starting point:", point)
#             tmp_opt_result = leastsq(model_EF, point, method, **leastsq_kwargs)
#             opt_results.update({ii: tmp_opt_result})
#             pickle.dump(opt_results, open(opt_results_file, "wb+"))

#     # Comparing the cost and get the point with the lowest cost
#     new_bestfit, opt_cost, opt_result = compare_opt_results(opt_results)
#     np.savetxt(STEP_DIR / "bestfit.txt", new_bestfit)

#     print(opt_result[1]["converged"], opt_result[1]["msg"])
#     print("New best fit:", new_bestfit)
#     print("Optimal cost:", opt_cost)
#     summary.update(opt_result, "model training", gamma=gamma)

#     ############################################################################

#     # Calculate the new energy vs lattice constant curve
#     preds_file = STEP_DIR / "new_predictions.txt"
#     if preds_file.exists():
#         new_preds = np.loadtxt(preds_file)
#     else:
#         new_preds = model_PD.predictions_dict(new_bestfit)["energies"].flatten()
#         np.savetxt(preds_file, new_preds)

#     print("New predictions:", new_preds)
#     summary.update(new_preds, "new predictions")

#     ############################################################################

#     # Check termination condition
#     periodic = False
#     terminate = False
#     if step <= warmup:
#         # We are not going to check the termination condition if we just
#         # started.
#         pass
#     else:
#         if step > 2:
#             results_array = get_results_array(
#                 ["convex optimization", "relative error"], step
#             )
#             # Check periodicity only when periodicity not detected.
#             # Use the convex optimization result to check the periodicity.
#             periodic, start, period = check_periodicity(results_array)
#             if periodic:
#                 print(f"Start at {start}, period {period}")
#                 print("Periodicity detected:", periodic)
#                 if len(results_array) >= np.ceil(start + 2.5 * period):
#                     terminate = True

#         if not periodic:
#             terminate = check_convergence(step, tol=cvx_tol ** 0.5)
#             print("Converged:", terminate)

#     if terminate:
#         break

#     step += 1
