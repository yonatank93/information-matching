from pathlib import Path
import os

import numpy as np
import jinja2
import matplotlib.pyplot as plt
from .base import Model

try:
    import lammps
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Module ``lammps`` not found, this test cannot be used"
    ) from exc

CUR_DIR = Path(__file__).absolute().parent
eps = np.finfo(float).eps

# Initiate dictionary to store the results
results = {
    "a0": {"desc": "Equilibrium lattice constant a", "unit": "angstrom"},
    "b0": {"desc": "Equilibrium lattice constant b", "unit": "angstrom"},
    "ecell": {"desc": "Energy per primitive unit cell", "unit": "eV/cell"},
    "alist": {"desc": "List of lattice constants a", "unit": "angstrom"},
    "blist": {"desc": "List of lattice constants b", "unit": "angstrom"},
    "elist": {"desc": "List of energies per unit cell", "unit": "eV/cell"},
}

# A list of perturbation to try as the initial guess for the equilibrium
# lattice constant
a0_init = 3.2  # Initial guess of the equilibrium a
b0_init = 3.19  # Initial guess of the equilibrium a

# Default list of the length perturbation
perturb_length = np.arange(0.0, 10.1, 0.5)
# LAMMPS template to check if the initial energy is finite
with open(CUR_DIR / "initial_energy.tpl", "r") as f:
    initeng_tpl = f.read()
# LAMMPS script that contains commands to create the atoms ans simulation box
conf_path = CUR_DIR / "conf_T0.lmp"
# Main LAMMPS template to compute energy given lattice constants
lammps_tpl_file = CUR_DIR / "energy_latconst.tpl"

# To keep the simulation stable, we don't allow the initial lattice constant to
# be too small or the initial energy to be too big.
latconst_threshold = 0.1  # Lowest lattice constant allowed
energy_inf = 1e12  # Largest energy allowed


class EnergyVsLatticeConstant(Model):
    """Compute the energy per primitive unit cell vs lattice constant curve of
    an MoS2 sheet. This is done by first equilibrating the system and get the
    equilibrium lattice constants (a0 and b0) and the equilibrium energy per
    unit cell. Note that since there are 3 atoms per unit cell (1 Mo atom and
    2 S atoms), this value would be approximately 3 times the cohesive energy.
    After that, a list of linearly spaced points of lattice constants a is
    created in the range (a-0.5, a+0.5). Energy per unit cell is then computed
    for each lattice constant a value in this list, with relaxation done only
    in the z direction, so that value of b is also different for each a. All
    the results are stored in a dictionary.

    Parameters
    ----------
    model_id: str
        KIM model ID to use. So far, it is tested to work with SW MoS2.
    da_list: np.ndarray (optional)
        List of :math:`\Delta a` to use.
    lmpfile: str or Path-like (optional)
        Path to a file in which the lammps script will be written.
    """

    def __init__(self, model_id, da_list=np.arange(-0.5, 0.51, 0.05), lmpfile=None):
        super().__init__("MoS2", None, model_id)
        self.predictions = results
        self.pred_keys = list(self.predictions)
        self.num_pred = len(self.predictions)
        self.da_list = np.sort(da_list)

        # Retrieve default parameters to reset the parameters later
        self._def_params = self.get_model_parameters()

        # Define the parameters attribute as a caching purpose
        self._convert_parameters(self._def_params)

        # Get the cutoff radius
        self._cutoff = np.array(self.get_default_cutoff()[1])

        # Prepare to load template file
        self.lmpfile = lmpfile
        if self.lmpfile is None:
            # Don't write lammps file, use lmp.commands_string to run
            self._env = jinja2.Environment()
            # Templates
            with open(lammps_tpl_file, "r") as f:
                self.lammps_tpl = f.read()
        else:
            # Write lammps file and use lmp
            file_loader = jinja2.FileSystemLoader(CUR_DIR)
            self._env = jinja2.Environment(loader=file_loader)
            # Templates
            self.lammps_tpl = lammps_tpl_file.name

        # Initialize other attributes
        self.verbose = False
        self._calc_changed = True

    def _convert_parameters(self, parameters):
        """Convert parameter dictionary to string that can be input to the
        lammps input file.
        {A: [[1], [value]]} -> A 1 value
        """
        params = []
        for name, info in parameters.items():
            idx, values = info
            if isinstance(idx, (int, float)):
                idx = [idx]
            if isinstance(values, (int, float)):
                values = [values]
            for ii, val in zip(idx, values):
                temp = [name, ii + 1, val]
                str_temp = " ".join([str(el) for el in temp])
                params.append(str_temp)
        str_params = " ".join([el for el in params])
        self._params_str = str_params
        self._calc_changed = True

    def _equilibrium_state(self):
        """Get the prediction related to the equilibrium structure of the
        system. This returns the equilibrium lattice constants (a and b) and
        the energy per unit cell.
        """
        # Perform relaxation, with initial a and b set to 3.20 and 3.19
        # angstrom. These values are obtained from DFT calculation.
        stage = "equilibration"
        a0, b0, ecell = self.relaxation(a0_init, b0_init, stage)
        return a0, b0, ecell

    def _ecell(self, a, b, stage):
        """Compute the energy per unit cell and lattice constant b given
        lattice constant a.
        """
        # Load the template file
        if self.lmpfile is None:
            template = self._env.from_string(self.lammps_tpl)
        else:
            template = self._env.get_template(self.lammps_tpl)

        # Generate lammps script from the template
        script = template.render(
            modelname=self.model_id,
            a=a,
            b=b,
            conf_path=conf_path,
            parameters=self._params_str,
            stage=stage,
        )

        # Create lammps object
        if self.verbose:
            lmp = lammps.lammps()
        else:
            lmp = lammps.lammps(cmdargs=["-screen", os.devnull, "-nocite"])
        # Run lammps script
        if self.lmpfile is None:
            lmp.commands_string(script)
        else:
            # Export the generated lammps script
            with open(self.lmpfile, "w+") as f:
                f.write(script)
            lmp.file(self.lmpfile)

        # Extract the cohesive energy value and put it in the list
        a = lmp.extract_variable("a")
        b = lmp.extract_variable("b")
        ecell = lmp.extract_variable("ecell")
        lmp.close()

        return a, b, ecell

    def _check_initial_energy(self, a, b):
        """Check if the initial energy for given initial lattice constants a
        and b has a finite value. Otherwise, the calculation with these initial
        lattice constant values will be skipped.
        """
        if self.verbose:
            print("Check if initial configuration has a finite energy")
        # Load the template file
        template = self._env.from_string(initeng_tpl)

        # Generate lammps script from the template
        script = template.render(
            modelname=self.model_id,
            a=a,
            b=b,
            conf_path=conf_path,
            parameters=self._params_str,
        )

        # Create lammps object
        if self.verbose:
            lmp = lammps.lammps()
        else:
            lmp = lammps.lammps(cmdargs=["-screen", os.devnull, "-nocite"])
        # Run lammps script
        lmp.commands_string(script)

        # Extract the potential energy value
        pe = lmp.extract_variable("pe")
        lmp.close()

        if self.verbose:
            print(f"Initial energy: {pe}")

        return not np.isnan(pe) and np.abs(pe) < energy_inf

    def _ecell_v_latconst(self):
        """Generate a list of lattice constants a from the equilibrium lattice
        constants and run lammps simulation to get the energy per unit cell
        for given lattice constant.
        """
        stage = "measurement"
        # Get the equilibrium state's predictions
        if self.verbose:
            print("Initial relaxation")
        a0, b0, e0 = self._equilibrium_state()

        # Change the lattice constant and run lammps simulation
        # List of da's
        da_left, da_right = self._split_dalist()

        # Create lists of lattice constants
        # For the list of a on the left of a0 reverse the order, so that the left most
        # element is the closest to a0
        aleft = da_left[::-1] + a0
        npoints_left = len(aleft)
        aright = da_right + a0
        npoints_right = len(aright)
        npoints = npoints_left + npoints_right
        # Generate indices to indicate where to put the results in the final
        # array. a will increase as we move to the right in the array.
        ialeft = np.arange(npoints_left)[::-1]
        iaright = np.arange(npoints_left, npoints)

        elist = np.zeros(npoints)
        alist = np.zeros(npoints)
        blist = np.zeros(npoints)

        # Iterate over the lattice constant a in the left list
        for ii, (idx, aa) in enumerate(zip(ialeft, aleft)):
            if self.verbose:
                print(f"L{idx}, {aa}")
            if aa == a0:
                a = a0
                b = b0
                ecell = e0
            else:
                a, b, ecell = self.relaxation(aa, b0_init, stage)
            alist[idx] = a
            blist[idx] = b
            elist[idx] = ecell

        # Iterate over the lattice constant a in the right list
        for ii, (idx, aa) in enumerate(zip(iaright, aright)):
            if self.verbose:
                print(f"R{idx}, {aa}")
            a, b, ecell = self.relaxation(aa, b0_init, stage)
            alist[idx] = a
            blist[idx] = b
            elist[idx] = ecell

        return a0, b0, e0, alist, blist, elist

    def _split_dalist(self):
        """Split the list of da into 2: list of da <= 0.0 and da > 0.0."""
        # Find index of element that is closest to 0.0
        idx = np.argmin(np.abs(self.da_list))
        # Fix the index to ensure that left list is less than 0.0
        if self.da_list[idx] > eps**0.5:
            idx = idx - 1
        # Splitted lists
        da_left = self.da_list[: idx + 1]
        da_right = self.da_list[idx + 1 :]
        return da_left, da_right

    def relaxation(self, ainit, binit, stage):
        """Perform the relaxation. The relaxation is done using several trial
        initial guess.
        """
        success = False
        if self.verbose:
            print(stage)
        for pert in perturb_length:
            if stage == "measurement":
                a_init = ainit
            elif stage == "equilibration":
                a_init = ainit + pert
            b_init = binit + pert

            # Check if the initial guess is too large
            initial_cell_dim_ok = self._initial_cell_dim_ok(a_init, b_init)

            if initial_cell_dim_ok:
                if a_init > latconst_threshold and b_init > latconst_threshold:
                    if self.verbose:
                        print(
                            "Try calculation with initial lattice constant "
                            f"a={a_init} and b={b_init}"
                        )
                    # Check if the initial configuration has finite
                    # energy
                    finite_energy = self._check_initial_energy(a_init, b_init)
                    if finite_energy:
                        try:
                            a, b, ecell = self._ecell(a_init, b_init, stage)
                            success = True
                            break
                        except Exception:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
        if not success:
            a = a_init
            b = np.nan
            ecell = np.inf
            # raise Exception("Failed in relaxing the system")
        return a, b, ecell

    def _initial_cell_dim_ok(self, ainit, binit):
        """Check if in the initial configuration, we still have interacting
        atoms by checking the atomic distance and the initial cell dimension.
        """
        rc_MoMo, rc_MoS, rc_SS = self._cutoff
        # Check for Mo-Mo interaction
        MoMo_ok = ainit < rc_MoMo
        # Check for S-S interaction
        SS_ok = ainit < rc_SS
        # Check for Mo-S interaction
        d_MoS = np.sqrt((binit / 2) ** 2 + (ainit / (2 * np.sqrt(3))) ** 2)
        MoS_ok = d_MoS < rc_MoS
        return MoMo_ok or MoS_ok or SS_ok

    def reset_calculator(self):
        """Reset calculator to use default parameters."""
        self._convert_parameters(self._def_params)
        self._calc_changed = True

    def compute(self, verbose=False, **kwargs):
        """Compute the energy per unit cell vs lattice constant of an MoS2
        system, given the parameter values of the KIM model.

        Parameters
        ----------
        **kwargs:
            User-defined parameters in a dictionary. The format of the
            dictionary is::

            kwargs = dict(param_name1: [idx, value], ...)

        Returns
        -------
        dict
            A dictionary containing the predictions
        """
        self.verbose = verbose
        if kwargs:
            self._convert_parameters(kwargs)

        if self._calc_changed:
            preds = self._ecell_v_latconst()
            for val, key in zip(preds, self.predictions):
                self.predictions[key].update({"value": val})

        self.computed = True
        self._calc_changed = False
        return self.predictions


if __name__ == "__main__":
    desc = "Compute energy vs lattice constant curve of MoS2 sheet."
    print(desc)

    # Simulation setup: define the model and parametersto use
    modelname = "SW_MX2_WenShirodkarPlechac_2017_MoS__MO_201919462778_001"
    parameters = {
        "A": [[0, 1, 2], [3.97818048, 11.37974144, 1.19073558]],
        "B": [[0, 1, 2], [0.44460213, 0.52666882, 0.90151527]],
        "p": [[0, 1, 2], [5, 5, 5]],
        "q": [[0, 1, 2], [0, 0, 0]],
        "sigma": [[0, 1, 2], [2.85295, 2.17517, 2.84133]],
        "gamma": [[0, 1, 2], [1.3566322, 1.3566322, 1.3566322]],
        "cutoff": [[0, 1, 2], [5.5466, 4.02692, 4.51956]],
        "lambda": [
            [0, 1, 2],
            [7.47675292e000, 8.15951812e000, 1.58101007e-322],
        ],
        "cosbeta0": [
            [0, 1, 2],
            [1.42856958e-001, 1.42856958e-001, 1.58101007e-322],
        ],
        "cutoff_jk": [
            [0, 1, 2],
            [3.86095e000, 5.54660e000, 1.58101e-322],
        ],
    }

    # Define the model
    model = EnergyVsLatticeConstant(modelname)

    # Get the predictions
    results = model.compute(verbose=True, **parameters)

    # Extract the information from the results dictionary
    a0 = results["a0"]["value"]
    b0 = results["b0"]["value"]
    Ec = results["ecell"]["value"]
    alist = results["alist"]["value"]
    blist = results["blist"]["value"]
    elist = results["elist"]["value"]

    # Plot the energy per unit cell curve
    plt.close("all")
    plt.figure()
    plt.title(f"$a={a0:0.2f}\ \AA, b={b0:0.2f}\ \AA, E_c={Ec:0.2f}$ (eV)")
    plt.plot(alist - a0, elist - Ec, "o-")
    plt.xlim(-0.5, 0.5)
    plt.xlabel(r"$a-a_0 (\AA)$")
    plt.ylabel(r"$E-E_c$ (eV)")

    plt.figure()
    plt.title(f"$a={a0:0.2f}\ \AA, b={b0:0.2f}\ \AA, E_c={Ec:0.2f}$ (eV)")
    plt.plot(alist - a0, blist - b0, "o-")
    plt.xlim(-0.5, 0.5)
    plt.xlabel(r"$a-a_0 (\AA)$")
    plt.ylabel(r"$b-b_0 (\AA)$")

    plt.show()
