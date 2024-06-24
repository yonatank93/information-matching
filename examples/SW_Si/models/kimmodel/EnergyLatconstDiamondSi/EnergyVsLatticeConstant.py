from pathlib import Path
import os

import numpy as np
import jinja2
from .. import base

try:
    import lammps
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Module ``lammps`` not found, this test cannot be used"
    ) from exc

CUR_DIR = Path(__file__).absolute().parent

# Initiate dictionary to store the results
results = {
    "a0": {"desc": "Equilibrium lattice constant a", "unit": "angstrom"},
    "ecoh": {"desc": "Energy per primitive unit cell", "unit": "eV/cell"},
    "alist": {"desc": "List of lattice constants a", "unit": "angstrom"},
    "elist": {"desc": "List of energies per unit cell", "unit": "eV/cell"},
}

# A list of perturbation to try as the initial guess for the equilibrium
# lattice constant
a0_init = 5.4309575703125  # 5.431 # Initial guess of the equilibrium from NIST


class EnergyVsLatticeConstant(base.Model):
    """Compute the energy per primitive unit cell vs lattice constant curve of
    an diamond silocon. This is done by first equilibrating the system and get the
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
        KIM model ID to use. Here, we only support SW-type of potential.
    da_list: np.ndarray (optional)
        List of :math:`\Delta a` to use.
    lmpfil: str or Path-like (optional)
        Path to a file in which the lammps script will be written.
    """

    def __init__(self, model_id, da_list=np.arange(-0.5, 0.51, 0.05), lmpfile=None):
        super().__init__("Si", None, model_id)
        self.predictions = results
        self.pred_keys = list(self.predictions)
        self.num_pred = len(self.predictions)
        self.da_list = da_list

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
            with open(CUR_DIR / "equilibration.tpl", "r") as f:
                self.equilibration_tpl = f.read()
            with open(CUR_DIR / "measurement.tpl", "r") as f:
                self.measurement_tpl = f.read()
        else:
            # Write lammps file and use lmp
            file_loader = jinja2.FileSystemLoader(CUR_DIR)
            self._env = jinja2.Environment(loader=file_loader)
            # Templates
            self.equilibration_tpl = "equilibration.tpl"
            self.measurement_tpl = "measurement.tpl"
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
        a0, ecoh = self._ecoh(a0_init, self.equilibration_tpl)
        return a0, ecoh

    def _ecoh(self, a, lammps_tpl):
        """Compute the energy per unit cell and lattice constant b given
        lattice constant a.
        """
        # Load the template file
        if self.lmpfile is None:
            template = self._env.from_string(lammps_tpl)
        else:
            template = self._env.get_template(lammps_tpl)

        # Generate lammps script from the template
        script = template.render(
            modelname=self.model_id,
            a=a,
            parameters=self._params_str,
        )

        # Create lammps object
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
        ecoh = lmp.extract_variable("ecoh")
        lmp.close()

        return a, ecoh

    def _ecoh_v_latconst(self):
        """Generate a list of lattice constants a from the equilibrium lattice
        constants and run lammps simulation to get the energy per unit cell
        for given lattice constant.
        """
        # Get the equilibrium state's predictions
        a0, e0 = self._equilibrium_state()

        # Change the lattice constant and run lammps simulation
        alist = self.da_list + a0

        # Iterate over the lattice constant a
        elist = np.zeros(len(alist))
        for ii, aa in enumerate(alist):
            # print(f"A{ii}, {aa}")
            a, ecoh = self._ecoh(aa, self.measurement_tpl)
            elist[ii] = ecoh

        return a0, e0, alist, elist

    def reset_calculator(self):
        """Reset calculator to use default parameters."""
        self._convert_parameters(self._def_params)
        self._calc_changed = True

    def compute(self, **kwargs):
        """Compute the energy per unit cell vs lattice constant of an Si
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
        if kwargs:
            self._convert_parameters(kwargs)

        if self._calc_changed:
            preds = self._ecoh_v_latconst()
            for val, key in zip(preds, self.predictions):
                self.predictions[key].update({"value": val})

        self.computed = True
        self._calc_changed = False
        return self.predictions
