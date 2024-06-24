import os
import jinja2
import lammps
from kliff.models import KIMModel
import numpy as np


class LatticeConstantCubicEnergy:
    command_tpl = """
kim             init {{ potential }} metal unit_conversion_mode
boundary	p p p
atom_style	atomic

# # # # # # # # # # Create simulation box and atoms
variable        ainit equal 4.04*${_u_distance}
lattice		{{ lattice }} ${ainit}
# Create simulation box
region		simbox block 0 1 0 1 0 1 units lattice
create_box	1 simbox
# Create atoms
create_atoms	1 box
mass		* 1.0  # Mass is not important for this static calculation

# # # # # # # # # # KIM stuffs
# Define map between atomic species and LAMMPS atom types
kim		interactions {{ element }}

# Update the parameters of the model that are used in the simulation
{{ parameters }}

# # # # # # # # # # Equilibration
variable	pe_metal equal c_thermo_pe/v__u_energy
variable	lx_metal equal lx/${_u_distance}
variable	ly_metal equal ly/${_u_distance}
variable	lz_metal equal lz/${_u_distance}
variable	press_metal equal c_thermo_press/v__u_pressure

# Set what thermodynamic information to print to log
reset_timestep	0
fix		1 all box/relax iso 0.0 vmax 0.001
thermo		10
thermo_style	custom step atoms v_pe_metal press v_press_metal v_lx_metal

# Perform static minimization using conjugate gradient method.
min_style       cg
min_modify	line backtrack
minimize	1e-25 1e-25 5000 10000

variable	natoms equal "count(all)"
variable	ecell equal v_pe_metal  # Energy per unit cell
variable        a equal v_lx_metal
variable	ecoh equal "v_ecell/v_natoms"
"""

    def __init__(self, element, lattice, potential):
        self.element = element
        self.lattice = lattice
        self.potential = potential

        # These lines are to get information about the parameters
        self.kimmodel = KIMModel(potential)
        params_kliff = self.kimmodel.get_model_params()
        self.params_dict = {name: item.value for name, item in params_kliff.items()}
        self.param_names = list(self.params_dict)

        # For caching
        self._calc_changed = False
        self.computed = False
        self.predictions = None

    def _compute_lattice_constant(self, params_str):
        # Render lammps commands
        env = jinja2.Environment()
        template = env.from_string(self.command_tpl)
        command = template.render(
            element=self.element,
            lattice=self.lattice,
            potential=self.potential,
            parameters=params_str,
        )

        # Run lammps
        lmp = lammps.lammps(cmdargs=["-screen", os.devnull, "-nocite"])
        try:
            lmp.commands_string(command)
        except Exception as e:
            lmp.close()
            raise e

        # Extract predictions
        a = lmp.extract_variable("a")
        ecoh = lmp.extract_variable("ecoh")
        lmp.close()
        return a, ecoh

    def _convert_params_dict_to_lammps_command(self, params_dict):
        """For each parameter, we want to have string
        kim param set 1:nparams <params_list>
        """
        params_commands_list = []
        for name, item in params_dict.items():
            # Each item contains the indices and parameter values
            idx = item[0]
            values = item[1]
            nvals = len(values)

            if np.all(sorted(idx) == np.arange(nvals)):
                # We change all values of the parameters
                if nvals == 1:
                    # There is only 1 value
                    idx_str = "1"
                    val_str = str(values[0])
                else:
                    # The parameter has multiple values
                    idx_str = f"1:{nvals}"
                    val_str = " ".join([str(val) for val in values])
                params_command = f"kim param set {name} {idx_str} {val_str}"
            else:
                # The case where we only change some values of the parameters. Remember
                # that LAMMPS uses base-1 index.
                param_str = [f"{name} {ii+1} {val}" for ii, val in zip(idx, values)]
                params_command = "kim param set " + " ".join(param_str)

            # Collect the commands to update parameters
            params_commands_list.append(params_command)

        # Combined commands to set parameters
        params_command = "\n".join(params_commands_list)
        return params_command

    def reset(self):
        self._calc_changed = False
        self.computed = False
        self.predictions = None

    def compute(self, **params):
        if not bool(params):
            # No parameters input
            if not self.computed:
                # The first time this method is called. Do the calculation.
                a, ecoh = self._compute_lattice_constant("")
                self.predictions = {"a": a, "ecoh": ecoh}
            else:
                # When there is no parameters input, return the results of the previous
                # calculation.
                pass
        else:
            # The parameter dictionary is not empty, we need to use the requested
            # parameters
            # Convert parameter dictionary into parameter command
            params_command = self._convert_params_dict_to_lammps_command(params)

            # Run the calculation
            a, ecoh = self._compute_lattice_constant(params_command)
            self.predictions = {"a": a, "ecoh": ecoh}
            self._calc_changed = True

        self.computed = True
        return self.predictions


if __name__ == "__main__":
    element = "Al"
    lattice = "fcc"
    potential = "EAM_Dynamo_MishinFarkasMehl_1999_Al__MO_651801486679_005"

    model = LatticeConstantCubicEnergy(element, lattice, potential)

    # Get parameter values
    kimmodel = KIMModel(potential)
    params_dict = kimmodel.get_model_params()
    cutoff = params_dict["cutoff"].value
    deltaRho = params_dict["deltaRho"].value
    deltaR = params_dict["deltaR"].value
    embeddingData = params_dict["embeddingData"].value
    rPhiData = params_dict["rPhiData"].value
    densityData = params_dict["densityData"].value
    nrho = len(embeddingData)
    nr = len(rPhiData)

    # Generate parameter dictionary
    params_dict = {
        "embeddingData": [np.arange(nrho), embeddingData * (1 + np.random.randn())],
        "rPhiData": [np.arange(nr), rPhiData * (1 + np.random.randn())],
        "densityData": [np.arange(nr), densityData * (1 + np.random.randn())],
    }
    preds = model.compute(**params_dict)
    print(preds)
