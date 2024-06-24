# Specify KIM potential and specify that all numerical values in this file
# are in LAMMPS `metals` units. Unit conversion is enabled, which means that
# unit coversion factors _u_* between the potential's unit system and the
# specified `metals` units will be computed. Multiplying by these factors
# translates the values in the file to those expected by the potential, and
# dividing by them translates from values computed using potential units to
# `metals` units. This is done throughout the file.
kim		init {{ modelname }} metal unit_conversion_mode
variable	stage string {{ stage }}

# Apply periodic boundary conditions, z direction is big enough so that images
# don't interact
boundary	p p s
atom_style	atomic

# Several variables needed to define the simulation box and atoms
variable	inita equal {{ a }}*${_u_distance}
variable	initb equal {{ b }}*${_u_distance}

# Create simulation box and atoms
include		{{ conf_path }}

# Define map between atomic species and LAMMPS atom types
kim		interactions Mo S

# Update the parameters of the model that are used in the simulation
kim		param set &
		{{ parameters }}

# Variables used to rescale the energy and stress so that the quantities in the
# thermo output are in the original metal units (eV and bars) even if we're
# running with a Simulator Model that uses different units.
variable	pe_metal equal c_thermo_pe/v__u_energy
variable	lx_metal equal lx/${_u_distance}
variable	ly_metal equal ly/${_u_distance}
variable	lz_metal equal lz/${_u_distance}
variable	press_metal equal c_thermo_press/v__u_pressure
compute		zpos all property/atom zu
compute		zmin all reduce min c_zpos
compute		zmax all reduce max c_zpos
variable	b_metal equal (c_zmax-c_zmin)/${_u_distance}

# Set what thermodynamic information to print to log
reset_timestep	0 
if "${stage} == equilibration" then "fix 1 all box/relax x 0.0 y 0.0"
thermo		100 # Print every timestep
thermo_style	custom step atoms v_pe_metal press v_press_metal &
		v_lx_metal v_ly_metal v_b_metal

# Perform static minimization using conjugate gradient method.
min_modify	line backtrack
minimize	1e-8 1e-8 6000 10000

# Define auxiliary variables to contain cohesive energy, and equilibrium
# lattice constants. The unit cell used in this simulation is twice as big as
# the primitive unit cell.
variable	natoms equal "count(all)" 
variable	ecell equal v_pe_metal*0.5  # Energy per unit cell
variable        a equal v_lx_metal/${nx}
variable        b equal v_b_metal
variable	ecoh equal "v_ecell/v_natoms"
