# Specify KIM potential and specify that all numerical values in this file
# are in LAMMPS `metals` units. Unit conversion is enabled, which means that
# unit coversion factors _u_* between the potential's unit system and the
# specified `metals` units will be computed. Multiplying by these factors
# translates the values in the file to those expected by the potential, and
# dividing by them translates from values computed using potential units to
# `metals` units. This is done throughout the file.
kim		init {{ modelname }} metal unit_conversion_mode

# Apply periodic boundary conditions, z direction is big enough so that images
# don't interact
boundary	p p p
atom_style	atomic

# # # # # # # # # # Create simulation box and atoms
# Several variables needed to define the simulation box and atoms
variable	inita equal {{ a }}*${_u_distance}
variable	nx equal 1  # Number of cell repetition in x direction

# Create simulation box and atoms
lattice		diamond ${inita}

# Create simulation box
region		simbox block 0 ${nx} 0 ${nx} 0 ${nx} units lattice
create_box	1 simbox
# Create atoms
create_atoms	1 box

# Define mass
mass		* 28.0855

# # # # # # # # # # KIM stuffs
# Define map between atomic species and LAMMPS atom types
kim		interactions Si

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

run		0

# Define auxiliary variables to contain cohesive energy, and equilibrium
# lattice constants. The unit cell used in this simulation is twice as big as
# the primitive unit cell.
variable	natoms equal "count(all)" 
variable	ecell equal v_pe_metal  # Energy per unit cell
variable        a equal v_lx_metal/${nx}
variable	ecoh equal "v_ecell/v_natoms"
