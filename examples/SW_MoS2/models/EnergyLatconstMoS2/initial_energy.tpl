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

run		0
variable	pe equal pe  # Initial potential energy
