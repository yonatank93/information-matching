#!/usr/bin/env python3

import matplotlib
import pylab as plt
from ase.build import bulk
from ase.phonons import Phonons
from ase.calculators.kim.kim import KIM
import kim_python_utils.ase as kim_ase_utils

from byukim.kim_tests import LatticeConstantCubicEnergy

symbol = "Si"
lattice = "diamond"
model = "SW_StillingerWeber_1985_Si__MO_405512056662_006"

# Compute lattice constant
test_alat = LatticeConstantCubicEnergy(symbol, lattice, model)
latticeconstant = test_alat.compute()["a"]
print(latticeconstant)

# if lattice.lower() != "fcc":
#     raise ValueError("Only fcc lattices are currently supported")

# Check if atoms have an energy interaction for this model
atoms_interacting = kim_ase_utils.check_if_atoms_interacting(
    model, symbols=[symbol, symbol], check_force=False, etol=1e-6
)
if not atoms_interacting:
    raise RuntimeError(
        "The model provided, {}, does not possess a non-trivial "
        "energy interaction for species {} as required by this Test.  Aborting."
        "".format(model, symbol)
    )

# Set up crystal and calculator
calc = KIM(model)
atoms = bulk(symbol, lattice, a=latticeconstant)

# Phonon calculator
N = 7
ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
ph.run()

# Read forces and assemble the dynamical matrix
ph.read(acoustic=True)
ph.clean()

# Band structure
path = atoms.cell.bandpath("GXULGK", npoints=100)
bs = ph.get_band_structure(path)
branches = ["TA", "TA", "LA"]  # Hard-coded to FCC for now (1-atom basis)
# Extract 100 x 3 array of frequencies and convert eV -> meV
omega_kn = 1000 * bs.energies[0, :, :]
wave_numbers, special_points, special_point_names = bs.get_labels()

for i, name in enumerate(special_point_names):
    if name == "G":
        special_point_names[i] = r"$\Gamma$"

# Calculate phonon DOS
dos_energies, dos_weights = ph.dos(kpts=(50, 50, 50), npts=1000, delta=5e-4)
dos_energies *= 1000  # Convert to meV

# Plot the band structure and DOS
emax = 40  # meV
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_axes([0.1, 0.07, 0.67, 0.85])
for n, branch in enumerate(branches):
    omega_n = omega_kn[:, n]
    plt.plot(wave_numbers, omega_n, "r-", lw=2)

plt.xticks(special_points, special_point_names, fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(wave_numbers[0], wave_numbers[-1])
plt.ylabel("Frequency ($\mathrm{meV}$)", fontsize=22)
plt.grid(True)

plt.axes([0.8, 0.07, 0.17, 0.85])
plt.fill_between(
    dos_weights, dos_energies, y2=0, color="blue", edgecolor="k", lw=1
)
plt.ylim(0, emax)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel("DOS", fontsize=18)
