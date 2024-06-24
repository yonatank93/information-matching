import numpy as np
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.phonons import Phonons
import kim_python_utils.ase as kim_ase_utils

# Dependencies
from .. import base
from ..latticeconstant import LatticeConstantCubicEnergy


# Some physical constants
h = 4.1357e-15  # Planck's constant in eV s
c = 2.99792458e10  # Speed of light in vacuum in cm/s


class PhononDispersionCurveCubic(base.Model):
    """A class that uses ASE to compute the phonon dispersion curve of a cubic
    lattice at 0 K.

    Parameters
    ----------
    symbol: str
        Element to simulate, e.g., "Si".
    lattice: str ("bcc", "fcc", "sc", "diamond")
        Structure of atoms.
    model_id: str
        KIM_ID for the potential.
    a: float or None (optional)
        Lattice constant of the cubic lattice. If None, then the equilibrium
        lattice constant will be computed using the potential specified.
    bandpath: str (optional)
        String containing the directions of the band path.
    npoitns: int (optional)
        Number of pointst to compute for each mode.
    """

    def __init__(self, symbol, lattice, model_id, a=None, bandpath="GXULGK", npoints=100):
        if lattice.lower() not in ["bcc", "fcc", "sc", "diamond"]:
            raise ValueError("Only cubic lattices are currently supported")

        super().__init__(symbol, lattice, model_id)
        self.lattice = self.config
        # Check if atoms have an energy interaction for this model
        atoms_interacting = kim_ase_utils.check_if_atoms_interacting(
            self.model_id,
            symbols=[self.symbol, self.symbol],
            check_force=False,
            etol=1e-6,
        )
        if not atoms_interacting:
            raise RuntimeError(
                "The model provided, {}, does not possess a non-trivial "
                "energy interaction for species {} as required by this Test. "
                "Aborting.".format(self.model_id, symbol)
            )

        # Band path
        self.bandpath = bandpath
        self.bs = None

        self._a = a
        self.npoints = npoints

    def _get_lattice_constant(self):
        """Compute the equilibrium lattice constant."""
        test_alat = LatticeConstantCubicEnergy(self.symbol, self.lattice, self.model_id)
        return test_alat.compute()["a"]

    def compute(self, **kwargs):
        """Compute the phonon dispersion curve.

        Parameters
        ----------
        **kwargs: dict
            Parameters of the potential.

        Returns
        -------
        predictions: {"energies", "labels"}
        """
        # Update model parameters in the calculator
        self.update_calculator(**kwargs)

        if self._calc_changed:
            # Set up crystal
            if self._a is None:
                self.latticeconstant = self._get_lattice_constant()
            else:
                self.latticeconstant = self._a
            atoms = bulk(self.symbol, self.lattice, a=self.latticeconstant)

            # Phonon calculator
            N = 7
            ph = Phonons(atoms, self.calc, supercell=(N, N, N), delta=0.05)
            ph.run()

            # Read forces and assemble the dynamical matrix
            ph.read(acoustic=True)
            ph.clean()

            path = atoms.cell.bandpath(self.bandpath, npoints=self.npoints)
            self.bs = ph.get_band_structure(path)

            # Parse the result
            energies = self.bs.energies
            labels = self.bs.get_labels()
            self.predictions = {"energies": energies, "labels": labels}

        self.computed = True
        self._calc_changed = False
        return self.predictions

    def plot(self, *args, **kwargs):
        self.bs.plot(*args, **kwargs)


if __name__ == "__main__":
    symbol = "Si"
    lattice = "diamond"
    model_list = [
        # "LJ_ElliottAkerson_2015_Universal__MO_959249795837_003",
        "SW_StillingerWeber_1985_Si__MO_405512056662_006",
        "EDIP_JustoBazantKaxiras_1998_Si__MO_958932894036_002",
        "Tersoff_LAMMPS_ErhartAlbe_2005_SiC__MO_903987585848_005",
        # "ThreeBodyBondOrder_KDS_KhorDasSarma_1988_Si__MO_722489435928_000",
        # "MFF_MistriotisFlytzanisFarantos_1989_Si__MO_080526771943_001",
        # "MEAM_LAMMPS_Lee_2007_Si__MO_774917820956_000",
    ]
    modelname_short = [
        # "LJ",
        "SW",
        "EDIP",
        "Tersoff",
        # "KDS",
        # "MFF",
        # "MEAM",
    ]
    colors = [
        # "black",
        "red",
        "green",
        "blue",
        # "cyan",
        # "magenta",
        # "orange",
    ]
    bandpath = "GXWKGLUWLUX"
    a = 5.47  # Lattice constant in angstrom, from materials project

    _, ax = plt.subplots()

    for ii, model in enumerate(model_list):
        print(modelname_short[ii])
        # Compute band structure
        model_bs = PhononDispersionCurveCubic(symbol, lattice, model, a, bandpath)
        preds = model_bs.compute()

        xcoords, labels_xcoords, labels = preds["labels"]
        energies = preds["energies"]
        if ii == 0:
            energies_ensemble = energies
        else:
            energies_ensemble = np.concatenate((energies_ensemble, energies), axis=0)

        for jj, eng in enumerate(energies[0].T):
            if jj == 0:
                label = modelname_short[ii]
            else:
                label = None
            ax.plot(xcoords, eng / h / c, c=colors[ii], label=label)

    # # Plot the errorbars
    # energies_means = np.mean(energies_ensemble / h / c, axis=0)
    # energies_errors = np.std(energies_ensemble / h / c, axis=0)
    # for eng, err in zip(energies_means.T, energies_errors.T):
    #     ax.errorbar(xcoords, eng, err, c="k")

    for x in labels_xcoords:
        ax.axvline(x, ls="--")
    ax.set_xlim(labels_xcoords[[0, -1]])
    ax.set_xticks(labels_xcoords, labels)

    ax.set_ylabel(r"Frequency $(cm^{-1})$")
    ax.set_ylim(bottom=0.0)
    ax.legend(bbox_to_anchor=(1, 1))
    plt.show()
