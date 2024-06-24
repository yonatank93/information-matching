#!/usr/bin/env python
"""This is a modified version of the Elastic constant cubic test in OpenKIM."""

from __future__ import print_function
from ase.build import bulk
from ase.units import GPa
import numpy as np
import numdifftools as ndt
from scipy.optimize import fmin
from scipy.optimize import minimize
import json


# Dependencies
from .. import base
from ..latticeconstant import LatticeConstantCubicEnergy
from . import config as C


class ElasticConstantsCubic(base.Model):
    """Determine the cubic elastic constants by numerically determining the
    Hessian.
    """

    def __init__(self, symbol, config, model_id):
        super().__init__(symbol, config, model_id)
        self.num_pred = 6
        self.pred_keys = ["a", "ecoh", "c11", "c12", "c44", "bulk_modulus"]

    def _setup(self, **kwargs):
        if kwargs:
            self.update_calculator(**kwargs)
        a_ecoh = LatticeConstantCubicEnergy(
            self.symbol, self.config, self.model_id
        ).compute(**kwargs)
        self.latticeconst = a_ecoh["a"]
        self.energy = a_ecoh["ecoh"]
        self.slab = self._create_slab()
        self.o_cell = self.slab.get_cell()
        self.slab.set_calculator(self.calc)
        self.o_volume = self.slab.get_volume()

    def _create_slab(self):
        slab = bulk(
            self.symbol,
            a=self.latticeconst,
            crystalstructure=self.config,
            cubic=True,
        )
        return slab

    def _voigt_to_matrix(self, voigt_vec):
        """Convert a voigt notation vector to a matrix"""
        matrix = np.zeros((3, 3))
        matrix[0, 0] = voigt_vec[0]
        matrix[1, 1] = voigt_vec[1]
        matrix[2, 2] = voigt_vec[2]
        matrix[tuple([[1, 2], [2, 1]])] = voigt_vec[3]
        matrix[tuple([[0, 2], [2, 0]])] = voigt_vec[4]
        matrix[tuple([[0, 1], [1, 0]])] = voigt_vec[5]

        return matrix

    def _get_energy_from_positions(self, pos):
        natoms = int(len(pos) / 3)
        self.slab.set_positions(np.reshape(pos, (natoms, 3)))
        energy = self.slab.get_potential_energy()
        return energy

    def _get_gradient_from_positions(self, pos):
        natoms = int(len(pos) / 3)
        self.slab.set_positions(np.reshape(pos, (natoms, 3)))
        forces = self.slab.get_forces()
        return -forces.flatten()

    def _energy_from_strain(self, strain_vec):
        """Apply a strain according to the strain_vec"""
        strain_mat = self._voigt_to_matrix(strain_vec)
        old_cell = self.o_cell
        new_cell = old_cell + np.dot(old_cell, strain_mat)
        self.slab.set_cell(new_cell, scale_atoms=True)
        if self.config == "diamond":
            pos0 = self.slab.get_positions().flatten()
            res = minimize(
                self._get_energy_from_positions,
                pos0,
                jac=self._get_gradient_from_positions,
            )
            energy = res["fun"]
        else:
            energy = self.slab.get_potential_energy()

        return (energy / self.o_volume) / GPa

    def _energy_from_scale(self, scale):
        old_cell = self.o_cell
        new_cell = old_cell * (1 + scale)
        self.slab.set_cell(new_cell, scale_atoms=True)
        energy = self.slab.get_potential_energy()
        return energy

    def _results(self):
        """Return the cubic elastic constants"""
        # get the minimum
        self.minscale = float(fmin(self._energy_from_scale, 0, xtol=0, ftol=1e-7))
        self.oo_cell = self.o_cell.copy()
        self.o_cell = self.o_cell * (1 + self.minscale)

        func = self._energy_from_strain
        hess = ndt.Hessian(func, step=0.001, full_output=True)

        elastic_constants, info = hess(np.zeros(6, dtype=float))
        error_estimate = info.error_estimate

        inds11 = tuple([[0, 1, 2], [0, 1, 2]])
        C11 = elastic_constants[inds11].mean()
        C11sig = np.sqrt(((1.0 / 3 * error_estimate[inds11]) ** 2).sum())

        inds12 = tuple([[1, 2, 2, 0, 0, 1], [0, 0, 1, 1, 2, 2]])
        C12 = elastic_constants[inds12].mean()
        C12sig = np.sqrt(((1.0 / 6 * error_estimate[inds12]) ** 2).sum())

        inds44 = tuple([[3, 4, 5], [3, 4, 5]])
        C44 = 1.0 / 4 * elastic_constants[inds44].mean()
        C44sig = 1.0 / 4 * np.sqrt(((1.0 / 3 * error_estimate[inds44]) ** 2).sum())

        B = 1.0 / 3 * (C11 + 2 * C12)
        Bsig = np.sqrt((1.0 / 3 * C11sig) ** 2 + (2.0 / 3 * C12sig) ** 2)

        excessinds = tuple(
            [
                [
                    3,
                    4,
                    5,
                    3,
                    4,
                    5,
                    3,
                    4,
                    5,
                    0,
                    1,
                    2,
                    4,
                    5,
                    0,
                    1,
                    2,
                    3,
                    5,
                    0,
                    1,
                    2,
                    3,
                    4,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                ],
            ]
        )
        excess = np.abs(elastic_constants[excessinds]).mean()
        excess_sig = np.sqrt(((1.0 / 24 * error_estimate[excessinds]) ** 2).sum())

        results_dict = {
            "C11": C11,
            "C11_sig": C11sig,
            "C12": C12,
            "C12_sig": C12sig,
            "C44": C44,
            "C44_sig": C44sig,
            "B": B,
            "B_sig": Bsig,
            "excess": excess,
            "excess_sig": excess_sig,
            "units": "GPa",
            "element": self.symbol,
            "crystal_structure": self.config,
            "space_group": C.SPACE_GROUPS[self.config],
            "wyckoff_code": C.WYCKOFF_CODES[self.config],
            "lattice_constant": self.latticeconst,
            "scale_discrepency": self.minscale,
        }

        return results_dict

    def compute(self, **kwargs):
        """The calculation returns the equilibrium lattice constant, cohesive
        energy, c11, c12, c44, and bulk modulus.

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
        normed_basis = {
            lattice: json.dumps(
                bulk(self.symbol, self.config, a=1, cubic=True).positions.tolist(),
                separators=(" ", " "),
            )
            for lattice in C.SPACE_GROUPS.keys()
        }

        # get lattice constant and energy
        self._setup(**kwargs)

        if self._calc_changed:
            count = bulk(self.symbol, self.config, a=1, cubic=True).positions.shape[0]
            results = self._results()
            results.update({"basis_coordinates": normed_basis[self.config]})

            # Repeat symbols to match normed basis
            results.update({"species": '" "'.join([self.symbol] * count)})

            self.num_pred = 6
            self.predictions = {
                self.pred_keys[0]: self.latticeconst,
                self.pred_keys[1]: self.energy,
                self.pred_keys[2]: results["C11"],
                self.pred_keys[3]: results["C12"],
                self.pred_keys[4]: results["C44"],
                self.pred_keys[5]: results["B"],
            }

        self.computed = True
        self._calc_changed = False
        return self.predictions
