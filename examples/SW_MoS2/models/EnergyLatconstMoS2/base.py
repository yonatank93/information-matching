import numpy as np
from ase.calculators.kim import KIM
from ase import Atoms
import kimpy


class Model(object):
    """Parent class for ``kim_tests.tests``.

    Parameters
    ----------
    symbol: str
        String of the element to use, e.g. "Ar" or "Si".
    config: str
        String to describe the configuration of atoms, e.g. "bcc".
    model_id: str
        KIM ID of the interatomic model.
    """

    def __init__(self, symbol, config, model_id):
        self.symbol = symbol
        self.config = config
        self.model_id = model_id
        self.calc = self._create_calculator()
        self.predictions = None
        self.computed = False
        self.kim_model = self.create_model()[1]
        self.num_params = self.kim_model.get_number_of_parameters()
        self.default_params = self.get_model_parameters()

    def _create_calculator(self):
        calc = KIM(self.model_id)
        self._calc_changed = True
        return calc

    def create_model(self):
        """Wrapper to ``kimpy.model.create``. Extract information in the
        installed KIM model specified and store them as ``kim_model`` that is
        used to do some computation.

        Returns
        -------
        requestedUnitsAccepted: bool
        kim_model
        error: bool
        """
        requestedUnitsAccepted, kim_model = kimpy.model.create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            self.model_id,
        )

        return requestedUnitsAccepted, kim_model

    def get_model_parameters(self):
        """Get the model parameters as a dictionary or numpy array.

        Returns
        -------
        parameters: dict
            Dictionary containing parameters stored in the model.
        """
        parameters = {}

        for i in range(self.num_params):
            out = self.kim_model.get_parameter_metadata(i)
            _, extent, name, _ = out
            idx = np.arange(extent)
            parameters.update(self.calc.get_parameters(**{name: idx}))
        return parameters

    def get_default_cutoff(self):
        """Get default cutoff value.

        Returns
        -------
        default_cutoff: {floats, list}
            Values of the default cutoff(s)
        """
        try:
            default_cutoff = self.get_model_parameters()["cutoffs"]
        except KeyError:
            default_cutoff = self.get_model_parameters()["cutoff"]

        self.default_cutoff = default_cutoff
        return default_cutoff

    def _get_atomic_number(self):
        """Get atomic number of the element for one species system."""
        atom = Atoms(self.symbol)
        return atom.get_atomic_numbers()[0]

    def update_calculator(self, **kwargs):
        """Update ASE's kim calculator with the parameters given.

        Parameters
        ----------
        **kwargs:
            User-defined parameters in a dictionary. The format of the
            dictionary is::

            kwargs = dict(param_name1: [idx, value], ...)
        """
        self.calc.set_parameters(**kwargs)
        self._calc_changed = True

    def reset_calculator(self):
        """Reset calculator to use default parameters."""
        self.calc = self._create_calculator()
        self._calc_changed = True
