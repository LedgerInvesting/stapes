from typing import Optional, List, Dict, Any, Set

import numpy as np

from ..utils import ConfigParameter, StanCode, get_data_type, DataType, StanStem
from ..data import DataCoords, DataValue


class Parameter(object):
    def __init__(self, name: str, dtype: str):
        self.name = name
        self.dtype: DataType = get_data_type(dtype)
        self.stem: Optional[StanStem] = None
        self.samples: Optional[Dict[str, np.ndarray]] = None
        self.pred_cache: Dict[Any, Any] = {}

    def set_samples(self, samples: Dict[str, np.ndarray]):
        """Set the samples attribute on the parameter from the overall sample dictionary."""

        def _demunge_name(name: str) -> str:
            if name == self.name:
                return ".."
            dundered_name = f"{self.name}__"
            dundered_len = len(dundered_name)
            if name[:dundered_len] == dundered_name:
                return "." + name[dundered_len:]
            else:
                return name

        self.samples = {
            _demunge_name(name): value
            for name, value in samples.items()
            if _demunge_name(name)[0] == "."
        }

    def evaluate(
        self,
        state: np.random.Generator,
        data: Dict[DataCoords, DataValue],
        coords: DataCoords
    ) -> np.ndarray:
        raise NotImplementedError("Must implement evaluate method")

    @property
    def likelihood_index(self) -> str:
        """The indexing expression that must be attached to the parameter when used in a
        likelihood expression."""
        raise NotImplementedError("Must implement likelihood_text method")

    @property
    def stan_code(self) -> StanCode:
        """Stan code to implement the parameter (except for the final transform)."""
        return self.stem.stan_code

    @property
    def config_parameters(self) -> List[ConfigParameter]:
        """A list of all configuration parameters associated with the parameter."""
        return self.stem.config_parameters

    @property
    def variables(self) -> Set[str]:
        """The set of all unique variables used by the parameter."""
        return set()
