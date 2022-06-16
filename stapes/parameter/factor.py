from typing import Dict, Set
from pathlib import Path

import numpy as np
import scipy.stats

from ..data import DataCoords, DataValue, get_variable_value
from ..utils import process_stem
from .parameter import Parameter


class Factor(Parameter):
    """A basic hierarchical parameter."""

    def __init__(self, name: str, dtype: str, args: Dict[str, str]):
        self.group_name = args.get("group")
        if not self.group_name:
            raise Exception("Vector parameter must have the `group` argument provided")
        self.is_centered = args.get("is_centered", "false") == "true"
        super().__init__(name, dtype)

        self.stem = process_stem(
            Path(__file__).resolve().parent / "factor.stem",
            self.name,
            {
                "is_centered": self.is_centered,
                "range_name": f"{self.group_name}__count",
                "dtype": self.dtype.stan_dtype,
                "transform": self.dtype.transform,
            },
        )

    def evaluate(
        self,
        state: np.random.Generator,
        data: Dict[DataCoords, DataValue],
        coords: DataCoords
    ) -> np.ndarray:
        idx = int(get_variable_value(data, coords, self.group_name))
        max_idx = self.samples[".."].shape[1]
        # If the index id was present in the training data, use the posterior samples.
        if idx <= max_idx:
            return self.samples[".."][:, idx - 1]
        # If we've already generated samples for a new index value, use the cached samples.
        elif idx in self.pred_cache:
            return self.pred_cache[idx]
        # Otherwise, generate a set of samples for the new index value.
        else:
            mu = self.samples.get(".mu", 0.0)
            sigma = self.samples[".sigma"]
            raw_values = scipy.stats.norm(loc=mu, scale=sigma).rvs(random_state=state)
            values = self.dtype.transform_fn(raw_values)
            self.pred_cache[idx] = values
            return values

    @property
    def likelihood_index(self) -> str:
        return f"[{self.group_name}[n]]"

    @property
    def variables(self) -> Set[str]:
        """The set of all unique variables used by the parameter."""
        return {self.group_name}
