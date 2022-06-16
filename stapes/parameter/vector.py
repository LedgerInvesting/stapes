from typing import Dict, Set
from pathlib import Path

import numpy as np

from ..data import DataCoords, DataValue
from ..utils import process_stem
from .parameter import Parameter


class Vector(Parameter):
    """A basic vectorized parameter."""

    def __init__(self, name: str, dtype: str, args: Dict[str, str]):
        self.group_name = args.get("group")
        if not self.group_name:
            raise Exception("Vector parameter must have the `group` argument provided")
        self.anchor = args.get("anchor", "none")
        if self.anchor not in ["none", "first", "last"]:
            raise Exception("Vector argument `anchor` must be one of 'none', 'first', 'or 'last'")
        super().__init__(name, dtype)

        self.stem = process_stem(
            Path(__file__).resolve().parent / "vector.stem",
            self.name,
            {
                "anchor": self.anchor,
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
        idx = int(data[coords])
        max_idx = self.samples[".."].shape[1]
        # If the index value isn't present in the training data, throw an exception. Unlike factors,
        # vectors are unable to extrapolate to unseen indices.
        if idx > max_idx:
            raise Exception("Cannot extrapolate to index values not in training data")
        return self.samples[".."][:, idx - 1]

    @property
    def likelihood_index(self) -> str:
        return f"[{self.group_name}[n]]"

    @property
    def variables(self) -> Set[str]:
        """The set of all unique variables used by the parameter."""
        return {self.group_name}
