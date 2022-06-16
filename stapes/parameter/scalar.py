from typing import Dict
from pathlib import Path

import numpy as np

from ..data import DataCoords, DataValue
from ..utils import process_stem
from .parameter import Parameter


class Scalar(Parameter):
    """A simple scalar without any additional structure."""

    def __init__(self, name: str, dtype: str, args: Dict[str, str]):
        if len(args) > 0:
            raise Exception("Scalar parameters do not accept any arguments")
        super().__init__(name, dtype)

        self.stem = process_stem(
            Path(__file__).resolve().parent / "scalar.stem",
            self.name,
            {
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
        return self.samples[".."]

    @property
    def likelihood_index(self) -> str:
        return ""
