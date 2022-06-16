from dataclasses import dataclass
from typing import Callable, Optional
import math

from .data_type import DataType


@dataclass
class ConfigParameter(object):
    """A configuration parameter for a model that is exposed in the model constructor."""

    name: str
    data_type: DataType
    default_value: Optional[float] = None

    @property
    def inv_transform(self) -> Callable[[float], float]:
        return self.data_type.inv_transform

    @property
    def min_value(self) -> float:
        return -math.inf

    @property
    def max_value(self) -> float:
        return math.inf
