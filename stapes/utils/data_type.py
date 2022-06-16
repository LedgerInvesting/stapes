import math
from typing import Callable
from dataclasses import dataclass

import numpy as np


def _logit(x):
    return np.log(x / (1 - x))


def _inv_logit(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class DataType(object):
    """Specification of Stapes datatypes.

    Fields:
        name: Name of the datatype in Marrow source code.
        stan_dtype: Stan syntax for the datatype.
        transform: Stan expression to map from a real to the datatype.
        transform_fn: Python function that maps from the reals to the datatype's domain.
        inv_transform: Python function that maps from the datatype's domain to the reals.
        min_value: The minimum allowable value the parameter can take.
        max_value: The maximum allowable value the parameter can take.
        base_default_value: Assumed default value (on the scale of the reals).
    """

    name: str
    stan_dtype: str
    transform: str
    transform_fn: Callable[[float], float]
    inv_transform: Callable[[float], float]
    min_value: float = -math.inf
    max_value: float = math.inf
    base_default_value: float = 0.0


_DATA_TYPE_DEFS = [
    DataType(
        name="pos",
        stan_dtype="real<lower=0>",
        transform="exp",
        transform_fn=np.exp,
        inv_transform=np.log,
        min_value=0,
    ),
    DataType(
        name="real",
        stan_dtype="real",
        transform="",
        transform_fn=lambda x: x,
        inv_transform=lambda x: x,
    ),
    DataType(
        name="scale",
        stan_dtype="real<lower=0>",
        transform="",
        transform_fn=lambda x: x,
        inv_transform=lambda x: x,
        min_value=0,
        base_default_value=1,
    ),
    DataType(
        name="unit",
        stan_dtype="real<lower=0, upper=1>",
        transform="inv_logit",
        transform_fn=_inv_logit,
        inv_transform=_logit,
        min_value=0,
        max_value=1,
    ),
    DataType(
        name="int",
        stan_dtype="int<lower=1>",
        transform="",
        transform_fn=lambda x: x,
        inv_transform=lambda x: x,
        min_value=1,
        max_value=999_999,
    )
]

_DATA_TYPES = {dtype.name: dtype for dtype in _DATA_TYPE_DEFS}


def get_data_type(name: str) -> DataType:
    """Look up a Marrow datatype by its name."""
    try:
        return _DATA_TYPES[name]
    except KeyError:
        raise Exception(f"DataType name {name} is not fully supported yet")
