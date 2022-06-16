from typing import Dict, Optional

from .parameter import Parameter
from .scalar import Scalar
from .vector import Vector
from .factor import Factor


METHOD_LOOKUP = {
    None: Scalar,
    "scalar": Scalar,
    "vector": Vector,
    "factor": Factor,
}


def make_parameter(
    method: Optional[str], name: str, dtype: str, args: Dict[str, str]
) -> Parameter:
    """Create a parameter object from a method name, args, and kwargs."""
    try:
        return METHOD_LOOKUP[method](name, dtype, args)
    except KeyError:
        raise Exception(f"Unrecognized parameter method {method}")
