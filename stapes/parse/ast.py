from typing import List, Union, Dict, Callable, Any, Set
from dataclasses import dataclass


@dataclass
class Operation(object):
    """An invocation of a prefix or infix operator (e.g., `+`, `-`, `*`, unary minus)."""
    operator: str
    operands: List["Operand"]


@dataclass
class VariableOperand(object):
    """A value in a likelihood expression that refers to input variable."""
    name: str
    modifiers: List[str]


@dataclass
class OpCall(object):
    """An invocation of a function in a likelihood expression (e.g., `log`, `sqrt`)."""
    name: str
    arg: "Operand"


"""A legal value in the right-hand side of a likelihood expression.
Bare floats are numeric literals, bare strings are references to parameter names.
"""
Operand = Union[OpCall, Operation, VariableOperand, str, float]


@dataclass
class Likelihood(object):
    """A single likelihood expression."""
    variable: str
    aspect: str
    value: Operand


@dataclass
class ParamSpec(object):
    """A function call within the context of a parameter specification."""
    name: str
    data_type: str
    param_type: str
    kwargs: Dict[str, Union[str, float]]


@dataclass
class ModelAst(object):
    params: List[ParamSpec]
    likelihoods: List[Likelihood]


def recurse_over_variables(expr: Operand, func: Callable[[VariableOperand], Any]) -> Set[Any]:
    if isinstance(expr, VariableOperand):
        return {func(expr)}
    elif isinstance(expr, Operation):
        return set().union(
            *[recurse_over_variables(operand, func) for operand in expr.operands]
        )
    elif isinstance(expr, OpCall):
        return recurse_over_variables(expr.arg, func)
    else:
        return set()


def get_all_parameters(expr: Operand) -> Set[str]:
    if isinstance(expr, str):
        return {expr[1:]}
    elif isinstance(expr, Operation):
        return set().union(
            *[get_all_parameters(operand) for operand in expr.operands]
        )
    elif isinstance(expr, OpCall):
        return get_all_parameters(expr.arg)
    else:
        return set()
