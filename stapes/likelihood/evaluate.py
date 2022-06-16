from typing import Dict, Union

import numpy as np

from ..parse import ast
from ..data import DataCoords, DataValue
from ..parameter import Parameter


def _minus(*args):
    # The `-` operator isn't a clean lambda because it needs to handle the unary (negation)
    # and binary (subtraction) cases
    if len(args) == 1:
        return -args[0]
    else:
        return args[0] - args[1]


def logit(x):
    return np.log(x / (1 - x))


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


# A lookup from operator names to operator functions.
OPERATIONS = {
    "+": lambda x, y: x + y,
    "-": _minus,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "^": lambda x, y: x ** y,
}


# A lookup from function names to function implementations.
OP_CALLS = {
    "log": np.log,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "logit": logit,
    "inv_logit": inv_logit,
}


def evaluate_operand(
    operand: ast.Operand,
    params: Dict[str, Parameter],
    state: np.random.Generator,
    data: Dict[DataCoords, DataValue],
    coords: DataCoords,
) -> Union[np.ndarray, float]:
    """Evaluate a single Operand in a likelihood expression at a single predicted cell."""
    if isinstance(operand, ast.VariableOperand):
        # If it's a variable, unpack it into a variable name and offset
        variable = operand.name
        dev_offset = sum([1 if mod == "prev_dev" else 0 for mod in operand.modifiers])
        exp_offset = sum([1 if mod == "prev_exp" else 0 for mod in operand.modifiers])
        field, slc, dev, exp = coords
        # Then use data_accessor to get the value of the variable for this cell
        return data[(field, slc, dev-dev_offset, exp-exp_offset)]
    elif isinstance(operand, float):
        # If it's a float, it's a plain numeric literal.
        return operand
    elif isinstance(operand, ast.Operation):
        # If it's an operation, evaluate each of the operands, then apply the operator
        clean_sub_operands = [
            evaluate_operand(op, params, state, data, coords) for op in operand.operands
        ]
        return OPERATIONS[operand.operator](*clean_sub_operands)
    elif isinstance(operand, ast.OpCall):
        # If it's a function call, evaluate the operand, then apply the function
        clean_arg = evaluate_operand(operand.arg, params, state, data, coords)
        return OP_CALLS[operand.name](clean_arg)
    elif isinstance(operand, str):
        # If it's a string, it's the name of a parameter. Clean up the parameter name,
        # then evaluate it at the current cell.
        param = params[operand[1:]]
        return param.evaluate(state, data, coords)
    else:
        raise Exception(f"Unrecognized Operand type {operand.__class__.__name__}")
