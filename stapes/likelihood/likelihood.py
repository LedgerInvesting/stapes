from dataclasses import dataclass
from typing import Set, Tuple, List, Dict, Optional
from pathlib import Path

import numpy as np

from ..data import DataCoords, DataValue
from ..parse import ast, recurse_over_variables, get_all_parameters
from ..utils import StanCode, ConfigParameter, get_data_type, process_stem
from ..parameter import Parameter
from .random import variates_from_mean_variance
from .evaluate import evaluate_operand

FAMILY_NAME_LOOKUP = ["normal", "lognormal", "gamma"]
FAMILY_INDEX_LOOKUP = {
    name: idx+1 for idx, name in enumerate(FAMILY_NAME_LOOKUP)
}


@dataclass
class Likelihood(object):
    variable: str
    mean_def: ast.Operand
    variance_def: ast.Operand

    @property
    def offsets(self) -> Set[Tuple[int, int]]:
        """The set of all distinct offset operator combinations in the likelihood."""
        result = (
            {(0, 0)}
            | recurse_over_variables(self.mean_def, _variable_offset)
            | recurse_over_variables(self.variance_def, _variable_offset)
        )
        return result

    @property
    def variables(self) -> Set[str]:
        """The set of all distinct variables in the RHS of the likelihood."""
        return (
            {self.variable}
            | recurse_over_variables(self.mean_def, lambda x: x.name)
            | recurse_over_variables(self.variance_def, lambda x: x.name)
        )

    @property
    def parameters(self) -> Set[str]:
        return get_all_parameters(self.mean_def) | get_all_parameters(self.variance_def)

    @property
    def config_parameters(self) -> List[ConfigParameter]:
        """The list of all configuration parameters required by the likelihood."""
        return [
            ConfigParameter(
                name=f"{self.variable}__family",
                data_type=get_data_type("int"),
                default_value=1,
            )
        ]

    def stan_code(self, params: Dict[str, Parameter]) -> StanCode:
        stem = process_stem(
            Path(__file__).resolve().parent / "likelihood.stem",
            self.variable,
            {
                "mean_definition": _generate_op_text(self.mean_def, params),
                "variance_definition": _generate_op_text(self.variance_def, params),
            }
        )
        return stem.stan_code

    def predict(
        self,
        params: Dict[str, Parameter],
        state: np.random.Generator,
        distribution_id: int,
        data: Dict[DataCoords, DataValue],
        coords: DataCoords,
    ) -> np.ndarray:
        mean = evaluate_operand(self.mean_def, params, state, data, coords)
        variance = evaluate_operand(self.variance_def, params, state, data, coords)
        distribution = FAMILY_NAME_LOOKUP[distribution_id-1]
        return variates_from_mean_variance(mean,  variance, distribution, state)


def _generate_op_text(op: ast.Operand, params: Dict[str, Parameter]) -> str:
    """Recurse through an expression and generate the relevant Stan code."""
    if isinstance(op, ast.Operation):
        if len(op.operands) == 1:
            # Unary operators are all prefix operators
            op_text = _generate_op_text(op.operands[0], params)
            return f"{op.operator}{op_text}"
        elif len(op.operands) == 2:
            # Binary operators are all infix operators
            left_op_text = _generate_op_text(op.operands[0], params)
            right_op_text = _generate_op_text(op.operands[1], params)
            return f"({left_op_text} {op.operator} {right_op_text})"
        else:
            # All operators should be either unary or binary
            raise Exception(f"Unknown operation arity {len(op.operands)}")
    elif isinstance(op, ast.VariableOperand):
        offset_name = _get_offset_name(_variable_offset(op))
        if offset_name:
            return f"{op.name}[{offset_name}[n]]"
        else:
            return f"{op.name}[n]"
    elif isinstance(op, ast.OpCall):
        return op.name + "(" + _generate_op_text(op.arg, params) + ")"
    elif isinstance(op, str):
        param_name = op[1:]
        index_expr = params[param_name].likelihood_index
        return f"{param_name}{index_expr}"
    else:
        return str(op)


def _variable_offset(variable: ast.VariableOperand) -> Tuple[int, int]:
    """Compute the offset tuple from a list of offset properties."""
    experience_offset, development_offset = 0, 0

    for elem in variable.modifiers:
        if elem == "prev_dev":
            development_offset += 1
        elif elem == "prev_exp":
            experience_offset += 1
        else:
            raise Exception(f"Unknown modifier {elem}")

    return experience_offset, development_offset


def _get_offset_name(offset: Tuple[int, int]) -> Optional[str]:
    """Get the name of the Stan variable that has the appropriate offset indices."""
    if offset == (0, 0):
        return None

    exp_offset, dev_offset = offset
    return "Lag" + "T" * exp_offset + "D" * dev_offset
