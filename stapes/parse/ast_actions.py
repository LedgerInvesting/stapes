from .ast import (
    Likelihood,
    ParamSpec,
    ModelAst,
    VariableOperand,
    Operation,
    OpCall,
)


# noinspection PyMethodMayBeStatic
class ModelAstActions(object):
    def start(self, ast):
        param_specs = [e for e in ast if isinstance(e, ParamSpec)]
        likelihoods = [e for e in ast if isinstance(e, Likelihood)]

        return ModelAst(
            likelihoods=likelihoods,
            params=param_specs,
        )

    def likelihood(self, ast):
        return Likelihood(
            variable=ast["variable"],
            aspect=ast["aspect"],
            value=ast["value"],
        )

    def expression(self, ast):
        return self._resolve_operation(ast)

    def add_operand(self, ast):
        return self._resolve_operation(ast)

    def mul_operand(self, ast):
        return self._resolve_operation(ast)

    def exp_operand(self, ast):
        if isinstance(ast, dict) and "op" in ast:
            return Operation(operator=ast["op"], operands=ast["args"])
        else:
            return ast

    def _resolve_operation(self, ast):
        if isinstance(ast, (VariableOperand, Operation, str, float, OpCall)):
            return ast
        else:
            return Operation(operator=ast["op"], operands=ast["args"])

    def variable_operand(self, ast):
        return VariableOperand(name=ast["name"], modifiers=sorted(ast.get("modifiers", [])))

    def number(self, ast):
        return float(ast)

    def param_spec(self, ast):
        return ParamSpec(
            name=ast["param"],
            data_type=ast.get("dtype", "real"),
            param_type=ast["spec"][0],
            kwargs=ast["spec"][1],
        )

    def func_call(self, ast):
        args = {}
        for arg in ast.get("args", []):
            if arg == ",":
                continue
            else:
                args[arg["arg_name"]] = arg["value"]
        return ast["name"], args

    def op_call(self, ast):
        return OpCall(name=ast["name"], arg=ast["arg"])
