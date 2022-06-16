import re
import pathlib
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any, Union

from .stan import StanCode
from .config_parameter import ConfigParameter
from .data_type import get_data_type


class StemException(Exception):
    pass


@dataclass
class StanStem(object):
    stan_code: StanCode = field(default_factory=StanCode)
    config_parameters: List[ConfigParameter] = field(default_factory=list)
    stan_parameters: List[str] = field(default_factory=list)


@dataclass
class StemConditionalArm(object):
    condition: Optional[str]
    lines: List["StemLine"]


StemConditional = List[StemConditionalArm]
StemLine = Union[str, StemConditional]


def process_stem(
    stemfile: Union[str, pathlib.Path],
    namespace: str,
    environment: Dict[str, Any],
) -> StanStem:
    with open(stemfile, "r") as infile:
        raw_stem_lines = [line[:-1] for line in infile.readlines()]
    stem_lines = parse_stem(raw_stem_lines)
    resolved_lines = resolve_stem_lines(stem_lines, namespace, environment)

    stan_fragments = defaultdict(list)
    config_parameters = []
    stan_parameters = []
    context = None

    for line in resolved_lines:
        # Update state, extract, parameters, etc.
        raw_line = line
        line = line.strip()
        context, line = _update_context(line, raw_line, context)
        config_parameters, line = _update_config_parameters(line, context, config_parameters)
        stan_parameters, line = _update_stan_parameters(line, context, stan_parameters)
        if context is not None and line != "":
            stan_fragments[context].append(line)

    concat_stan_fragments = {context: "\n".join(lines) for context, lines in stan_fragments.items()}

    return StanStem(
        stan_code=StanCode(**concat_stan_fragments),
        config_parameters=config_parameters,
        stan_parameters=stan_parameters,
    )


def parse_stem(raw_stem_lines: List[str]) -> List[StemLine]:
    stem_lines = []
    while raw_stem_lines:
        new_line, raw_stem_lines = parse_stem_line(raw_stem_lines)
        stem_lines.append(new_line)
    return stem_lines


def parse_stem_line(raw_stem_lines: List[str]) -> Tuple[StemLine, List[str]]:
    """Parse a single stem line.

    A single logical stem line may correspond to multiple physical lines, so the function
    accepts a list of all remaining lines in the file and returns the lines that it
    has not consumed.
    """
    trimmed_line = raw_stem_lines[0].strip()
    if trimmed_line[:4] == "#if ":
        return parse_conditional(raw_stem_lines)
    else:
        return raw_stem_lines[0], raw_stem_lines[1:]


def parse_conditional(raw_stem_lines: List[str]) -> Tuple[StemLine, List[str]]:
    trimmed_line = raw_stem_lines[0].strip()
    conditionals = []
    condition = trimmed_line[4:]
    lines = []
    raw_stem_lines.pop(0)

    while True:
        if not raw_stem_lines:
            raise StemException("Reached EOF before a matching #endif")
        trimmed_line = raw_stem_lines[0].strip()
        if trimmed_line[:6] == "#elif ":
            conditionals.append(StemConditionalArm(condition=condition, lines=lines))
            raw_stem_lines.pop(0)
            condition = trimmed_line[6:]
            lines = []
        elif trimmed_line[:5] == "#else":
            conditionals.append(StemConditionalArm(condition=condition, lines=lines))
            raw_stem_lines.pop(0)
            condition = None
            lines = []
        elif trimmed_line[:6] == "#endif":
            conditionals.append(StemConditionalArm(condition=condition, lines=lines))
            raw_stem_lines.pop(0)
            break
        else:
            new_line, raw_stem_lines = parse_stem_line(raw_stem_lines)
            lines.append(new_line)

    if any([cond.condition is None for cond in conditionals[:-1]]):
        raise StemException("#else clauses cannot be followed by other #elif clauses")

    return conditionals, raw_stem_lines


def resolve_stem_lines(
    stem_lines: List[StemLine],
    namespace: str,
    environment: Dict[str, Any],
) -> List[str]:
    result = []
    for line in stem_lines:
        if isinstance(line, str):
            result.append(_resolve_symbols(line, namespace, environment))
        else:  # If it hits this branch, it must be a StemConditional
            for arm in line:
                if not arm.condition or eval(arm.condition, environment):
                    result += [_resolve_symbols(ln, namespace, environment) for ln in arm.lines]
                    break
    return result


# Stem lines that always map to a specific context
_CONTEXT_MAP = {
    "data {": "data",
    "transformed data {": "trans_data_decl",
    "parameters {": "param_decl",
    "transformed parameters {": "trans_decl",
    "model {": "model_decl",
    "}": None,
}

# Stem lines that may map to another context, depending on the current context
_CONDITIONAL_CONTEXT_MAP = {
    ("trans_data_decl", "// !definitions"): "trans_data_def",
    ("trans_decl", "// !definitions"): "trans_def",
    ("model_decl", "// !definitions"): "model_def",
}


def _update_context(line: str, raw_line: str, context: Optional[str]) -> Tuple[Optional[str], str]:
    if raw_line in _CONTEXT_MAP:
        return _CONTEXT_MAP[line], ""
    if (context, line) in _CONDITIONAL_CONTEXT_MAP:
        return _CONDITIONAL_CONTEXT_MAP[(context, line.strip())], ""
    else:
        return context, line


def _update_stan_parameters(line: str, context: str, params: List[str]) -> Tuple[List[str], str]:
    if context in ["param_decl", "trans_decl", "gq_decl"] and line[:6] == "$core ":
        new_param = line.split()[-1][:-1].split("[")[0]
        return params + [new_param], line[6:]
    else:
        return params, line


def _update_config_parameters(
    line: str,
    context: str,
    params: List[ConfigParameter],
) -> Tuple[List[ConfigParameter], str]:
    if context == "data" and line[:8] == "$config ":
        dtype_name, param_name = line[8:-1].split()
        dtype = get_data_type(dtype_name[1:])
        param = ConfigParameter(
            name=param_name,
            data_type=dtype,
            default_value=dtype.base_default_value
        )
        clean_line = f"{dtype.stan_dtype} {param_name};"
        return params + [param], clean_line
    else:
        return params, line


PURE_NAME = re.compile(r"\.\.")
NAMESPACE = re.compile(r"(\.)(@?[a-z][a-z0-9_]*)")
EXPRESSION = re.compile(r"@(([a-z][a-z0-9_]*)|({.+}))")


def _resolve_symbols(
    line: str,
    namespace: str,
    environment: Dict[str, Any],
) -> str:
    def _resolve_namespace(matchobj):
        return namespace + "__" + matchobj.group(2)

    def _resolve_expression(matchobj):
        raw_expr = matchobj.group(0)[1:]
        if raw_expr[0] == "{":
            raw_expr = raw_expr[1:-1]
        return str(eval(raw_expr, environment))

    line = re.sub(NAMESPACE, _resolve_namespace, line)
    line = re.sub(EXPRESSION, _resolve_expression, line)
    line = re.sub(PURE_NAME, namespace, line)
    return line
