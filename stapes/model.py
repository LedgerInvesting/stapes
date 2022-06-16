from typing import Dict, Tuple, Optional, List
import tempfile

import cmdstanpy as csp

from .parameter import Parameter, make_parameter
from .likelihood import Likelihood
from .parse import parse_text
from .utils import StanCode, ConfigParameter, UTIL_FUNCTIONS
from .variable import get_variable_stan
from .data import build_stan_data


class Model(object):
    """A completely specified Stapes model."""
    def __init__(self, parameters: Dict[str, Parameter], likelihoods: Dict[str, Likelihood]):
        self.parameters = parameters
        self.likelihoods = likelihoods

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        """List of all distinct (dev_lag, exp_period) offset tuples used in the model."""
        result = set()
        for lik in self.likelihoods.values():
            result |= lik.offsets
        return list(result)

    @property
    def variables(self) -> List[str]:
        """List of all distinct variables used in the Marrow model."""
        result = set()
        for param in self.parameters.values():
            result |= param.variables
        for lik in self.likelihoods.values():
            result |= lik.variables
        return list(result)

    @property
    def config_parameters(self) -> List[ConfigParameter]:
        """A list of all configuration parameters the model accepts."""
        result = []
        for param in self.parameters.values():
            result += param.config_parameters
        for lik in self.likelihoods.values():
            result += lik.config_parameters
        return result

    @property
    def stan_code(self) -> StanCode:
        """Stan source code representation of the Marrow model."""
        result = StanCode(data="int<lower=1> N;\nint<lower=N> T;")
        for offset in self.offsets:
            result += _offset_to_stan_code(offset)
        for variable in self.variables:
            result += get_variable_stan(variable)
        for param in self.parameters.values():
            result += param.stan_code
        for lik in self.likelihoods.values():
            result += lik.stan_code(self.parameters)
        return result

    @property
    def full_stan_code(self) -> str:
        """Stan source code representation of the Marrow model, including all utility functions."""
        return UTIL_FUNCTIONS + str(self.stan_code)

    @property
    def stan_model(self) -> csp.CmdStanModel:
        # Write the Stan code to a tempfile so cmdstanpy can pick it up
        with tempfile.NamedTemporaryFile(suffix=".stan", delete=False) as tmp:
            tmp.write(str(self.full_stan_code).encode("utf-8"))
            tmp_name = tmp.name
        return csp.CmdStanModel(stan_file=tmp_name)

    def fit(self, train_data, config: Dict[str, float]):
        stan_data = build_stan_data(train_data, self.offsets)
        stan_model = self.stan_model
        fit = stan_model.sample(data={**stan_data, **config})
        samples = fit.stan_variables()
        for param in self.parameters.values():
            param.set_samples(samples)


def build_model(text: str):
    ast = parse_text(text)

    lik_aspects = {}
    for lik in ast.likelihoods:
        if lik.variable not in lik_aspects:
            lik_aspects[lik.variable] = {lik.aspect: lik.value}
        else:
            lik_aspects[lik.variable][lik.aspect] = lik.value

    likelihoods = {}
    for variable, aspects in lik_aspects.items():
        if set(aspects.keys()) != {"mean", "variance"}:
            raise Exception(f"Variable {variable} must have mean and variance aspect specified")
        likelihoods[variable] = Likelihood(
            variable=variable,
            mean_def=aspects["mean"],
            variance_def=aspects["variance"],
        )

    parameters = {}
    for param_ast in ast.params:
        param = make_parameter(
            param_ast.param_type,
            param_ast.name[1:],
            param_ast.data_type,
            param_ast.kwargs
        )
        parameters[param.name] = param

    # Check for declared parameters that aren't used in the model
    used_param_names = set()
    for lik in likelihoods.values():
        used_param_names |= lik.parameters
    for param_name in parameters:
        if param_name not in used_param_names:
            raise Exception(f"Parameter {param_name} is never used")

    # Add implicitly declared parameters
    for name in used_param_names:
        if name not in parameters:
            parameters[name] = make_parameter(None, name, "real", {})

    return Model(parameters=parameters, likelihoods=likelihoods)


def _offset_to_stan_code(offset: Tuple[int, int]) -> StanCode:
    if offset == (0, 0):
        return StanCode()

    return StanCode(data=f"array[N] int<lower=1, upper=T> {_get_offset_name(offset)};")


def _get_offset_name(offset: Tuple[int, int]) -> Optional[str]:
    """Get the name of the Stan variable that has the appropriate offset indices."""
    if offset == (0, 0):
        return None

    exp_offset, dev_offset = offset
    return "Lag" + "T" * exp_offset + "D" * dev_offset
