from pathlib import Path

from ..utils import StanCode, process_stem


def get_variable_stan(name: str) -> StanCode:
    if name[-2:] == "Id":
        return process_stem(
            Path(__file__).resolve().parent / "coordinate.stem",
            name,
            {}
        ).stan_code
    else:
        return process_stem(
            Path(__file__).resolve().parent / "variable.stem",
            name,
            {}
        ).stan_code

