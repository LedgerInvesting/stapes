from pathlib import Path

from ..utils import process_stem, StanStem


def get_variable_stan(name: str) -> StanStem:
    if name[-2:] == "Id":
        return process_stem(
            Path(__file__).resolve().parent / "coordinate.stem",
            name,
            {}
        )
    else:
        return process_stem(
            Path(__file__).resolve().parent / "variable.stem",
            name,
            {}
        )
