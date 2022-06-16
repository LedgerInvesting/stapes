import re
from dataclasses import dataclass


@dataclass
class StanCode(object):
    """Text-oriented representation of a Stan program.

    Stan programs are broken up into discrete blocks -- data, transformed data, parameters, etc.
    Within each of these discrete blocks, there are additional structural restrictions, such as
    requiring declarations to precede definitions. Finally, there are some useful categorizations
    within blocks that aren't strictly required by the language, like keeping prior and likelihood
    sampling statements separate within the model block.

    The StanCode class represents each of these logical units of a Stan program as a separate
    string-typed field. This makes composition of a Stan program straightforward: the sum of two
    separate Stan programs is the sum of each of their fields. It is also easy to generate a
    valid `.stan` file from this representation.
    """

    data: str = ""
    trans_data_decl: str = ""
    trans_data_def: str = ""
    param_decl: str = ""
    trans_decl: str = ""
    trans_def: str = ""
    model_decl: str = ""
    model_def: str = ""

    def __add__(self, other: "StanCode") -> "StanCode":
        """Concatenate two StanCode fragments."""
        # Special case for StanCode + 0.
        if isinstance(other, int):
            return self
        return StanCode(
            data=self.data + other.data,
            trans_data_decl=self.trans_data_decl + other.trans_data_decl,
            trans_data_def=self.trans_data_def +other.trans_data_def,
            param_decl=self.param_decl + other.param_decl,
            trans_decl=self.trans_decl + other.trans_decl,
            trans_def=self.trans_def + other.trans_def,
            model_decl=self.model_decl + other.model_decl,
            model_def=self.model_def + other.model_def,
        )

    def __radd__(self, other: "StanCode") -> "StanCode":
        return other + self

    def __str__(self) -> str:
        """Convert a StanCode object to a string representation of a legal Stan program."""
        # fmt: off
        raw_join = "\n".join([
            "data {", self.data, "}\n\n",
            "transformed data {", self.trans_data_decl, "    real delta = 1e-8;", "\n",
            self.trans_data_def, "}\n",
            "parameters {", self.param_decl, "}\n",
            "transformed parameters {", self.trans_decl, "\n", self.trans_def, "}\n",
            "model {", self.model_decl, "\n", self.model_def, "}\n",
        ])
        # fmt: on
        # Remove sequences of 3 or more consecutive newlines
        no_triples = re.sub("\n{3,}", "\n\n", raw_join)
        return no_triples
