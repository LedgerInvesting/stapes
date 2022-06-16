from pathlib import Path

import tatsu

from .ast import ModelAst
from .ast_actions import ModelAstActions

# Initialize the language parser
GRAMMAR_FILENAME = Path(__file__).parent / "grammar.ebnf"
with open(GRAMMAR_FILENAME, "r") as infile:
    GRAMMAR_TEXT = infile.read()
MARROW_PARSER = tatsu.compile(GRAMMAR_TEXT)


def parse_text(text: str, trace: bool = False) -> ModelAst:
    return MARROW_PARSER.parse(text, semantics=ModelAstActions(), trace=trace)
