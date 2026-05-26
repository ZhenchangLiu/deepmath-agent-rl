"""Answer extraction and verification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .protocol import extract_boxed_answer


WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.removeprefix("$").removesuffix("$")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = WHITESPACE_RE.sub("", text)
    return text


# def normalize_for_sympy(text: str) -> str:
#     text = text.strip()
#     text = text.removeprefix("$").removesuffix("$")
#     text = text.replace("\\(", "").replace("\\)", "")
#     text = text.replace("\\left", "").replace("\\right", "")
#     text = text.replace("\\cdot", "*").replace("\\times", "*")
#     text = text.replace("{", "").replace("}", "")
#     return text


# def verify_sympy_expression(predicted: str, gold: str) -> bool:
#     try:
#         import sympy as sp
#         from sympy.parsing.sympy_parser import (
#             convert_xor,
#             implicit_multiplication_application,
#             parse_expr,
#             standard_transformations,
#         )
#     except ImportError:
#         return False

#     transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
#     # transformations = standard_transformations
#     try:
#         pred_expr = parse_expr(normalize_for_sympy(predicted), transformations=transformations)
#         gold_expr = parse_expr(normalize_for_sympy(gold), transformations=transformations)
#         return bool(sp.simplify(pred_expr - gold_expr) == 0)
#     except Exception:
#         return False


def verify_answer(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False

    if normalize_answer(predicted) == normalize_answer(gold):
        return True

    try:
        from math_verify import parse, verify
        return bool(verify(parse(predicted), parse(gold)))
    except Exception:
        return False

    # try:
    #     from math_verify import parse, verify
    # except ImportError:
    #     return False

    # try:
    #     if bool(verify(parse(predicted), parse(gold))):
    #         return True
    # except Exception:
    #     pass

    # return verify_sympy_expression(predicted, gold)


@dataclass(frozen=True)
class EvalResult:
    problem_id: str
    predicted: str | None
    gold: str
    correct: bool
    stopped_reason: str
    steps: int


def extract_answer_from_text(text: str) -> str | None:
    return extract_boxed_answer(text)
