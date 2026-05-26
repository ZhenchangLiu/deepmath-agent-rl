"""Custom VeRL reward function for DeepMath-style RLVR."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:  # Support VeRL loading this file directly by path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepmath_lite.eval import verify_answer
from deepmath_lite.protocol import extract_boxed_answer


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """Return a sparse rule reward for VeRL.

    VeRL passes the generated response as ``solution_str`` and the parquet
    ``reward_model.ground_truth`` value as ``ground_truth``. Keep the reward
    binary for the first RL smoke run: correct boxed answer gets 1, everything
    else gets 0.
    """

    del data_source, extra_info
    predicted = extract_boxed_answer(solution_str or "")
    return 1.0 if verify_answer(predicted, str(ground_truth)) else 0.0


reward_func = compute_score
