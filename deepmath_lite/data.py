"""Dataset loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MathProblem:
    id: str
    question: str
    answer: str


def load_jsonl(path: str | Path) -> list[MathProblem]:
    problems: list[MathProblem] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                problem = MathProblem(
                    id=str(raw.get("id", line_no)),
                    question=str(raw["question"]),
                    answer=str(raw["answer"]),
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing key: {exc}") from exc
            problems.append(problem)
    return problems


def iter_limited(items: Iterable[MathProblem], limit: int | None) -> Iterable[MathProblem]:
    for index, item in enumerate(items):
        if limit is not None and index >= limit:
            break
        yield item

