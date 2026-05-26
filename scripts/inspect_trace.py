#!/usr/bin/env python
"""Inspect trace jsonl files emitted by scripts/run_eval.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Trace jsonl path")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--incorrect-only", action="store_true")
    parser.add_argument("--show-prompts", action="store_true")
    parser.add_argument("--max-chars", type=int, default=1200)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid json") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def trim(text: Any, max_chars: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def is_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("eval", {}).get("correct"))


def print_summary(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    correct = sum(is_correct(row) for row in rows)
    incorrect = total - correct
    accuracy = correct / total if total else 0.0
    stopped = {}
    for row in rows:
        reason = row.get("eval", {}).get("stopped_reason") or row.get("trace", {}).get("stopped_reason")
        stopped[reason] = stopped.get(reason, 0) + 1

    print(f"records: {total}")
    print(f"correct: {correct}")
    print(f"incorrect: {incorrect}")
    print(f"accuracy: {accuracy:.4f}")
    print("stopped reasons:")
    for reason, count in sorted(stopped.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"  {reason}: {count}")


def print_row(row: dict[str, Any], max_chars: int, show_prompts: bool) -> None:
    problem = row.get("problem", {})
    trace = row.get("trace", {})
    evaluation = row.get("eval", {})

    print("\n" + "=" * 80)
    print(f"line: {row.get('_line_no')}")
    print(f"id: {problem.get('id') or evaluation.get('problem_id')}")
    print(f"correct: {evaluation.get('correct')}")
    print(f"predicted: {evaluation.get('predicted')}")
    print(f"gold: {evaluation.get('gold') or problem.get('answer')}")
    print(f"stopped_reason: {evaluation.get('stopped_reason') or trace.get('stopped_reason')}")
    if trace.get("error"):
        print(f"error: {trim(trace.get('error'), max_chars)}")
    print(f"question: {trim(problem.get('question') or trace.get('question'), max_chars)}")

    for index, step in enumerate(trace.get("steps", []), start=1):
        print(f"\n-- step {index} --")
        if show_prompts:
            print("prompt:")
            print(trim(step.get("prompt"), max_chars))
        print("model_output:")
        print(trim(step.get("model_output"), max_chars))
        if step.get("code") is not None:
            print("code:")
            print(trim(step.get("code"), max_chars))
        execution = step.get("execution")
        if execution is not None:
            print("execution:")
            print(
                json.dumps(
                    {
                        "stdout": trim(execution.get("stdout"), max_chars),
                        "error": trim(execution.get("error"), max_chars),
                        "timed_out": execution.get("timed_out"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if step.get("observation") is not None:
            print("observation:")
            print(trim(step.get("observation"), max_chars))


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    print_summary(rows)

    selected = [row for row in rows if not is_correct(row)] if args.incorrect_only else rows
    print(f"\nshowing: {min(args.limit, len(selected))} of {len(selected)} selected")
    for row in selected[: args.limit]:
        print_row(row, max_chars=args.max_chars, show_prompts=args.show_prompts)


if __name__ == "__main__":
    main()
