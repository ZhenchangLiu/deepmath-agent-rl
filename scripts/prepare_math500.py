#!/usr/bin/env python
"""Convert MATH-500-style data to DeepMath Lite jsonl format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument(
        "--input",
        default=None,
        help="Local json/jsonl file. If omitted, load from Hugging Face datasets.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="Keep the full worked solution in metadata. Off by default to keep files compact.",
    )
    return parser.parse_args()


def read_local_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        records = []
        with source.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{source}:{line_no} invalid json") from exc
        return records

    with source.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "rows", "examples", "test"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"{source} must be a json list, jsonl file, or dict with a data-like list")


def read_hf_records(dataset: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets or pass --input with a local json/jsonl file") from exc

    ds = load_dataset(dataset, split=split)
    return [dict(item) for item in ds]


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def normalize_record(record: dict[str, Any], index: int, include_solution: bool) -> dict[str, Any]:
    question = first_present(record, ("question", "problem", "prompt"))
    answer = first_present(record, ("answer", "final_answer", "target"))
    if question is None:
        raise ValueError(f"record {index} missing question/problem/prompt")
    if answer is None:
        raise ValueError(f"record {index} missing answer/final_answer/target")

    raw_id = first_present(record, ("id", "unique_id", "uid"))
    problem_id = str(raw_id) if raw_id is not None else f"math500-{index:04d}"

    metadata_keys = ("subject", "level", "type", "source", "unique_id")
    metadata = {key: record[key] for key in metadata_keys if key in record and record[key] is not None}
    metadata.setdefault("source", DEFAULT_DATASET)
    if include_solution and record.get("solution") is not None:
        metadata["solution"] = record["solution"]

    return {
        "id": problem_id,
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "metadata": metadata,
    }


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    raw_records = (
        read_local_records(args.input)
        if args.input
        else read_hf_records(dataset=args.dataset, split=args.split)
    )
    if args.limit is not None:
        raw_records = raw_records[: args.limit]

    records = [
        normalize_record(record, index=index, include_solution=args.include_solution)
        for index, record in enumerate(raw_records, start=1)
    ]
    count = write_jsonl(records, args.output)
    print(json.dumps({"output": args.output, "count": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
