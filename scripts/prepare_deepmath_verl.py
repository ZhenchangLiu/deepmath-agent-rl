#!/usr/bin/env python
"""Prepare DeepMath-103K data in VeRL parquet format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_DATASET = "zwhe99/DeepMath-103K"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = "data_verl/deepmath_103k"
DEFAULT_INSTRUCTION = "Solve the problem. Put your final answer in \\boxed{...}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--input",
        default=None,
        help="Optional local json/jsonl/parquet file. If omitted, load from Hugging Face datasets.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-name", default="train.parquet")
    parser.add_argument("--val-name", default="val.parquet")
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep dataset order before train/val split. By default records are shuffled.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid json") from exc
    return records


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "train", "rows", "examples"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"{path} must be a json list or a dict with a data-like list")


def read_local_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(source)
    if suffix == ".json":
        return read_json(source)
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas/pyarrow to read local parquet files") from exc
        return pd.read_parquet(source).to_dict(orient="records")
    raise ValueError(f"unsupported input suffix: {source.suffix}")


def read_hf_records(dataset: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets or pass --input with a local file") from exc

    ds = load_dataset(dataset, split=split)
    return [dict(item) for item in ds]


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def build_question(question: str, instruction: str) -> str:
    question = question.strip()
    instruction = instruction.strip()
    if not instruction:
        return question
    return f"{question}\n\n{instruction}"


def normalize_record(record: dict[str, Any], index: int, instruction: str, data_source: str) -> dict[str, Any]:
    question = first_present(record, ("question", "problem", "prompt"))
    answer = first_present(record, ("final_answer", "answer", "target", "ground_truth"))
    if question is None:
        raise ValueError(f"record {index} missing question/problem/prompt")
    if answer is None:
        raise ValueError(f"record {index} missing final_answer/answer/target/ground_truth")

    difficulty = first_present(record, ("difficulty", "level"))
    topic = first_present(record, ("topic", "subject", "type"))
    source_id = first_present(record, ("id", "unique_id", "uid"))

    extra_info: dict[str, Any] = {
        "index": index,
        "source_id": str(source_id) if source_id is not None else None,
    }
    if difficulty is not None:
        extra_info["difficulty"] = difficulty
    if topic is not None:
        extra_info["topic"] = topic

    for key in ("r1_solution_1", "r1_solution_2", "r1_solution_3", "r1_solution_4"):
        if key in record and record[key] is not None:
            extra_info[key] = record[key]

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": build_question(str(question), instruction),
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(answer).strip(),
        },
        "extra_info": extra_info,
    }


def split_records(
    records: list[dict[str, Any]],
    val_size: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(records)
    if shuffle:
        random.Random(seed).shuffle(rows)
    val_count = min(max(val_size, 0), len(rows))
    return rows[val_count:], rows[:val_count]


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas/pyarrow to write parquet files") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


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
        normalize_record(record, index=index, instruction=args.instruction, data_source=args.dataset)
        for index, record in enumerate(raw_records)
    ]
    train_records, val_records = split_records(
        records,
        val_size=args.val_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    output_dir = Path(args.output_dir)
    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    write_parquet(train_records, train_path)
    write_parquet(val_records, val_path)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "input": args.input,
        "output_dir": str(output_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "instruction": args.instruction,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
