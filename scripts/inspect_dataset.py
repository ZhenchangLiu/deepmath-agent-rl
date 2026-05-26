#!/usr/bin/env python
"""Inspect a DeepMath Lite jsonl dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Dataset jsonl path")
    parser.add_argument("--examples", type=int, default=3, help="Number of examples to print")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid json") from exc
            record["_line_no"] = line_no
            records.append(record)
    return records


def short(text: Any, max_len: int = 160) -> str:
    value = str(text).replace("\n", "\\n")
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def print_counter(title: str, counter: Counter[Any]) -> None:
    if not counter:
        return
    print(f"\n{title}:")
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"  {key}: {count}")


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    ids = [str(record.get("id", "")) for record in records]
    id_counts = Counter(ids)
    duplicate_ids = {key: count for key, count in id_counts.items() if key and count > 1}

    missing_id = [record["_line_no"] for record in records if not str(record.get("id", "")).strip()]
    missing_question = [record["_line_no"] for record in records if not str(record.get("question", "")).strip()]
    missing_answer = [record["_line_no"] for record in records if not str(record.get("answer", "")).strip()]

    answer_lengths = [len(str(record.get("answer", ""))) for record in records]
    question_lengths = [len(str(record.get("question", ""))) for record in records]
    subjects = Counter(record.get("metadata", {}).get("subject") for record in records if record.get("metadata"))
    levels = Counter(record.get("metadata", {}).get("level") for record in records if record.get("metadata"))
    sources = Counter(record.get("metadata", {}).get("source") for record in records if record.get("metadata"))

    print(f"file: {args.input}")
    print(f"records: {len(records)}")
    print(f"missing id: {len(missing_id)}")
    print(f"missing question: {len(missing_question)}")
    print(f"missing answer: {len(missing_answer)}")
    print(f"duplicate ids: {len(duplicate_ids)}")
    if records:
        print(
            "question length: "
            f"min={min(question_lengths)} max={max(question_lengths)} "
            f"avg={sum(question_lengths) / len(question_lengths):.1f}"
        )
        print(
            "answer length: "
            f"min={min(answer_lengths)} max={max(answer_lengths)} "
            f"avg={sum(answer_lengths) / len(answer_lengths):.1f}"
        )

    if duplicate_ids:
        print("\nduplicate id samples:")
        for key, count in list(duplicate_ids.items())[:10]:
            print(f"  {key}: {count}")
    if missing_question:
        print(f"\nmissing question lines: {missing_question[:20]}")
    if missing_answer:
        print(f"\nmissing answer lines: {missing_answer[:20]}")

    print_counter("subjects", subjects)
    print_counter("levels", levels)
    print_counter("sources", sources)

    print(f"\nfirst {min(args.examples, len(records))} examples:")
    for record in records[: args.examples]:
        print(f"- line {record['_line_no']} id={record.get('id')}")
        print(f"  question: {short(record.get('question'))}")
        print(f"  answer: {short(record.get('answer'))}")
        metadata = record.get("metadata")
        if metadata:
            print(f"  metadata: {metadata}")


if __name__ == "__main__":
    main()
