#!/usr/bin/env python
"""Evaluate the agent on a jsonl file."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepmath_lite.agent import ToolAgent
from deepmath_lite.data import MathProblem
from deepmath_lite.data import iter_limited, load_jsonl
from deepmath_lite.eval import EvalResult, verify_answer
from deepmath_lite.models import MockModel, OpenAIChatBackend, OpenAICompatibleBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="jsonl file with id/question/answer")
    parser.add_argument(
        "--trace-output",
        default=None,
        help="Trace jsonl path. Defaults to outputs_inference/{input_stem}__{backend}-{model}__inference.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", choices=["mock", "openai", "chat"], default="mock")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking", choices=["disabled", "enabled", "omit"], default="disabled")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def safe_filename_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return cleaned.strip("-") or "unknown"


def default_model_name(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.backend == "chat":
        return "deepseek-v4-flash"
    if args.backend == "openai":
        return "Qwen-Qwen3-4B"
    return "mock"


def default_trace_output(args: argparse.Namespace) -> Path:
    input_stem = safe_filename_part(Path(args.input).stem)
    inference_name = safe_filename_part(f"{args.backend}-{default_model_name(args)}")
    filename = f"{input_stem}__{inference_name}__inference.jsonl"
    return Path("outputs_inference") / filename


def build_backend(args: argparse.Namespace) -> Any:
    if args.backend == "mock":
        return MockModel()
    if args.backend == "chat":
        return OpenAIChatBackend(
            base_url=args.base_url,
            model=args.model or "deepseek-v4-flash",
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking_type=None if args.thinking == "omit" else args.thinking,
        )
    return OpenAICompatibleBackend(
        base_url=args.base_url,
        model=args.model or "Qwen/Qwen3-4B",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def evaluate_one(problem: MathProblem, args: argparse.Namespace) -> dict[str, Any]:
    try:
        agent = ToolAgent(model=build_backend(args))
        trace = agent.solve(problem.question)
        return {"problem": asdict(problem), "trace": asdict(trace)}
    except Exception as exc:  # noqa: BLE001 - keep long eval jobs alive and inspectable.
        error = f"{type(exc).__name__}: {exc}"
        trace = {
            "question": problem.question,
            "steps": [],
            "final_text": "",
            "final_answer": None,
            "stopped_reason": "worker_error",
            "error": error,
            "traceback": traceback.format_exc(limit=8),
        }
        return {"problem": asdict(problem), "trace": trace}


def attach_eval(row: dict[str, Any]) -> dict[str, Any]:
    """Attach evaluation in the caller thread.

    math-verify uses signal-based timeouts by default and is not safe to call
    from worker threads. Keep model calls concurrent, then judge answers here.
    """

    problem = row["problem"]
    trace = row["trace"]
    predicted = trace.get("final_answer")
    gold = problem["answer"]
    result = EvalResult(
        problem_id=problem["id"],
        predicted=predicted,
        gold=gold,
        correct=verify_answer(predicted, gold),
        stopped_reason=trace.get("stopped_reason", ""),
        steps=len(trace.get("steps") or []),
    )
    row["eval"] = asdict(result)
    return row


def print_progress(done: int, total: int, correct_count: int) -> None:
    accuracy = correct_count / done if done else 0.0
    print(f"completed {done}/{total} correct={correct_count} accuracy={accuracy:.4f}", flush=True)


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    problems = list(iter_limited(load_jsonl(args.input), args.limit))
    results: list[EvalResult] = []
    trace_path = Path(args.trace_output) if args.trace_output else default_trace_output(args)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with trace_path.open("w", encoding="utf-8") as trace_file:
        if args.concurrency == 1:
            for problem in problems:
                row = attach_eval(evaluate_one(problem, args))
                result = EvalResult(**row["eval"])
                results.append(result)
                trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                if args.progress_every > 0 and len(results) % args.progress_every == 0:
                    print_progress(len(results), len(problems), sum(item.correct for item in results))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(evaluate_one, problem, args) for problem in problems]
                for future in as_completed(futures):
                    row = attach_eval(future.result())
                    result = EvalResult(**row["eval"])
                    results.append(result)
                    trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    trace_file.flush()
                    if args.progress_every > 0 and len(results) % args.progress_every == 0:
                        print_progress(len(results), len(problems), sum(item.correct for item in results))

    total = len(results)
    correct_count = sum(item.correct for item in results)
    accuracy = correct_count / total if total else 0.0
    print(json.dumps({"total": total, "correct": correct_count, "accuracy": accuracy}, indent=2))
    print(f"wrote traces to {trace_path}")


if __name__ == "__main__":
    main()
