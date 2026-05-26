#!/usr/bin/env python
"""Run one DeepMath Lite agent example."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepmath_lite.agent import ToolAgent
from deepmath_lite.models import MockModel, OpenAIChatBackend, OpenAICompatibleBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--backend", choices=["mock", "openai", "chat"], default="mock")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking", choices=["disabled", "enabled", "omit"], default="disabled")
    parser.add_argument("--max-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "mock":
        backend = MockModel()
    elif args.backend == "chat":
        backend = OpenAIChatBackend(
            base_url=args.base_url,
            model=args.model or "deepseek-v4-flash",
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking_type=None if args.thinking == "omit" else args.thinking,
        )
    else:
        backend = OpenAICompatibleBackend(
            base_url=args.base_url,
            model=args.model or "Qwen/Qwen3-4B",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    agent = ToolAgent(model=backend, max_steps=args.max_steps)
    trace = agent.solve(args.question)
    print(json.dumps(asdict(trace), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
