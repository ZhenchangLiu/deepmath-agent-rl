"""Tool-augmented reasoning loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .executor import ExecutionResult, run_python
from .models import ModelBackend
from .protocol import (
    build_prompt,
    extract_boxed_answer,
    find_first_code_block,
    has_malformed_boxed_answer,
    has_malformed_code_block,
    has_markdown_code_block,
    make_observation,
)


@dataclass
class AgentStep:
    prompt: str
    model_output: str
    model_metadata: dict[str, Any] | None = None
    code: str | None = None
    execution: ExecutionResult | None = None
    observation: str | None = None


@dataclass
class AgentTrace:
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_text: str = ""
    final_answer: str | None = None
    stopped_reason: str = ""


@dataclass
class ToolAgent:
    model: ModelBackend
    max_steps: int = 5
    timeout_s: float = 2.0

    def solve(self, question: str) -> AgentTrace:
        history: list[str] = [] # For LLM
        trace = AgentTrace(question=question) # For DEV

        for _ in range(self.max_steps):
            prompt = build_prompt(question, history)
            model_output = self.model.generate(prompt)
            model_metadata = getattr(self.model, "last_metadata", None)
            step = AgentStep(prompt=prompt, model_output=model_output, model_metadata=model_metadata)
            trace.steps.append(step)

            block = find_first_code_block(model_output)
            answer = extract_boxed_answer(model_output)
            if "<observation>" in model_output:
                trace.final_text = model_output
                trace.final_answer = None
                trace.stopped_reason = "protocol_violation_fabricated_observation"
                return trace

            if block is not None and answer is not None:
                trace.final_text = model_output
                trace.final_answer = None
                trace.stopped_reason = "protocol_violation_code_and_answer"
                return trace

            if has_markdown_code_block(model_output):
                trace.final_text = model_output
                trace.final_answer = None
                trace.stopped_reason = "protocol_violation_markdown_code_block"
                return trace

            if block is not None:
                result = run_python(block.code, timeout_s=self.timeout_s)
                observation = make_observation(result.stdout, result.error)
                step.code = block.code
                step.execution = result
                step.observation = observation

                history.append(model_output)
                history.append(observation)
                continue

            if answer is not None:
                trace.final_text = model_output
                trace.final_answer = answer
                trace.stopped_reason = "boxed_answer"
                return trace

            if isinstance(model_metadata, dict) and model_metadata.get("finish_reason") == "length":
                trace.final_text = model_output
                trace.stopped_reason = "truncated_generation"
                return trace

            if has_malformed_code_block(model_output):
                trace.final_text = model_output
                trace.stopped_reason = "malformed_code_block"
                return trace

            if has_malformed_boxed_answer(model_output):
                trace.final_text = model_output
                trace.stopped_reason = "malformed_boxed_answer"
                return trace

            trace.final_text = model_output
            trace.stopped_reason = "no_code_or_answer"
            return trace

        trace.final_text = trace.steps[-1].model_output if trace.steps else ""
        trace.final_answer = extract_boxed_answer(trace.final_text)
        trace.stopped_reason = "max_steps"
        return trace
