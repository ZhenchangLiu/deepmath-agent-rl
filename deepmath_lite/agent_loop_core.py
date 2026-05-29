"""Local core for VeRL-style agent rollouts.

This module deliberately avoids importing VeRL. It models the rollout contract
we need before adapting it to VeRL's AgentLoopOutput on the H800 environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .executor import ExecutionResult, run_python
from .protocol import build_prompt, extract_boxed_answer, find_first_code_block, make_observation


ASSISTANT_ROLE = "assistant"
OBSERVATION_ROLE = "observation"


class TextModelRunner(Protocol):
    """Minimal text generation interface used by the local loop core."""

    def generate(self, prompt: str) -> str:
        ...


class TextTokenizer(Protocol):
    """Tokenizer interface needed to build response ids and masks."""

    def encode(self, text: str) -> list[int]:
        ...


@dataclass(frozen=True)
class RolloutSpan:
    """One contiguous response span in an agent rollout."""

    role: str
    text: str

    @property
    def is_model_action(self) -> bool:
        return self.role == ASSISTANT_ROLE


@dataclass
class RolloutStep:
    prompt: str
    model_output: str
    code: str | None = None
    execution: ExecutionResult | None = None
    observation: str | None = None


@dataclass
class AgentRollout:
    question: str
    steps: list[RolloutStep] = field(default_factory=list)
    spans: list[RolloutSpan] = field(default_factory=list)
    final_answer: str | None = None
    stopped_reason: str = ""

    @property
    def response_text(self) -> str:
        return "".join(span.text for span in self.spans)

    @property
    def num_turns(self) -> int:
        return sum(1 for span in self.spans if span.role == ASSISTANT_ROLE)

    def build_response_ids_and_mask(self, tokenizer: TextTokenizer) -> tuple[list[int], list[int]]:
        response_ids: list[int] = []
        response_mask: list[int] = []
        for span in self.spans:
            token_ids = tokenizer.encode(span.text)
            response_ids.extend(token_ids)
            response_mask.extend([1 if span.is_model_action else 0] * len(token_ids))
        return response_ids, response_mask


@dataclass
class AgentLoopCore:
    """Run a code-execution agent loop and preserve action/observation masks."""

    model: TextModelRunner
    max_steps: int = 5
    timeout_s: float = 2.0

    def run(self, question: str) -> AgentRollout:
        history: list[str] = []
        rollout = AgentRollout(question=question)

        for _ in range(self.max_steps):
            prompt = build_prompt(question, history)
            model_output = self.model.generate(prompt)
            rollout.spans.append(RolloutSpan(role=ASSISTANT_ROLE, text=model_output))

            block = find_first_code_block(model_output)
            answer = extract_boxed_answer(model_output)
            step = RolloutStep(prompt=prompt, model_output=model_output)
            rollout.steps.append(step)

            if block is not None and answer is not None:
                rollout.final_answer = None
                rollout.stopped_reason = "protocol_violation_code_and_answer"
                return rollout

            if block is not None:
                result = run_python(block.code, timeout_s=self.timeout_s)
                observation = make_observation(result.stdout, result.error)
                step.code = block.code
                step.execution = result
                step.observation = observation

                rollout.spans.append(RolloutSpan(role=OBSERVATION_ROLE, text=observation))
                history.append(model_output)
                history.append(observation)
                continue

            if answer is not None:
                rollout.final_answer = answer
                rollout.stopped_reason = "boxed_answer"
                return rollout

            rollout.final_answer = None
            rollout.stopped_reason = "no_code_or_answer"
            return rollout

        rollout.final_answer = extract_boxed_answer(rollout.steps[-1].model_output) if rollout.steps else None
        rollout.stopped_reason = "max_steps"
        return rollout
