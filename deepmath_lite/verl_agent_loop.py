"""VeRL AgentLoop adapter skeleton for DeepMath Lite.

The real VeRL package is imported lazily because local development should not
depend on a fully working VeRL/Ray/vLLM stack. The core rollout logic lives in
``verl_agent_loop_core.py`` and is intentionally framework-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocol import build_prompt
from .verl_agent_loop_core import AgentLoopCore, AgentRollout, TextModelRunner, TextTokenizer


def import_verl_agent_loop_types() -> tuple[Any, Any, Any, Any]:
    """Import VeRL AgentLoop symbols only in an H800/VeRL runtime."""

    from verl.experimental.agent_loop.agent_loop import (  # noqa: PLC0415
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )

    return AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register


@dataclass(frozen=True)
class AgentLoopOutputPayload:
    """Framework-neutral payload matching VeRL's AgentLoopOutput fields."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    num_turns: int
    metrics: dict[str, float | int] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)


class VeRLPromptTokenizer(TextTokenizer):
    """Small adapter around Hugging Face tokenizers used by VeRL."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


def build_output_payload(
    question: str,
    rollout: AgentRollout,
    tokenizer: TextTokenizer,
    extra_fields: dict[str, Any] | None = None,
) -> AgentLoopOutputPayload:
    """Convert the local rollout into the fields VeRL expects."""

    prompt_ids = tokenizer.encode(build_prompt(question, history=None))
    response_ids, response_mask = rollout.build_response_ids_and_mask(tokenizer)
    if len(response_ids) != len(response_mask):
        raise ValueError("response_ids and response_mask must have the same length")

    return AgentLoopOutputPayload(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        num_turns=rollout.num_turns,
        metrics={
            "tool_calls": sum(1 for step in rollout.steps if step.code is not None),
            "generate_sequences": rollout.num_turns,
            "compute_score": 0.0,
        },
        extra_fields={
            "stopped_reason": rollout.stopped_reason,
            "final_answer": rollout.final_answer,
            **(extra_fields or {}),
        },
    )


def to_verl_output(payload: AgentLoopOutputPayload) -> Any:
    """Instantiate VeRL's AgentLoopOutput in a real VeRL runtime."""

    _, AgentLoopMetrics, AgentLoopOutput, _ = import_verl_agent_loop_types()
    metrics = AgentLoopMetrics(**payload.metrics)
    return AgentLoopOutput(
        prompt_ids=payload.prompt_ids,
        response_ids=payload.response_ids,
        response_mask=payload.response_mask,
        num_turns=payload.num_turns,
        metrics=metrics,
        extra_fields=payload.extra_fields,
    )


class VeRLServerModelRunner(TextModelRunner):
    """Placeholder for the VeRL LLM server manager integration.

    This is intentionally not implemented locally. The H800 integration should
    call VeRL's async LLM server manager here and return one assistant span per
    agent step.
    """

    def __init__(self, server_manager: Any, sampling_params: dict[str, Any]):
        self.server_manager = server_manager
        self.sampling_params = sampling_params

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("VeRL server_manager generation is wired in the H800 adapter step")


def build_deepmath_agent_loop_class() -> type:
    """Build a VeRL AgentLoop subclass when VeRL is available."""

    AgentLoopBase, _, _, register = import_verl_agent_loop_types()

    @register("deepmath_lite")
    class DeepMathLiteAgentLoop(AgentLoopBase):  # type: ignore[misc, valid-type]
        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            question = extract_question(kwargs)
            tokenizer = VeRLPromptTokenizer(self.tokenizer)
            model = VeRLServerModelRunner(self.server_manager, sampling_params)
            rollout = AgentLoopCore(model=model).run(question)
            payload = build_output_payload(question, rollout, tokenizer)
            return to_verl_output(payload)

    return DeepMathLiteAgentLoop


def extract_question(dataset_fields: dict[str, Any]) -> str:
    """Extract a plain question from common VeRL dataset field shapes."""

    prompt = dataset_fields.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])

    question = dataset_fields.get("question") or dataset_fields.get("problem")
    if question is not None:
        return str(question)

    raise ValueError("expected prompt, question, or problem in dataset fields")
