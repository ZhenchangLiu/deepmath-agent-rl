"""VeRL AgentLoop adapter skeleton for DeepMath Lite.

The real VeRL package is imported lazily because local development should not
depend on a fully working VeRL/Ray/vLLM stack. The core rollout logic lives in
``verl_agent_loop_core.py`` and is intentionally framework-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .eval import verify_answer
from .protocol import build_prompt
from .verl_agent_loop_core import AsyncAgentLoopCore, AgentRollout, AsyncTextModelRunner, TextTokenizer


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
    reward_score: float | None = None
    metrics: dict[str, float | int] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)


class VeRLPromptTokenizer(TextTokenizer):
    """Small adapter around Hugging Face tokenizers used by VeRL."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def build_output_payload(
    question: str,
    rollout: AgentRollout,
    tokenizer: TextTokenizer,
    reward_score: float | None = None,
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
        reward_score=reward_score,
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
        reward_score=payload.reward_score,
        num_turns=payload.num_turns,
        metrics=metrics,
        extra_fields=payload.extra_fields,
    )


def score_agent_rollout(rollout: AgentRollout, ground_truth: str) -> tuple[float, dict[str, Any]]:
    """Return shaped reward components for one agent rollout."""

    format_ok = rollout.stopped_reason in {"boxed_answer", "max_steps"}
    has_code_error = any(step.execution is not None and not step.execution.ok for step in rollout.steps)
    answer_correct = format_ok and verify_answer(rollout.final_answer, ground_truth)

    format_reward = 0.2 if format_ok else 0.0
    answer_reward = 0.8 if answer_correct else 0.0
    code_error_penalty = 0.2 if has_code_error else 0.0
    score = max(0.0, format_reward + answer_reward - code_error_penalty)

    return score, {
        "score": score,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        "code_error_penalty": code_error_penalty,
        "format_ok": format_ok,
        "answer_correct": answer_correct,
        "has_code_error": has_code_error,
        "stopped_reason": rollout.stopped_reason,
        "final_answer": rollout.final_answer,
    }


class VeRLServerModelRunner(AsyncTextModelRunner):
    """Async model runner backed by VeRL's LLM server manager."""

    def __init__(self, server_manager: Any, tokenizer: VeRLPromptTokenizer, sampling_params: dict[str, Any]):
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.last_metadata: dict[str, Any] | None = None

    async def generate(self, prompt: str) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=dict(self.sampling_params),
        )
        token_ids = list(output.token_ids)
        self.last_metadata = {
            "provider": "verl_server_manager",
            "finish_reason": getattr(output, "stop_reason", None),
            "token_count": len(token_ids),
            "num_preempted": getattr(output, "num_preempted", None),
            "extra_fields": getattr(output, "extra_fields", {}),
        }
        return self.tokenizer.decode(token_ids)


class DeepMathLiteAgentLoop:
    """Hydra-loadable VeRL AgentLoop implementation for DeepMath Lite."""

    def __init__(
        self,
        trainer_config: Any,
        server_manager: Any,
        tokenizer: Any,
        processor: Any = None,
        dataset_cls: Any = None,
        data_config: Any = None,
        max_steps: int = 5,
        timeout_s: float = 2.0,
        **_: Any,
    ):
        self.trainer_config = trainer_config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config
        self.max_steps = max_steps
        self.timeout_s = timeout_s

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        question = extract_question(kwargs)
        ground_truth = extract_ground_truth(kwargs)
        tokenizer = VeRLPromptTokenizer(self.tokenizer)
        model = VeRLServerModelRunner(self.server_manager, tokenizer, sampling_params)
        rollout = await AsyncAgentLoopCore(
            model=model,
            max_steps=self.max_steps,
            timeout_s=self.timeout_s,
        ).run(question)
        reward_score, reward_extra_info = score_agent_rollout(rollout, ground_truth)
        payload = build_output_payload(
            question,
            rollout,
            tokenizer,
            reward_score=reward_score,
            extra_fields={"reward_extra_info": reward_extra_info},
        )
        return to_verl_output(payload)


def build_deepmath_agent_loop_class() -> type:
    """Build a VeRL AgentLoop subclass when VeRL is available."""

    AgentLoopBase, _, _, register = import_verl_agent_loop_types()

    @register("deepmath_lite")
    class DeepMathLiteAgentLoop(AgentLoopBase):  # type: ignore[misc, valid-type]
        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            question = extract_question(kwargs)
            ground_truth = extract_ground_truth(kwargs)
            tokenizer = VeRLPromptTokenizer(self.tokenizer)
            model = VeRLServerModelRunner(self.server_manager, tokenizer, sampling_params)
            rollout = await AsyncAgentLoopCore(model=model).run(question)
            reward_score, reward_extra_info = score_agent_rollout(rollout, ground_truth)
            payload = build_output_payload(
                question,
                rollout,
                tokenizer,
                reward_score=reward_score,
                extra_fields={"reward_extra_info": reward_extra_info},
            )
            return to_verl_output(payload)

    return DeepMathLiteAgentLoop


def extract_question(dataset_fields: dict[str, Any]) -> str:
    """Extract a plain question from common VeRL dataset field shapes."""

    raw_prompt = dataset_fields.get("raw_prompt")
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, list) and raw_prompt:
        last = raw_prompt[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])

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


def extract_ground_truth(dataset_fields: dict[str, Any]) -> str:
    """Extract ground truth answer from common VeRL reward field shapes."""

    reward_model = dataset_fields.get("reward_model")
    if hasattr(reward_model, "item"):
        reward_model = reward_model.item()
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])

    for key in ("ground_truth", "answer", "target", "final_answer"):
        value = dataset_fields.get(key)
        if value is not None:
            if hasattr(value, "item"):
                value = value.item()
            return str(value)

    raise ValueError("expected reward_model.ground_truth or answer field in dataset fields")
