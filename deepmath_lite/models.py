"""Model backends used by the agent loop."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class ModelBackend(Protocol):
    """A minimal text-generation interface."""

    def generate(self, prompt: str) -> str:
        ...


def _dump_openai_obj(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


@dataclass
class MockModel:
    """Deterministic backend for local tests and demos.

    It intentionally solves only tiny arithmetic-looking prompts. The point is
    to verify the system loop without needing a GPU.
    """

    def generate(self, prompt: str) -> str:
        scratchpad = prompt.rsplit("\nScratchpad:\n", 1)[-1] if "\nScratchpad:\n" in prompt else ""
        if "<observation>" in scratchpad:
            observation = scratchpad.rsplit("<observation>", 1)[1].split("</observation>", 1)[0].strip()
            answer = observation.splitlines()[-1].strip()
            return f"The calculation result is {answer}. Therefore \\boxed{{{answer}}}."
        return "I will compute it exactly.\n<python>\nprint(99 + 88)\n</python>"


@dataclass
class OpenAICompatibleBackend:
    """Backend for vLLM's OpenAI-compatible server.

    Example endpoint on the H800 machine:
    http://127.0.0.1:8000/v1
    """

    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.2
    max_tokens: int = 1024
    last_metadata: dict[str, Any] | None = None

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError("Install openai to use OpenAICompatibleBackend") from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0]
        self.last_metadata = {
            "provider": "openai_compatible_completion",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", self.model),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": _dump_openai_obj(getattr(response, "usage", None)),
        }
        return choice.text


@dataclass
class OpenAIChatBackend:
    """Backend for OpenAI-compatible chat APIs such as DeepSeek."""

    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 2048
    thinking_type: str | None = "disabled"
    last_metadata: dict[str, Any] | None = None

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError("Install openai to use OpenAIChatBackend") from exc

        api_key = self.api_key or os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} or pass an API key to use OpenAIChatBackend")

        client = OpenAI(base_url=self.base_url, api_key=api_key)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking_type:
            request["extra_body"] = {"thinking": {"type": self.thinking_type}}

        response = client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)
        self.last_metadata = {
            "provider": "openai_compatible_chat",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", self.model),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": _dump_openai_obj(getattr(response, "usage", None)),
            "content_length": len(content),
            "reasoning_content_length": len(reasoning_content or ""),
            "reasoning_content": reasoning_content,
        }
        return content
