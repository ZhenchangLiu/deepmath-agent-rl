import asyncio
import unittest
from dataclasses import dataclass

from deepmath_lite.verl_agent_loop import VeRLPromptTokenizer, VeRLServerModelRunner


class CharacterHFTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


@dataclass
class FakeTokenOutput:
    token_ids: list[int]
    stop_reason: str = "completed"
    num_preempted: int = 0
    extra_fields: dict | None = None


class FakeServerManager:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeTokenOutput(token_ids=[ord("\\"), *map(ord, "boxed{5}")], extra_fields={"global_steps": 1})


class VerlAgentLoopAdapterTests(unittest.TestCase):
    def test_server_model_runner_calls_verl_generate_and_decodes_tokens(self):
        async def run_case():
            tokenizer = VeRLPromptTokenizer(CharacterHFTokenizer())
            server_manager = FakeServerManager()
            runner = VeRLServerModelRunner(
                server_manager=server_manager,
                tokenizer=tokenizer,
                sampling_params={"max_tokens": 16, "temperature": 0.7},
            )
            text = await runner.generate("Question?")
            return text, runner, server_manager

        text, runner, server_manager = asyncio.run(run_case())
        self.assertEqual(text, "\\boxed{5}")
        self.assertEqual(len(server_manager.calls), 1)
        call = server_manager.calls[0]
        self.assertEqual(call["prompt_ids"], [ord(char) for char in "Question?"])
        self.assertEqual(call["sampling_params"], {"max_tokens": 16, "temperature": 0.7})
        self.assertIn("request_id", call)
        self.assertEqual(runner.last_metadata["provider"], "verl_server_manager")
        self.assertEqual(runner.last_metadata["finish_reason"], "completed")
        self.assertEqual(runner.last_metadata["token_count"], 9)


if __name__ == "__main__":
    unittest.main()
