import asyncio
import unittest
from dataclasses import dataclass

from deepmath_lite.executor import ExecutionResult
from deepmath_lite.verl_agent_loop import (
    VeRLPromptTokenizer,
    VeRLServerModelRunner,
    extract_ground_truth,
    resolve_max_response_length,
    score_agent_rollout,
)
from deepmath_lite.verl_agent_loop_core import AgentRollout, RolloutStep


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


class FakeConfigWrap:
    def __init__(self, config):
        self.config = config


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

    def test_scores_formatted_wrong_answer(self):
        rollout = AgentRollout(
            question="1+1?",
            steps=[RolloutStep(prompt="", model_output="\\boxed{3}")],
            final_answer="3",
            stopped_reason="boxed_answer",
        )

        score, info = score_agent_rollout(rollout, "2")

        self.assertEqual(score, 0.2)
        self.assertEqual(info["format_reward"], 0.2)
        self.assertEqual(info["answer_reward"], 0.0)
        self.assertFalse(info["answer_correct"])

    def test_scores_formatted_correct_answer(self):
        rollout = AgentRollout(
            question="1+1?",
            steps=[RolloutStep(prompt="", model_output="\\boxed{2}")],
            final_answer="2",
            stopped_reason="boxed_answer",
        )

        score, info = score_agent_rollout(rollout, "2")

        self.assertEqual(score, 1.0)
        self.assertEqual(info["format_reward"], 0.2)
        self.assertEqual(info["answer_reward"], 0.8)
        self.assertTrue(info["answer_correct"])

    def test_scores_bad_format_as_zero_even_with_answer_text(self):
        rollout = AgentRollout(
            question="1+1?",
            steps=[RolloutStep(prompt="", model_output="```python\nprint(2)\n```\n\\boxed{2}")],
            final_answer=None,
            stopped_reason="protocol_violation_markdown_code_block",
        )

        score, info = score_agent_rollout(rollout, "2")

        self.assertEqual(score, 0.0)
        self.assertEqual(info["format_reward"], 0.0)
        self.assertEqual(info["answer_reward"], 0.0)

    def test_code_error_cancels_format_reward(self):
        step = RolloutStep(
            prompt="",
            model_output="<python>\n1 / 0\n</python>",
            code="1 / 0",
            execution=ExecutionResult(stdout="", error="ZeroDivisionError: division by zero"),
        )
        rollout = AgentRollout(
            question="1+1?",
            steps=[step],
            final_answer=None,
            stopped_reason="max_steps",
        )

        score, info = score_agent_rollout(rollout, "2")

        self.assertEqual(score, 0.0)
        self.assertEqual(info["format_reward"], 0.2)
        self.assertEqual(info["code_error_penalty"], 0.2)
        self.assertTrue(info["has_code_error"])

    def test_extract_ground_truth_from_reward_model(self):
        self.assertEqual(extract_ground_truth({"reward_model": {"ground_truth": "42"}}), "42")

    def test_resolves_max_response_length_from_verl_config_wrap(self):
        trainer_config = FakeConfigWrap({"data": {"max_response_length": 1024}})

        length = resolve_max_response_length({}, trainer_config=trainer_config)

        self.assertEqual(length, 1024)

    def test_agent_loop_config_length_takes_precedence(self):
        trainer_config = FakeConfigWrap({"data": {"max_response_length": 1024}})

        length = resolve_max_response_length(
            {"max_tokens": 2048},
            trainer_config=trainer_config,
            agent_loop_max_response_length="512",
        )

        self.assertEqual(length, 512)


if __name__ == "__main__":
    unittest.main()
