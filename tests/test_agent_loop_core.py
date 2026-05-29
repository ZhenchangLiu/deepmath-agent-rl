import unittest

from deepmath_lite.agent_loop_core import (
    ASSISTANT_ROLE,
    OBSERVATION_ROLE,
    AgentLoopCore,
)


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("model called more times than scripted")
        return self.outputs.pop(0)


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class AgentLoopCoreTests(unittest.TestCase):
    def test_executes_code_and_masks_observation_tokens(self):
        model = ScriptedModel(
            [
                "Compute exactly.\n<python>\nprint(2 + 3)\n</python>",
                "The result is 5, so \\boxed{5}.",
            ]
        )
        rollout = AgentLoopCore(model=model).run("What is 2+3?")

        self.assertEqual(rollout.stopped_reason, "boxed_answer")
        self.assertEqual(rollout.final_answer, "5")
        self.assertEqual([span.role for span in rollout.spans], [ASSISTANT_ROLE, OBSERVATION_ROLE, ASSISTANT_ROLE])
        self.assertIn("<observation>\n5\n</observation>", rollout.response_text)
        self.assertIn("<observation>\n5\n</observation>", model.prompts[1])

        response_ids, response_mask = rollout.build_response_ids_and_mask(CharacterTokenizer())
        self.assertEqual(len(response_ids), len(response_mask))
        self.assertEqual(len(response_ids), len(rollout.response_text))

        cursor = 0
        for span in rollout.spans:
            span_len = len(span.text)
            mask_slice = response_mask[cursor : cursor + span_len]
            expected = 1 if span.role == ASSISTANT_ROLE else 0
            self.assertEqual(mask_slice, [expected] * span_len)
            cursor += span_len

    def test_keeps_multiple_tool_turns_in_order(self):
        model = ScriptedModel(
            [
                "<python>\nprint(2 + 3)\n</python>",
                "<python>\nprint(5 * 4)\n</python>",
                "\\boxed{20}",
            ]
        )
        rollout = AgentLoopCore(model=model).run("Use two calculations.")

        self.assertEqual(rollout.final_answer, "20")
        self.assertEqual(
            [span.role for span in rollout.spans],
            [ASSISTANT_ROLE, OBSERVATION_ROLE, ASSISTANT_ROLE, OBSERVATION_ROLE, ASSISTANT_ROLE],
        )
        self.assertIn("<observation>\n5\n</observation>", rollout.response_text)
        self.assertIn("<observation>\n20\n</observation>", rollout.response_text)
        self.assertEqual(rollout.num_turns, 3)

    def test_rejects_code_and_answer_in_same_model_action(self):
        model = ScriptedModel(["<python>\nprint(2 + 3)\n</python>\n\\boxed{5}"])
        rollout = AgentLoopCore(model=model).run("What is 2+3?")

        self.assertEqual(rollout.stopped_reason, "protocol_violation_code_and_answer")
        self.assertIsNone(rollout.final_answer)
        self.assertEqual([span.role for span in rollout.spans], [ASSISTANT_ROLE])


if __name__ == "__main__":
    unittest.main()
