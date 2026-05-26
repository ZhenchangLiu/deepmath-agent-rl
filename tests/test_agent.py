import unittest

from deepmath_lite.agent import ToolAgent


class CodeAndAnswerModel:
    def __init__(self):
        self.calls = 0
        self.last_metadata = None

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_metadata = {"call": self.calls, "finish_reason": "stop"}
        if self.calls == 1:
            return "<python>\nprint(2 + 3)\n</python>"
        return "\\boxed{5}"


class FabricatedObservationModel:
    last_metadata = {"finish_reason": "stop"}

    def generate(self, prompt: str) -> str:
        return "<observation>\n999\n</observation>\n\\boxed{999}"


class CodeAndBoxedModel:
    last_metadata = {"finish_reason": "stop"}

    def generate(self, prompt: str) -> str:
        return "<python>\nprint(2 + 3)\n</python>\n\\boxed{5}"


class StaticModel:
    def __init__(self, output: str, finish_reason: str = "stop"):
        self.output = output
        self.last_metadata = {"finish_reason": finish_reason}

    def generate(self, prompt: str) -> str:
        return self.output


class AgentTests(unittest.TestCase):
    def test_rejects_fabricated_observation(self):
        trace = ToolAgent(model=FabricatedObservationModel()).solve("What is 2+3?")
        self.assertEqual(trace.stopped_reason, "protocol_violation_fabricated_observation")
        self.assertIsNone(trace.final_answer)

    def test_rejects_code_and_answer_in_same_response(self):
        trace = ToolAgent(model=CodeAndBoxedModel()).solve("What is 2+3?")
        self.assertEqual(trace.stopped_reason, "protocol_violation_code_and_answer")
        self.assertIsNone(trace.steps[0].execution)

    def test_rejects_markdown_code_block_before_boxed_answer(self):
        model = StaticModel("```python\nprint(2 + 3)\n```\n\\boxed{5}")
        trace = ToolAgent(model=model).solve("What is 2+3?")
        self.assertEqual(trace.stopped_reason, "protocol_violation_markdown_code_block")
        self.assertIsNone(trace.final_answer)

    def test_marks_truncated_generation(self):
        model = StaticModel("We need to keep solving", finish_reason="length")
        trace = ToolAgent(model=model).solve("What is 2+3?")
        self.assertEqual(trace.stopped_reason, "truncated_generation")

    def test_marks_malformed_code_block(self):
        model = StaticModel("<python>\nprint(2 + 3)")
        trace = ToolAgent(model=model).solve("What is 2+3?")
        self.assertEqual(trace.stopped_reason, "malformed_code_block")

    def test_marks_malformed_boxed_answer(self):
        model = StaticModel("\\boxed{\\frac{1}{2}")
        trace = ToolAgent(model=model).solve("What is 1/2?")
        self.assertEqual(trace.stopped_reason, "malformed_boxed_answer")

    def test_accepts_nested_boxed_answer(self):
        model = StaticModel("\\boxed{\\frac{3\\sqrt{3}}{4}}")
        trace = ToolAgent(model=model).solve("Simplify.")
        self.assertEqual(trace.stopped_reason, "boxed_answer")
        self.assertEqual(trace.final_answer, "\\frac{3\\sqrt{3}}{4}")

    def test_executes_code_then_accepts_next_turn_answer(self):
        trace = ToolAgent(model=CodeAndAnswerModel()).solve("What is 2+3?")
        self.assertEqual(len(trace.steps), 2)
        self.assertEqual(trace.steps[0].execution.stdout.strip(), "5")
        self.assertEqual(trace.final_answer, "5")
        self.assertEqual(trace.steps[0].model_metadata["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
