import threading
import unittest

from scripts import run_eval


class RunEvalTests(unittest.TestCase):
    def test_attach_eval_judges_in_caller_thread(self):
        original_verify_answer = run_eval.verify_answer
        calls = []

        def fake_verify_answer(predicted, gold):
            calls.append((threading.current_thread().name, predicted, gold))
            return predicted == gold

        row = {
            "problem": {"id": "demo", "question": "What is 2+3?", "answer": "5"},
            "trace": {"final_answer": "5", "stopped_reason": "boxed_answer", "steps": [{}]},
        }

        try:
            run_eval.verify_answer = fake_verify_answer
            result = run_eval.attach_eval(row)
        finally:
            run_eval.verify_answer = original_verify_answer

        self.assertEqual(calls, [(threading.current_thread().name, "5", "5")])
        self.assertTrue(result["eval"]["correct"])
        self.assertEqual(result["eval"]["predicted"], "5")
        self.assertEqual(result["eval"]["stopped_reason"], "boxed_answer")


if __name__ == "__main__":
    unittest.main()
