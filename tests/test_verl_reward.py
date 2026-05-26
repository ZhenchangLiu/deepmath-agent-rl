import unittest

from deepmath_lite.verl_reward import compute_score


class VerlRewardTests(unittest.TestCase):
    def test_rewards_correct_boxed_answer(self):
        self.assertEqual(
            compute_score(
                data_source="zwhe99/DeepMath-103K",
                solution_str="The final answer is \\boxed{5}.",
                ground_truth="5",
            ),
            1.0,
        )

    def test_rejects_missing_boxed_answer(self):
        self.assertEqual(
            compute_score(
                data_source="zwhe99/DeepMath-103K",
                solution_str="The final answer is 5.",
                ground_truth="5",
            ),
            0.0,
        )

    def test_rejects_wrong_boxed_answer(self):
        self.assertEqual(
            compute_score(
                data_source="zwhe99/DeepMath-103K",
                solution_str="The final answer is \\boxed{6}.",
                ground_truth="5",
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
