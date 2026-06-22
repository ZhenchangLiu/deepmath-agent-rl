import unittest
from types import SimpleNamespace

import numpy as np

from deepmath_lite.verl_main_ppo_reward_metrics import extract_reward_component_values


class RewardMetricExtractionTests(unittest.TestCase):
    def test_extracts_flat_component_array(self):
        batch = SimpleNamespace(non_tensor_batch={"format_reward": np.array([0.2, 0.0, 0.2])})

        values = extract_reward_component_values(batch, "format_reward")

        self.assertTrue(np.array_equal(values, np.array([0.2, 0.0, 0.2], dtype=np.float32)))

    def test_extracts_nested_reward_extra_info(self):
        batch = SimpleNamespace(
            non_tensor_batch={
                "reward_extra_info": np.array(
                    [
                        {"answer_reward": 0.8},
                        {"answer_reward": 0.0},
                    ],
                    dtype=object,
                )
            }
        )

        values = extract_reward_component_values(batch, "answer_reward")

        self.assertTrue(np.array_equal(values, np.array([0.8, 0.0], dtype=np.float32)))

    def test_extracts_extra_fields_reward_extra_info(self):
        batch = SimpleNamespace(
            non_tensor_batch={
                "extra_fields": np.array(
                    [
                        {"reward_extra_info": {"code_error_penalty": 0.2}},
                        {"reward_extra_info": {"code_error_penalty": 0.0}},
                    ],
                    dtype=object,
                )
            }
        )

        values = extract_reward_component_values(batch, "code_error_penalty")

        self.assertTrue(np.array_equal(values, np.array([0.2, 0.0], dtype=np.float32)))

    def test_extracts_json_encoded_extra_fields(self):
        batch = SimpleNamespace(
            non_tensor_batch={
                "extra_fields": np.array(
                    [
                        '{"reward_extra_info": {"format_reward": 0.2}}',
                        '{"reward_extra_info": {"format_reward": 0.0}}',
                    ],
                    dtype=object,
                )
            }
        )

        values = extract_reward_component_values(batch, "format_reward")

        self.assertTrue(np.array_equal(values, np.array([0.2, 0.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
