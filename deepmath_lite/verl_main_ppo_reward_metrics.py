"""VeRL PPO entrypoint with DeepMath reward-component logging.

This module intentionally leaves VeRL training behavior untouched. It only
wraps the metric collector so reward components emitted by the AgentLoop show
up as scalar training metrics in console/wandb.
"""

from __future__ import annotations

from typing import Any

import numpy as np


REWARD_COMPONENT_KEYS = (
    "format_reward",
    "answer_reward",
    "code_error_penalty",
)


def _numeric_values(values: Any) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return array


def install_reward_component_metric_patch() -> None:
    """Patch VeRL's trainer metric collector in-process."""

    from verl.trainer.ppo import ray_trainer  # noqa: PLC0415

    original_compute_data_metrics = ray_trainer.compute_data_metrics
    if getattr(original_compute_data_metrics, "_deepmath_reward_metrics", False):
        return

    def compute_data_metrics_with_reward_components(batch: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        metrics = original_compute_data_metrics(batch, *args, **kwargs)
        for key in REWARD_COMPONENT_KEYS:
            if key not in batch.non_tensor_batch:
                continue
            values = _numeric_values(batch.non_tensor_batch[key])
            if values is None:
                continue
            metrics[f"critic/{key}/mean"] = float(np.mean(values))
            metrics[f"critic/{key}/max"] = float(np.max(values))
            metrics[f"critic/{key}/min"] = float(np.min(values))
        return metrics

    compute_data_metrics_with_reward_components._deepmath_reward_metrics = True
    ray_trainer.compute_data_metrics = compute_data_metrics_with_reward_components


install_reward_component_metric_patch()

from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    main()
