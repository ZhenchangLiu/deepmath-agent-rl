"""VeRL PPO entrypoint with DeepMath reward-component logging.

This module intentionally leaves VeRL training behavior untouched. It only
wraps the metric collector so reward components emitted by the AgentLoop show
up as scalar training metrics in console/wandb.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


REWARD_COMPONENT_KEYS = (
    "format_reward",
    "answer_reward",
    "code_error_penalty",
)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _iter_items(value: Any) -> Any:
    value = _maybe_json(value)
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, np.ndarray):
        for item in value.reshape(-1):
            yield from _iter_items(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_items(item)
        return
    if hasattr(value, "item"):
        try:
            yield from _iter_items(value.item())
            return
        except (TypeError, ValueError):
            pass
    yield value


def _collect_component_values(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    for item in _iter_items(value):
        if not isinstance(item, dict):
            continue
        if key in item:
            values.append(item[key])
        for nested_key in ("reward_extra_info", "extra_fields"):
            if nested_key in item:
                values.extend(_collect_component_values(item[nested_key], key))
    return values


def _numeric_values(values: Any) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return array


def extract_reward_component_values(batch: Any, key: str) -> np.ndarray | None:
    """Extract a reward component from common VeRL DataProto shapes."""

    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if not non_tensor_batch:
        return None

    collected: list[Any] = []
    if key in non_tensor_batch:
        collected.extend(_collect_component_values(non_tensor_batch[key], key))
        if not collected:
            values = _numeric_values(non_tensor_batch[key])
            if values is not None:
                return values

    for field_name in ("reward_extra_info", "extra_fields"):
        if field_name in non_tensor_batch:
            collected.extend(_collect_component_values(non_tensor_batch[field_name], key))

    if not collected:
        return None
    return _numeric_values(collected)


def install_reward_component_metric_patch() -> None:
    """Patch VeRL's trainer metric collector in-process."""

    from verl.trainer.ppo import ray_trainer  # noqa: PLC0415

    original_compute_data_metrics = ray_trainer.compute_data_metrics
    if getattr(original_compute_data_metrics, "_deepmath_reward_metrics", False):
        return

    def compute_data_metrics_with_reward_components(batch: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        metrics = original_compute_data_metrics(batch, *args, **kwargs)
        for key in REWARD_COMPONENT_KEYS:
            values = extract_reward_component_values(batch, key)
            if values is None:
                continue
            metrics[f"critic/{key}/mean"] = float(np.mean(values))
            metrics[f"critic/{key}/max"] = float(np.max(values))
            metrics[f"critic/{key}/min"] = float(np.min(values))
        return metrics

    compute_data_metrics_with_reward_components._deepmath_reward_metrics = True
    ray_trainer.compute_data_metrics = compute_data_metrics_with_reward_components


def main() -> Any:
    install_reward_component_metric_patch()

    from verl.trainer.main_ppo import main as verl_main  # noqa: PLC0415

    return verl_main()


if __name__ == "__main__":
    main()
