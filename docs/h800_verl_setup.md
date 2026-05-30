# H800 VeRL Setup And Smoke Plan

This document is the handoff checklist for moving DeepMath Lite from local
development to an H800 machine. The goal is not to start full training
immediately. The first H800 milestone is a clean VeRL environment plus smoke
checks for data, reward, AgentLoop imports, and local tests.

## Principles

- Do not reuse the local Mac `llm` environment.
- Use a fresh H800 conda environment or a clean container.
- Keep MATH-500 as evaluation-only data.
- Use DeepMath-103K / OpenMathReasoning-TIR style data for RL training.
- Start with smoke checks before any multi-GPU GRPO run.

## Recommended Environment

Create a dedicated environment:

```bash
conda create -n verl-deepmath python=3.12 -y
conda activate verl-deepmath
```

Install PyTorch, vLLM, and VeRL using versions compatible with the H800 CUDA
driver. Prefer the official VeRL installation guide for exact CUDA-specific
commands.

Suggested source install pattern:

```bash
git clone https://github.com/verl-project/verl.git ~/verl
cd ~/verl
pip install --no-deps -e .
```

Then install missing runtime dependencies according to the VeRL documentation
and `pip check`. On H800, this should be resolved in the dedicated environment,
not in the local Mac `llm` environment.

## Project Checkout

Use git if possible:

```bash
git clone <YOUR_REPO_URL> ~/math
cd ~/math
```

If no remote exists yet, use `rsync` or `scp` once, then initialize git remote
later.

## Data

The full DeepMath-103K VeRL parquet can be generated on H800:

```bash
python scripts/prepare_deepmath_verl.py \
  --output-dir data_verl/deepmath_103k
```

For smoke checks, use a tiny subset:

```bash
python scripts/prepare_deepmath_verl.py \
  --limit 32 \
  --val-size 4 \
  --output-dir /tmp/deepmath_verl_smoke_data
```

Expected output:

```text
train_count > 0
val_count >= 0
skipped_count is reported
```

## Smoke Checks

Run the bundled smoke script:

```bash
bash scripts/h800_smoke.sh
```

This script checks:

- Python package imports.
- VeRL AgentLoop API import.
- Project unit tests.
- Tiny DeepMath-103K parquet generation.
- Local AgentLoop payload construction without launching vLLM.

The smoke script intentionally does not:

- Start Ray.
- Start vLLM.
- Run GRPO.
- Allocate multiple GPUs.

## Manual Checks

If the smoke script fails, run these manually:

```bash
python -c "from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics; print('verl agent loop ok')"
python -c "from deepmath_lite.verl_agent_loop import build_deepmath_agent_loop_class; print(build_deepmath_agent_loop_class())"
python -m unittest discover -s tests
```

Check available GPUs:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Next Implementation Step On H800

The current placeholder is:

```text
deepmath_lite.verl_agent_loop.VeRLServerModelRunner.generate
```

It must be replaced by a real VeRL rollout call against:

```python
await self.server_manager.generate(
    request_id=...,
    prompt_ids=...,
    sampling_params=...,
)
```

This should be implemented on H800 because it needs the real VeRL server manager
and tokenizer behavior.

## First Training Smoke Target

After the AgentLoop server-manager call works, run a tiny GRPO smoke:

```text
samples: 8-32
num_generations: 2
max_steps: 1-5
response_length: small, e.g. 512-1024
goal: one optimizer step and one checkpoint
```

The success condition is not accuracy. The success condition is:

```text
rollout runs
sandbox observations get response_mask = 0
assistant tokens get response_mask = 1
reward is attached
checkpoint is saved
```

## Known Local Caveat

The local Mac `llm` environment was briefly modified while probing VeRL. Do not
use that environment as the reference for H800 dependency versions. The H800
environment should be built cleanly.
