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

The VeRL server-manager adapter is now wired through:

```text
deepmath_lite.verl_agent_loop.DeepMathLiteAgentLoop
configs/verl/deepmath_lite_agent_loop.yaml
```

The first real training smoke is:

```bash
bash scripts/h800_train_agent_grpo_smoke.sh
```

The smoke script preflights the vLLM V1 NumPy/Numba compatibility before
starting Ray. If it reports `Numba needs NumPy 2.2 or less`, fix the dedicated
H800 environment:

```bash
python -m pip install "numpy<2.3"
```

By default this uses:

```text
model: /mmu_nlp_hdd/dujiazhen03/model/Qwen2.5-1.5B-Instruct
trainer: verl.trainer.main_ppo
rollout: vLLM async AgentLoop
data: 32 DeepMath-103K samples, 4 validation samples
steps: 1 optimizer step
gpus: 1
vllm: V1 engine, eager mode, small context/concurrency
agent loop workers: 1
```

The script appends `VLLM_USE_V1` and `VLLM_LOGGING_LEVEL` through
`+ray_kwargs.ray_init.runtime_env.env_vars`, because VeRL creates vLLM inside
Ray actors and plain shell exports may not be visible in the server actor. The
values are quoted so Hydra keeps them as strings for Ray's `runtime_env`.

To switch the same path to 7B later:

```bash
MODEL_PATH=/mmu_nlp_hdd/dujiazhen03/model/Qwen2.5-7B-Instruct \
NGPUS_PER_NODE=8 \
ROLLOUT_TP=2 \
ROLLOUT_GPU_MEM_UTIL=0.6 \
bash scripts/h800_train_agent_grpo_smoke.sh
```

If the vLLM server fails before entering `DeepMathLiteAgentLoop.run`, inspect
the Ray worker logs for the real engine error:

```bash
bash scripts/h800_collect_ray_logs.sh
```

Useful toggles for isolating vLLM startup issues:

```bash
VLLM_USE_V1=0 bash scripts/h800_train_agent_grpo_smoke.sh
VLLM_LOGGING_LEVEL=DEBUG bash scripts/h800_train_agent_grpo_smoke.sh
ROLLOUT_LOAD_FORMAT=auto bash scripts/h800_train_agent_grpo_smoke.sh
ROLLOUT_GPU_MEM_UTIL=0.5 bash scripts/h800_train_agent_grpo_smoke.sh
AGENT_LOOP_WORKERS=4 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 bash scripts/h800_train_agent_grpo_smoke.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/h800_train_agent_grpo_smoke.sh
```

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
