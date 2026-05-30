#!/usr/bin/env bash
# Tiny end-to-end VeRL GRPO smoke for DeepMath Lite AgentLoop on H800.

set -xeuo pipefail

ROOT_DIR=${ROOT_DIR:-/share/liuzhenchang/deepmath-agent-rl}
MODEL_PATH=${MODEL_PATH:-/mmu_nlp_hdd/dujiazhen03/model/Qwen2.5-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/tmp/deepmath_verl_smoke_data/train.parquet}
VAL_FILE=${VAL_FILE:-/tmp/deepmath_verl_smoke_data/val.parquet}
DATA_DIR=${DATA_DIR:-/tmp/deepmath_verl_smoke_data}
AGENT_LOOP_CONFIG=${AGENT_LOOP_CONFIG:-${ROOT_DIR}/configs/verl/deepmath_lite_agent_loop.yaml}
REWARD_PATH=${REWARD_PATH:-${ROOT_DIR}/deepmath_lite/verl_reward.py}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1536}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-dummy}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
SAVE_FREQ=${SAVE_FREQ:-1}
TEST_FREQ=${TEST_FREQ:--1}

PROJECT_NAME=${PROJECT_NAME:-deepmath_lite}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-agent_grpo_smoke_qwen25_1p5b}
TRAINER_MODULE=${TRAINER_MODULE:-verl.trainer.main_ppo}
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}"

python scripts/prepare_deepmath_verl.py \
    --limit 32 \
    --val-size 4 \
    --output-dir "${DATA_DIR}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.agent.default_agent_loop=deepmath_lite
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.custom_reward_function.path="${REWARD_PATH}"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
    reward.reward_model.enable=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.val_before_train=False
)

RAY_ENV=(
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'${VLLM_USE_V1}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL="'${VLLM_LOGGING_LEVEL}'"
)

python -m "${TRAINER_MODULE}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${RAY_ENV[@]}" \
    "$@"
