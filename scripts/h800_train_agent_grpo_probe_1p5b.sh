#!/usr/bin/env bash
# A slightly larger 1.5B H800 probe run with wandb-ready logging.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/share/liuzhenchang/deepmath-agent-rl}
export ROOT_DIR

export MODEL_PATH=${MODEL_PATH:-/mmu_nlp_hdd/dujiazhen03/model/Qwen2.5-1.5B-Instruct}
export DATA_DIR=${DATA_DIR:-/tmp/deepmath_verl_probe_1p5b_data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}
export DATA_LIMIT=${DATA_LIMIT:-256}
export VAL_SIZE=${VAL_SIZE:-16}

export PROJECT_NAME=${PROJECT_NAME:-deepmath_lite}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-agent_grpo_probe5_qwen25_1p5b}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:--1}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_N=${ROLLOUT_N:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-3072}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-2}

exec "${ROOT_DIR}/scripts/h800_train_agent_grpo_smoke.sh" "$@"
