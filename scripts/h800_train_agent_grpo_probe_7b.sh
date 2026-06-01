#!/usr/bin/env bash
# 7B H800 probe run for the shaped DeepMath AgentLoop reward.

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/share/liuzhenchang/deepmath-agent-rl}
export ROOT_DIR

export MODEL_PATH=${MODEL_PATH:-/mmu_nlp_hdd/dujiazhen03/model/Qwen2.5-7B-Instruct}
export DATA_DIR=${DATA_DIR:-/tmp/deepmath_verl_probe_7b_data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}
export DATA_LIMIT=${DATA_LIMIT:-512}
export VAL_SIZE=${VAL_SIZE:-32}

export PROJECT_NAME=${PROJECT_NAME:-deepmath_lite}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-agent_grpo_shaped_probe_qwen25_7b}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}

export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:--1}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_N=${ROLLOUT_N:-4}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-3072}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-4}

exec "${ROOT_DIR}/scripts/h800_train_agent_grpo_smoke.sh" "$@"
