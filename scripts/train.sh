#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the Wan2.1-T2V-14B directory}"
: "${INIT_WEIGHTS:?Set INIT_WEIGHTS to the initial transformer weights}"
: "${DATA_ROOT:?Set DATA_ROOT to the training dataset directory}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
  --config_file "${ROOT_DIR}/configs/accelerate.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  "${ROOT_DIR}/train.py" \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --load_wan_path "${INIT_WEIGHTS}" \
  --train_data_roots "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler polynomial \
  --lr_warmup_steps 20 \
  --max_train_steps 10000 \
  --checkpointing_steps 500 \
  --validation_steps 500 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --dataloader_num_workers 8 \
  "$@"
