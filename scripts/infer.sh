#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the Wan2.1-T2V-14B directory}"
: "${MODEL_WEIGHTS:?Set MODEL_WEIGHTS to a trained transformer checkpoint}"
: "${COND_VIDEO:?Set COND_VIDEO to a condition video}"
: "${REF_IMAGE:?Set REF_IMAGE to a reference image}"
: "${DEPTH_VIDEO:?Set DEPTH_VIDEO to a merged depth video}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/inference}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python "${ROOT_DIR}/infer.py" \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --load_wan_path "${MODEL_WEIGHTS}" \
  --video_list "${COND_VIDEO}" \
  --image_list "${REF_IMAGE}" \
  --depth_video_list "${DEPTH_VIDEO}" \
  --output_dir "${OUTPUT_DIR}" \
  --frame_num 81 \
  --sampling_steps 40 \
  --target_short_size 480 \
  --dtype bf16 \
  "$@"
