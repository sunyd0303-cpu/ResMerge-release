#!/usr/bin/env bash
set -euo pipefail

# Example ResMerge command with placeholder paths.
# All expert models must be fine-tuned from the same base model and must have
# identical parameter names and tensor shapes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL="/path/to/base_model"
EXPERT_1="/path/to/expert_1"
EXPERT_2="/path/to/expert_2"
EXPERT_3="/path/to/expert_3"
OUTPUT_DIR="/path/to/resmerge_output"

python "${REPO_DIR}/resmerge-main.py" \
  --base_model "${BASE_MODEL}" \
  --task_models "${EXPERT_1}" "${EXPERT_2}" "${EXPERT_3}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda \
  --gpu_ids 0,1,2,3 \
  --merge_impl stream \
  --rank_topk 1 \
  --head_ratio_budget 0.20 \
  --head_reliability_rule pos_mean \
  --save_dtype bf16 \
  --save_attn_implementation eager \
  --retry_cuda_oom
