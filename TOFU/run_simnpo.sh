#!/bin/bash
# SimNPO unlearning on 2 GPUs (global batch 32 via accum=16 in forget.yaml).
# Prerequisites:
#   1. HF login + models downloaded to data disk (see README)
#   2. forget.yaml model_path points to the local origin model directory
#   3. Run from the TOFU/ directory

set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export master_port="${master_port:-29500}"
# Put caches on the data disk (system disk is often only ~30GB)
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"

NPROC="${NPROC:-2}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node=${NPROC}"
echo "HF_HOME=${HF_HOME}"
echo "master_port=${master_port}"

echo "=== Running SimNPO forget05 ==="
torchrun --nproc_per_node="${NPROC}" --master_port="${master_port}" \
  forget.py --config-name=forget.yaml \
  split=forget05 npo_coeff=0.1375 beta=2.5

echo "=== Running SimNPO forget10 ==="
torchrun --nproc_per_node="${NPROC}" --master_port=$((master_port + 1)) \
  forget.py --config-name=forget.yaml \
  split=forget10 npo_coeff=0.125 beta=4.5

echo "Done. Check \${save_dir}/checkpoint/aggregate_stat.txt under the model unlearned/ folders."
