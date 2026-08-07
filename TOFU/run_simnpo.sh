#!/bin/bash
# SimNPO unlearning on 2 GPUs (global batch 32 via accum=16 in forget.yaml).
# Runs ONE split at a time so you can inspect results before the next one.
#
# Usage:
#   bash run_simnpo.sh              # default: forget05
#   bash run_simnpo.sh forget05
#   bash run_simnpo.sh forget10
#
# Prerequisites:
#   1. HF login + models downloaded to data disk (see docs/复现指南.md)
#   2. forget.yaml model_path points to the local origin model directory
#   3. Run from the TOFU/ directory

set -euo pipefail

cd "$(dirname "$0")"

SPLIT="${1:-forget05}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export master_port="${master_port:-29500}"
# Put caches on the data disk (system disk is often only ~30GB)
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"

NPROC="${NPROC:-2}"

case "$SPLIT" in
  forget05) NPO_COEFF=0.1375; BETA=2.5 ;;
  forget10) NPO_COEFF=0.125;  BETA=4.5 ;;
  *) echo "Unsupported split: $SPLIT (use forget05 or forget10)"; exit 1 ;;
esac

echo "split=${SPLIT} npo_coeff=${NPO_COEFF} beta=${BETA}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc_per_node=${NPROC}"
echo "HF_HOME=${HF_HOME} master_port=${master_port}"

echo "=== Running SimNPO ${SPLIT} ==="
torchrun --nproc_per_node="${NPROC}" --master_port="${master_port}" \
  forget.py --config-name=forget.yaml \
  split="${SPLIT}" npo_coeff="${NPO_COEFF}" beta="${BETA}"

echo "Done ${SPLIT}. Check \${save_dir}/checkpoint/aggregate_stat.txt under the model unlearned/ folder."
