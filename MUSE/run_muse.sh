#!/bin/bash
# SimNPO+GDR unlearning on MUSE (News / Books), 2 GPUs via device_map=auto.
#
# Usage:
#   bash run_muse.sh              # default: news
#   bash run_muse.sh news
#   bash run_muse.sh books
#
# Prerequisites:
#   1. source /root/autodl-tmp/env_hf.sh (or rely on exports below)
#   2. MUSE/data and MUSE/baselines/ckpt symlinks point at data disk
#   3. Run from the MUSE/ directory (or via this script's cd)

set -euo pipefail

cd "$(dirname "$0")"

CORPUS="${1:-news}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -f /root/autodl-tmp/env_hf.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/env_hf.sh
fi

TOKENIZER_DIR="${TOKENIZER_DIR:-NousResearch/Llama-2-7b-hf}"
MAX_LEN="${MAX_LEN:-2048}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-4}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "${LOG_DIR}"

case "${CORPUS}" in
  news)
    MODEL_DIR="muse-bench/MUSE-News_target"
    DATA_FILE="../data/news/raw/forget.txt"
    RETAIN_FILE="../data/news/raw/retain1.txt"
    OUT_DIR="./ckpt/news/simnpo_gdr"
    BETA=0.7
    ;;
  books)
    MODEL_DIR="muse-bench/MUSE-Books_target"
    DATA_FILE="../data/books/raw/forget.txt"
    RETAIN_FILE="../data/books/raw/retain1.txt"
    OUT_DIR="./ckpt/books/simnpo_gdr"
    BETA=0.75
    ;;
  *)
    echo "Unsupported corpus: ${CORPUS} (use news or books)"
    exit 1
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/muse_${CORPUS}_simnpo_gdr_${STAMP}.log"

echo "corpus=${CORPUS} algo=simnpo_gdr beta=${BETA} epochs=${EPOCHS} batch=${BATCH} max_len=${MAX_LEN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME}"
echo "out_dir=${OUT_DIR}"
echo "log=${LOG_FILE}"

cd baselines
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python unlearn.py \
  --algo simnpo_gdr \
  --model_dir "${MODEL_DIR}" \
  --tokenizer_dir "${TOKENIZER_DIR}" \
  --data_file "${DATA_FILE}" \
  --retain_data_file "${RETAIN_FILE}" \
  --out_dir "${OUT_DIR}" \
  --max_len "${MAX_LEN}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --per_device_batch_size "${BATCH}" \
  --beta "${BETA}" \
  --coeff 0.1 \
  --npo_coeff 1.0 \
  2>&1 | tee "${LOG_FILE}"

echo "Done ${CORPUS}. Checkpoints under MUSE/baselines/ckpt/${CORPUS}/simnpo_gdr (→ /root/autodl-tmp/MUSE_ckpt)."
echo "Eval example:"
echo "  cd /usr/local/Unlearn-Simple/MUSE && python eval.py --model_dirs baselines/ckpt/${CORPUS}/simnpo_gdr --names simnpo_gdr --corpus ${CORPUS} --tokenizer_dir ${TOKENIZER_DIR} --out_file out_${CORPUS}.csv"
