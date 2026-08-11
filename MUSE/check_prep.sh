#!/bin/bash
# Pre-flight checks before MUSE training / eval.
set -euo pipefail

cd "$(dirname "$0")"

ok=0
fail=0
check() {
  local name="$1"
  shift
  if "$@"; then
    echo "[OK]  ${name}"
    ok=$((ok + 1))
  else
    echo "[FAIL] ${name}"
    fail=$((fail + 1))
  fi
}

if [[ -f /root/autodl-tmp/env_hf.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/env_hf.sh
fi

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

check "data symlink" bash -c '[[ -L data && -d data ]]'
check "ckpt symlink" bash -c '[[ -L baselines/ckpt && -d baselines/ckpt ]]'
check "news forget.txt" test -f data/news/raw/forget.txt
check "news retain1.txt" test -f data/news/raw/retain1.txt
check "books forget.txt" test -f data/books/raw/forget.txt
check "books retain1.txt" test -f data/books/raw/retain1.txt
check "News target in HF cache" bash -c 'ls -d "$HUGGINGFACE_HUB_CACHE"/models--muse-bench--MUSE-News_target >/dev/null 2>&1'
check "Books target in HF cache" bash -c 'ls -d "$HUGGINGFACE_HUB_CACHE"/models--muse-bench--MUSE-Books_target >/dev/null 2>&1'
check "tokenizer in HF cache" bash -c 'ls -d "$HUGGINGFACE_HUB_CACHE"/models--NousResearch--Llama-2-7b-hf >/dev/null 2>&1'
check "hub < 1.0" python -c "import huggingface_hub as h; v=h.__version__; assert v.startswith('0.'), v"
check "transformers import" python -c "import transformers; assert transformers.__version__.startswith('4.')"
check "CUDA available" python -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()>=1"
check "data disk free >= 40G" bash -c 'avail=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc 0-9); [[ "${avail:-0}" -ge 40 ]]'

echo "---"
echo "passed=${ok} failed=${fail}"
if [[ "${fail}" -gt 0 ]]; then
  exit 1
fi
echo "Ready to run: bash run_muse.sh news|books"
