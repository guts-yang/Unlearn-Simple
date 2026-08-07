# TOFU

## Installation

```bash
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

If your cloud image already provides a recent PyTorch + CUDA (e.g. PyTorch 2.8 / CUDA 12.8), skip the conda CUDA pin and install only the Python deps.

`pip install flash-attn --no-build-isolation` compiles from source and needs a full CUDA toolkit; images that ship only PyTorch's bundled CUDA runtime have no `nvcc` and the build fails immediately. Install the official prebuilt wheel instead, matching `cu12` / `torch<major.minor>` / `cxx11abi<TRUE|FALSE>` / `cp<pyver>` to your interpreter:

```bash
python -c "import torch; print(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI)"
# pick the matching asset from https://github.com/Dao-AILab/flash-attention/releases
pip install --no-deps <flash_attn-...whl>
```

If you cannot install it at all, set `llama2-7b.flash_attention2` to `"false"` in `config/model_config.yaml` to fall back to SDPA. Both are exact attention, so metrics are unaffected.

## HuggingFace setup (required before running)

Unlearning loads two resources:

| Role | HuggingFace repo | Notes |
|------|------------------|-------|
| Origin / ft weights (`model_path`) | [`locuslab/tofu_ft_llama2-7b`](https://huggingface.co/locuslab/tofu_ft_llama2-7b) | Public. Must be a **local directory** (see below). |
| Tokenizer / base config (`hf_key`) | [`NousResearch/Llama-2-7b-chat-hf`](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) | May require accepting the Llama-2 license. |
| Dataset | [`locuslab/TOFU`](https://huggingface.co/datasets/locuslab/TOFU) | Auto-downloaded for forget01/05/10. |

`forget.py` uses `os.listdir(model_path)`, so **do not** set `model_path` to a Hub repo id.

```bash
pip install -U "huggingface_hub[cli]"
# Everything goes to the data disk — the system disk is only ~30GB.
# HUGGINGFACE_HUB_CACHE must be $HF_HOME/hub, or the existing cache is bypassed.
export HF_ENDPOINT=https://hf-mirror.com     # required where huggingface.co is unreachable
export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
hf auth login   # paste a Read token (the repo root .env has one)

hf download locuslab/tofu_ft_llama2-7b
hf download NousResearch/Llama-2-7b-chat-hf
```

On this machine `source /root/autodl-tmp/env_hf.sh` sets all of the above (and reads the token from `.env`).

No `--local-dir` needed: the cache snapshot directory is itself a valid local directory. Point `model_path` in `config/forget.yaml` at it (already the default in this repo):

```bash
ls -d $HF_HOME/hub/models--locuslab--tofu_ft_llama2-7b/snapshots/*/
```

The snapshot hash changes when re-downloading, so re-read it after a fresh download.

Alternative: download the author-provided origin model from [Google Drive](https://drive.google.com/drive/folders/1L47Hf813gal8RD581S3XrWHnY_0ll4y4?usp=sharing) into a local folder and set `model_path` accordingly.

## 2-GPU settings (global batch aligned with paper)

Paper used 8 GPUs with `batch_size=1` and `gradient_accumulation_steps=4` → global batch `1×4×8=32`.

This repo defaults to **2 GPUs** with `gradient_accumulation_steps=16` → global batch `1×16×2=32` (same optimizer steps / dynamics). `num_epochs`, `lr`, and `batch_size` are unchanged. Eval `batch_size` is set to `8` to reduce peak VRAM on 2×80GB.

## Get and evaluate the unlearned model

* Confirm `model_path` in `config/forget.yaml` points to the local origin model directory.
* `save_dir` defaults to `/root/autodl-tmp/TOFU_results/2GPU_<hparams>`. Keep it on the data disk, and do **not** put it under `model_path` — that writes checkpoints into the HuggingFace cache.

One-shot (recommended):

```bash
cd TOFU
source /root/autodl-tmp/env_hf.sh
bash run_simnpo.sh
```

Or manually:

```bash
export master_port=29500
source /root/autodl-tmp/env_hf.sh

# forget05
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port \
  forget.py --config-name=forget.yaml split=forget05 npo_coeff=0.1375 beta=2.5

# forget10
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port \
  forget.py --config-name=forget.yaml split=forget10 npo_coeff=0.125 beta=4.5
```

* Results are written to `/root/autodl-tmp/TOFU_results/2GPU_<hparams>/checkpoint*/aggregate_stat.txt`.
