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

If your cloud image already provides a recent PyTorch + CUDA (e.g. PyTorch 2.8 / CUDA 12.8), you can skip the conda CUDA pin and only install Python deps + `flash-attn`. If `flash-attn` fails to build, set `llama2-7b.flash_attention2` to `"false"` in `config/model_config.yaml`.

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
# Prefer the data disk — system disks are often ~30GB
export HF_HOME=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
huggingface-cli login   # paste a Read token

huggingface-cli download locuslab/tofu_ft_llama2-7b \
  --local-dir /data/models/tofu_ft_llama2-7b
huggingface-cli download NousResearch/Llama-2-7b-chat-hf \
  --local-dir /data/models/Llama-2-7b-chat-hf
```

Then set `model_path` in `config/forget.yaml` to `/data/models/tofu_ft_llama2-7b` (already the default in this repo). Optionally point `hf_key` in `config/model_config.yaml` to `/data/models/Llama-2-7b-chat-hf`.

Alternative: download the author-provided origin model from [Google Drive](https://drive.google.com/drive/folders/1L47Hf813gal8RD581S3XrWHnY_0ll4y4?usp=sharing) into a local folder and set `model_path` accordingly.

## 2-GPU settings (global batch aligned with paper)

Paper used 8 GPUs with `batch_size=1` and `gradient_accumulation_steps=4` → global batch `1×4×8=32`.

This repo defaults to **2 GPUs** with `gradient_accumulation_steps=16` → global batch `1×16×2=32` (same optimizer steps / dynamics). `num_epochs`, `lr`, and `batch_size` are unchanged. Eval `batch_size` is set to `8` to reduce peak VRAM on 2×80GB.

## Get and evaluate the unlearned model

* Confirm `model_path` in `config/forget.yaml` points to the local origin model directory.
* Optionally edit `save_dir` for where checkpoints are written.

One-shot (recommended):

```bash
cd TOFU
export HF_HOME=/data/hf_cache
bash run_simnpo.sh
```

Or manually:

```bash
export master_port=29500
export HF_HOME=/data/hf_cache

# forget05
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port \
  forget.py --config-name=forget.yaml split=forget05 npo_coeff=0.1375 beta=2.5

# forget10
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port \
  forget.py --config-name=forget.yaml split=forget10 npo_coeff=0.125 beta=4.5
```

* Results are written to `${save_dir}/checkpoint/aggregate_stat.txt`.
