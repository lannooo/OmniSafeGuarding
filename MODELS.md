# Finetuning Your own Omni-modal LLM-baesd Guardrail

This document shows the details of model downloading and guardrail finetuning workflows.

### Base Model Preparation

Download base models used for inference and finetuning:

```bash
bash prepare_base_models_hf.sh

# optional: use a custom model root
bash prepare_base_models_hf.sh /path/to/models
```

## Train/Fine-tune an Omni Guardrail

### Step 0. Environment Setup (skip if already done)

Create and install the recommended environment with one script:

```bash
bash setup_env.sh
```

By default, the script creates conda env `omniguard` with Python 3.10 and installs the training/inference/data dependencies used in this project (PyTorch stack, vLLM, transformers, Qwen utils, datasets, and common tooling).

Optional environment variables:

```bash
# custom env name / python version
ENV_NAME=omniguard PYTHON_VERSION=3.10 bash setup_env.sh

# optional pip mirror
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh

# flash-attn behavior: auto (default) | 1 (force install) | 0 (skip)
INSTALL_FLASH_ATTN=auto bash setup_env.sh

# activate env
conda activate omniguard
```

Notes:

- `flash-attn` is installed automatically only when CUDA/NVIDIA conditions are detected (or when force-enabled). On macOS or non-CUDA machines it is skipped with a warning.
- After setup, activate with `conda activate omniguard` (or your custom `ENV_NAME`).

Besure that swift library is installed for LLM fine-tuning. Refer to [SWIFT](https://github.com/modelscope/ms-swift).

```Shell
pip install ms-swift -U

# Using uv
pip install uv
uv pip install ms-swift -U --torch-backend=auto
```

### Step 1. Fetch multimodal training media

Training media resources are provided via an anonymous Zenodo release [Zenodo-Link](https://zenodo.org/records/19979658):

Run the following script to download it and extract files:

```bash
bash finetune_fetch_media.sh ./ms-swift
```

After completion, media files are placed under ``ms-swift/media``.

### Step 2. Prepare finetuning JSON data

Rewrite dataset media prefixes to your local media directory:

```bash
bash finetune_prepare.sh ./ms-swift/media
```

This regenerates finetuning data files under ``ms-swift/data``.

### Step 3. Launch swift finetuning

Run the LoRA finetuning script for Qwen2.5 Base model:

```bash
bash ms-swift/finetune/qwen25/sft_lora_enhance.sh
```

Before running, check the path variables in the script:

- MODEL_NAME: base model path
- INPUT: finetuning JSON file
- OUTPUT: checkpoint output directory

## Post-Training Merge

After training, merge LoRA adapter weights into a standalone Guard model:

```bash
bash ms-swift/finetune/sft_lora_merge.sh <checkpoint_path> <base_model_path>
```

Example:

```bash
bash ms-swift/finetune/sft_lora_merge.sh ms-swift/ckpt/your_adapter_ckpt models/base/Qwen2.5-Omni-7B

# then, moving them to the guardrail directory, e.g.,
mv ms-swift/ckpt/your_adapter_ckpt_merged models/guards/omguard-qwen25-custom
```

Important: Naming the merged guardrail model name with family-key identifiers (qwen25 / qwen3 / phi / minicpm), so inference logic can correctly load the the correct model wrapper.

## Notes

- Multimodal finetuning may hit OOM. Start with smaller per-device batch size (often 1 for multimodal), then use gradient accumulation for effective batch size.
- If OOM persists, use more GPUs via CUDA_VISIBLE_DEVICES and/or switch to a DeepSpeed setup, for example ms-swift/finetune/qwen3/zero3.sh.
- Script comments in ms-swift/finetune/qwen25/sft_lora_enhance.sh already include practical memory guidance; follow those first.
