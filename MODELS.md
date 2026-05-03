# Models, Training Data, and Finetuning

This document focuses on model download and guardrail finetuning workflows.

## Base Model Preparation

Download base models used for inference and finetuning:

```bash
bash prepare_base_models_hf.sh
```

Use a custom model root if needed:

```bash
bash prepare_base_models_hf.sh /path/to/models
```

## Train Your Own Guardrail

Run finetuning in 3 steps.

### Step 1. Fetch multimodal training media

Training media resources are provided via an anonymous Zenodo release:

- https://zenodo.org/records/19979658

Download and extract directly under ms-swift:

```bash
bash finetune_fetch_media.sh ./ms-swift
```

After completion, media files are placed under ms-swift/media.

### Step 2. Prepare finetuning JSON data

Rewrite dataset media prefixes to your local media directory:

```bash
bash finetune_prepare.sh ./ms-swift/media
```

This regenerates finetuning data files under ms-swift/data.

### Step 3. Launch swift finetuning

Run the Qwen2.5 LoRA finetuning script:

```bash
bash ms-swift/finetune/qwen25/sft_lora_enhance.sh
```

Before running, check the path variables in the script:

- MODEL_NAME: base model path
- INPUT: finetuning JSON file
- OUTPUT: checkpoint output directory

## Finetuning Notes

- Multimodal finetuning may hit OOM. Start with smaller per-device batch size (often 1 for multimodal), then use gradient accumulation for effective batch size.
- If OOM persists, use more GPUs via CUDA_VISIBLE_DEVICES and/or switch to a DeepSpeed setup, for example ms-swift/finetune/qwen3/zero3.sh.
- Script comments in ms-swift/finetune/qwen25/sft_lora_enhance.sh already include practical memory guidance; follow those first.

## Post-Training Merge

After training, merge LoRA adapter weights into a standalone Guard model:

```bash
bash ms-swift/finetune/sft_lora_merge.sh <checkpoint_path> <base_model_path>
```

Example:

```bash
bash ms-swift/finetune/sft_lora_merge.sh ms-swift/ckpt/your_adapter_ckpt models/base/Qwen2.5-Omni-7B
```

Important naming note:

- Keep a model-family key in the merged Guard model name (for example qwen25), so downstream loading logic can identify the correct model family.
