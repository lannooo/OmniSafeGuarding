# Unified Yet Fragile? Uncovering the Cross-modal Pitfalls of Omni-Modal Safety Guardrails
We present **OmniUBench**, a comprehensive framework for systematic omni-modal assessment across safety, *utility and reliability*, and **OmGuard**, an enhanced guardrail system that addresses modality bias and spurious correlations to achieve state-of-the-art moderation performance across text, image, audio, and video.

> [!CAUTION]
> Harmful Content Disclaimer:
> This project involves safety benchmarks that may contain harmful, disturbing, or offensive examples.
> These data are provided strictly for safety research and guardrail development.

## Quick Start

### Environment Setup [Required]

This framework is built around Python 3.10, PyTorch 2.10, Transformers >= 5.2, and vLLM.
Create and install the recommended environment with one script:

```bash
# Prerequisite: conda installation
bash env_setup.sh

# Optional: disable flash-attn install
# INSTALL_FLASH_ATTN=0 bash env_setup.sh

# Optional: manually install dependencies in an existing Python 3.10 env
# pip install -r requirements.txt
```

Notes: `flash-attn` is optional. In some setups, `flash-attn` installation may fail due to CUDA/toolchain compatibility; this does not block running the benchmark pipeline. But we recommend enabling it for better performance.

### Prepare Benchmark [Required]

Most benchmark datasets are publicly accessible from Hugging Face and can be prepared with `prepare_omniubench_hf.sh`.
Some datasets requiring alternative fetching (e.g., `git clone`) and those require huggingface login/access, we also provide packaged resources (or a `mini` version) under `data/external`.

#### Huggingface Datasets:

This step can take a long time. Make sure you have enough free disk space (around 150GB). Full download may consume huge network traffic (around 80GB). See `hf_datasets.txt` for the dataset list.

```bash
# By default downloaded under data/...
bash prepare_omniubench_hf.sh
```
Recommendation: login to Hugging Face and generate an access token, and login to ensure the required access permissions:
```bash
hf auth login
```


#### Local Datasets:

This step fetch data from github repo, and extracts local packaged datasets (as well as labels) into the data directory. Make sure required files already exist under `data/external` before running.

```bash
# Prerequisite: git
bash prepare_omniubench_external.sh
```

The following mini datasets may be missing in anonymous repo releases, 
(too large and are managed via Git LFS). They are disabled in default evaluation configs; after preparing local data, uncomment related eval sets.:
- `data/external/JailBreakV-28k-mini.zip`
- `data/external/MML-SafeBench.zip`
- `data/external/Omniguard_Custom_mini.zip`: download from anonymous Zenodo (https://zenodo.org/records/19999112) and place under `data/external/`


For dataset-specific issues and access reminders (HF terms/login), see [BENCHMARKS.md](BENCHMARKS.md).


### Prepare Base Models [Required]

Model training and inference both require base models.
By default, models are saved into `models/base`, and with only `Qwen/Qwen2.5-Omni-7B` be downloaded, feel free to uncomment other model options in `prepare_base_models_hf.sh`.
Since these model checkpoints are large, make sure you have sufficient free disk space.

```bash
bash prepare_base_models_hf.sh
```

For model download options, training data, and finetuning preparation, see [MODELS.md](MODELS.md).

### A Quick Inference Example

This is a key sanity-check step for the full pipeline.
Running this script verifies that your environment and base model setup are working correctly.
The script loads a base Omni model and runs some test cases for either multimodal QA or zero-shot guardrail inference.

```bash
python inference_example.py
```

### Run a Mini Benchmark validation

This step loads your base model and runs a mini benchmarking with `fast` subset. 

```bash
# Example: run mini benchmark on GPU 0
CUDA_VISIBLE_DEVICES=0 python eval_omniguard_sft_all.py \
  --model_path ./models/base/Qwen2.5-Omni-7B \
  --enable_vllm
# If custom data_dir is specified
DATA_DIR=/path/to/data CUDA_VISIBLE_DEVICES=0 python eval_omniguard_sft_all.py --model_path ./models/base/Qwen2.5-Omni-7B --enable_vllm
```


OmniUBench uses a multi-dimensional evaluation protocol (Safety-Utility-Reliability),
and datasets are grouped accordingly for target-oriented evaluation:
- safety: includes basic safety and jailbreak defense (detecting malicious queries and jailbreak samples)
- utility: evaluates recognition/interference on normal and safe samples
- reliability: evaluates cross-modal modality bias and spurious correlations

Be free to uncomment the benchmark settings in `eval_omniguard_sft_all.py` for a full benchmarking!
```python
eval_sets = {
  # For fast evaluate the whole inference framework
  "fast": eval_fast_validate,

  # Uncomment for complete evaluation
  "utility": eval_general,
  "safety": eval_basic_safety,
  "jailbk": eval_ext_jailbreaks,
}
```


Expected outputs:
- Displayed Metrics for each dataset/benchmark
- Per-dataset predictions under `output/metric.logs/<model_namh e>.<prompt-version>/`
- Brief log under `logs/<model_name>.<version>.metric.log`

### Benchmarking fine-tuned Safety Guardrail 
Merging LoRa adapter weights:
```bash
bash ms-swift/finetune/sft_lora_merge.sh <ckpt-path> <base-model-path>
# e.g., merge our OmGuard weights with Qwen2.5-Omni-7B 
bash ms-swift/finetune/sft_lora_merge.sh ms-swift/ckpt/OmGuard-Qwen25-7B-Enhance-lora-adapter models/base/Qwen2.5-Omni-7B
# move the merged model, give it a name containing Qwen25, e.g.,
mv ms-swift/ckpt/OmGuard-Qwen25-7B-Enhance-lora-adapter_merged models/guard/OmGuard-Qwen25-7B-Enhance
```

Now we can benchmark the merged fine-tuned OmGuard model:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_omniguard_sft_all.py \
  --model_path ./models/guard/OmGuard-Qwen25-7B-Enhance \
  --enable_vllm
```

### Project Layout
- `module/`: shared data/model/inference utilities
- `baseline/`: baseline guardrail implementations
- `data/`: benchmark and external datasets
- `ms-swift/`: finetuning and related scripts
- `output/`: prediction artifacts and metric outputs


## Reproduce Metrics

There are two clear reproduction paths.

### Path 1: Reproduce by Running Inference

Use this path when you want full end-to-end reproducibility from model outputs (it will take more time).

```bash
# benchmarking in one script
python eval_omniguard_sft_all.py

# or, inference on specific dataset
python eval_omniguard_sft.py
```

This produces prediction outputs and logs that can be used for metric calculation and report generation.

### Path 2: Reproduce Directly from Intermediate Results

Use this path when you want quick metric verification without rerunning all model inference.

- Reuse released intermediate outputs in `output/`
- Intermediate results are stored in `output/experiments/<model_name>/`
- Open `metrics_report.ipynb` and modify three selections in the main comparison cell:

```python
# Step1: select models above to compare
select_models = qwen3_30B_guards

# Step2: select evaluate dimention
## [General]: benign queries (utility)
## [Safety]: harmful queries (safety)
## [Jailbreak]: jailbreak queries (safety)
## [FalseReject]: false reject queries, constructed with specific benign text-patterns and benign image/audio/video (reliability)
## [ModalityBias]: queries with only one modality showing harmful, while the other modality(ies) showing benign (reliability)
dimension = "Safety"

# Step3: select modalities to compare: "Text", "Image", "Audio", "Video"
modalities = ["Image"]
```

- Run the notebook cell to regenerate metric tables from intermediate outputs.


## Benchmarking Other Baselines

### Prepare Baseline Guardrails
The baseline guardrails can be downloaded directly from Hugging Face. We provide a dedicated downloader under `baseline/`. Ensure that you have sufficient free space. 
```bash
bash baseline/prepare_guardrails_hf.sh
```
Current baseline model options are as follows. By default, the script downloads the model into `models/guard`, be free to comment those you do not want to download:
- `Qwen/Qwen3Guard-Gen-8B`
- `Qwen/Qwen3Guard-Gen-8B`
- `openai/gpt-oss-safeguard-20b`
- `meta-llama/Llama-Guard-3-8B`
- `meta-llama/Llama-Guard-3-11B-Vision`
- `AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf`
- `yueliu1999/GuardReasoner-VL-7B`
- `zhu-thu-22/GuardReasoner-Omni-4B`
- `anonymousICML/OmniGuard-7B`

### Benchmarking Baselines

Each baseline script under `baseline/` can be run directly with a model path, a target dataset, and a GPU id.

Example: benchmark Qwen3Guard on a text dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/bsl_qwen3guard.py \
  --model_path ./models/guard/Qwen3Guard-Gen-8B \
  --dataset advbench
```

The detailed predictions are alwo writen into the metric logs under `output/metric.logs/`.
For other baselines, see the corresponding script in `baseline/` for more details.


## Train your own Guardrail

Training your own guardrail within three steps.

### Step 1. Fetch multimodal training media

First download the multimodal resources used in training, including images, audio, and other media files.
Because these resources are large, we provide them through a completely anonymous record on zenodo. 

Conduct the script to download:

```bash
bash finetune_fetch_media.sh ./ms-swift
```

This downloads and extracts the training media under `ms-swift/media`.

### Step 2. fine-tuning with `ms-swift` framework

Before training, prepare finetuning json files by rewriting media paths to your actual local directory.

```bash
# Example: media downloaded to ms-swift/media in Step 1
bash finetune_prepare.sh ./ms-swift/media
```

This regenerates finetuning data files under `ms-swift/data` with updated media prefixes.

### Step 3. Run the swift finetuning script

```bash
bash ms-swift/finetune/qwen25/sft_lora_enhance.sh
```

Before running, make sure the following paths in the script are correct for your setup:

- `MODEL_NAME` (base model path)
- `INPUT` (finetuning json path)
- `OUTPUT` (checkpoint output path)

Notes:

- Multimodal training may cause OOM (ensure that GPU has memory more than 40GB). If that happens, adjust settings in the finetuning script (more visible GPUs and/or reduce batch size).

After fine-tuning, you can obtain your own Guard model through the same merge step, and re-run them for benchmarking!


## License & Usage

- **Code**: Unless otherwise noted, this repository's code is intended to be released under the MIT License to facilitate study in this field.
- **Dataset/Training Data**: The OmniUBench-related dataset resources are intended for academic research only, under CC BY-NC 4.0.
- **Model Weights and Third-Party Assets**: Follow the original license and terms of each upstream model/dataset provider (e.g., Hugging Face datasets/models).

By using this repository, you agree to:

- Use the resources for lawful and responsible safety research.
- Not use the data/models to facilitate harmful, abusive, or illegal activities.
- Comply with all applicable upstream licenses, usage terms, and local regulations.