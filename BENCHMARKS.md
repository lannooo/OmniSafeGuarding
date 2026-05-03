# OmniUBench

OmniUBench is a large, unified, multi-modal safety benchmark collection for guardrail research.
It consolidates benchmark data across text, image, video, and audio modalities under a common risk-labeling scheme.

This repo now includes a one-click dataset preparation script for downloading public Hugging Face datasets used by OmniUBench.

The Hugging Face dataset list and local directory names are managed in a single mapping file:

- `hf_datasets.txt`

## Evaluation Dimensions

OmniUBench adopts a multi-dimensional protocol: Safety-Utility-Reliability.
To support target-oriented evaluation, benchmark datasets are grouped by evaluation objective:

- safety: basic safety + jailbreak defense (detect malicious queries and jailbreak samples)
- utility: evaluate recognition/interference on normal and safe samples
- reliability: evaluate cross-modal modality bias and spurious correlations

## Quick Start

> [!WARNING]
> Preparing the complete OmniUBench collection may require more than 150GB disk space.
> Downloading most complete datasets covered by the scripts may consume around 80GB network traffic.
> Please choose a target path with enough free space before downloading or extracting datasets.

> [!CAUTION]
> Harmful Content Disclaimer:
> This benchmark collection may include examples that are disturbing, harmful, or upsetting.
> It covers high-risk categories such as discriminatory language, abuse, violence, self-harm, sexual content, misinformation, and other unsafe multimodal content.
> These data are intended to support safety research and development of safer LLM/MLLM systems.
> Do not train a model exclusively on harmful examples.

### Environment Setup

Create and install the recommended environment with one script:

```bash
bash setup_omguard_env.sh
```

By default, the script creates conda env `omniguard` with Python 3.10 and installs the training/inference/data dependencies used in this project (PyTorch stack, vLLM, transformers, Qwen utils, datasets, and common tooling).

Optional environment variables:

```bash
# custom env name / python version
ENV_NAME=omniguard PYTHON_VERSION=3.10 bash setup_omguard_env.sh

# optional pip mirror
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_omguard_env.sh

# flash-attn behavior: auto (default) | 1 (force install) | 0 (skip)
INSTALL_FLASH_ATTN=auto bash setup_omguard_env.sh
```

Notes:

- `flash-attn` is installed automatically only when CUDA/NVIDIA conditions are detected (or when force-enabled). On macOS or non-CUDA machines it is skipped with a warning.
- After setup, activate with `conda activate omniguard` (or your custom `ENV_NAME`).

### Prepare Benchmark Data

Run from the repository root:

**Hugging Face downloader:**

```bash
bash prepare_omniubench_data_hf.sh
```

The downloader reads per-dataset method from `hf_datasets.txt`:
- `[hf]` force `hf download`
- `[hfd]` use `hfd.sh` if present, otherwise fallback to `hf download`
- `[skip]` skip this dataset line
- no prefix means default `[hf]`

**Prepare external (non-Hugging Face) datasets:**

```bash
bash prepare_omniubench_data_external.sh
```

Optionally specify a custom target directory:

```bash
bash prepare_omniubench_data_hf.sh /path/to/data
bash prepare_omniubench_data_external.sh /path/to/data
```

Model download, training data, and finetuning instructions were moved to [MODELS.md](MODELS.md).

### Hugging Face Access Reminder (Terms + Login)

For smoother setup, we recommend opening the following Hugging Face dataset pages in a browser, confirming access, and accepting any required terms first:

- [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench)
- [ys-zong/VLGuard](https://huggingface.co/datasets/ys-zong/VLGuard)
- [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench)
- [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)
- [BAAI/Video-SafetyBench](https://huggingface.co/datasets/BAAI/Video-SafetyBench)
- [etri-vilab/holisafe-bench](https://huggingface.co/datasets/etri-vilab/holisafe-bench)
- [Virtue-AI-HUB/SafeWatch-Bench](https://huggingface.co/datasets/Virtue-AI-HUB/SafeWatch-Bench)


Although these datasets are public, Hugging Face CLI download may still require account login and terms acceptance.

Some datasets may additionally require manual approval before access is granted. `Virtue-AI-HUB/SafeWatch-Bench` is one such case.

Use a token to login before running downloader scripts:

```bash
hf auth login
```

You can create the token in Hugging Face account settings.

## Hugging Face Datasets

The Hugging Face dataset list is defined in `hf_datasets.txt`:

- HuggingFaceH4/mt_bench_prompts -> mt_bench_prompts
- domenicrosati/TruthfulQA -> TruthfulQA
- walledai/AdvBench -> AdvBench
- nvidia/Aegis-AI-Content-Safety-Dataset-2.0 -> Aegis-2.0
- PKU-Alignment/BeaverTails -> BeaverTails
- PKU-Alignment/PKU-SafeRLHF-QA -> PKU-SafeRLHF-QA
- Bertievidgen/SimpleSafetyTests -> SimpleSafetyTests
- mmathys/openai-moderation-api-evaluation -> openaiMod
- walledai/HarmBench -> HarmBench
- Paul/XSTest -> XSTest
- allenai/wildguardmix -> wildguard
- lmsys/toxic-chat -> toxic-chat
- hlt-lab/voicebench -> voicebench
- JailbreakV-28K/JailBreakV-28k -> JailBreakV-28k
- Foreshhh/vlsbench -> vlsbench
- ys-zong/VLGuard -> VLGuard
- sinwang/SIUO -> SIUO
- etri-vilab/holisafe-bench -> HoliSafe
- Monosail/HADES -> Hades
- MMInstruction/RedTeamingVLM -> RedTeamingVLM
- BAAI/Video-SafetyBench -> Video-SafetyBench
- Virtue-AI-HUB/SafeWatch-Bench -> SafeWatch-Bench

### VoiceBench Post-Processing

`hlt-lab/voicebench` is downloaded as parquet files containing audio bytes.
The HF preparation script now runs VoiceBench conversion automatically after download.

If you need to rerun it manually:

```bash
python data/voicebench_convert.py
```

### MMBench-Video Post-Processing

After downloading MMBench-Video, the HF preparation script automatically unwraps pkl payloads into video files.
The conversion script is `data/mmbench_video_unwrap_pkl.py`, and outputs are placed under the dataset's `video/` directory.

### Video-SafetyBench Post-Processing

After downloading `BAAI/Video-SafetyBench`, the HF preparation script automatically extracts `video.tar.gz` under the dataset directory, producing a `video/` subdirectory.

## External Datasets

Datasets not available on Hugging Face are prepared via `prepare_omniubench_data_external.sh`.

### JailBreakV-28k (Mini Package)

For quick evaluation, this repository provides a mini package:

- `data/external/JailBreakV-28k-mini.zip` -> `data/JailBreakV-28k`

This mini package is a subset and does not include the full dataset.
If you need the full JailBreakV-28K dataset, download it separately from the official source.

### JailbreakBench

**Preferred method:** Place `JailbreakBench.zip` in `data/external/` and run the script. It will extract to `data/JailbreakBench`.

**Fallback method:** If the zip is not available and `git` is installed, the script will clone from [JailbreakBench/artifacts](https://github.com/JailbreakBench/artifacts.git).

Usage:

```bash
# Ensure JailbreakBench.zip is in data/external/
bash prepare_omniubench_data_external.sh
```

### CipherChat

Place `CipherChat.zip` in `data/external/` and run the script. It will extract to `data/CipherChat`.

Usage:

```bash
# Ensure CipherChat.zip is in data/external/
bash prepare_omniubench_data_external.sh
```

### jailbreak_llms (DAN)

**Preferred method:** Place `DAN.zip` in `data/external/` and run the script. It will extract to `data/jailbreak_llms`.

**Fallback method:** If the zip is not available and `git` is installed, the script will clone from [verazuo/jailbreak_llms](https://github.com/verazuo/jailbreak_llms.git) into `data/jailbreak_llms`.

After clone, the script also unzips `data/jailbreak_llms/data/forbidden_question/forbidden_question_set_with_prompts.csv.zip` in place.

Usage:

```bash
# Ensure DAN.zip is in data/external/ (preferred)
bash prepare_omniubench_data_external.sh
```

### FigStep

**Preferred method:** Place `FigStep.zip` in `data/external/` and run the script. It will extract to `data/FigStep`.

**Fallback method:** If the zip is not available and `git` is installed, the script will clone from [CryptoAILab/FigStep](https://github.com/CryptoAILab/FigStep.git) into `data/FigStep`.

Usage:

```bash
# Ensure FigStep.zip is in data/external/ (preferred)
bash prepare_omniubench_data_external.sh
```

### MML-SafeBench

Place `MML-SafeBench.zip` in `data/external/` and run the script. It will extract to `data/MML-SafeBench`.

This package is a subset for quick evaluation and does not include the full dataset.

For the full dataset, follow the download instructions linked from the project's GitHub page:

- [wangyu-ovo/MML](https://github.com/wangyu-ovo/MML)

### SafeWatch-Bench (Mini Package)

`Virtue-AI-HUB/SafeWatch-Bench` on Hugging Face requires both form confirmation and approval before access is granted.

For quick evaluation, this repository provides a mini package:

- `data/external/SafeWatch-Bench-mini.zip` -> `data/SafeWatch-Bench`

This mini package is a subset and does not include the full dataset.

### Omniguard_Custom

Place `Omni_Custom.zip` in `data/external/` and run the script. It will extract to `data/Omniguard_Custom`.

### AIAH

Place `AIAH.zip` in `data/external/` and run the script. It will extract to `data/AIAH`.

### LVLM_NLF

Place `LVLM_NLF.zip` in `data/external/` and run the script. It will extract to `data/LVLM_NLF`.

### AdvBench

Place `AdvBench.zip` in `data/external/` and run the script. It will extract to `data/AdvBench`.

### HarmBench

Place `HarmBench.zip` in `data/external/` and run the script. It will extract to `data/HarmBench`.

### wildguard

Place `wildguard.zip` in `data/external/` and run the script. It will extract to `data/wildguard`.

## Mapping Format

Each non-comment line in `hf_datasets.txt` must follow:

```text
[method] repo_id -> local_subdir
```

Method values:
- `[hf]`: always use `hf download`
- `[hfd]`: use `hfd.sh` when available, fallback to `hf download` when `hfd.sh` is absent
- `[skip]`: skip this dataset line
- no prefix: default is `[hf]`

Example:

```text
[hfd] hlt-lab/voicebench -> voicebench
walledai/AdvBench -> AdvBench
[skip] some/repo -> local_dir
```

This keeps local paths stable even if you later switch to another repo ID for the same dataset.

## Notes

- Some OmniUBench datasets are not available on Hugging Face.
- For datasets with mini/partial external packages, the provided files are subsets for quick evaluation and do not include full datasets.
- Non-Hugging Face datasets can be placed directly in this repository (or another agreed path) and managed together with the downloaded datasets.
- Some datasets may require extra external files beyond the Hugging Face snapshot. By convention, place those supplemental files under `data/external/`. For example, Aegis-2.0 may need `data/external/Suicide_Detection_filtered.csv` to restore missing self-harm-related content during loading.
- As you provide more dataset repo IDs, just edit `hf_datasets.txt`.
- As you add more external datasets, extend `prepare_omniubench_data_external.sh` with additional functions.
- If the pure `hf` script fails with network errors, you can retry directly. In environments with restricted access, you can also configure a mirror endpoint before running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash prepare_omniubench_data_hf.sh
```