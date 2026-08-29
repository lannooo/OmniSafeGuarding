# Prepare the Evaluation and Benchmarks

> [!CAUTION]
> Harmful Content Disclaimer: This dataset collection may include examples that are disturbing, harmful, or upsetting. It covers high-risk categories such as discriminatory language, abuse, violence, self-harm, sexual content, misinformation, and other unsafe multimodal content. These data are intended to support safety research and development of safer LLM/MLLM systems.

We provide a unified framework for guardrail evaluation under scenarios of multi-modal safety moderation (see ``module/load_datasets.py``,`module/predefined_datasets.py`,`module/predefined_modalities.py`). The framework consolidates various benchmarks across text, image, video, and audio modalities (as well as cross-modal) under a common risk-labeling scheme.

This repo includes **one-click** scripts for downloading and preparing public Hugging Face datasets or downloading from other resources.

## Quick Start

> [!WARNING]
> Preparing the complete collection of datasets may require more than 150GB disk space. Downloading most complete datasets covered by the scripts may consume around 80GB network traffic. Please choose a target path with enough free space before downloading or extracting datasets.

### Prepare Benchmark Data

#### Access Reminder

Some HF dataset require confirmation or approval for granted access. We recommend to manually operate them in browser, confirming access, and accepting any required terms first before download:

- [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench)
- [ys-zong/VLGuard](https://huggingface.co/datasets/ys-zong/VLGuard)
- [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench)
- [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)
- [BAAI/Video-SafetyBench](https://huggingface.co/datasets/BAAI/Video-SafetyBench)
- [etri-vilab/holisafe-bench](https://huggingface.co/datasets/etri-vilab/holisafe-bench)
- [Virtue-AI-HUB/SafeWatch-Bench](https://huggingface.co/datasets/Virtue-AI-HUB/SafeWatch-Bench)

#### Huggingface Login

Login your hugging face account before running downloader scripts:

```bash
# using token, You can create the token in Hugging Face account settings
hf auth login
```

#### **Hugging Face downloader**

```bash
bash prepare_omniubench_hf.sh
```

The downloader reads per-dataset method from `hf_datasets.txt`:

- `[hf]` force `hf download`
- `[hfd]` use `hfd.sh` if present, otherwise fallback to `hf download`
- `[skip]` skip this dataset line
- no prefix means default `[hf]`

#### **Prepare external (non-Hugging Face) datasets**

```bash
bash prepare_omniubench_external.sh
```

Optionally specify a custom target directory:

```bash
bash prepare_omniubench_hf.sh /path/to/data
bash prepare_omniubench_external.sh /path/to/data
```

### Hugging Face Datasets

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

### Post-Processing

#### VoiceBench Post-Processing

`hlt-lab/voicebench` is downloaded as parquet files containing audio bytes.
The HF preparation script now runs VoiceBench conversion automatically after download.

If you need to rerun it manually:

```bash
python data/voicebench_convert.py
```

#### MMBench-Video Post-Processing

After downloading MMBench-Video, the HF preparation script automatically unwraps pkl payloads into video files.
The conversion script is `data/mmbench_video_unwrap_pkl.py`, and outputs are placed under the dataset's `video/` directory.

#### Video-SafetyBench Post-Processing

After downloading `BAAI/Video-SafetyBench`, the HF preparation script automatically extracts `video.tar.gz` under the dataset directory, producing a `video/` subdirectory.

## External Datasets

Datasets not available on Hugging Face are prepared via `prepare_omniubench_data_external.sh`.

```bash
# Ensure the external datasets zip files are in data/external/
bash prepare_omniubench_external.sh
```

**Note**: Since these subsets/media are large and typically require ``git-lfs``, they may be unavailable in anonymous repo snapshots. For this reason, related evaluation datasets are commented out in default evaluation configs. After you prepare these datasets locally, uncomment the corresponding eval entries.

The following datasets are provided in mini/partial form for quick runs, while full datasets can be downloaded separately in the following instructions:

- `JailBreakV-28k`
- `MML-SafeBench`
- `Omniguard_Custom`
- `SafeWatch-Bench`

### JailBreakV-28k (Mini)

For quick evaluation, this repository provides a mini package:

- `data/external/JailBreakV-28k-mini.zip` -> `data/JailBreakV-28k`

If you need the full JailBreakV-28K dataset, download it separately from the official source (full dataset: [JailbreakV](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)).

### JailbreakBench

**Preferred method:** Place `JailbreakBench.zip` in `data/external/` and run the script. It will extract to `data/JailbreakBench`.

**Fallback method:** If the zip is not available and `git` is installed, the script will clone from [JailbreakBench/artifacts](https://github.com/JailbreakBench/artifacts.git).

### CipherChat

Place `CipherChat.zip` in `data/external/` and run the script. It will extract to `data/CipherChat`.

### jailbreak_llms (DAN)

**Preferred method:** Place `DAN.zip` in `data/external/` and run the script. It will extract to `data/jailbreak_llms`.

**Fallback method:** If the zip is not available and `git` is installed, the script will clone from [verazuo/jailbreak_llms](https://github.com/verazuo/jailbreak_llms.git) into `data/jailbreak_llms`.

After clone, the script also unzips `data/jailbreak_llms/data/forbidden_question/forbidden_question_set_with_prompts.csv.zip` in place.

### FigStep

**Preferred method:** Place `FigStep.zip` in `data/external/` and run the script. It will extract to `data/FigStep`.

**Fallback method:** If the zip is not available and `git` is installed, the script will clone from [CryptoAILab/FigStep](https://github.com/CryptoAILab/FigStep.git) into `data/FigStep`.

### MML-SafeBench (Mini)

Place `MML-SafeBench.zip` in `data/external/` and run the script. It will extract to `data/MML-SafeBench`.

This package is a subset for quick evaluation. For the full dataset, follow the download instructions linked from the GitHub page: [wangyu-ovo/MML](https://github.com/wangyu-ovo/MML).

### SafeWatch-Bench (Mini)

`Virtue-AI-HUB/SafeWatch-Bench` on Hugging Face requires both form confirmation and approval before access is granted.

For quick evaluation, this repository provides a mini package:

- `data/external/SafeWatch-Bench-mini.zip` -> `data/SafeWatch-Bench`

### Omniguard_Custom (Mini)

This dataset is used for evaluating modality bias, this repo includes a mini version for quick evaluation. You can also download the full dataset from **Anonymous Zenodo: [Modality-Bias-Eval](https://zenodo.org/records/22157542) . (place under `data/external/`)**

Or generate by your own, see `scripts/generate_omni_custom_quard.py`.

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

- Some datasets are not available on Hugging Face. Non-Hugging Face datasets can be placed directly in this repository (or another agreed path) and managed together with the downloaded datasets.
- For datasets with mini/partial external packages, the provided files are subsets for quick evaluation and do not include full datasets.
- Some datasets may require extra external files beyond the Hugging Face snapshot. By convention, place those supplemental files under `data/external/`. For example, Aegis-2.0 may need `data/external/Suicide_Detection_filtered.csv` to restore missing self-harm-related content during loading.
- As you provide more dataset repo IDs, just edit `hf_datasets.txt`.
- As you add more external datasets, extend `prepare_omniubench_data_external.sh` with additional functions.
- If the pure `hf` script fails with network errors, you can retry directly. In environments with restricted access, you can also configure a mirror endpoint before running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash prepare_omniubench_data_hf.sh
```
