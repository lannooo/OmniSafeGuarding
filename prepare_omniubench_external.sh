#!/usr/bin/env bash
set -euo pipefail

# Prepare external OmniUBench datasets that are not on Hugging Face.
# Supports both local zip files and git clone as fallback.
# Usage:
#   bash prepare_omniubench_data_external.sh
#   bash prepare_omniubench_data_external.sh /path/to/custom/data_dir

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/data}"
EXTERNAL_DIR="${ROOT_DIR}/data/external"

echo "[INFO] Target data directory: ${DATA_DIR}"
echo "[INFO] External source directory: ${EXTERNAL_DIR}"
mkdir -p "${DATA_DIR}"

# Prepare JailbreakBench from zip or git clone
prepare_jailbreakbench() {
  local target_dir="${DATA_DIR}/JailbreakBench"
  local zip_path="${EXTERNAL_DIR}/JailbreakBench.zip"
  local git_url="https://github.com/JailbreakBench/artifacts.git"

  echo "[INFO] Preparing JailbreakBench ..."

  if [[ -f "${zip_path}" ]]; then
    echo "[INFO] Found ${zip_path}, extracting ..."
    mkdir -p "${target_dir}"
    unzip -q "${zip_path}" -d "${target_dir}"
    echo "[INFO] Extracted JailbreakBench to ${target_dir}"
  elif command -v git >/dev/null 2>&1; then
    echo "[INFO] Zip not found. Attempting git clone from ${git_url} ..."
    git clone "${git_url}" "${target_dir}"
    echo "[INFO] Cloned JailbreakBench to ${target_dir}"
  else
    echo "[ERROR] JailbreakBench preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR]   - Git not available for fallback clone"
    echo "[ERROR] Please provide JailbreakBench.zip in ${EXTERNAL_DIR}/ or install git."
    return 1
  fi
}

prepare_cipherchat() {
  local target_dir="${DATA_DIR}/CipherChat"
  local zip_path="${EXTERNAL_DIR}/CipherChat.zip"

  echo "[INFO] Preparing CipherChat ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] CipherChat preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide CipherChat.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted CipherChat to ${target_dir}"
}

prepare_jailbreak_llms() {
  local target_dir="${DATA_DIR}/jailbreak_llms"
  local zip_path="${EXTERNAL_DIR}/DAN.zip"
  local git_url="https://github.com/verazuo/jailbreak_llms.git"
  local fq_dir="${target_dir}/data/forbidden_question"
  local fq_zip="${fq_dir}/forbidden_question_set_with_prompts.csv.zip"

  echo "[INFO] Preparing jailbreak_llms ..."

  if [[ -f "${zip_path}" ]]; then
    echo "[INFO] Found ${zip_path}, extracting ..."
    mkdir -p "${target_dir}"
    unzip -q "${zip_path}" -d "${target_dir}"
    echo "[INFO] Extracted jailbreak_llms to ${target_dir}"
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] jailbreak_llms preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR]   - Git not available for fallback clone"
    echo "[ERROR] Please provide DAN.zip in ${EXTERNAL_DIR}/ or install git."
    return 1
  fi

  if [[ -d "${target_dir}/.git" ]]; then
    echo "[INFO] Existing git repository found at ${target_dir}, skipping clone."
  elif [[ -d "${target_dir}" && -n "$(ls -A "${target_dir}" 2>/dev/null)" ]]; then
    echo "[ERROR] Target directory already exists and is not empty: ${target_dir}"
    echo "[ERROR] Please clear it or provide ${zip_path}."
    return 1
  else
    echo "[INFO] Zip not found. Attempting git clone from ${git_url} ..."
    git clone "${git_url}" "${target_dir}"
    echo "[INFO] Cloned jailbreak_llms to ${target_dir}"
  fi

  if [[ -f "${fq_zip}" ]]; then
    echo "[INFO] Extracting $(basename "${fq_zip}") in ${fq_dir} ..."
    unzip -o -q "${fq_zip}" -d "${fq_dir}"
    echo "[INFO] Extracted forbidden question csv in ${fq_dir}"
  else
    echo "[WARN] Expected zip not found after clone: ${fq_zip}"
  fi
}

prepare_aiah() {
  local target_dir="${DATA_DIR}/AIAH"
  local zip_path="${EXTERNAL_DIR}/AIAH.zip"

  echo "[INFO] Preparing AIAH ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] AIAH preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide AIAH.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted AIAH to ${target_dir}"
}

prepare_lvlm_nlf() {
  local target_dir="${DATA_DIR}/LVLM_NLF"
  local zip_path="${EXTERNAL_DIR}/LVLM_NLF.zip"

  echo "[INFO] Preparing LVLM_NLF ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] LVLM_NLF preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide LVLM_NLF.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted LVLM_NLF to ${target_dir}"
}

prepare_jailbreakv28k_mini() {
  local target_dir="${DATA_DIR}/JailBreakV-28k"
  local zip_path="${EXTERNAL_DIR}/JailBreakV-28k-mini.zip"

  echo "[INFO] Preparing JailBreakV-28k (mini) ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] JailBreakV-28k preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide JailBreakV-28k-mini.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted JailBreakV-28k (mini) to ${target_dir}"
}

prepare_advbench() {
  local target_dir="${DATA_DIR}/AdvBench"
  local zip_path="${EXTERNAL_DIR}/AdvBench.zip"

  echo "[INFO] Preparing AdvBench ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] AdvBench preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide AdvBench.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted AdvBench to ${target_dir}"
}

prepare_harmbench() {
  local target_dir="${DATA_DIR}/HarmBench"
  local zip_path="${EXTERNAL_DIR}/HarmBench.zip"

  echo "[INFO] Preparing HarmBench ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] HarmBench preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide HarmBench.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted HarmBench to ${target_dir}"
}

prepare_wildguard() {
  local target_dir="${DATA_DIR}/wildguard"
  local zip_path="${EXTERNAL_DIR}/wildguard.zip"

  echo "[INFO] Preparing wildguard ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] wildguard preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide wildguard.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted wildguard to ${target_dir}"
}

prepare_figstep() {
  local target_dir="${DATA_DIR}/FigStep"
  local zip_path="${EXTERNAL_DIR}/FigStep.zip"
  local git_url="https://github.com/CryptoAILab/FigStep.git"

  echo "[INFO] Preparing FigStep ..."

  if [[ -f "${zip_path}" ]]; then
    echo "[INFO] Found ${zip_path}, extracting ..."
    mkdir -p "${target_dir}"
    unzip -q "${zip_path}" -d "${target_dir}"
    echo "[INFO] Extracted FigStep to ${target_dir}"
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] FigStep preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR]   - Git not available for fallback clone"
    echo "[ERROR] Please provide FigStep.zip in ${EXTERNAL_DIR}/ or install git."
    return 1
  fi

  if [[ -d "${target_dir}/.git" ]]; then
    echo "[INFO] Existing git repository found at ${target_dir}, skipping clone."
    return 0
  elif [[ -d "${target_dir}" && -n "$(ls -A "${target_dir}" 2>/dev/null)" ]]; then
    echo "[ERROR] Target directory already exists and is not empty: ${target_dir}"
    echo "[ERROR] Please clear it or provide ${zip_path}."
    return 1
  fi

  echo "[INFO] Zip not found. Attempting git clone from ${git_url} ..."
  git clone "${git_url}" "${target_dir}"
  echo "[INFO] Cloned FigStep to ${target_dir}"
}

prepare_mml_safebench() {
  local target_dir="${DATA_DIR}/MML-SafeBench"
  local zip_path="${EXTERNAL_DIR}/MML-SafeBench.zip"

  echo "[INFO] Preparing MML-SafeBench ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] MML-SafeBench preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide MML-SafeBench.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted MML-SafeBench to ${target_dir}"
}

prepare_safewatch_bench_mini() {
  local target_dir="${DATA_DIR}/SafeWatch-Bench"
  local zip_path="${EXTERNAL_DIR}/SafeWatch-Bench-mini.zip"

  echo "[INFO] Preparing SafeWatch-Bench (mini) ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] SafeWatch-Bench preparation failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide SafeWatch-Bench-mini.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  mkdir -p "${target_dir}"
  unzip -q "${zip_path}" -d "${target_dir}"
  echo "[INFO] Extracted SafeWatch-Bench (mini) to ${target_dir}"
}

prepare_omnicustom_bench_mini() {
  local target_dir="${DATA_DIR}/Omniguard_Custom"
  local full_zip_path="${EXTERNAL_DIR}/Omni_Custom_v4.zip"
  local mini_zip_path="${EXTERNAL_DIR}/Omniguard_Custom_mini.zip"

  echo "[INFO] Preparing OmniCustom ..."

  if [[ -f "${full_zip_path}" ]]; then
    echo "[INFO] Found ${full_zip_path}, extracting full dataset ..."
    mkdir -p "${target_dir}"
    unzip -q "${full_zip_path}" -d "${target_dir}"
    echo "[INFO] Extracted OmniCustom (full) to ${target_dir}"
    return 0
  fi

  if [[ -f "${mini_zip_path}" ]]; then
    echo "[INFO] Full dataset zip not found. Falling back to mini package: ${mini_zip_path}"
    mkdir -p "${target_dir}"
    unzip -q "${mini_zip_path}" -d "${target_dir}"
    echo "[INFO] Extracted OmniCustom (mini) to ${target_dir}"
    return 0
  fi

  echo "[ERROR] OmniCustom preparation failed:"
  echo "[ERROR]   - Full zip not found: ${full_zip_path}"
  echo "[ERROR]   - Mini zip not found: ${mini_zip_path}"
  echo "[ERROR] Please provide Omni_Custom_v4.zip (preferred) or Omniguard_Custom_mini.zip in ${EXTERNAL_DIR}/."
  return 1
}



extract_labels_in_place() {
  local zip_path="${EXTERNAL_DIR}/labels.zip"

  echo "[INFO] Extracting labels in place ..."

  if [[ ! -f "${zip_path}" ]]; then
    echo "[ERROR] labels extraction failed:"
    echo "[ERROR]   - Zip not found: ${zip_path}"
    echo "[ERROR] Please provide labels.zip in ${EXTERNAL_DIR}/."
    return 1
  fi

  unzip -q "${zip_path}" -d "${EXTERNAL_DIR}"
  echo "[INFO] Extracted labels.zip to ${EXTERNAL_DIR}"
}

prepare_jailbreakbench
prepare_jailbreakv28k_mini
prepare_cipherchat
prepare_jailbreak_llms
prepare_aiah
prepare_lvlm_nlf
prepare_advbench
prepare_harmbench
prepare_wildguard
prepare_figstep
prepare_mml_safebench
prepare_safewatch_bench_mini
prepare_omnicustom_bench_mini

echo "[INFO] External datasets are prepared."

extract_labels_in_place

echo "[INFO] Labels resources are prepared."