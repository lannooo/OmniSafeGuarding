#!/usr/bin/env bash
set -euo pipefail

# One-click downloader for baseline guardrail models from Hugging Face.
# Usage:
#   bash prepare_guardrails_hf.sh
#   bash prepare_guardrails_hf.sh /path/to/custom/model_dir

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
MODEL_DIR="${1:-${REPO_ROOT}/models/guard}"

# Model download options:
# - Use [hfd] prefix to download with hfd.sh (if available and configured).
# - Use [skip] prefix to skip downloading a model.
MODEL_REPOS=(
  # "Qwen/Qwen3Guard-Gen-8B",
  # "openai/gpt-oss-safeguard-20b",
  # "meta-llama/Llama-Guard-3-8B",
  # "meta-llama/Llama-Guard-3-11B-Vision",
  # "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf",
  # "yueliu1999/GuardReasoner-VL-7B",
  "zhu-thu-22/GuardReasoner-Omni-3B",
  "zhu-thu-22/GuardReasoner-Omni-7B",
  "anonymousICML/OmniGuard-3B",
  "anonymousICML/OmniGuard-7B",
)

echo "[INFO] Target model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[ERROR] Hugging Face CLI (hf) is required but not found in PATH."
  exit 1
fi

ensure_hf_login() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[INFO] HF_TOKEN detected. Logging in to Hugging Face CLI ..."
    hf auth login --token "${HF_TOKEN}" >/dev/null
    return 0
  fi

  if hf auth whoami >/dev/null 2>&1; then
    echo "[INFO] Hugging Face CLI is already logged in."
    return 0
  fi

  echo "[INFO] Hugging Face CLI is not logged in. Starting login flow ..."
  hf auth login
}

download_one_model() {
  local repo_id="$1"
  local target_dir="$2"
  local method="$3"

  if [[ "${method}" == "hfd" && -f "${REPO_ROOT}/hfd.sh" ]]; then
    bash "${REPO_ROOT}/hfd.sh" "${repo_id}" --local-dir "${target_dir}"
  else
    hf download "${repo_id}" --local-dir "${target_dir}"
  fi
}

download_model_with_retry() {
  local repo_id="$1"
  local target_dir="$2"
  local method="$3"
  local max_attempts=3

  for attempt in $(seq 1 "${max_attempts}"); do
    echo "[INFO] Attempt ${attempt}/${max_attempts}: ${repo_id}"
    if download_one_model "${repo_id}" "${target_dir}" "${method}"; then
      return 0
    fi
    [[ ${attempt} -lt ${max_attempts} ]] && echo "[WARN] Retrying ${repo_id} ..."
  done

  echo "[ERROR] Failed to download ${repo_id} after ${max_attempts} attempts."
  return 1
}

ensure_hf_login

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "${s}"
}

for raw_entry in "${MODEL_REPOS[@]}"; do
  line="$(trim "${raw_entry}")"

  method="hf"
  if [[ "${line}" =~ ^\[([a-z]+)\](.*)$ ]]; then
    method="${BASH_REMATCH[1]}"
    line="$(trim "${BASH_REMATCH[2]}")"
  fi

  if [[ "${method}" != "hf" && "${method}" != "hfd" && "${method}" != "skip" ]]; then
    echo "[ERROR] Invalid method '[${method}]' in entry: ${raw_entry}"
    echo "[ERROR] Allowed methods: [hf], [hfd], [skip]"
    exit 1
  fi

  repo_id="${line}"
  if [[ -z "${repo_id}" ]]; then
    echo "[ERROR] Empty repo_id in entry: ${raw_entry}"
    exit 1
  fi

  if [[ "${method}" == "skip" ]]; then
    echo "[INFO] Skipping ${repo_id}"
    continue
  fi

  model_name="${repo_id##*/}"
  target_dir="${MODEL_DIR}/${model_name}"
  mkdir -p "${target_dir}"

  echo "[INFO] Downloading model [${method}] ${repo_id} -> ${target_dir}"
  download_model_with_retry "${repo_id}" "${target_dir}" "${method}"
done

echo "[INFO] Baseline models are ready."
