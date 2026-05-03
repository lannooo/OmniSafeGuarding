#!/usr/bin/env bash
set -euo pipefail

# One-click downloader for public OmniUBench datasets via hfd.sh.
# Usage:
#   bash prepare_omniubench_data_hf.sh
#   bash prepare_omniubench_data_hf.sh /path/to/custom/data_dir

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/data}"
CONFIG_FILE="${ROOT_DIR}/hf_datasets.txt"

echo "[INFO] Target data directory: ${DATA_DIR}"
echo "[INFO] Dataset mapping file:  ${CONFIG_FILE}"
mkdir -p "${DATA_DIR}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Mapping file not found: ${CONFIG_FILE}"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 is required but not found in PATH."
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "[ERROR] Hugging Face CLI (hf) is required but not found in PATH."
  exit 1
fi

ensure_hf_login() {
  # Prefer non-interactive auth when HF_TOKEN is provided.
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

ensure_hf_login

download_one() {
  local repo_id="$1"
  local target_dir="$2"
  local method="$3"  # hf | hfd

  if [[ "${method}" == "hfd" && -f "${ROOT_DIR}/hfd.sh" ]]; then
    bash "${ROOT_DIR}/hfd.sh" "${repo_id}" --dataset --local-dir "${target_dir}"
  else
    hf download --repo-type dataset "${repo_id}" --local-dir "${target_dir}"
  fi
}

download_with_retry() {
  local repo_id="$1"
  local target_dir="$2"
  local method="$3"
  local max_attempts=3

  for attempt in $(seq 1 "${max_attempts}"); do
    echo "[INFO] Attempt ${attempt}/${max_attempts}: ${repo_id}"
    if download_one "${repo_id}" "${target_dir}" "${method}"; then
      return 0
    fi
    [[ ${attempt} -lt ${max_attempts} ]] && echo "[WARN] Retrying ${repo_id} ..."
  done

  echo "[ERROR] Failed to download ${repo_id} after ${max_attempts} attempts."
  return 1
}

run_voicebench_postprocess() {
  local voicebench_root="$1"
  local convert_script="${ROOT_DIR}/data/voicebench_convert.py"

  if [[ ! -d "${voicebench_root}" ]]; then
    echo "[WARN] voicebench directory not found: ${voicebench_root}. Skipping post-processing."
    return 0
  fi

  if [[ ! -f "${convert_script}" ]]; then
    echo "[WARN] VoiceBench convert script not found: ${convert_script}. Skipping post-processing."
    return 0
  fi

  echo "[INFO] Running VoiceBench post-processing ..."
  VOICEBENCH_ROOT="${voicebench_root}" python3 "${convert_script}"
}

run_audiojailbreak_postprocess() {
  local audiojailbreak_root="$1"
  local replacement_readme="${DATA_DIR}/external/AudioJailbreak_Replace_README.md"
  local target_readme="${audiojailbreak_root}/README.md"

  if [[ ! -f "${replacement_readme}" ]]; then
    echo "[WARN] AudioJailbreak replacement README not found: ${replacement_readme}. Skipping README replacement."
    return 0
  fi

  cp "${replacement_readme}" "${target_readme}"
  echo "[INFO] Replaced AudioJailbreak README with ${replacement_readme}"
}

run_safebench_postprocess() {
  local safebench_root="$1"
  local merged_tar="${safebench_root}/safebench.tar.gz"

  if ! compgen -G "${safebench_root}/part_*" >/dev/null; then
    echo "[WARN] No split files found for safebench under ${safebench_root}. Skipping merge/extract."
    return 0
  fi

  echo "[INFO] Merging safebench parts into ${merged_tar} ..."
  (
    cd "${safebench_root}"
    cat part_* > safebench.tar.gz
  )

  echo "[INFO] Extracting ${merged_tar} ..."
  tar -xzf "${merged_tar}" -C "${safebench_root}"

  if [[ ! -d "${safebench_root}/final_bench" ]]; then
    echo "[ERROR] safebench post-processing finished but final_bench not found in ${safebench_root}"
    return 1
  fi

  echo "[INFO] safebench post-processing done: final_bench is ready"
}

run_omni_safetybench_postprocess() {
  local osb_root="$1"
  local mm_data_root="${osb_root}/mm_data"
  local tar_count=0
  local extracted_files=0

  if [[ ! -d "${mm_data_root}" ]]; then
    echo "[ERROR] Omni-SafetyBench mm_data directory not found: ${mm_data_root}"
    return 1
  fi

  tar_count="$(find "${mm_data_root}" -type f -name data.tar | wc -l | tr -d '[:space:]')"
  if [[ "${tar_count}" == "0" ]]; then
    echo "[WARN] No data.tar found under ${mm_data_root}. Skipping extraction."
    return 0
  fi

  echo "[INFO] Extracting Omni-SafetyBench multimedia archives under ${mm_data_root} ..."
  while IFS= read -r tar_file; do
    [[ -z "${tar_file}" ]] && continue
    local tar_dir
    tar_dir="$(dirname "${tar_file}")"
    echo "[INFO] Extracting ${tar_file} -> ${tar_dir}"
    tar -xf "${tar_file}" -C "${tar_dir}"
  done < <(find "${mm_data_root}" -type f -name data.tar | sort)

  extracted_files="$(find "${mm_data_root}" -type f ! -name data.tar | wc -l | tr -d '[:space:]')"
  if [[ "${extracted_files}" == "0" ]]; then
    echo "[ERROR] Omni-SafetyBench extraction produced no multimedia files under ${mm_data_root}"
    return 1
  fi

  echo "[INFO] Omni-SafetyBench post-processing done: extracted ${extracted_files} files"
}

run_mmbench_video_postprocess() {
  local mmbench_video_root="$1"
  local unwrap_script="${ROOT_DIR}/data/mmbench_video_unwrap_pkl.py"

  if [[ ! -f "${unwrap_script}" ]]; then
    echo "[WARN] MMBench-Video unwrap script not found: ${unwrap_script}. Skipping post-processing."
    return 0
  fi

  echo "[INFO] Running MMBench-Video post-processing ..."
  MMBENCH_VIDEO_ROOT="${mmbench_video_root}" python3 "${unwrap_script}" "${mmbench_video_root}"
}

run_video_safetybench_postprocess() {
  local root="$1"
  local tar_path="${root}/video.tar.gz"

  if [[ ! -f "${tar_path}" ]]; then
    echo "[WARN] Video-SafetyBench video.tar.gz not found: ${tar_path}. Skipping post-processing."
    return 0
  fi

  echo "[INFO] Extracting Video-SafetyBench video.tar.gz ..."
  tar -xzf "${tar_path}" -C "${root}"
  echo "[INFO] Video-SafetyBench post-processing done: video/ extracted"
}

run_vlguard_postprocess() {
  local vlguard_root="$1"

  echo "[INFO] Running VLGuard post-processing ..."
  for split in test train; do
    local zip_path="${vlguard_root}/${split}.zip"
    if [[ -f "${zip_path}" ]]; then
      echo "[INFO] Extracting ${zip_path} ..."
      unzip -q "${zip_path}" -d "${vlguard_root}"
    else
      echo "[WARN] VLGuard ${split}.zip not found: ${zip_path}"
    fi
  done
  echo "[INFO] VLGuard post-processing done"
}

run_postprocess_for_dataset() {
  local repo_id="$1"
  local target_dir="$2"

  case "${repo_id}" in
    MBZUAI/AudioJailbreak)
      run_audiojailbreak_postprocess "${target_dir}"
      ;;
    opencompass/MMBench-Video)
      run_mmbench_video_postprocess "${target_dir}"
      ;;
    Leyiii/Omni-SafetyBench)
      run_omni_safetybench_postprocess "${target_dir}"
      ;;
    Zonghao2025/safebench)
      run_safebench_postprocess "${target_dir}"
      ;;
    hlt-lab/voicebench)
      run_voicebench_postprocess "${target_dir}"
      ;;
    ys-zong/VLGuard)
      run_vlguard_postprocess "${target_dir}"
      ;;
    BAAI/Video-SafetyBench)
      run_video_safetybench_postprocess "${target_dir}"
      ;;
  esac
}

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "${s}"
}

while IFS= read -r line || [[ -n "${line}" ]]; do
  line="$(trim "${line}")"
  [[ -z "${line}" || "${line}" == \#* ]] && continue

  if [[ "${line}" != *"->"* ]]; then
    echo "[ERROR] Invalid mapping line: ${line}"
    echo "[ERROR] Expected format: repo_id -> local_subdir"
    exit 1
  fi

  # Parse optional [method] prefix
  method="hf"
  if [[ "${line}" =~ ^\[([a-z]+)\](.*)$ ]]; then
    method="${BASH_REMATCH[1]}"
    line="$(trim "${BASH_REMATCH[2]}")"
  fi

  if [[ "${method}" != "hf" && "${method}" != "hfd" && "${method}" != "skip" ]]; then
    echo "[ERROR] Invalid method '[${method}]' in line: ${line}"
    echo "[ERROR] Allowed methods: [hf], [hfd], [skip]"
    exit 1
  fi

  repo_id="$(trim "${line%%->*}")"
  local_subdir="$(trim "${line#*->}")"

  if [[ -z "${repo_id}" || -z "${local_subdir}" ]]; then
    echo "[ERROR] Empty repo_id or local_subdir in: ${line}"
    exit 1
  fi

  if [[ "${method}" == "skip" ]]; then
    echo "[INFO] Skipping ${repo_id} -> ${local_subdir}"
    continue
  fi

  target_dir="${DATA_DIR}/${local_subdir}"
  mkdir -p "${target_dir}"
  echo "[INFO] Downloading [${method}] ${repo_id} -> ${target_dir}"
  download_with_retry "${repo_id}" "${target_dir}" "${method}"
  run_postprocess_for_dataset "${repo_id}" "${target_dir}"
done < "${CONFIG_FILE}"

echo "[INFO] All datasets downloaded."
