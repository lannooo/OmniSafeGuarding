#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-omguard}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Optional env vars:
#   ENV_NAME=omniguard
#   PYTHON_VERSION=3.10
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   INSTALL_FLASH_ATTN=auto   # auto | 1 | 0

TORCH_PACKAGES=(
  torch==2.10.0
  torchaudio==2.10.0
  torchvision==0.25.0
  torchcodec==0.10.0
)

MODEL_PACKAGES=(
  vllm
  "transformers>=5.2.0"
  accelerate
  qwen-vl-utils
  qwen-omni-utils
  ms-swift
)

DATA_PACKAGES=(
  datasets
  huggingface_hub
  pandas
  scipy
  socksio
  pyarrow
)

UTILITY_PACKAGES=(
  tomlkit
  moviepy
  librosa
  pathos
  umap-learn
  tqdm
  requests
  matplotlib
  dashscope
  openai
  openai-harmony
  ipython
  ipykernel
  ipywidgets
)

echo "[INFO] Setting up conda environment: ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Please install Conda or Miniconda first."
  exit 1
fi

if [[ "$(uname -s)" == "Linux" ]] && [[ -f /etc/os-release ]]; then
  if ! command -v aria2c >/dev/null 2>&1 && grep -qi '^ID=ubuntu\|^ID_LIKE=.*ubuntu' /etc/os-release; then
    echo "[INFO] Installing aria2 for hfd.sh support on Ubuntu ..."
    sudo apt-get update
    sudo apt-get install -y aria2
  fi
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda environment already exists: ${ENV_NAME}"
else
  echo "[INFO] Creating conda environment ${ENV_NAME} with Python ${PYTHON_VERSION} ..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo "[INFO] Upgrading pip ..."
python -m pip install --upgrade pip

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  echo "[INFO] Setting pip index-url: ${PIP_INDEX_URL}"
  python -m pip config set global.index-url "${PIP_INDEX_URL}"
fi

echo "[INFO] Installing PyTorch stack ..."
python -m pip install "${TORCH_PACKAGES[@]}"

echo "[INFO] Installing model/training packages ..."
python -m pip install -U "${MODEL_PACKAGES[@]}"

echo "[INFO] Installing data packages ..."
python -m pip install "${DATA_PACKAGES[@]}"

echo "[INFO] Installing utility packages ..."
python -m pip install "${UTILITY_PACKAGES[@]}"

install_flash_attn="${INSTALL_FLASH_ATTN:-auto}"
if [[ "${install_flash_attn}" == "0" ]]; then
  echo "[INFO] INSTALL_FLASH_ATTN=0, skipping flash-attn installation."
elif [[ "${install_flash_attn}" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[WARN] macOS detected. Skipping flash-attn (typically Linux/CUDA only)."
  elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] No NVIDIA GPU detected. Skipping flash-attn."
  elif ! command -v nvcc >/dev/null 2>&1; then
    echo "[WARN] nvcc not found. flash-attn source build requires CUDA toolkit (nvcc)."
    echo "[WARN] Install CUDA toolkit matching torch CUDA runtime (current torch reports below), then retry."
    python - <<'PY'
import torch
print(f"[WARN] torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
PY
  else
    torch_cuda="$(python - <<'PY'
import torch
print((torch.version.cuda or "").strip())
PY
)"
    nvcc_cuda="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
    if [[ -n "${torch_cuda}" ]] && [[ -n "${nvcc_cuda}" ]] && [[ "${torch_cuda}" != "${nvcc_cuda}" ]]; then
      echo "[WARN] CUDA mismatch: torch=${torch_cuda}, nvcc=${nvcc_cuda}."
      echo "[WARN] flash-attn may fail to compile with mismatched CUDA versions. Skipping in auto mode."
    else
      echo "[INFO] Installing flash-attn ..."
      if ! python -m pip install -U flash-attn --no-build-isolation; then
        echo "[WARN] flash-attn installation failed. Continue without it."
      fi
    fi
  fi
else
  echo "[INFO] INSTALL_FLASH_ATTN=${install_flash_attn}, installing flash-attn ..."
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found. flash-attn source build requires CUDA toolkit (nvcc)."
    exit 1
  fi
  python -m pip install -U flash-attn --no-build-isolation
fi

echo "[INFO] Environment ready."
echo "[INFO] Activate it with: conda activate ${ENV_NAME}"