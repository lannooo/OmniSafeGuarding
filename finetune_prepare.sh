#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash finetune_prepare.sh <training_media_dir>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PREFIX="$1"

python "$SCRIPT_DIR/ms-swift/rewrite_media_prefix.py" "$TARGET_PREFIX"

echo "[INFO] finetune data prepared under ms-swift/data"