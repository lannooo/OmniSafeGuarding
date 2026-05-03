#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-.}"
TARGET_DIR="$(realpath "$TARGET_DIR")"

mkdir -p "$TARGET_DIR"

wget -c -O "$TARGET_DIR/omniubench_train.tar.gz.aa" "https://zenodo.org/records/19979658/files/omniubench_train.tar.gz.aa?download=1"
wget -c -O "$TARGET_DIR/omniubench_train.tar.gz.ab" "https://zenodo.org/records/19979658/files/omniubench_train.tar.gz.ab?download=1"
wget -c -O "$TARGET_DIR/omniubench_train.tar.gz.ac" "https://zenodo.org/records/19979658/files/omniubench_train.tar.gz.ac?download=1"
wget -c -O "$TARGET_DIR/omniubench_train.tar.gz.ad" "https://zenodo.org/records/19979658/files/omniubench_train.tar.gz.ad?download=1"

(
	cd "$TARGET_DIR"
	cat omniubench_train.tar.gz.* | tar -xzvf -

	if [[ ! -d training_dataset ]]; then
		echo "[ERROR] training_dataset not found after extraction: $TARGET_DIR/training_dataset"
		exit 1
	fi

	if [[ -e media ]]; then
		echo "[ERROR] target directory already exists: $TARGET_DIR/media"
		exit 1
	fi

	mv training_dataset media
)

echo "[INFO] media directory: $TARGET_DIR/media"