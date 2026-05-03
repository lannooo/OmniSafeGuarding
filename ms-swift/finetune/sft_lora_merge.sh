#!/bin/bash

set -e

CHECKPOINT_PATH=""
BASE_MODEL_PATH=""

show_help() {
    echo "Usage: $0 [CHECKPOINT_PATH] [BASE_MODEL_PATH]"
    echo ""
    echo "Rewrite adapter_config.json(base_model_name_or_path) then merge LoRA with ms-swift."
    echo "Example:"
    echo "  $0 qwen_output/v1-20251217-122357/checkpoint-24 models/base/Qwen2.5-Omni-7B"
    echo "  $0 /absolute/path/to/checkpoint /absolute/path/to/base-model"
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
fi

if [[ -n "$1" ]]; then
    CHECKPOINT_PATH="$1"
else
    read -p "Enter the checkpoint path (e.g., qwen_output/v1-20251217-122357/checkpoint-24): " CHECKPOINT_PATH
fi

# obtain base path from command line
if [[ -n "$2" ]]; then
    BASE_MODEL_PATH="$2"
else
    read -p "Enter the base model path (e.g., models/base/Qwen2.5-Omni-7B): " BASE_MODEL_PATH
fi

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "❌ Error: Checkpoint path cannot be empty."
    show_help
fi

if [[ -z "$BASE_MODEL_PATH" ]]; then
    echo "❌ Error: Base model path cannot be empty."
    show_help
fi

if [[ ! "$CHECKPOINT_PATH" == /* ]]; then
    CHECKPOINT_PATH="$PWD/$CHECKPOINT_PATH"
fi

if [[ ! "$BASE_MODEL_PATH" == /* ]]; then
    BASE_MODEL_PATH="$PWD/$BASE_MODEL_PATH"
fi

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "❌ Error: Checkpoint directory does not exist: $CHECKPOINT_PATH"
    exit 1
fi

if [[ ! -d "$BASE_MODEL_PATH" ]]; then
    echo "❌ Error: Base model directory does not exist: $BASE_MODEL_PATH"
    exit 1
fi

ADAPTER_CONFIG_PATH="$CHECKPOINT_PATH/adapter_config.json"
if [[ ! -f "$ADAPTER_CONFIG_PATH" ]]; then
    echo "❌ Error: adapter_config.json not found: $ADAPTER_CONFIG_PATH"
    exit 1
fi

ARGS_JSON_PATH="$CHECKPOINT_PATH/args.json"

TMP_JSON_FILE="$(mktemp)"

# modify the original adapter_config.json / args.json with the actual base model path
if command -v jq >/dev/null 2>&1; then
    jq --arg base "$BASE_MODEL_PATH" '.base_model_name_or_path = $base' "$ADAPTER_CONFIG_PATH" > "$TMP_JSON_FILE"
else
    python3 - "$ADAPTER_CONFIG_PATH" "$BASE_MODEL_PATH" "$TMP_JSON_FILE" <<'PY'
import json
import sys

adapter_config_path = sys.argv[1]
base_model_path = sys.argv[2]
output_path = sys.argv[3]

with open(adapter_config_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["base_model_name_or_path"] = base_model_path

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
fi

mv "$TMP_JSON_FILE" "$ADAPTER_CONFIG_PATH"

if [[ -f "$ARGS_JSON_PATH" ]]; then
    TMP_JSON_FILE="$(mktemp)"
    if command -v jq >/dev/null 2>&1; then
        jq --arg base "$BASE_MODEL_PATH" '.model = $base | .model_dir = $base' "$ARGS_JSON_PATH" > "$TMP_JSON_FILE"
    else
        python3 - "$ARGS_JSON_PATH" "$BASE_MODEL_PATH" "$TMP_JSON_FILE" <<'PY'
import json
import sys

args_json_path = sys.argv[1]
base_model_path = sys.argv[2]
output_path = sys.argv[3]

with open(args_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["model"] = base_model_path
data["model_dir"] = base_model_path

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
    fi

    mv "$TMP_JSON_FILE" "$ARGS_JSON_PATH"
    echo "✅ Updated: $ARGS_JSON_PATH -> model, model_dir"
else
    echo "⚠️ Skip: args.json not found under checkpoint path"
fi

echo "✅ Checkpoint path: $CHECKPOINT_PATH"
echo "✅ Base model path: $BASE_MODEL_PATH"
echo "✅ Updated: $ADAPTER_CONFIG_PATH -> base_model_name_or_path"
echo "🚀 Starting LoRA merge with ms-swift..."

# merge the lora-ckpt with base model to the final model
swift export \
    --adapters "$CHECKPOINT_PATH" \
    --output_dir "${CHECKPOINT_PATH}_merged" \
    --merge_lora true

# [Notice] If flash_attn not installed, uncomment this and try to export with sdpa instead,
# it **MAY** lead to unexpected results, since the provided ckpts were all trained with 'flash_attn'
# swift export \
#     --adapters "$CHECKPOINT_PATH" \
#     --output_dir "${CHECKPOINT_PATH}_merged" \
#     --attn_impl sdpa \
#     --merge_lora true

echo "✅ LoRA merge completed"
echo "Merged model saved to: ${CHECKPOINT_PATH}_merged"
