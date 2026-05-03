#!/bin/bash

# **Note**: For text-only data, we can use larger batch size per device, 
# a single GPU with about 40 GB memory is enough for 7B model fine-tuning. 
# While for multimodal data, reduce the batch size per device to 1 or 2, 
# use gradient accumulation to achieve a larger effective batch size.
# If encounter OOM issue in multimodal training, please try to further 
# reduce the per_device_train_batch_size to 1, or also use multiple GPUs
# during trining (by setting CUDA_VISIBLE_DEVICES), or use DeepSpeed for
# optimization (see qwen3/zero3.sh).

MODEL_NAME="models/base/Qwen2.5-Omni-7B"
MODEL_TYPE="qwen2_5_omni"
INPUT="ms-swift/data/omguard_test_only.json"
OUTPUT="ms-swift/output/qwen25_lora_text"

# batch-size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# --attn_impl flash_attn if flash-attn is installed

CUDA_VISIBLE_DEVICES=0 \
swift sft \
      --model=$MODEL_NAME \
      --model_type $MODEL_TYPE \
      --dataset=$INPUT \
      --load_from_cache_file true \
      --split_dataset_ratio 0.01 \
      --tuner_type lora \
      --num_train_epochs 2 \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 4 \
      --learning_rate 5e-5 \
      --lora_rank 8 \
      --lora_alpha 32 \
      --target_modules all-linear \
      --freeze_vit true \
      --freeze_aligner true \
      --packing false \
      --torch_dtype bfloat16 \
      --gradient_accumulation_steps 8 \
      --save_steps 50 \
      --save_total_limit 8 \
      --save_only_model true \
      --logging_steps 10 \
      --max_length 8196 \
      --output_dir=$OUTPUT \
      --warmup_ratio 0.05 \
      --dataset_num_proc 1 \
      --dataloader_num_workers 4

