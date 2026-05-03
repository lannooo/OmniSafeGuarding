#!/bin/bash

MODEL_NAME="models/base/Qwen2.5-Omni-7B"
MODEL_TYPE="qwen2_5_omni"
INPUT="ms-swift/data/omguard_multimodal_basic.json"
OUTPUT="ms-swift/output/qwen25_full_multimodal_basic"

CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
      --model=$MODEL_NAME \
      --model_type $MODEL_TYPE \
      --dataset=$INPUT \
      --load_from_cache_file true \
      --split_dataset_ratio 0.01 \
      --tuner_type full \
      --num_train_epochs 2 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --learning_rate 2e-5 \
      --freeze_vit true \
      --freeze_aligner true \
      --packing false \
      --torch_dtype bfloat16 \
      --gradient_accumulation_steps 16 \
      --save_steps 50 \
      --save_total_limit 8 \
      --save_only_model true \
      --logging_steps 10 \
      --max_length 8196 \
      --output_dir=$OUTPUT \
      --warmup_ratio 0.05 \
      --dataset_num_proc 1 \
      --dataloader_num_workers 4

