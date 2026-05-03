#!/bin/bash

MODEL_NAME="models/base/MiniCPM-o-4_5"
INPUT="ms-swift/data/omguard_multimodal_basic.json"
OUTPUT="ms-swift/output/minicpm_lora_multimodal_basic"

MODEL_TYPE="minicpmo"
TEMPLATE="minicpmo4_5"

CUDA_VISIBLE_DEVICES=0 \
swift sft \
      --model=$MODEL_NAME \
      --dataset=$INPUT \
      --model_type $MODEL_TYPE \
      --template $TEMPLATE \
      --load_from_cache_file true \
      --split_dataset_ratio 0.01 \
      --tuner_type lora \
      --num_train_epochs 2 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --learning_rate 2e-5 \
      --lora_rank 8 \
      --lora_alpha 32 \
      --attn_impl sdpa \
      --target_modules all-linear \
      --freeze_vit true \
      --freeze_aligner true \
      --packing false \
      --torch_dtype bfloat16 \
      --gradient_accumulation_steps 32 \
      --save_steps 50 \
      --save_total_limit 8 \
      --save_only_model true \
      --logging_steps 10 \
      --max_length 8196 \
      --output_dir=$OUTPUT \
      --warmup_ratio 0.05 \
      --dataset_num_proc 1 \
      --dataloader_num_workers 1