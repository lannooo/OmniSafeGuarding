#!/bin/bash

MODEL_NAME="models/base/Phi-4-multimodal-instruct"
INPUT="ms-swift/data/omguard_multimodal_basic.json"
OUTPUT="ms-swift/output/phi4_full_multimodal_basic"

MODEL_TYPE="phi4_multimodal"
TEMPLATE="phi4_multimodal"

CUDA_VISIBLE_DEVICES=0 \
swift sft \
      --model=$MODEL_NAME \
      --dataset=$INPUT \
      --model_type $MODEL_TYPE \
      --template $TEMPLATE \
      --load_from_cache_file true \
      --split_dataset_ratio 0.01 \
      --tuner_type full \
      --torch_dtype bfloat16 \
      --attn_impl flash_attn \
      --num_train_epochs 2 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --learning_rate 2e-5 \
      --freeze_vit true \
      --freeze_aligner true \
      --gradient_accumulation_steps 32 \
      --save_steps 100 \
      --save_only_model true \
      --save_total_limit 2 \
      --logging_steps 10 \
      --max_length 8196 \
      --output_dir=$OUTPUT \
      --warmup_ratio 0.05 \
      --dataset_num_proc 1 \
      --dataloader_num_workers 1