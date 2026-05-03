#!/bin/bash
# Qwen3-MoE-architecture requires at least 2 * 60GiB GPU memory for training, 
# we use deepspeed zero3 to reduce GPU memory usage.

MODEL_NAME="models/base/Qwen3-Omni-30B-A3B-Instruct"
INPUT="ms-swift/data/omguard_multimodal_mix_mrb_sae.json"
OUTPUT="ms-swift/output/qwen3_lora_zero3_multimodal_enhance"

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export MASTER_PORT=29500
export NCCL_TIMEOUT=360000
export MAX_PIXELS=301056 # 301056, 401408
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export NPROC_PER_NODE=2
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export CUDA_VISIBLE_DEVICES=0,1

swift sft \
    --model=$MODEL_NAME \
    --dataset=$INPUT \
    --split_dataset_ratio 0.0 \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --attn_impl flash_attn \
    --experts_impl grouped_mm \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner false \
    --gradient_accumulation_steps 8 \
    --save_steps 50 \
    --save_total_limit 8 \
    --logging_steps 10 \
    --max_length 8196 \
    --output_dir=$OUTPUT \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --dataloader_num_workers 4