#!/usr/bin/env bash

pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting training..."

accelerate launch --gpu_ids=0 train.py --config configs/config_1.toml &
accelerate launch --gpu_ids=1 train.py --config configs/config_2.toml &
accelerate launch --gpu_ids=3 train.py --config configs/config_3.toml &
