#!/usr/bin/env bash

pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting training..."

accelerate launch --gpu_ids=0 train.py --config configs/config_1.toml --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &
accelerate launch --gpu_ids=1 train.py --config configs/config_2.toml --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &
accelerate launch --gpu_ids=2 train.py --config configs/config_3.toml --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &
