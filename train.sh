#!/usr/bin/env bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting training..."

export TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxasexport
export TORCHINDUCTOR_CACHE_DIR=./data/inductor
mkdir -p ./data/inductor

#accelerate launch --gpu_ids=0 train.py --config configs/config_1.toml --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &
#accelerate launch --gpu_ids=1 train.py --config configs/config_2.toml --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &
#accelerate launch --gpu_ids=2 train.py --config configs/config_3.toml --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &

python train.py --config config.toml --traindataset /home/data/train.jsonl --testdataset /home/data/test.jsonl
