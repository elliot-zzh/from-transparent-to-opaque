#/bin/bash

pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
python generate_experiments.py config.toml $GPU_COUNT 1000
