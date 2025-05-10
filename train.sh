#!/usr/bin/env bash

pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True
python train.py
