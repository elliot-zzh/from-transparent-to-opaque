#!/bin/bash

# Run distributed VAE training with accelerate launch
# This script launches distributed VAE training across multiple GPUs

GPUS=${1:-"all"}  # Use all GPUs by default, or specify number as first arg
PER_DEVICE_BATCH_SIZE=${2:-4}  # Default batch size per device

# Validate arguments
if [ "$GPUS" != "all" ] && ! [[ "$GPUS" =~ ^[0-9]+$ ]]; then
  echo "Error: First argument must be 'all' or a number"
  exit 1
fi

if ! [[ "$PER_DEVICE_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
  echo "Error: Second argument (batch size) must be a number"
  exit 1
fi

echo "Starting distributed VAE training with $GPUS GPUs and batch size $PER_DEVICE_BATCH_SIZE per device"

# Run with accelerate launch
cd vae_train
accelerate launch \
  --multi_gpu \
  --num_processes=$GPUS \
  --mixed_precision=bf16 \
  train.py

echo "Distributed VAE training complete"