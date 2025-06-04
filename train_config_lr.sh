#!/bin/bash

set -e  # Exit on any error

echo "Starting parallel GPU training..."

. ./log_vram_usage.sh vram_usage_log.csv 5 &

echo "Launching train_config_lr_gpu0.sh"
./train_config_lr_gpu0.sh &

echo "Waiting for all GPU processes to complete..."
wait
echo "All GPU processes completed"
