#!/bin/bash

set -e  # Exit on any error

echo "Starting parallel GPU training..."

. ./log_vram_usage.sh vram_usage_log.csv 5 &

echo "Launching train_gpu0.sh"
./train_gpu0.sh &
echo "Launching train_gpu1.sh"
./train_gpu1.sh &
echo "Launching train_gpu2.sh"
./train_gpu2.sh &
echo "Launching train_gpu3.sh"
./train_gpu3.sh &

echo "Waiting for all GPU processes to complete..."
wait
echo "All GPU processes completed"
