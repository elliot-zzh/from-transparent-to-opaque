#!/bin/bash

set -e  # Exit on any error

echo "Starting parallel GPU training..."

echo "Launching train_config_lr_gpu0.sh"
./train_config_lr_gpu0.sh &
echo "Launching train_config_lr_gpu1.sh"
./train_config_lr_gpu1.sh &
echo "Launching train_config_lr_gpu2.sh"
./train_config_lr_gpu2.sh &
echo "Launching train_config_lr_gpu3.sh"
./train_config_lr_gpu3.sh &
echo "Launching train_config_lr_gpu4.sh"
./train_config_lr_gpu4.sh &

echo "Waiting for all GPU processes to complete..."
wait
echo "All GPU processes completed"
