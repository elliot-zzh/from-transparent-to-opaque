#!/bin/bash

set -e  # Exit on any error

echo "Starting parallel GPU training..."

echo "Launching train_config_batch_gpu0.sh"
./train_config_batch_gpu0.sh &

echo "Waiting for all GPU processes to complete..."
wait
echo "All GPU processes completed"
