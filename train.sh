#!/bin/bash

set -e  # Exit on any error

echo "Starting hyperparameter search..."

echo "Running hyperparameter sweep: train_config_lr.sh"
./train_config_lr.sh
echo "Completed hyperparameter sweep: train_config_lr.sh"

echo "Running hyperparameter sweep: train_config_batch.sh"
./train_config_batch.sh
echo "Completed hyperparameter sweep: train_config_batch.sh"

echo "All hyperparameter sweeps completed"
