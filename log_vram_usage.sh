#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <log_file> <interval_seconds>"
    exit 1
fi

LOG_FILE=$1
INTERVAL=$2

# Infinite loop to continuously log VRAM usage
while true; do
    # Query VRAM usage and append to the log file
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits >> $LOG_FILE
    # Wait for the specified interval before the next query
    sleep $INTERVAL
done