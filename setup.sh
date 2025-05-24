#/bin/bash

pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python generate_config.py config.toml 5 1000
