#/bin/bash

pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python generate_config.py config.toml $CUDA_VISIBLE_DEVICES 8 # 8 -> 1000 later
