#/bin/bash

pip install -r requirements.txt
cp ./fsdp.yml ~/.cache/huggingface/accelerate/default_config.yaml
python generate_experiments.py config.toml 5 160
