#/bin/bash

pip install -r requirements.txt
python generate_experiments.py config.toml 5 100
