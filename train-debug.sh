echo "Starting..."
CUDA_VISIBLE_DEVICES=0 accelerate run train.py --config=config-debug.toml
# python train.py
