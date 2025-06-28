echo "Starting..."
CUDA_VISIBLE_DEVICES=0 python train.py --config=config-debug.toml
# python train.py