echo "Starting training..."
CUDA_VISIBLE_DEVICES=0 python train.py --config=config-debug.toml --traindataset=/home/featurize/data/open-r1-math-220k-256.jsonl
# python train.py