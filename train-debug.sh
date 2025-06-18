echo "Starting..."
CUDA_VISIBLE_DEVICES=0 python train.py --config=config-debug.toml --traindataset=/home/featurize/data/open-r1-math-220k.jsonl
# python train.py