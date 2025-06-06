echo "Starting training..."
accelerate launch --gpu_ids=0 --num-processes=1 --mixed-precision=bf16 train.py --config=config-debug.toml --traindataset=/home/featurize/data/open-r1-math-220k-256.jsonl
# python train.py