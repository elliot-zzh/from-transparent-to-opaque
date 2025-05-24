pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting training..."
accelerate launch --gpu_ids=0 --num-processes=1 --mixed-precision=bf16 train.py --traindataset=/home/featurize/data/open-r1-math-220k.jsonl
# python train.py