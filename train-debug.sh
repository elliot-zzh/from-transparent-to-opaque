pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting training..."
accelerate launch --gpu_ids=0 --mixed-precision=bf16 train.py
# python train.py