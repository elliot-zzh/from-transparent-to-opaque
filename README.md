# From Transparent to Opaque

## Before Launching Scripts

Run:

```bash
conda activate <your conda env>
./setup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./train.sh # or use srun etc to launch
```