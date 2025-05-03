# From Transparent to Opaque

## Multi-GPU Training

This codebase now supports distributed training across multiple GPUs using Hugging Face's Accelerate library.

### Main Model Training

To run the main model training on multiple GPUs:

```bash
# Run on all available GPUs
./run_distributed.sh

# Specify number of GPUs and batch size per device
./run_distributed.sh 2 8  # Run on 2 GPUs with batch size 8 per GPU
```

### VAE Training

To run the VAE training on multiple GPUs:

```bash
# Run on all available GPUs
./run_vae_distributed.sh

# Specify number of GPUs and batch size per device
./run_vae_distributed.sh 2 8  # Run on 2 GPUs with batch size 8 per GPU
```

### Implementation Details

- Uses Accelerate's distributed training capabilities
- Automatically shards data across GPUs
- Handles gradient synchronization
- Mixed precision training with bfloat16
- Properly handles model saving and loading in distributed setting
- Distributes model layers across multiple GPUs when possible