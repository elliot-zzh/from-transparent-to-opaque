pip install -r requirements.txt
pip install packaging ninja
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [[ "$gpu_name" == *"H100"* || "$gpu_name" == *"H200"* || "$gpu_name" == *"H800"* ]]; then
    arch="Hopper"
elif [[ "$gpu_name" == *"A100"* || "$gpu_name" == *"A200"* || "$gpu_name" == *"A800"* ]]; then
    arch="Ampere"
elif [[ "$gpu_name" == *"V100"* ]]; then
    arch="Volta"
elif [[ "$gpu_name" == *"3090"* || "$gpu_name" == *"3080"* ]]; then
    arch="Ampere"
elif [[ "$gpu_name" == *"4090"* || "$gpu_name" == *"4080"* ]]; then
    arch="Ada"
else
    arch="Unknown"
fi

echo "GPU: $gpu_name"
echo "Architecture: $arch"

if [[ "$gpu_name" == *"H100"* || "$gpu_name" == *"H200"* || "$gpu_name" == *"H800"* ]]; then
    if python -c "import importlib.util; exit(importlib.util.find_spec('flash_attn_3') is None)"; then
        echo "flash-attn-3 is installed"
    else
        echo "installing flash-attn-3"
        base=$(basename "$PWD")
        cd ..
        git clone https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention
        cd hopper
        python setup.py install
        cd ..
        cd ..
        cd $base
    fi
else
    if python -c "import importlib.util; exit(importlib.util.find_spec('flash_attn') is None)"; then
        echo "flash-attn is installed"
    else
        echo "installing flash-attn 2..."
        mem_gb=$(free -g | awk '/^Mem:/ { print $2 }')
        if [ "$mem_gb" -lt 96 ]; then
            export MAX_JOBS=4
        fi
        pip install flash-attn --no-build-isolation
    fi
fi
echo "flash-attn instsalled!"
